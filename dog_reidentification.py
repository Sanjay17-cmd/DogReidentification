import hashlib
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from postgres_database import (
    initialize_database,
    load_database,
    migrate_json_database,
    save_database as save_postgres_database,
)
import torch
from PIL import Image, ImageOps
from transformers import CLIPModel, CLIPProcessor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn

BASE_DIR = Path(__file__).resolve().parent
IMAGE_FOLDER = BASE_DIR / "images"
TEST_FOLDER = BASE_DIR / "test"
MATCH_THRESHOLD = 0.80
DOG_DETECTION_THRESHOLD = 0.60
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MODEL_NAME = "openai/clip-vit-base-patch32"
PREPROCESSING_VERSION = "dog-detection-crop-v1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def print_banner() -> None:
    print("=" * 56)
    print("           DOG RE-IDENTIFICATION SYSTEM")
    print("=" * 56)
    print()
    print(f"Project directory:\n{BASE_DIR}")
    print()
    print(f"Reference image directory:\n{IMAGE_FOLDER}")
    print()


def print_error(message: str) -> None:
    print(f"[ERROR] {message}")


def sha256_file(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def list_reference_images(directory: Path) -> List[Path]:
    if not directory.exists() or not directory.is_dir():
        return []

    images = []
    def natural_name_key(path: Path) -> List[object]:
        return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.name.lower())]

    for item in sorted(directory.iterdir(), key=natural_name_key):
        if item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS:
            images.append(item)
    return images


def normalize_embedding(vector: np.ndarray) -> np.ndarray:
    vec = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        return vec
    return vec / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = normalize_embedding(a)
    b_norm = normalize_embedding(b)
    return float(np.dot(a_norm, b_norm))


def reference_embeddings_matrix(database: Dict) -> np.ndarray:
    embeddings = np.asarray(
        [dog.get("embedding", []) for dog in database.get("dogs", [])],
        dtype=np.float32,
    )
    if embeddings.size == 0:
        return np.empty((0, 0), dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-12)


def preprocess_image(image_path: Path) -> Image.Image:
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image


def get_dog_detector():
    try:
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        detector = fasterrcnn_resnet50_fpn(weights=weights)
        detector.to(DEVICE)
        detector.eval()
        return detector, weights.transforms()
    except Exception as exc:
        raise RuntimeError(f"Failed to load dog detector: {exc}") from exc


def maybe_crop_dog(image: Image.Image, detector, detector_transform) -> Image.Image:
    image_width, image_height = image.size
    detector_input = detector_transform(image).to(DEVICE)
    with torch.inference_mode():
        prediction = detector([detector_input])[0]

    dog_boxes = [
        box.cpu().numpy()
        for box, label, score in zip(
            prediction["boxes"], prediction["labels"], prediction["scores"]
        )
        if int(label) == 18 and float(score) >= DOG_DETECTION_THRESHOLD
    ]
    if not dog_boxes:
        return image

    left, top, right, bottom = np.asarray(dog_boxes[0], dtype=np.float32)
    padding_x = (right - left) * 0.10
    padding_y = (bottom - top) * 0.10
    left = max(0, int(left - padding_x))
    top = max(0, int(top - padding_y))
    right = min(image_width, int(right + padding_x))
    bottom = min(image_height, int(bottom + padding_y))
    return image.crop((left, top, right, bottom))


def get_model_and_processor():
    try:
        processor = CLIPProcessor.from_pretrained(MODEL_NAME)
        model = CLIPModel.from_pretrained(MODEL_NAME)
        model.to(DEVICE)
        model.eval()
        return processor, model
    except Exception as exc:
        raise RuntimeError(f"Failed to load model '{MODEL_NAME}': {exc}") from exc


def generate_embedding_for_image(image_path: Path, processor, model, detector, detector_transform) -> np.ndarray:
    return generate_embeddings_for_images([image_path], processor, model, detector, detector_transform)[0]


def generate_embeddings_for_images(
    image_paths: List[Path], processor, model, detector, detector_transform, batch_size: int = 8
) -> List[np.ndarray]:
    embeddings = []
    for start in range(0, len(image_paths), batch_size):
        images = [
            maybe_crop_dog(preprocess_image(path), detector, detector_transform)
            for path in image_paths[start:start + batch_size]
        ]
        inputs = processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(DEVICE)

        with torch.inference_mode():
            image_features = model.get_image_features(pixel_values=pixel_values)

        if hasattr(image_features, "pooler_output"):
            image_features = image_features.pooler_output
        batch_embeddings = image_features.cpu().numpy().astype(np.float32)
        norms = np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
        batch_embeddings /= np.maximum(norms, 1e-12)
        embeddings.extend(batch_embeddings)

    return embeddings


def save_database(database: Dict) -> None:
    save_postgres_database(database)


def validate_existing_database(database: Dict) -> bool:
    if not isinstance(database, dict):
        return False
    if database.get("model") != MODEL_NAME:
        return False
    if database.get("preprocessing") != PREPROCESSING_VERSION:
        return False
    dogs = database.get("dogs")
    if not isinstance(dogs, list):
        return False
    for dog in dogs:
        if not isinstance(dog, dict):
            return False
        if not {"dog_id", "image_name", "embedding", "embedding_dimension", "model"}.issubset(dog.keys()):
            return False
        embedding = dog.get("embedding")
        if not isinstance(embedding, list):
            return False
        if len(embedding) != int(dog.get("embedding_dimension", 0)):
            return False
    return True


def ensure_reference_database(reference_files: List[Path], processor, model, detector, detector_transform) -> Dict:
    if not reference_files:
        raise ValueError("No valid images found in the reference folder.")

    current_manifest = {
        file_path.name: sha256_file(file_path)
        for file_path in reference_files
    }

    existing_db = load_database()
    if validate_existing_database(existing_db):
        db_by_name = {dog.get("image_name"): dog for dog in existing_db.get("dogs", []) if isinstance(dog, dict)}
        if len(db_by_name) == len(reference_files) and all(
            file_name in db_by_name
            and db_by_name[file_name].get("image_hash") == current_manifest[file_name]
            and db_by_name[file_name].get("dog_id") == f"DOG_{index:03d}"
            for index, file_name in enumerate(current_manifest, start=1)
        ):
            print("Loading existing embedding database from PostgreSQL...")
            return existing_db

    print("Checking embedding database...")
    if existing_db.get("dogs"):
        print("Reference images changed or the database is missing metadata. Rebuilding the database...")
    else:
        print("No PostgreSQL embedding database found. Creating it...")

    dogs = []
    embeddings = generate_embeddings_for_images(reference_files, processor, model, detector, detector_transform)
    for index, image_path in enumerate(reference_files, start=1):
        dog_id = f"DOG_{index:03d}"
        print(f"[{index:02d}/{len(reference_files)}] {image_path.name} -> {dog_id}")
        embedding = embeddings[index - 1]
        dogs.append(
            {
                "dog_id": dog_id,
                "image_name": image_path.name,
                "image_hash": current_manifest[image_path.name],
                "embedding": embedding.astype(float).tolist(),
                "embedding_dimension": int(len(embedding)),
                "model": MODEL_NAME,
            }
        )

    database = {
        "model": MODEL_NAME,
        "preprocessing": PREPROCESSING_VERSION,
        "embedding_dimension": int(len(dogs[0]["embedding"])) if dogs else 0,
        "dogs": dogs,
    }
    save_database(database)
    print()
    print("Embedding database created successfully.")
    print("Database: PostgreSQL")
    print()
    return database


def resolve_new_image_path(raw_input: str) -> Optional[Path]:
    candidate = raw_input.strip().strip('"')
    if not candidate:
        return None

    candidate = os.path.expanduser(candidate)
    candidate_path = Path(candidate)

    # Full absolute path, such as: E:\Python model\testdog.jpg or C:\Users\Name\dog.jpg
    if candidate_path.is_absolute():
        resolved = candidate_path.resolve()
        if resolved.exists() and resolved.is_file():
            return resolved
        return None

    # Relative paths from the current working directory, project root, or images folder.
    search_locations = [
        Path.cwd(),
        BASE_DIR,
        IMAGE_FOLDER,
    ]

    for base_dir in search_locations:
        resolved = (base_dir / candidate).resolve()
        if resolved.exists() and resolved.is_file():
            return resolved

    # Allow a bare filename from the project root too.
    project_root_candidate = (BASE_DIR / candidate).resolve()
    if project_root_candidate.exists() and project_root_candidate.is_file():
        return project_root_candidate

    # Allow a simple filename that exists in the reference images folder.
    images_candidate = (IMAGE_FOLDER / candidate).resolve()
    if images_candidate.exists() and images_candidate.is_file():
        return images_candidate

    return None


def show_top_matches(matches: List[Dict]) -> None:
    print("Top 5 matches")
    print("-" * 56)
    for idx, match in enumerate(matches, start=1):
        print(f"{idx}. {match['dog_id']:<8} {match['similarity']:.4f}")
    print("-" * 56)
    print()


def print_result(best_match: Dict) -> None:
    print("=" * 56)
    print("RESULT")
    print("=" * 56)
    print(f"Input Image : {best_match['input_name']}")
    print(f"Best Match  : {best_match['dog_id']}")
    print(f"Similarity  : {best_match['similarity']:.4f}")
    print("Status      : EXISTING DOG")
    print()
    print(f"Reference Image : {best_match['image_name']}")
    print(f"This dog matches {best_match['dog_id']}.")
    print("=" * 56)


def print_new_dog_result(best_similarity: float, input_name: str) -> None:
    print("=" * 56)
    print("RESULT")
    print("=" * 56)
    print(f"Input Image : {input_name}")
    print(f"Best Similarity : {best_similarity:.4f}")
    print("Status          : NEW DOG")
    print()
    print("No sufficiently strong match was found.")
    print("=" * 56)


def next_dog_id(current_db: Dict) -> str:
    existing_ids = []
    for dog in current_db.get("dogs", []):
        dog_id = str(dog.get("dog_id", ""))
        try:
            number = int(dog_id.split("_")[-1])
            existing_ids.append(number)
        except ValueError:
            continue

    next_number = max(existing_ids) + 1 if existing_ids else 1
    return f"DOG_{next_number:03d}"


def add_new_dog_to_database(current_db: Dict, image_path: Path, embedding: np.ndarray) -> str:
    dog_id = next_dog_id(current_db)
    new_entry = {
        "dog_id": dog_id,
        "image_name": image_path.name,
        "image_hash": sha256_file(image_path),
        "embedding": embedding.astype(float).tolist(),
        "embedding_dimension": int(len(embedding)),
        "model": MODEL_NAME,
    }
    current_db["dogs"].append(new_entry)
    save_database(current_db)
    return dog_id


def check_test_images(test_files: List[Path], database: Dict, processor, model, detector, detector_transform) -> None:
    if not test_files:
        print_error(f"No supported image files were found in {TEST_FOLDER}")
        return

    print(f"Checking {len(test_files)} test image(s)...")
    test_embeddings = generate_embeddings_for_images(
        test_files, processor, model, detector, detector_transform
    )
    dogs = database.get("dogs", [])
    reference_matrix = reference_embeddings_matrix(database)
    dogs_by_hash = {
        dog.get("image_hash"): dog
        for dog in dogs
        if isinstance(dog, dict)
    }

    for image_path, new_embedding in zip(test_files, test_embeddings):
        input_hash = sha256_file(image_path)
        exact_match = dogs_by_hash.get(input_hash)
        if exact_match is not None:
            best_match = {
                "dog_id": exact_match.get("dog_id"),
                "image_name": exact_match.get("image_name"),
                "similarity": 1.0,
            }
        else:
            similarities = reference_matrix @ new_embedding
            best_index = int(np.argmax(similarities))
            best_match = {
                "dog_id": dogs[best_index].get("dog_id"),
                "image_name": dogs[best_index].get("image_name"),
                "similarity": float(similarities[best_index]),
            }

        print()
        if best_match["similarity"] >= MATCH_THRESHOLD:
            print_result({**best_match, "input_name": image_path.name})
        else:
            print_new_dog_result(best_match["similarity"], image_path.name)


def main() -> int:
    print_banner()

    if not IMAGE_FOLDER.exists():
        print_error(f"Reference image directory does not exist: {IMAGE_FOLDER}")
        return 1

    reference_images = list_reference_images(IMAGE_FOLDER)
    if not reference_images:
        print_error(f"No supported image files were found in {IMAGE_FOLDER}")
        return 1

    print(f"Checking reference images...")
    print(f"Found {len(reference_images)} images.")
    print()

    try:
        initialize_database()
        if migrate_json_database(BASE_DIR / "dog_embeddings.json"):
            print("Migrated dog_embeddings.json into PostgreSQL.")
        print("Loading embedding model...")
        processor, model = get_model_and_processor()
        print("Model loaded successfully.")
        print("Loading dog detector...")
        detector, detector_transform = get_dog_detector()
        print("Dog detector loaded successfully.")
        print()
    except Exception as exc:
        print_error(str(exc))
        return 1

    try:
        database = ensure_reference_database(
            reference_images, processor, model, detector, detector_transform
        )
    except Exception as exc:
        print_error(str(exc))
        return 1

    print(f"Reference dogs available: {len(database.get('dogs', []))}")
    print()

    if TEST_FOLDER.exists() and TEST_FOLDER.is_dir():
        test_images = list_reference_images(TEST_FOLDER)
        check_test_images(test_images, database, processor, model, detector, detector_transform)
        return 0

    while True:
        try:
            user_input = input("Enter the path of the new dog image:\n> ").strip()
        except EOFError:
            print()
            print("No input received. Exiting.")
            return 0

        if not user_input:
            print_error("Please enter a valid image path or filename.")
            continue

        image_path = resolve_new_image_path(user_input)
        if image_path is None:
            print_error("Invalid image path. Use a full path, a project-relative path, or a filename from the project root or images folder.")
            print_error(f"Example: E:\\Python model\\testdog.jpg  |  images\\testdog.jpg  |  testdog.jpg")
            continue

        try:
            print("Processing image...")
            print("Generating embedding...")
            new_embedding = generate_embedding_for_image(
                image_path, processor, model, detector, detector_transform
            )
        except Exception as exc:
            print_error(f"Could not process image '{image_path}': {exc}")
            continue

        exact_match = None
        input_hash = sha256_file(image_path)
        for dog in database.get("dogs", []):
            if dog.get("image_hash") == input_hash:
                exact_match = {
                    "dog_id": dog.get("dog_id"),
                    "image_name": dog.get("image_name"),
                    "similarity": 1.0,
                }
                break

        if exact_match is not None:
            print("Comparing against {} dogs...".format(len(database.get("dogs", []))))
            print()
            print("Top 5 matches")
            print("-" * 56)
            print(f"1. {exact_match['dog_id']:<8} 1.0000")
            print("-" * 56)
            print()
            result = {
                "dog_id": exact_match["dog_id"],
                "image_name": exact_match["image_name"],
                "similarity": 1.0,
                "input_name": image_path.name,
            }
            print_result(result)

            while True:
                try:
                    repeat_choice = input("Do you want to identify another dog? [Y/N]: ").strip().upper()
                except EOFError:
                    repeat_choice = "N"
                if repeat_choice in {"Y", "YES"}:
                    break
                if repeat_choice in {"N", "NO", ""}:
                    print("Exiting dog re-identification system.")
                    return 0
                print_error("Please answer Y or N.")
            continue

        dogs = database.get("dogs", [])
        reference_matrix = reference_embeddings_matrix(database)
        similarities = reference_matrix @ new_embedding
        ranked_indexes = np.argsort(similarities)[::-1][:5]
        top_five = [
            {
                "dog_id": dogs[index].get("dog_id"),
                "image_name": dogs[index].get("image_name"),
                "similarity": float(similarities[index]),
            }
            for index in ranked_indexes
        ]

        print("Comparing against {} dogs...".format(len(dogs)))
        print()
        show_top_matches(top_five)

        best_match = top_five[0]
        best_similarity = float(best_match["similarity"])

        if best_similarity >= MATCH_THRESHOLD:
            result = {
                "dog_id": best_match["dog_id"],
                "image_name": best_match["image_name"],
                "similarity": best_similarity,
                "input_name": image_path.name,
            }
            print_result(result)
        else:
            print_new_dog_result(best_similarity, image_path.name)
            while True:
                try:
                    add_choice = input("Do you want to add this new dog to the database? [Y/N]: ").strip().upper()
                except EOFError:
                    add_choice = "N"
                if add_choice in {"Y", "YES"}:
                    new_id = add_new_dog_to_database(database, image_path, new_embedding)
                    print()
                    print("New dog added successfully.")
                    print(f"Assigned ID: {new_id}")
                    print("Embedding database updated: PostgreSQL")
                    break
                if add_choice in {"N", "NO", ""}:
                    print("New dog not added to the database.")
                    break
                print_error("Please answer Y or N.")

        while True:
            try:
                repeat_choice = input("Do you want to identify another dog? [Y/N]: ").strip().upper()
            except EOFError:
                repeat_choice = "N"
            if repeat_choice in {"Y", "YES"}:
                break
            if repeat_choice in {"N", "NO", ""}:
                print("Exiting dog re-identification system.")
                return 0
            print_error("Please answer Y or N.")


if __name__ == "__main__":
    sys.exit(main())
