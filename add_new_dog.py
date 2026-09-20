from pathlib import Path

from dog_reidentification import (
    BASE_DIR,
    generate_embedding_for_image,
    get_dog_detector,
    get_model_and_processor,
    sha256_file,
)
from postgres_database import initialize_database, load_database, save_database


def next_dog_id(dogs: list[dict]) -> str:
    numbers = []
    for dog in dogs:
        try:
            numbers.append(int(str(dog.get("dog_id", "")).split("_")[-1]))
        except ValueError:
            continue
    return f"DOG_{max(numbers, default=0) + 1:03d}"


def main() -> int:
    try:
        initialize_database()
        database = load_database()
        if not database.get("dogs"):
            print("No existing dogs found. Run enroll_dogs.py first.")
            return 1

        filename = input("Enter the new dog image filename in the project root: ").strip().strip('"')
        image_path = (BASE_DIR / Path(filename).name).resolve()
        if not filename or Path(filename).name != filename or not image_path.is_file():
            print(f"Image not found in project root: {filename}")
            return 1

        image_hash = sha256_file(image_path)
        if any(dog.get("image_hash") == image_hash for dog in database["dogs"]):
            print("This image is already stored in PostgreSQL.")
            return 1

        print("Loading CLIP model and dog detector...")
        processor, model = get_model_and_processor()
        detector, detector_transform = get_dog_detector()
        embedding = generate_embedding_for_image(
            image_path, processor, model, detector, detector_transform
        )
        dog_id = next_dog_id(database["dogs"])
        database["dogs"].append(
            {
                "dog_id": dog_id,
                "image_name": image_path.name,
                "image_hash": image_hash,
                "embedding": embedding.astype(float).tolist(),
                "embedding_dimension": int(len(embedding)),
                "model": database.get("model", "openai/clip-vit-base-patch32"),
            }
        )
        save_database(database)
        print(f"Added {image_path.name} to PostgreSQL as {dog_id}.")
        return 0
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())