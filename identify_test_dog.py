from pathlib import Path

import numpy as np

from dog_reidentification import (
    BASE_DIR,
    MATCH_THRESHOLD,
    generate_embedding_for_image,
    get_dog_detector,
    get_model_and_processor,
    reference_embeddings_matrix,
    sha256_file,
)
from postgres_database import initialize_database, load_database


def main() -> int:
    try:
        initialize_database()
        database = load_database()
        dogs = database.get("dogs", [])
        if not dogs:
            print("No dogs are stored. Run enroll_dogs.py first.")
            return 1

        filename = input("Enter the test image filename in the project root: ").strip().strip('"')
        image_path = (BASE_DIR / Path(filename).name).resolve()
        if not filename or Path(filename).name != filename or not image_path.is_file():
            print(f"Image not found in project root: {filename}")
            return 1

        print("Loading CLIP model and dog detector...")
        processor, model = get_model_and_processor()
        detector, detector_transform = get_dog_detector()
        embedding = generate_embedding_for_image(
            image_path, processor, model, detector, detector_transform
        )

        matching_hash = sha256_file(image_path)
        exact_match = next(
            (dog for dog in dogs if dog.get("image_hash") == matching_hash), None
        )
        if exact_match:
            print(f"Matching dog ID: {exact_match['dog_id']}")
            print("Similarity: 1.0000")
            return 0

        similarities = reference_embeddings_matrix(database) @ embedding
        best_index = int(np.argmax(similarities))
        best_dog = dogs[best_index]
        similarity = float(similarities[best_index])
        print(f"Best dog ID: {best_dog['dog_id']}")
        print(f"Similarity: {similarity:.4f}")
        if similarity < MATCH_THRESHOLD:
            print("Result: no reliable match; this may be a new dog.")
        else:
            print("Result: matching dog found.")
        return 0
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())