from dog_reidentification import (
    BASE_DIR,
    IMAGE_FOLDER,
    ensure_reference_database,
    get_dog_detector,
    get_model_and_processor,
    list_reference_images,
)
from postgres_database import initialize_database


def main() -> int:
    try:
        initialize_database()
        image_paths = list_reference_images(IMAGE_FOLDER)
        if not image_paths:
            print(f"No supported images found in {IMAGE_FOLDER}")
            return 1

        print(f"Found {len(image_paths)} dog image(s) in {IMAGE_FOLDER}")
        print("Loading CLIP model and dog detector...")
        processor, model = get_model_and_processor()
        detector, detector_transform = get_dog_detector()
        database = ensure_reference_database(
            image_paths, processor, model, detector, detector_transform
        )
        print(f"PostgreSQL now contains {len(database['dogs'])} dog(s).")
        print(f"Project: {BASE_DIR}")
        return 0
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())