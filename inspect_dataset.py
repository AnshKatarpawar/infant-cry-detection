
from pathlib import Path
from collections import Counter

PROJECT_DIR = Path(__file__).resolve().parent

# Locate the extracted dataset
DATASET_PATH = (
    PROJECT_DIR
    / "dataset"
    / "donateacry_corpus_cleaned_and_updated_data"
)

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

if not DATASET_PATH.exists():
    print("Dataset folder not found:", DATASET_PATH)
else:
    category_counts = Counter()

    for category_folder in DATASET_PATH.iterdir():
        if category_folder.is_dir():
            count = sum(
                1 for file in category_folder.rglob("*")
                if file.is_file()
                and file.suffix.lower() in AUDIO_EXTENSIONS
            )

            category_counts[category_folder.name] = count

    print("\n=== DATASET CATEGORY SUMMARY ===")
    print("Dataset path:", DATASET_PATH)
    print("Total recordings:", sum(category_counts.values()))

    for category, count in sorted(category_counts.items()):
        print(f"{category:20} : {count} files")
