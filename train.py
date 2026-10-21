
from pathlib import Path

import numpy as np
import librosa
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent
DATASET_PATH = (
    PROJECT_DIR
    / "dataset"
    / "donateacry_corpus_cleaned_and_updated_data"
)
MODEL_PATH = PROJECT_DIR / "models" / "cry_classifier.joblib"

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


# Extract numerical features from an audio recording
def extract_features(file_path):
    audio, sr = librosa.load(
        str(file_path),
        sr=16000,
        mono=True,
        duration=5
    )

    if len(audio) == 0:
        raise ValueError("Empty audio file")

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=20
    )

    # Mean and standard deviation of MFCC features
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # Additional audio characteristics
    rms = librosa.feature.rms(y=audio)[0]
    zcr = librosa.feature.zero_crossing_rate(audio)[0]
    centroid = librosa.feature.spectral_centroid(
        y=audio, sr=sr
    )[0]

    extra_features = [
        np.mean(rms), np.std(rms),
        np.mean(zcr), np.std(zcr),
        np.mean(centroid), np.std(centroid)
    ]

    return np.concatenate([
        mfcc_mean,
        mfcc_std,
        extra_features
    ])


def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {DATASET_PATH}"
        )

    features = []
    labels = []

    print("Reading dataset...")

    # Each immediate subfolder represents one category
    for category_folder in sorted(DATASET_PATH.iterdir()):
        if not category_folder.is_dir():
            continue

        category = category_folder.name

        files = [
            file for file in category_folder.rglob("*")
            if file.is_file()
            and file.suffix.lower() in AUDIO_EXTENSIONS
        ]

        print(f"{category}: {len(files)} recordings")

        for file_path in files:
            try:
                feature_vector = extract_features(file_path)
                features.append(feature_vector)
                labels.append(category)

            except Exception as error:
                print(f"Skipped {file_path.name}: {error}")

    if len(features) == 0:
        raise ValueError("No valid audio recordings found.")

    X = np.asarray(features)
    y = np.asarray(labels)

    print("\nUsable recordings:", len(y))
    print("Feature vector size:", X.shape[1])

    # A stratified split preserves category proportions
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("Training recordings:", len(y_train))
    print("Testing recordings:", len(y_test))

    # Train a baseline classifier
    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42
    )

    print("\nTraining model...")
    model.fit(X_train, y_train)

    # Evaluate on the held-out test set
    predictions = model.predict(X_test)

    print("\n=== MODEL EVALUATION ===")
    print(
        "Accuracy:",
        round(accuracy_score(y_test, predictions), 4)
    )

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=model.classes_,
            zero_division=0
        )
    )

    print("Confusion matrix:")
    print(model.classes_)
    print(confusion_matrix(
        y_test,
        predictions,
        labels=model.classes_
    ))

    # Save model and feature settings
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "sample_rate": 16000,
            "duration": 5,
            "n_mfcc": 20,
            "categories": list(model.classes_)
        },
        MODEL_PATH
    )

    print("\nModel saved successfully!")
    print("Location:", MODEL_PATH)


if __name__ == "__main__":
    main()
