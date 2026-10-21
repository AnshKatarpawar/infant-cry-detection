
from pathlib import Path

import joblib
import librosa
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "cry_classifier.joblib"

# Load the saved model and its settings
saved_data = joblib.load(MODEL_PATH)
model = saved_data["model"]

def extract_features(file_path):
    audio, sr = librosa.load(
        str(file_path),
        sr=16000,
        mono=True,
        duration=5
    )

    if len(audio) == 0:
        raise ValueError("The audio file is empty.")

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=20
    )

    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    rms = librosa.feature.rms(y=audio)[0]
    zcr = librosa.feature.zero_crossing_rate(audio)[0]
    centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sr
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
    ]).reshape(1, -1)


if __name__ == "__main__":
    audio_path = input(
        "Enter the full path of a WAV audio file: "
    ).strip().strip('"')

    path = Path(audio_path)

    if not path.is_file():
        print("Error: Audio file not found.")
    else:
        try:
            features = extract_features(path)

            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]

            print("\nPredicted category:", prediction)
            print("\nModel probability estimates:")

            for category, probability in sorted(
                zip(model.classes_, probabilities),
                key=lambda item: item[1],
                reverse=True
            ):
                print(f"{category}: {probability:.1%}")

            print(
                "\nNote: This is an experimental classification, "
                "not a medical assessment."
            )

        except Exception as error:
            print("Could not process the audio:", error)
