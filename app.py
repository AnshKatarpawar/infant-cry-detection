
from pathlib import Path

import streamlit as st
import joblib
import librosa
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "cry_classifier.joblib"

st.set_page_config(
    page_title="Infant Cry Classifier",
    page_icon="🔊",
    layout="centered"
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

def extract_features(file_path):
    audio, sr = librosa.load(
        str(file_path),
        sr=16000,
        mono=True,
        duration=5
    )

    if len(audio) == 0:
        raise ValueError("The audio recording is empty.")

    mfcc = librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=20
    )

    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

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
        mfcc_mean, mfcc_std, extra_features
    ]).reshape(1, -1)


st.title("Infant Cry Classification")
st.write(
    "Upload a short audio recording to see the model's "
    "predicted category."
)

st.info(
    "This is an experimental classifier. It does not "
    "diagnose medical conditions or reliably determine "
    "an infant's actual needs."
)

try:
    saved_data = load_model()
    model = saved_data["model"]

    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=["wav", "mp3", "flac", "ogg", "m4a"]
    )

    if uploaded_file is not None:
        st.audio(uploaded_file)

        if st.button("Analyze recording"):
            try:
                # Use a temporary file that is deleted afterward
                suffix = Path(uploaded_file.name).suffix.lower()

                import tempfile

                with tempfile.NamedTemporaryFile(
                    suffix=suffix, delete=False
                ) as temp:
                    temp.write(uploaded_file.getvalue())
                    temp_path = Path(temp.name)

                try:
                    features = extract_features(temp_path)
                finally:
                    temp_path.unlink(missing_ok=True)

                prediction = model.predict(features)[0]
                probabilities = model.predict_proba(features)[0]

                st.subheader("Predicted category")
                st.success(prediction.replace("_", " ").title())

                st.subheader("Model probability estimates")

                for category, probability in sorted(
                    zip(model.classes_, probabilities),
                    key=lambda item: item[1],
                    reverse=True
                ):
                    st.write(
                        f"**{category.replace('_', ' ').title()}**"
                    )
                    st.progress(float(probability))
                    st.caption(f"{probability:.1%}")

            except Exception as error:
                st.error(f"Audio analysis failed: {error}")

except Exception as error:
    st.error(f"Could not load the trained model: {error}")
