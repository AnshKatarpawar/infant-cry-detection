# Infant Cry Detection and Classification

A Python-based machine learning project that classifies infant cry audio recordings into five categories using audio feature extraction and a Random Forest classifier.

## Features

- Audio feature extraction using MFCCs, RMS energy, zero-crossing rate, and spectral centroid.
- Random Forest classification.
- Model evaluation using accuracy, precision, recall, F1-score, and a confusion matrix.
- Audio upload and prediction through a Streamlit web application.

## Categories

- Belly pain
- Burping
- Discomfort
- Hungry
- Tired

## Technologies Used

- Python
- Librosa
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- Joblib

## Project Structure

```text
InfantCryDetection/
├── dataset/                 # Download separately
├── models/
│   └── cry_classifier.joblib
├── inspect_dataset.py
├── train.py
├── predict.py
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

1. Clone this repository.
2. Create and activate a Python virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Download the Donate-a-Cry dataset from the [dataset repository](https://github.com/gveres/donateacry-corpus).
5. Place the extracted `donateacry_corpus_cleaned_and_updated_data` folder inside `dataset/`.
6. Train the model:

```bash
python train.py
```

7. Run the prediction script:

```bash
python predict.py
```

8. Launch the web application:

```bash
python -m streamlit run app.py
```

## Dataset

The project uses the Donate-a-Cry corpus. The original audio recordings are not included in this repository. Download the dataset separately and follow its applicable licence and usage terms.

## Limitations

The current model classifies recordings into five dataset categories. It does not independently distinguish infant crying from non-crying audio. The dataset is imbalanced, and classification performance varies significantly between categories.

Predictions are experimental and should not be used to diagnose a medical condition or determine an infant's actual needs.

## Future Improvements

- Add a separate crying-versus-non-crying detector.
- Improve performance on underrepresented classes.
- Evaluate on recordings from previously unseen infants.
- Add real-time audio analysis.
- Improve the visualization of model performance.

## Disclaimer

This project is for educational and research purposes only. It is not a medical device.
