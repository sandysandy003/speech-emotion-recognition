
from pathlib import Path
import numpy as np
import librosa
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# RAVDESS filename emotion codes
EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}


def extract_mfcc(file_path, max_length=174):
    audio, sr = librosa.load(file_path, sr=22050, mono=True)
    if audio.size == 0:
        raise ValueError(f"Empty audio: {file_path}")

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)

    if mfcc.shape[1] < max_length:
        mfcc = np.pad(
            mfcc,
            ((0, 0), (0, max_length - mfcc.shape[1])),
        )
    else:
        mfcc = mfcc[:, :max_length]

    return mfcc.T.astype(np.float32)


def load_dataset():
    features, labels = [], []

    for path in DATA_DIR.rglob("*.wav"):
        # RAVDESS filenames contain an emotion code in field 3.
        parts = path.stem.split("-")
        if len(parts) != 7:
            continue

        emotion = EMOTIONS.get(parts[2])
        if emotion is None:
            continue

        try:
            features.append(extract_mfcc(path))
            labels.append(emotion)
        except (ValueError, OSError) as exc:
            print(f"Skipping {path}: {exc}")

    if not features:
        raise RuntimeError(
            "No valid RAVDESS WAV files found in data/. "
            "Extract the dataset into data/ first."
        )

    return np.stack(features), np.array(labels)


def build_model(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv1D(
        64, 5, padding="same", activation="relu"
    )(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True)
    )(x)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(32)
    )(x)

    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(
        num_classes, activation="softmax"
    )(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    print("Loading audio dataset...")
    X, labels = load_dataset()

    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_model(X.shape[1:], len(encoder.classes_))
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_DIR / "speech_emotion_model.keras",
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
    )

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    predictions = np.argmax(model.predict(X_test, verbose=0), axis=1)

    print(f"\nTest loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")
    print(classification_report(
        y_test,
        predictions,
        labels=np.arange(len(encoder.classes_)),
        target_names=encoder.classes_,
        zero_division=0,
    ))

    np.save(MODEL_DIR / "classes.npy", encoder.classes_)
    print("Model and class labels saved in models/.")


if __name__ == "__main__":
    main()

import numpy as np
import librosa


def extract_features(file_path):
    """
    Extract audio features from a speech recording.
    Returns a numerical feature vector.
    """
    try:
        audio, sample_rate = librosa.load(
            file_path,
            sr=22050,
            mono=True
        )

        if len(audio) == 0:
            raise ValueError("Audio file is empty.")

        # MFCC features
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sample_rate,
            n_mfcc=40
        )

        # Delta MFCC features
        delta_mfcc = librosa.feature.delta(mfcc)

        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(
            y=audio,
            sr=sample_rate
        )

        zero_crossing_rate = librosa.feature.zero_crossing_rate(
            y=audio
        )

        # Combine statistical summaries
        features = np.concatenate([
            np.mean(mfcc, axis=1),
            np.std(mfcc, axis=1),
            np.mean(delta_mfcc, axis=1),
            np.mean(spectral_centroid, axis=1),
            np.mean(zero_crossing_rate, axis=1)
        ])

        return features.astype(np.float32)

    except Exception as error:
        raise RuntimeError(
            f"Could not process audio file {file_path}: {error}"
        ) from error

from pathlib import Path
import sys
import re

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "raw"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"

sys.path.insert(0, str(ROOT / "src"))
from features import extract_features

# RAVDESS filename emotion codes
EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {DATA_DIR}"
        )

    audio_files = sorted(DATA_DIR.glob("Actor_*/*.wav"))

    if not audio_files:
        raise FileNotFoundError(
            "No WAV files found. Extract the RAVDESS "
            "Actor folders into data/raw."
        )

    features_list = []
    labels = []

    print(f"Found {len(audio_files)} audio files.")

    for index, audio_path in enumerate(audio_files, start=1):
        # RAVDESS filename example:
        # 03-01-05-01-02-01-12.wav
        match = re.match(
            r"\d{2}-\d{2}-(\d{2})-\d{2}-\d{2}-\d{2}-\d{2}\.wav$",
            audio_path.name,
            re.IGNORECASE,
        )

        if not match:
            continue

        emotion_code = match.group(1)

        # Use the eight emotion categories, including calm.
        if emotion_code not in EMOTIONS:
            continue

        try:
            features = extract_features(audio_path)

            if not np.all(np.isfinite(features)):
                print(f"Skipping invalid features: {audio_path.name}")
                continue

            features_list.append(features)
            labels.append(EMOTIONS[emotion_code])

        except Exception as error:
            print(f"Skipping {audio_path.name}: {error}")

        if index % 100 == 0:
            print(f"Processed {index}/{len(audio_files)} files")

    if len(features_list) < 20:
        raise RuntimeError(
            "Not enough valid recordings. Check the dataset "
            "and audio-processing installation."
        )

    X = np.vstack(features_list)
    y = np.array(labels)

    print("\nDataset size:", len(y))
    print("Emotion counts:")
    print(pd.Series(y).value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    )

    print("\nTraining model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"\nTest accuracy: {accuracy:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))

    joblib.dump(model, MODEL_DIR / "emotion_model.pkl")
    joblib.dump(
        sorted(EMOTIONS.values()),
        MODEL_DIR / "emotion_labels.pkl",
    )

    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(
        OUTPUT_DIR / "classification_report.csv"
    )

    pd.DataFrame(
        confusion_matrix(
            y_test,
            predictions,
            labels=sorted(EMOTIONS.values()),
        ),
        index=sorted(EMOTIONS.values()),
        columns=sorted(EMOTIONS.values()),
    ).to_csv(OUTPUT_DIR / "confusion_matrix.csv")

    print("\nTraining complete!")
    print("Saved model:", MODEL_DIR / "emotion_model.pkl")
    print("Saved evaluation files in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
if __name__ == "__main__":
    main()

    print("\nTraining complete!")
    print("Saved model:", MODEL_DIR / "emotion_model.pkl")
    print("Saved evaluation files in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()

