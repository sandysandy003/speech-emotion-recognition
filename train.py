 
from pathlib import Path

import numpy as np
import librosa
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "raw"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

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

MAX_LENGTH = 174
SAMPLE_RATE = 22050
N_MFCC = 40


def extract_mfcc(file_path):
    audio, sr = librosa.load(
        str(file_path), sr=SAMPLE_RATE, mono=True
    )

    if audio.size == 0:
        raise ValueError(f"Empty audio file: {file_path}")

    mfcc = librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=N_MFCC
    )

    if mfcc.shape[1] < MAX_LENGTH:
        mfcc = np.pad(
            mfcc,
            ((0, 0), (0, MAX_LENGTH - mfcc.shape[1]))
        )
    else:
        mfcc = mfcc[:, :MAX_LENGTH]

    return mfcc.T.astype(np.float32)


def load_dataset():
    features, labels = [], []

    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {DATA_DIR}"
        )

    for path in DATA_DIR.rglob("*.wav"):
        parts = path.stem.split("-")

        if len(parts) != 7:
            continue

        emotion = EMOTIONS.get(parts[2])
        if emotion is None:
            continue

        try:
            features.append(extract_mfcc(path))
            labels.append(emotion)
        except Exception as exc:
            print(f"Skipping {path.name}: {exc}")

    if not features:
        raise RuntimeError(
            "No valid WAV files found in data/raw."
        )

    X = np.stack(features)
    y = np.asarray(labels)

    print(f"Loaded recordings: {len(y)}")
    print("Emotion counts:")
    for emotion in sorted(set(labels)):
        print(f"  {emotion}: {labels.count(emotion)}")

    return X, y


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
    print("Loading speech emotion dataset...")
    X, labels = load_dataset()

    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = build_model(
        X.shape[1:], len(encoder.classes_)
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(MODEL_DIR / "speech_emotion_model.keras"),
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=16,
        callbacks=callbacks,
    )

    loss, accuracy = model.evaluate(
        X_test, y_test, verbose=0
    )
    predictions = np.argmax(
        model.predict(X_test, verbose=0), axis=1
    )

    print(f"\nTest loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")
    print(classification_report(
        y_test,
        predictions,
        labels=np.arange(len(encoder.classes_)),
        target_names=encoder.classes_,
        zero_division=0,
    ))

    np.save(
        MODEL_DIR / "classes.npy",
        encoder.classes_,
    )
    print("\nTraining finished. Model saved in models/.")


if __name__ == "__main__":
    main()

