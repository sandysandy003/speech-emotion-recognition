
import os
import tempfile

import librosa
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
from fastapi import FastAPI, File, Header, HTTPException, UploadFile

load_dotenv()

API_KEY = os.getenv("API_KEY")

if not API_KEY:
    raise RuntimeError("API_KEY is missing. Check your .env file.")

MODEL_PATH = "models/speech_emotion_model.keras"
CLASSES_PATH = "models/classes.npy"

app = FastAPI(title="Speech Emotion Recognition API")

model = tf.keras.models.load_model(MODEL_PATH)
classes = np.load(CLASSES_PATH, allow_pickle=True)


def extract_features(audio_path):
    audio, sr = librosa.load(audio_path, sr=22050, mono=True)

    # Match the 40-MFCC, 174-frame input used by the model.
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=40
    )

    features = np.zeros((40, 174), dtype=np.float32)
    frames = min(mfcc.shape[1], 174)
    features[:, :frames] = mfcc[:, :frames]

    # Model input: batch, time steps, MFCC features.
    return features.T[np.newaxis, :, :]


@app.get("/")
def home():
    return {
        "message": "Speech Emotion Recognition API",
        "status": "running"
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    x_api_key: str | None = Header(default=None)
):
    if x_api_key is None or x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key"
        )

    allowed_types = {
        "audio/wav",
        "audio/x-wav",
        "audio/mpeg",
        "audio/ogg",
        "audio/flac",
        "audio/mp4",
        "application/octet-stream"
    }

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail="Unsupported audio format"
        )

    suffix = os.path.splitext(file.filename or "")[1].lower()
    if suffix not in {".wav", ".mp3", ".ogg", ".flac", ".m4a"}:
        raise HTTPException(
            status_code=415,
            detail="Upload a WAV, MP3, OGG, FLAC, or M4A file"
        )

    temp_path = None

    try:
        contents = await file.read()

        if not contents:
            raise HTTPException(status_code=400, detail="Empty audio file")

        if len(contents) > 15 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="Audio file must be 15 MB or smaller"
            )

        with tempfile.NamedTemporaryFile(
            suffix=suffix, delete=False
        ) as temp:
            temp.write(contents)
            temp_path = temp.name

        features = extract_features(temp_path)
        prediction = model.predict(features, verbose=0)[0]
        index = int(np.argmax(prediction))

        return {
            "emotion": str(classes[index]),
            "confidence": round(float(prediction[index]), 4)
        }

    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not process the audio. Check the file format."
        )
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        await file.close()
