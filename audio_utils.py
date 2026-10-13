
import librosa
import numpy as np


def extract_features(file_path, max_length=174):
    """Extract a fixed-size MFCC feature matrix from an audio file."""
    audio, sample_rate = librosa.load(
        file_path,
        sr=22050,
        mono=True
    )

    if audio.size == 0:
        raise ValueError("The audio file is empty.")

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=40
    )

    # Keep feature dimensions consistent for model training.
    if mfcc.shape[1] < max_length:
        mfcc = np.pad(
            mfcc,
            ((0, 0), (0, max_length - mfcc.shape[1])),
            mode="constant"
        )
    else:
        mfcc = mfcc[:, :max_length]

    return mfcc.astype(np.float32)
