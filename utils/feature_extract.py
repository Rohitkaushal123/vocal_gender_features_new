import numpy as np
import librosa

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr).mean(axis=1)
    zcr = librosa.feature.zero_crossing_rate(y).mean()
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()

    features = np.hstack([mfcc, chroma, zcr, rolloff, centroid])
    return features
