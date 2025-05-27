import pickle
import numpy as np
import streamlit as st
import librosa

# Load model and preprocessors
with open('randomModel.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('selectk.pkl', 'rb') as f:
    selector = pickle.load(f)

# Streamlit UI
st.title("🎤 Voice Gender Classification")
st.markdown("Upload a `.wav` or `.mp3` file to predict gender.")

audio_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

# Feature extraction function
def extract_all_43_features(file):
    try:
        y, sr = librosa.load(file, sr=None)

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        rms = librosa.feature.rms(y=y)
        pitches, _ = librosa.piptrack(y=y, sr=sr)
        min_pitch = pitches[pitches > 0].min() if np.any(pitches > 0) else 0.0

        def safe_mean(arr): return np.mean(arr) if arr.size else 0.0
        def safe_std(arr): return np.std(arr) if arr.size else 0.0

        features = np.array([
            safe_mean(spectral_centroid), safe_std(spectral_centroid),
            safe_mean(spectral_bandwidth), safe_std(spectral_bandwidth),
            safe_mean(contrast), safe_std(contrast),
            safe_mean(rms), safe_std(rms),
            safe_mean(zcr), safe_std(zcr),
            safe_mean(chroma), safe_std(chroma),
            min_pitch
        ] + [
            safe_mean(mfcc[i]) if i < mfcc.shape[0] else 0.0 for i in range(13)
        ] + [
            safe_std(mfcc[i]) if i < mfcc.shape[0] else 0.0 for i in range(13)
        ])

        print("🔍 Extracted features length:", len(features))

        # Force 43 features
        if len(features) < 43:
            features = np.append(features, [0.0] * (43 - len(features)))
        elif len(features) > 43:
            features = features[:43]

        return features.reshape(1, -1)

    except Exception as e:
        print(f"❌ Error extracting features: {e}")
        return np.zeros((1, 43))

# Run prediction
if audio_file is not None:
    st.audio(audio_file)

    try:
        features = extract_all_43_features(audio_file)

        if features.shape[1] != 43:
            st.error("❌ Feature extraction failed. Got wrong number of features.")
        else:
            scaled = scaler.transform(features)
            selected = selector.transform(scaled)
            prediction = model.predict(selected)

            # label_map = {0: "Male", 1: "Female"}
            label_map = {1: "Male", 0: "Female"}
            st.success(f"🎯 Predicted: **{label_map[prediction[0]]}**")

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
