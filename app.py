import streamlit as st
import os
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load YAMNet model
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

# Load trained model
try:
    model = tf.keras.models.load_model("deep_fake_voice.keras")
except:
    st.error("🚨 Model file not found! Ensure 'deep_fake_voice.h5' is in the same folder.")

# Function to extract YAMNet + MFCC + Spectral Contrast features
def extract_features(file_path):
    """Extracts YAMNet + MFCC + Spectral Contrast features"""
    try:
        audio, sr = librosa.load(file_path, sr=16000)
        
        # YAMNet Embeddings
        waveform = tf.convert_to_tensor(audio, dtype=tf.float32)
        scores, embeddings, spectrogram = yamnet_model(waveform)
        yamnet_features = embeddings.numpy().mean(axis=0)  # Average embeddings
        
        # MFCC Features (Reduced from 40 → 20)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
        mfcc_features = np.mean(mfcc, axis=1)
        
        # Spectral Contrast Features
        spec_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        spec_contrast_features = np.mean(spec_contrast, axis=1)
        
        # Combine all features
        return np.concatenate((yamnet_features, mfcc_features, spec_contrast_features)), audio, sr, mfcc
    
    except Exception as e:
        st.error(f"Error processing {file_path}: {e}")
        return None, None, None, None

# Streamlit UI
st.title("🔊 Deepfake Voice Detection")
st.write("Upload an audio file to check if it's **REAL** or **FAKE**.")

uploaded_file = st.file_uploader("Upload an audio file (.wav, .mp3)", type=["wav", "mp3"])

if uploaded_file is not None:
    # Save uploaded file
    file_path = "temp_audio.wav"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Extract features and get audio data for visualization
    st.write("📊 Extracting features and analyzing audio...")
    features, audio, sr, mfcc = extract_features(file_path)
    
    if features is not None:
        # Display waveform
        st.subheader("🎧 Audio Waveform")
        plt.figure(figsize=(10, 4))
        librosa.display.waveshow(audio, sr=sr)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        st.pyplot(plt)

        # Display spectrogram
        st.subheader("📈 Spectrogram")
        plt.figure(figsize=(10, 4))
        X = librosa.stft(audio)
        Xdb = librosa.amplitude_to_db(abs(X))
        librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
        plt.colorbar(format="%+2.0f dB")
        st.pyplot(plt)

        # Display MFCC
        st.subheader("🎵 MFCC (Mel-Frequency Cepstral Coefficients)")
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mfcc, sr=sr, x_axis='time')
        plt.colorbar(format="%+2.0f dB")
        plt.ylabel("MFCC Coefficients")
        st.pyplot(plt)

        # Make prediction
        features = np.expand_dims(features, axis=0)  # Add batch dimension
        prediction = model.predict(features)

        # Determine if the audio is REAL or FAKE
        predicted_class = "REAL" if prediction[0][1] > 0.5 else "FAKE"

        # Display prediction result
        st.subheader("🎯 Prediction Result")
        st.write(f"**The audio is classified as:** {predicted_class}")

    # Remove temporary file
    os.remove(file_path)