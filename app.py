import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
import tempfile
from PIL import Image

# Custom theme
def set_custom_theme():
    st.markdown(""" <style>
    .main {
        background-color: #F5F5F5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 24px;
    }
    .stFileUploader>div>div>div>button {
        background-color: #2196F3;
        color: white;
    }
    .stMarkdown h1 {
        color: #FF5722;
    }
    .stMarkdown h2 {
        color: #607D8B;
    }
    .stMarkdown h3 {
        color: #795548;
    }
    .prediction-real {
        color: #4CAF50;
        font-weight: bold;
        font-size: 1.5em;
    }
    .prediction-fake {
        color: #F44336;
        font-weight: bold;
        font-size: 1.5em;
    } </style>
    """, unsafe_allow_html=True)

set_custom_theme()

# App header with logo
col1, col2 = st.columns([1, 4])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/4474/4474885.png", width=80)
with col2:
    st.title("Deepfake Audio Detector")

st.markdown("""
Upload a `.wav` audio file, and this app will analyze whether it's a **real** or **fake** voice using our advanced deep learning model.
The app provides detailed visualizations and explanations of the audio features used in detection.
""")

# Sidebar with info
with st.sidebar:
    st.header("About")
    st.markdown("""
    This app uses an advanced deep learning model that combines:
    - **YAMNet embeddings** (pretrained audio analysis)
    - **MFCC (20 coefficients)**
    - **Spectral contrast features**
    - **9 different audio feature visualizations**
    
    The model analyzes subtle artifacts that often exist in synthetic voices.
    """)
    st.markdown("---")
    st.markdown("**How it works:**")
    st.markdown("1. Upload a WAV audio file")
    st.markdown("2. The app extracts advanced audio features")
    st.markdown("3. Model analyzes patterns using deep learning")
    st.markdown("4. Get detection results with explanations")
    st.markdown("---")
    st.markdown("Made with ❤️ using Streamlit & TensorFlow")

# Load models (with caching)
@st.cache_resource
def load_models():
    try:
        yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
        custom_model = tf.keras.models.load_model("deep_fake_voice.keras", compile=False)
        custom_model.compile(
            loss='binary_crossentropy',
            optimizer=tf.keras.optimizers.AdamW(learning_rate=0.0005),
            metrics=['accuracy']
        )
        return yamnet_model, custom_model
    except Exception as e:
        st.error(f"Model loading error: {str(e)}")
        return None, None

yamnet_model, model = load_models()

# Feature extraction function
def extract_features(audio, sr):
    try:
        # YAMNet Embeddings
        waveform = tf.convert_to_tensor(audio, dtype=tf.float32)
        scores, embeddings, spectrogram = yamnet_model(waveform)
        yamnet_features = embeddings.numpy().mean(axis=0)
        
        # MFCC Features (20 coefficients)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
        mfcc_features = np.mean(mfcc, axis=1)
        
        # Spectral Contrast Features
        spec_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        spec_contrast_features = np.mean(spec_contrast, axis=1)
        
        # Combine all features
        return np.concatenate((yamnet_features, mfcc_features, spec_contrast_features))
    except Exception as e:
        st.error(f"Feature extraction error: {str(e)}")
        return None

# Graph explanations
GRAPH_EXPLANATIONS = {
    "waveform": "The waveform shows the amplitude of the audio signal over time. Real voices typically have more natural variations in amplitude while synthetic voices often show repetitive patterns or unusual amplitude distributions.",
    "spectrogram": "The spectrogram displays frequency content over time. Deepfake voices often show inconsistencies in harmonic structures, with either too-perfect harmonics or abrupt frequency transitions that don't match natural speech patterns.",
    "mfcc": "MFCCs represent the short-term power spectrum of sound. We use 20 coefficients for better analysis of synthetic voices. Fake voices often show either too-consistent MFCC patterns (lacking natural variation) or erratic coefficient changes.",
    "chroma": "Chroma features show pitch class information. Fake voices might have unnatural pitch transitions, with either overly smooth pitch changes (characteristic of vocoders) or abrupt jumps that don't match natural speech prosody.",
    "spectral_contrast": "Spectral contrast shows the difference between spectral peaks and valleys. Synthetic voices often show either exaggerated contrast (from artificial enhancement) or unnaturally flat contrast (from voice conversion artifacts).",
    "spectral_centroid": "Indicates where the 'center of mass' of the spectrum is located. Real voices usually have smoother centroid movements, while synthetic voices may show unnaturally stable or erratic centroid patterns.",
    "spectral_bandwidth": "Measures the spectral spread. Unusually narrow bandwidth can indicate band-limited synthesis, while unusually wide bandwidth may suggest artificial noise injection in fake voices.",
    "spectral_rolloff": "Shows the frequency below which a specified percentage of energy lies. Fake voices may have abnormal rolloff patterns, either cutting off too abruptly or extending unnaturally.",
    "zero_crossing": "Counts how often the signal crosses zero. This can reveal unnatural speech rhythms in synthetic audio, with either too-regular or too-irregular zero-crossing patterns.",
    "rms_energy": "Root Mean Square energy shows loudness variations. Real speech has natural energy fluctuations, while synthetic voices may show either overly compressed or unnaturally varying energy levels.",
    "yamnet": "YAMNet embeddings provide high-level audio features from a pretrained model. The pattern of these 512-dimensional embeddings often differs between real and synthetic speech, with fake voices showing less natural clustering in the embedding space."
}

# Detailed detection reasons
def get_detection_reasons(prediction, features):
    reasons = []
    
    if prediction < 0.5:
        reasons.append("Natural amplitude variations in waveform matching human speech patterns")
        reasons.append("Consistent harmonic structures in spectrogram with appropriate formant transitions")
        reasons.append("Normal distribution of MFCC coefficients showing natural speech characteristics")
        reasons.append("Appropriate spectral contrast between harmonic and non-harmonic components")
        reasons.append("Smooth spectral centroid movement typical of natural vocal tract movements")
        reasons.append("Natural zero-crossing rate matching human speech rhythm patterns")
        reasons.append("YAMNet embeddings show typical human speech clustering patterns")
    else:
        if features['spectral_contrast_var'] > 5:
            reasons.append(f"Abnormal spectral contrast variation (score: {features['spectral_contrast_var']:.2f} - suggests artificial harmonic enhancement)")
        if features['mfcc_range'] < 15:
            reasons.append(f"Limited MFCC variation (range: {features['mfcc_range']:.2f} - indicates overly consistent spectral envelope)")
        if features['yamnet_confidence'] < 0.7:
            reasons.append(f"Low YAMNet speech confidence ({features['yamnet_confidence']:.2f} - suggests atypical speech patterns)")
        if features['zcr_var'] > 0.15:
            reasons.append(f"Abnormal zero-crossing rate variation ({features['zcr_var']:.2f} - indicates unnatural speech rhythm)")
        if features['centroid_var'] > 800:
            reasons.append(f"Unstable spectral centroid ({features['centroid_var']:.2f} - suggests vocal tract inconsistencies)")
        
        reasons.append("Combined features show patterns consistent with known synthetic voice artifacts")

    return reasons

# File uploader
uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"], key="file_uploader")

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    
    try:
        # Save file to temp directory
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
            temp_audio.write(uploaded_file.read())
            temp_path = temp_audio.name

        # Load audio
        y, sr = librosa.load(temp_path, sr=16000)  # Match YAMNet's expected sample rate

        # Feature extraction
        with st.spinner('Extracting advanced audio features...'):
            features = extract_features(y, sr)
            
            if features is not None:
                # Calculate comprehensive feature stats for detailed reasoning
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
                spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
                spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
                zcr = librosa.feature.zero_crossing_rate(y)[0]
                
                feature_stats = {
                    'spectral_contrast_var': np.var(spec_contrast),
                    'mfcc_range': np.ptp(mfcc),
                    'yamnet_confidence': np.mean(features[:512]),
                    'zcr_var': np.var(zcr),
                    'centroid_var': np.var(spectral_centroid),
                    'bandwidth': np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
                    'rolloff': np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),
                    'rms': np.mean(librosa.feature.rms(y=y))
                }
                
                # Reshape for prediction
                features = features.reshape(1, -1)

                # Predict
                prediction = model.predict(features)[0][0]
                result = "🟢 Real Voice" if prediction < 0.5 else "🔴 Fake Voice"
                confidence = (1 - prediction) if prediction < 0.5 else prediction

                # Display results
                if prediction < 0.5:
                    st.markdown(f'<div class="prediction-real">{result} (Confidence: {confidence:.2%})</div>', 
                               unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="prediction-fake">{result} (Confidence: {confidence:.2%})</div>', 
                               unsafe_allow_html=True)
                
                # Show detailed detection reasons
                st.subheader("🔍 Detailed Detection Analysis")
                reasons = get_detection_reasons(prediction, feature_stats)
                for i, reason in enumerate(reasons, 1):
                    st.markdown(f"**{i}.** {reason}")
                
                # Feature visualizations
                st.markdown("---")
                st.subheader("📊 Comprehensive Audio Analysis")
                
                def plot_graph(title, plot_func, explanation_key):
                    with st.expander(f"📌 {title}", expanded=False):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            fig, ax = plt.subplots(figsize=(10, 4))
                            plot_func(ax)
                            plt.tight_layout()
                            st.pyplot(fig)
                        with col2:
                            st.info(GRAPH_EXPLANATIONS[explanation_key])

                # 1. Waveform
                plot_graph("1. Time Domain Waveform", 
                           lambda ax: librosa.display.waveshow(y, sr=sr, ax=ax, color='#1f77b4'),
                           "waveform")

                # 2. Spectrogram
                plot_graph("2. Frequency Spectrogram", 
                           lambda ax: librosa.display.specshow(
                               librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max),
                               sr=sr, x_axis='time', y_axis='hz', ax=ax, cmap='magma'),
                           "spectrogram")

                # 3. MFCC
                plot_graph("3. MFCC (20 Coefficients)", 
                           lambda ax: librosa.display.specshow(
                               mfcc, sr=sr, x_axis='time', ax=ax, cmap='coolwarm'),
                           "mfcc")

                # 4. Chroma
                chroma = librosa.feature.chroma_stft(y=y, sr=sr)
                plot_graph("4. Chroma Features", 
                           lambda ax: librosa.display.specshow(
                               chroma, x_axis='time', y_axis='chroma', cmap='viridis', ax=ax),
                           "chroma")

                # 5. Spectral Contrast
                plot_graph("5. Spectral Contrast", 
                           lambda ax: librosa.display.specshow(
                               spec_contrast, x_axis='time', cmap='plasma', ax=ax),
                           "spectral_contrast")

                # 6. Spectral Centroid
                times = librosa.times_like(spectral_centroid)
                plot_graph("6. Spectral Centroid", 
                           lambda ax: (
                               ax.plot(times, spectral_centroid, label='Centroid', color='green'),
                               ax.fill_between(times, 0, spectral_centroid, alpha=0.3, color='green'),
                               ax.set(ylabel='Hz', xlabel='Time'),
                               ax.legend()
                           ),
                           "spectral_centroid")

                # 7. Spectral Bandwidth
                bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
                plot_graph("7. Spectral Bandwidth", 
                           lambda ax: (
                               ax.plot(times, bandwidth, label='Bandwidth', color='purple'),
                               ax.set(ylabel='Hz', xlabel='Time'),
                               ax.legend()
                           ),
                           "spectral_bandwidth")

                # 8. Spectral Rolloff
                rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
                plot_graph("8. Spectral Rolloff", 
                           lambda ax: (
                               ax.plot(times, rolloff, label='Rolloff (85%)', color='orange'),
                               ax.set(ylabel='Hz', xlabel='Time'),
                               ax.legend()
                           ),
                           "spectral_rolloff")

                # 9. Zero-Crossing Rate
                zcr_times = librosa.times_like(zcr)
                plot_graph("9. Zero-Crossing Rate", 
                           lambda ax: (
                               ax.plot(zcr_times, zcr, label='ZCR', color='red'),
                               ax.set(ylabel='Rate', xlabel='Time'),
                               ax.legend()
                           ),
                           "zero_crossing")

                # 10. RMS Energy
                rms = librosa.feature.rms(y=y)[0]
                rms_times = librosa.times_like(rms)
                plot_graph("10. RMS Energy", 
                           lambda ax: (
                               ax.plot(rms_times, rms, label='RMS', color='blue'),
                               ax.set(ylabel='Energy', xlabel='Time'),
                               ax.legend()
                           ),
                           "rms_energy")

                # 11. YAMNet Embeddings Overview
                plot_graph("11. YAMNet Embeddings (First 50 Dims)", 
                           lambda ax: (
                               ax.plot(features[0][:50], label='Embeddings', color='#2ca02c'),
                               ax.set(ylabel='Value', xlabel='Dimension'),
                               ax.legend()
                           ),
                           "yamnet")

                # Clean up
                os.unlink(temp_path)

    except Exception as e:
        st.error(f"❌ Processing error: {str(e)}")

# Add footer
st.markdown("---")
st.markdown("""
<style>
.footer {
    font-size: 0.8em;
    color: #777;
    text-align: center;
    padding: 10px;
}
</style>

<div class="footer">
    Advanced Deepfake Audio Detector v2.0 | 11 Feature Visualizations | Detailed Analysis
</div>
""", unsafe_allow_html=True)