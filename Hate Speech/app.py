
import os
import streamlit as st
import numpy as np
import tensorflow as tf
import speech_recognition as sr
from pydub import AudioSegment
import pickle
import time
import logging
from typing import Tuple, Optional
import plotly.express as px
import pandas as pd
import uuid
import io
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------
# Constants and Configuration
# ---------------------------
MAX_SEQUENCE_LENGTH = 100
SUPPORTED_AUDIO_FORMATS = ["wav", "mp3"]  # Limited to formats pydub handles well without ffmpeg
MAX_AUDIO_SIZE_MB = 25
TEMP_DIR = "temp_files"

# Label mapping with emojis
LABEL_MAP = {
    0: "Hate Speech", 
    1: "Offensive Language", 
    2: "Normal"
}

LABEL_COLORS = {
    "Hate Speech": "#e74c3c",      # Red
    "Offensive Language": "#f39c12", # Orange
    "Normal": "#27ae60"            # Green
}

LABEL_EMOJIS = {
    "Hate Speech": "🚫",
    "Offensive Language": "⚠️",
    "Normal": "✅"
}

# ---------------------------
# Utility Functions
# ---------------------------
def create_temp_dir():
    """Create temporary directory if it doesn't exist"""
    os.makedirs(TEMP_DIR, exist_ok=True)

def cleanup_temp_files():
    """Clean up old temporary files"""
    try:
        if os.path.exists(TEMP_DIR):
            for file in os.listdir(TEMP_DIR):
                file_path = os.path.join(TEMP_DIR, file)
                if os.path.isfile(file_path):
                    try:
                        if time.time() - os.path.getctime(file_path) > 3600:
                            os.remove(file_path)
                    except:
                        pass
    except Exception as e:
        logger.error(f"Error cleaning temp files: {e}")

def validate_audio_file(uploaded_file) -> Tuple[bool, str]:
    """Validate uploaded audio file"""
    if uploaded_file is None:
        return False, "No file uploaded"

    # Check file size
    file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
    if file_size_mb > MAX_AUDIO_SIZE_MB:
        return False, f"File size ({file_size_mb:.1f}MB) exceeds limit ({MAX_AUDIO_SIZE_MB}MB)"

    # Check file extension
    file_extension = uploaded_file.name.split(".")[-1].lower()
    if file_extension not in SUPPORTED_AUDIO_FORMATS:
        return False, f"Unsupported format. Supported: {', '.join(SUPPORTED_AUDIO_FORMATS).upper()}. Please convert to WAV or MP3."

    return True, "Valid file"

# ---------------------------
# Model Loading
# ---------------------------
@st.cache_resource(show_spinner="Loading AI model...")
def load_model_and_tokenizer():
    """Load the hate speech detection model and tokenizer"""
    try:
        if not os.path.exists("hate_speech_rnn_model.h5"):
            st.error("❌ Model file 'hate_speech_rnn_model.h5' not found!")
            st.info("Please ensure the model file is in the same directory as this app.")
            st.stop()

        if not os.path.exists("tokenizer.pkl"):
            st.error("❌ Tokenizer file 'tokenizer.pkl' not found!")
            st.info("Please ensure the tokenizer file is in the same directory as this app.")
            st.stop()

        model = tf.keras.models.load_model("hate_speech_rnn_model.h5")
        with open("tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)

        logger.info("Model and tokenizer loaded successfully")
        return model, tokenizer

    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

# ---------------------------
# Simple Audio Processing with Pydub Only
# ---------------------------
def convert_to_wav_simple(input_file: str) -> str:
    """Simple audio conversion using only pydub"""
    if input_file.lower().endswith(".wav"):
        return input_file

    output_file = os.path.join(TEMP_DIR, f"{uuid.uuid4().hex}.wav")

    try:
        # Get file extension
        file_ext = input_file.split('.')[-1].lower()

        # Load audio based on format
        if file_ext == 'mp3':
            audio = AudioSegment.from_mp3(input_file)
        elif file_ext == 'wav':
            audio = AudioSegment.from_wav(input_file)
        else:
            # Try generic loader
            audio = AudioSegment.from_file(input_file)

        # Convert to standard format: 16kHz, mono
        audio = audio.set_frame_rate(16000).set_channels(1)

        # Export as WAV
        audio.export(output_file, format="wav")

        if os.path.exists(output_file):
            return output_file
        else:
            raise Exception("Failed to create output file")

    except Exception as e:
        raise Exception(f"Audio conversion failed: {str(e)}. Please ensure the file is a valid {file_ext.upper()} file.")

@st.cache_data(ttl=300, show_spinner="Converting speech to text...")
def audio_to_text_simple(audio_file: str, language_preference: str = "auto") -> Tuple[str, str]:
    """Simple audio to text conversion"""
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 4000
    recognizer.dynamic_energy_threshold = True

    wav_file = None
    try:
        # Convert to WAV if needed
        wav_file = convert_to_wav_simple(audio_file)

        # Process audio
        with sr.AudioFile(wav_file) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio_data = recognizer.record(source)

        # Try different languages
        languages = ["en-US", "hi-IN"] if language_preference == "auto" else [language_preference]

        for lang in languages:
            try:
                text = recognizer.recognize_google(audio_data, language=lang)
                detected_lang = "English" if lang.startswith("en") else "Hindi"
                return text, detected_lang
            except sr.UnknownValueError:
                continue
            except sr.RequestError as e:
                return f"❌ Speech recognition service error: {str(e)}", "Unknown"

        return "❌ Could not understand the audio. Please speak clearly and try again.", "Unknown"

    except Exception as e:
        return f"❌ Audio processing error: {str(e)}", "Unknown"

    finally:
        # Cleanup
        if wav_file and wav_file != audio_file and os.path.exists(wav_file):
            try:
                os.remove(wav_file)
            except:
                pass

# ---------------------------
# Text Processing
# ---------------------------
def preprocess_and_predict(text: str, model, tokenizer) -> Tuple[str, np.ndarray, float]:
    """Text preprocessing and prediction"""
    try:
        text = text.strip()
        if not text:
            return "Normal", np.array([0, 0, 1]), 1.0

        sequences = tokenizer.texts_to_sequences([text])
        padded = tf.keras.preprocessing.sequence.pad_sequences(
            sequences, 
            maxlen=MAX_SEQUENCE_LENGTH, 
            padding="post", 
            truncating="post"
        )

        probabilities = model.predict(padded, verbose=0)[0]
        predicted_idx = np.argmax(probabilities)
        label = LABEL_MAP[predicted_idx]
        confidence = float(np.max(probabilities))

        return label, probabilities, confidence

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return "Error", np.array([0, 0, 0]), 0.0

# ---------------------------
# Visualization
# ---------------------------
def create_probability_chart(probabilities: np.ndarray) -> None:
    """Create probability visualization"""
    df = pd.DataFrame({
        'Classification': list(LABEL_MAP.values()),
        'Probability': probabilities,
        'Color': [LABEL_COLORS[label] for label in LABEL_MAP.values()]
    })

    fig = px.bar(
        df, 
        x='Probability', 
        y='Classification',
        orientation='h',
        color='Color',
        color_discrete_map={color: color for color in df['Color']},
        title="Classification Probabilities"
    )

    fig.update_layout(
        height=300,
        showlegend=False,
        title_x=0.5,
        xaxis=dict(range=[0, 1])
    )

    st.plotly_chart(fig, use_container_width=True)

def show_result(label: str, probabilities: np.ndarray, confidence: float, text: str = "") -> None:
    """Display results with visualization"""
    color = LABEL_COLORS[label]
    emoji = LABEL_EMOJIS[label]

    # Main result
    st.markdown(
        f"""
        <div style='
            padding: 20px; 
            border-radius: 10px; 
            background: linear-gradient(135deg, {color}20 0%, {color}30 100%);
            border-left: 5px solid {color};
            margin: 15px 0;
        '>
            <div style='text-align: center;'>
                <h2 style='color: {color}; margin: 0;'>
                    {emoji} {label}
                </h2>
                <p style='color: #666; margin: 10px 0 0 0;'>
                    Confidence: {confidence:.1%}
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Progress bar
    st.progress(confidence)

    # Detailed scores
    with st.expander("📊 Detailed Scores"):
        create_probability_chart(probabilities)

        for idx, lbl in LABEL_MAP.items():
            prob = probabilities[idx]
            st.write(f"{LABEL_EMOJIS[lbl]} **{lbl}**: {prob:.3f} ({prob:.1%})")

    # Text insights
    if text:
        with st.expander("📝 Text Analysis"):
            words = len(text.split())
            chars = len(text)
            st.write(f"- **Words**: {words}")
            st.write(f"- **Characters**: {chars}")
            st.write(f"- **Length**: {'Short' if words < 10 else 'Medium' if words < 50 else 'Long'}")

# ---------------------------
# App Interface
# ---------------------------
def main():
    """Main application"""
    st.set_page_config(
        page_title="🎤 Hate Speech Detector",
        page_icon="🎤",
        layout="centered"
    )

    create_temp_dir()
    cleanup_temp_files()

    # Load model
    model, tokenizer = load_model_and_tokenizer()

    # Header
    st.title("🎤 Hate Speech Detection")
    st.markdown(
        """
        <div style='text-align: center; padding: 15px; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                    border-radius: 10px; color: white; margin-bottom: 20px;'>
            <h3 style='margin: 0; color: white;'>AI-Powered Text & Audio Analysis</h3>
            <p style='margin: 5px 0 0 0; opacity: 0.9;'>
                Simple • Fast • Accurate
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        lang_pref = st.selectbox(
            "Language",
            ["auto", "en-US", "hi-IN"],
            format_func=lambda x: {"auto": "🔄 Auto", "en-US": "🇺🇸 English", "hi-IN": "🇮🇳 Hindi"}[x]
        )

        st.header("📊 Info")
        st.info("""
        **Supported Audio**: WAV, MP3
        **Max Size**: 25MB
        **Languages**: English, Hindi
        **Processing**: Pydub only
        """)

        st.header("💡 Tips")
        st.success("""
        - Use clear audio recordings
        - WAV format works best
        - Speak clearly and slowly
        - Avoid background noise
        """)

    # Main tabs
    tab1, tab2 = st.tabs(["📝 Text Analysis", "🎵 Audio Analysis"])

    with tab1:
        st.subheader("Text Input")

        user_text = st.text_area(
            "Enter text to analyze:",
            placeholder="Type your text here...",
            height=100
        )

        if st.button("🔍 Analyze Text", type="primary", use_container_width=True):
            if user_text.strip():
                with st.spinner("Analyzing..."):
                    label, probs, confidence = preprocess_and_predict(user_text, model, tokenizer)
                    show_result(label, probs, confidence, user_text)
            else:
                st.warning("Please enter some text")

    with tab2:
        st.subheader("Audio Input")

        # File uploader
        uploaded_file = st.file_uploader(
            "Upload audio file (WAV or MP3 only)",
            type=["wav", "mp3"],
            help="Maximum 25MB. WAV format recommended for best results."
        )

        if uploaded_file:
            # Validate
            is_valid, message = validate_audio_file(uploaded_file)

            if not is_valid:
                st.error(message)
            else:
                file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
                st.success(f"✅ {uploaded_file.name} ({file_size:.1f}MB)")

                # Audio player
                st.audio(uploaded_file)

                if st.button("🎯 Analyze Audio", type="primary", use_container_width=True):
                    temp_path = os.path.join(TEMP_DIR, f"{uuid.uuid4().hex}_{uploaded_file.name}")

                    try:
                        # Save file
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getvalue())

                        # Process
                        with st.spinner("Converting speech to text..."):
                            text, detected_lang = audio_to_text_simple(temp_path, lang_pref)

                        # Show results
                        if not text.startswith("❌"):
                            st.success(f"🎯 **Transcribed ({detected_lang})**: {text}")

                            with st.spinner("Analyzing text..."):
                                label, probs, confidence = preprocess_and_predict(text, model, tokenizer)
                                show_result(label, probs, confidence, text)
                        else:
                            st.error(text)
                            st.info("💡 Try uploading a WAV file or check audio quality")

                    except Exception as e:
                        st.error(f"Error: {str(e)}")

                    finally:
                        if os.path.exists(temp_path):
                            try:
                                os.remove(temp_path)
                            except:
                                pass

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <small>🔒 Privacy: Files processed locally • 🗑️ Auto-deleted after processing</small>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
