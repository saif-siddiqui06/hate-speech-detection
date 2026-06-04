from pathlib import Path
import tempfile
import wave

import numpy as np
import streamlit as st


@st.cache_resource(show_spinner="Loading optional Whisper model...")
def load_whisper_model():
    try:
        import whisper
    except ImportError:
        return None

    return whisper.load_model("base")


def transcribe_audio_file(uploaded_file):
    suffix = Path(uploaded_file.name).suffix or ".wav"
    audio_bytes = uploaded_file.getvalue()
    return transcribe_audio_bytes(audio_bytes, suffix)


def transcribe_audio_bytes(audio_bytes, suffix=".wav"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(audio_bytes)
        temp_path = Path(temp_file.name)

    try:
        model = load_whisper_model()
        if model is not None:
            result = model.transcribe(str(temp_path), fp16=False)
            language = result.get("language", "unknown")
            return result.get("text", "").strip(), language

        return transcribe_with_speech_recognition(temp_path), "english"
    finally:
        temp_path.unlink(missing_ok=True)


def transcribe_with_speech_recognition(audio_path):
    import speech_recognition as sr
    from pydub import AudioSegment

    recognizer = sr.Recognizer()
    wav_path = audio_path

    if audio_path.suffix.lower() != ".wav":
        wav_path = audio_path.with_suffix(".wav")
        audio = AudioSegment.from_file(audio_path)
        audio = audio.set_channels(1).set_frame_rate(16000)
        audio.export(wav_path, format="wav")

    try:
        with sr.AudioFile(str(wav_path)) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.3)
            audio_data = recognizer.record(source)
        return recognizer.recognize_google(audio_data)
    finally:
        if wav_path != audio_path:
            wav_path.unlink(missing_ok=True)


def audio_metrics(audio_bytes, text, language="unknown"):
    duration = wav_duration(audio_bytes)
    words = len(text.split())
    speaking_rate = round((words / duration) * 60, 1) if duration else 0
    return {
        "Duration": f"{duration:.1f} sec" if duration else "Unknown",
        "Words Detected": words,
        "Speaking Speed": f"{speaking_rate} WPM" if speaking_rate else "Unknown",
        "Detected Language": language.title() if language else "Unknown",
    }


def wav_duration(audio_bytes):
    try:
        with tempfile.NamedTemporaryFile(delete=True, suffix=".wav") as temp_file:
            temp_file.write(audio_bytes)
            temp_file.flush()
            with wave.open(temp_file.name, "rb") as wav_file:
                frames = wav_file.getnframes()
                rate = wav_file.getframerate()
                return frames / float(rate)
    except Exception:
        return None


def waveform_dataframe(audio_bytes, max_points=240):
    data = np.frombuffer(audio_bytes, dtype=np.uint8).astype(float)
    if data.size == 0:
        return None

    chunk_size = max(1, data.size // max_points)
    trimmed = data[: chunk_size * min(max_points, data.size // chunk_size)]
    if trimmed.size == 0:
        return None

    waveform = trimmed.reshape(-1, chunk_size).mean(axis=1)
    waveform = (waveform - waveform.min()) / max(waveform.max() - waveform.min(), 1)
    return {
        "Frame": list(range(len(waveform))),
        "Amplitude": waveform.tolist(),
    }
