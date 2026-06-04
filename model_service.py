from pathlib import Path
import pickle
import re
from html import escape

import numpy as np
import streamlit as st


BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "Hate Speech" / "hate_speech_rnn_model.h5"
TOKENIZER_PATH = BASE_DIR / "Hate Speech" / "tokenizer.pkl"
MAX_SEQUENCE_LENGTH = 100

MODEL_LABELS = {
    0: "Hate Speech",
    1: "Abusive",
    2: "Normal",
}

LABEL_COLORS = {
    "Normal": "#22c55e",
    "Abusive": "#f59e0b",
    "Hate Speech": "#ef4444",
}

HARMFUL_KEYWORDS = {
    "idiot",
    "stupid",
    "dumb",
    "worthless",
    "hate",
    "trash",
    "moron",
    "fool",
    "shut",
    "kill",
    "ugly",
    "loser",
}

EMOTION_KEYWORDS = {
    "Anger": {"hate", "angry", "stupid", "idiot", "trash", "shut", "kill", "moron"},
    "Fear": {"afraid", "fear", "scared", "threat", "danger", "unsafe"},
    "Sadness": {"sad", "cry", "hurt", "worthless", "alone", "depressed"},
    "Joy": {"happy", "great", "good", "love", "nice", "thanks", "excellent"},
}


@st.cache_resource(show_spinner="Loading trained hate speech classifier...")
def load_model_and_tokenizer():
    try:
        import tensorflow as tf
    except ImportError:
        st.warning(
            "TensorFlow is not available in this environment. "
            "The app is using a lightweight demo classifier so the dashboard can run on Streamlit Cloud."
        )
        return None, None

    if not MODEL_PATH.exists():
        st.error(f"Model file not found: {MODEL_PATH}")
        st.stop()

    if not TOKENIZER_PATH.exists():
        st.error(f"Tokenizer file not found: {TOKENIZER_PATH}")
        st.stop()

    model = tf.keras.models.load_model(MODEL_PATH)
    with TOKENIZER_PATH.open("rb") as file:
        tokenizer = pickle.load(file)
    return model, tokenizer


def classify_text(text, model, tokenizer):
    clean_text = text.strip()
    if model is None or tokenizer is None:
        return classify_text_lightweight(clean_text)

    import tensorflow as tf

    sequence = tokenizer.texts_to_sequences([clean_text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(
        sequence,
        maxlen=MAX_SEQUENCE_LENGTH,
        padding="post",
        truncating="post",
    )
    probabilities = model.predict(padded, verbose=0)[0]
    prediction_index = int(np.argmax(probabilities))
    prediction = MODEL_LABELS[prediction_index]
    confidence = float(probabilities[prediction_index])
    toxicity_score = calculate_toxicity_score(prediction, probabilities)
    severity = get_severity(toxicity_score)
    emotion, emotion_scores = detect_emotion(clean_text)
    keywords = detect_harmful_keywords(clean_text)

    return {
        "text": clean_text,
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": {
            MODEL_LABELS[index]: float(probability)
            for index, probability in enumerate(probabilities)
        },
        "toxicity_score": toxicity_score,
        "severity": severity,
        "emotion": emotion,
        "emotion_scores": emotion_scores,
        "keywords": keywords,
        "highlighted_text": highlight_keywords(clean_text, keywords),
        "rewrite": polite_rewrite(clean_text, prediction),
    }


def calculate_toxicity_score(prediction, probabilities):
    hate = float(probabilities[0])
    abusive = float(probabilities[1])
    normal = float(probabilities[2])
    if prediction == "Normal":
        return round(max(0.0, (1.0 - normal) * 45), 1)
    return round(min(100.0, (hate * 100) + (abusive * 72)), 1)


def classify_text_lightweight(text):
    keywords = detect_harmful_keywords(text)
    lower_text = text.lower()
    threat_terms = {"kill", "hate", "worthless"}
    strong_hits = len(set(re.findall(r"\b[\w']+\b", lower_text)).intersection(threat_terms))

    if strong_hits:
        probabilities = {"Hate Speech": 0.72, "Abusive": 0.22, "Normal": 0.06}
        prediction = "Hate Speech"
    elif keywords:
        probabilities = {"Hate Speech": 0.18, "Abusive": 0.68, "Normal": 0.14}
        prediction = "Abusive"
    else:
        probabilities = {"Hate Speech": 0.05, "Abusive": 0.12, "Normal": 0.83}
        prediction = "Normal"

    confidence = probabilities[prediction]
    toxicity_score = calculate_toxicity_score(
        prediction,
        [probabilities["Hate Speech"], probabilities["Abusive"], probabilities["Normal"]],
    )
    severity = get_severity(toxicity_score)
    emotion, emotion_scores = detect_emotion(text)

    return {
        "text": text,
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": probabilities,
        "toxicity_score": toxicity_score,
        "severity": severity,
        "emotion": emotion,
        "emotion_scores": emotion_scores,
        "keywords": keywords,
        "highlighted_text": highlight_keywords(text, keywords),
        "rewrite": polite_rewrite(text, prediction),
    }


def get_severity(score):
    if score <= 30:
        return "Low"
    if score <= 70:
        return "Medium"
    return "High"


def detect_harmful_keywords(text):
    words = re.findall(r"\b[\w']+\b", text.lower())
    return sorted({word for word in words if word in HARMFUL_KEYWORDS})


def highlight_keywords(text, keywords):
    highlighted = escape(text)
    for keyword in sorted(keywords, key=len, reverse=True):
        pattern = re.compile(rf"\b({re.escape(keyword)})\b", re.IGNORECASE)
        highlighted = pattern.sub(
            r"<mark class='keyword-highlight'>\1</mark>",
            highlighted,
        )
    return highlighted


def detect_emotion(text):
    tokens = set(re.findall(r"\b[\w']+\b", text.lower()))
    raw_scores = {}
    for emotion, words in EMOTION_KEYWORDS.items():
        raw_scores[emotion] = len(tokens.intersection(words))

    if sum(raw_scores.values()) == 0:
        raw_scores["Neutral"] = 4
    else:
        raw_scores["Neutral"] = 1

    total = sum(raw_scores.values()) or 1
    scores = {
        emotion: round((score / total) * 100, 1)
        for emotion, score in raw_scores.items()
    }
    detected = max(scores, key=scores.get)
    return detected, scores


def polite_rewrite(text, prediction):
    if prediction == "Normal":
        return ""

    lower_text = text.lower()
    if any(word in lower_text for word in ["stupid", "idiot", "dumb", "moron", "fool"]):
        return "I respectfully disagree with your opinion and would like to discuss it constructively."
    if any(word in lower_text for word in ["hate", "trash", "worthless"]):
        return "I have serious concerns about this, but I will express them respectfully."
    return "I disagree with this viewpoint, but I will communicate my response in a respectful way."
