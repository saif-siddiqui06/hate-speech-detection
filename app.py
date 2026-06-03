from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf


BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "Hate Speech" / "hate_speech_rnn_model.h5"
TOKENIZER_PATH = BASE_DIR / "Hate Speech" / "tokenizer.pkl"
MAX_SEQUENCE_LENGTH = 100

LABELS = {
    0: "Hate Speech",
    1: "Offensive Language",
    2: "Normal",
}


@st.cache_resource(show_spinner="Loading model...")
def load_model_and_tokenizer():
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


def predict_text(text, model, tokenizer):
    sequence = tokenizer.texts_to_sequences([text.strip()])
    padded = tf.keras.preprocessing.sequence.pad_sequences(
        sequence,
        maxlen=MAX_SEQUENCE_LENGTH,
        padding="post",
        truncating="post",
    )
    probabilities = model.predict(padded, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    confidence = float(np.max(probabilities))
    return LABELS[predicted_index], probabilities, confidence


def main():
    st.set_page_config(
        page_title="Hate Speech Detection",
        page_icon="!",
        layout="centered",
    )

    st.title("Hate Speech Detection")
    st.write("Classify text as hate speech, offensive language, or normal content.")

    model, tokenizer = load_model_and_tokenizer()

    text = st.text_area(
        "Enter text to analyze",
        placeholder="Type a sentence or message here...",
        height=140,
    )

    if st.button("Analyze", type="primary", use_container_width=True):
        if not text.strip():
            st.warning("Please enter some text first.")
            return

        label, probabilities, confidence = predict_text(text, model, tokenizer)

        st.subheader(label)
        st.metric("Confidence", f"{confidence:.1%}")

        chart_data = pd.DataFrame(
            {
                "Class": list(LABELS.values()),
                "Probability": probabilities,
            }
        ).set_index("Class")
        st.bar_chart(chart_data)

        with st.expander("Scores"):
            for index, class_name in LABELS.items():
                st.write(f"{class_name}: {probabilities[index]:.3f}")


if __name__ == "__main__":
    main()
