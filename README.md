# Hate Speech Detection

A Streamlit machine learning app that classifies text as:

- Hate Speech
- Offensive Language
- Normal

## Run locally

```powershell
python -m venv myenv
myenv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this project to GitHub.
2. Go to `https://share.streamlit.io`.
3. Select the GitHub repository.
4. Set the main file path to `app.py`.
5. Deploy.

The trained model is expected at:

```text
Hate Speech/hate_speech_rnn_model.h5
```

The tokenizer is expected at:

```text
Hate Speech/tokenizer.pkl
```

## Dashboard Features

- Dashboard overview with total analyses, class distribution, trends, and recent activity
- Text analysis with toxicity score, severity level, emotion analysis, harmful keyword detection, and polite rewrite suggestion
- Audio analysis with upload/record input, speech-to-text transcription, waveform preview, audio metrics, and model prediction
- Live monitoring prototype for short microphone samples
- Analytics dashboard with prediction, severity, emotion, and weekly trend charts
- Social media moderation simulator with flagged comment cards
- About page with methodology, technologies, and responsible use notes
