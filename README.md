# AI Hate Speech Detection and Content Moderation System

This project is a Streamlit-based machine learning dashboard for detecting harmful text content. It supports direct text analysis and speech-audio analysis by converting audio into text before sending it to the trained hate speech detection model.

The application is designed as an M.Tech-level project demo with a clean dashboard UI, model prediction insights, visual analytics, and a social media moderation simulator.

## Live Demo

Try the deployed app here:

```text
https://hate-speech-detection-grmxl9ny798gwmzbu2pwwr.streamlit.app/
```

## Project Objective

Online platforms receive a large amount of user-generated content every day. Some of this content may be abusive, hateful, or unsafe for public discussion. This project demonstrates how a machine learning model can assist content moderation by classifying text into:

- `Normal`
- `Abusive`
- `Hate Speech`

The system is not meant to replace human moderation. It is a decision-support prototype that shows how AI can help identify risky content and present useful insights for review.

## Key Features

- Text-based hate speech detection
- Speech-audio input with speech-to-text transcription
- Toxicity score and severity level
- Prediction confidence and probability charts
- Emotion estimation
- Harmful keyword detection and highlighting
- Polite rewrite suggestion for harmful text
- Dashboard with recent activity and prediction distribution
- Analytics page with trends and downloadable report
- Social media moderation simulator
- Lightweight Streamlit deployment setup

## System Workflow

### Text Analysis Pipeline

```text
User Text
   -> Tokenization
   -> Sequence Padding
   -> Trained RNN Model
   -> Prediction and Insights
```

### Audio Analysis Pipeline

```text
Audio Input
   -> Temporary Audio Processing
   -> Speech-to-Text Transcription
   -> Text Classification
   -> Prediction and Insights
```

Both text and audio inputs use the same trained classifier. For audio, the transcribed text becomes the model input.

## Dashboard Pages

### Dashboard

Shows a high-level overview of the system:

- Total analyses
- Normal predictions
- Abusive predictions
- Hate speech predictions
- Prediction distribution chart
- Daily trend chart
- Recent activity table

### Text Analysis

Allows the user to enter text and analyze it using the model. The result includes:

- Predicted class
- Confidence score
- Toxicity score
- Severity category
- Emotion estimate
- Harmful keywords
- Highlighted input text
- Suggested respectful rewrite

### Audio Analysis

Allows the user to upload or record speech audio. The app transcribes the audio and classifies the resulting text. It also displays:

- Transcribed text
- Audio duration
- Words detected
- Speaking speed
- Detected language
- Waveform visualization
- Same prediction insights as text analysis

### Live Monitoring

Provides a lightweight prototype for real-time moderation using short microphone samples. Each recorded statement is transcribed, classified, and added to the latest monitored statements table.

### Analytics

Displays aggregated results from the current session:

- Prediction distribution
- Severity distribution
- Emotion distribution
- Weekly toxic message trend
- CSV report download

### Social Media Moderator

Simulates a social media comment section. When a comment is submitted, the system automatically analyzes it and displays whether it should be flagged.

## Technologies Used

- Python
- Streamlit
- TensorFlow / Keras
- Pandas
- NumPy
- Plotly
- OpenAI Whisper
- FFmpeg
- Natural Language Processing

## Project Structure

```text
hate-speech-detection/
├── app.py
├── streamlit_hate_speech_app.py
├── model_service.py
├── audio_service.py
├── analytics_service.py
├── ui_components.py
├── train_model.py
├── requirements.txt
├── packages.txt
├── runtime.txt
├── Dataset---Hate-Speech-Detection-using-Deep-Learning.csv
└── Hate Speech/
    ├── hate_speech_rnn_model.h5
    └── tokenizer.pkl
```

## Important Files

- `app.py`: Main Streamlit dashboard application
- `streamlit_hate_speech_app.py`: Compatibility entry point that runs `app.py`
- `model_service.py`: Loads the model, tokenizer, and performs prediction
- `audio_service.py`: Handles audio transcription and audio metrics
- `analytics_service.py`: Stores session analytics and report data
- `ui_components.py`: Reusable dashboard UI and chart components
- `train_model.py`: Training script for regenerating the model
- `requirements.txt`: Python dependencies
- `packages.txt`: Linux system dependency for Streamlit Cloud, mainly `ffmpeg`
- `runtime.txt`: Python runtime version for Streamlit Cloud

## Run Locally

Create and activate a virtual environment:

```powershell
python -m venv myenv
myenv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run the app:

```powershell
streamlit run app.py
```

Then open the local URL shown by Streamlit, usually:

```text
http://localhost:8501
```

## Deploy on Streamlit Community Cloud

1. Push the project to GitHub.
2. Go to `https://share.streamlit.io`.
3. Click `Create app`.
4. Select the GitHub repository.
5. Set the branch to `main`.
6. Set the main file path to `app.py`.
7. Deploy the app.

For this repository, use:

```text
Repository: saif-siddiqui06/hate-speech-detection
Branch: main
Main file path: app.py
```

## Model and Tokenizer Paths

The app expects the trained model at:

```text
Hate Speech/hate_speech_rnn_model.h5
```

The app expects the tokenizer at:

```text
Hate Speech/tokenizer.pkl
```

Do not remove or rename these files unless you also update the paths in `model_service.py`.

## Limitations

- The model prediction depends on the training dataset quality.
- Short or sarcastic text may be difficult to classify correctly.
- Audio results depend on transcription quality.
- MP3 and M4A support may require FFmpeg in the environment.
- Emotion detection and keyword detection are lightweight demo features, not full psychological analysis.
- Final moderation decisions should include human review.

## Future Improvements

- Add a larger multilingual dataset
- Add advanced text preprocessing
- Improve emotion detection using a trained emotion classifier
- Add database storage for historical analytics
- Add user authentication for moderation dashboards
- Add REST API support for external integration

## Author

M.Tech AI / Machine Learning Project

AI-Powered Hate Speech Detection and Moderation System
