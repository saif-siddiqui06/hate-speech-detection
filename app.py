import pandas as pd
import streamlit as st

import analytics_service as analytics
import ui_components as ui
from audio_service import audio_metrics, transcribe_audio_bytes, transcribe_audio_file, waveform_dataframe
from model_service import LABEL_COLORS, classify_text, load_model_and_tokenizer


PAGES = [
    "Dashboard",
    "Text Analysis",
    "Audio Analysis",
    "Live Monitoring",
    "Analytics",
    "Social Media Moderator",
    "About Project",
]


def main():
    st.set_page_config(
        page_title="AI Hate Speech Moderation",
        page_icon="AI",
        layout="wide",
    )
    ui.inject_dashboard_css()
    analytics.init_state()
    model, tokenizer = load_model_and_tokenizer()

    page = sidebar()
    if page == "Dashboard":
        dashboard_page()
    elif page == "Text Analysis":
        text_analysis_page(model, tokenizer)
    elif page == "Audio Analysis":
        audio_analysis_page(model, tokenizer)
    elif page == "Live Monitoring":
        live_monitoring_page(model, tokenizer)
    elif page == "Analytics":
        analytics_page()
    elif page == "Social Media Moderator":
        social_moderator_page(model, tokenizer)
    else:
        about_page()


def sidebar():
    with st.sidebar:
        st.title("AI Moderator")
        st.caption("M.Tech Project Console")
        page = st.radio(
            "Navigation",
            PAGES,
            format_func=lambda item: {
                "Dashboard": "Dashboard",
                "Text Analysis": "Text Analysis",
                "Audio Analysis": "Audio Analysis",
                "Live Monitoring": "Live Monitoring",
                "Analytics": "Analytics",
                "Social Media Moderator": "Social Media Moderator",
                "About Project": "About Project",
            }[item],
        )
        st.divider()
        st.metric("Model Classes", 3)
        st.metric("Input Modes", 2)
        st.caption("Text | Audio | Moderation Simulation")
    return page


def dashboard_page():
    ui.hero()
    counts = analytics.summary_counts()
    metric_cols = st.columns(4)
    for column, (label, value) in zip(metric_cols, counts.items()):
        with column:
            ui.metric_card(label, value)

    chart_col, trend_col = st.columns([1, 1.25])
    with chart_col:
        distribution = analytics.distribution("Prediction", ["Normal", "Abusive", "Hate Speech"])
        st.plotly_chart(ui.pie_chart(distribution, "Prediction Distribution"), use_container_width=True)
    with trend_col:
        st.plotly_chart(ui.line_chart(analytics.weekly_trend(), "Date", "Total", "Daily Predictions Trend"), use_container_width=True)

    st.subheader("Recent Activity")
    df = analytics.dataframe()
    if df.empty:
        st.info("No analyses yet. Run text, audio, or moderator analysis to populate the dashboard.")
    else:
        st.dataframe(
            df[["Timestamp", "Input Type", "Prediction", "Severity", "Toxicity Score", "Emotion"]].head(10),
            use_container_width=True,
        )


def text_analysis_page(model, tokenizer):
    st.title("AI Hate Speech Analyzer")
    ui.pipeline(
        [
            ("Text Input", "User enters message or social media comment."),
            ("Preprocessing", "Text is converted into tokenizer sequence."),
            ("Sequence Padding", "Input is normalized to fixed length."),
            ("Prediction", "Trained model predicts content class."),
            ("Insights", "Severity, emotion, keywords, and rewrite are generated."),
        ]
    )

    text = st.text_area("Enter text to analyze", height=180, placeholder="Enter text to analyze...")
    if st.button("Analyze Text", type="primary", use_container_width=True):
        if not text.strip():
            st.warning("Please enter text before analysis.")
            return
        result = classify_text(text, model, tokenizer)
        analytics.record_analysis(result, "Text")
        render_full_analysis(result)


def audio_analysis_page(model, tokenizer):
    st.title("Speech-Based Hate Speech Detection")
    ui.pipeline(
        [
            ("Audio Uploaded", "Upload or record speech audio."),
            ("Speech-to-Text", "Whisper converts speech into text."),
            ("Text Cleaning", "Transcript is prepared for classifier."),
            ("Prediction", "Model classifies the transcript."),
            ("Results", "Moderation insights are displayed."),
        ]
    )

    upload_tab, record_tab = st.tabs(["Upload Audio", "Record Audio"])
    audio_bytes = None
    source_name = ""
    suffix = ".wav"

    with upload_tab:
        uploaded_file = st.file_uploader("Upload mp3, wav, or m4a audio", type=["mp3", "wav", "m4a"])
        if uploaded_file:
            audio_bytes = uploaded_file.getvalue()
            source_name = uploaded_file.name
            suffix = "." + uploaded_file.name.split(".")[-1].lower()
            st.audio(uploaded_file)

    with record_tab:
        if hasattr(st, "audio_input"):
            recorded_audio = st.audio_input("Record speech from microphone")
            if recorded_audio:
                audio_bytes = recorded_audio.getvalue()
                source_name = "microphone_recording.wav"
                suffix = ".wav"
                st.audio(recorded_audio)
        else:
            st.info("Microphone recording needs a newer Streamlit version. Please use audio upload.")

    if st.button("Analyze Audio", type="primary", use_container_width=True):
        if not audio_bytes:
            st.warning("Please upload or record audio first.")
            return

        with st.status("Running audio moderation pipeline...", expanded=True) as status:
            st.write("Audio received")
            st.write("Running speech-to-text")
            if uploaded_file and source_name == uploaded_file.name:
                transcript, language = transcribe_audio_file(uploaded_file)
            else:
                transcript, language = transcribe_audio_bytes(audio_bytes, suffix)
            st.write("Running hate speech classifier")
            result = classify_text(transcript, model, tokenizer)
            status.update(label="Audio pipeline completed", state="complete")

        analytics.record_analysis(result, "Audio")
        st.subheader("Transcribed Text")
        st.write(transcript)

        metric_cols = st.columns(4)
        for column, (label, value) in zip(metric_cols, audio_metrics(audio_bytes, transcript, language).items()):
            with column:
                ui.metric_card(label, value)

        wave_data = waveform_dataframe(audio_bytes)
        if wave_data:
            st.subheader("Audio Waveform Visualization")
            st.line_chart(pd.DataFrame(wave_data).set_index("Frame"))

        render_full_analysis(result)


def live_monitoring_page(model, tokenizer):
    st.title("Real-Time Speech Moderation")
    st.caption("Lightweight local demo: capture short microphone clips and analyze the latest statement.")

    if not hasattr(st, "audio_input"):
        st.warning("Your Streamlit version does not expose microphone recording. Use the Audio Analysis upload page.")
        return

    if st.button("Start Monitoring", type="primary"):
        st.session_state.monitoring_active = True

    if st.session_state.get("monitoring_active"):
        st.success("Listening... record a short statement below.")
        recorded_audio = st.audio_input("Live microphone sample")
        if recorded_audio:
            with st.spinner("Transcribing and analyzing current statement..."):
                transcript, language = transcribe_audio_bytes(recorded_audio.getvalue(), ".wav")
                result = classify_text(transcript, model, tokenizer)
                analytics.record_analysis(result, "Live Monitoring")
                st.session_state.monitoring_feed.insert(0, {
                    "Transcript": transcript,
                    "Risk Level": result["severity"],
                    "Prediction": result["prediction"],
                    "Language": language,
                })

            st.subheader("Current Transcript")
            st.write(transcript)
            st.subheader("Current Risk Level")
            st.markdown(
                f"<h2 style='color:{ui.SEVERITY_COLORS[result['severity']]};'>{result['severity']}</h2>",
                unsafe_allow_html=True,
            )
            render_full_analysis(result)

    st.subheader("Latest 10 Monitored Statements")
    feed = st.session_state.get("monitoring_feed", [])[:10]
    if feed:
        st.dataframe(pd.DataFrame(feed), use_container_width=True)
    else:
        st.info("No monitored statements yet.")


def analytics_page():
    st.title("Analytics Dashboard")
    prediction_df = analytics.distribution("Prediction", ["Normal", "Abusive", "Hate Speech"])
    severity_df = analytics.distribution("Severity", ["Low", "Medium", "High"])
    emotion_df = analytics.distribution("Emotion", ["Anger", "Fear", "Sadness", "Joy", "Neutral"])

    chart_col_1, chart_col_2 = st.columns(2)
    with chart_col_1:
        st.plotly_chart(ui.pie_chart(prediction_df, "Prediction Distribution"), use_container_width=True)
    with chart_col_2:
        st.plotly_chart(ui.bar_chart(severity_df, "Label", "Count", "Severity Distribution"), use_container_width=True)

    chart_col_3, chart_col_4 = st.columns(2)
    with chart_col_3:
        st.plotly_chart(ui.bar_chart(emotion_df, "Label", "Count", "Emotion Analysis"), use_container_width=True)
    with chart_col_4:
        st.plotly_chart(ui.line_chart(analytics.weekly_trend(), "Date", "Toxic Messages", "Weekly Toxic Messages Trend"), use_container_width=True)

    st.download_button(
        "Download Analysis Report",
        data=analytics.report_csv(),
        file_name="hate_speech_analysis_report.csv",
        mime="text/csv",
        use_container_width=True,
    )


def social_moderator_page(model, tokenizer):
    st.title("Social Media Moderator")
    comment = st.text_area("Post a comment...", height=120)
    if st.button("Submit Comment", type="primary", use_container_width=True):
        if not comment.strip():
            st.warning("Please enter a comment.")
            return
        result = classify_text(comment, model, tokenizer)
        analytics.record_analysis(result, "Social Media")
        st.session_state.moderation_feed.insert(0, result)

    for result in st.session_state.get("moderation_feed", []):
        badge = "Flagged Content" if result["prediction"] != "Normal" else "Approved"
        color = LABEL_COLORS[result["prediction"]]
        st.markdown(
            f"""
            <div class="feed-card">
                <div style="color:#475569;">User Comment</div>
                <div style="font-size:17px;margin:8px 0;color:#0f172a;">{result["text"]}</div>
                <div style="color:{color};font-weight:800;">{badge}</div>
                <div style="color:#334155;">Prediction: {result["prediction"]} | Severity: {result["severity"]} | Emotion: {result["emotion"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if result["rewrite"]:
            ui.rewrite_card(result["rewrite"])


def about_page():
    st.title("About Project")
    st.markdown(
        """
        ### AI-Powered Hate Speech Detection & Moderation System

        **M.Tech Project**

        This system demonstrates an end-to-end AI moderation workflow:

        - Text based hate speech classification
        - Speech audio to text conversion
        - Toxicity severity scoring
        - Emotion estimation
        - Offensive keyword detection
        - AI-style polite rewrite suggestions
        - Live monitoring prototype
        - Analytics and report export

        **Technologies Used**

        Python, Streamlit, TensorFlow/Keras, Machine Learning, Natural Language Processing,
        Whisper Speech Recognition, Pandas, and Plotly.

        **Responsible Use Note**

        The model output should be treated as decision support. Final moderation decisions
        should include human review, especially for sarcasm, code-mixed language, slang,
        or context-sensitive speech.
        """
    )


def render_full_analysis(result):
    left, right = st.columns([1, 1])
    with left:
        ui.result_panel(result)
        st.plotly_chart(ui.gauge(result["toxicity_score"]), use_container_width=True)
    with right:
        st.plotly_chart(ui.probability_chart(result["probabilities"]), use_container_width=True)

    st.subheader("Emotion Detection")
    emotion_col, chart_col = st.columns([0.85, 1.5])
    with emotion_col:
        ui.metric_card("Detected Emotion", result["emotion"])
    with chart_col:
        st.plotly_chart(ui.emotion_chart(result["emotion_scores"]), use_container_width=True)

    st.subheader("Detected Harmful Keywords")
    ui.keyword_tags(result["keywords"])

    st.subheader("Highlighted Input")
    st.markdown(result["highlighted_text"], unsafe_allow_html=True)

    if result["prediction"] != "Normal":
        st.subheader("AI Polite Rewriter")
        if st.button("Rewrite by AI", key=f"rewrite_{len(st.session_state.analyses)}"):
            ui.rewrite_card(result["rewrite"])
        else:
            ui.rewrite_card(result["rewrite"])


if __name__ == "__main__":
    main()
