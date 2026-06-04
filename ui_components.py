import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from model_service import LABEL_COLORS


SEVERITY_COLORS = {
    "Low": "#22c55e",
    "Medium": "#f59e0b",
    "High": "#ef4444",
}

EMOTION_ICONS = {
    "Anger": "Anger",
    "Fear": "Fear",
    "Sadness": "Sadness",
    "Joy": "Joy",
    "Neutral": "Neutral",
}


def inject_dashboard_css():
    st.markdown(
        """
        <style>
            .stApp {
                background: #0b1120;
                color: #e5e7eb;
            }
            .block-container {
                max-width: 1220px;
                padding-top: 1.4rem;
                padding-bottom: 3rem;
            }
            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #020617 0%, #111827 100%);
                border-right: 1px solid rgba(148, 163, 184, 0.18);
            }
            h1, h2, h3 {
                color: #f8fafc;
                letter-spacing: 0;
            }
            .dashboard-hero {
                border: 1px solid rgba(148, 163, 184, 0.22);
                background:
                    radial-gradient(circle at top right, rgba(59, 130, 246, 0.22), transparent 28rem),
                    linear-gradient(135deg, #111827 0%, #172033 62%, #0f172a 100%);
                border-radius: 18px;
                padding: 28px;
                box-shadow: 0 24px 70px rgba(0, 0, 0, 0.35);
                margin-bottom: 20px;
            }
            .dashboard-hero h1 {
                margin: 0;
                font-size: 42px;
                line-height: 1.05;
            }
            .dashboard-hero p {
                color: #cbd5e1;
                max-width: 820px;
                margin-top: 12px;
                font-size: 16px;
            }
            .metric-card, .glass-card {
                border: 1px solid rgba(148, 163, 184, 0.22);
                background: rgba(15, 23, 42, 0.76);
                border-radius: 16px;
                padding: 18px;
                box-shadow: 0 16px 36px rgba(0, 0, 0, 0.24);
            }
            .metric-label {
                color: #94a3b8;
                font-size: 13px;
                margin-bottom: 6px;
            }
            .metric-value {
                color: #f8fafc;
                font-size: 30px;
                font-weight: 800;
            }
            .pipeline-grid {
                display: grid;
                grid-template-columns: repeat(5, minmax(0, 1fr));
                gap: 10px;
                margin: 14px 0;
            }
            .pipeline-step {
                background: #111827;
                border: 1px solid rgba(148, 163, 184, 0.22);
                border-radius: 14px;
                padding: 14px;
                min-height: 92px;
            }
            .step-id {
                color: #38bdf8;
                font-weight: 800;
                font-size: 13px;
            }
            .step-title {
                color: #f8fafc;
                font-weight: 700;
                margin-top: 6px;
            }
            .step-copy {
                color: #94a3b8;
                font-size: 13px;
                margin-top: 4px;
            }
            .result-card {
                border-radius: 18px;
                padding: 22px;
                border: 1px solid rgba(148, 163, 184, 0.25);
                background: rgba(15, 23, 42, 0.82);
                box-shadow: 0 20px 46px rgba(0, 0, 0, 0.25);
            }
            .tag {
                display: inline-block;
                padding: 7px 10px;
                margin: 4px 6px 4px 0;
                border-radius: 999px;
                background: rgba(239, 68, 68, 0.16);
                color: #fecaca;
                border: 1px solid rgba(248, 113, 113, 0.45);
                font-size: 13px;
                font-weight: 700;
            }
            .keyword-highlight {
                background: rgba(239, 68, 68, 0.34);
                color: #fff;
                padding: 2px 5px;
                border-radius: 5px;
            }
            .rewrite-card {
                border-left: 5px solid #38bdf8;
                background: rgba(14, 165, 233, 0.12);
                border-radius: 14px;
                padding: 18px;
                color: #e0f2fe;
            }
            .feed-card {
                border: 1px solid rgba(148, 163, 184, 0.22);
                background: rgba(15, 23, 42, 0.8);
                border-radius: 16px;
                padding: 16px;
                margin-bottom: 12px;
            }
            @media (max-width: 820px) {
                .dashboard-hero h1 { font-size: 30px; }
                .pipeline-grid { grid-template-columns: 1fr; }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero():
    st.markdown(
        """
        <section class="dashboard-hero">
            <h1>AI Content Moderation System</h1>
            <p>
                M.Tech-level hate speech detection dashboard for text and speech audio.
                It combines trained neural classification, speech-to-text, toxicity scoring,
                emotion analysis, keyword highlighting, and moderation-ready reporting.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def pipeline(steps):
    html = ['<div class="pipeline-grid">']
    for index, (title, copy) in enumerate(steps, start=1):
        html.append(
            f"""
            <div class="pipeline-step">
                <div class="step-id">STEP {index}</div>
                <div class="step-title">{title}</div>
                <div class="step-copy">{copy}</div>
            </div>
            """
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def pie_chart(df, title):
    fig = px.pie(df, names="Label", values="Count", hole=0.45, title=title)
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def bar_chart(df, x, y, title, color=None):
    fig = px.bar(df, x=x, y=y, title=title, color=color)
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def line_chart(df, x, y, title):
    fig = px.line(df, x=x, y=y, markers=True, title=title)
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def gauge(score):
    if score <= 30:
        color = "#22c55e"
    elif score <= 70:
        color = "#f59e0b"
    else:
        color = "#ef4444"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 30], "color": "rgba(34, 197, 94, 0.25)"},
                    {"range": [31, 70], "color": "rgba(245, 158, 11, 0.25)"},
                    {"range": [71, 100], "color": "rgba(239, 68, 68, 0.25)"},
                ],
            },
            title={"text": "Toxicity Score"},
        )
    )
    fig.update_layout(template="plotly_dark", height=270, paper_bgcolor="rgba(0,0,0,0)")
    return fig


def result_panel(result):
    prediction = result["prediction"]
    color = LABEL_COLORS[prediction]
    st.markdown(
        f"""
        <div class="result-card">
            <div style="color:#94a3b8;font-size:13px;">Prediction Result</div>
            <div style="font-size:34px;font-weight:850;color:{color};">{prediction}</div>
            <div style="color:#cbd5e1;margin-top:6px;">Confidence: {result["confidence"] * 100:.1f}%</div>
            <div style="color:#cbd5e1;margin-top:3px;">Severity: {result["severity"]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def emotion_chart(scores):
    df = pd.DataFrame({"Emotion": list(scores.keys()), "Probability": list(scores.values())})
    return bar_chart(df, "Emotion", "Probability", "Emotion Probability Chart", "Emotion")


def probability_chart(probabilities):
    df = pd.DataFrame({"Class": list(probabilities.keys()), "Probability": [v * 100 for v in probabilities.values()]})
    fig = px.bar(df, x="Class", y="Probability", color="Class", title="Prediction Probability")
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def keyword_tags(keywords):
    if not keywords:
        st.caption("No harmful keywords detected from the local keyword list.")
        return
    html = "".join(f"<span class='tag'>{keyword}</span>" for keyword in keywords)
    st.markdown(html, unsafe_allow_html=True)


def rewrite_card(text):
    st.markdown(
        f"""
        <div class="rewrite-card">
            <div style="font-weight:800;margin-bottom:8px;">Suggested Respectful Alternative</div>
            <div>{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
