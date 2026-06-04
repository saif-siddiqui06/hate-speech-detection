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
                background:
                    radial-gradient(circle at top left, rgba(37, 99, 235, 0.14), transparent 28rem),
                    radial-gradient(circle at top right, rgba(20, 184, 166, 0.16), transparent 30rem),
                    linear-gradient(180deg, #f8fafc 0%, #e0f2fe 48%, #eef2ff 100%);
                color: #0f172a;
            }
            .block-container {
                max-width: 1220px;
                padding-top: 1.4rem;
                padding-bottom: 3rem;
            }
            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #0f172a 0%, #1e3a8a 100%);
                border-right: 1px solid rgba(255, 255, 255, 0.18);
            }
            h1, h2, h3 {
                color: #0f172a;
                letter-spacing: 0;
            }
            [data-testid="stSidebar"] h1,
            [data-testid="stSidebar"] h2,
            [data-testid="stSidebar"] h3,
            [data-testid="stSidebar"] label,
            [data-testid="stSidebar"] p,
            [data-testid="stSidebar"] span {
                color: #f8fafc;
            }
            .dashboard-hero {
                border: 1px solid rgba(30, 64, 175, 0.25);
                background:
                    radial-gradient(circle at top right, rgba(45, 212, 191, 0.42), transparent 23rem),
                    linear-gradient(135deg, #1d4ed8 0%, #2563eb 45%, #0891b2 100%);
                border-radius: 18px;
                padding: 28px;
                box-shadow: 0 24px 60px rgba(37, 99, 235, 0.22);
                margin-bottom: 20px;
            }
            .dashboard-hero h1 {
                margin: 0;
                font-size: 42px;
                line-height: 1.05;
                color: #ffffff;
            }
            .dashboard-hero p {
                color: #eff6ff;
                max-width: 820px;
                margin-top: 12px;
                font-size: 16px;
            }
            .metric-card, .glass-card {
                border: 1px solid rgba(37, 99, 235, 0.16);
                background: rgba(255, 255, 255, 0.92);
                border-radius: 16px;
                padding: 18px;
                box-shadow: 0 16px 34px rgba(15, 23, 42, 0.10);
            }
            .metric-label {
                color: #475569;
                font-size: 13px;
                margin-bottom: 6px;
            }
            .metric-value {
                color: #1d4ed8;
                font-size: 30px;
                font-weight: 800;
            }
            .pipeline-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(155px, 1fr));
                gap: 8px;
                margin: 10px 0 18px 0;
            }
            .pipeline-step {
                background: rgba(255, 255, 255, 0.94);
                border: 1px solid rgba(14, 165, 233, 0.22);
                border-radius: 12px;
                padding: 11px 12px;
                min-height: 0;
                box-shadow: 0 10px 26px rgba(15, 23, 42, 0.08);
            }
            .step-id {
                color: #0891b2;
                font-weight: 800;
                font-size: 11px;
            }
            .step-title {
                color: #0f172a;
                font-weight: 700;
                margin-top: 4px;
                font-size: 15px;
            }
            .step-copy {
                color: #475569;
                font-size: 12px;
                margin-top: 3px;
                line-height: 1.35;
            }
            .result-card {
                border-radius: 18px;
                padding: 22px;
                border: 1px solid rgba(37, 99, 235, 0.18);
                background: rgba(255, 255, 255, 0.95);
                box-shadow: 0 20px 42px rgba(15, 23, 42, 0.12);
            }
            .tag {
                display: inline-block;
                padding: 7px 10px;
                margin: 4px 6px 4px 0;
                border-radius: 999px;
                background: #fee2e2;
                color: #991b1b;
                border: 1px solid #fca5a5;
                font-size: 13px;
                font-weight: 700;
            }
            .keyword-highlight {
                background: #fecaca;
                color: #7f1d1d;
                padding: 2px 5px;
                border-radius: 5px;
                font-weight: 800;
            }
            .rewrite-card {
                border-left: 5px solid #0d9488;
                background: #ccfbf1;
                border-radius: 14px;
                padding: 18px;
                color: #134e4a;
            }
            .feed-card {
                border: 1px solid rgba(37, 99, 235, 0.16);
                background: rgba(255, 255, 255, 0.94);
                border-radius: 16px;
                padding: 16px;
                margin-bottom: 12px;
                box-shadow: 0 12px 28px rgba(15, 23, 42, 0.09);
            }
            div[data-testid="stDataFrame"],
            div[data-testid="stTable"] {
                background: rgba(255, 255, 255, 0.94);
                border-radius: 12px;
            }
            .stTabs [data-baseweb="tab-list"] {
                gap: 8px;
            }
            .stTabs [data-baseweb="tab"] {
                background: rgba(255, 255, 255, 0.78);
                border-radius: 999px;
                color: #1e293b;
                padding: 8px 16px;
                border: 1px solid rgba(37, 99, 235, 0.14);
            }
            .stTabs [aria-selected="true"] {
                background: #2563eb;
                color: #ffffff;
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
            f'<div class="pipeline-step">'
            f'<div class="step-id">STEP {index}</div>'
            f'<div class="step-title">{title}</div>'
            f'<div class="step-copy">{copy}</div>'
            f'</div>'
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def pie_chart(df, title):
    fig = px.pie(df, names="Label", values="Count", hole=0.45, title=title)
    fig.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def bar_chart(df, x, y, title, color=None):
    fig = px.bar(df, x=x, y=y, title=title, color=color)
    fig.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def line_chart(df, x, y, title):
    fig = px.line(df, x=x, y=y, markers=True, title=title)
    fig.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
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
    fig.update_layout(template="plotly_white", height=270, paper_bgcolor="rgba(0,0,0,0)")
    return fig


def result_panel(result):
    prediction = result["prediction"]
    color = LABEL_COLORS[prediction]
    st.markdown(
        f"""
        <div class="result-card">
            <div style="color:#475569;font-size:13px;">Prediction Result</div>
            <div style="font-size:34px;font-weight:850;color:{color};">{prediction}</div>
            <div style="color:#334155;margin-top:6px;">Confidence: {result["confidence"] * 100:.1f}%</div>
            <div style="color:#334155;margin-top:3px;">Severity: {result["severity"]}</div>
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
    fig.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
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
