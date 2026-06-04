from datetime import datetime, timedelta

import pandas as pd
import streamlit as st


def init_state():
    if "analyses" not in st.session_state:
        st.session_state.analyses = []
    if "moderation_feed" not in st.session_state:
        st.session_state.moderation_feed = []
    if "monitoring_feed" not in st.session_state:
        st.session_state.monitoring_feed = []


def record_analysis(result, input_type):
    entry = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Date": datetime.now().strftime("%Y-%m-%d"),
        "Input Type": input_type,
        "Prediction": result["prediction"],
        "Severity": result["severity"],
        "Toxicity Score": result["toxicity_score"],
        "Emotion": result["emotion"],
        "Confidence": round(result["confidence"] * 100, 1),
        "Text": result["text"],
        "Keywords": ", ".join(result["keywords"]) if result["keywords"] else "None",
    }
    st.session_state.analyses.insert(0, entry)
    return entry


def dataframe():
    return pd.DataFrame(st.session_state.get("analyses", []))


def summary_counts():
    df = dataframe()
    counts = {"Normal": 0, "Abusive": 0, "Hate Speech": 0}
    if not df.empty:
        counts.update(df["Prediction"].value_counts().to_dict())
    return {
        "Total Analyses": len(df),
        "Normal Predictions": counts.get("Normal", 0),
        "Abusive Predictions": counts.get("Abusive", 0),
        "Hate Speech Predictions": counts.get("Hate Speech", 0),
    }


def distribution(column, labels):
    df = dataframe()
    if df.empty:
        return pd.DataFrame({"Label": labels, "Count": [0 for _ in labels]})
    counts = df[column].value_counts().reindex(labels, fill_value=0)
    return pd.DataFrame({"Label": counts.index, "Count": counts.values})


def weekly_trend():
    df = dataframe()
    dates = [(datetime.now() - timedelta(days=day)).strftime("%Y-%m-%d") for day in range(6, -1, -1)]
    if df.empty:
        return pd.DataFrame({"Date": dates, "Toxic Messages": [0] * 7, "Total": [0] * 7})

    grouped_total = df.groupby("Date").size().reindex(dates, fill_value=0)
    toxic_df = df[df["Prediction"].isin(["Abusive", "Hate Speech"])]
    grouped_toxic = toxic_df.groupby("Date").size().reindex(dates, fill_value=0)
    return pd.DataFrame(
        {
            "Date": dates,
            "Toxic Messages": grouped_toxic.values,
            "Total": grouped_total.values,
        }
    )


def report_csv():
    df = dataframe()
    if df.empty:
        return "No analyses available."
    return df.to_csv(index=False)
