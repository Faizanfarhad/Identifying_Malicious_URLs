from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from keras import models

from src.encoding.unicode_encoding import EncodingwithPadding


CLASS_LABELS = ["phishing", "benign", "defacement", "malware"]
CLASS_STYLES = {
    "benign": {"badge": "badge-benign", "emoji": "Safe"},
    "phishing": {"badge": "badge-phishing", "emoji": "Alert"},
    "defacement": {"badge": "badge-defacement", "emoji": "Warning"},
    "malware": {"badge": "badge-malware", "emoji": "Danger"},
}
MODEL_PATHS = [
    Path("savedModel/malicious_url_checker_model.keras"),
    Path("src/malicious_url_checker_model.keras"),
]


st.set_page_config(
    page_title="Malicious URL Detector",
    page_icon="🛡️",
    layout="wide",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=DM+Sans:wght@400;500;700&display=swap');

        :root {
            --bg-1: #07111f;
            --bg-2: #10253e;
            --panel: rgba(7, 17, 31, 0.72);
            --panel-strong: rgba(13, 30, 51, 0.88);
            --border: rgba(158, 208, 255, 0.18);
            --text: #f2f6fb;
            --muted: #a9bbcd;
            --accent: #61d0ff;
            --accent-2: #8cffc7;
            --danger: #ff7b7b;
            --warning: #ffbf69;
            --safe: #78f0b6;
        }

        .stApp {
            background:
                radial-gradient(circle at 20% 20%, rgba(97, 208, 255, 0.18), transparent 26%),
                radial-gradient(circle at 80% 10%, rgba(140, 255, 199, 0.14), transparent 22%),
                radial-gradient(circle at 50% 100%, rgba(75, 125, 255, 0.18), transparent 28%),
                linear-gradient(135deg, var(--bg-1), var(--bg-2));
            color: var(--text);
            font-family: "DM Sans", sans-serif;
        }

        .stApp::before,
        .stApp::after {
            content: "";
            position: fixed;
            inset: auto;
            width: 32rem;
            height: 32rem;
            border-radius: 50%;
            filter: blur(26px);
            opacity: 0.34;
            z-index: 0;
            pointer-events: none;
            animation: drift 12s ease-in-out infinite alternate;
        }

        .stApp::before {
            top: 8%;
            left: -8%;
            background: rgba(97, 208, 255, 0.18);
        }

        .stApp::after {
            right: -10%;
            bottom: 4%;
            background: rgba(140, 255, 199, 0.14);
            animation-duration: 15s;
        }

        @keyframes drift {
            from { transform: translate3d(0, 0, 0) scale(1); }
            to { transform: translate3d(40px, -20px, 0) scale(1.08); }
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        [data-testid="stSidebar"] {
            background: rgba(5, 12, 22, 0.72);
            border-right: 1px solid var(--border);
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            position: relative;
            z-index: 1;
        }

        .hero-card,
        .glass-card,
        .result-card {
            background: linear-gradient(180deg, rgba(14, 29, 49, 0.88), rgba(7, 17, 31, 0.8));
            border: 1px solid var(--border);
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.24);
            backdrop-filter: blur(12px);
            border-radius: 24px;
        }

        .hero-card {
            padding: 2rem;
            min-height: 220px;
        }

        .glass-card {
            padding: 1.25rem 1.25rem 1rem;
            height: 100%;
        }

        .result-card {
            padding: 1rem 1rem 0.9rem;
            margin-bottom: 0.9rem;
            transition: transform 0.2s ease, border-color 0.2s ease;
        }

        .result-card:hover {
            transform: translateY(-2px);
            border-color: rgba(158, 208, 255, 0.34);
        }

        .eyebrow {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.18em;
            color: var(--accent-2);
            margin-bottom: 0.6rem;
            font-weight: 700;
        }

        .hero-title {
            font-family: "Space Grotesk", sans-serif;
            font-size: clamp(2rem, 4vw, 3.4rem);
            line-height: 1;
            margin: 0 0 0.85rem 0;
        }

        .hero-copy,
        .card-copy {
            color: var(--muted);
            font-size: 1rem;
            line-height: 1.65;
        }

        .mini-stat {
            display: inline-block;
            padding: 0.75rem 1rem;
            margin: 0.4rem 0.6rem 0 0;
            border-radius: 16px;
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.06);
        }

        .mini-stat strong {
            display: block;
            font-size: 1.1rem;
            color: var(--text);
        }

        .panel-title {
            font-family: "Space Grotesk", sans-serif;
            font-size: 1.25rem;
            margin-bottom: 0.35rem;
        }

        .result-header {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            align-items: center;
            margin-bottom: 0.6rem;
        }

        .url-chip {
            color: var(--text);
            font-weight: 500;
            word-break: break-all;
            font-size: 0.98rem;
        }

        .badge {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            border: 1px solid transparent;
        }

        .badge-benign {
            background: rgba(120, 240, 182, 0.12);
            color: var(--safe);
            border-color: rgba(120, 240, 182, 0.18);
        }

        .badge-phishing {
            background: rgba(255, 123, 123, 0.12);
            color: #ff9d9d;
            border-color: rgba(255, 123, 123, 0.18);
        }

        .badge-defacement {
            background: rgba(255, 191, 105, 0.12);
            color: var(--warning);
            border-color: rgba(255, 191, 105, 0.18);
        }

        .badge-malware {
            background: rgba(255, 94, 94, 0.14);
            color: #ff7a7a;
            border-color: rgba(255, 94, 94, 0.18);
        }

        .confidence {
            color: var(--muted);
            font-size: 0.9rem;
        }

        .confidence strong {
            color: var(--text);
        }

        .stTextArea textarea {
            min-height: 200px;
            border-radius: 18px;
            border: 1px solid rgba(158, 208, 255, 0.16);
            background: rgba(4, 10, 19, 0.72);
            color: var(--text);
            font-size: 0.98rem;
        }

        .stButton button {
            border-radius: 999px;
            padding: 0.75rem 1.2rem;
            border: none;
            background: linear-gradient(135deg, #61d0ff, #8cffc7);
            color: #04111f;
            font-weight: 800;
        }

        .stDataFrame, div[data-testid="stMetric"] {
            background: transparent;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_model():
    for path in MODEL_PATHS:
        if path.exists():
            return models.load_model(path)
    raise FileNotFoundError(
        "Could not find the trained model. Expected one of: "
        + ", ".join(str(path) for path in MODEL_PATHS)
    )


def parse_urls(raw_text: str) -> list[str]:
    tokens = re.split(r"[\n,]+", raw_text)
    urls = []
    for token in tokens:
        cleaned = token.strip().strip('"').strip("'")
        if cleaned:
            urls.append(cleaned)
    return urls


def predict_urls(model, urls: list[str]) -> pd.DataFrame:
    encoded_urls = []
    for url in urls:
        encoded = EncodingwithPadding.encode_eachChar(url)
        encoded = np.array(encoded, dtype=np.int32)
        encoded_urls.append(encoded)

    batch = np.stack(encoded_urls, axis=0)
    probabilities = model.predict(batch, verbose=0)

    rows = []
    for url, probs in zip(urls, probabilities):
        label_index = int(np.argmax(probs))
        label = CLASS_LABELS[label_index]
        rows.append(
            {
                "url": url,
                "prediction": label,
                "confidence": float(probs[label_index]),
                "phishing_score": float(probs[0]),
                "benign_score": float(probs[1]),
                "defacement_score": float(probs[2]),
                "malware_score": float(probs[3]),
            }
        )
    return pd.DataFrame(rows)


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <div class="eyebrow">Threat Intelligence Dashboard</div>
            <h1 class="hero-title">Scan suspicious URLs with a cleaner, interactive interface.</h1>
            <p class="hero-copy">
                Paste one URL or a whole list, run the model, and review the predictions in a dedicated
                output panel with confidence scores and a quick summary.
            </p>
            <div class="mini-stat"><strong>4 Classes</strong> benign, phishing, defacement, malware</div>
            <div class="mini-stat"><strong>Batch Input</strong> newline or comma-separated URLs</div>
            <div class="mini-stat"><strong>Live Panel</strong> cards, table, summary chart</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_results(results: pd.DataFrame) -> None:
    safe_count = int((results["prediction"] == "benign").sum())
    risky_count = int(len(results) - safe_count)
    top_threat = results["prediction"].value_counts().idxmax()

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("URLs Scanned", len(results))
    metric_col2.metric("Potential Threats", risky_count)
    metric_col3.metric("Most Common Result", top_threat.title())

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="panel-title">Prediction Output Panel</div>
        <p class="card-copy">Each card shows the predicted class and the model confidence for that URL.</p>
        """,
        unsafe_allow_html=True,
    )

    for row in results.itertuples(index=False):
        style = CLASS_STYLES[row.prediction]
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-header">
                    <div class="url-chip">{row.url}</div>
                    <div class="badge {style["badge"]}">{style["emoji"]} {row.prediction}</div>
                </div>
                <div class="confidence">
                    Confidence: <strong>{row.confidence * 100:.2f}%</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    table_df = results.copy()
    table_df["confidence"] = table_df["confidence"].map(lambda value: f"{value * 100:.2f}%")
    st.dataframe(
        table_df[["url", "prediction", "confidence"]],
        use_container_width=True,
        hide_index=True,
    )

    summary = results["prediction"].value_counts().rename_axis("class").reset_index(name="count")
    st.bar_chart(summary.set_index("class"))


def main() -> None:
    inject_styles()
    render_hero()

    with st.sidebar:
        st.markdown("### Input Guide")
        st.write("Add one URL per line or separate multiple URLs with commas.")
        st.write("The app uses your saved Keras model and the same Unicode encoding flow from the notebook.")
        st.code("g00gle.com\nexample.com/login\npaypal-security-check.net")

    left_col, right_col = st.columns([1.1, 1], gap="large")

    with left_col:
        st.markdown(
            """
            <div class="glass-card">
                <div class="panel-title">URL Input</div>
                <p class="card-copy">Submit a single URL or batch-scan a list to review predictions together.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        raw_input = st.text_area(
            "URLs",
            placeholder="Enter one URL per line or comma-separated values...",
            label_visibility="collapsed",
        )
        run_prediction = st.button("Analyze URLs", use_container_width=True)

    with right_col:
        st.markdown(
            """
            <div class="glass-card">
                <div class="panel-title">Status</div>
                <p class="card-copy">Run the scan to populate the output panel with your results.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if not run_prediction:
        return

    urls = parse_urls(raw_input)
    if not urls:
        st.warning("Please enter at least one URL before running the scan.")
        return

    try:
        model = load_model()
        results = predict_urls(model, urls)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        return

    render_results(results)


if __name__ == "__main__":
    main()
