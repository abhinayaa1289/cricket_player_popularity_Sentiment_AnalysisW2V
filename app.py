"""
🏏 Cricket Player Popularity & Sentiment Analyzer
--------------------------------------------------
A colorful, modern Streamlit app that:
  1. Looks up a player in the Cricket_Player_Popularity dataset and shows
     ALL their stored info (RecordID, Date, Team, Positive, Neutral,
     Negative, PopularityScore).
  2. Predicts overall sentiment (Positive / Neutral / Negative) using the
     trained GloVe + Neural Network model.

Required files in the same folder as this script:
    - Cricket_Player_Popularity.csv   (the dataset)
    - sentiment_model.keras           (trained Keras model)
    - scaler.pkl                      (fitted StandardScaler)
    - label_encoder.pkl               (fitted LabelEncoder)
    - embedding_lookup.pkl            (dict: word -> 200-d GloVe vector)

Run with:
    streamlit run app.py
"""

import os
import re
import pickle

import numpy as np
import pandas as pd
import streamlit as st

# Folder this script lives in — used so file lookups work no matter which
# directory `streamlit run` is launched from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------------------------------------------------------
# Page configuration
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Cricket Sentiment & Popularity",
    page_icon="🏏",
    layout="centered",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------------------------
# Custom CSS — gradients, glassmorphism, animated buttons
# -------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }

    .stApp {
        background: linear-gradient(-45deg, #6a11cb, #2575fc, #ff6a88, #22c1c3);
        background-size: 400% 400%;
        animation: gradientShift 16s ease infinite;
    }
    @keyframes gradientShift {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    #MainMenu, footer, header {visibility: hidden;}

    .glass-card {
        background: rgba(255, 255, 255, 0.16);
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        border-radius: 22px;
        border: 1px solid rgba(255, 255, 255, 0.35);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
        padding: 26px 28px;
        margin-bottom: 22px;
        animation: fadeInUp 0.6s ease;
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(18px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    .app-title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: 800;
        color: white;
        text-shadow: 0 4px 18px rgba(0,0,0,0.35);
        margin-bottom: 0;
    }
    .app-subtitle {
        text-align: center;
        color: rgba(255,255,255,0.9);
        font-size: 1.02rem;
        margin-top: 4px;
        margin-bottom: 26px;
    }

    .stTextInput > div > div > input {
        background: rgba(255,255,255,0.85);
        border-radius: 14px;
        border: none;
        padding: 12px 16px;
        font-size: 1.05rem;
        font-weight: 600;
        color: #2b2b40;
    }

    div.stButton > button {
        width: 100%;
        border: none;
        border-radius: 14px;
        padding: 0.85em 1.2em;
        font-size: 1.05rem;
        font-weight: 700;
        color: white;
        background: linear-gradient(90deg, #ff6a88, #ff9a44, #6a11cb, #2575fc);
        background-size: 300% 300%;
        animation: gradientShift 6s ease infinite;
        box-shadow: 0 6px 20px rgba(0,0,0,0.25);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 10px 26px rgba(0,0,0,0.35);
        color: white;
    }
    div.stButton > button:active { transform: translateY(0px) scale(0.98); }

    .sentiment-badge {
        display: inline-block;
        padding: 10px 26px;
        border-radius: 999px;
        font-size: 1.4rem;
        font-weight: 800;
        color: white;
        box-shadow: 0 6px 18px rgba(0,0,0,0.3);
        animation: pop 0.45s ease;
    }
    @keyframes pop {
        0%   { transform: scale(0.7); opacity: 0; }
        70%  { transform: scale(1.08); opacity: 1; }
        100% { transform: scale(1); }
    }

    .prob-label { color: white; font-weight: 600; font-size: 0.92rem; margin-bottom: 2px; }
    .prob-track {
        width: 100%; height: 14px; border-radius: 10px;
        background: rgba(255,255,255,0.25); overflow: hidden; margin-bottom: 14px;
    }
    .prob-fill { height: 100%; border-radius: 10px; transition: width 1s ease-in-out; }

    /* Stat mini-cards */
    .stat-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
        gap: 14px;
        margin: 10px 0 4px 0;
    }
    .stat-box {
        background: rgba(255,255,255,0.14);
        border: 1px solid rgba(255,255,255,0.3);
        border-radius: 16px;
        padding: 14px 10px;
        text-align: center;
        color: white;
        animation: fadeInUp 0.5s ease;
    }
    .stat-value { font-size: 1.4rem; font-weight: 800; }
    .stat-label { font-size: 0.78rem; opacity: 0.85; margin-top: 2px; }

    .footer-note {
        text-align: center;
        color: rgba(255,255,255,0.75);
        font-size: 0.8rem;
        margin-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------
# Header
# -------------------------------------------------------------------
st.markdown('<div class="app-title">🏏 Cricket Popularity & Sentiment Analyzer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Search a player to see their full stats + AI-predicted sentiment ✨</div>',
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------
EMBED_DIM = 200

# Primary CSV location (your machine). Falls back to a CSV sitting next to
# app.py if this exact path isn't found — so the app still works if you move
# the folder or share it with someone else.
_HARDCODED_CSV_PATH = r"C:\Users\abhin\OneDrive\Desktop\w2v\Cricket_Player_Popularity_updated.csv"
_FALLBACK_CSV_PATH = os.path.join(BASE_DIR, "Cricket_Player_Popularity_updated.csv")
CSV_PATH = _HARDCODED_CSV_PATH if os.path.exists(_HARDCODED_CSV_PATH) else _FALLBACK_CSV_PATH

MODEL_PATH = os.path.join(BASE_DIR, "sentiment_model.keras")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")
EMBEDDING_PATH = os.path.join(BASE_DIR, "embedding_lookup.pkl")

SENTIMENT_COLORS = {"Positive": "#2ecc71", "Neutral": "#f1c40f", "Negative": "#e74c3c"}
SENTIMENT_EMOJI = {"Positive": "😃", "Neutral": "😐", "Negative": "😞"}


# -------------------------------------------------------------------
# Text cleaning — identical logic to the training notebook
# -------------------------------------------------------------------
def clean_text(text: str):
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"@\w+", " @ ", text)
    text = re.sub(r"[^a-z@ ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()


def sentence_vector(text: str, embedding_lookup: dict, dim: int = EMBED_DIM) -> np.ndarray:
    words = clean_text(text)
    vectors = [embedding_lookup[w] for w in words if w in embedding_lookup]
    if not vectors:
        return np.zeros(dim)
    return np.mean(vectors, axis=0)


# -------------------------------------------------------------------
# Cached loaders
# -------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model_artifacts():
    import tensorflow as tf

    required = {
        "sentiment_model.keras": MODEL_PATH,
        "scaler.pkl": SCALER_PATH,
        "label_encoder.pkl": ENCODER_PATH,
        "embedding_lookup.pkl": EMBEDDING_PATH,
    }
    missing = [name for name, path in required.items() if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            "Missing required file(s) in "
            f"{BASE_DIR}: {', '.join(missing)}"
        )

    model = tf.keras.models.load_model(MODEL_PATH)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f:
        encoder = pickle.load(f)
    with open(EMBEDDING_PATH, "rb") as f:
        embedding_lookup = pickle.load(f)
    return model, scaler, encoder, embedding_lookup


@st.cache_data(show_spinner=False)
def load_dataset():
    df = pd.read_csv(CSV_PATH)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


# -------------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ℹ️ About")
    st.write(
        "Search any player name to pull up every record stored for them "
        "(mentions, sentiment counts, popularity score) plus an AI sentiment "
        "prediction based on a GloVe + Neural Network model."
    )
    st.markdown("### 🎨 Legend")
    for label, color in SENTIMENT_COLORS.items():
        st.markdown(
            f'<span style="background:{color};padding:4px 10px;border-radius:8px;'
            f'color:white;font-weight:600;">{SENTIMENT_EMOJI[label]} {label}</span>',
            unsafe_allow_html=True,
        )
    st.markdown("---")
    st.caption(f"Model files expected in: {BASE_DIR}")

# -------------------------------------------------------------------
# Search input
# -------------------------------------------------------------------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
player_name = st.text_input("Enter a cricket player's name", placeholder="e.g. Virat Kohli")
search = st.button("🔍 Get Player Insights")
st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------------------------
# Main logic
# -------------------------------------------------------------------
if search:
    if not player_name.strip():
        st.warning("Please enter a player name first! 🙂")
    else:
        # ---- Load dataset (optional but needed for stats lookup) ----
        try:
            df = load_dataset()
            matches = df[df["Player"].str.contains(player_name.strip(), case=False, na=False)]
        except FileNotFoundError:
            df = None
            matches = pd.DataFrame()
            st.info(f"ℹ️ Dataset file '{CSV_PATH}' not found — showing AI prediction only.")

        # ================= DATASET RESULTS =================
        if df is not None:
            if matches.empty:
                st.warning(f"No records found for **{player_name}** in the dataset. Showing AI prediction only.")
            else:
                team = matches["Team"].mode().iloc[0] if not matches["Team"].mode().empty else "N/A"
                total_pos = int(matches["Positive"].sum())
                total_neu = int(matches["Neutral"].sum())
                total_neg = int(matches["Negative"].sum())
                avg_pop = matches["PopularityScore"].mean()
                record_count = len(matches)
                date_min = matches["Date"].min()
                date_max = matches["Date"].max()

                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown(
                    f'<div style="color:white;font-size:1.3rem;font-weight:800;">'
                    f'🧑\u200d🦱 {matches["Player"].iloc[0]} &nbsp; '
                    f'<span style="font-size:0.9rem;font-weight:500;opacity:0.85;">({team})</span></div>',
                    unsafe_allow_html=True,
                )

                st.markdown('<div class="stat-grid">', unsafe_allow_html=True)
                stat_html = ""
                stat_items = [
                    ("Records", record_count),
                    ("Positive", total_pos),
                    ("Neutral", total_neu),
                    ("Negative", total_neg),
                    ("Avg. Popularity", f"{avg_pop:.1f}"),
                ]
                for label, val in stat_items:
                    stat_html += (
                        f'<div class="stat-box"><div class="stat-value">{val}</div>'
                        f'<div class="stat-label">{label}</div></div>'
                    )
                st.markdown(stat_html, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

                if pd.notnull(date_min) and pd.notnull(date_max):
                    st.markdown(
                        f'<div style="color:rgba(255,255,255,0.85);font-size:0.85rem;margin-top:6px;">'
                        f'📅 Data range: {date_min.date()} → {date_max.date()}</div>',
                        unsafe_allow_html=True,
                    )
                st.markdown("</div>", unsafe_allow_html=True)

                # ---- Popularity trend chart ----
                if record_count > 1:
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    st.markdown(
                        '<div style="color:white;font-weight:700;margin-bottom:8px;">📈 Popularity Score Over Time</div>',
                        unsafe_allow_html=True,
                    )
                    trend = matches.sort_values("Date").set_index("Date")["PopularityScore"]
                    st.line_chart(trend)
                    st.markdown("</div>", unsafe_allow_html=True)

                # ---- Raw records table ----
                with st.expander(f"📋 View all {record_count} raw record(s)"):
                    st.dataframe(
                        matches[
                            ["RecordID", "Date", "Player", "Team", "Positive", "Neutral", "Negative", "PopularityScore"]
                        ].sort_values("Date"),
                        use_container_width=True,
                    )

        # ================= AI SENTIMENT PREDICTION =================
        try:
            model, scaler, encoder, embedding_lookup = load_model_artifacts()
        except FileNotFoundError as e:
            st.error(
                "⚠️ Couldn't load the sentiment model. "
                f"{e}"
            )
            st.stop()

        with st.spinner("Reading the crowd's mood... 🏟️"):
            vec = sentence_vector(player_name, embedding_lookup).reshape(1, -1)
            vec_scaled = scaler.transform(vec)
            probs = model.predict(vec_scaled, verbose=0)[0]
            pred_idx = int(np.argmax(probs))
            pred_label = encoder.classes_[pred_idx]

        color = SENTIMENT_COLORS.get(pred_label, "#888888")
        emoji = SENTIMENT_EMOJI.get(pred_label, "")

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown(
            '<div style="color:white;font-weight:700;margin-bottom:10px;">🤖 AI-Predicted Sentiment</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="text-align:center;">'
            f'<span class="sentiment-badge" style="background:{color};">{emoji} {pred_label}</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        for label, prob in sorted(zip(encoder.classes_, probs), key=lambda x: -x[1]):
            bar_color = SENTIMENT_COLORS.get(label, "#888888")
            pct = prob * 100
            st.markdown(
                f'<div class="prob-label">{SENTIMENT_EMOJI.get(label, "")} {label} — {pct:.1f}%</div>'
                f'<div class="prob-track"><div class="prob-fill" '
                f'style="width:{pct:.1f}%;background:{bar_color};"></div></div>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    '<div class="footer-note">Built with Streamlit · GloVe Embeddings · Keras Neural Network</div>',
    unsafe_allow_html=True,
)