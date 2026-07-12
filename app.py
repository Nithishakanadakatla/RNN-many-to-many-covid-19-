"""
app.py
------
Streamlit app for the Many-to-Many Simple RNN COVID-19 forecaster.
The user manually enters the previous 7 days of case counts and gets
the next 7 days predicted, with a polished, card-based UI.

Run with:
    streamlit run app.py
"""

import pickle
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from tensorflow import keras

MODEL_PATH = "covid_rnn_model.keras"
SCALER_PATH = "scaler.pkl"
DEFAULT_CSV = "day_wise.csv"

st.set_page_config(
    page_title="COVID-19 RNN Forecaster",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #0f1420 0%, #141b2d 100%);
    }

    .hero {
        padding: 2rem 2.2rem;
        border-radius: 18px;
        background: linear-gradient(120deg, #4b6cb7 0%, #834d9b 100%);
        color: white;
        margin-bottom: 1.6rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.25);
    }
    .hero h1 {
        margin: 0 0 0.3rem 0;
        font-size: 2.1rem;
        font-weight: 800;
    }
    .hero p {
        margin: 0;
        font-size: 1.02rem;
        opacity: 0.92;
    }

    .section-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.2rem;
    }
    .section-card h3 {
        margin-top: 0;
    }

    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 0.8rem 0.6rem;
    }

    .stButton>button {
        background: linear-gradient(120deg, #ff5f6d 0%, #ffc371 100%);
        color: #201400;
        font-weight: 700;
        border: none;
        border-radius: 12px;
        padding: 0.6rem 1.4rem;
        font-size: 1.02rem;
        transition: transform 0.15s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px) scale(1.02);
        color: #201400;
    }

    .day-label {
        font-size: 0.82rem;
        opacity: 0.75;
        margin-bottom: -0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model_and_scaler():
    model = keras.models.load_model(MODEL_PATH)
    with open(SCALER_PATH, "rb") as f:
        cfg = pickle.load(f)
    return model, cfg


@st.cache_data
def load_default_history():
    """Used only to pre-fill sensible starting numbers for the input boxes."""
    try:
        df = pd.read_csv(DEFAULT_CSV).sort_values("Date").reset_index(drop=True)
        return df
    except Exception:
        return None


def predict_next_days(model, cfg, last_window_raw: np.ndarray) -> np.ndarray:
    scaler = cfg["scaler"]
    n_in = cfg["n_steps_in"]

    window_scaled = scaler.transform(last_window_raw.reshape(-1, 1))
    X = window_scaled.reshape(1, n_in, 1)

    pred_scaled = model.predict(X, verbose=0)
    pred_raw = scaler.inverse_transform(pred_scaled)
    return pred_raw.flatten()


def main():
    st.markdown(
        """
        <div class="hero">
            <h1>🦠 COVID-19 Forecaster</h1>
            <p>Many-to-Many Simple RNN &nbsp;•&nbsp; type in 7 days of case counts,
            get the next 7 days predicted instantly.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        model, cfg = load_model_and_scaler()
    except Exception as e:
        st.error(
            "Could not load the trained model. Run `python train.py` first "
            f"to generate '{MODEL_PATH}' and '{SCALER_PATH}'.\n\nDetails: {e}"
        )
        return

    feature_col = cfg["feature_col"]
    n_in = cfg["n_steps_in"]
    n_out = cfg["n_steps_out"]

    default_df = load_default_history()
    default_values = [0] * n_in
    if default_df is not None and feature_col in default_df.columns and len(default_df) >= n_in:
        default_values = default_df[feature_col].values[-n_in:].astype(int).tolist()

    # ---------------- Sidebar ----------------
    with st.sidebar:
        st.header("⚙️ Settings")
        st.caption(f"Model predicts **{feature_col}** ({n_in} days in → {n_out} days out)")
        start_date = st.date_input("Date of Day 1 (first input day)", value=datetime.today() - timedelta(days=7))
        if st.button("↻ Reset to sample values"):
            for i in range(n_in):
                st.session_state[f"day_{i}"] = default_values[i]
            st.rerun()
        st.divider()
        st.caption("Model architecture: SimpleRNN(64) → SimpleRNN(32) → Dense(32) → Dense(7)")

    # ---------------- Manual input section ----------------
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown(f"### ✍️ Enter the last {n_in} days of **{feature_col}**")
    st.caption("Type your own numbers below, or use the sidebar to reset to sample data.")

    cols = st.columns(n_in)
    user_values = []
    for i, col in enumerate(cols):
        day_date = start_date + timedelta(days=i)
        with col:
            st.markdown(f'<div class="day-label">{day_date.strftime("%b %d")}</div>', unsafe_allow_html=True)
            val = st.number_input(
                f"Day {i + 1}",
                min_value=0,
                value=int(default_values[i]) if f"day_{i}" not in st.session_state else st.session_state[f"day_{i}"],
                step=1,
                key=f"day_{i}",
                label_visibility="visible",
            )
            user_values.append(val)
    st.markdown("</div>", unsafe_allow_html=True)

    predict_clicked = st.button("🔮 Predict next 7 days", use_container_width=True)

    if predict_clicked:
        last_window = np.array(user_values, dtype=float)

        if np.all(last_window == 0):
            st.warning("All input values are 0 — enter some real numbers for a meaningful prediction.")

        predictions = predict_next_days(model, cfg, last_window)
        future_dates = [start_date + timedelta(days=n_in + i) for i in range(n_out)]
        input_dates = [start_date + timedelta(days=i) for i in range(n_in)]

        # ---------------- Results ----------------
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### 📈 Predicted next 7 days")

        metric_cols = st.columns(n_out)
        for i, col in enumerate(metric_cols):
            if i == 0:
                delta = int(predictions[i] - last_window[-1])
            else:
                delta = int(predictions[i] - predictions[i - 1])
            with col:
                st.metric(
                    label=future_dates[i].strftime("%b %d"),
                    value=f"{int(predictions[i]):,}",
                    delta=f"{delta:+,}",
                )
        st.markdown("</div>", unsafe_allow_html=True)

        # ---------------- Chart ----------------
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Your input vs Predicted")

        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")

        ax.plot(input_dates, last_window, marker="o", linewidth=2.5,
                label="Your input (last 7 days)", color="#4fc3f7")
        ax.plot(future_dates, predictions, marker="o", linestyle="--", linewidth=2.5,
                label="Predicted (next 7 days)", color="#ff8a65")

        # connect the two lines visually
        ax.plot([input_dates[-1], future_dates[0]], [last_window[-1], predictions[0]],
                linestyle="--", linewidth=2.5, color="#ff8a65")

        ax.set_title(f"{feature_col}: Input vs Forecast", fontsize=13, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel(feature_col)
        ax.legend(frameon=False)
        ax.grid(alpha=0.15)
        fig.autofmt_xdate()

        st.pyplot(fig, transparent=True)

        with st.expander("See raw predicted values as a table"):
            pred_df = pd.DataFrame({
                "Date": [d.strftime("%Y-%m-%d") for d in future_dates],
                f"Predicted {feature_col}": predictions.round(0).astype(int),
            })
            st.dataframe(pred_df, use_container_width=True, hide_index=True)

        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()