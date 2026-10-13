"""app.py - Streamlit page for the crypto signal project.
Uses the functions in crypto.py. Paper money only, never places real orders.

Run in terminal:   streamlit run app.py
"""
import io

import numpy as np
import pandas as pd
import requests
import streamlit as st

import crypto

st.set_page_config(page_title="Crypto signal")

# ------------------------------------------------------------------ 1. TITLE + SIDEBAR
st.title("Crypto signal: Decision Tree / Random Forest / XGBoost")
st.caption("Supervised learning on free Binance data. Paper money only, not financial advice.")

st.sidebar.header("Settings")
sec = st.sidebar.slider("Bar seconds", 1, 60, 1)                  # 1 bar = this many seconds of trades
ahead = st.sidebar.slider("Predict ahead (bars)", 1, 60, 10)      # how far in the future we look
share = st.sidebar.slider("BUY / SELL share (%)", 10, 40, 25)     # top share = BUY, bottom share = SELL
cash = st.sidebar.number_input("Paper cash (USDT)", min_value=100.0, value=1000.0, step=100.0)
risk = st.sidebar.slider("Risk per trade (%)", 1, 50, 10)


# ------------------------------------------------------------------ 2. LOAD DATA
@st.cache_data(show_spinner="Reading trades...")
def bars_from_file(raw, sec):
    """uploaded file (bytes) -> trades -> bars -> features"""
    trades = crypto.load_trades(io.BytesIO(raw), nrows=500000)
    return crypto.features(crypto.to_bars(trades, sec)).tail(30000).reset_index(drop=True)


@st.cache_data
def demo_bars(sec):
    """random fake trades so the app works without a file (NOT real prices)"""
    rng = np.random.default_rng(42)
    n = 30000
    t = pd.Timestamp("2026-10-11") + pd.to_timedelta(np.sort(rng.uniform(0, 3600, n)), unit="s")
    price = 60000 * np.exp(np.cumsum(rng.normal(0, 0.00005, n)))
    d = pd.DataFrame({"timestamp": t, "price": price,
                      "quantity": rng.exponential(0.01, n) + 0.0001,
                      "side": rng.choice(["buy", "sell"], n)})
    return crypto.features(crypto.to_bars(d, sec)).reset_index(drop=True)


uploaded = st.file_uploader("Upload a Binance trades CSV", type="csv")
use_demo = st.checkbox("Use random demo data instead (no file needed)")

if use_demo:
    bars = demo_bars(sec)
    st.caption("Demo data: random numbers, not real prices.")
elif uploaded is not None:
    try:
        bars = bars_from_file(uploaded.getvalue(), sec)
    except Exception as e:
        st.error(f"Could not read this file: {e}")
        st.stop()
else:
    st.info("Upload a trades file, or tick the demo box above.")
    st.stop()

st.write(f"Loaded {len(bars):,} bars")
if len(bars) < 200:
    st.warning("Very few bars. Use a smaller 'Bar seconds' value or a bigger file.")
st.line_chart(bars.set_index("timestamp")["price"])

# ------------------------------------------------------------------ 3. TRAIN
# st.session_state keeps the trained models, because Streamlit re-runs this
# whole file every time a button or slider is touched.
if st.button("Train models"):
    models = crypto.get_models()
    rows = []
    try:
        with st.spinner("Training..."):
            for name, model in models.items():
                acc, base = crypto.train(bars, model, ahead=ahead, tail=share / 100)
                rows.append({"Model": name, "Accuracy": round(acc, 3), "Baseline": round(base, 3)})
    except Exception as e:
        st.error(f"Training failed: {e}")
    else:
        st.session_state["models"] = models
        st.session_state["results"] = pd.DataFrame(rows)
        st.session_state["trained_sec"] = sec

if "results" in st.session_state:
    results = st.session_state["results"]
    st.table(results.set_index("Model"))
    st.caption("Accuracy should beat Baseline, otherwise the model learned nothing useful.")
    if (results["Accuracy"] <= results["Baseline"]).all():
        st.warning("No model beat the baseline, so do not trust the signal below.")

# ------------------------------------------------------------------ 4. LIVE SIGNAL
if "models" in st.session_state:
    choice = st.selectbox("Model for live signal", list(st.session_state["models"]))
    if st.session_state.get("trained_sec") != sec:
        st.warning("Bar seconds changed after training. Train again before asking for a signal.")

    if st.button("Get live signal"):
        try:
            signal, conf, price = crypto.live_signal(st.session_state["models"][choice], sec)
        except requests.exceptions.RequestException:
            st.error("Could not reach Binance. Check your internet connection and try again.")
        except Exception:
            st.error("Not enough recent trades to build the features. Try a smaller 'Bar seconds' value.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Signal", signal)
            c2.metric("Confidence", f"{conf:.0%}")
            c3.metric("Price", f"{price:,.2f}")
            st.info(crypto.paper_plan(signal, price, cash=cash, risk_pct=risk))
