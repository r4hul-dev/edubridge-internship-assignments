"""crypto.py - crypto signal engine WITHOUT Streamlit (supervised learning only).
Models: Decision Tree, Random Forest, XGBoost  ->  BUY / HOLD / SELL

Run in terminal:   python crypto.py
Students: build the Streamlit page on top of the functions below (see TODO at the bottom).
"""
import os
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

try:
    from dotenv import load_dotenv
    load_dotenv()                          # reads the .env file
except Exception:
    pass
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

BASE = os.getenv("BINANCE_BASE", "https://data-api.binance.vision")   # free, no key
SYMBOL = os.getenv("SYMBOL", "BTCUSDT")
DATA_FILE = os.getenv("DATA_FILE", "BTCUSDT-trades-2026-10-11.csv")
NAMES = {0: "SELL", 1: "HOLD", 2: "BUY"}
FEATS = ["ret_1", "ret_5", "vol_20", "flow", "flow_5", "log_qty"]


# 1) LOAD: Binance daily trades file (no header) or your own CSV -> timestamp, price, quantity, side
def load_trades(file, nrows=None):
    head = pd.read_csv(file, header=None, nrows=1)
    file.seek(0)
    if str(head.iloc[0, 0]).isdigit():                     # Binance file has no header
        d = pd.read_csv(file, header=None, nrows=nrows,
                        names=["id", "price", "quantity", "qq", "time", "maker", "bm"])
        unit = "us" if d["time"].max() > 1e14 else "ms"
        d["timestamp"] = pd.to_datetime(d["time"], unit=unit)
        d["side"] = np.where(d["maker"].astype(str).str.lower() == "true", "sell", "buy")
    else:                                                  # your own CSV with a header
        d = pd.read_csv(file, nrows=nrows)
        d["timestamp"] = pd.to_datetime(d["timestamp"])
    d = d[["timestamp", "price", "quantity", "side"]].dropna().drop_duplicates()
    return d.sort_values("timestamp")


# 2) BARS: group trades into small time bars (1 row per few seconds)
def to_bars(d, sec=1):
    d = d.copy()
    d["buy"] = np.where(d["side"] == "buy", d["quantity"], 0.0)
    g = d.set_index("timestamp").resample(f"{sec}s")
    b = pd.DataFrame({"price": g["price"].last(), "qty": g["quantity"].sum(),
                      "buy": g["buy"].sum(), "n": g["price"].count()})
    return b[b["n"] > 0].reset_index()


# 3) FEATURES: numbers the model learns from
def features(b):
    b["ret_1"] = b["price"].pct_change()                   # last price change
    b["ret_5"] = b["price"].pct_change(5)                  # change over 5 bars
    b["vol_20"] = b["ret_1"].rolling(20).std()             # volatility
    b["flow"] = 2 * b["buy"] / b["qty"] - 1                # +1 all buying, -1 all selling
    b["flow_5"] = b["flow"].rolling(5).mean()
    b["log_qty"] = np.log1p(b["qty"])                      # trade volume
    return b


# 4) MODELS + TRAIN: label = what the price did N bars later (SELL / HOLD / BUY)
def get_models():
    models = {"Decision Tree": DecisionTreeClassifier(max_depth=5, class_weight="balanced"),
              "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=5,
                                                      class_weight="balanced", n_jobs=-1)}
    if XGBClassifier:
        models["XGBoost"] = XGBClassifier(n_estimators=200, max_depth=5)
    return models


def train(b, model, ahead=10, tail=0.25, test=0.3):
    b = b.copy()
    b["future"] = b["price"].shift(-ahead) / b["price"] - 1
    d = b.dropna(subset=FEATS + ["future"])
    cut = int(len(d) * (1 - test))                         # past = train, newest = test
    tr, te = d.iloc[:cut], d.iloc[cut:]
    lo, hi = tr["future"].quantile([tail, 1 - tail])       # SELL below lo, BUY above hi
    label = lambda x: np.where(x["future"] > hi, 2, np.where(x["future"] < lo, 0, 1))
    model.fit(tr[FEATS], label(tr))
    acc = (model.predict(te[FEATS]) == label(te)).mean()
    base = pd.Series(label(te)).value_counts(normalize=True).max()   # "always guess most common"
    return acc, base


# 5) LIVE: free Binance API -> same features -> predict
def live_signal(model, sec=1):
    r = requests.get(f"{BASE}/api/v3/trades", params={"symbol": SYMBOL, "limit": 1000}, timeout=10)
    r.raise_for_status()
    d = pd.DataFrame(r.json())
    d["timestamp"] = pd.to_datetime(d["time"], unit="ms")
    d["price"], d["quantity"] = d["price"].astype(float), d["qty"].astype(float)
    d["side"] = np.where(d["isBuyerMaker"], "sell", "buy")
    live = features(to_bars(d, sec)).dropna(subset=FEATS)
    last = live.iloc[[-1]]
    p = model.predict_proba(last[FEATS])[0]
    return NAMES[int(model.classes_[p.argmax()])], float(p.max()), float(last["price"].iloc[0])


# 6) PAPER TRADE: fake-money suggestion (never places real orders)
def paper_plan(signal, price, cash=1000.0, risk_pct=10):
    spend = cash * risk_pct / 100
    if signal == "BUY":
        return f"Paper BUY: spend {spend:,.2f} USDT = {spend / price:.6f} coin"
    if signal == "SELL":
        return "Paper SELL: sell part of the coin you hold"
    return "HOLD: do nothing"


# ---------------------------------------------------------------- RUN IN TERMINAL
if __name__ == "__main__":
    with open(DATA_FILE, "rb") as f:
        bars = features(to_bars(load_trades(f, nrows=500000), 1)).tail(30000)
    print(f"{len(bars)} bars loaded from {DATA_FILE}")
    models = get_models()
    for name, m in models.items():
        acc, base = train(bars, m)
        print(f"{name:14s} accuracy={acc:.3f}  baseline={base:.3f}")
    sig, conf, price = live_signal(models["Random Forest"])
    print(f"Live: {sig} (confidence {conf:.0%}) at {price:,.2f}")
    print(paper_plan(sig, price))

# ------------------------------------------------------------------ STUDENT TODO
# Build app.py with Streamlit using the functions above (import crypto):
#   1. st.title(...) and sidebar sliders for: bar seconds, predict ahead, BUY/SELL share
#   2. st.file_uploader -> crypto.load_trades -> to_bars -> features -> st.line_chart(price)
#   3. Button "Train models": loop crypto.get_models(), call crypto.train, show st.table of accuracy
#   4. Button "Live signal": crypto.live_signal -> st.metric, then st.info(crypto.paper_plan(...))
# Run with:  streamlit run app.py
