import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from datetime import datetime

# --- PAGE CONFIG ---
st.set_page_config(page_title="Basha AI V7 Hybrid", page_icon="🦅", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .stMetric { background-color: #1e1e1e; border-radius: 10px; padding: 10px; border: 1px solid #333; }
    .big-font { font-size:20px !important; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
st.sidebar.title("🦅 BASHA AI V7.0")
st.sidebar.info("Hybrid: V5 Scanner + V6 AI Prediction")
mode = st.sidebar.radio("Select Mode", ["☪️ Halal Sniper (Scanner)", "🤖 AI Stock Predictor", "⚡ Index Scalper"])
capital = st.sidebar.number_input("Capital (₹)", 10000, step=1000)

# --- HELPER FUNCTIONS ---
def get_data(ticker, period="1y", interval="1d"):
    data = yf.Ticker(ticker).history(period=period, interval=interval)
    if len(data) > 0:
        # Indicators
        data.ta.rsi(length=14, append=True)
        data.ta.ema(length=50, append=True)
        data.ta.ema(length=200, append=True)
        # Volume Whale Detector
        data['Vol_SMA'] = data['Volume'].rolling(window=20).mean()
        return data
    return None

def check_whale(row):
    """V5 Feature: Detect Big Volume"""
    if row['Volume'] > (1.5 * row['Vol_SMA']):
        return "🐋 WHALE DETECTED"
    return "Normal"

def ai_predict_price(df):
    """V6 Feature: Machine Learning Prediction"""
    df['Numbers'] = list(range(0, len(df)))
    X = np.array(df['Numbers']).reshape(-1, 1)
    y = df['Close'].values
    model = LinearRegression()
    model.fit(X, y)
    next_day_index = np.array([[len(df) + 1]])
    predicted_price = model.predict(next_day_index)[0]
    return predicted_price, model.score(X, y) * 100

def plot_chart(df, ticker, whale_signal):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                low=df['Low'], close=df['Close'], name="Price"))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_50'], line=dict(color='orange', width=1), name="EMA 50"))
    fig.update_layout(title=f"{ticker} | {whale_signal}", template="plotly_dark", height=500)
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# ☪️ MODE 1: HALAL SNIPER (V5 FEATURES)
# ==========================================
if mode == "☪️ Halal Sniper (Scanner)":
    st.title("☪️ Halal Sniper Dashboard (V5)")
    st.caption("Scanner | Whale Detector | Score System | Signals")
    
    halal_stocks = ["TATASTEEL.NS", "ASHOKLEY.NS", "WIPRO.NS", "INFY.NS", "HCLTECH.NS", 
                    "TITAN.NS", "SUNPHARMA.NS", "ULTRACEMCO.NS", "MARUTI.NS", "RELIANCE.NS"]
    
    if st.button("🚀 SCAN MARKET"):
        results = []
        bar = st.progress(0)
        for i, stock in enumerate(halal_stocks):
            bar.progress((i+1)/len(halal_stocks))
            try:
                df = get_data(stock)
                if df is not None:
                    curr = df.iloc[-1]
                    rsi = curr['RSI_14']
                    whale = check_whale(curr)
                    
                    # V5 SCORING LOGIC
                    score = 0
                    if rsi < 45: score += 20
                    if curr['Close'] > curr['EMA_200']: score += 30
                    if "WHALE" in whale: score += 30
                    if curr['Close'] > curr['EMA_50']: score += 20
                    
                    action = "⚪ WAIT"
                    if score >= 80: action = "🔥 STRONG BUY"
                    elif score >= 50: action = "✅ BUY WATCH"
                    elif score < 30: action = "🔴 AVOID"
                    
                    sl = int(curr['Close'] * 0.95)
                    tgt = int(curr['Close'] * 1.10)
                    
                    results.append({
                        "Stock": stock,
                        "Price": f"₹{round(curr['Close'],1)}",
                        "Score": f"{score}/100",
                        "ACTION": action,
                        "Volume": whale,
                        "SL": sl,
                        "Target": tgt
                    })
            except:
                continue
        bar.empty()
        
        # SHOW V5 TABLE
        df_res = pd.DataFrame(results)
        st.dataframe(df_res.style.map(lambda x: 'color: #00FF00; font-weight: bold' if 'BUY' in str(x) else 'color: white'))
        st.info("💡 Tip: Only trade stocks with Score > 50 and Green Action.")

# ==========================================
# 🤖 MODE 2: AI PREDICTOR (V6 FEATURES)
# ==========================================
elif mode == "🤖 AI Stock Predictor":
    st.title("🤖 AI Future Price Predictor (V6)")
    st.caption("Select a stock to see Machine Learning Prediction")
    
    stocks = ["TATASTEEL.NS", "RELIANCE.NS", "INFY.NS", "HDFCBANK.NS", "MARUTI.NS"]
    selected = st.selectbox("Select Stock", stocks)
    
    if st.button("🔮 PREDICT NEXT TREND"):
        df = get_data(selected)
        if df is not None:
            curr = df['Close'].iloc[-1]
            pred, conf = ai_predict_price(df)
            change = ((pred - curr) / curr) * 100
            direction = "UP 🚀" if change > 0 else "DOWN 📉"
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Current Price", f"₹{round(curr, 2)}")
            c2.metric("AI Predicted Price", f"₹{round(pred, 2)}")
            c3.metric("Expected Move", f"{round(change, 2)}%", direction)
            
            st.write("---")
            if conf > 70: st.success(f"✅ AI Confidence: {round(conf,2)}% (High)")
            else: st.warning(f"⚠️ AI Confidence: {round(conf,2)}% (Low - Volatile)")
            
            # Chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Price"))
            # Trend Line
            z = np.polyfit(range(len(df)), df['Close'], 1)
            p = np.poly1d(z)
            fig.add_trace(go.Scatter(x=df.index, y=p(range(len(df))), name="AI Trend", line=dict(color='orange', dash='dash')))
            st.plotly_chart(fig, use_container_width=True)

# ==========================================
# ⚡ MODE 3: SCALPER
# ==========================================
elif mode == "⚡ Index Scalper":
    st.title("⚡ Nifty/BankNifty Scalper")
    idx = st.selectbox("Select Index", ["^NSEI", "^NSEBANK"])
    
    if st.button("⚡ ANALYZE"):
        df = get_data(idx, period="5d", interval="5m")
        if df is not None:
            curr = df.iloc[-1]
            rsi = curr['RSI_14']
            whale = check_whale(curr)
            
            signal = "SIDEWAYS 😴"
            if curr['Close'] > curr['EMA_50'] and rsi < 60: signal = "🚀 CALL (BUY)"
            elif curr['Close'] < curr['EMA_50'] and rsi > 40: signal = "📉 PUT (SELL)"
            
            c1, c2 = st.columns(2)
            c1.metric("Price", f"₹{round(curr['Close'],2)}")
            c2.metric("Signal", signal)
            
            plot_chart(df, idx, whale)