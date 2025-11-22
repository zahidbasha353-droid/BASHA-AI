import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

# --- PAGE CONFIG ---
st.set_page_config(page_title="Basha AI V6 God Mode", page_icon="💎", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .stMetric { background-color: #121212; border: 1px solid #333; border-radius: 10px; padding: 15px; }
    h1, h2, h3 { color: #00e676; font-family: 'Courier New', monospace; }
    .big-font { font-size:20px !important; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
st.sidebar.title("💎 BASHA AI V6.0")
st.sidebar.info("Feature: AI Prediction | PCR Analysis | Whale Tracker")
mode = st.sidebar.radio("Select Mode", ["🤖 AI Stock Predictor", "⚡ F&O Option Chain Analyst"])

# --- HELPER FUNCTIONS ---
def get_data(ticker, period="2y", interval="1d"): # Fetches 2 years data for AI training
    data = yf.Ticker(ticker).history(period=period, interval=interval)
    if len(data) > 0:
        data.ta.rsi(length=14, append=True)
        data.ta.ema(length=50, append=True)
        return data
    return None

def ai_predict_price(df):
    """Simple Machine Learning Model to Predict Next Day Price"""
    df['Numbers'] = list(range(0, len(df)))
    X = np.array(df['Numbers']).reshape(-1, 1)
    y = df['Close'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    next_day_index = np.array([[len(df) + 1]])
    predicted_price = model.predict(next_day_index)[0]
    return predicted_price, model.score(X, y) * 100  # Confidence Score

def get_pcr(ticker):
    """Get Put-Call Ratio (Approximate for Index)"""
    # Note: Real-time PCR needs paid API, here we estimate using Volume/Price action
    # This acts as a 'Sentiment Indicator'
    try:
        stock = yf.Ticker(ticker)
        # Logic: If Volume is high and Price is up -> Bullish Sentiment
        hist = stock.history(period="5d")
        change = (hist['Close'].iloc[-1] - hist['Open'].iloc[-1])
        
        if change > 0:
            return "BULLISH (Call Writers Trapped) 🟢"
        else:
            return "BEARISH (Put Writers Trapped) 🔴"
    except:
        return "NEUTRAL ⚪"

# ==========================================
# 🤖 MODE 1: AI STOCK PREDICTOR
# ==========================================
if mode == "🤖 AI Stock Predictor":
    st.title("🤖 AI Future Price Predictor")
    st.caption("Uses Machine Learning (Linear Regression) to project trends.")
    
    stocks = ["TATASTEEL.NS", "RELIANCE.NS", "INFY.NS", "HDFCBANK.NS", "MARUTI.NS", "ADANIENT.NS", "TCS.NS"]
    selected_stock = st.selectbox("Select Stock to Predict", stocks)
    
    if st.button("🔮 ACTIVATE AI PREDICTION"):
        with st.spinner("Training AI Model... Please wait..."):
            df = get_data(selected_stock)
            
            if df is not None:
                curr_price = df['Close'].iloc[-1]
                pred_price, confidence = ai_predict_price(df)
                
                # Calculate Potential
                change = ((pred_price - curr_price) / curr_price) * 100
                direction = "UP 🚀" if change > 0 else "DOWN 📉"
                color = "green" if change > 0 else "red"
                
                # --- DISPLAY RESULTS ---
                c1, c2, c3 = st.columns(3)
                c1.metric("Current Price", f"₹{round(curr_price, 2)}")
                c2.metric("AI Predicted Price (Next Trend)", f"₹{round(pred_price, 2)}")
                c3.metric("Expected Move", f"{round(change, 2)}%", direction)
                
                st.write("---")
                st.subheader(f"🧠 AI Confidence: {round(confidence, 2)}%")
                if confidence > 70:
                    st.success("✅ The AI is highly confident in this trend.")
                else:
                    st.warning("⚠️ Market is volatile. AI confidence is low.")
                
                # --- CHART ---
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Actual Price"))
                
                # Trend Line
                z = np.polyfit(range(len(df)), df['Close'], 1)
                p = np.poly1d(z)
                fig.add_trace(go.Scatter(x=df.index, y=p(range(len(df))), name="AI Trend Line", line=dict(color='orange', dash='dash')))
                
                st.plotly_chart(fig, use_container_width=True)

# ==========================================
# ⚡ MODE 2: F&O OPTION ANALYST
# ==========================================
elif mode == "⚡ F&O Option Chain Analyst":
    st.title("⚡ Option Chain & Sentiment Decoder")
    st.warning("Designed for NIFTY / BANKNIFTY Traders")
    
    idx = st.selectbox("Select Index", ["^NSEI", "^NSEBANK"], format_func=lambda x: "NIFTY 50" if x == "^NSEI" else "BANK NIFTY")
    
    if st.button("⚡ DECODE MARKET DATA"):
        df = get_data(idx, period="1mo", interval="1d")
        
        if df is not None:
            curr = df.iloc[-1]
            rsi = curr['RSI_14']
            sentiment = get_pcr(idx)
            
            # MARKET MOOD METER
            mood = "NEUTRAL"
            if rsi > 60 and "BULLISH" in sentiment: mood = "🔥 GREED (Super Bullish)"
            elif rsi < 40 and "BEARISH" in sentiment: mood = "😨 FEAR (Super Bearish)"
            
            # DISPLAY DASHBOARD
            c1, c2, c3 = st.columns(3)
            c1.metric("Index Price", f"₹{round(curr['Close'], 2)}")
            c2.metric("Market Mood", mood)
            c3.metric("Data Sentiment", sentiment)
            
            # SCALPING LEVELS
            st.subheader("🎯 Key Levels for Tomorrow")
            pivot = (curr['High'] + curr['Low'] + curr['Close']) / 3
            r1 = (2 * pivot) - curr['Low']
            s1 = (2 * pivot) - curr['High']
            
            col1, col2 = st.columns(2)
            col1.success(f"resistance (Take Profit): ₹{int(r1)}")
            col2.error(f"Support (Buy Zone): ₹{int(s1)}")
            
            # VISUAL CHART
            fig = go.Figure(data=[go.Candlestick(x=df.index,
                open=df['Open'], high=df['High'],
                low=df['Low'], close=df['Close'], name="Index")])
            fig.update_layout(title="Market Momentum", template="plotly_dark", height=500)
            st.plotly_chart(fig, use_container_width=True)