import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from datetime import datetime

# --- PAGE CONFIG ---
st.set_page_config(page_title="Basha AI Ultimate", page_icon="🤖", layout="wide")

# --- SIDEBAR & NAVIGATION ---
st.sidebar.title("🤖 BASHA AI 3.0")
mode = st.sidebar.radio("Select Trading Mode", ["☪️ Halal Swing (Safe)", "⚡ Index Scalper (Risky)"])

st.sidebar.write("---")
user_name = st.sidebar.text_input("Trader Name", "Zahid Basha")
capital = st.sidebar.number_input("Capital (₹)", 5000, step=1000)

# --- HELPER FUNCTIONS ---

def get_data(ticker, period="5d", interval="15m"):
    """Fetch Data & Calculate Indicators"""
    data = yf.Ticker(ticker).history(period=period, interval=interval)
    if len(data) > 0:
        # RSI
        data.ta.rsi(length=14, append=True)
        # EMA (Trend)
        data.ta.ema(length=50, append=True)
        data.ta.ema(length=200, append=True)
        return data
    return None

def check_patterns(df):
    """Simple Candle Pattern Recognition"""
    last_row = df.iloc[-1]
    prev_row = df.iloc[-2]
    
    pattern = "None"
    
    # Bullish Engulfing (Green eats Red)
    if (prev_row['Close'] < prev_row['Open']) and \
       (last_row['Close'] > last_row['Open']) and \
       (last_row['Close'] > prev_row['Open']) and \
       (last_row['Open'] < prev_row['Close']):
        pattern = "Bullish Engulfing 🐂"

    # Bearish Engulfing (Red eats Green)
    elif (prev_row['Close'] > prev_row['Open']) and \
         (last_row['Close'] < last_row['Open']) and \
         (last_row['Open'] > prev_row['Close']) and \
         (last_row['Close'] < prev_row['Open']):
        pattern = "Bearish Engulfing 🐻"
        
    # Hammer (Long wick at bottom)
    body = abs(last_row['Close'] - last_row['Open'])
    lower_wick = min(last_row['Close'], last_row['Open']) - last_row['Low']
    if lower_wick > (2 * body):
        pattern = "Hammer 🔨"

    return pattern

def plot_chart(df, ticker):
    """Professional Candle Chart"""
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                open=df['Open'], high=df['High'],
                low=df['Low'], close=df['Close'], name=ticker)])
    fig.update_layout(title=f"{ticker} Live Chart", template="plotly_dark", height=500)
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# ☪️ MODE 1: HALAL SWING TRADING
# ==========================================
if mode == "☪️ Halal Swing (Safe)":
    st.title(f"☪️ {user_name}'s Halal Dashboard")
    st.info("Strategy: Buy & Hold (Swing) | No Interest | Ethical Stocks")
    
    halal_stocks = ["TATASTEEL.NS", "ASHOKLEY.NS", "WIPRO.NS", "INFY.NS", "HCLTECH.NS", "TITAN.NS"]
    
    if st.button("🔍 SCAN HALAL STOCKS"):
        st.write("### Scanning for High Probability Setups...")
        
        report = []
        for stock in halal_stocks:
            df = get_data(stock, period="6mo", interval="1d") # Daily timeframe for Swing
            if df is not None:
                rsi = df['RSI_14'].iloc[-1]
                price = df['Close'].iloc[-1]
                pattern = check_patterns(df)
                
                # MATRIX LOGIC for Halal
                score = 0
                if rsi < 40: score += 1 # Cheap
                if pattern != "None" and "Bullish" in pattern: score += 1 # Good Pattern
                
                decision = "WAIT"
                if score >= 2: decision = "🔥 STRONG BUY"
                elif score == 1: decision = "✅ BUY WATCH"
                
                report.append([stock, round(price,2), round(rsi,1), pattern, decision])
        
        results = pd.DataFrame(report, columns=["Stock", "Price", "RSI", "Pattern", "Signal"])
        st.dataframe(results.style.map(lambda x: 'color: green' if 'BUY' in str(x) else 'color: white'))

# ==========================================
# ⚡ MODE 2: NIFTY/BANKNIFTY SCALPER
# ==========================================
elif mode == "⚡ Index Scalper (Risky)":
    st.title(f"⚡ {user_name}'s Scalping Matrix")
    st.warning("⚠️ High Risk Mode: Future & Options Analysis (Check Halal status yourself)")
    
    indices = {"NIFTY 50": "^NSEI", "BANK NIFTY": "^NSEBANK"}
    selected_index = st.selectbox("Select Index", list(indices.keys()))
    ticker = indices[selected_index]
    
    # Auto Refresh Logic
    if st.button("⚡ ANALYZE MARKET MATRIX"):
        df = get_data(ticker, period="5d", interval="5m") # 5 Minute Candles for Scalping
        
        if df is not None:
            # 1. LIVE METRICS
            current_price = df['Close'].iloc[-1]
            rsi = df['RSI_14'].iloc[-1]
            ema_50 = df['EMA_50'].iloc[-1]
            pattern = check_patterns(df)
            
            # 2. MATRIX CALCULATION (90% Accuracy Logic attempt)
            trend = "UP 🟢" if current_price > ema_50 else "DOWN 🔴"
            rsi_signal = "OVERSOLD (BUY) 🟢" if rsi < 30 else ("OVERBOUGHT (SELL) 🔴" if rsi > 70 else "NEUTRAL ⚪")
            
            # Display Top Metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Current Price", f"₹{round(current_price, 2)}")
            col2.metric("RSI Strength", f"{round(rsi, 2)}", rsi_signal)
            col3.metric("Trend (EMA)", trend)
            col4.metric("Candle Pattern", pattern)
            
            # 3. CHART
            plot_chart(df, ticker)
            
            # 4. FINAL MATRIX DECISION
            st.subheader("🔮 AI Probability Matrix")
            
            buy_score = 0
            if current_price > ema_50: buy_score += 1
            if rsi < 40: buy_score += 1
            if "Bullish" in pattern or "Hammer" in pattern: buy_score += 2 # Strong Pattern gets more points
            
            sell_score = 0
            if current_price < ema_50: sell_score += 1
            if rsi > 70: sell_score += 1
            if "Bearish" in pattern: sell_score += 2
            
            if buy_score >= 3:
                st.success(f"🚀 **STRONG BUY CALL** | Confidence: High | StopLoss: {int(current_price * 0.995)}")
            elif sell_score >= 3:
                st.error(f"📉 **STRONG SELL (PUT) CALL** | Confidence: High | StopLoss: {int(current_price * 1.005)}")
            else:
                st.info("⏳ MARKET IS SIDEWAYS / NO CLEAR SIGNAL (Wait)")
                
            # Data Table
            st.caption("Recent Data (Last 5 Candles)")
            st.dataframe(df.tail(5)[['Open', 'High', 'Low', 'Close', 'RSI_14']])