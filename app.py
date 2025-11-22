import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from datetime import datetime

# --- PAGE CONFIG ---
st.set_page_config(page_title="Basha AI V5 Ultimate", page_icon="🚀", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .stMetric { background-color: #262730; border-radius: 10px; padding: 10px; }
    div[data-testid="stMarkdownContainer"] > h1 { color: #00FF00; }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
st.sidebar.title("🚀 BASHA AI V5.0")
st.sidebar.caption("Whale Detector | Fundamentals | Multi-Timeframe")
mode = st.sidebar.radio("Trading Mode", ["☪️ Halal Swing (Safe)", "⚡ Index Scalper (Risky)"])
capital = st.sidebar.number_input("Capital (₹)", 10000, step=1000)

# --- HELPER FUNCTIONS ---
def get_data(ticker, period="1y", interval="1d"):
    data = yf.Ticker(ticker).history(period=period, interval=interval)
    if len(data) > 0:
        # Indicators
        data.ta.rsi(length=14, append=True)
        data.ta.ema(length=50, append=True)
        data.ta.ema(length=200, append=True) # Long term trend
        # Volume SMA for Whale Detection
        data['Vol_SMA'] = data['Volume'].rolling(window=20).mean()
        return data
    return None

def check_whale(row):
    """Check if Volume is 1.5x higher than average"""
    if row['Volume'] > (1.5 * row['Vol_SMA']):
        return "🐋 WHALE DETECTED"
    return "Normal"

def get_fundamentals(ticker):
    """Fetch basic company info"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "PE Ratio": info.get('forwardPE', 'N/A'),
            "Market Cap": info.get('marketCap', 'N/A'),
            "Debt To Equity": info.get('debtToEquity', 'N/A'),
            "Sector": info.get('sector', 'N/A')
        }
    except:
        return None

def plot_chart_v5(df, ticker, whale_signal):
    fig = go.Figure()
    
    # Candlestick
    fig.add_trace(go.Candlestick(x=df.index,
                open=df['Open'], high=df['High'],
                low=df['Low'], close=df['Close'], name="Price"))
    
    # EMA Lines
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_50'], line=dict(color='orange', width=1), name="EMA 50"))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_200'], line=dict(color='blue', width=1), name="EMA 200 (Trend)"))
    
    # Update Layout
    title_text = f"{ticker} | {whale_signal}"
    fig.update_layout(title=title_text, template="plotly_dark", height=600, xaxis_rangeslider_visible=False)
    
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# ☪️ MODE 1: HALAL ULTIMATE SWING
# ==========================================
if mode == "☪️ Halal Swing (Safe)":
    st.title("☪️ Halal Ultimate Dashboard")
    
    # Main Watchlist
    halal_stocks = ["TATASTEEL.NS", "ASHOKLEY.NS", "WIPRO.NS", "INFY.NS", "HCLTECH.NS", 
                    "TITAN.NS", "SUNPHARMA.NS", "ULTRACEMCO.NS", "POWERGRID.NS", "MARUTI.NS"]
    
    tab1, tab2 = st.tabs(["🚀 AI SCANNER", "🔬 DEEP ANALYSIS"])
    
    with tab1:
        if st.button("🚀 RUN V5 SCANNER"):
            st.write("Scanning Price, Volume, and Trends...")
            results = []
            
            scan_bar = st.progress(0)
            for i, stock in enumerate(halal_stocks):
                scan_bar.progress((i+1)/len(halal_stocks))
                try:
                    df = get_data(stock)
                    if df is not None:
                        curr = df.iloc[-1]
                        rsi = curr['RSI_14']
                        ema50 = curr['EMA_50']
                        ema200 = curr['EMA_200']
                        whale = check_whale(curr)
                        
                        # SCORING SYSTEM (The Brain)
                        score = 0
                        if rsi < 45: score += 20        # Cheap
                        if curr['Close'] > ema200: score += 30 # Long term Up Trend
                        if "WHALE" in whale: score += 30 # Big Money Flow
                        if curr['Close'] > ema50: score += 20  # Short term momentum
                        
                        action = "⚪ WAIT"
                        if score >= 80: action = "🔥 STRONG BUY"
                        elif score >= 50: action = "✅ BUY WATCH"
                        elif score < 30: action = "🔴 AVOID"
                        
                        # Targets
                        sl = int(curr['Close'] * 0.94)
                        tgt = int(curr['Close'] * 1.12)
                        
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
            
            scan_bar.empty()
            df_res = pd.DataFrame(results)
            
            # Color Formatting
            st.dataframe(df_res.style.map(lambda x: 'color: #00FF00; font-weight: bold' if 'BUY' in str(x) else 'color: white'))
            
    with tab2:
        st.header("🔬 Fundamental & Technical Deep Dive")
        selected = st.selectbox("Select Stock for Checkup", halal_stocks)
        
        if st.button("🔍 ANALYZE STOCK"):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("🏢 Fundamentals")
                info = get_fundamentals(selected)
                if info:
                    st.metric("Sector", info['Sector'])
                    st.metric("Market Cap", f"₹ {info['Market Cap']}")
                    st.metric("P/E Ratio", info['PE Ratio'])
                    st.metric("Debt/Equity", info['Debt To Equity'], delta_color="inverse")
                    if info['Debt To Equity'] != 'N/A' and float(info['Debt To Equity']) > 1.0:
                        st.error("⚠️ High Debt Warning!")
                    else:
                        st.success("✅ Debt Looks Safe")
            
            with col2:
                st.subheader("📊 Technical Chart")
                df_chart = get_data(selected)
                if df_chart is not None:
                    last_row = df_chart.iloc[-1]
                    whale_status = check_whale(last_row)
                    plot_chart_v5(df_chart, selected, whale_status)

# ==========================================
# ⚡ MODE 2: SCALPER ULTIMATE
# ==========================================
elif mode == "⚡ Index Scalper (Risky)":
    st.title("⚡ Index Scalper V5 (Whale Tracker)")
    
    idx = st.selectbox("Select Index", ["^NSEI", "^NSEBANK"], format_func=lambda x: "NIFTY 50" if x == "^NSEI" else "BANK NIFTY")
    
    if st.button("⚡ SCAN INDEX"):
        df = get_data(idx, period="5d", interval="5m") # 5 Minute Data
        
        if df is not None:
            curr = df.iloc[-1]
            rsi = curr['RSI_14']
            whale = check_whale(curr)
            
            # V5 SCALPING MATRIX
            signal = "SIDEWAYS 😴"
            confidence = "Low"
            
            # Bullish Case
            if curr['Close'] > curr['EMA_50'] and rsi < 60:
                signal = "🚀 CALL (BUY)"
                confidence = "High" if "WHALE" in whale else "Medium"
                sl = int(curr['Close'] - 40)
                tgt = int(curr['Close'] + 80)
                
            # Bearish Case
            elif curr['Close'] < curr['EMA_50'] and rsi > 40:
                signal = "📉 PUT (SELL)"
                confidence = "High" if "WHALE" in whale else "Medium"
                sl = int(curr['Close'] + 40)
                tgt = int(curr['Close'] - 80)
            
            else:
                sl, tgt = 0, 0

            # DISPLAY DASHBOARD
            c1, c2, c3 = st.columns(3)
            c1.metric("Current Price", f"₹{round(curr['Close'],2)}", f"{whale}")
            c2.metric("Signal", signal, f"Conf: {confidence}")
            c3.metric("RSI Strength", f"{round(rsi, 2)}")
            
            if sl > 0:
                st.success(f"🎯 TARGET: {tgt} | 🛑 STOP LOSS: {sl}")
            
            plot_chart_v5(df, idx, whale)
            st.caption("Showing last 5 days of 5-minute candles")