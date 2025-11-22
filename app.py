import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go

# --- PAGE CONFIG ---
st.set_page_config(page_title="Basha AI V4 Sniper", page_icon="🎯", layout="wide")

# --- SIDEBAR ---
st.sidebar.title("🎯 BASHA AI V4.0")
st.sidebar.info("Status: Sniper Mode Activated")
mode = st.sidebar.radio("Trading Mode", ["☪️ Halal Swing (Safe)", "⚡ Index Scalper (Risky)"])
capital = st.sidebar.number_input("Capital (₹)", 5000, step=1000)

# --- HELPER FUNCTIONS ---
def get_data(ticker, period="1mo", interval="1d"):
    data = yf.Ticker(ticker).history(period=period, interval=interval)
    if len(data) > 0:
        data.ta.rsi(length=14, append=True)
        data.ta.ema(length=50, append=True)
        return data
    return None

def plot_chart(df, ticker, buy_signal=False):
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                open=df['Open'], high=df['High'],
                low=df['Low'], close=df['Close'], name=ticker)])
    
    # Add EMA Line
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_50'], line=dict(color='orange', width=1), name="EMA 50"))
    
    title_color = "green" if buy_signal else "white"
    fig.update_layout(title=f"{ticker} Price Action", template="plotly_dark", height=500,
                     title_font_color=title_color)
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# ☪️ MODE 1: HALAL SWING SNIPER
# ==========================================
if mode == "☪️ Halal Swing (Safe)":
    st.title("☪️ Halal Sniper Dashboard")
    st.caption("Clear Action | Stop Loss | Targets | Live Charts")
    
    halal_stocks = ["TATASTEEL.NS", "ASHOKLEY.NS", "WIPRO.NS", "INFY.NS", "HCLTECH.NS", "TITAN.NS", "SUNPHARMA.NS"]
    
    if st.button("🎯 SCAN MARKET"):
        st.write("Analyzing Market Structure...")
        
        signals = []
        for stock in halal_stocks:
            try:
                df = get_data(stock, period="6mo", interval="1d")
                if df is not None:
                    current_price = df['Close'].iloc[-1]
                    rsi = df['RSI_14'].iloc[-1]
                    ema = df['EMA_50'].iloc[-1]
                    
                    # STRATEGY: RSI Oversold + Above EMA or Reversal
                    action = "⚪ WAIT"
                    color = "white"
                    
                    if rsi < 40:
                        action = "🟢 BUY NOW"
                    elif rsi > 70:
                        action = "🔴 SELL/BOOK"
                    
                    # Calculate SL and Target
                    sl = int(current_price * 0.95) # 5% Loss limit
                    tgt = int(current_price * 1.10) # 10% Profit Target
                    
                    signals.append({
                        "Stock": stock,
                        "Price": f"₹{round(current_price,1)}",
                        "RSI": round(rsi,1),
                        "ACTION": action,
                        "🛑 Stop Loss": f"₹{sl}",
                        "🎯 Target": f"₹{tgt}"
                    })
            except:
                continue
        
        # Show Table
        df_res = pd.DataFrame(signals)
        st.dataframe(df_res, use_container_width=True)
        
        # --- CHART SECTION ---
        st.write("---")
        st.subheader("📊 Live Chart Analysis")
        selected = st.selectbox("Select Stock to View Chart", halal_stocks)
        
        df_chart = get_data(selected, period="6mo", interval="1d")
        if df_chart is not None:
            # Show Key Levels
            curr = df_chart['Close'].iloc[-1]
            c1, c2, c3 = st.columns(3)
            c1.metric("Current Price", f"₹{round(curr,2)}")
            c2.metric("Stop Loss (Support)", f"₹{int(curr*0.95)}")
            c3.metric("Target (Resistance)", f"₹{int(curr*1.10)}")
            
            plot_chart(df_chart, selected)

# ==========================================
# ⚡ MODE 2: SCALPER SNIPER (INDEX)
# ==========================================
elif mode == "⚡ Index Scalper (Risky)":
    st.title("⚡ Index Scalper Matrix")
    st.warning("⚠️ High Risk: Future & Options (Nifty/BankNifty)")
    
    idx = st.selectbox("Select Index", ["^NSEI", "^NSEBANK"], format_func=lambda x: "NIFTY 50" if x == "^NSEI" else "BANK NIFTY")
    
    if st.button("⚡ ANALYZE SIGNAL"):
        df = get_data(idx, period="5d", interval="5m")
        
        if df is not None:
            curr = df['Close'].iloc[-1]
            rsi = df['RSI_14'].iloc[-1]
            ema = df['EMA_50'].iloc[-1]
            
            # MATRIX DECISION
            signal = "SIDEWAYS 😴"
            signal_color = "gray"
            
            if curr > ema and rsi < 50:
                signal = "🚀 CALL (BUY) NOW"
                signal_color = "green"
                sl = curr - 50
                tgt = curr + 100
            elif curr < ema and rsi > 50:
                signal = "📉 PUT (SELL) NOW"
                signal_color = "red"
                sl = curr + 50
                tgt = curr - 100
            else:
                sl = 0
                tgt = 0
            
            # DISPLAY BIG SIGNAL
            st.markdown(f"""
            <div style='text-align: center; background-color: #1e1e1e; padding: 20px; border-radius: 10px; border: 2px solid {signal_color};'>
                <h2 style='color: {signal_color}; margin:0;'>{signal}</h2>
                <h1 style='font-size: 50px; margin:0;'>₹{round(curr, 2)}</h1>
            </div>
            """, unsafe_allow_html=True)
            
            # TARGETS
            if sl > 0:
                c1, c2 = st.columns(2)
                c1.error(f"🛑 STOP LOSS: {int(sl)}")
                c2.success(f"🎯 TARGET: {int(tgt)}")
            
            # CHART
            plot_chart(df, "Index Live")
            
            st.dataframe(df.tail(5))