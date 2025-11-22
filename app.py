import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# --- PAGE CONFIG ---
st.set_page_config(page_title="Basha AI V16.0", page_icon="🦅", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .stMetric { background-color: #1e1e1e; border-radius: 10px; padding: 15px; border: 1px solid #333; }
    .ticker-wrap { width: 100%; overflow: hidden; background-color: #121212; color: #00ff00; padding: 10px; border-bottom: 1px solid #333; margin-bottom: 20px; }
    .ticker-move { display: inline-block; white-space: nowrap; animation: ticker 30s linear infinite; font-family: 'Courier New', monospace; font-weight: bold; }
    @keyframes ticker { 0% { transform: translateX(100%); } 100% { transform: translateX(-100%); } }
</style>
""", unsafe_allow_html=True)

# --- 🌍 HUGE STOCK DATABASE (A-Z) ---
full_market_list = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", 
    "LICI.NS", "HINDUNILVR.NS", "TATASTEEL.NS", "MARUTI.NS", "TITAN.NS", "AXISBANK.NS", "SUNPHARMA.NS",
    "ULTRACEMCO.NS", "BAJFINANCE.NS", "KOTAKBANK.NS", "ASIANPAINT.NS", "HCLTECH.NS", "NTPC.NS", "POWERGRID.NS",
    "ADANIENT.NS", "ONGC.NS", "TATAMOTORS.NS", "COALINDIA.NS", "ADANIPORTS.NS", "JSWSTEEL.NS", "GRASIM.NS",
    "ZOMATO.NS", "PAYTM.NS", "TATAPOWER.NS", "SUZLON.NS", "JIOFIN.NS", "YESBANK.NS", "IDEA.NS", "IRFC.NS", "RVNL.NS",
    "POLYCAB.NS", "HAVELLS.NS", "TRENT.NS", "HAL.NS", "BEL.NS", "BHEL.NS", "IOC.NS", "VEDL.NS", "DLF.NS", "VARUN.NS",
    "SIEMENS.NS", "TATAELXSI.NS", "KPITTECH.NS", "LTIM.NS", "TECHM.NS", "WIPRO.NS", "PERSISTENT.NS", "MPHASIS.NS",
    "ASHOKLEY.NS", "EICHERMOT.NS", "TVSMOTOR.NS", "HEROMOTOCO.NS", "BOSCHLTD.NS", "MRF.NS", "BALKRISIND.NS",
    "PNB.NS", "BANKBARODA.NS", "CANBK.NS", "IDFCFIRSTB.NS", "FEDERALBNK.NS", "AUBANK.NS", "MUTHOOTFIN.NS",
    "CHOLAFIN.NS", "SHRIRAMFIN.NS", "M&MFIN.NS", "BAJAJHLDNG.NS", "PFC.NS", "RECLTD.NS", "OFSS.NS", "DMART.NS",
    "PIDILITIND.NS", "BERGEPAINT.NS", "GODREJCP.NS", "DABUR.NS", "MARICO.NS", "COLPAL.NS", "BRITANNIA.NS", "NESTLEIND.NS",
    "APOLLOHOSP.NS", "MAXHEALTH.NS", "LALPATHLAB.NS", "METROPOLIS.NS", "SYNGENE.NS", "BIOCON.NS", "LUPIN.NS",
    "AUROPHARMA.NS", "ALKEM.NS", "TORNTPHARM.NS", "ZYDUSLIFE.NS", "DRREDDY.NS", "DIVISLAB.NS"
]
full_market_list.sort()

# Halal List
halal_list = ["TATASTEEL.NS", "ASHOKLEY.NS", "WIPRO.NS", "INFY.NS", "HCLTECH.NS", "TITAN.NS", "MARUTI.NS", "RELIANCE.NS", "SUNPHARMA.NS", "ASIANPAINT.NS", "HINDUNILVR.NS", "CIPLA.NS", "ULTRACEMCO.NS", "HEROMOTOCO.NS", "TECHM.NS"]
halal_list.sort()

# --- SESSION STATE ---
if 'balance' not in st.session_state: st.session_state.balance = 1000000
if 'portfolio' not in st.session_state: st.session_state.portfolio = []
if 'pnl' not in st.session_state: st.session_state.pnl = 0

# --- SIDEBAR ---
st.sidebar.title("🦅 BASHA AI V16.0")
st.sidebar.info("Features: Split Lists (Halal/All) in AI Predictor")

capital = st.sidebar.number_input("Total Trading Capital (₹)", value=10000, step=5000)

mode = st.sidebar.radio("Select Tool", [
    "📊 Target Dashboard (Home)", 
    "🌍 Universal Scanner (Halal/All)", 
    "🎮 Paper Trading (Practice)", 
    "🤖 AI Predictor (Split Lists)", 
    "⚡ Index Scalper"
])

# --- HELPER FUNCTIONS ---
def get_data(ticker, period="1y", interval="1d"):
    try:
        data = yf.Ticker(ticker).history(period=period, interval=interval)
        if len(data) > 0:
            data.ta.rsi(length=14, append=True)
            data.ta.ema(length=50, append=True)
            data.ta.ema(length=200, append=True)
            data['Vol_SMA'] = data['Volume'].rolling(window=20).mean()
            return data
    except: return None
    return None

def check_whale(row):
    if row['Volume'] > (1.5 * row['Vol_SMA']): return "🐋 WHALE"
    return "Normal"

def get_category(stock_name):
    if stock_name in halal_list: return "☪️ HALAL"
    return "🚫 NORMAL / UNCHECKED"

def ai_predict(df):
    df['Numbers'] = list(range(0, len(df)))
    X = np.array(df['Numbers']).reshape(-1, 1)
    y = df['Close'].values
    model = LinearRegression().fit(X, y)
    return model.predict(np.array([[len(df) + 1]]))[0], model.score(X, y) * 100

# --- LIVE TICKER ---
try:
    nifty = yf.Ticker("^NSEI").history(period="1d")['Close'].iloc[-1]
    ticker_text = f"🚀 NIFTY 50: ₹{round(nifty,2)} &nbsp;&nbsp; | &nbsp;&nbsp; 🦅 BASHA AI V16 LIVE &nbsp;&nbsp; | &nbsp;&nbsp; 🤖 DUAL LIST SUPPORT ADDED"
except:
    ticker_text = "🚀 MARKET DATA LOADING... | 🦅 BASHA AI V16 LIVE"

st.markdown(f"""<div class="ticker-wrap"><div class="ticker-move">{ticker_text}</div></div>""", unsafe_allow_html=True)

# ==========================================
# 📊 MODE 0: TARGET DASHBOARD
# ==========================================
if mode == "📊 Target Dashboard (Home)":
    st.title(f"💰 Financial Freedom Plan")
    st.metric("🌞 Daily Target (1%)", f"₹{int(capital * 0.01)}")
    st.info("👈 Use 'AI Predictor' to see the new Dual List feature.")

# ==========================================
# 🌍 MODE 1: UNIVERSAL SCANNER
# ==========================================
elif mode == "🌍 Universal Scanner (Halal/All)":
    st.title("🌍 Market Scanner")
    filter_opt = st.radio("Select List:", ["Only ☪️ Halal Stocks", "🔥 Top 50 Popular Stocks (Nifty 50 Mix)"], horizontal=True)
    
    if filter_opt == "Only ☪️ Halal Stocks": scan_list = halal_list
    else: scan_list = full_market_list[:50] 
    
    if st.button("🚀 SCAN NOW"):
        res = []
        bar = st.progress(0)
        for i, s in enumerate(scan_list):
            bar.progress((i+1)/len(scan_list))
            try:
                df = get_data(s)
                if df is not None:
                    curr = df.iloc[-1]
                    rsi = curr['RSI_14']
                    whale = check_whale(curr)
                    cat = get_category(s)
                    score = 0
                    if rsi < 45: score += 20
                    if curr['Close'] > curr['EMA_200']: score += 30
                    if "WHALE" in whale: score += 30
                    if curr['Close'] > curr['EMA_50']: score += 20
                    
                    action = "⚪ WAIT"
                    if score >= 80: action = "🔥 STRONG BUY"
                    elif score >= 50: action = "✅ BUY WATCH"
                    
                    res.append({"Stock": s, "Category": cat, "Price": round(curr['Close'],1), "Score": score, "ACTION": action, "Volume": whale})
            except: continue
        bar.empty()
        df_final = pd.DataFrame(res)
        if not df_final.empty:
            st.dataframe(df_final.style.map(lambda x: 'color: #00FF00' if 'BUY' in str(x) else ('color: white')))

# ==========================================
# 🎮 MODE 2: PAPER TRADING
# ==========================================
elif mode == "🎮 Paper Trading (Practice)":
    st.title("🎮 Virtual Trading Simulator")
    c1, c2 = st.columns(2)
    c1.metric("💰 Virtual Balance", f"₹{int(st.session_state.balance)}")
    c2.metric("📈 Profit/Loss", f"₹{st.session_state.pnl}")

    col1, col2, col3, col4 = st.columns(4)
    # Search box for Paper Trading
    s_sym = col1.selectbox("Search Stock", full_market_list) 
    act = col2.selectbox("Action", ["BUY", "SELL"])
    qty = col3.number_input("Qty", 1)
    
    if col4.button("⚡ EXECUTE"):
        try:
            cp = yf.Ticker(s_sym).history(period="1d")['Close'].iloc[-1]
            val = cp * qty
            if act == "BUY":
                if val <= st.session_state.balance:
                    st.session_state.balance -= val
                    st.session_state.portfolio.append({"Stock": s_sym, "Type": "BUY", "Qty": qty, "Price": cp})
                    st.success(f"Bought {qty} of {s_sym} @ ₹{round(cp,1)}")
                else: st.error("No Cash!")
            elif act == "SELL":
                st.session_state.balance += val
                st.session_state.pnl += (val * 0.01)
                st.success("Sold! P&L Updated.")
        except: st.error("Market Data Error")

# ==========================================
# 🤖 MODE 3: AI PREDICTOR (UPDATED: DUAL LIST)
# ==========================================
elif mode == "🤖 AI Predictor (Split Lists)":
    st.title("🤖 AI Future Predictor")
    
    # --- THE NEW UPDATE IS HERE ---
    st.write("### 🔍 Step 1: Select Market Segment")
    list_choice = st.radio("", ["☪️ Halal Stocks Only", "🌍 All Market Stocks (A-Z)"], horizontal=True)
    
    if list_choice == "☪️ Halal Stocks Only":
        display_stocks = halal_list
    else:
        display_stocks = full_market_list
        
    st.write("### 🔍 Step 2: Choose Stock")
    sel = st.selectbox("", display_stocks)
    # ------------------------------
    
    if st.button("🔮 PREDICT"):
        df = get_data(sel)
        if df is not None:
            pred, conf = ai_predict(df)
            curr = df['Close'].iloc[-1]
            change = ((pred-curr)/curr)*100
            dir_ = "UP 🚀" if change > 0 else "DOWN 📉"
            
            st.metric("Category", get_category(sel))
            c1, c2 = st.columns(2)
            c1.metric("Current Price", f"₹{round(curr,2)}")
            c2.metric("AI Prediction", f"₹{round(pred,2)}", f"{round(change,2)}% {dir_}")
            
            fig = go.Figure(data=[go.Scatter(x=df.index, y=df['Close'], name="Price")])
            st.plotly_chart(fig)
        else:
            st.error("Could not fetch data for this stock. Try another.")

# ==========================================
# ⚡ MODE 4: SCALPER
# ==========================================
elif mode == "⚡ Index Scalper":
    st.title("⚡ Nifty Scalper")
    idx = st.selectbox("Index", ["^NSEI", "^NSEBANK"])
    
    if st.button("⚡ ANALYZE"):
        df = get_data(idx, period="5d", interval="5m")
        if df is not None:
            curr = df.iloc[-1]
            rsi = curr['RSI_14']
            signal = "SIDEWAYS"
            if curr['Close'] > curr['EMA_50'] and rsi < 60: signal = "🚀 CALL (BUY)"
            elif curr['Close'] < curr['EMA_50'] and rsi > 40: signal = "📉 PUT (SELL)"
            
            c1, c2 = st.columns(2)
            c1.metric("Price", f"₹{round(curr['Close'],2)}")
            c2.metric("Signal", signal)
            
            fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA_50'], line=dict(color='orange'), name="EMA 50"))
            fig.update_layout(template="plotly_dark", height=500)
            st.plotly_chart(fig, use_container_width=True)