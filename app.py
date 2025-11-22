import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# --- PAGE CONFIG ---
st.set_page_config(page_title="Basha AI V13 Hybrid", page_icon="🦅", layout="wide")

# --- CUSTOM CSS (V10 Design + V12 Tags) ---
st.markdown("""
<style>
    .stMetric { background-color: #1e1e1e; border-radius: 10px; padding: 15px; border: 1px solid #333; }
    .target-box { background-color: #112D20; padding: 15px; border-radius: 10px; border-left: 5px solid #00FF00; margin-bottom: 20px; }
    /* TAGS */
    .halal-tag { background-color: #004d00; color: #00ff00; padding: 2px 6px; border-radius: 4px; font-weight: bold; }
    .normal-tag { background-color: #4d0000; color: #ff4d4d; padding: 2px 6px; border-radius: 4px; font-weight: bold; }
    /* TICKER */
    .ticker-wrap { width: 100%; overflow: hidden; background-color: #121212; color: #00ff00; padding: 10px; border-bottom: 1px solid #333; margin-bottom: 20px; }
    .ticker-move { display: inline-block; white-space: nowrap; animation: ticker 30s linear infinite; font-family: 'Courier New', monospace; font-weight: bold; }
    @keyframes ticker { 0% { transform: translateX(100%); } 100% { transform: translateX(-100%); } }
</style>
""", unsafe_allow_html=True)

# --- STOCK LISTS (V12 DATA) ---
halal_list = ["TATASTEEL.NS", "ASHOKLEY.NS", "WIPRO.NS", "INFY.NS", "HCLTECH.NS", "TITAN.NS", "MARUTI.NS", "RELIANCE.NS", "SUNPHARMA.NS", "ASIANPAINT.NS"]
normal_list = ["HDFCBANK.NS", "SBIN.NS", "ICICIBANK.NS", "ITC.NS", "AXISBANK.NS", "BAJFINANCE.NS", "KOTAKBANK.NS", "MCDOWELL-N.NS"]
all_stocks = halal_list + normal_list

# --- SESSION STATE ---
if 'balance' not in st.session_state: st.session_state.balance = 1000000
if 'portfolio' not in st.session_state: st.session_state.portfolio = []
if 'pnl' not in st.session_state: st.session_state.pnl = 0

# --- SIDEBAR (V10 Style) ---
st.sidebar.title("🦅 BASHA AI V13.0")
st.sidebar.info("V10 Design + V12 Scanner")

# CAPITAL INPUT
st.sidebar.header("💼 Your Investment")
capital = st.sidebar.number_input("Total Trading Capital (₹)", value=10000, step=5000)

# UPDATED MENU
mode = st.sidebar.radio("Select Tool", [
    "📊 Target Dashboard (Home)", 
    "🌍 Universal Scanner (Halal/All)", 
    "🎮 Paper Trading (Practice)", 
    "🤖 AI Predictor", 
    "⚡ Index Scalper"
])

# --- HELPER FUNCTIONS ---
def get_data(ticker, period="1y"):
    try:
        data = yf.Ticker(ticker).history(period=period)
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
    return "🚫 NORMAL"

def ai_predict(df):
    df['Numbers'] = list(range(0, len(df)))
    X = np.array(df['Numbers']).reshape(-1, 1)
    y = df['Close'].values
    model = LinearRegression().fit(X, y)
    return model.predict(np.array([[len(df) + 1]]))[0], model.score(X, y) * 100

# --- 📺 LIVE TICKER (V10 Style) ---
try:
    nifty = yf.Ticker("^NSEI").history(period="1d")['Close'].iloc[-1]
    banknifty = yf.Ticker("^NSEBANK").history(period="1d")['Close'].iloc[-1]
    gold = yf.Ticker("GC=F").history(period="1d")['Close'].iloc[-1]
    ticker_text = f"🚀 NIFTY 50: ₹{round(nifty,2)} &nbsp;&nbsp; | &nbsp;&nbsp; 🏦 BANK NIFTY: ₹{round(banknifty,2)} &nbsp;&nbsp; | &nbsp;&nbsp; 💎 GOLD: ${round(gold,2)} &nbsp;&nbsp; | &nbsp;&nbsp; 🦅 BASHA AI V13 LIVE &nbsp;&nbsp; | &nbsp;&nbsp; 🎯 FOCUS: 1% DAILY PROFIT"
except:
    ticker_text = "🚀 MARKET DATA LOADING... | 🦅 BASHA AI V13 LIVE | 🎯 FOCUS: 1% DAILY PROFIT"

st.markdown(f"""<div class="ticker-wrap"><div class="ticker-move">{ticker_text}</div></div>""", unsafe_allow_html=True)

# ==========================================
# 📊 MODE 0: TARGET DASHBOARD (V10)
# ==========================================
if mode == "📊 Target Dashboard (Home)":
    st.title(f"💰 Financial Freedom Plan")
    
    daily_target = capital * 0.01
    weekly_target = capital * 0.05
    monthly_target = capital * 0.15

    c1, c2, c3 = st.columns(3)
    c1.metric("🌞 Daily Target (Intraday)", f"₹{int(daily_target)}", "1% Growth")
    c2.metric("📅 Weekly Target (Swing)", f"₹{int(weekly_target)}", "5% Growth")
    c3.metric("🚀 Monthly Target (Long)", f"₹{int(monthly_target)}", "15% Growth")

    st.markdown("---")
    st.subheader("📝 Discipline Rules")
    st.markdown(f"""
    <div class="target-box">
    <b>1. Daily Rule:</b> If you hit <b>₹{int(daily_target)}</b> profit, STOP TRADING immediately.<br>
    <b>2. Stop Loss:</b> Never lose more than <b>₹{int(daily_target/2)}</b> in a single day.
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 🌍 MODE 1: UNIVERSAL SCANNER (V12 INTEGRATION)
# ==========================================
elif mode == "🌍 Universal Scanner (Halal/All)":
    st.title("🌍 Market Scanner & Risk Calculator")
    
    # V12 FILTER
    st.write("### 🔍 Select Market Segment")
    filter_opt = st.radio("", ["Show Only ☪️ Halal Stocks", "Show All (Banks/Normal Included)"], horizontal=True)
    
    # V10 RISK CALCULATOR
    with st.expander("🧮 Risk Management Calculator"):
        rc1, rc2, rc3 = st.columns(3)
        r_cap = rc1.number_input("Trade Capital", value=float(capital))
        r_risk = rc2.number_input("Risk % per Trade", 1.0)
        r_sl = rc3.number_input("Stop Loss Price", 0.0)
        
        if r_sl > 0:
            risk_amount = r_cap * (r_risk/100)
            approx_entry = r_sl * 1.01 
            qty_safe = int(risk_amount / (approx_entry - r_sl))
            st.success(f"✅ Safe Quantity: **{qty_safe} Shares** (Max Loss: ₹{int(risk_amount)})")

    # SCAN LOGIC
    if filter_opt == "Show Only ☪️ Halal Stocks": scan_list = halal_list
    else: scan_list = all_stocks
    
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
                    
                    sl = int(curr['Close'] * 0.95)
                    tgt = int(curr['Close'] * 1.10)
                    
                    res.append({
                        "Stock": s, "Category": cat, "Price": round(curr['Close'],1), 
                        "Score": score, "ACTION": action, "Volume": whale, 
                        "SL": sl, "Target": tgt
                    })
            except: continue
        bar.empty()
        
        # DISPLAY TABLE WITH COLORS
        df_final = pd.DataFrame(res)
        if not df_final.empty:
            st.dataframe(df_final.style.map(lambda x: 'color: #00FF00' if 'BUY' in str(x) else ('color: #FF4444' if 'NORMAL' in str(x) else 'color: white')))
        else:
            st.warning("No stocks found.")

# ==========================================
# 🎮 MODE 2: PAPER TRADING (V10)
# ==========================================
elif mode == "🎮 Paper Trading (Practice)":
    st.title("🎮 Virtual Trading Simulator")
    daily_goal = capital * 0.01
    
    c1, c2, c3 = st.columns(3)
    c1.metric("💰 Virtual Balance", f"₹{int(st.session_state.balance)}")
    c2.metric("📈 Today's P&L", f"₹{st.session_state.pnl}")
    c3.metric("🎯 Goal Remaining", f"₹{max(0, int(daily_goal - st.session_state.pnl))}")

    if st.session_state.pnl >= daily_goal:
        st.balloons()
        st.success("🎉 DAILY TARGET HIT! YOU WON TODAY!")

    col1, col2, col3, col4 = st.columns(4)
    s_sym = col1.text_input("Symbol (e.g. HDFCBANK.NS)")
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
                st.session_state.pnl += (val * 0.02) # Demo Profit
                st.success("Sold! P&L Updated.")
        except: st.error("Invalid Symbol")

# ==========================================
# 🤖 MODE 3: AI PREDICTOR (V10 + V12 DATA)
# ==========================================
elif mode == "🤖 AI Predictor":
    st.title("🤖 AI Future Predictor")
    sel = st.selectbox("Select Stock", all_stocks) # Now shows ALL stocks
    if st.button("🔮 PREDICT"):
        df = get_data(sel)
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

# ==========================================
# ⚡ MODE 4: SCALPER (V10)
# ==========================================
elif mode == "⚡ Index Scalper":
    st.title("⚡ Nifty Scalper")
    idx = st.selectbox("Index", ["^NSEI", "^NSEBANK"])
    if st.button("⚡ ANALYZE"):
        df = get_data(idx, period="5d", interval="5m")
        curr = df.iloc[-1]
        rsi = curr['RSI_14']
        signal = "SIDEWAYS"
        if curr['Close'] > curr['EMA_50'] and rsi < 60: signal = "🚀 CALL (BUY)"
        elif curr['Close'] < curr['EMA_50'] and rsi > 40: signal = "📉 PUT (SELL)"
        
        c1, c2 = st.columns(2)
        c1.metric("Price", f"₹{round(curr['Close'],2)}")
        c2.metric("Signal", signal)