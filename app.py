import streamlit as st
import yfinance as yf
import pandas as pd
import time

# --- PAGE SETUP ---
st.set_page_config(page_title="Halal Trading Bot", page_icon="📈", layout="wide")

# --- SIDEBAR SETTINGS ---
st.sidebar.title("⚙️ Bot Settings")
user_name = st.sidebar.text_input("Unga Name Enna?", "Boss")
capital = st.sidebar.number_input("Capital Investment (₹)", value=5000, step=500)
target_percent = st.sidebar.slider("Profit Target (%)", 1.0, 5.0, 2.0)

st.sidebar.write("---")
st.sidebar.info("Designed for Halal Swing Trading")

# --- MAIN SCREEN ---
st.title(f"🚀 {user_name}'s Halal Trading Dashboard")
st.write(f"**Capital:** ₹{capital} | **Target Profit:** ₹{int(capital * (target_percent/100))}")

# --- HALAL STOCK LIST (Safe Stocks) ---
halal_stocks = [
    "TATASTEEL.NS", "ASHOKLEY.NS", "WIPRO.NS", "INFY.NS", 
    "HCLTECH.NS", "TITAN.NS", "ASIANPAINT.NS", 
    "HAVELLS.NS", "MARUTI.NS", "HINDUNILVR.NS"
]

# --- SCANNING BUTTON ---
if st.button("🔍 SCAN MARKET NOW"):
    st.write("### Scanning Live Market Data...")
    progress = st.progress(0)
    status_text = st.empty()
    
    found_stocks = []
    
    for i, stock in enumerate(halal_stocks):
        status_text.text(f"Checking {stock}...")
        progress.progress((i + 1) / len(halal_stocks))
        
        try:
            # Get Data
            data = yf.Ticker(stock).history(period="5d")
            
            if len(data) > 0:
                current_price = data['Close'].iloc[-1]
                prev_close = data['Close'].iloc[-2]
                
                # Simple Logic: Price dip aagi iruka? (Buy Opportunity)
                change_percent = ((current_price - prev_close) / prev_close) * 100
                
                # Budget Check
                if current_price <= capital:
                    max_qty = int(capital / current_price)
                    potential_profit = (max_qty * current_price) * (target_percent / 100)
                    
                    # Logic: Only show if potential profit is decent (> ₹50)
                    if potential_profit > 50:
                        found_stocks.append({
                            "Stock": stock,
                            "Price": f"₹{round(current_price, 2)}",
                            "Change": f"{round(change_percent, 2)}%",
                            "Buy Qty": max_qty,
                            "Est. Profit": f"₹{round(potential_profit, 2)}"
                        })
                        
        except Exception as e:
            continue
            
    progress.empty()
    status_text.empty()
    
    # --- DISPLAY RESULTS ---
    if found_stocks:
        st.success(f"✅ Scanning Complete! Found {len(found_stocks)} opportunities.")
        df = pd.DataFrame(found_stocks)
        st.table(df)
        st.markdown("💡 **Tip:** Buy in Cash (Delivery) and hold for 2-3 days.")
    else:
        st.warning("⚠️ No good trades found right now. Market might be expensive or slow.")

else:
    st.info("👆 Click the SCAN button to find stocks.")
