#!/usr/bin/env python3
"""
Entry point for the local BTC-USD Trading Bot.
Initializes the Streamlit UI, starts the trading bot,
and displays a continuously updating interactive chart with a trade log.
"""

import streamlit as st
import time
import pandas as pd
from model import get_crypto_data_from_coinmarketcap, get_monday_levels
from chart import generate_live_chart
from bot import start_trading_bot, stop_trading_bot, get_trade_log, get_signals

# Set Streamlit page configuration
st.set_page_config(layout="wide", page_title="BTC-USD Trading Bot", initial_sidebar_state="expanded")

# Custom CSS to force the sidebar always open (this CSS may be tweaked as needed)
st.markdown("""
    <style>
    /* Force sidebar to always be visible */
    .css-1d391kg { 
        position: fixed;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("BTC-USD Trading Bot – Live Trading Mode")

# Sidebar controls (always visible)
st.sidebar.header("Trading Bot Controls")
trade_amount = st.sidebar.number_input("Trade Amount (USD)", value=100, step=10)
refresh_interval = st.sidebar.number_input("Refresh Interval (seconds)", value=5, step=1)

# Start/stop buttons for the live trading bot
if st.sidebar.button("Start Trading Bot"):
    start_trading_bot(trade_amount=trade_amount)
    st.sidebar.success("Trading Bot Started!")
if st.sidebar.button("Stop Trading Bot"):
    stop_trading_bot()
    st.sidebar.info("Trading Bot Stopped.")

# Placeholders for the chart and trade log below the sidebar
chart_placeholder = st.empty()
trade_log_placeholder = st.empty()

# Use Streamlit's query parameters to update the UI every few seconds
count = st.query_params.get("count", [0])[0]
try:
    count = int(count) + 1
except Exception:
    count = 0
st.query_params["count"] = count

# Main update loop (this will re-run the script repeatedly)
# Fetch market data and signals, generate the chart, and update the trade log
data = get_crypto_data_from_coinmarketcap(days=7)
monday_high, monday_low = get_monday_levels()
buy_signals, sell_signals = get_signals()
trade_log = get_trade_log()

# Generate an interactive Plotly candlestick chart with a trend line
fig = generate_live_chart(data, monday_high, monday_low, buy_signals, sell_signals)
# Set a fixed size for the chart (users can still zoom and pan)
fig.update_layout(height=900, width=1200)
chart_placeholder.plotly_chart(fig, use_container_width=True)

# Display the trade log as a table
if trade_log:
    df_log = pd.DataFrame(trade_log)
    if 'date' in df_log.columns:
        df_log['date'] = pd.to_datetime(df_log['date']).dt.strftime("%Y-%m-%d %H:%M:%S")
    trade_log_placeholder.dataframe(df_log)
else:
    trade_log_placeholder.write("No trades logged yet.")

# Wait a short period before auto-refresh (Streamlit re-runs the script on each change)
time.sleep(refresh_interval)
st.experimental_rerun()