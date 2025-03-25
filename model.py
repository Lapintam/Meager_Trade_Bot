import os
import time
import hmac
import hashlib
import base64
import json
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

SYMBOL = "BTC-USD"
TIMEFRAME = "1h"
TRADE_AMOUNT_USD = 100

def get_monday_levels():
    """
    Retrieve Monday high/low levels from historical data.
    """
    try:
        today = datetime.now()
        days_since_monday = today.weekday()
        last_monday = today - timedelta(days=days_since_monday)
        last_monday = last_monday.replace(hour=0, minute=0, second=0, microsecond=0)

        data = get_crypto_data_from_coinmarketcap(days=days_since_monday + 1, interval=TIMEFRAME)
        if data.empty:
            print("No data available for the current week.")
            return None, None

        monday_data = data[data.index.dayofweek == 0]
        if monday_data.empty:
            print("No Monday data available.")
            return None, None

        monday_high = monday_data['High'].max()
        monday_low = monday_data['Low'].min()
        return monday_high, monday_low
    except Exception as e:
        print(f"Error getting Monday levels: {e}")
        return None, None

def get_current_price():
    """
    Retrieve the current BTC price using yfinance.
    """
    try:
        ticker = yf.Ticker(SYMBOL)
        data = ticker.history(period="1d", interval="1m")
        if data.empty:
            print("No current price data available")
            return None
        return data['Close'].iloc[-1]
    except Exception as e:
        print(f"Error fetching current price: {e}")
        return None

def calculate_btc_amount(usd_amount, btc_price):
    """
    Calculate BTC amount from USD amount.
    """
    if not btc_price or btc_price <= 0:
        print("Invalid BTC price for calculation")
        return 0
    return round(usd_amount / btc_price, 8)

def get_crypto_data_from_coinmarketcap(days=30, interval='1h'):
    """
    Fetch historical BTC data using yfinance.
    """
    try:
        ticker = yf.Ticker(SYMBOL)
        data = ticker.history(period=f"{days}d", interval=interval)
        if data.empty:
            print("No historical data available")
            return pd.DataFrame()
            
        data.index = pd.to_datetime(data.index)
        return data
    except Exception as e:
        print(f"Error fetching historical data: {e}")
        return pd.DataFrame()

def calculate_performance(trades_df):
    """
    Calculate performance metrics based on trade data.
    """
    if trades_df.empty:
        return {
            'total_trades': 0,
            'profit_loss': 0,
            'win_rate': 0,
            'average_profit': 0
        }
        
    metrics = {
        'total_trades': len(trades_df),
        'profit_loss': trades_df.get('profit_loss', 0).sum(),
        'win_rate': (trades_df['profit_loss'] > 0).mean() if 'profit_loss' in trades_df else 0,
        'average_profit': trades_df['profit_loss'].mean() if 'profit_loss' in trades_df else 0
    }
    return metrics

def backtest_strategy(days=30):
    """
    Backtest the trading strategy using historical data.
    """
    pass  # To be implemented if needed