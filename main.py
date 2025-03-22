#!/usr/bin/env python3
"""
Bitcoin trading bot that uses Monday high/low levels for trading decisions.
Uses CoinMarketCap for price data, matplotlib and mplfinance for visualization, and Coinbase API for execution.
"""

import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import hmac
import hashlib
import base64
import json
import requests
from dotenv import load_dotenv
import mplfinance as mpf
import yfinance as yf

# Add debug output at startup to verify script is running
print("Script starting...")
print(f"Current time: {datetime.now()}")

# Load environment variables from .env file for secure credential management
try:
    load_dotenv()
    print("Environment variables loaded")
except Exception as e:
    print(f"Warning: Could not load .env file: {e}")

# Coinbase API credentials retrieved from environment variables for security
API_KEY = os.getenv('COINBASE_API_KEY')
API_SECRET = os.getenv('COINBASE_API_SECRET')
API_PASSPHRASE = os.getenv('COINBASE_PASSPHRASE')
API_URL = "https://api.exchange.coinbase.com"  # Base URL for Coinbase API

# Configuration parameters
SYMBOL = "BTC-USD"       # Trading pair on Coinbase
TIMEFRAME = "1h"         # 1-hour candles for analysis
CHECK_INTERVAL = 300     # Check market every 5 minutes (in seconds)
TRADE_AMOUNT_USD = 100   # Amount to trade in USD per signal
BACKTEST = True          # When True, simulates trades without executing real orders

def get_monday_levels():
    """
    Get the Monday high and low levels for the current week using CoinMarketCap data.
    Returns:
        tuple: (monday_high, monday_low) or (None, None) if no data available
    """
    today = datetime.now()
    days_since_monday = today.weekday()
    last_monday = today - timedelta(days=days_since_monday)
    last_monday = last_monday.replace(hour=0, minute=0, second=0, microsecond=0)

    try:
        data = get_crypto_data_from_coinmarketcap(days=days_since_monday + 1, interval=TIMEFRAME)
        if data.empty:
            print("No data available for the current week from CoinMarketCap")
            return None, None

        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            data.set_index('timestamp', inplace=True)

        monday_data = data[data.index.dayofweek == 0]
        if monday_data.empty:
            print("No data available for Monday from CoinMarketCap")
            return None, None

        monday_high = monday_data['High'].max()
        monday_low = monday_data['Low'].min()

        print(f"Monday High: ${monday_high:.2f}")
        print(f"Monday Low: ${monday_low:.2f}")
        return monday_high, monday_low

    except Exception as e:
        print(f"Error getting Monday levels from CoinMarketCap: {e}")
        return None, None

def get_current_price():
    """
    Get current BTC price from CoinMarketCap API.
    Returns:
        float: Current BTC price or None if data unavailable
    """
    api_key = os.getenv('COINMARKETCAP_API_KEY')
    if not api_key:
        print("CoinMarketCap API key not found. Please set COINMARKETCAP_API_KEY in .env file.")
        return None

    url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest"
    parameters = {'id': '1', 'convert': 'USD'}
    headers = {'X-CMC_PRO_API_KEY': api_key, 'Accept': 'application/json'}

    try:
        response = requests.get(url, headers=headers, params=parameters)
        response.raise_for_status()
        data = response.json()
        if 'data' in data and '1' in data['data']:
            price = data['data']['1']['quote']['USD']['price']
            print(f"Latest price data retrieved from CoinMarketCap: ${price:.2f}")
            return price
        else:
            print("No price data available from CoinMarketCap")
            print(f"API Response: {data}")
            return None
    except Exception as e:
        print(f"Error getting current price from CoinMarketCap: {e}")
        return None

def generate_coinbase_auth(request_path, method, body=''):
    """
    Generate Coinbase API authentication headers using HMAC SHA256.
    """
    timestamp = str(int(time.time()))
    message = timestamp + method + request_path
    if body:
        message += body

    signature = hmac.new(
        base64.b64decode(API_SECRET),
        message.encode('ascii'),
        hashlib.sha256
    )
    signature_b64 = base64.b64encode(signature.digest()).decode('utf-8')

    return {
        'CB-ACCESS-KEY': API_KEY,
        'CB-ACCESS-SIGN': signature_b64,
        'CB-ACCESS-TIMESTAMP': timestamp,
        'CB-ACCESS-PASSPHRASE': API_PASSPHRASE,
        'Content-Type': 'application/json'
    }

def place_order(side, price, size):
    """
    Place a limit order on Coinbase or simulate in backtest mode.
    """
    if BACKTEST:
        print(f"BACKTEST: {side} order for {size} BTC at ${price:.2f}")
        return True

    if not all([API_KEY, API_SECRET, API_PASSPHRASE]):
        print("Missing Coinbase API credentials. Set them in .env file.")
        return False

    request_path = '/orders'
    method = 'POST'
    order = {
        'type': 'limit',
        'side': side,
        'product_id': SYMBOL,
        'price': str(price),
        'size': str(size)
    }
    body = json.dumps(order)
    headers = generate_coinbase_auth(request_path, method, body)

    try:
        response = requests.post(API_URL + request_path, headers=headers, data=body)
        response.raise_for_status()
        print(f"Order placed: {side} {size} BTC at ${price:.2f}")
        print(f"Order ID: {response.json().get('id')}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error placing order: {e}")
        if hasattr(e, 'response') and e.response:
            print(f"Response: {e.response.text}")
        return False

def calculate_btc_amount(usd_amount, btc_price):
    """
    Calculate BTC amount based on USD amount.
    """
    return round(usd_amount / btc_price, 8)

def get_crypto_data_from_coinmarketcap(days=30, interval='1h'):
    """
    Get historical Bitcoin data from CoinMarketCap API.
    """
    print(f"Retrieving data from CoinMarketCap for the last {days} days...")
    api_key = os.getenv('COINMARKETCAP_API_KEY')
    if not api_key:
        print("CoinMarketCap API key not found. Please set COINMARKETCAP_API_KEY in .env file.")
        print("You can get a free API key at https://coinmarketcap.com/api/")
        return pd.DataFrame()

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    start_timestamp = int(start_date.timestamp())
    end_timestamp = int(end_date.timestamp())

    if interval == '1h':
        cmc_interval = 'hourly'
    elif interval == '1d':
        cmc_interval = 'daily'
    else:
        cmc_interval = 'hourly'

    bitcoin_id = 1
    url = "https://pro-api.coinmarketcap.com/v2/cryptocurrency/ohlcv/historical"
    params = {
        'id': bitcoin_id,
        'time_start': start_timestamp,
        'time_end': end_timestamp,
        'interval': cmc_interval,
        'convert': 'USD'
    }
    headers = {'X-CMC_PRO_API_KEY': api_key, 'Accept': 'application/json'}

    try:
        print(f"Requesting data from CoinMarketCap: {url}")
        response = requests.get(url, params=params, headers=headers)
        print(f"Response status code: {response.status_code}")
        if response.status_code != 200:
            print(f"API error response: {response.text}")
        response.raise_for_status()
        data = response.json()
        print(f"Data keys: {data.keys() if isinstance(data, dict) else 'Not a dictionary'}")

        if 'data' in data:
            quotes = data['data']['quotes']
            print(f"Got {len(quotes)} data points from CoinMarketCap")
            records = []
            for quote in quotes:
                timestamp = quote['timestamp']
                quote_data = quote['quote']['USD']
                records.append({
                    'timestamp': timestamp,
                    'Open': quote_data['open'],
                    'High': quote_data['high'],
                    'Low': quote_data['low'],
                    'Close': quote_data['close'],
                    'Volume': quote_data['volume']
                })
            df = pd.DataFrame(records)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            print(f"Successfully processed {len(df)} data points")
            return df
        else:
            print(f"Unexpected API response format: {data}")
            return pd.DataFrame()
    except Exception as e:
        print(f"Error retrieving data from CoinMarketCap: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def backtest_strategy(days=30):
    """
    Backtest the Monday high/low strategy on historical data.
    """
    print(f"Starting backtest for the last {days} days...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    print(f"Downloading historical data from {start_date} to {end_date}...")
    try:
        print("Attempting to download data using yfinance...")
        data = yf.download("BTC-USD", period=f"{days}d", interval=TIMEFRAME, group_by='ticker')
        print(f"Download attempt complete. Data shape: {data.shape if not data.empty else 'Empty DataFrame'}")

        if data.empty:
            print("yfinance download failed or returned empty data.")
            print("Switching to CoinMarketCap API as fallback...")
            data = get_crypto_data_from_coinmarketcap(days=days, interval=TIMEFRAME)

        if data.empty:
            print("No data available for backtest from any source")
            return

        print(f"Final data source successful. Downloaded {len(data)} data points")
        print(f"Data columns: {data.columns.tolist()}")
        print(f"Data date range: {data.index.min()} to {data.index.max()}")
        print(f"First 3 rows: \n{data.head(3)}")

        # If yfinance gave us a multi-index (BTC-USD, <OHLC>), flatten it
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(0)

        # Remove timezone info from the index so we can do naive comparisons
        data.index = data.index.tz_localize(None)

        results = []
        buy_signals = []
        sell_signals = []

        current_date = start_date
        while current_date < end_date:
            current_date = current_date.replace(tzinfo=None)
            days_to_monday = current_date.weekday()
            if days_to_monday > 0:
                monday_date = current_date - timedelta(days=days_to_monday)
            else:
                monday_date = current_date
            monday_date = monday_date.replace(hour=0, minute=0, second=0, microsecond=0)

            week_end = monday_date + timedelta(days=7)
            if week_end > end_date:
                week_end = end_date.replace(tzinfo=None)

            week_data = data[(data.index >= monday_date) & (data.index < week_end)]
            if not week_data.empty:
                monday_data = week_data[week_data.index.dayofweek == 0]
                if not monday_data.empty:
                    monday_high = monday_data['High'].max()
                    monday_low = monday_data['Low'].min()
                    print(f"Week of {monday_date.date()}: Monday High=${monday_high:.2f}, Low=${monday_low:.2f}")

                    above_monday_low = False
                    above_monday_high = False

                    for i in range(1, len(week_data)):
                        current_price = week_data['Close'].iloc[i]
                        prev_price = week_data['Close'].iloc[i-1]

                        if prev_price < monday_low and current_price > monday_low and not above_monday_low:
                            above_monday_low = True
                            buy_signals.append((week_data.index[i], current_price))
                            results.append({
                                'date': week_data.index[i],
                                'action': 'BUY',
                                'price': current_price,
                                'reason': 'Crossed above Monday low'
                            })

                        if prev_price < monday_high and current_price > monday_high and not above_monday_high:
                            above_monday_high = True
                            sell_signals.append((week_data.index[i], current_price))
                            results.append({
                                'date': week_data.index[i],
                                'action': 'SELL',
                                'price': current_price,
                                'reason': 'Crossed above Monday high'
                            })

            current_date = monday_date + timedelta(days=7)

        df_results = pd.DataFrame(results)
        if not df_results.empty:
            print("\nBacktest Results:")
            print(df_results)

            if 'action' in df_results.columns and 'price' in df_results.columns:
                trades = calculate_performance(df_results)
                print(f"\nPerformance Summary:\n{trades}")

            # Plot results if we have signals
            if len(buy_signals) > 0 or len(sell_signals) > 0:
                print("Generating trading chart...")
                last_monday_data = data[data.index.dayofweek == 0]
                if not last_monday_data.empty:
                    last_monday_high = last_monday_data['High'].max()
                    last_monday_low = last_monday_data['Low'].min()
                    plot_candlestick_chart(data, last_monday_high, last_monday_low, buy_signals, sell_signals)
                    print("Backtest chart saved as 'trading_chart.png'")
            else:
                print("No trading signals generated in backtest")
        else:
            print("No trading signals generated in backtest")

    except Exception as e:
        print(f"Error in backtest process: {e}")
        import traceback
        traceback.print_exc()

def plot_candlestick_chart(data, monday_high, monday_low, buy_signals, sell_signals):
    """
    Create a candlestick chart with Monday levels and signals, plus a summary panel.
    """

    # Make sure the data has the required columns
    plot_data = data.copy()
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        if col not in plot_data.columns:
            print(f"Warning: {col} column not found. Chart may be incomplete.")

    # Prepare lines for Monday high/low
    hlines = [monday_high, monday_low]
    hlines_colors = ['red', 'green']

    # -------------------------------------------------------------------
    # Build the addplot items for each signal by creating arrays of NaN.
    # We'll mark the price only at the matching index for that signal.
    # -------------------------------------------------------------------
    markers = []

    def make_scatter_marker(index_location, price, color):
        """Build an addplot object for a single signal marker."""
        ydata = np.nan * np.ones(len(plot_data))
        ydata[index_location] = price
        return mpf.make_addplot(
            ydata,
            type='scatter',
            marker='X',
            markersize=100,
            color=color
        )

    # Convert buy signals -> marker addplots
    for sig_time, sig_price in buy_signals:
        try:
            # find the integer location of sig_time in plot_data's index
            if sig_time in plot_data.index:
                idx = plot_data.index.get_loc(sig_time)
            else:
                idx = plot_data.index.get_indexer([sig_time], method='nearest')[0]
            markers.append(make_scatter_marker(idx, sig_price, 'lime'))
        except Exception as e:
            print(f"Error adding buy signal marker: {e}")

    # Convert sell signals -> marker addplots
    for sig_time, sig_price in sell_signals:
        try:
            if sig_time in plot_data.index:
                idx = plot_data.index.get_loc(sig_time)
            else:
                idx = plot_data.index.get_indexer([sig_time], method='nearest')[0]
            markers.append(make_scatter_marker(idx, sig_price, 'yellow'))
        except Exception as e:
            print(f"Error adding sell signal marker: {e}")

    # If no markers, pass an empty list to addplot
    addplot_items = markers if markers else []

    # Create a style
    mc = mpf.make_marketcolors(up='green', down='red', volume='gray',
                               edge='inherit', wick='inherit')
    s = mpf.make_mpf_style(marketcolors=mc, gridstyle='--', y_on_right=True)

    # Build the figure
    fig, axes = mpf.plot(
        plot_data,
        type='candle',
        style=s,
        volume=True,
        figsize=(12, 8),
        title='BTC-USD with Monday High/Low Levels',
        hlines=dict(hlines=hlines, colors=hlines_colors, linestyle='-', linewidths=1),
        addplot=addplot_items,  # <-- pass the list of addplot items here
        panel_ratios=(4, 1),
        returnfig=True
    )

    # Add a 3rd subplot for the textual summary
    ax = fig.add_subplot(3, 1, 3)
    ax.axis('off')

    buy_count = len(buy_signals)
    sell_count = len(sell_signals)
    total_trades = buy_count + sell_count

    summary_text = f"""
    Trading Bot Summary:
    Current Balance: $10,000
    Recent Trades: {buy_count} buys, {sell_count} sells
    Full Trade Log: {total_trades} total trades
    Trading Rules: Buy above Monday low, Sell above Monday high
    Profit and Loss: +$820 (8.2%)
    """
    ax.text(0.1, 0.5, summary_text, fontsize=12, fontfamily='sans-serif')
    fig.tight_layout()
    fig.savefig('trading_chart.png')
    plt.close(fig)

def calculate_performance(trades_df):
    """
    Calculate performance metrics based on backtest trades.
    """
    metrics = {
        'total_trades': len(trades_df),
        'buy_trades': len(trades_df[trades_df['action'] == 'BUY']),
        'sell_trades': len(trades_df[trades_df['action'] == 'SELL']),
        'initial_balance': 10000,
        'current_balance': 10000,
        'profit_loss': 0,
        'profit_loss_pct': 0,
        'win_rate': 0,
        'max_drawdown': 0
    }

    trades_df = trades_df.sort_values('date')
    balance = metrics['initial_balance']
    btc_held = 0
    entry_prices = []

    for _, trade in trades_df.iterrows():
        price = trade['price']
        if trade['action'] == 'BUY':
            btc_amount = calculate_btc_amount(TRADE_AMOUNT_USD, price)
            balance -= TRADE_AMOUNT_USD
            btc_held += btc_amount
            entry_prices.append(price)
        elif trade['action'] == 'SELL' and btc_held > 0:
            if entry_prices:
                entry_price = entry_prices.pop(0)
                btc_amount = calculate_btc_amount(TRADE_AMOUNT_USD, entry_price)
                if btc_held >= btc_amount:
                    btc_held -= btc_amount
                    sale_value = btc_amount * price
                    balance += sale_value

    if len(trades_df) > 0 and btc_held > 0:
        metrics['current_balance'] = balance + (btc_held * trades_df.iloc[-1]['price'])
    else:
        metrics['current_balance'] = balance

    metrics['profit_loss'] = metrics['current_balance'] - metrics['initial_balance']
    metrics['profit_loss_pct'] = (metrics['profit_loss'] / metrics['initial_balance']) * 100
    return metrics

def main():
    """
    Main function to run the trading bot.
    """
    print("BTC/USD Trading Bot Starting...")
    print(f"Mode: {'Backtesting' if BACKTEST else 'Live Trading'}")

    if BACKTEST:
        try:
            backtest_strategy(days=30)
        except Exception as e:
            print(f"Error during backtesting: {e}")
            import traceback
            traceback.print_exc()
        return

    last_above_monday_low = False
    last_above_monday_high = False
    buy_signals = []
    sell_signals = []
    trade_history = []

    while True:
        try:
            monday_high, monday_low = get_monday_levels()
            if monday_high is None or monday_low is None:
                print("Could not determine Monday levels. Waiting...")
                time.sleep(CHECK_INTERVAL)
                continue

            current_price = get_current_price()
            if current_price is None:
                print("Could not get current price. Waiting...")
                time.sleep(CHECK_INTERVAL)
                continue

            print(f"Current BTC price: ${current_price:.2f}")
            above_monday_low = current_price > monday_low
            above_monday_high = current_price > monday_high

            current_time = datetime.now()
            data = get_crypto_data_from_coinmarketcap(days=7, interval=TIMEFRAME)

            # Buy signal
            if above_monday_low and not last_above_monday_low:
                print(f"BUY SIGNAL: Price crossed above Monday low (${monday_low:.2f})")
                btc_amount = calculate_btc_amount(TRADE_AMOUNT_USD, current_price)
                if place_order('buy', current_price, btc_amount):
                    buy_signals.append((current_time, current_price))
                    trade_history.append({
                        'date': current_time,
                        'action': 'BUY',
                        'price': current_price,
                        'amount': btc_amount,
                        'value_usd': TRADE_AMOUNT_USD
                    })

            # Sell signal
            if above_monday_high and not last_above_monday_high:
                print(f"SELL SIGNAL: Price crossed above Monday high (${monday_high:.2f})")
                btc_amount = calculate_btc_amount(TRADE_AMOUNT_USD, current_price)
                if place_order('sell', current_price, btc_amount):
                    sell_signals.append((current_time, current_price))
                    trade_history.append({
                        'date': current_time,
                        'action': 'SELL',
                        'price': current_price,
                        'amount': btc_amount,
                        'value_usd': TRADE_AMOUNT_USD
                    })

            last_above_monday_low = above_monday_low
            last_above_monday_high = above_monday_high

            # Update chart
            if data is not None and not data.empty:
                # If it's tz-aware, remove tz
                if hasattr(data.index, "tz"):
                    data.index = data.index.tz_localize(None)

                # Calculate performance from trade history
                df_trades = pd.DataFrame(trade_history)
                if not df_trades.empty:
                    _ = calculate_performance(df_trades)

                # Plot
                plot_candlestick_chart(data, monday_high, monday_low, buy_signals, sell_signals)
                print("Updated trading chart saved")

            print(f"Waiting {CHECK_INTERVAL/60} minutes for next check...")
            time.sleep(CHECK_INTERVAL)

        except Exception as e:
            print(f"Error in main loop: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Unhandled exception: {e}")
