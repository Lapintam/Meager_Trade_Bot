import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import yfinance as yf
from datetime import datetime, timedelta

# ---------- Example config ----------
SYMBOL = "BTC-USD"
TIMEFRAME = "1h"          # 1-hour data
REFRESH_INTERVAL_MS = 30000  # Update chart every 30 seconds

# Matplotlib objects
fig, ax = plt.subplots(figsize=(10,6))

# We’ll store references to our plot elements here so we can update them
price_line, = ax.plot([], [], label="BTC Price", color="blue")
monday_high_line = ax.axhline(y=0, color="red", linestyle="--", label="Monday High")
monday_low_line = ax.axhline(y=0, color="green", linestyle="--", label="Monday Low")

# For signals (buy/sell “X” markers), we’ll keep scatter plots:
buy_scatter = ax.scatter([], [], marker='X', color='lime', s=100, zorder=5, label='Buy Signals')
sell_scatter = ax.scatter([], [], marker='X', color='yellow', s=100, zorder=5, label='Sell Signals')

# 1) Grab data
def fetch_data(days=7):
    """
    Fetch the last `days` of historical data from yfinance.
    Return a DataFrame with columns [Open, High, Low, Close, Volume].
    """
    # Example: yfinance
    df = yf.download(SYMBOL, period=f"{days}d", interval=TIMEFRAME, progress=False)

    # Flatten multi-index if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(0)

    # Remove timezone from index for simpler plotting
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    return df

# 2) Compute Monday high/low
def compute_monday_levels(df):
    """
    Look for all rows that are Monday, find the Monday’s high and low.
    For a real-time chart, we’ll just use the most recent Monday we have in the data.
    """
    if df.empty:
        return None, None

    # Filter just Monday data
    monday_df = df[df.index.dayofweek == 0]
    if monday_df.empty:
        return None, None

    # We only care about the last Monday in the dataset
    last_monday_date = monday_df.index[-1].date()
    # Filter the dataset to that last Monday only
    last_monday = df[df.index.date == last_monday_date]

    if last_monday.empty:
        return None, None

    return last_monday['High'].max(), last_monday['Low'].min()

# 3) Simple signal detection
def detect_signals(df, monday_high, monday_low):
    """
    In your real bot, you’d do your real-time signal logic here.
    This is a toy example:
    - 'Buy' when we cross above Monday low
    - 'Sell' when we cross above Monday high
    Returns lists of (x, y) for buy and sell points.
    """
    if df.empty or monday_high is None or monday_low is None:
        return [], []

    buy_points = []
    sell_points = []

    close_prices = df['Close'].values
    timestamps = df.index

    # We’ll do a naive loop checking if we cross above monday_low or monday_high
    # For a real-time approach, you’d do something more robust
    above_low = False
    above_high = False

    for i in range(1, len(close_prices)):
        prev_price = close_prices[i-1]
        curr_price = close_prices[i]
        if (prev_price < monday_low) and (curr_price > monday_low) and not above_low:
            above_low = True
            buy_points.append((timestamps[i], curr_price))
        if (prev_price < monday_high) and (curr_price > monday_high) and not above_high:
            above_high = True
            sell_points.append((timestamps[i], curr_price))

    return buy_points, sell_points

# 4) Matplotlib animation init function
def init():
    """
    This runs once at the beginning of the animation to set up the axes, etc.
    """
    ax.set_title(f"{SYMBOL} Real-Time Chart")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price (USD)")
    ax.legend(loc="upper left")
    return (price_line, monday_high_line, monday_low_line, buy_scatter, sell_scatter)

# 5) Animation update function
def update(frame):
    """
    This function is called periodically by FuncAnimation. It:
      - Fetches fresh data
      - Recomputes Monday high/low
      - Redetects signals
      - Updates the lines/scatter in the chart
    """
    df = fetch_data(days=7)
    if df.empty:
        return (price_line, monday_high_line, monday_low_line, buy_scatter, sell_scatter)

    # X and Y data for the main price line
    xdata = df.index
    ydata = df['Close'].values

    # Update the price line
    price_line.set_data(xdata, ydata)
    ax.set_xlim(xdata[0], xdata[-1])
    ax.set_ylim(ydata.min() * 0.99, ydata.max() * 1.01)

    # Monday high/low
    monday_high, monday_low = compute_monday_levels(df)
    if monday_high is not None and monday_low is not None:
        monday_high_line.set_ydata(monday_high)
        monday_low_line.set_ydata(monday_low)
    else:
        # If we don’t have Monday levels, put them off the chart
        monday_high_line.set_ydata(np.nan)
        monday_low_line.set_ydata(np.nan)

    # Detect signals
    buy_points, sell_points = detect_signals(df, monday_high, monday_low)

    # Convert buy_points and sell_points to arrays for scatter
    if buy_points:
        buy_x, buy_y = zip(*buy_points)
    else:
        buy_x, buy_y = [], []
    if sell_points:
        sell_x, sell_y = zip(*sell_points)
    else:
        sell_x, sell_y = [], []

    # Update scatter data
    buy_scatter.set_offsets(np.column_stack([buy_x, buy_y]) if buy_x else [])
    sell_scatter.set_offsets(np.column_stack([sell_x, sell_y]) if sell_x else [])

    return (price_line, monday_high_line, monday_low_line, buy_scatter, sell_scatter)

# 6) Launch the animation
ani = animation.FuncAnimation(
    fig,            # figure to animate
    update,         # update function
    init_func=init, # init function
    interval=REFRESH_INTERVAL_MS,
    blit=False      # set blit=True if you want performance gains, but be mindful of multiple artists
)

plt.tight_layout()
plt.show()
