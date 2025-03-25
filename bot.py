import os
import time
import hmac
import hashlib
import base64
import json
import threading
from datetime import datetime
from model import get_monday_levels, get_current_price, calculate_btc_amount, get_crypto_data_from_coinmarketcap

# Global variables for live updates
live_signals = {"buy": [], "sell": []}
trade_log = []  # Each trade: date, action, price, trade_amount, profit_loss, etc.
trading_bot_active = False
bot_thread = None

CHECK_INTERVAL = 60  # Check interval (in seconds)

def place_order(side, price, size):
    """
    Simulate placing an order. In a production bot, integrate with your brokerage API.
    """
    print(f"Simulated {side} order for {size} BTC at ${price:.2f}")
    return True

def live_trading_loop(trade_amount=100):
    global trading_bot_active, live_signals, trade_log
    trading_bot_active = True
    last_above_monday_low = False
    last_above_monday_high = False
    print("Starting Live Trading Bot...")
    while trading_bot_active:
        try:
            monday_high, monday_low = get_monday_levels()
            if monday_high is None or monday_low is None:
                print("Could not determine Monday levels. Waiting...")
                time.sleep(CHECK_INTERVAL)
                continue

            current_price = get_current_price()
            if current_price is None:
                print("Could not retrieve current price. Waiting...")
                time.sleep(CHECK_INTERVAL)
                continue

            print(f"Current BTC price: ${current_price:.2f}")
            above_monday_low = current_price > monday_low
            above_monday_high = current_price > monday_high
            current_time = datetime.now()

            # Retrieve recent market data (for logging or trend purposes)
            data = get_crypto_data_from_coinmarketcap(days=7, interval="1h")

            # Simulate buy signal (price crossing above Monday low)
            if above_monday_low and not last_above_monday_low:
                btc_amount = calculate_btc_amount(trade_amount, current_price)
                if place_order('buy', current_price, btc_amount):
                    live_signals["buy"].append((current_time, current_price))
                    trade_log.append({
                        'date': current_time,
                        'action': 'BUY',
                        'price': current_price,
                        'trade_amount_usd': trade_amount,
                        'profit_loss': 0
                    })

            # Simulate sell signal (price crossing above Monday high)
            if above_monday_high and not last_above_monday_high:
                btc_amount = calculate_btc_amount(trade_amount, current_price)
                if place_order('sell', current_price, btc_amount):
                    # For illustration, calculate a simulated profit/loss:
                    profit_loss = current_price * btc_amount - trade_amount
                    live_signals["sell"].append((current_time, current_price))
                    trade_log.append({
                        'date': current_time,
                        'action': 'SELL',
                        'price': current_price,
                        'trade_amount_usd': trade_amount,
                        'profit_loss': profit_loss
                    })

            last_above_monday_low = above_monday_low
            last_above_monday_high = above_monday_high

            time.sleep(CHECK_INTERVAL)
        except Exception as e:
            print("Error in trading loop:", e)
            time.sleep(CHECK_INTERVAL)
    print("Live Trading Bot has been stopped.")

def start_trading_bot(trade_amount=100):
    global bot_thread, trading_bot_active
    if not trading_bot_active:
        bot_thread = threading.Thread(target=live_trading_loop, kwargs={"trade_amount": trade_amount}, daemon=True)
        bot_thread.start()

def stop_trading_bot():
    global trading_bot_active
    trading_bot_active = False

def get_trade_log():
    return trade_log

def get_signals():
    return live_signals.get("buy", []), live_signals.get("sell", [])