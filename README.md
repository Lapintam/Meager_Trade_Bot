# Meager Trade Bot

Meager Trade Bot is a live interactive BTC-USD trading bot that simulates trading decisions based on Monday high/low levels. The app displays a continuously updating, TradingView‑like interactive candlestick chart (with zoom/pan features), a fixed sidebar for controls, and a trade log table. It is built using Python, Streamlit, Plotly, yfinance, and other supporting libraries.

## Features

- **Live Interactive Chart:**  
  - Displays a candlestick chart with a TradingView‑like dark theme.
  - Fixed chart size with zoom and pan capabilities.
  - Overlays horizontal lines for Monday high/low levels.
  - Includes a trend line calculated via linear regression on BTC‑USD close prices.
  - Overlays buy/sell signal markers on the chart.

- **Fixed Sidebar:**  
  - Always visible with controls for trade amount and refresh interval.
  - Buttons to start/stop the live trading bot.

- **Trade Logging:**  
  - Logs simulated trades (including date, trade action, price, trade amount, and profit/loss).
  - Trade log is displayed in a table below the chart.

## Installation

1. Clone the Repository

   ```bash
   git clone https://github.com/yourusername/Meager_Trade_Bot.git
   cd Meager_Trade_Bot

2.	Set Up a Virtual Environment (Recommended)
    Create and activate a virtual environment:
	•	On macOS/Linux:
        ```bash
        python3 -m venv venv
        source venv/bin/activate

3.	Install Dependencies
Install the required packages using the provided requirements.txt file:
    ```bash
    pip install -r requirements.txt

4.	Configure Environment Variables (Optional)
If your application requires API keys or other sensitive settings, create a .env file in the root directory. For example:
    ```bash
    COINMARKETCAP_API_KEY=your_api_key_here
    COINBASE_API_KEY=your_api_key_here
    COINBASE_API_SECRET=your_api_secret_here
    COINBASE_PASSPHRASE=your_api_passphrase_here


Usage

To run the application locally:
    ```bash
    streamlit run main.py