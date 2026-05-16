import yfinance as yf
import pandas as pd
import os
from ta.momentum import RSIIndicator
from ta.trend import MACD


def prepare_gold_data():
    ticker = "GC=F"
    print(f"Downloading data for {ticker} (Gold)...")

    # 1. Download data from the last 5 years
    df = yf.download(ticker, period="5y")

    if df.empty:
        print("No data found. Check your internet connection.")
        return

    # FIX: Flatten the double header from yfinance to a single row
    # This prevents putting "GC=F" text and <null> values into our data rows
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    print("Adding RSI and MACD indicators...")

    # 2. Get close prices as a simple 1D list
    close_prices = df['Close'].squeeze()

    # 3. Calculate RSI and MACD indicators
    rsi_indicator = RSIIndicator(close=close_prices, window=14)
    df['RSI'] = rsi_indicator.rsi()

    macd = MACD(close=close_prices, window_slow=26, window_fast=12,
                window_sign=9)
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()

    print("Removing empty rows...")
    # 4. Drop empty rows (NaN) so we do not break the LSTM model
    df = df.dropna()

    # 5. Safely save the file to the 'data/' folder
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_dir = os.path.join(project_root, "data")

    os.makedirs(data_dir, exist_ok=True)
    output_path = os.path.join(data_dir, "Gold_features.csv")

    df.to_csv(output_path)

    print(f"Success! Saved {len(df)} rows of clean data to: {output_path}")


if __name__ == "__main__":
    prepare_gold_data()
