import yfinance as yf
import pandas as pd
import os
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange


def prepare_gold_data():
    ticker = "GC=F"
    print(f"Downloading advanced data for {ticker} (Gold)...")

    # 1. Download data from the last 5 years
    df = yf.download(ticker, period="5y")

    if df.empty:
        print("No data found. Check your internet connection.")
        return

    # FIX: Flatten the double header from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    print("Adding advanced technical indicators...")

    # Extract columns to flat 1D lists for the 'ta' library
    close_prices = df['Close'].squeeze()
    high_prices = df['High'].squeeze()
    low_prices = df['Low'].squeeze()

    # --- 1. MOMENTUM ---
    # RSI (Relative Strength Index)
    rsi_indicator = RSIIndicator(close=close_prices, window=14)
    df['RSI'] = rsi_indicator.rsi()

    # --- 2. TREND ---
    # MACD
    macd = MACD(close=close_prices, window_slow=26, window_fast=12,
                window_sign=9)
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()

    # EMA (Exponential Moving Average) - Reacts faster to recent price changes
    ema_indicator = EMAIndicator(close=close_prices, window=20)
    df['EMA_20'] = ema_indicator.ema_indicator()

    # --- 3. VOLATILITY ---
    # Bollinger Bands - Shows if the price is unusually high or low
    bb_indicator = BollingerBands(close=close_prices, window=20, window_dev=2)
    df['BB_High'] = bb_indicator.bollinger_hband()
    df['BB_Low'] = bb_indicator.bollinger_lband()

    # ATR (Average True Range) - Measures overall market volatility
    atr_indicator = AverageTrueRange(high=high_prices, low=low_prices,
                                     close=close_prices, window=14)
    df['ATR'] = atr_indicator.average_true_range()

    print("Removing empty rows...")
    # Drop rows with NaN values created by indicators (e.g., first 26 days for MACD)
    df = df.dropna()

    # Safely save the file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_dir = os.path.join(project_root, "data")

    os.makedirs(data_dir, exist_ok=True)
    output_path = os.path.join(data_dir, "Gold_features.csv")

    df.to_csv(output_path)

    print(f"Success! Saved {len(df)} rows with new features to: {output_path}")


if __name__ == "__main__":
    prepare_gold_data()