import yfinance as yf
import pandas as pd
import os

# test

def download_stock_data(ticker="AAPL", period="5y"):
    print(f"Fetching data for {ticker}...")
    # Pobieramy dane z ostatnich 5 lat
    df = yf.download(ticker, period=period)

    if df.empty:
        print("No data found. Check the ticker symbol.")
        return

    # Tworzymy ścieżkę do zapisu
    output_path = os.path.join("data", f"{ticker}_data.csv")
    df.to_csv(output_path)

    print(f"Success! Saved {len(df)} rows to {output_path}")
    print(df.head())


if __name__ == "__main__":
    # Możecie tu wpisać dowolny ticker, np. "TSLA", "BTC-USD" lub "ALE.WA"
    download_stock_data("AAPL")