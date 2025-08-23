import ccxt
import pandas as pd
import os
import argparse
from datetime import datetime

def download_data(ticker, timeframe, start_date, data_dir="data"):
    """
    Downloads historical OHLCV data for a given ticker and saves it as a Parquet file.
    """
    print(f"--- Downloading data for {ticker} ---")
    
    # Initialize exchange
    exchange = ccxt.binance() # You can change this to another exchange
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Convert start_date to milliseconds
    since = exchange.parse8601(start_date + 'T00:00:00Z')
    
    all_ohlcv = []
    
    while True:
        print(f"Fetching data from {exchange.iso8601(since)}...")
        ohlcv = exchange.fetch_ohlcv(ticker, timeframe, since)
        if len(ohlcv) == 0:
            break
        all_ohlcv.extend(ohlcv)
        since = ohlcv[-1][0] + 1
        
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Save data
    file_path = os.path.join(data_dir, f"{ticker.replace('/', '_')}_{timeframe}.parquet")
    df.to_parquet(file_path)
    print(f"Data for {ticker} saved to {file_path}")
    print(f"Downloaded {len(df)} records from {df['timestamp'].min()} to {df['timestamp'].max()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download historical crypto data.")
    parser.add_argument("--ticker", type=str, default="BTC/USDT", help="Ticker symbol (e.g., BTC/USDT)")
    parser.add_argument("--timeframe", type=str, default="1h", help="Timeframe (e.g., 1h, 1d)")
    parser.add_argument("--start_date", type=str, default="2021-01-01", help="Start date in YYYY-MM-DD format")
    
    args = parser.parse_args()
    download_data(args.ticker, args.timeframe, args.start_date)