# scripts/step01b_download_alt_data.py
import requests
import pandas as pd
import pandas_datareader.data as pdr
import os
import argparse
from datetime import datetime

def download_fear_and_greed(limit=0):
    """
    Downloads the Fear & Greed Index from alternative.me API.
    limit=0 fetches all available historical data.
    """
    print("--- Downloading Fear & Greed Index ---")
    url = f"https://api.alternative.me/fng/?limit={limit}"
    try:
        response = requests.get(url)
        response.raise_for_status() # Raise an exception for bad status codes
        data = response.json()['data']
        
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='s')
        df.rename(columns={'value': 'fear_and_greed'}, inplace=True)
        df.set_index('timestamp', inplace=True)
        df = df[['fear_and_greed']].astype(int)
        
        print(f"Successfully downloaded {len(df)} records for Fear & Greed Index.")
        return df
    except Exception as e:
        print(f"Error downloading Fear & Greed Index: {e}")
        return None

def download_fred_data(start_date, series_map):
    """
    Downloads specified series from FRED.
    """
    print("--- Downloading Macroeconomic Data from FRED ---")
    try:
        df = pdr.get_data_fred(series_map.keys(), start=start_date)
        df.rename(columns=series_map, inplace=True)
        
        print(f"Successfully downloaded {len(df)} records for {list(series_map.values())}.")
        return df
    except Exception as e:
        print(f"Error downloading FRED data: {e}")
        return None

def main(start_date, data_dir="data"):
    """
    Main function to download all alternative data, combine, and save.
    """
    # 1. Download Fear & Greed data
    fng_df = download_fear_and_greed()
    
    # 2. Download FRED data
    fred_series = {
        'VIXCLS': 'vix',           # Volatility Index
        'DGS10': 'treasury_10y'    # 10-Year Treasury Yield
    }
    fred_df = download_fred_data(start_date, fred_series)
    
    # 3. Combine the datasets
    if fng_df is not None and fred_df is not None:
        print("\n--- Combining alternative datasets ---")
        # Join the two dataframes on their datetime index
        combined_df = fng_df.join(fred_df, how='outer')
        
        # Forward-fill missing values (e.g., for weekends/holidays in FRED data)
        combined_df.ffill(inplace=True)
        combined_df.dropna(inplace=True) # Drop any remaining NaNs at the beginning
        
        # Ensure index is timezone-naive to match OHLCV data
        combined_df.index = combined_df.index.tz_localize(None)
        
        # 4. Save to Parquet file
        os.makedirs(data_dir, exist_ok=True)
        file_path = os.path.join(data_dir, "alt_data.parquet")
        combined_df.to_parquet(file_path)
        
        print(f"Combined alternative data saved to {file_path}")
        print(f"Final dataset has {len(combined_df)} records from {combined_df.index.min()} to {combined_df.index.max()}")
    else:
        print("Could not create combined dataset due to download errors.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download alternative data for the trading model.")
    parser.add_argument("--start_date", type=str, default="2021-01-01", help="Start date in YYYY-MM-DD format")
    args = parser.parse_args()
    
    main(args.start_date)