import pandas as pd
import pandas_ta as ta
import os
import argparse

def build_features(ticker, timeframe, use_alt_data=False, data_dir="data"):
    """
    Reads raw data, optionally merges alternative data, generates technical 
    indicators and custom features, and saves the combined dataset.
    """
    print(f"--- Building features for {ticker} ---")
    
    # --- Load Primary Data ---
    raw_file_path = os.path.join(data_dir, f"{ticker.replace('/', '_')}_{timeframe}.parquet")
    if not os.path.exists(raw_file_path):
        print(f"Error: Raw data not found at {raw_file_path}. Please run the download script first.")
        return None
    df = pd.read_parquet(raw_file_path)

    # --- Optional Alternative Data Integration ---
    if use_alt_data:
        print("--- Integrating alternative data ---")
        alt_data_path = os.path.join(data_dir, "alt_data.parquet")
        if os.path.exists(alt_data_path):
            df_alt = pd.read_parquet(alt_data_path)
            
            # Convert the index into a column
            df_alt = df_alt.reset_index() 
            
            # --- THIS IS THE FINAL FIX ---
            # Rename the newly created 'index' column to 'timestamp'
            df_alt = df_alt.rename(columns={'index': 'timestamp'})
            # --- END OF FIX ---

            df = pd.merge_asof(
                df.sort_values('timestamp'),
                df_alt.sort_values('timestamp'), # This will now work
                on='timestamp',
                direction='backward'
            )
            
            df.ffill(inplace=True)
            print("Alternative data successfully merged.")
        else:
            print(f"Warning: Alternative data file not found at {alt_data_path}. Proceeding without it.")
    
    # The rest of the script remains the same...
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.bbands(length=20, std=2, append=True)
    df.ta.atr(length=14, append=True)
    df.ta.ema(length=50, append=True)
    df.ta.ema(length=200, append=True)
    df.ta.obv(append=True)
    
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['rsi_binned'] = pd.cut(df['RSI_14'], bins=[0, 30, 70, 100], labels=['Oversold', 'Neutral', 'Overbought'])
    df['above_50_ema'] = (df['close'] > df['EMA_50']).astype(int)
    
    df['future_return'] = df['close'].pct_change().shift(-1)
    df['target'] = (df['future_return'] > 0).astype(int)
    
    df.dropna(inplace=True)
    
    features_file_path = os.path.join(data_dir, f"{ticker.replace('/', '_')}_{timeframe}_features.parquet")
    df.to_parquet(features_file_path)
    print(f"Features saved to {features_file_path}")
    print(f"Feature set has {df.shape[0]} records and {df.shape[1]} columns.")
    
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build features for the trading model.")
    parser.add_argument("--ticker", type=str, default="BTC/USDT", help="Ticker symbol")
    parser.add_argument("--timeframe", type=str, default="1h", help="Timeframe")
    parser.add_argument("--use-alt-data", action='store_true', help="Flag to enable using alternative data.")
    
    args = parser.parse_args()
    build_features(
        ticker=args.ticker, 
        timeframe=args.timeframe, 
        use_alt_data=args.use_alt_data
    )