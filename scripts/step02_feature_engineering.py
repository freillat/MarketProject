import pandas as pd
import pandas_ta as ta
import os

def build_features(ticker, timeframe, data_dir="data"):
    """
    Reads raw data and generates technical indicators and custom features.
    Saves the combined dataset.
    """
    print(f"--- Building features for {ticker} ---")
    
    # Load raw data
    raw_file_path = os.path.join(data_dir, f"{ticker.replace('/', '_')}_{timeframe}.parquet")
    if not os.path.exists(raw_file_path):
        print(f"Error: Raw data not found at {raw_file_path}. Please run the download script first.")
        return None
        
    df = pd.read_parquet(raw_file_path)
    
    # 1. Generate Technical Indicators using pandas-ta
    # This is where you can get creative and add more indicators to score points!
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.bbands(length=20, std=2, append=True)
    df.ta.atr(length=14, append=True)
    df.ta.ema(length=50, append=True)
    df.ta.ema(length=200, append=True)
    df.ta.obv(append=True)
    # Add 20+ more features here from the pandas-ta library
    
    # 2. Generate Custom Derived Features
    # This is another area for creativity and points.
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['rsi_binned'] = pd.cut(df['RSI_14'], bins=[0, 30, 70, 100], labels=['Oversold', 'Neutral', 'Overbought'])
    df['above_50_ema'] = (df['close'] > df['EMA_50']).astype(int)
    
    # 3. Define the Target Variable
    # Predict the return of the next hour.
    df['future_return'] = df['close'].pct_change().shift(-1)
    
    # Create a binary target: 1 if price goes up, 0 if it goes down.
    df['target'] = (df['future_return'] > 0).astype(int)
    
    # Clean up data
    df.dropna(inplace=True)
    
    # Save features
    features_file_path = os.path.join(data_dir, f"{ticker.replace('/', '_')}_{timeframe}_features.parquet")
    df.to_parquet(features_file_path)
    print(f"Features saved to {features_file_path}")
    print(f"Feature set has {df.shape[0]} records and {df.shape[1]} columns.")
    
    return df

if __name__ == "__main__":
    # Example usage
    build_features(ticker="BTC/USDT", timeframe="1h")