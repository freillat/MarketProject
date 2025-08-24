from scripts.step01_download_data import download_data
from scripts.step01b_download_alt_data import main as download_alt_data
from scripts.step02_feature_engineering import build_features
from scripts.step03_train_model import train_model
from scripts.step04_backtest import run_backtest

TICKER = "BTC/USDT"
TIMEFRAME = "1h"
START_DATE = "2021-01-01"
USE_ALT_DATA = True # Set to True or False to control the feature

def main():
    """
    Runs the full end-to-end ML trading pipeline.
    """
    print("--- Starting ML Trading Pipeline ---")
    
    # Step 1: Download Data
    download_data(ticker=TICKER, timeframe=TIMEFRAME, start_date=START_DATE)
    if USE_ALT_DATA:
        download_alt_data(start_date=START_DATE)
    
    # Step 2: Feature Engineering
    build_features(ticker=TICKER, timeframe=TIMEFRAME, use_alt_data=USE_ALT_DATA)
    
    # Step 3: Model Training
    train_model(ticker=TICKER, timeframe=TIMEFRAME)
    
    # Step 4: Backtesting
    run_backtest(ticker=TICKER, timeframe=TIMEFRAME)
    
    print("--- ML Trading Pipeline Finished ---")

if __name__ == "__main__":
    main()