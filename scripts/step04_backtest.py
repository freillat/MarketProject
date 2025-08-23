import pandas as pd
import joblib
import os
import numpy as np
from sklearn.model_selection import train_test_split

def calculate_performance_metrics(portfolio_df, timeframe='1h'):
    if portfolio_df['portfolio_value'].empty or portfolio_df['portfolio_value'].iloc[0] == 0:
        return {}
    periods_in_year = 24 * 365
    portfolio_df['returns'] = portfolio_df['portfolio_value'].pct_change().fillna(0)
    if portfolio_df['returns'].std() == 0: return {}
    sharpe_ratio = (portfolio_df['returns'].mean() / portfolio_df['returns'].std()) * np.sqrt(periods_in_year)
    annualized_volatility = portfolio_df['returns'].std() * np.sqrt(periods_in_year)
    running_max = portfolio_df['portfolio_value'].cummax()
    drawdown = (portfolio_df['portfolio_value'] - running_max) / running_max
    max_drawdown = drawdown.min()
    return {
        "Final Portfolio Value": portfolio_df['portfolio_value'].iloc[-1],
        "Total Return (%)": (portfolio_df['portfolio_value'].iloc[-1] / portfolio_df['portfolio_value'].iloc[0] - 1) * 100,
        "Sharpe Ratio": sharpe_ratio,
        "Annualized Volatility (%)": annualized_volatility * 100,
        "Max Drawdown (%)": max_drawdown * 100
    }

def run_backtest(ticker, timeframe, initial_capital=10000, data_dir="data", model_dir="models"):
    print(f"--- Running backtest for {ticker} ---")
    
    # Load data and model
    features_file_path = os.path.join(data_dir, f"{ticker.replace('/', '_')}_{timeframe}_features.parquet")
    model_path = os.path.join(model_dir, f"{ticker.replace('/', '_')}_{timeframe}_xgb_model.joblib")
    
    if not (os.path.exists(features_file_path) and os.path.exists(model_path)):
        print("Error: Data or model file not found.")
        return

    df = pd.read_parquet(features_file_path)
    model = joblib.load(model_path)
    
    # Isolate out-of-sample data
    features = [col for col in df.columns if col not in ['timestamp', 'future_return', 'target', 'rsi_binned']]
    X = df[features]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    backtest_df = X_test.copy()
    backtest_df[['timestamp', 'open', 'close']] = df.loc[X_test.index, ['timestamp', 'open', 'close']]
    
    print(f"Backtest will run on out-of-sample data from {backtest_df['timestamp'].min()} to {backtest_df['timestamp'].max()}")
    
    # Generate predictions for the test set
    backtest_df['prediction'] = model.predict(X_test)
    backtest_df['probability'] = model.predict_proba(X_test)[:, 1]

    print("\n--- Prediction Analysis ---")
    print("Prediction Counts (0=Down, 1=Up):")
    print(backtest_df['prediction'].value_counts())
    
    print("\nConfidence (Probability) Stats for 'UP' predictions:")
    up_predictions = backtest_df[backtest_df['prediction'] == 1]
    if not up_predictions.empty:
        print(up_predictions['probability'].describe())
    else:
        print("No 'UP' predictions were made.")

    # --- Initialize variables for both strategies ---
    capital_s1 = initial_capital
    position_s1 = 0
    portfolio_value_s1 = []
    
    capital_s2 = initial_capital
    position_s2 = 0
    portfolio_value_s2 = []
    
    # --- PARAMETERS ---
    TRANSACTION_COST_PERCENT = 0.0005 # 0.05% per trade
    PROBABILITY_THRESHOLD = 0.55 # can play with this value

    for i in range(len(backtest_df) - 1):
        current_row = backtest_df.iloc[i]
        next_row = backtest_df.iloc[i+1]
        trade_price = next_row['open']

        # Strategy 1
        if current_row['prediction'] == 1 and current_row['probability'] > PROBABILITY_THRESHOLD and position_s1 == 0:
            position_s1 = (capital_s1 * (1 - TRANSACTION_COST_PERCENT)) / trade_price
            capital_s1 = 0
        elif current_row['prediction'] == 0 and position_s1 > 0:
            capital_s1 = position_s1 * trade_price * (1 - TRANSACTION_COST_PERCENT)
            position_s1 = 0
        portfolio_value_s1.append(capital_s1 + (position_s1 * current_row['close']))

        # Strategy 2
        should_go_long = current_row['prediction'] == 1 and current_row['probability'] > PROBABILITY_THRESHOLD
        should_go_short = current_row['prediction'] == 0 and (1 - current_row['probability']) > PROBABILITY_THRESHOLD
        if position_s2 == 0:
            if should_go_long:
                position_s2 = (capital_s2 * (1 - TRANSACTION_COST_PERCENT)) / trade_price
                capital_s2 = 0
            elif should_go_short:
                position_s2 = -1 * (capital_s2 * (1 - TRANSACTION_COST_PERCENT)) / trade_price
                capital_s2 += abs(position_s2 * trade_price)
        elif position_s2 > 0 and not should_go_long:
            capital_s2 = position_s2 * trade_price * (1 - TRANSACTION_COST_PERCENT)
            position_s2 = 0
        elif position_s2 < 0 and not should_go_short:
            capital_s2 += (position_s2 * trade_price) * (1 - TRANSACTION_COST_PERCENT)
            position_s2 = 0
        portfolio_value_s2.append(capital_s2 + (position_s2 * current_row['close']))

    portfolio_value_s1.append(capital_s1 + (position_s1 * backtest_df.iloc[-1]['close']))
    portfolio_value_s2.append(capital_s2 + (position_s2 * backtest_df.iloc[-1]['close']))

# --- Performance Analysis ---
    backtest_df['portfolio_value_s1'] = portfolio_value_s1
    backtest_df['portfolio_value_s2'] = portfolio_value_s2
    
    # Buy and Hold portfolio value to the DataFrame
    backtest_df['portfolio_value_bh'] = initial_capital * (backtest_df['close'] / backtest_df['close'].iloc[0])

    # Calculate metrics using the new helper function
    metrics_s1 = calculate_performance_metrics(pd.DataFrame({'portfolio_value': backtest_df['portfolio_value_s1']}))
    metrics_s2 = calculate_performance_metrics(pd.DataFrame({'portfolio_value': backtest_df['portfolio_value_s2']}))
    metrics_bh = calculate_performance_metrics(pd.DataFrame({'portfolio_value': backtest_df['portfolio_value_bh']}))
    
    print("\n--- Backtest Results (Out-of-Sample) ---")
    results_table = pd.DataFrame({
        "Strategy 1 (Long-Only)": metrics_s1,
        "Strategy 2 (Long-Short)": metrics_s2,
        "Buy and Hold": metrics_bh
    }).round(2)
    print(results_table)

    # Save the updated DataFrame with all portfolio values
    results_path = os.path.join(data_dir, f"{ticker.replace('/', '_')}_{timeframe}_backtest_results.csv")
    backtest_df.to_csv(results_path, index=False)
    print(f"\nBacktest results saved to {results_path}")

if __name__ == "__main__":
    run_backtest(ticker="BTC/USDT", timeframe="1h")