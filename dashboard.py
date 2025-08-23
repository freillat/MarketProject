import streamlit as st
import pandas as pd
import os

# --- Page Config ---
st.set_page_config(page_title="Crypto Trading Bot Dashboard", layout="wide")

# --- Title ---
st.title("🤖 Crypto Trading Bot Dashboard")

# --- Load Data ---
@st.cache_data
def load_data(ticker="BTC_USDT", timeframe="1h"):
    file_path = os.path.join("data", f"{ticker}_{timeframe}_backtest_results.csv")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, parse_dates=['timestamp'])
        df.set_index('timestamp', inplace=True)
        return df
    return None

results_df = load_data()

if results_df is not None:
    st.header("Backtest Performance Comparison")
    
    # Plot all three equity curves on the same chart
    st.line_chart(results_df[['portfolio_value_s1', 'portfolio_value_s2', 'portfolio_value_bh']])

    # Display final values for all strategies
    final_s1 = results_df['portfolio_value_s1'].iloc[-1]
    final_s2 = results_df['portfolio_value_s2'].iloc[-1]
    final_bh = results_df['portfolio_value_bh'].iloc[-1]
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Strategy 1 (Long-Only)", f"${final_s1:,.2f}")
    col2.metric("Strategy 2 (Long-Short)", f"${final_s2:,.2f}")
    col3.metric("Buy & Hold", f"${final_bh:,.2f}")
    
    st.header("Recent Data and Signals")
    # Display recent signals and data
    st.dataframe(results_df[['close', 'prediction', 'probability', 'portfolio_value_s1', 'portfolio_value_s2']].tail(20))
else:
    st.warning("No backtest results found. Please run the main pipeline first to generate the results file.")