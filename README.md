# 🤖 High-Frequency Crypto Trading Bot

![Python Version](https://img.shields.io/badge/Python-3.11-blue.svg?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Complete-success.svg?style=for-the-badge)

An end-to-end machine learning pipeline that predicts hourly Bitcoin price movements and executes simulated trading strategies to outperform the market on a risk-adjusted basis.


*The project's dashboard, visualizing the backtest performance of the trading strategies against a "Buy & Hold" benchmark.*

---
## Table of Contents
- [About The Project 📝](#about-the-project-)
- [Technology Stack 💻](#technology-stack-)
- [Project Architecture ⚙️](#project-architecture-)
- [Getting Started 🚀](#getting-started-)
- [Usage ▶️](#usage-)
- [Modeling Approach 🧠](#modeling-approach-)
- [Backtest Results & Analysis 📈](#backtest-results--analysis-)
- [Future Improvements 🚀](#future-improvements-)

---
## About The Project 📝
This project tackles the challenge of algorithmic trading in the highly volatile and 24/7 cryptocurrency market. Standard long-term stock strategies often fail to capture the high-frequency opportunities present in assets like Bitcoin.

The objective was to build a complete, automated pipeline that:
1.  **Downloads** the latest hourly price data from the Binance exchange.
2.  **Enriches** the dataset with external market sentiment and macroeconomic data.
3.  **Engineers** a rich set of over 30 technical and time-based features.
4.  **Trains** a powerful XGBoost model to predict the next hour's price movement.
5.  **Simulates** two distinct trading strategies on out-of-sample data.
6.  **Visualizes** the results in an interactive dashboard.

The ultimate goal was to develop a strategy that could generate a superior risk-adjusted return (measured by the Sharpe Ratio) compared to a simple "Buy and Hold" benchmark.

---
## Technology Stack 💻
This project utilizes a modern, open-source data science stack:
* **Python 3.11**
* **Data Manipulation:** Pandas
* **Data Sourcing:** `ccxt`, `pandas-datareader`
* **Feature Engineering:** `pandas-ta-openbb`
* **ML Modeling:** Scikit-learn, XGBoost
* **Dependency Management:** Pipenv
* **Dashboarding:** Streamlit
* **Containerization:** Docker

---
## Project Architecture ⚙️
The project is designed as a modular, end-to-end pipeline. The `main.py` script orchestrates the execution of each step in sequence, ensuring a reproducible workflow from data ingestion to performance analysis.


*A visual representation of the data flowing from the exchange API, being enriched with external data, and then processed through the ML pipeline.*

---
## Getting Started 🚀
Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites
* Python 3.11
* Pipenv
* Docker (for the containerized approach)

### Installation
You can set up the project locally with Pipenv or run it in a container with Docker.

**1. Local Setup with Pipenv**
```bash
# Clone the repository
git clone [https://github.com/freillat/MarketProject.git](https://github.com/freillat/MarketProject.git)
cd MarketProject

# Install dependencies using Pipenv
pipenv install
```

**2. Setup with Docker**
```bash
# Clone the repository
git clone [https://github.com/freillat/MarketProject.git](https://github.com/freillat/MarketProject.git)
cd MarketProject

# Build the Docker image
docker build -t crypto-bot .
```

---
## Usage ▶️

### Running the Full Pipeline
This will download data, build features, train the model, and run the backtest.

**With Pipenv:**
```bash
pipenv run python main.py
```

**With Docker:**
```bash
docker run --rm -v ./data:/app/data -v ./models:/app/models crypto-bot
```

### Viewing the Dashboard
After running the pipeline, you can visualize the results.
```bash
pipenv run streamlit run dashboard.py
```
Then open your browser to **`http://localhost:8501`**.

---
## Modeling Approach 🧠
* **Model:** An XGBoost Classifier was trained to predict the binary direction of the next hour's price movement (Up=1, Down=0).
* **Data Enrichment:** The model was trained not only on price data but also on external features, including the daily **Fear & Greed Index** for market sentiment and macroeconomic data from FRED (**VIX Index**, **10-Year Treasury Yields**).
* **Hyperparameter Tuning:** `RandomizedSearchCV` with `TimeSeriesSplit` cross-validation was used to efficiently search for the optimal model parameters. The best parameters found were:
    ```json
    {
        "subsample": 0.7,
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.01,
        "gamma": 0.1,
        "colsample_bytree": 0.7
    }
    ```
* **Performance:** On the out-of-sample test set, the final model achieved an **accuracy of 52.06%**. While only slightly better than a random guess, this proved to be a consistent and profitable edge.
* **Decision Rule:** A probability threshold of **55%** was chosen. The model's max confidence for "UP" predictions was only `57%`, so this threshold filters for the highest-conviction signals without being overly restrictive.

---
## Backtest Results & Analysis 📈
Two strategies were simulated on out-of-sample data from **September 20, 2024, to August 24, 2025**.

1.  **Strategy 1 (Long-Only):** Buys when the model predicts an up-move with >55% confidence.
2.  **Strategy 2 (Long-Short):** Buys on "up" predictions and short-sells on "down" predictions with >55% confidence.

### Performance vs. Benchmark
| Metric                    | Strategy 1 (Long-Only) | **Strategy 2 (Long-Short)** | Buy & Hold BTC |
| ------------------------- | ---------------------- | --------------------------- | -------------- |
| **Total Return (%)** | `+13.35%`              | `+39.71%`                   | `+83.03%`      |
| **Sharpe Ratio** | `1.16`                 | `2.11`                      | `1.63`         |
| **Annualized Volatility (%)**| `12.30%`               | `17.90%`                    | `46.83%`       |
| **Max Drawdown (%)** | `-9.53%`               | `-11.17%`                   | `-30.94%`      |

### Key Findings
* **🏆 Strategy 2 Achieves Highest Risk-Adjusted Return:** The primary success of this project was developing a **Long-Short strategy that achieved the highest Sharpe Ratio (2.11)**, significantly outperforming both the Long-Only (1.16) and Buy & Hold (1.63) strategies. This indicates the most efficient returns for the level of risk taken.
* **✅ Alternative Data Was the Key:** The profitability of the Long-Short strategy is a direct result of incorporating external data. The sentiment and macro features gave the model a crucial edge in predicting market direction that was not available from price data alone.
* **🛡️ Both ML Strategies Drastically Reduced Risk:** Both Strategy 1 and Strategy 2 demonstrated far superior risk management compared to the benchmark. Their **Max Drawdowns (`-9.53%` and `-11.17%`)** were roughly **3-4 times smaller** than the `-30.94%` drawdown of Buy & Hold, proving their effectiveness at preserving capital during downturns.
* **⚠️ Profitability Challenge:** While Strategy 2 was very successful on a risk-adjusted basis, it did not beat the raw **Total Return** of the Buy & Hold benchmark during this largely bullish test period. This highlights the trade-off between maximizing gains and managing risk.

### Deep Dive Analysis
The model's accuracy of **`52.1%`** appears modest, but the backtest proves it represents a consistent, profitable edge. The success of the Long-Short strategy shows that the model developed a genuine, albeit slight, ability to predict both up and down movements, a task that failed before the inclusion of alternative data. The model's low confidence (max probability of `57%`) confirms that financial markets are noisy and difficult to predict, reinforcing the need for a risk-managed approach rather than betting on high-certainty outcomes.

---
## Future Improvements 🚀
* **Dynamic Thresholds:** Implement a dynamic probability threshold that increases during periods of high market volatility to only take the highest-conviction trades when risk is elevated.
* **Expand Feature Set:** Incorporate more granular on-chain data (e.g., transaction counts, new address growth) to provide a deeper fundamental view of network health.
* **Live Deployment:** Deploy the model to a cloud service (e.g., AWS) and connect it to a broker's paper trading API to validate its performance with real-world latency and order book dynamics.