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
2.  **Engineers** a rich set of over 30 technical and time-based features.
3.  **Trains** a powerful XGBoost model to predict the next hour's price movement.
4.  **Simulates** two distinct trading strategies on out-of-sample data.
5.  **Visualizes** the results in an interactive dashboard.

The ultimate goal was to develop a strategy that could generate returns with significantly lower risk than a simple "Buy and Hold" benchmark.

---
## Technology Stack 💻
This project utilizes a modern, open-source data science stack:
* **Python 3.11**
* **Data Manipulation:** Pandas
* **Data Sourcing:** `ccxt`
* **Feature Engineering:** `pandas-ta-openbb`
* **ML Modeling:** Scikit-learn, XGBoost
* **Dependency Management:** Pipenv
* **Dashboarding:** Streamlit
* **Containerization:** Docker

---
## Project Architecture ⚙️
The project is designed as a modular, end-to-end pipeline. The `main.py` script orchestrates the execution of each step in sequence, ensuring a reproducible workflow from data ingestion to performance analysis.


*A visual representation of the data flowing from the exchange API through feature engineering, model training, and finally to the backtest simulation.*

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
* **Hyperparameter Tuning:** `GridSearchCV` with `TimeSeriesSplit` cross-validation was used to find the optimal model parameters while respecting the temporal nature of the data. The best parameters were:
```json
{
"learning_rate": 0.01,
"max_depth": 3,
"n_estimators": 100,
"subsample": 0.8
}
```
* **Performance:** On the out-of-sample test set, the final model achieved an **accuracy of 52.38%**, indicating a slight, but potentially valuable, predictive edge over a random guess.
* **Decision Rule:** Based on analysis of the model's output, a probability threshold of **55%** was chosen. A trade is only initiated if the model's confidence exceeds this level, filtering for higher-conviction signals.

---
## Backtest Results & Analysis 📈
Two strategies were simulated on out-of-sample data from **September 20, 2024, to August 23, 2025**.

**Strategy 1 (Long-Only):** Buys when the model predicts an up-move with >55% confidence.

**Strategy 2 (Long-Short):** Buys on "up" predictions and short-sells on "down" predictions.

### Performance vs. Benchmark
| Metric                    | Strategy 1 (Long-Only) | Strategy 2 (Long-Short) | Buy & Hold BTC |
| ----------------------------- | ---------------------- | ----------------------- | ---------------- |
| **Total Return (%)** | `+11.83%`              | `-1.14%`                | `+83.52%`      |
| **Sharpe Ratio** | `1.33`                 | `-0.08`                 | `1.64`         |
| **Annualized Volatility (%)**| `9.40%`                | `9.45%`                 | `46.83%`       |
| **Max Drawdown (%)** | `-3.80%`               | `-10.38%`               | `-30.94%`      |

### Key Findings
* 🏆 **Superior Risk Management:** The primary success of this project was creating a strategy that dramatically reduces risk. **Strategy 1 (Long-Only)** achieved a positive return with an **Annualized Volatility (`9.40%`)** and **Max Drawdown (`-3.80%`)** that were nearly **5 times lower** than the Buy & Hold benchmark. This demonstrates a significant advantage in capital preservation.
* ⚠️ **Profitability Challenge:** During the tested period, which was largely bullish for Bitcoin, neither ML strategy managed to outperform the simple Buy & Hold's total return. This is a common outcome for risk-managed strategies in strong bull markets.
* 📉 **Long-Short Underperformance:** **Strategy 2 (Long-Short)** performed poorly, resulting in a net loss. This indicates that while the model has a slight edge in predicting upward movements, its predictions for downward movements (`prediction=0`) were not reliable enough to be profitable after accounting for transaction costs.

### Deep Dive Analysis
The model's overall accuracy of **`52.4%`** shows a small but real predictive signal. The "Prediction Analysis" revealed that the model's maximum confidence on any "UP" prediction was only **`56.4%`**. This confirms that financial markets are extremely difficult to predict with high certainty and explains why the strategy's returns are modest. The failure of the long-short strategy suggests that future work should focus on engineering features that can better predict downturns, as this is where the current model is weakest.

---
## Future Improvements 🚀
* **Improve Short Signal:** Engineer features specifically designed to capture market downturns (e.g., "fear and greed" index, high volatility indicators) to improve the performance of the long-short strategy.
* **Expand Asset Universe:** Test the model on other volatile assets like Ethereum (ETH) or Solana (SOL) to see if its predictive power is transferable.
* **Live Deployment:** Deploy the model to a cloud service (e.g., AWS, GCP) and connect it to a broker's paper trading API to simulate live performance with real-world latency and order book dynamics.