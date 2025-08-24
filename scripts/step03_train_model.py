import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def train_model(ticker, timeframe, data_dir="data", model_dir="models"):
    """
    Trains an XGBoost model using RandomizedSearchCV with TimeSeriesSplit 
    for more efficient hyperparameter tuning.
    """
    print(f"--- Training model for {ticker} ---")
    
    # Load feature data
    features_file_path = os.path.join(data_dir, f"{ticker.replace('/', '_')}_{timeframe}_features.parquet")
    if not os.path.exists(features_file_path):
        print(f"Error: Feature data not found at {features_file_path}.")
        return

    df = pd.read_parquet(features_file_path)

    # Define features (X) and target (y)
    features = [col for col in df.columns if col not in ['timestamp', 'future_return', 'target', 'rsi_binned']]
    X = df[features]
    y = df['target']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    print(f"Training data size: {len(X_train)}, Testing data size: {len(X_test)}")
    
    # --- RandomizedSearchCV Implementation ---
    
    # 1. Define the base model to be tuned
    model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False)
    
    # 2. Define a large parameter distribution to sample from
    param_dist = {
        'n_estimators': [100, 200, 300, 400],
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2]
    }
    
    # 3. Use TimeSeriesSplit for cross-validation, which is crucial for time series data
    tscv = TimeSeriesSplit(n_splits=3)
    
    # 4. Set up and run RandomizedSearchCV
    # n_iter=50 means it will test 50 random combinations from param_dist. Total fits = 50 * 3 = 150
    # n_jobs=-1 uses all available CPU cores to speed up the search.
    # verbose=2 will print progress updates.
    # random_state=42 ensures the "random" search is the same every time you run it.
    print("--- Starting RandomizedSearchCV for hyperparameter tuning... ---")
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=50,
        cv=tscv,
        scoring='accuracy',
        verbose=2,
        n_jobs=-1,
        random_state=42
    )
    
    random_search.fit(X_train, y_train)
    
    # 5. Get the best model and its parameters from the search
    best_model = random_search.best_estimator_
    print(f"--- RandomizedSearchCV Finished ---")
    print(f"Best parameters found: {random_search.best_params_}")
    
    # --- End of RandomizedSearchCV Implementation ---

    # Evaluate the *best* model on the out-of-sample test set
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nBest Model Accuracy on Test Set: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))

    # Save the best trained model
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{ticker.replace('/', '_')}_{timeframe}_xgb_model.joblib")
    joblib.dump(best_model, model_path)
    print(f"Best model saved to {model_path}")

if __name__ == "__main__":
    train_model(ticker="BTC/USDT", timeframe="1h")