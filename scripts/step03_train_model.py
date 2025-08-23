import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def train_model(ticker, timeframe, data_dir="data", model_dir="models"):
    """
    Trains an XGBoost model using GridSearchCV with TimeSeriesSplit for hyperparameter tuning.
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
    
    # --- GridSearchCV Implementation ---
    
    # 1. Define the model
    model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False)
    
    # 2. Define the parameter grid to search
    # Note: This is a small grid for demonstration. For a real project, you might expand this.
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.8, 1.0]
    }
    
    # 3. Use TimeSeriesSplit for cross-validation
    # This is crucial for time series data to prevent lookahead bias.
    tscv = TimeSeriesSplit(n_splits=3)
    
    # 4. Set up and run GridSearchCV
    # n_jobs=-1 uses all available CPU cores to speed up the search.
    # verbose=2 will print progress updates.
    print("--- Starting GridSearchCV for hyperparameter tuning... ---")
    grid_search = GridSearchCV(
        estimator=model, 
        param_grid=param_grid, 
        cv=tscv, 
        scoring='accuracy', 
        verbose=2, 
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    # 5. Get the best model and its parameters
    best_model = grid_search.best_estimator_
    print(f"--- GridSearchCV Finished ---")
    print(f"Best parameters found: {grid_search.best_params_}")
    
    # --- End of GridSearchCV Implementation ---

    # Evaluate the *best* model on the test set
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