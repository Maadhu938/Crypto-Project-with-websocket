"""
Automatically fetch data and train models for top 50 cryptocurrencies
"""
import os
import numpy as np
import pandas as pd
import requests
import time
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt
import joblib


def fetch_with_retry(url, params, max_retries=5, base_delay=10):
    """Fetch data with exponential backoff retry logic"""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 429:
                # Rate limited - wait longer
                wait_time = base_delay * (2 ** attempt)
                print(f"  ⏳ Rate limited. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise
            wait_time = base_delay * (2 ** attempt)
            print(f"  ⚠️  Request failed, retrying in {wait_time} seconds... ({attempt + 1}/{max_retries})")
            time.sleep(wait_time)
    return None

print("=" * 60)
print("AUTO-TRAINING SCRIPT FOR TOP 10 CRYPTOCURRENCIES")
print("=" * 60)

# Get top 10 coins from CoinGecko
print("\n📊 Fetching top 10 coins from CoinGecko...")
try:
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {"vs_currency": "usd", "order": "market_cap_desc", "per_page": 10, "page": 1}
    response = fetch_with_retry(url, params)
    if response is None:
        print("❌ Failed to fetch coins list")
        exit(1)
    coins = response.json()
    print(f"✅ Found {len(coins)} coins")
except Exception as e:
    print(f"❌ Failed to fetch coins: {e}")
    exit(1)

# Create models directory
os.makedirs("models", exist_ok=True)

success_count = 0
failed_coins = []

for idx, coin in enumerate(coins, 1):
    symbol = coin['id']
    name = coin['name']
    
    print(f"\n[{idx}/10] Processing {name} ({symbol})...")
    
    try:
        # Create coin-specific directory
        coin_dir = f"models/{symbol}"
        os.makedirs(coin_dir, exist_ok=True)
        
        # Check if models already exist
        if (os.path.exists(f"{coin_dir}/lstm_model.h5") and 
            os.path.exists(f"{coin_dir}/xgb_model.pkl")):
            print(f"  ⏭️  Models already exist, skipping...")
            success_count += 1
            continue
        
        # Fetch historical data (365 days)
        print(f"  📥 Fetching historical data...")
        url = f"https://api.coingecko.com/api/v3/coins/{symbol}/market_chart"
        params = {"vs_currency": "usd", "days": 365, "interval": "daily"}
        response = fetch_with_retry(url, params)
        if response is None:
            print(f"  ❌ Failed to fetch data after retries")
            failed_coins.append(symbol)
            continue
        data = response.json()
        prices = [p[1] for p in data['prices']]
        
        if len(prices) < 100:
            print(f"  ⚠️  Not enough data ({len(prices)} points), skipping...")
            failed_coins.append(symbol)
            continue
        
        print(f"  ✅ Got {len(prices)} price points")
        
        # ========== TRAIN LSTM MODEL ==========
        print(f"  🧠 Training LSTM model...")
        try:
            # Normalize prices
            scaler_lstm = MinMaxScaler()
            prices_norm = scaler_lstm.fit_transform(np.array(prices).reshape(-1, 1))
            
            # Prepare sequences (LOOKBACK = 60)
            LOOKBACK = 60
            X, y = [], []
            for i in range(LOOKBACK, len(prices_norm)):
                X.append(prices_norm[i - LOOKBACK:i, 0])
                y.append(prices_norm[i, 0])
            X, y = np.array(X), np.array(y)
            X = X.reshape((X.shape[0], X.shape[1], 1))
            
            # Train/test split
            split = int(len(X) * 0.8)
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]
            
            # Build LSTM model
            model_lstm = Sequential()
            model_lstm.add(LSTM(50, return_sequences=True, input_shape=(LOOKBACK, 1)))
            model_lstm.add(Dropout(0.2))
            model_lstm.add(LSTM(50))
            model_lstm.add(Dropout(0.2))
            model_lstm.add(Dense(1))
            model_lstm.compile(optimizer='adam', loss='mean_squared_error')
            
            # Train (reduced epochs for speed)
            model_lstm.fit(X_train, y_train, epochs=30, batch_size=64, 
                          validation_data=(X_test, y_test), verbose=0)
            
            # Evaluate LSTM model
            y_pred = model_lstm.predict(X_test, verbose=0)
            y_pred_rescaled = scaler_lstm.inverse_transform(y_pred)
            y_test_rescaled = scaler_lstm.inverse_transform(y_test.reshape(-1, 1))
            
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            from math import sqrt
            
            lstm_mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
            lstm_rmse = sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))
            lstm_mape = np.mean(np.abs((y_test_rescaled - y_pred_rescaled) / y_test_rescaled)) * 100
            lstm_r2 = r2_score(y_test_rescaled, y_pred_rescaled)
            
            # Save LSTM model, scaler, and metrics
            model_lstm.save(f"{coin_dir}/lstm_model.h5")
            joblib.dump(scaler_lstm, f"{coin_dir}/lstm_scaler.save")
            joblib.dump({
                'mae': float(lstm_mae),
                'rmse': float(lstm_rmse),
                'mape': float(lstm_mape),
                'r2': float(lstm_r2)
            }, f"{coin_dir}/lstm_metrics.pkl")
            print(f"  ✅ LSTM model saved (MAE: {lstm_mae:.2f}, R²: {lstm_r2:.4f})")
            
        except Exception as e:
            print(f"  ❌ LSTM training failed: {e}")
            failed_coins.append(symbol)
            continue
        
        # ========== TRAIN XGBOOST MODEL ==========
        print(f"  🌳 Training XGBoost model...")
        try:
            # Create dataframe with lag features
            df = pd.DataFrame({"Close": prices})
            for i in range(1, 8):  # lag1 to lag7
                df[f"lag_{i}"] = df["Close"].shift(i)
            
            df = df.dropna().reset_index(drop=True)
            
            X = df[[f"lag_{i}" for i in range(1, 8)]]
            y = df["Close"]
            
            # Scale features
            scaler_xgb = MinMaxScaler()
            X_scaled = scaler_xgb.fit_transform(X)
            
            # Train/test split
            split = int(len(X_scaled) * 0.8)
            X_train, X_test = X_scaled[:split], X_scaled[split:]
            y_train, y_test = y[:split], y[split:]
            
            # XGBoost model
            model_xgb = XGBRegressor(
                learning_rate=0.1,
                n_estimators=150,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            model_xgb.fit(X_train, y_train)
            
            # Evaluate XGBoost model
            y_pred = model_xgb.predict(X_test)
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            from math import sqrt
            
            xgb_mae = mean_absolute_error(y_test, y_pred)
            xgb_rmse = sqrt(mean_squared_error(y_test, y_pred))
            xgb_mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            xgb_r2 = r2_score(y_test, y_pred)
            
            # Save XGBoost model, scaler, and metrics
            joblib.dump(model_xgb, f"{coin_dir}/xgb_model.pkl")
            joblib.dump(scaler_xgb, f"{coin_dir}/xgb_scaler.pkl")
            joblib.dump({
                'mae': float(xgb_mae),
                'rmse': float(xgb_rmse),
                'mape': float(xgb_mape),
                'r2': float(xgb_r2)
            }, f"{coin_dir}/xgb_metrics.pkl")
            print(f"  ✅ XGBoost model saved (MAE: {xgb_mae:.2f}, R²: {xgb_r2:.4f})")
            
            success_count += 1
            print(f"  🎉 {name} training complete!")
            
        except Exception as e:
            print(f"  ❌ XGBoost training failed: {e}")
            failed_coins.append(symbol)
            continue
        
        # Rate limiting - be nice to CoinGecko API (longer delay)
        print(f"  ⏸️  Waiting 3 seconds before next coin...")
        time.sleep(3)
        
    except requests.exceptions.RequestException as e:
        print(f"  ❌ API request failed: {e}")
        failed_coins.append(symbol)
        continue
    except Exception as e:
        print(f"  ❌ Unexpected error: {e}")
        failed_coins.append(symbol)
        continue

# Summary
print("\n" + "=" * 60)
print("TRAINING SUMMARY")
print("=" * 60)
print(f"✅ Successfully trained: {success_count}/10")
if failed_coins:
    print(f"❌ Failed coins: {', '.join(failed_coins)}")
print("\n🎉 Auto-training complete!")
print("=" * 60)

