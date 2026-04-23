import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import requests
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO
from tensorflow.keras.models import load_model
import joblib
import time
import threading
from datetime import datetime

port = int(os.environ.get('PORT', 5000))

app = Flask(__name__)
CORS(app, origins=["https://cryptoapp.maadhuavati.in"])
socketio = SocketIO(app, cors_allowed_origins="*")

_model_cache = {}
_price_cache = {}
_PRICE_CACHE_TTL = 15 * 60  # 15 minutes

_LIVE_COINS = [
    "bitcoin",
    "ethereum",
    "tether",
    "solana",
    "binancecoin",
    "ripple"
]
_LIVE_COIN_LABELS = {
    "bitcoin": "Bitcoin (BTC)",
    "ethereum": "Ethereum (ETH)",
    "tether": "Tether (USDT)",
    "solana": "Solana (SOL)",
    "binancecoin": "BNB (BNB)",
    "ripple": "Ripple (XRP)"
}
_SYMBOL_MAP = {
    "bitcoin": "BTC",
    "ethereum": "ETH",
    "tether": "USDT",
    "solana": "SOL",
    "binancecoin": "BNB",
    "ripple": "XRP",
    "cardano": "ADA",
    "dogecoin": "DOGE",
    "polygon": "MATIC",
    "polkadot": "DOT"
}
_LIVE_COIN_IDS = {
    "bitcoin": "90",
    "ethereum": "80",
    "tether": "518",
    "solana": "48543",
    "binancecoin": "2710",
    "ripple": "58"
}
_live_thread = None
_live_thread_lock = threading.Lock()


def fetch_recent_prices(symbol, days=7):
    """Fetch recent prices from CoinGecko API with caching."""
    cache_key = (symbol, days)
    now = time.time()
    cached = _price_cache.get(cache_key)
    if cached and (now - cached["timestamp"] < _PRICE_CACHE_TTL):
        return cached["prices"]

    try:
        url = f"https://api.coingecko.com/api/v3/coins/{symbol}/market_chart"
        params = {"vs_currency": "usd", "days": days}
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        prices = [p[1] for p in response.json()['prices']]
        _price_cache[cache_key] = {"timestamp": now, "prices": prices}
        return prices
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to fetch prices: {str(e)}")


def fetch_simple_prices(symbols):
    """Fetch current prices for a list of symbols from CoinLore."""
    ids = ",".join([_LIVE_COIN_IDS.get(s, "90") for s in symbols])
    url = f"https://api.coinlore.net/api/ticker/?id={ids}"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    data = response.json()
    payload = []
    timestamp = int(time.time())
    for symbol in symbols:
        coin_id = _LIVE_COIN_IDS.get(symbol)
        entry = next((c for c in data if str(c["id"]) == coin_id), None)
        if not entry:
            continue
        payload.append({
            "id": symbol,
            "label": _LIVE_COIN_LABELS.get(symbol, symbol.title()),
            "price": float(entry.get("price_usd", 0)),
            "change24h": float(entry.get("percent_change_24h", 0.0))
        })
    return {"timestamp": timestamp, "coins": payload}

def load_models(symbol):
    """Load LSTM and XGBoost models for a given symbol"""
    model_dir = f"models/{symbol}"
    paths = {
        'lstm_model': os.path.join(model_dir, "lstm_model.h5"),
        'lstm_scaler': os.path.join(model_dir, "lstm_scaler.save"),
        'xgb_model': os.path.join(model_dir, "xgb_model.pkl"),
        'xgb_scaler': os.path.join(model_dir, "xgb_scaler.pkl"),
        'lstm_metrics': os.path.join(model_dir, "lstm_metrics.pkl"),
        'xgb_metrics': os.path.join(model_dir, "xgb_metrics.pkl")
    }
    
    # Try new structure
    if os.path.exists(model_dir) and all(os.path.exists(paths[k]) for k in ['lstm_model', 'lstm_scaler', 'xgb_model', 'xgb_scaler']):
        if symbol in _model_cache:
            return _model_cache[symbol]
        try:
            models = (
                load_model(paths['lstm_model']),
                joblib.load(paths['lstm_scaler']),
                joblib.load(paths['xgb_model']),
                joblib.load(paths['xgb_scaler']),
                joblib.load(paths['lstm_metrics']) if os.path.exists(paths['lstm_metrics']) else None,
                joblib.load(paths['xgb_metrics']) if os.path.exists(paths['xgb_metrics']) else None
            )
            _model_cache[symbol] = models
            return models
        except Exception as e:
            raise Exception(f"Failed to load models: {str(e)}")
    
    # Fallback: old structure for bitcoin
    if symbol == "bitcoin":
        old_paths = {
            'lstm_model': "models/lstm_model.h5",
            'lstm_scaler': "models/lstm_scaler.save",
            'xgb_model': "models/xgb_btc_model.pkl",
            'xgb_scaler': "models/xgb_scaler.pkl"
        }
        if all(os.path.exists(old_paths[k]) for k in old_paths):
            cache_key = "bitcoin_old"
            if cache_key in _model_cache:
                return _model_cache[cache_key]
            try:
                models = (
                    load_model(old_paths['lstm_model']),
                    joblib.load(old_paths['lstm_scaler']),
                    joblib.load(old_paths['xgb_model']),
                    joblib.load(old_paths['xgb_scaler']),
                    None, None
                )
                _model_cache[cache_key] = models
                return models
            except Exception as e:
                raise Exception(f"Failed to load models: {str(e)}")
    
    return None, None, None, None, None, None

def prepare_lstm_input(prices):
    """Prepare input for LSTM model (last 60 prices)"""
    if len(prices) < 60:
        raise ValueError("Not enough data for LSTM prediction (need at least 60 prices)")
    return np.array(prices[-60:]).reshape(-1, 1)

def prepare_xgb_input(prices):
    """Prepare input for XGBoost model (7 lag features)"""
    if len(prices) < 7:
        raise ValueError("Not enough data for XGBoost prediction (need at least 7 prices)")
    return np.array([prices[-i] for i in range(1, 8)]).reshape(1, -1)

@app.route("/predict", methods=["GET"])
def predict():
    """Predict price for any coin using LSTM and XGBoost models"""
    symbol = request.args.get("symbol")
    if not symbol:
        return jsonify({"error": "Missing symbol parameter"}), 400
    
    try:
        lstm_model, lstm_scaler, xgb_model, xgb_scaler, lstm_metrics, xgb_metrics = load_models(symbol)
        if lstm_model is None:
            return jsonify({"error": "Model not found"}), 404
        
        prices = fetch_recent_prices(symbol, days=365)
        if len(prices) < 60:
            return jsonify({"error": "Not enough data"}), 400
        
        # LSTM Prediction
        lstm_input = prepare_lstm_input(prices)
        lstm_scaled = lstm_scaler.transform(lstm_input).flatten()
        lstm_pred_norm = lstm_model.predict(np.array(lstm_scaled).reshape(1, 60, 1), verbose=0)[0][0]
        lstm_prediction = lstm_scaler.inverse_transform([[lstm_pred_norm]])[0][0]
        
        # XGBoost Prediction
        xgb_input = prepare_xgb_input(prices)
        xgb_scaled = xgb_scaler.transform(xgb_input)
        xgb_prediction = xgb_model.predict(xgb_scaled)[0]
        
        return jsonify({
            "symbol": symbol,
            "lstm_prediction": float(lstm_prediction),
            "xgboost_prediction": float(xgb_prediction),
            "history": prices[-30:] if len(prices) >= 30 else prices,
            "lstm_metrics": lstm_metrics if lstm_metrics else {},
            "xgb_metrics": xgb_metrics if xgb_metrics else {},
            "timestamp": int(time.time())
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict_lstm", methods=["POST"])
def predict_lstm():
    try:
        lstm_model, lstm_scaler, _, _, _, _ = load_models("bitcoin")
        if lstm_model is None:
            return jsonify({"error": "Bitcoin models not found"}), 404
        prices = fetch_recent_prices("bitcoin", days=365)
        if len(prices) < 60:
            return jsonify({"error": "Not enough data to predict"}), 400
        lstm_input = prepare_lstm_input(prices)
        lstm_scaled = lstm_scaler.transform(lstm_input).flatten()
        pred_norm = lstm_model.predict(np.array(lstm_scaled).reshape(1, 60, 1), verbose=0)[0][0]
        return jsonify({"predicted_price": float(lstm_scaler.inverse_transform([[pred_norm]])[0][0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict_xgb", methods=["POST"])
def predict_xgb():
    try:
        _, _, xgb_model, xgb_scaler, _, _ = load_models("bitcoin")
        if xgb_model is None:
            return jsonify({"error": "Bitcoin models not found"}), 404
        prices = fetch_recent_prices("bitcoin", days=365)
        if len(prices) < 7:
            return jsonify({"error": "Not enough data to predict"}), 400
        xgb_input = prepare_xgb_input(prices)
        xgb_scaled = xgb_scaler.transform(xgb_input)
        return jsonify({"predicted_price": float(xgb_model.predict(xgb_scaled)[0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict_with_history", methods=["POST"])
def predict_with_history():
    try:
        req = request.json
        model_type = req.get("model")
        lstm_model, lstm_scaler, xgb_model, xgb_scaler, _, _ = load_models("bitcoin")
        if lstm_model is None or xgb_model is None:
            return jsonify({"error": "Bitcoin models not found"}), 404
        prices = fetch_recent_prices("bitcoin", days=365)
        
        if model_type == "lstm":
            lstm_input = prepare_lstm_input(prices)
            lstm_scaled = lstm_scaler.transform(lstm_input).flatten()
            pred_norm = lstm_model.predict(np.array(lstm_scaled).reshape(1, 60, 1), verbose=0)[0][0]
            pred_price = lstm_scaler.inverse_transform([[pred_norm]])[0][0]
        elif model_type == "xgboost":
            xgb_input = prepare_xgb_input(prices)
            xgb_scaled = xgb_scaler.transform(xgb_input)
            pred_price = xgb_model.predict(xgb_scaled)[0]
        else:
            return jsonify({"error": "Invalid model"}), 400
        
        return jsonify({"history": prices, "predicted_price": float(pred_price)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _live_price_worker():
    """Background task that streams live prices over WebSocket."""
    while True:
        try:
            payload = fetch_simple_prices(_LIVE_COINS)
            if payload["coins"]:
                socketio.emit("price_update", payload)
        except requests.exceptions.RequestException as exc:
            socketio.emit("price_error", {"message": f"Live feed error: {exc}"})
        except Exception as exc:  # pragma: no cover - defensive
            socketio.emit("price_error", {"message": f"Unexpected live feed error: {exc}"})
        finally:
            socketio.sleep(10)


def _ensure_live_thread():
    global _live_thread
    with _live_thread_lock:
        if _live_thread is None:
            _live_thread = socketio.start_background_task(_live_price_worker)


@socketio.on("connect")
def handle_connect():
    _ensure_live_thread()


@socketio.on("subscribe_live")
def handle_subscribe_live(data):
    coins = data.get("coins") if isinstance(data, dict) else None
    response = {
        "coins": coins if coins else _LIVE_COINS,
        "timestamp": int(time.time())
    }
    socketio.emit("subscription_ack", response, to=request.sid)


@socketio.on("disconnect")
def handle_disconnect():
    # Nothing to clean up per-connection yet.
    pass


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=port, debug=False)
