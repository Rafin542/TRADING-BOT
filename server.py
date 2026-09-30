"""
🧠 Market Prediction AI — Web Dashboard Server

FastAPI-based web server with a beautiful frontend dashboard.
Run: python server.py
Open: http://localhost:8000
"""

import os
import sys
import json
import asyncio
import numpy as np
from datetime import datetime
from typing import Optional

# Fix Windows Unicode/emoji encoding issues
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CONFIG

app = FastAPI(title="🧠 Market Prediction AI", version="1.0.0")


# ═══════════════════════════════════════════════════════════════
# STATE — Global state for caching
# ═══════════════════════════════════════════════════════════════

class AppState:
    """Application state for caching data and models."""
    def __init__(self):
        self.fetcher = None
        self.feature_engine = None
        self.ensemble = None
        self.preprocessor = None
        self.brain = None  # 3-Layer MasterBrain
        self.models_loaded = False
        self.last_prediction = None
        self.prediction_history = []
        self.replay_engine = None

state = AppState()


def initialize_components():
    """Initialize data fetcher and feature engine (lazy loading)."""
    if state.fetcher is None:
        from data.fetcher import DataFetcher
        state.fetcher = DataFetcher(CONFIG)
        print("✅ DataFetcher initialized")

    if state.feature_engine is None:
        from features.feature_engine import FeatureEngine
        state.feature_engine = FeatureEngine(CONFIG)
        print("✅ FeatureEngine initialized")

    if state.preprocessor is None:
        from data.preprocessor import DataPreprocessor
        state.preprocessor = DataPreprocessor(CONFIG)
        print("✅ DataPreprocessor initialized")


def load_models_if_needed():
    """Load trained models if available."""
    if not state.models_loaded:
        meta_path = os.path.join(CONFIG.model_save_dir, 'meta.json')
        if os.path.exists(meta_path):
            from models.ensemble import MasterEnsemble
            state.ensemble = MasterEnsemble(CONFIG)
            state.ensemble.load_all()
            state.models_loaded = True
            print("✅ Models loaded")
        else:
            print("⚠️  No trained models found. Train first using: python main.py train")


# ═══════════════════════════════════════════════════════════════
# API ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the main dashboard HTML."""
    html_path = os.path.join(os.path.dirname(__file__), "templates", "dashboard.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/api/market-data")
async def get_market_data(symbol: str = "BTC/USDT", timeframe: str = "4h", limit: int = 200):
    """Fetch OHLCV market data."""
    try:
        symbol = symbol.upper().strip()
        if "USDT" in symbol and "/" not in symbol:
            symbol = symbol.replace("USDT", "/USDT")
            
        initialize_components()
        df = state.fetcher.fetch_ohlcv(symbol, timeframe, limit)

        if df is None or len(df) == 0:
            return JSONResponse({"error": "Failed to fetch data"}, status_code=500)

        # Convert to JSON-friendly format
        data = {
            "symbol": symbol,
            "timeframe": timeframe,
            "candles": len(df),
            "data": []
        }

        for _, row in df.iterrows():
            data["data"].append({
                "timestamp": str(row.get("timestamp", "")),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
            })

        return JSONResponse(data)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/predict")
async def predict(symbol: str = "BTC/USDT", timeframe: str = "4h"):
    """Generate 3-layer market prediction using MasterBrain."""
    try:
        symbol = symbol.upper().strip()
        if "USDT" in symbol and "/" not in symbol:
            symbol = symbol.replace("USDT", "/USDT")
            
        # Initialize brain if needed
        if state.brain is None:
            from brain import MasterBrain
            state.brain = MasterBrain(CONFIG)

        # Run 3-layer prediction
        output = state.brain.predict(symbol, timeframe)
        result = state.brain.to_dict(output)

        if "error" in result["layer1"]["details"]:
            return JSONResponse({"error": result["layer1"]["details"]["error"]}, status_code=400)

        # Store in history
        state.prediction_history.append({
            "timestamp": result["timestamp"],
            "symbol": symbol,
            "direction": result["direction"],
            "confidence": result["final_confidence"],
            "price": result["current_price"],
            "l1_conf": result["layer1"]["confidence"],
            "l2_conf": result["layer2"]["confidence"],
            "l3_conf": result["layer3"]["confidence"],
        })
        if len(state.prediction_history) > 100:
            state.prediction_history = state.prediction_history[-100:]

        return JSONResponse(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


def _generate_feature_based_prediction(featured_df):
    """
    Feature-based heuristic prediction when no trained models available.
    Uses technical indicators to generate a prediction.
    """
    last_row = featured_df.iloc[-1]

    signals = []
    model_preds = {}

    # RSI signal
    rsi = last_row.get("rsi", 50)
    if rsi < 30:
        signals.append(1)  # Oversold = bullish
        model_preds["RSI"] = {"direction": "UP", "probability": 0.7, "confidence": 0.6, "probabilities": {"UP": 0.7, "DOWN": 0.3}}
    elif rsi > 70:
        signals.append(-1)  # Overbought = bearish
        model_preds["RSI"] = {"direction": "DOWN", "probability": 0.7, "confidence": 0.6, "probabilities": {"UP": 0.3, "DOWN": 0.7}}
    else:
        signals.append(0)
        model_preds["RSI"] = {"direction": "UP" if rsi < 50 else "DOWN", "probability": 0.5 + abs(50-rsi)/100, "confidence": 0.4, "probabilities": {"UP": 0.5, "DOWN": 0.5}}

    # MACD signal
    macd_hist = last_row.get("macd_hist", 0)
    if macd_hist > 0:
        signals.append(1)
        model_preds["MACD"] = {"direction": "UP", "probability": 0.6, "confidence": 0.55, "probabilities": {"UP": 0.6, "DOWN": 0.4}}
    else:
        signals.append(-1)
        model_preds["MACD"] = {"direction": "DOWN", "probability": 0.6, "confidence": 0.55, "probabilities": {"UP": 0.4, "DOWN": 0.6}}

    # EMA crossover
    ema_8 = last_row.get("ema_8", 0)
    ema_21 = last_row.get("ema_21", 0)
    if ema_8 > 0 and ema_21 > 0:
        if ema_8 > ema_21:
            signals.append(1)
            model_preds["EMA_Cross"] = {"direction": "UP", "probability": 0.65, "confidence": 0.5, "probabilities": {"UP": 0.65, "DOWN": 0.35}}
        else:
            signals.append(-1)
            model_preds["EMA_Cross"] = {"direction": "DOWN", "probability": 0.65, "confidence": 0.5, "probabilities": {"UP": 0.35, "DOWN": 0.65}}

    # Bollinger Band position
    bb_pct = last_row.get("bb_percent_b", 0.5)
    if bb_pct < 0.2:
        signals.append(1)
        model_preds["Bollinger"] = {"direction": "UP", "probability": 0.6, "confidence": 0.5, "probabilities": {"UP": 0.6, "DOWN": 0.4}}
    elif bb_pct > 0.8:
        signals.append(-1)
        model_preds["Bollinger"] = {"direction": "DOWN", "probability": 0.6, "confidence": 0.5, "probabilities": {"UP": 0.4, "DOWN": 0.6}}
    else:
        signals.append(0)
        model_preds["Bollinger"] = {"direction": "UP", "probability": 0.5, "confidence": 0.3, "probabilities": {"UP": 0.5, "DOWN": 0.5}}

    # Volume trend
    rel_vol = last_row.get("relative_volume", 1.0)
    vol_signal = 1 if rel_vol > 1.2 else (-1 if rel_vol < 0.8 else 0)
    model_preds["Volume"] = {"direction": "UP" if vol_signal >= 0 else "DOWN", "probability": 0.55, "confidence": 0.4, "probabilities": {"UP": 0.55, "DOWN": 0.45}}

    # ADX trend strength
    adx = last_row.get("adx", 20)
    model_preds["ADX_Trend"] = {"direction": "UP" if sum(signals) > 0 else "DOWN", "probability": min(0.5 + adx/200, 0.8), "confidence": min(adx/50, 0.8), "probabilities": {"UP": 0.5, "DOWN": 0.5}}

    # Aggregate
    avg_signal = np.mean(signals) if signals else 0
    up_prob = 0.5 + avg_signal * 0.2
    up_prob = max(0.1, min(0.9, up_prob))

    direction = "UP" if up_prob > 0.5 else "DOWN"
    up_votes = sum(1 for s in signals if s > 0)
    agreement = max(up_votes, len(signals) - up_votes) / max(len(signals), 1)

    return {
        "direction": direction,
        "up_probability": float(up_prob),
        "down_probability": float(1 - up_prob),
        "confidence_score": float(abs(up_prob - 0.5) * 2 * agreement),
        "agreement_ratio": float(agreement),
        "regime": "heuristic_mode",
        "individual_model_predictions": model_preds,
        "risk_level": "MEDIUM",
        "note": "⚠️ Heuristic mode — train models for better accuracy: python main.py train"
    }


def _make_json_safe(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_json_safe(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


@app.get('/api/multi-tf')
async def get_multi_tf(symbol: str = 'BTC/USDT', timeframe: str = '4h'):
    try:
        initialize_components()
        symbol = symbol.replace('USDT', '/USDT') if '/' not in symbol else symbol
        
        if not state.brain:
            from brain import MasterBrain
            state.brain = MasterBrain(CONFIG)
        
        if state.brain.multi_tf_analyzer:
            result = state.brain.multi_tf_analyzer.analyze_all(symbol)
            return _make_json_safe(result)
        else:
            return {'error': 'Multi-timeframe analyzer not available'}
    except Exception as e:
        return {'error': str(e)}


@app.get('/api/patterns')
async def get_patterns(symbol: str = 'BTC/USDT', timeframe: str = '4h'):
    try:
        initialize_components()
        symbol = symbol.replace('USDT', '/USDT') if '/' not in symbol else symbol
        
        df = state.fetcher.fetch_ohlcv(symbol, timeframe, 500)
        if df is None or len(df) < 100:
            return {'error': 'Not enough data for pattern matching'}
        
        if not state.brain:
            from brain import MasterBrain
            state.brain = MasterBrain(CONFIG)
        
        if state.brain.pattern_matcher:
            result = state.brain.pattern_matcher.find_similar_patterns(df)
            return _make_json_safe(result)
        else:
            return {'error': 'Pattern matcher not available'}
    except Exception as e:
        return {'error': str(e)}


@app.get("/api/indicators")
async def get_indicators(symbol: str = "BTC/USDT", timeframe: str = "4h"):
    """Get current indicator values."""
    try:
        initialize_components()
        df = state.fetcher.fetch_ohlcv(symbol, timeframe, 300)

        if df is None or len(df) == 0:
            return JSONResponse({"error": "Failed to fetch data"}, status_code=500)

        featured_df = state.feature_engine.compute_all_features(df)
        featured_df = state.feature_engine.remove_nan_rows(featured_df)

        if len(featured_df) == 0:
            return JSONResponse({"error": "No data after feature computation"}, status_code=400)

        last = featured_df.iloc[-1]

        indicators = {
            "rsi_14": _safe_float(last.get("rsi")),
            "macd": _safe_float(last.get("macd")),
            "macd_signal": _safe_float(last.get("macd_signal")),
            "macd_histogram": _safe_float(last.get("macd_hist")),
            "ema_8": _safe_float(last.get("ema_8")),
            "ema_21": _safe_float(last.get("ema_21")),
            "ema_55": _safe_float(last.get("ema_55")),
            "ema_200": _safe_float(last.get("ema_200")),
            "bb_upper": _safe_float(last.get("bb_upper")),
            
            "bb_lower": _safe_float(last.get("bb_lower")),
            "bb_percent_b": _safe_float(last.get("bb_pct_b")),
            "atr_14": _safe_float(last.get("atr")),
            "adx": _safe_float(last.get("adx")),
            "obv": _safe_float(last.get("obv")),
            "volume": _safe_float(last.get("volume")),
            "close": _safe_float(last.get("close")),
        }

        return JSONResponse({"symbol": symbol, "timeframe": timeframe, "indicators": indicators})

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/history")
async def get_prediction_history():
    """Get prediction history."""
    return JSONResponse({"history": state.prediction_history})


@app.get("/api/symbols")
async def get_symbols():
    """Get supported symbols."""
    return JSONResponse({"symbols": CONFIG.data.supported_symbols, "timeframes": CONFIG.data.timeframes})


def _safe_float(val, default=0.0):
    """Safely convert to float."""
    try:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return default
        return round(float(val), 6)
    except:
        return default


# ═══════════════════════════════════════════════════════════════
# BACKTESTING & FAST MARKET REPLAY ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.get("/api/backtest/markets")
async def get_backtest_markets():
    """List markets available for backtesting & replay."""
    try:
        from backtesting.replay_engine import MarketReplayEngine
        if state.replay_engine is None:
            state.replay_engine = MarketReplayEngine(CONFIG)
        data = state.replay_engine.get_available_markets()
        return JSONResponse(_make_json_safe(data))
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/backtest/run")
@app.get("/api/backtest/run")
async def run_backtest_simulation(
    symbol: str = "BTC/USDT",
    timeframe: str = "4h",
    start_date: Optional[str] = None,
    candles: int = 100,
    min_confidence: float = 0.50,
    risk_reward: float = 1.5,
    request: Request = None
):
    """Run lookahead-free market simulation and return replay timeline."""
    try:
        if request and request.method == "POST":
            try:
                body = await request.json()
                symbol = body.get("symbol", symbol)
                timeframe = body.get("timeframe", timeframe)
                start_date = body.get("start_date", start_date)
                candles = int(body.get("candles", candles))
                min_confidence = float(body.get("min_confidence", min_confidence))
                risk_reward = float(body.get("risk_reward", risk_reward))
            except Exception:
                pass

        symbol = symbol.upper().strip()
        if "USDT" in symbol and "/" not in symbol:
            symbol = symbol.replace("USDT", "/USDT")

        from backtesting.replay_engine import MarketReplayEngine
        if state.replay_engine is None:
            state.replay_engine = MarketReplayEngine(CONFIG)

        result = state.replay_engine.run_simulation(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            num_candles=candles,
            min_confidence=min_confidence,
            risk_reward_ratio=risk_reward
        )

        return JSONResponse(_make_json_safe(result))
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


# ═══════════════════════════════════════════════════════════════
# RUN SERVER
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧠 Market Prediction AI — Dashboard Server")
    print("="*60)
    print("📡 Starting server at: http://localhost:8000")
    print("🌐 Open your browser and go to: http://localhost:8000")
    print("="*60 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
