"""
🧠 Self-Improving Market Prediction AI — Main Entry Point

Usage:
    python main.py train --symbol BTC/USDT --timeframe 4h
    python main.py predict --symbol BTC/USDT --timeframe 4h
    python main.py backtest --symbol BTC/USDT --timeframe 4h
    python main.py evolve --symbol BTC/USDT
"""

import argparse
import sys
import os
import json
import numpy as np
from datetime import datetime

# Fix Windows Unicode/emoji encoding issues
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from config import CONFIG, MasterConfig
from utils.helpers import set_seed, setup_logging, create_directories, format_prediction


logger = setup_logging(CONFIG.log_dir)


# ═══════════════════════════════════════════════════════════════
# SIGNAL DISPLAY — Beautiful prediction output
# ═══════════════════════════════════════════════════════════════

def signal_display(prediction: dict, symbol: str = "BTC/USDT", timeframe: str = "4h"):
    """Beautifully formatted prediction output with emojis and details."""
    direction = prediction.get('direction', 'UNKNOWN')
    up_prob = prediction.get('up_probability', 0.5)
    down_prob = prediction.get('down_probability', 0.5)
    confidence = prediction.get('confidence_score', 0.0)
    agreement = prediction.get('agreement_ratio', 0.0)
    regime = prediction.get('regime', 'unknown')
    risk_level = prediction.get('risk_level', 'UNKNOWN')

    # Direction emoji
    if direction == 'UP':
        dir_emoji = "🟩 ▲ UP"
        dir_color = "BULLISH"
    elif direction == 'DOWN':
        dir_emoji = "🟥 ▼ DOWN"
        dir_color = "BEARISH"
    else:
        dir_emoji = "⬜ ◆ NEUTRAL"
        dir_color = "NEUTRAL"

    # Confidence bar
    conf_blocks = int(confidence * 20)
    conf_bar = "█" * conf_blocks + "░" * (20 - conf_blocks)

    print("\n")
    print("╔══════════════════════════════════════════════════════════╗")
    print("║     🧠 MARKET PREDICTION AI — SIGNAL REPORT            ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║  📊 Market:     {symbol:<40} ║")
    print(f"║  ⏱️  Timeframe:  {timeframe:<40} ║")
    print(f"║  🕐 Time:       {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC'):<40} ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║                                                        ║")
    print(f"║  🔮 DIRECTION:   {dir_emoji:<39} ║")
    print(f"║                                                        ║")
    print(f"║  📈 UP Probability:    {up_prob*100:>6.2f}%                        ║")
    print(f"║  📉 DOWN Probability:  {down_prob*100:>6.2f}%                        ║")
    print(f"║  🎯 Confidence:        {confidence*100:>6.2f}%                        ║")
    print(f"║  🤝 Model Agreement:   {agreement*100:>6.2f}%                        ║")
    print(f"║                                                        ║")
    print(f"║  Confidence: [{conf_bar}] {confidence*100:.1f}%       ║")
    print(f"║                                                        ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║  📊 Market Regime:  {regime:<36} ║")
    print(f"║  ⚠️  Risk Level:     {risk_level:<36} ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print("║  🤖 Individual Model Predictions:                      ║")
    print("║  ─────────────────────────────────────────────          ║")

    individual = prediction.get('individual_model_predictions', {})
    for model_name, model_pred in individual.items():
        m_dir = model_pred.get('direction', 'UP')
        m_prob = model_pred.get('probability', 0.5)
        m_emoji = "🟩" if m_dir == "UP" else "🟥"
        m_line = f"{model_name:<14} {m_emoji} {m_dir:<5} ({m_prob*100:.1f}%)"
        print(f"║    {m_line:<52} ║")

    print("╠══════════════════════════════════════════════════════════╣")
    print("║  💡 Summary:                                           ║")

    if confidence >= 0.70:
        print("║  ✅ HIGH CONFIDENCE signal — Strong conviction         ║")
    elif confidence >= 0.60:
        print("║  ⚠️  MODERATE CONFIDENCE — Proceed with caution        ║")
    else:
        print("║  ❌ LOW CONFIDENCE — Consider waiting for better setup  ║")

    print("╚══════════════════════════════════════════════════════════╝")
    print()


# ═══════════════════════════════════════════════════════════════
# TRAIN COMMAND — Full training pipeline
# ═══════════════════════════════════════════════════════════════

def run_train(symbol: str, timeframe: str):
    """Fetch data → compute features → train all models → save."""
    from data.fetcher import DataFetcher
    from data.preprocessor import DataPreprocessor
    from features.feature_engine import FeatureEngine
    from models.ensemble import MasterEnsemble

    logger.info(f"=== TRAINING PIPELINE START: {symbol} @ {timeframe} ===")

    # Step 1: Fetch data
    print(f"\n📥 Step 1/5: Fetching data for {symbol}...")
    fetcher = DataFetcher(CONFIG)
    
    # Ensure the requested timeframe is always fetched
    fetch_tfs = list(set(CONFIG.data.timeframes + [timeframe]))
    data_dict = fetcher.fetch_multi_timeframe(symbol, fetch_tfs)
    primary_df = data_dict.get(timeframe)

    if primary_df is None or len(primary_df) < CONFIG.data.min_candles:
        print(f"❌ Not enough data for {symbol} @ {timeframe}. Got {len(primary_df) if primary_df is not None else 0} candles, need {CONFIG.data.min_candles}.")
        return

    print(f"   ✅ Fetched {len(primary_df)} candles for primary timeframe {timeframe}")
    for tf, df in data_dict.items():
        if df is not None:
            print(f"   ✅ {tf}: {len(df)} candles")

    # Step 2: Compute features
    print(f"\n⚙️  Step 2/5: Computing features...")
    feature_engine = FeatureEngine(CONFIG)
    featured_df = feature_engine.compute_all_features(primary_df, higher_tf_data=data_dict)
    featured_df = feature_engine.remove_nan_rows(featured_df)
    feature_names = feature_engine.get_feature_names()
    print(f"   ✅ Computed {len(feature_names)} features, {len(featured_df)} valid samples")

    # Step 3: Preprocess & create labels
    print(f"\n🔧 Step 3/5: Preprocessing & creating labels...")
    preprocessor = DataPreprocessor(CONFIG)
    featured_df = preprocessor.handle_missing_data(featured_df)
    featured_df = preprocessor.create_labels(featured_df, horizon=1, method='binary')
    featured_df.dropna(subset=['target'], inplace=True)

    # Prepare X, y
    label_col = 'target'
    exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', label_col, 'future_return']
    feature_cols = [c for c in featured_df.columns if c not in exclude_cols]
    feature_cols = [c for c in feature_cols if featured_df[c].dtype in ['float64', 'float32', 'int64', 'int32']]

    X = featured_df[feature_cols].values.astype(np.float32)
    y = featured_df[label_col].values.astype(np.int64)

    # Handle NaN/Inf in features
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"   ✅ X shape: {X.shape}, y shape: {y.shape}")
    print(f"   ✅ Label distribution: UP={np.sum(y==1)}, DOWN={np.sum(y==0)}")

    # Step 4: Walk-forward split
    print(f"\n📊 Step 4/5: Walk-forward data splitting...")
    splits = list(preprocessor.walk_forward_split(
        featured_df, 
        train_window=CONFIG.training.train_window,
        val_window=CONFIG.training.val_window,
        test_window=CONFIG.training.test_window,
        step=CONFIG.training.step_size
    ))

    if len(splits) == 0:
        print("   ⚠️  Not enough data for walk-forward splitting. Using simple 70/15/15 split...")
        n = len(X)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)

        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]
        
        train_df = featured_df.iloc[:train_end]
        val_df = featured_df.iloc[train_end:val_end]
        test_df = featured_df.iloc[val_end:]
    else:
        # Use the last split for training
        last_split = splits[-1]
        train_df, val_df, test_df = last_split

        X_train = train_df[feature_cols].values.astype(np.float32)
        y_train = train_df[label_col].values.astype(np.int64)
        X_val = val_df[feature_cols].values.astype(np.float32)
        y_val = val_df[label_col].values.astype(np.int64)
        X_test = test_df[feature_cols].values.astype(np.float32)
        y_test = test_df[label_col].values.astype(np.int64)

        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"   ✅ Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Step 5: Train ensemble
    print(f"\n🧠 Step 5/5: Training Master Ensemble (all models)...")
    ensemble = MasterEnsemble(CONFIG)

    # Returns data for regime detection
    close_prices = featured_df['close'].values
    returns_data = np.diff(np.log(close_prices + 1e-10))

    # Create sequences for deep learning models (they need 3D input)
    seq_len = CONFIG.transformer.sequence_length
    train_seq_df = train_df[feature_cols + [label_col]]
    X_train_seq, y_train_seq = preprocessor.create_sequences(train_seq_df, seq_len, target_col=label_col)
    val_seq_df = val_df[feature_cols + [label_col]]
    if len(val_seq_df) > seq_len:
        X_val_seq, y_val_seq = preprocessor.create_sequences(val_seq_df, seq_len, target_col=label_col)
    else:
        X_val_seq, y_val_seq = X_val, y_val

    ensemble.fit_all(
        X_train_seq, y_train_seq,
        X_val_seq, y_val_seq,
        returns_data=returns_data
    )

    # Save models
    ensemble.save_all()

    # Save feature column names for prediction time
    meta = {
        'feature_cols': feature_cols,
        'symbol': symbol,
        'timeframe': timeframe,
        'trained_at': datetime.utcnow().isoformat(),
        'num_features': len(feature_cols),
        'train_samples': len(X_train),
    }
    meta_path = os.path.join(CONFIG.model_save_dir, 'meta.json')
    os.makedirs(CONFIG.model_save_dir, exist_ok=True)
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅ Training complete! Models saved to {CONFIG.model_save_dir}/")
    print(f"   📋 {len(feature_cols)} features, {len(X_train)} training samples")

    # Quick evaluation on test set
    print(f"\n📈 Quick Test Evaluation:")
    test_seq_df = test_df[feature_cols + [label_col]]
    X_test_seq, y_test_seq = preprocessor.create_sequences(test_seq_df, seq_len, target_col=label_col)

    if len(X_test_seq) > 0:
        correct = 0
        total = 0
        for i in range(min(len(X_test_seq), 50)):
            pred = ensemble.predict(X_test_seq[i:i+1])
            predicted_dir = 1 if pred['direction'] == 'UP' else 0
            actual = y_test_seq[i]
            if predicted_dir == actual:
                correct += 1
            total += 1

        accuracy = correct / total if total > 0 else 0
        print(f"   🎯 Test Accuracy (first 50 samples): {accuracy*100:.1f}%")

    logger.info("=== TRAINING PIPELINE COMPLETE ===")


# ═══════════════════════════════════════════════════════════════
# PREDICT COMMAND — Generate live prediction
# ═══════════════════════════════════════════════════════════════

def run_predict(symbol: str, timeframe: str):
    """Load models → fetch latest data → generate prediction signal."""
    from data.fetcher import DataFetcher
    from data.preprocessor import DataPreprocessor
    from features.feature_engine import FeatureEngine
    from models.ensemble import MasterEnsemble
    from calibration.probability_calibrator import ProbabilityCalibrator
    from risk.manager import RiskManager

    logger.info(f"=== PREDICTION: {symbol} @ {timeframe} ===")

    # Load meta
    meta_path = os.path.join(CONFIG.model_save_dir, 'meta.json')
    if not os.path.exists(meta_path):
        print("❌ No trained models found. Run 'train' first.")
        return

    with open(meta_path, 'r') as f:
        meta = json.load(f)

    feature_cols = meta['feature_cols']

    # Fetch latest data
    print(f"📥 Fetching latest data for {symbol}...")
    fetcher = DataFetcher(CONFIG)
    data_dict = fetcher.fetch_multi_timeframe(symbol, CONFIG.data.timeframes)
    primary_df = data_dict.get(timeframe)

    if primary_df is None or len(primary_df) < CONFIG.data.min_candles:
        print(f"❌ Not enough data. Got {len(primary_df) if primary_df is not None else 0} candles.")
        return

    # Compute features
    print("⚙️  Computing features...")
    feature_engine = FeatureEngine(CONFIG)
    featured_df = feature_engine.compute_all_features(primary_df, higher_tf_data=data_dict)
    featured_df = feature_engine.remove_nan_rows(featured_df)

    # Ensure we have the same feature columns
    available_cols = [c for c in feature_cols if c in featured_df.columns]
    missing_cols = [c for c in feature_cols if c not in featured_df.columns]
    if missing_cols:
        print(f"   ⚠️  {len(missing_cols)} features missing, filling with 0")
        for col in missing_cols:
            featured_df[col] = 0.0

    X = featured_df[feature_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Create sequence for latest data point
    preprocessor = DataPreprocessor(CONFIG)
    seq_len = CONFIG.transformer.sequence_length
    df_for_seq = featured_df[feature_cols].copy()
    df_for_seq['dummy_target'] = 0
    X_seq, _ = preprocessor.create_sequences(df_for_seq, seq_len, target_col='dummy_target')

    if len(X_seq) == 0:
        print("❌ Not enough data for sequence creation.")
        return

    # Load ensemble
    print("🧠 Loading models...")
    ensemble = MasterEnsemble(CONFIG)
    ensemble.load_all()

    # Get latest data point prediction
    latest_X = X_seq[-1:]  # Last sequence

    # Returns for regime detection
    close_prices = featured_df['close'].values
    returns_data = np.diff(np.log(close_prices + 1e-10))

    # Predict
    print("🔮 Generating prediction...")
    prediction = ensemble.predict(latest_X, current_returns_data=returns_data)

    # Risk assessment
    risk_manager = RiskManager()
    should_trade = risk_manager.should_trade(prediction)
    prediction['should_trade'] = should_trade

    # Display
    signal_display(prediction, symbol=symbol, timeframe=timeframe)

    if not should_trade:
        print("⚠️  Risk Manager says: DO NOT TRADE this signal (low confidence or high risk)")

    logger.info(f"Prediction: {prediction['direction']}, Confidence: {prediction['confidence_score']:.2%}")


# ═══════════════════════════════════════════════════════════════
# BACKTEST COMMAND — Walk-forward backtesting
# ═══════════════════════════════════════════════════════════════

def run_backtest(symbol: str, timeframe: str):
    """Run walk-forward backtest and display comprehensive metrics."""
    from data.fetcher import DataFetcher
    from data.preprocessor import DataPreprocessor
    from features.feature_engine import FeatureEngine
    from models.ensemble import MasterEnsemble
    from backtesting.metrics import MetricsCalculator

    logger.info(f"=== BACKTEST: {symbol} @ {timeframe} ===")

    # Fetch data
    print(f"📥 Fetching historical data for {symbol}...")
    fetcher = DataFetcher(CONFIG)
    data_dict = fetcher.fetch_multi_timeframe(symbol, CONFIG.data.timeframes)
    primary_df = data_dict.get(timeframe)

    if primary_df is None or len(primary_df) < CONFIG.data.min_candles:
        print(f"❌ Not enough data for backtesting.")
        return

    # Compute features
    print("⚙️  Computing features...")
    feature_engine = FeatureEngine(CONFIG)
    featured_df = feature_engine.compute_all_features(primary_df, higher_tf_data=data_dict)
    featured_df = feature_engine.remove_nan_rows(featured_df)

    # Prepare data
    preprocessor = DataPreprocessor(CONFIG)
    featured_df = preprocessor.handle_missing_data(featured_df)
    featured_df = preprocessor.create_labels(featured_df, horizon=1, method='binary')

    label_col = 'target'
    exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', label_col, 'future_return']
    feature_cols = [c for c in featured_df.columns if c not in exclude_cols]
    feature_cols = [c for c in feature_cols if featured_df[c].dtype in ['float64', 'float32', 'int64', 'int32']]

    X = featured_df[feature_cols].values.astype(np.float32)
    y = featured_df[label_col].values.astype(np.int64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Walk-forward backtest
    print("\n📊 Running walk-forward backtest...")
    splits = list(preprocessor.walk_forward_split(
        featured_df,
        train_window=CONFIG.training.train_window,
        val_window=CONFIG.training.val_window,
        test_window=CONFIG.training.test_window,
        step=CONFIG.training.step_size
    ))

    if len(splits) == 0:
        print("⚠️  Not enough data for walk-forward splits. Using 70/30 split...")
        n = len(X)
        train_end = int(n * 0.70)
        splits = [(featured_df.iloc[:train_end], featured_df.iloc[train_end:train_end+50], featured_df.iloc[train_end+50:])]

    all_predictions = []
    all_actuals = []
    all_probabilities = []
    seq_len = CONFIG.transformer.sequence_length

    for fold_idx, (train_df, val_df, test_df) in enumerate(splits):
        print(f"\n  📂 Fold {fold_idx+1}/{len(splits)}: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

        X_train = train_df[feature_cols].values.astype(np.float32)
        y_train = train_df[label_col].values.astype(np.int64)
        X_val = val_df[feature_cols].values.astype(np.float32)
        y_val = val_df[label_col].values.astype(np.int64)
        X_test = test_df[feature_cols].values.astype(np.float32)
        y_test = test_df[label_col].values.astype(np.int64)

        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

        # Create sequences
        train_seq_df = train_df[feature_cols + [label_col]]
        X_train_seq, y_train_seq = preprocessor.create_sequences(train_seq_df, seq_len, target_col=label_col)
        val_seq_df = val_df[feature_cols + [label_col]]
        if len(val_seq_df) > seq_len:
            X_val_seq, y_val_seq = preprocessor.create_sequences(val_seq_df, seq_len, target_col=label_col)
        else:
            X_val_seq, y_val_seq = X_val, y_val

        # Train
        ensemble = MasterEnsemble(CONFIG)
        returns_data = np.diff(np.log(train_df['close'].values + 1e-10))

        try:
            ensemble.fit_all(X_train_seq, y_train_seq, X_val_seq, y_val_seq, returns_data=returns_data)
        except Exception as e:
            print(f"    ⚠️  Training error in fold {fold_idx+1}: {e}")
            continue

        # Test
        test_seq_df = test_df[feature_cols + [label_col]]
        if len(test_seq_df) > seq_len:
            X_test_seq, y_test_seq = preprocessor.create_sequences(test_seq_df, seq_len, target_col=label_col)
        else:
            X_test_seq, y_test_seq = [], []

        for i in range(len(X_test_seq)):
            try:
                pred = ensemble.predict(X_test_seq[i:i+1])
                pred_dir = 1 if pred['direction'] == 'UP' else 0
                all_predictions.append(pred_dir)
                all_actuals.append(y_test_seq[i])
                all_probabilities.append(pred['up_probability'])
            except Exception as e:
                continue

    # Calculate metrics
    if len(all_predictions) == 0:
        print("\n❌ No predictions were generated during backtest.")
        return

    print(f"\n{'='*60}")
    print(f"📊 BACKTEST RESULTS — {symbol} @ {timeframe}")
    print(f"{'='*60}")
    print(f"Total predictions: {len(all_predictions)}")

    metrics_calc = MetricsCalculator()
    metrics = metrics_calc.compute_all(
        np.array(all_predictions),
        np.array(all_actuals),
        np.array(all_probabilities)
    )

    metrics_calc.print_report(metrics)

    logger.info(f"Backtest complete: {len(all_predictions)} predictions, Accuracy={metrics.get('accuracy', 0):.2%}")


# ═══════════════════════════════════════════════════════════════
# EVOLVE COMMAND — Genetic programming feature discovery
# ═══════════════════════════════════════════════════════════════

def run_evolve(symbol: str, timeframe: str):
    """Run genetic programming to discover new features/indicators."""
    from data.fetcher import DataFetcher
    from data.preprocessor import DataPreprocessor
    from features.feature_engine import FeatureEngine
    from self_improvement.genetic_features import GeneticFeatureEvolver

    logger.info(f"=== GENETIC EVOLUTION: {symbol} @ {timeframe} ===")

    # Fetch and prepare data
    print(f"📥 Fetching data for {symbol}...")
    fetcher = DataFetcher(CONFIG)
    primary_df = fetcher.fetch_ohlcv(symbol, timeframe, CONFIG.data.history_length)

    if primary_df is None or len(primary_df) < CONFIG.data.min_candles:
        print("❌ Not enough data.")
        return

    # Create target (future returns)
    preprocessor = DataPreprocessor(CONFIG)
    primary_df = preprocessor.create_labels(primary_df, horizon=1, method='binary')

    # Prepare data for GP
    target = primary_df['target'].values[:-1]  # Remove last row (no future data)
    data = primary_df[['open', 'high', 'low', 'close', 'volume']].values[:-1]

    # Run evolution
    print(f"\n🧬 Starting Genetic Programming Evolution...")
    print(f"   Population: {CONFIG.self_improvement.gp_population_size}")
    print(f"   Generations: {CONFIG.self_improvement.gp_generations}")
    print(f"   Max tree depth: {CONFIG.self_improvement.gp_max_tree_depth}")

    evolver = GeneticFeatureEvolver()
    evolver.evolve(primary_df, target_col="target")

    # Get best features
    best_features = evolver.get_best_features(top_k=10)

    print(f"\n🏆 Top {len(best_features)} Discovered Features:")
    print("-" * 50)
    for i, feature in enumerate(best_features):
        print(f"  {i+1}. Formula: {feature}")

    # Save evolved features
    save_path = os.path.join(CONFIG.model_save_dir, 'evolved_features.json')
    os.makedirs(CONFIG.model_save_dir, exist_ok=True)
    evolver.save_formulas(save_path)
    print(f"\n✅ Evolved features saved to {save_path}")

    logger.info("=== GENETIC EVOLUTION COMPLETE ===")


# ═══════════════════════════════════════════════════════════════
# REPLAY COMMAND — Fast Historical Market Replay
# ═══════════════════════════════════════════════════════════════

def run_replay(symbol: str, timeframe: str, start_date: str = None, candles: int = 100, speed: float = 0.2, min_confidence: float = 0.50):
    """Replay historical market candle-by-candle with the current algorithm."""
    import time
    from backtesting.replay_engine import MarketReplayEngine

    print(f"\n{'='*65}")
    print(f"⏪ FAST MARKET REPLAY BACKTEST: {symbol} @ {timeframe}")
    print(f"{'='*65}")
    print(f"⏱️  Speed: {speed}s per candle | Candles: {candles} | Min Conf: {min_confidence*100:.0f}%")
    print(f"⏳ Loading historical data and simulating algorithm...")

    engine = MarketReplayEngine(CONFIG)
    simulation = engine.run_simulation(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        num_candles=candles,
        min_confidence=min_confidence
    )

    frames = simulation["frames"]
    if not frames:
        print("❌ No frames generated for replay.")
        return

    print(f"✅ Simulation ready: {len(frames)} candles from {simulation['start_time']} to {simulation['end_time']}")
    print(f"{'-'*65}")
    print(f"Step | Time                 | Price      | Signal | Conf  | Event / Active Trade           | WinRate")
    print(f"{'-'*65}")

    for f in frames:
        c = f["candle"]
        p = f["prediction"]
        m = f["metrics"]
        act = f["active_trade"]
        evt = f["trade_event"]

        dir_sym = "▲ UP" if p["direction"] == "UP" else ("▼ DN" if p["direction"] == "DOWN" else "  --")
        conf_str = f"{p['confidence']*100:.1f}%"

        event_str = "Idle"
        if evt:
            if evt["type"] == "ENTRY":
                event_str = f"🚀 ENTRY {evt['trade']['side']} @ {evt['trade']['entry_price']}"
            elif evt["type"] == "EXIT":
                res_sym = "✅" if evt['trade']['result'] == "WIN" else "❌"
                event_str = f"{res_sym} EXIT {evt['trade']['result']} {evt['trade']['pnl_pct']:+.2f}%"
        elif act:
            event_str = f"Hold {act['side']} ({act['current_pnl_pct']:+.2f}%)"

        wr_str = f"WR: {m['win_rate']:.1f}% ({m['wins']}W/{m['losses']}L)"
        print(f"{f['step']+1:>4} | {c['timestamp'][:19]} | ${c['close']:>9.2f} | {dir_sym} | {conf_str:>5} | {event_str:<30} | {wr_str}")

        if speed > 0:
            time.sleep(speed)

    print(f"{'='*65}")
    print(f"📊 FINAL BACKTEST REPLAY SUMMARY — {symbol} @ {timeframe}")
    print(f"{'='*65}")
    s = simulation["summary"]
    print(f"Total Trades Taken:    {s['total_trades']}")
    print(f"Wins / Losses:         {s['wins']} Wins / {s['losses']} Losses")
    print(f"Final Win Rate:        {s['win_rate']:.1f}%")
    print(f"Net Cumulative PnL:    {s['net_pnl_pct']:+.2f}%")
    print(f"Directional Accuracy:  {s['directional_accuracy']:.1f}%")
    print(f"{'='*65}\n")


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="🧠 Self-Improving Market Prediction AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py train --symbol BTC/USDT --timeframe 4h
  python main.py predict --symbol ETH/USDT --timeframe 1h
  python main.py backtest --symbol BTC/USDT --timeframe 4h
  python main.py replay --symbol BTC/USDT --timeframe 4h --candles 100 --speed 0.2
  python main.py evolve --symbol BTC/USDT --timeframe 4h
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train
    train_parser = subparsers.add_parser("train", help="Train all models on historical data")
    train_parser.add_argument("--symbol", type=str, default="BTC/USDT", help="Trading pair (e.g., BTC/USDT)")
    train_parser.add_argument("--timeframe", type=str, default="4h", help="Primary timeframe (e.g., 1h, 4h, 1d)")

    # Predict
    predict_parser = subparsers.add_parser("predict", help="Generate latest market prediction")
    predict_parser.add_argument("--symbol", type=str, default="BTC/USDT")
    predict_parser.add_argument("--timeframe", type=str, default="4h")

    # Backtest
    bt_parser = subparsers.add_parser("backtest", help="Run walk-forward backtest")
    bt_parser.add_argument("--symbol", type=str, default="BTC/USDT")
    bt_parser.add_argument("--timeframe", type=str, default="4h")

    # Replay
    replay_parser = subparsers.add_parser("replay", help="Fast historical candle-by-candle replay backtest")
    replay_parser.add_argument("--symbol", type=str, default="BTC/USDT")
    replay_parser.add_argument("--timeframe", type=str, default="4h")
    replay_parser.add_argument("--start", type=str, default=None, help="Start date YYYY-MM-DD")
    replay_parser.add_argument("--candles", type=int, default=100, help="Number of replay candles")
    replay_parser.add_argument("--speed", type=float, default=0.2, help="Seconds per candle (e.g. 0.1, 0.2, 1.0)")
    replay_parser.add_argument("--min-conf", type=float, default=0.50, help="Minimum confidence threshold to open trade")

    # Evolve
    evolve_parser = subparsers.add_parser("evolve", help="Run genetic programming to discover new features")
    evolve_parser.add_argument("--symbol", type=str, default="BTC/USDT")
    evolve_parser.add_argument("--timeframe", type=str, default="4h")

    args = parser.parse_args()

    # Initialize
    set_seed(CONFIG.training.seed)
    create_directories(CONFIG)

    if args.command == "train":
        run_train(args.symbol, args.timeframe)
    elif args.command == "predict":
        run_predict(args.symbol, args.timeframe)
    elif args.command == "backtest":
        run_backtest(args.symbol, args.timeframe)
    elif args.command == "replay":
        run_replay(args.symbol, args.timeframe, args.start, args.candles, args.speed, args.min_conf)
    elif args.command == "evolve":
        run_evolve(args.symbol, args.timeframe)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
