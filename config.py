"""
🧠 Self-Improving Market Prediction AI — Master Configuration

সমস্ত hyperparameters, settings, API configs এখানে কেন্দ্রীভূত।
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import os


# ═══════════════════════════════════════════════════════════════
# DATA CONFIGURATION
# ═══════════════════════════════════════════════════════════════

@dataclass
class DataConfig:
    """ডেটা সংগ্রহ ও প্রক্রিয়াকরণের সেটিংস"""

    # Exchange settings
    exchange_id: str = "kucoin"
    api_key: str = os.getenv("EXCHANGE_API_KEY", "")
    api_secret: str = os.getenv("EXCHANGE_API_SECRET", "")

    # Default market
    default_symbol: str = "BTC/USDT"

    # Supported markets (extensible)
    supported_symbols: List[str] = field(default_factory=lambda: [
        "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT",
        "XRP/USDT", "ADA/USDT", "DOGE/USDT", "AVAX/USDT",
    ])

    # Timeframes for multi-timeframe analysis
    timeframes: List[str] = field(default_factory=lambda: [
        "5m", "15m", "1h", "4h", "1d"
    ])

    # Primary timeframe for prediction
    primary_timeframe: str = "4h"

    # Historical candles to fetch per timeframe
    history_length: int = 2000

    # Data cache directory
    cache_dir: str = "data_cache"

    # Minimum required candles for feature computation
    min_candles: int = 500


# ═══════════════════════════════════════════════════════════════
# FEATURE ENGINEERING CONFIGURATION
# ═══════════════════════════════════════════════════════════════

@dataclass
class FeatureConfig:
    """ফিচার ইঞ্জিনিয়ারিং সেটিংস"""

    # EMA periods
    ema_periods: List[int] = field(default_factory=lambda: [
        8, 13, 21, 34, 55, 89, 144, 200
    ])

    # SMA periods
    sma_periods: List[int] = field(default_factory=lambda: [
        10, 20, 50, 100, 200
    ])

    # RSI period
    rsi_period: int = 14

    # MACD settings
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # Bollinger Bands
    bb_period: int = 20
    bb_std: float = 2.0

    # ATR period
    atr_period: int = 14

    # Stochastic RSI
    stoch_rsi_period: int = 14
    stoch_k_period: int = 3
    stoch_d_period: int = 3

    # ADX period
    adx_period: int = 14

    # Ichimoku
    ichimoku_tenkan: int = 9
    ichimoku_kijun: int = 26
    ichimoku_senkou_b: int = 52

    # Volume profile bins
    volume_profile_bins: int = 50

    # Lookback periods for statistical features
    stat_lookbacks: List[int] = field(default_factory=lambda: [
        10, 20, 50, 100
    ])

    # Swing detection sensitivity
    swing_lookback: int = 5

    # Support/Resistance detection
    sr_lookback: int = 100
    sr_sensitivity: float = 0.02  # 2% price zone


# ═══════════════════════════════════════════════════════════════
# MODEL CONFIGURATION
# ═══════════════════════════════════════════════════════════════

@dataclass
class TransformerConfig:
    """Temporal Fusion Transformer সেটিংস"""
    d_model: int = 64
    n_heads: int = 4
    n_encoder_layers: int = 2
    n_decoder_layers: int = 1
    d_ff: int = 128
    dropout: float = 0.1
    sequence_length: int = 60
    prediction_horizon: int = 1  # candles ahead


@dataclass
class LSTMConfig:
    """BiLSTM + Attention সেটিংস"""
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = True
    attention_heads: int = 4
    sequence_length: int = 60


@dataclass
class WaveNetConfig:
    """WaveNet সেটিংস"""
    residual_channels: int = 32
    skip_channels: int = 32
    n_layers: int = 6  # dilation: 1, 2, 4, ..., 32
    kernel_size: int = 2
    sequence_length: int = 60


@dataclass
class CNNConfig:
    """CNN Pattern Recognition সেটিংস"""
    image_size: int = 64  # GAF image size
    n_candles_window: int = 60
    channels: List[int] = field(default_factory=lambda: [32, 64, 128, 256])


@dataclass
class XGBoostConfig:
    """XGBoost/LightGBM সেটিংস"""
    n_estimators: int = 1000
    max_depth: int = 8
    learning_rate: float = 0.05
    num_leaves: int = 63
    feature_fraction: float = 0.7
    bagging_fraction: float = 0.7
    reg_alpha: float = 0.1
    reg_lambda: float = 0.1
    early_stopping_rounds: int = 50
    use_lightgbm: bool = True  # True=LightGBM, False=XGBoost


@dataclass
class RegimeConfig:
    """HMM Regime Detection সেটিংস"""
    n_regimes: int = 5
    covariance_type: str = "full"
    n_iter: int = 200
    lookback: int = 252  # ~1 year daily


# ═══════════════════════════════════════════════════════════════
# TRAINING CONFIGURATION
# ═══════════════════════════════════════════════════════════════

@dataclass
class TrainingConfig:
    """ট্রেনিং সেটিংস"""
    batch_size: int = 64
    max_epochs: int = 30
    early_stopping_patience: int = 7
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    max_lr: float = 1e-3  # OneCycleLR
    gradient_clip: float = 1.0

    # Walk-forward validation
    train_window: int = 1500  # candles for training
    val_window: int = 300     # candles for validation
    test_window: int = 200    # candles for testing
    step_size: int = 100      # walk-forward step

    # Class balancing
    use_focal_loss: bool = True
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0

    # Device
    device: str = "cpu"  # GPU নেই, CPU ব্যবহার হবে
    num_workers: int = 4

    # Seed for reproducibility
    seed: int = 42


# ═══════════════════════════════════════════════════════════════
# ENSEMBLE CONFIGURATION
# ═══════════════════════════════════════════════════════════════

@dataclass
class EnsembleConfig:
    """Ensemble Meta-Learner সেটিংস"""

    # Model weights (default, adjusted by RL agent)
    default_weights: Dict[str, float] = field(default_factory=lambda: {
        "transformer": 0.20,
        "lstm": 0.15,
        "wavenet": 0.10,
        "cnn": 0.10,
        "xgboost": 0.20,
        "regime_hmm": 0.10,
        "dtw": 0.10,
        "bayesian": 0.05,
    })

    # Minimum models that must agree for a signal
    min_agreement: float = 0.55

    # Confidence threshold for signal generation
    min_confidence: float = 0.60


# ═══════════════════════════════════════════════════════════════
# SELF-IMPROVEMENT CONFIGURATION
# ═══════════════════════════════════════════════════════════════

@dataclass
class SelfImprovementConfig:
    """Self-Improvement Engine সেটিংস"""

    # Genetic Programming
    gp_population_size: int = 300
    gp_generations: int = 50
    gp_crossover_prob: float = 0.7
    gp_mutation_prob: float = 0.2
    gp_tournament_size: int = 5
    gp_max_tree_depth: int = 6

    # RL Weight Optimizer (PPO)
    rl_total_timesteps: int = 100_000
    rl_learning_rate: float = 3e-4
    rl_n_steps: int = 2048
    rl_batch_size: int = 64
    rl_n_epochs: int = 10
    rl_gamma: float = 0.99
    rl_clip_range: float = 0.2

    # Self-Reflection
    reflection_window: int = 50  # analyze last N predictions
    reflection_trigger: float = 0.50  # trigger if accuracy drops below

    # Performance tracking
    rolling_accuracy_window: int = 100
    retrain_trigger_accuracy: float = 0.52  # retrain if below this
    concept_drift_threshold: float = 0.05


# ═══════════════════════════════════════════════════════════════
# RISK MANAGEMENT CONFIGURATION
# ═══════════════════════════════════════════════════════════════

@dataclass
class RiskConfig:
    """Risk Management সেটিংস"""
    min_confidence_for_signal: float = 0.65
    min_model_agreement: float = 0.60
    max_signals_per_day: int = 5
    cooldown_after_loss_streak: int = 3
    kelly_fraction: float = 0.25
    max_position_pct: float = 0.05
    daily_loss_limit_pct: float = 0.03
    weekly_loss_limit_pct: float = 0.07


# ═══════════════════════════════════════════════════════════════
# MASTER CONFIGURATION
# ═══════════════════════════════════════════════════════════════

@dataclass
class MasterConfig:
    """সম্পূর্ণ সিস্টেমের মাস্টার কনফিগারেশন"""
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    lstm: LSTMConfig = field(default_factory=LSTMConfig)
    wavenet: WaveNetConfig = field(default_factory=WaveNetConfig)
    cnn: CNNConfig = field(default_factory=CNNConfig)
    xgboost: XGBoostConfig = field(default_factory=XGBoostConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    self_improvement: SelfImprovementConfig = field(default_factory=SelfImprovementConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)

    # Paths
    model_save_dir: str = "saved_models"
    log_dir: str = "logs"
    results_dir: str = "results"


# Global config instance
CONFIG = MasterConfig()
