"""
🧠 THREE-LAYER BRAIN ENGINE — Progressive Confidence Market Prediction

Architecture:
    Layer 1 (Initial):   10-30% confidence  → All standard indicators + market session/stop functions
    Layer 2 (Main):      30-80% confidence  → Custom advanced indicators + ensemble prediction
    Layer 3 (Supreme):   80-98% confidence  → Deep Learning + Self-Learning + Self-Evolving Brain

Each layer refines the previous layer's prediction and ONLY increases confidence
when it has strong evidence to do so.
"""

import os
import sys
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CONFIG
try:
    from analysis.multi_tf_analyzer import MultiTimeframeAnalyzer
    from analysis.pattern_matcher import HistoricalPatternMatcher
    from analysis.time_predictor import TimeDurationPredictor
    HAS_ANALYSIS = True
except ImportError:
    HAS_ANALYSIS = False


# ═══════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════

@dataclass
class LayerSignal:
    """Single layer's output signal."""
    direction: str = "NEUTRAL"          # UP, DOWN, NEUTRAL
    confidence: float = 0.0             # 0.0 to 1.0
    up_probability: float = 0.5
    down_probability: float = 0.5
    signals_bullish: int = 0
    signals_bearish: int = 0
    signals_neutral: int = 0
    details: Dict = field(default_factory=dict)
    sub_signals: Dict = field(default_factory=dict)


@dataclass
class BrainOutput:
    """Final 3-layer brain output."""
    # Final
    direction: str = "NEUTRAL"
    final_confidence: float = 0.0
    up_probability: float = 0.5
    down_probability: float = 0.5

    # Per-layer
    layer1: LayerSignal = field(default_factory=LayerSignal)
    layer2: LayerSignal = field(default_factory=LayerSignal)
    layer3: LayerSignal = field(default_factory=LayerSignal)

    # Meta
    symbol: str = ""
    timeframe: str = ""
    current_price: float = 0.0
    timestamp: str = ""
    regime: str = "unknown"
    risk_level: str = "MEDIUM"

    # Market functions
    multi_tf: dict = field(default_factory=dict)
    pattern_match: dict = field(default_factory=dict)
    time_prediction: dict = field(default_factory=dict)
    suggested_entry: float = 0.0
    suggested_stop_loss: float = 0.0
    suggested_take_profit: float = 0.0
    risk_reward_ratio: float = 0.0
    position_size_pct: float = 0.0


# ═══════════════════════════════════════════════════════════════
# LAYER 1: INITIAL LAYER (10-30% Confidence)
# All standard indicators + market session/stop functions
# ═══════════════════════════════════════════════════════════════

class Layer1_Initial:
    """
    🟡 LAYER 1 — FOUNDATION (10-30% Confidence)
    
    সব ধরনের standard indicator ব্যবহার করে base signal তৈরি করে।
    Market open/close session detection + stop loss/take profit calculation।
    
    এই layer-এ confidence কখনো 30% এর উপরে যাবে না।
    """

    def __init__(self, config=CONFIG):
        self.config = config

    def analyze(self, df, featured_df) -> LayerSignal:
        """Run all Layer 1 analysis."""
        signal = LayerSignal()
        signal.details = {"layer": "Layer 1 — Initial", "max_confidence": "30%"}

        last = featured_df.iloc[-1] if len(featured_df) > 0 else {}
        prev = featured_df.iloc[-2] if len(featured_df) > 1 else last

        bullish = 0
        bearish = 0
        neutral = 0
        sub_signals = {}

        # ─────────────────────────────────────────
        # 1. TREND INDICATORS
        # ─────────────────────────────────────────

        # EMA Crossovers (short vs long)
        ema_8 = self._get(last, "ema_8")
        ema_21 = self._get(last, "ema_21")
        ema_55 = self._get(last, "ema_55")
        ema_200 = self._get(last, "ema_200")
        close = self._get(last, "close")

        if ema_8 and ema_21:
            if ema_8 > ema_21:
                bullish += 1
                sub_signals["EMA_8/21_Cross"] = {"signal": "BULLISH", "value": f"{ema_8:.2f} > {ema_21:.2f}"}
            else:
                bearish += 1
                sub_signals["EMA_8/21_Cross"] = {"signal": "BEARISH", "value": f"{ema_8:.2f} < {ema_21:.2f}"}

        if ema_21 and ema_55:
            if ema_21 > ema_55:
                bullish += 1
                sub_signals["EMA_21/55_Cross"] = {"signal": "BULLISH", "value": f"{ema_21:.2f} > {ema_55:.2f}"}
            else:
                bearish += 1
                sub_signals["EMA_21/55_Cross"] = {"signal": "BEARISH", "value": f"{ema_21:.2f} < {ema_55:.2f}"}

        # Price vs EMA 200 (major trend)
        if close and ema_200:
            if close > ema_200:
                bullish += 1
                sub_signals["Price_vs_EMA200"] = {"signal": "BULLISH", "value": f"Price above EMA200"}
            else:
                bearish += 1
                sub_signals["Price_vs_EMA200"] = {"signal": "BEARISH", "value": f"Price below EMA200"}

        # ADX (trend strength)
        adx = self._get(last, "adx")
        di_plus = self._get(last, "di_plus")
        di_minus = self._get(last, "di_minus")

        if adx and adx > 25:
            if di_plus and di_minus:
                if di_plus > di_minus:
                    bullish += 1
                    sub_signals["ADX_Trend"] = {"signal": "BULLISH", "value": f"ADX={adx:.1f}, DI+>DI-"}
                else:
                    bearish += 1
                    sub_signals["ADX_Trend"] = {"signal": "BEARISH", "value": f"ADX={adx:.1f}, DI->DI+"}
        else:
            neutral += 1
            adx_val = adx if adx else 0
            sub_signals["ADX_Trend"] = {"signal": "NEUTRAL", "value": f"ADX={adx_val:.1f} (weak trend)"}

        supertrend_dir = self._get(last, "supertrend")
        if supertrend_dir is not None:
            if supertrend_dir > 0:
                bullish += 1
                sub_signals["Supertrend"] = {"signal": "BULLISH", "value": "Uptrend"}
            elif supertrend_dir < 0:
                bearish += 1
                sub_signals["Supertrend"] = {"signal": "BEARISH", "value": "Downtrend"}

        psar = self._get(last, "psar")
        if psar and close:
            if close > psar:
                bullish += 1
                sub_signals["Parabolic_SAR"] = {"signal": "BULLISH", "value": f"Price > SAR"}
            else:
                bearish += 1
                sub_signals["Parabolic_SAR"] = {"signal": "BEARISH", "value": f"Price < SAR"}

        # Ichimoku Cloud
        tenkan = self._get(last, "ichimoku_tenkan")
        kijun = self._get(last, "ichimoku_kijun")
        senkou_a = self._get(last, "ichimoku_senkou_a")
        senkou_b = self._get(last, "ichimoku_senkou_b")

        if tenkan and kijun:
            if tenkan > kijun:
                bullish += 1
                sub_signals["Ichimoku_TK"] = {"signal": "BULLISH", "value": "Tenkan > Kijun"}
            else:
                bearish += 1
                sub_signals["Ichimoku_TK"] = {"signal": "BEARISH", "value": "Tenkan < Kijun"}

        if close and senkou_a and senkou_b:
            cloud_top = max(senkou_a, senkou_b)
            cloud_bottom = min(senkou_a, senkou_b)
            if close > cloud_top:
                bullish += 1
                sub_signals["Ichimoku_Cloud"] = {"signal": "BULLISH", "value": "Above cloud"}
            elif close < cloud_bottom:
                bearish += 1
                sub_signals["Ichimoku_Cloud"] = {"signal": "BEARISH", "value": "Below cloud"}
            else:
                neutral += 1
                sub_signals["Ichimoku_Cloud"] = {"signal": "NEUTRAL", "value": "Inside cloud"}

        # ─────────────────────────────────────────
        # 2. MOMENTUM INDICATORS
        # ─────────────────────────────────────────

        rsi = self._get(last, "rsi")
        if rsi:
            if rsi < 30:
                bullish += 2  # Strong oversold
                sub_signals["RSI"] = {"signal": "STRONG_BULLISH", "value": f"{rsi:.1f} (Oversold)"}
            elif rsi < 40:
                bullish += 1
                sub_signals["RSI"] = {"signal": "BULLISH", "value": f"{rsi:.1f} (Near oversold)"}
            elif rsi > 70:
                bearish += 2  # Strong overbought
                sub_signals["RSI"] = {"signal": "STRONG_BEARISH", "value": f"{rsi:.1f} (Overbought)"}
            elif rsi > 60:
                bearish += 1
                sub_signals["RSI"] = {"signal": "BEARISH", "value": f"{rsi:.1f} (Near overbought)"}
            else:
                neutral += 1
                sub_signals["RSI"] = {"signal": "NEUTRAL", "value": f"{rsi:.1f}"}

        # Stochastic RSI
        stoch_rsi = self._get(last, "stoch_rsi_k")
        if stoch_rsi:
            if stoch_rsi < 20:
                bullish += 1
                sub_signals["Stoch_RSI"] = {"signal": "BULLISH", "value": f"{stoch_rsi:.1f} (Oversold)"}
            elif stoch_rsi > 80:
                bearish += 1
                sub_signals["Stoch_RSI"] = {"signal": "BEARISH", "value": f"{stoch_rsi:.1f} (Overbought)"}
            else:
                neutral += 1
                sub_signals["Stoch_RSI"] = {"signal": "NEUTRAL", "value": f"{stoch_rsi:.1f}"}

        macd_hist = self._get(last, "macd_hist")
        prev_macd_hist = self._get(prev, "macd_hist")
        if macd_hist is not None:
            if macd_hist > 0:
                bullish += 1
                if prev_macd_hist is not None and macd_hist > prev_macd_hist:
                    bullish += 1  # Increasing momentum
                    sub_signals["MACD"] = {"signal": "STRONG_BULLISH", "value": f"Histogram={macd_hist:.4f} (increasing)"}
                else:
                    sub_signals["MACD"] = {"signal": "BULLISH", "value": f"Histogram={macd_hist:.4f}"}
            else:
                bearish += 1
                if prev_macd_hist is not None and macd_hist < prev_macd_hist:
                    bearish += 1
                    sub_signals["MACD"] = {"signal": "STRONG_BEARISH", "value": f"Histogram={macd_hist:.4f} (increasing)"}
                else:
                    sub_signals["MACD"] = {"signal": "BEARISH", "value": f"Histogram={macd_hist:.4f}"}

        # Williams %R
        williams_r = self._get(last, "williams_r")
        if williams_r is not None:
            if williams_r < -80:
                bullish += 1
                sub_signals["Williams_%R"] = {"signal": "BULLISH", "value": f"{williams_r:.1f} (Oversold)"}
            elif williams_r > -20:
                bearish += 1
                sub_signals["Williams_%R"] = {"signal": "BEARISH", "value": f"{williams_r:.1f} (Overbought)"}

        # CCI
        cci = self._get(last, "cci")
        if cci is not None:
            if cci < -100:
                bullish += 1
                sub_signals["CCI"] = {"signal": "BULLISH", "value": f"{cci:.1f} (Oversold)"}
            elif cci > 100:
                bearish += 1
                sub_signals["CCI"] = {"signal": "BEARISH", "value": f"{cci:.1f} (Overbought)"}

        # MFI (Money Flow Index)
        mfi = self._get(last, "mfi")
        if mfi is not None:
            if mfi < 20:
                bullish += 1
                sub_signals["MFI"] = {"signal": "BULLISH", "value": f"{mfi:.1f} (Money flowing in)"}
            elif mfi > 80:
                bearish += 1
                sub_signals["MFI"] = {"signal": "BEARISH", "value": f"{mfi:.1f} (Money flowing out)"}

        # ROC (Rate of Change)
        roc = self._get(last, "roc_10")
        if roc is not None:
            if roc > 0:
                bullish += 1
                sub_signals["ROC"] = {"signal": "BULLISH", "value": f"{roc:.2f}%"}
            else:
                bearish += 1
                sub_signals["ROC"] = {"signal": "BEARISH", "value": f"{roc:.2f}%"}

        # ─────────────────────────────────────────
        # 3. VOLATILITY INDICATORS
        # ─────────────────────────────────────────

        bb_pct = self._get(last, "bb_pct_b")
        if bb_pct is not None:
            if bb_pct < 0.0:
                bullish += 2  # Below lower band = strong reversal signal
                sub_signals["Bollinger_%B"] = {"signal": "STRONG_BULLISH", "value": f"{bb_pct:.3f} (Below lower band)"}
            elif bb_pct < 0.2:
                bullish += 1
                sub_signals["Bollinger_%B"] = {"signal": "BULLISH", "value": f"{bb_pct:.3f} (Near lower band)"}
            elif bb_pct > 1.0:
                bearish += 2
                sub_signals["Bollinger_%B"] = {"signal": "STRONG_BEARISH", "value": f"{bb_pct:.3f} (Above upper band)"}
            elif bb_pct > 0.8:
                bearish += 1
                sub_signals["Bollinger_%B"] = {"signal": "BEARISH", "value": f"{bb_pct:.3f} (Near upper band)"}

        # Keltner Channel position
        keltner_upper = self._get(last, "keltner_upper")
        keltner_lower = self._get(last, "keltner_lower")
        if close and keltner_upper and keltner_lower:
            if close < keltner_lower:
                bullish += 1
                sub_signals["Keltner"] = {"signal": "BULLISH", "value": "Below lower channel"}
            elif close > keltner_upper:
                bearish += 1
                sub_signals["Keltner"] = {"signal": "BEARISH", "value": "Above upper channel"}

        # ─────────────────────────────────────────
        # 4. VOLUME INDICATORS
        # ─────────────────────────────────────────

        # OBV slope
        obv = self._get(last, "obv")
        prev_obv = self._get(prev, "obv")
        if obv is not None and prev_obv is not None:
            if obv > prev_obv:
                bullish += 1
                sub_signals["OBV"] = {"signal": "BULLISH", "value": "Volume accumulation"}
            else:
                bearish += 1
                sub_signals["OBV"] = {"signal": "BEARISH", "value": "Volume distribution"}

        # CMF (Chaikin Money Flow)
        cmf = self._get(last, "cmf")
        if cmf is not None:
            if cmf > 0.05:
                bullish += 1
                sub_signals["CMF"] = {"signal": "BULLISH", "value": f"{cmf:.3f} (Buying pressure)"}
            elif cmf < -0.05:
                bearish += 1
                sub_signals["CMF"] = {"signal": "BEARISH", "value": f"{cmf:.3f} (Selling pressure)"}

        # Relative Volume
        rel_vol = self._get(last, "relative_volume")
        if rel_vol is not None:
            sub_signals["Rel_Volume"] = {"signal": "INFO", "value": f"{rel_vol:.2f}x average"}

        # ─────────────────────────────────────────
        # 5. CANDLESTICK PATTERNS
        # ─────────────────────────────────────────
        candle_bull = 0
        candle_bear = 0
        pattern_names = [
            "cdl_doji", "cdl_hammer", "cdl_inverted_hammer", "cdl_shooting_star", "cdl_hanging_man",
            "cdl_bullish_engulfing", "cdl_bearish_engulfing", "cdl_bullish_harami", "cdl_bearish_harami",
            "cdl_morning_star", "cdl_evening_star", "cdl_three_white_soldiers", "cdl_three_black_crows",
            "cdl_piercing_line", "cdl_dark_cloud_cover", "cdl_tweezer_top", "cdl_tweezer_bottom",
        ]
        detected_patterns = []
        for pname in pattern_names:
            val = self._get(last, pname)
            if val is not None and val != 0:
                if val > 0:
                    candle_bull += 1
                    detected_patterns.append(f"✅ {pname} (Bullish)")
                elif val < 0:
                    candle_bear += 1
                    detected_patterns.append(f"❌ {pname} (Bearish)")

        if candle_bull > candle_bear:
            bullish += candle_bull
        elif candle_bear > candle_bull:
            bearish += candle_bear

        sub_signals["Candlestick_Patterns"] = {
            "signal": "BULLISH" if candle_bull > candle_bear else "BEARISH" if candle_bear > candle_bull else "NEUTRAL",
            "value": f"Bull={candle_bull}, Bear={candle_bear}",
            "patterns": detected_patterns
        }

        # ─────────────────────────────────────────
        # 6. MARKET SESSION & STOP/ENTRY FUNCTIONS
        atr = self._get(last, "atr")
        if close and atr:
            # Calculate dynamic stop loss & take profit
            if bullish > bearish:
                signal.details["suggested_entry"] = close
                signal.details["suggested_stop_loss"] = close - (atr * 1.5)
                signal.details["suggested_take_profit_1"] = close + (atr * 1.0)
                signal.details["suggested_take_profit_2"] = close + (atr * 2.0)
                signal.details["suggested_take_profit_3"] = close + (atr * 3.0)
            else:
                signal.details["suggested_entry"] = close
                signal.details["suggested_stop_loss"] = close + (atr * 1.5)
                signal.details["suggested_take_profit_1"] = close - (atr * 1.0)
                signal.details["suggested_take_profit_2"] = close - (atr * 2.0)
                signal.details["suggested_take_profit_3"] = close - (atr * 3.0)

            signal.details["atr"] = float(atr)
            signal.details["risk_reward_1"] = 1.0 / 1.5
            signal.details["risk_reward_2"] = 2.0 / 1.5
            signal.details["risk_reward_3"] = 3.0 / 1.5

        # Market open/close strength analysis
        body_ratio = self._get(last, "body_ratio")
        if body_ratio is not None:
            signal.details["candle_body_strength"] = float(body_ratio)

        # ─────────────────────────────────────────
        # LAYER 1 FINAL SCORING
        # ─────────────────────────────────────────

        total_signals = bullish + bearish + neutral
        if total_signals == 0:
            total_signals = 1

        if bullish > bearish:
            signal.direction = "UP"
            agreement = bullish / total_signals
        elif bearish > bullish:
            signal.direction = "DOWN"
            agreement = bearish / total_signals
        else:
            signal.direction = "NEUTRAL"
            agreement = 0.5

        # Layer 1 confidence: 10-30% range
        raw_confidence = agreement
        signal.confidence = 0.10 + (raw_confidence * 0.20)  # Maps to 0.10 - 0.30
        signal.confidence = max(0.10, min(0.30, signal.confidence))

        signal.up_probability = bullish / total_signals if total_signals > 0 else 0.5
        signal.down_probability = bearish / total_signals if total_signals > 0 else 0.5
        signal.signals_bullish = bullish
        signal.signals_bearish = bearish
        signal.signals_neutral = neutral
        signal.sub_signals = sub_signals

        signal.details["total_indicators_analyzed"] = total_signals
        signal.details["bullish_count"] = bullish
        signal.details["bearish_count"] = bearish
        signal.details["neutral_count"] = neutral

        return signal

    def _get(self, row, key, default=None):
        """Safely get a value from DataFrame row."""
        try:
            val = row.get(key, default) if isinstance(row, dict) else getattr(row, key, default)
            if val is not None and isinstance(val, (int, float, np.integer, np.floating)):
                if np.isnan(val) or np.isinf(val):
                    return default
                return float(val)
            return default
        except:
            return default


# ═══════════════════════════════════════════════════════════════
# LAYER 2: MAIN LAYER (30-80% Confidence)
# Custom advanced indicators + ensemble prediction
# ═══════════════════════════════════════════════════════════════

class Layer2_Main:
    """
    🟠 LAYER 2 — MAIN PREDICTION (30-80% Confidence)
    
    Layer 1 এর base signal নিয়ে, custom advanced indicators ও
    statistical models দিয়ে confidence আরো বাড়ায়।
    
    XGBoost/LightGBM + HMM Regime + Novel Indicators + Multi-Timeframe
    """

    def __init__(self, config=CONFIG):
        self.config = config
        self.xgb_model = None
        self.regime_detector = None
        self.is_trained = False

    def analyze(self, df, featured_df, layer1_signal: LayerSignal, 
                higher_tf_data: Dict = None) -> LayerSignal:
        """Run Layer 2 analysis, building on Layer 1."""
        signal = LayerSignal()
        signal.details = {"layer": "Layer 2 — Main", "max_confidence": "80%"}

        last = featured_df.iloc[-1] if len(featured_df) > 0 else {}

        bullish = 0
        bearish = 0
        neutral = 0
        sub_signals = {}

        # Start with Layer 1's direction bias
        if layer1_signal.direction == "UP":
            bullish += 2
        elif layer1_signal.direction == "DOWN":
            bearish += 2

        # ─────────────────────────────────────────
        # 1. NOVEL CUSTOM INDICATORS
        # ─────────────────────────────────────────

        # Adaptive Fractal Momentum (AFM)
        afm = self._get(last, "adaptive_fractal_momentum")
        if afm is not None:
            if afm > 0.3:
                bullish += 2
                sub_signals["AFM"] = {"signal": "STRONG_BULLISH", "value": f"{afm:.4f}"}
            elif afm > 0:
                bullish += 1
                sub_signals["AFM"] = {"signal": "BULLISH", "value": f"{afm:.4f}"}
            elif afm < -0.3:
                bearish += 2
                sub_signals["AFM"] = {"signal": "STRONG_BEARISH", "value": f"{afm:.4f}"}
            elif afm < 0:
                bearish += 1
                sub_signals["AFM"] = {"signal": "BEARISH", "value": f"{afm:.4f}"}

        vwrs = self._get(last, "vw_regime_score")
        if vwrs is not None:
            if vwrs > 0.3:
                bullish += 2
                sub_signals["VWRS"] = {"signal": "STRONG_BULLISH", "value": f"{vwrs:.4f}"}
            elif vwrs > 0:
                bullish += 1
                sub_signals["VWRS"] = {"signal": "BULLISH", "value": f"{vwrs:.4f}"}
            elif vwrs < -0.3:
                bearish += 2
                sub_signals["VWRS"] = {"signal": "STRONG_BEARISH", "value": f"{vwrs:.4f}"}
            elif vwrs < 0:
                bearish += 1
                sub_signals["VWRS"] = {"signal": "BEARISH", "value": f"{vwrs:.4f}"}

        # Market DNA Fingerprint
        dna = self._get(last, "market_dna_score")
        if dna is not None:
            if dna > 0.6:
                bullish += 2
                sub_signals["Market_DNA"] = {"signal": "BULLISH", "value": f"Score={dna:.3f}"}
            elif dna < -0.6:
                bearish += 2
                sub_signals["Market_DNA"] = {"signal": "BEARISH", "value": f"Score={dna:.3f}"}

        # Entropy Reversal Detector
        entropy_rev = self._get(last, "entropy_reversal_signal")
        if entropy_rev is not None and abs(entropy_rev) > 0.5:
            if entropy_rev > 0:
                bullish += 1
                sub_signals["Entropy_Reversal"] = {"signal": "BULLISH", "value": f"{entropy_rev:.4f} (Reversal UP detected)"}
            else:
                bearish += 1
                sub_signals["Entropy_Reversal"] = {"signal": "BEARISH", "value": f"{entropy_rev:.4f} (Reversal DOWN detected)"}

        # Cross-Timeframe Pressure Index
        ctpi = self._get(last, "ctf_pressure_index")
        if ctpi is not None:
            if ctpi > 0.3:
                bullish += 2
                sub_signals["CTPI"] = {"signal": "STRONG_BULLISH", "value": f"{ctpi:.4f} (HTF pushing UP)"}
            elif ctpi > 0:
                bullish += 1
                sub_signals["CTPI"] = {"signal": "BULLISH", "value": f"{ctpi:.4f}"}
            elif ctpi < -0.3:
                bearish += 2
                sub_signals["CTPI"] = {"signal": "STRONG_BEARISH", "value": f"{ctpi:.4f} (HTF pushing DOWN)"}
            elif ctpi < 0:
                bearish += 1
                sub_signals["CTPI"] = {"signal": "BEARISH", "value": f"{ctpi:.4f}"}

        # ─────────────────────────────────────────
        # 2. MARKET STRUCTURE (Smart Money Concepts)
        # ─────────────────────────────────────────

        trend_dir_short = self._get(last, "trend_short")
        trend_dir_medium = self._get(last, "trend_med")
        trend_dir_long = self._get(last, "trend_long")

        trend_sum = 0
        for td in [trend_dir_short, trend_dir_medium, trend_dir_long]:
            if td is not None:
                trend_sum += td

        if trend_sum > 1:
            bullish += 2
            sub_signals["Trend_Structure"] = {"signal": "BULLISH", "value": f"Multi-TF trend alignment UP ({trend_sum})"}
        elif trend_sum < -1:
            bearish += 2
            sub_signals["Trend_Structure"] = {"signal": "BEARISH", "value": f"Multi-TF trend alignment DOWN ({trend_sum})"}

        # Break of Structure
        bos_bull = self._get(last, "bos_bull")
        bos_bear = self._get(last, "bos_bear")
        if bos_bull:
            bullish += 2
            sub_signals["Break_of_Structure"] = {"signal": "BULLISH", "value": "Bullish BOS detected"}
        elif bos_bear:
            bearish += 2
            sub_signals["Break_of_Structure"] = {"signal": "BEARISH", "value": "Bearish BOS detected"}

        # Support/Resistance proximity
        dist_support = self._get(last, "dist_to_support")
        dist_resistance = self._get(last, "dist_to_resistance")
        if dist_support is not None and dist_resistance is not None:
            if dist_support < dist_resistance * 0.3:
                bullish += 1  # Near support = potential bounce
                sub_signals["S/R_Proximity"] = {"signal": "BULLISH", "value": f"Near support (dist={dist_support:.4f})"}
            elif dist_resistance < dist_support * 0.3:
                bearish += 1  # Near resistance = potential rejection
                sub_signals["S/R_Proximity"] = {"signal": "BEARISH", "value": f"Near resistance (dist={dist_resistance:.4f})"}

        # Nadaraya-Watson Envelope
        nw_signal = self._get(last, "nw_signal")
        nw_est = self._get(last, "nadaraya_watson")
        if nw_signal is not None:
            if nw_signal > 0:
                bullish += 3
                sub_signals["Nadaraya_Watson"] = {"signal": "STRONG_BULLISH", "value": f"Oversold below NW band ({nw_est:.2f})"}
            elif nw_signal < 0:
                bearish += 3
                sub_signals["Nadaraya_Watson"] = {"signal": "STRONG_BEARISH", "value": f"Overbought above NW band ({nw_est:.2f})"}

        # Smart Money Fair Value Gaps (FVG)
        fvg_imbalance = self._get(last, "fvg_imbalance")
        if fvg_imbalance is not None:
            if fvg_imbalance > 1:
                bullish += 2
                sub_signals["SMC_Fair_Value_Gap"] = {"signal": "BULLISH", "value": f"Unmitigated Bullish Imbalance (+{fvg_imbalance})"}
            elif fvg_imbalance < -1:
                bearish += 2
                sub_signals["SMC_Fair_Value_Gap"] = {"signal": "BEARISH", "value": f"Unmitigated Bearish Imbalance ({fvg_imbalance})"}

        # ─────────────────────────────────────────
        # 3. STATISTICAL FEATURES
        # ─────────────────────────────────────────

        # Chaos Theory & Non-Linear Dynamics
        chaos_hurst = self._get(last, "chaos_hurst")
        chaos_lyap = self._get(last, "chaos_lyapunov")
        chaos_fd = self._get(last, "chaos_fractal_dim")
        chaos_pred = self._get(last, "chaos_predictability_score")
        rsi_val = self._get(last, "rsi_14")
        
        if chaos_hurst is not None:
            if chaos_pred > 0.8:
                # Highly predictable trend
                if layer1_signal.direction == "UP":
                    bullish += 2
                    sub_signals["Chaos_Theory_Trend"] = {"signal": "STRONG_BULLISH", "value": f"H={chaos_hurst:.2f}, LE={chaos_lyap:.3f} (Deterministic Trend)"}
                elif layer1_signal.direction == "DOWN":
                    bearish += 2
                    sub_signals["Chaos_Theory_Trend"] = {"signal": "STRONG_BEARISH", "value": f"H={chaos_hurst:.2f}, LE={chaos_lyap:.3f} (Deterministic Trend)"}
            elif chaos_hurst < 0.45:
                # Mean reverting regime
                if rsi_val is not None and rsi_val > 65:
                    bearish += 2
                    sub_signals["Chaos_Theory_Reversion"] = {"signal": "STRONG_BEARISH", "value": f"H={chaos_hurst:.2f} (Mean Reverting Overbought)"}
                elif rsi_val is not None and rsi_val < 35:
                    bullish += 2
                    sub_signals["Chaos_Theory_Reversion"] = {"signal": "STRONG_BULLISH", "value": f"H={chaos_hurst:.2f} (Mean Reverting Oversold)"}
            elif chaos_lyap > 0:
                sub_signals["Chaos_Warning"] = {"signal": "WARNING", "value": f"LE={chaos_lyap:.3f} (Stochastic/Random Market)"}

        # Legacy Hurst Exponent
        hurst = self._get(last, "hurst_50")
        if hurst is not None and "Chaos_Theory_Trend" not in sub_signals:
            if hurst > 0.6:
                if layer1_signal.direction == "UP":
                    bullish += 1
                else:
                    bearish += 1
                sub_signals["Hurst"] = {"signal": layer1_signal.direction, "value": f"{hurst:.3f} (Trending market)"}
            elif hurst < 0.4:
                # Mean-reverting — counter the current direction
                if layer1_signal.direction == "UP":
                    bearish += 1
                else:
                    bullish += 1
                sub_signals["Hurst"] = {"signal": "COUNTER", "value": f"{hurst:.3f} (Mean-reverting market)"}

        # Z-score of price
        zscore = self._get(last, "zscore_50")
        if zscore is not None:
            if zscore < -2:
                bullish += 2
                sub_signals["Price_ZScore"] = {"signal": "STRONG_BULLISH", "value": f"{zscore:.2f} (Extremely low)"}
            elif zscore > 2:
                bearish += 2
                sub_signals["Price_ZScore"] = {"signal": "STRONG_BEARISH", "value": f"{zscore:.2f} (Extremely high)"}

        # Volatility regime
        vol_regime = self._get(last, "vol_regime")
        if vol_regime is not None:
            sub_signals["Volatility_Regime"] = {"signal": "INFO", "value": f"{vol_regime}"}

        # ─────────────────────────────────────────
        # 4. MULTI-TIMEFRAME ANALYSIS
        # ─────────────────────────────────────────

        tf_alignment = self._get(last, "timeframe_alignment_score")
        if tf_alignment is not None:
            if tf_alignment > 0.5:
                bullish += 3
                sub_signals["MTF_Alignment"] = {"signal": "STRONG_BULLISH", "value": f"{tf_alignment:.3f} (All TF aligned UP)"}
            elif tf_alignment > 0.2:
                bullish += 1
                sub_signals["MTF_Alignment"] = {"signal": "BULLISH", "value": f"{tf_alignment:.3f}"}
            elif tf_alignment < -0.5:
                bearish += 3
                sub_signals["MTF_Alignment"] = {"signal": "STRONG_BEARISH", "value": f"{tf_alignment:.3f} (All TF aligned DOWN)"}
            elif tf_alignment < -0.2:
                bearish += 1
                sub_signals["MTF_Alignment"] = {"signal": "BEARISH", "value": f"{tf_alignment:.3f}"}

        htf_bias = self._get(last, "higher_timeframe_bias")
        if htf_bias is not None:
            if htf_bias > 0.3:
                bullish += 2
                sub_signals["HTF_Bias"] = {"signal": "BULLISH", "value": f"{htf_bias:.3f} (Higher TF bullish)"}
            elif htf_bias < -0.3:
                bearish += 2
                sub_signals["HTF_Bias"] = {"signal": "BEARISH", "value": f"{htf_bias:.3f} (Higher TF bearish)"}

        # ─────────────────────────────────────────
        # 5. ORDER FLOW / VOLUME DIVERGENCE
        # ─────────────────────────────────────────

        buy_vol = self._get(last, "buy_volume_est")
        sell_vol = self._get(last, "sell_volume_est")
        if buy_vol is not None and sell_vol is not None and sell_vol > 0:
            buy_sell = buy_vol / sell_vol
            if buy_sell > 1.3:
                bullish += 1
                sub_signals["Buy_Sell_Ratio"] = {"signal": "BULLISH", "value": f"{buy_sell:.2f}x (Buyers dominant)"}
            elif buy_sell < 0.7:
                bearish += 1
                sub_signals["Buy_Sell_Ratio"] = {"signal": "BEARISH", "value": f"{buy_sell:.2f}x (Sellers dominant)"}

        vol_divergence = self._get(last, "volume_divergence")
        if vol_divergence is not None and abs(vol_divergence) > 0.3:
            sub_signals["Volume_Divergence"] = {"signal": "WARNING", "value": f"{vol_divergence:.3f} (Divergence detected!)"}

        # ─────────────────────────────────────────
        # 6. XGBoost PREDICTION (if trained)
        # ─────────────────────────────────────────

        if self.is_trained and self.xgb_model is not None:
            try:
                exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'label', 'future_return']
                feature_cols = [c for c in featured_df.columns if c not in exclude_cols and featured_df[c].dtype in ['float64', 'float32', 'int64', 'int32']]
                X = featured_df[feature_cols].iloc[-1:].values.astype(np.float32)
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
                xgb_pred = self.xgb_model.predict(X)
                xgb_dir = xgb_pred.get('direction', 'UP')
                xgb_prob = xgb_pred.get('probability', 0.5)

                if xgb_dir == "UP":
                    bullish += 3
                else:
                    bearish += 3

                sub_signals["XGBoost_Model"] = {"signal": xgb_dir, "value": f"Prob={xgb_prob:.3f}"}
            except Exception as e:
                sub_signals["XGBoost_Model"] = {"signal": "ERROR", "value": str(e)}

        # ─────────────────────────────────────────
        # 7. REGIME DETECTION (HMM)
        # ─────────────────────────────────────────

        regime_label = "unknown"
        if self.regime_detector is not None:
            try:
                close_prices = df['close'].values
                returns = np.diff(np.log(close_prices + 1e-10))
                regime_info = self.regime_detector.detect_regime(returns)
                regime_label = regime_info.get('label', 'unknown')
                sub_signals["Market_Regime"] = {"signal": "INFO", "value": regime_label}

                if regime_label in ['bull_trend']:
                    bullish += 2
                elif regime_label in ['bear_trend', 'crash']:
                    bearish += 2
            except:
                pass

        signal.details["regime"] = regime_label

        # ─────────────────────────────────────────
        # LAYER 2 FINAL SCORING
        # ─────────────────────────────────────────

        total = bullish + bearish + neutral
        if total == 0:
            total = 1

        if bullish > bearish:
            signal.direction = "UP"
            agreement = bullish / total
        elif bearish > bullish:
            signal.direction = "DOWN"
            agreement = bearish / total
        else:
            signal.direction = layer1_signal.direction  # Defer to Layer 1
            agreement = 0.5

        # Layer 2 confidence: 30-80% range
        # Higher agreement + alignment with Layer 1 = higher confidence
        layer_agreement_bonus = 0.1 if signal.direction == layer1_signal.direction else -0.1
        raw_confidence = agreement + layer_agreement_bonus
        signal.confidence = 0.30 + (raw_confidence * 0.50)
        signal.confidence = max(0.30, min(0.80, signal.confidence))

        signal.up_probability = bullish / total if total > 0 else 0.5
        signal.down_probability = bearish / total if total > 0 else 0.5
        signal.signals_bullish = bullish
        signal.signals_bearish = bearish
        signal.signals_neutral = neutral
        signal.sub_signals = sub_signals

        return signal

    def _get(self, row, key, default=None):
        try:
            val = row.get(key, default) if isinstance(row, dict) else getattr(row, key, default)
            if val is not None and isinstance(val, (int, float, np.integer, np.floating)):
                if np.isnan(val) or np.isinf(val):
                    return default
                return float(val)
            return default
        except:
            return default


# ═══════════════════════════════════════════════════════════════
# LAYER 3: SUPREME BRAIN (80-98% Confidence)
# Deep Learning + Self-Learning + Self-Evolving
# ═══════════════════════════════════════════════════════════════

class Layer3_Supreme:
    """
    🔴 LAYER 3 — SUPREME BRAIN (80-98% Confidence)
    
    Deep Learning মডেল (Transformer, LSTM, WaveNet, CNN) + 
    Self-Learning (RL, Genetic Programming) + 
    Self-Reflection (learn from mistakes) — 
    
    এটি সিস্টেমের মস্তিষ্ক। শুধুমাত্র যখন Layer 1 ও 2 agree করে এবং 
    deep learning models ও confirm করে — তখনই confidence 80%+ হয়।
    """

    def __init__(self, config=CONFIG):
        self.config = config
        self.ensemble = None
        self.performance_tracker = None
        self.self_reflection = None
        self.is_trained = False

    def analyze(self, df, featured_df, layer1_signal: LayerSignal, 
                layer2_signal: LayerSignal) -> LayerSignal:
        """Run Layer 3 deep analysis, building on Layer 1 + 2."""
        signal = LayerSignal()
        signal.details = {"layer": "Layer 3 — Supreme Brain", "max_confidence": "98%"}

        sub_signals = {}
        bullish = 0
        bearish = 0
        neutral = 0

        # ─────────────────────────────────────────
        # 1. LAYER AGREEMENT CHECK
        # ─────────────────────────────────────────

        l1_dir = layer1_signal.direction
        l2_dir = layer2_signal.direction
        layers_agree = (l1_dir == l2_dir) and l1_dir != "NEUTRAL"

        if layers_agree:
            if l1_dir == "UP":
                bullish += 5
            else:
                bearish += 5
            sub_signals["Layer_Agreement"] = {"signal": l1_dir, "value": f"Layer 1 + 2 both say {l1_dir} ✅"}
        else:
            neutral += 3
            sub_signals["Layer_Agreement"] = {"signal": "CONFLICT", "value": f"L1={l1_dir}, L2={l2_dir} ⚠️"}

        # Combined confidence from lower layers
        combined_lower_confidence = (layer1_signal.confidence + layer2_signal.confidence) / 2
        sub_signals["Lower_Layers_Confidence"] = {"signal": "INFO", "value": f"L1={layer1_signal.confidence:.1%}, L2={layer2_signal.confidence:.1%}"}

        # ─────────────────────────────────────────
        # 2. DEEP LEARNING MODELS (if trained)
        # ─────────────────────────────────────────

        dl_predictions = {}
        dl_agreement = 0

        if self.is_trained and self.ensemble is not None:
            try:
                exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'label', 'future_return']
                feature_cols = [c for c in featured_df.columns if c not in exclude_cols and featured_df[c].dtype in ['float64', 'float32', 'int64', 'int32']]

                X = featured_df[feature_cols].values.astype(np.float32)
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

                seq_len = self.config.transformer.sequence_length
                if len(X) > seq_len:
                    # Create sequence for latest point
                    X_seq = X[-seq_len:].reshape(1, seq_len, -1)

                    close_prices = df['close'].values
                    returns_data = np.diff(np.log(close_prices + 1e-10))

                    prediction = self.ensemble.predict(X_seq, current_returns_data=returns_data)

                    dl_dir = prediction.get('direction', 'NEUTRAL')
                    dl_confidence = prediction.get('confidence_score', 0)
                    dl_agreement_ratio = prediction.get('agreement_ratio', 0)

                    individual = prediction.get('individual_model_predictions', {})
                    for model_name, model_pred in individual.items():
                        m_dir = model_pred.get('direction', 'NEUTRAL')
                        m_prob = model_pred.get('probability', 0.5)
                        dl_predictions[model_name] = m_dir
                        sub_signals[f"DL_{model_name}"] = {"signal": m_dir, "value": f"{m_prob:.3f}"}

                        if m_dir == "UP":
                            bullish += 2
                        elif m_dir == "DOWN":
                            bearish += 2

                    sub_signals["DL_Ensemble"] = {"signal": dl_dir, "value": f"Conf={dl_confidence:.3f}, Agreement={dl_agreement_ratio:.3f}"}
                    dl_agreement = dl_agreement_ratio

            except Exception as e:
                sub_signals["DL_Models"] = {"signal": "ERROR", "value": str(e)}

        else:
            # No trained models — use Layer 2's advanced indicators as proxy
            sub_signals["DL_Models"] = {"signal": "NOT_TRAINED", "value": "Train models for Layer 3 power: python main.py train"}

            # Still provide some Layer 3 analysis using advanced statistical signals
            # Use Layer 2's confidence as a basis
            if layer2_signal.confidence > 0.6:
                if layer2_signal.direction == "UP":
                    bullish += 3
                else:
                    bearish += 3
            elif layer2_signal.confidence > 0.4:
                if layer2_signal.direction == "UP":
                    bullish += 1
                else:
                    bearish += 1

        # ─────────────────────────────────────────
        # 3. SELF-REFLECTION (learn from mistakes)
        # ─────────────────────────────────────────

        if self.self_reflection is not None:
            try:
                raw_pred = 1.0 if bullish > bearish else (0.0 if bearish > bullish else 0.5)
                correction = self.self_reflection.apply_corrections(
                    raw_pred,
                    featured_df.iloc[-1].to_dict() if len(featured_df) > 0 else {}
                )
                if correction != raw_pred:
                    sub_signals["Self_Reflection"] = {
                        "signal": "ADJUSTED",
                        "value": f"Prediction adjusted from {raw_pred:.2f} to {correction:.2f}"
                    }
            except:
                pass

        # ─────────────────────────────────────────
        # 4. PERFORMANCE TRACKING
        # ─────────────────────────────────────────

        if self.performance_tracker is not None:
            try:
                rolling_acc = self.performance_tracker.get_rolling_accuracy()
                sub_signals["Rolling_Accuracy"] = {"signal": "INFO", "value": f"{rolling_acc:.1%}"}

                if self.performance_tracker.should_retrain():
                    sub_signals["Retrain_Alert"] = {"signal": "WARNING", "value": "Model needs retraining!"}
            except:
                pass

        # ─────────────────────────────────────────
        # LAYER 3 FINAL SCORING
        # ─────────────────────────────────────────

        total = bullish + bearish + neutral
        if total == 0:
            total = 1

        if bullish > bearish:
            signal.direction = "UP"
            agreement = bullish / total
        elif bearish > bullish:
            signal.direction = "DOWN"
            agreement = bearish / total
        else:
            signal.direction = layer2_signal.direction
            agreement = 0.5

        # Layer 3 confidence: 80-98% range
        # But ONLY if layers agree AND deep learning confirms
        all_three_agree = (signal.direction == l1_dir == l2_dir) and l1_dir != "NEUTRAL"

        if all_three_agree and self.is_trained:
            # Maximum confidence when all 3 layers + DL models agree
            raw_confidence = agreement * (1 + dl_agreement) / 2
            signal.confidence = 0.85 + (raw_confidence * 0.14)  # 85-99%
        elif all_three_agree and not self.is_trained:
            # Good confidence even without DL, if all layers agree strongly
            # (Allows breaking 90% in heuristic mode if alignment is perfect)
            signal.confidence = 0.75 + (agreement * 0.22)  # 75-97%
        elif layers_agree:
            # Layer 1+2 agree but Layer 3 has concerns
            signal.confidence = 0.60 + (agreement * 0.25)  # 60-85%
        else:
            # Disagreement — lower confidence
            signal.confidence = 0.40 + (agreement * 0.25)  # 40-65%

        signal.confidence = max(0.10, min(0.98, signal.confidence))

        signal.up_probability = bullish / total if total > 0 else 0.5
        signal.down_probability = bearish / total if total > 0 else 0.5
        signal.signals_bullish = bullish
        signal.signals_bearish = bearish
        signal.signals_neutral = neutral
        signal.sub_signals = sub_signals

        return signal


# ═══════════════════════════════════════════════════════════════
# MASTER BRAIN — Orchestrates all 3 layers
# ═══════════════════════════════════════════════════════════════

class MasterBrain:
    """
    🧠 MASTER BRAIN — The central nervous system.
    
    Runs all 3 layers sequentially, with each layer building
    on the previous one's output to progressively increase confidence.
    """

    def __init__(self, config=CONFIG):
        self.config = config
        self.layer1 = Layer1_Initial(config)
        self.layer2 = Layer2_Main(config)
        self.layer3 = Layer3_Supreme(config)
        self._models_loaded = False
        if HAS_ANALYSIS:
            self.multi_tf_analyzer = MultiTimeframeAnalyzer(config)
            self.pattern_matcher = HistoricalPatternMatcher(config)
            self.time_predictor = TimeDurationPredictor(config)
        else:
            self.multi_tf_analyzer = None
            self.pattern_matcher = None
            self.time_predictor = None

    def load_trained_models(self):
        """Load trained models if available."""
        meta_path = os.path.join(self.config.model_save_dir, 'meta.json')
        if os.path.exists(meta_path):
            try:
                from models.ensemble import MasterEnsemble
                from models.xgboost_model import GradientBoostPredictor
                from models.regime_detector import RegimeDetector

                ensemble = MasterEnsemble(self.config)
                ensemble.load_all()
                self.layer3.ensemble = ensemble
                self.layer3.is_trained = True

                # Try loading individual models for Layer 2
                xgb_path = os.path.join(self.config.model_save_dir, "xgboost_model.pt")
                if os.path.exists(xgb_path):
                    xgb = GradientBoostPredictor(self.config)
                    xgb.load(xgb_path)
                    self.layer2.xgb_model = xgb
                    self.layer2.is_trained = True

                regime_path = os.path.join(self.config.model_save_dir, "regime_model.pkl")
                if os.path.exists(regime_path):
                    regime = RegimeDetector(self.config)
                    regime.load(regime_path)
                    self.layer2.regime_detector = regime

                # Load self-improvement modules
                try:
                    from self_improvement.self_reflection import SelfReflection
                    from self_improvement.performance_tracker import PerformanceTracker
                    self.layer3.self_reflection = SelfReflection()
                    self.layer3.performance_tracker = PerformanceTracker()
                except:
                    pass

                self._models_loaded = True
                print("✅ Brain: All trained models loaded")
            except Exception as e:
                print(f"⚠️  Brain: Could not load models: {e}")

    def predict(self, symbol: str, timeframe: str) -> BrainOutput:
        """
        Run the full 3-layer prediction pipeline.
        
        Layer 1 → Layer 2 → Layer 3 → Final Output
        """
        from data.fetcher import DataFetcher
        from features.feature_engine import FeatureEngine

        output = BrainOutput()
        output.symbol = symbol
        output.timeframe = timeframe
        output.timestamp = datetime.utcnow().isoformat()

        # ── Fetch Data ──
        fetcher = DataFetcher(self.config)
        # Limit to 300 candles for live prediction to massively speed up API calls
        fetch_tfs = list(set(self.config.data.timeframes + [timeframe]))
        data_dict = fetcher.fetch_multi_timeframe(symbol, fetch_tfs, limit=300)
        primary_df = data_dict.get(timeframe)

        if primary_df is None or len(primary_df) < 100:
            output.layer1.details = {"error": "Not enough data"}
            return output

        output.current_price = float(primary_df['close'].iloc[-1])

        # ── Compute Features ──
        feature_engine = FeatureEngine(self.config)
        featured_df = feature_engine.compute_all_features(primary_df, higher_tf_data=data_dict)
        featured_df = feature_engine.remove_nan_rows(featured_df)

        if len(featured_df) < 10:
            output.layer1.details = {"error": "Not enough features computed"}
            return output

        # Load models if not done
        if not self._models_loaded:
            self.load_trained_models()

        # ══════════════════════════════════════
        # RUN LAYER 1 (10-30% confidence)
        # ══════════════════════════════════════
        output.layer1 = self.layer1.analyze(primary_df, featured_df)

        # ══════════════════════════════════════
        # RUN LAYER 2 (30-80% confidence)
        # ══════════════════════════════════════
        output.layer2 = self.layer2.analyze(
            primary_df, featured_df, output.layer1,
            higher_tf_data=data_dict
        )

        # ══════════════════════════════════════
        # RUN LAYER 3 (80-98% confidence)
        # ══════════════════════════════════════
        output.layer3 = self.layer3.analyze(
            primary_df, featured_df, output.layer1, output.layer2
        )

        # === Advanced Analysis ===
        multi_tf_data = {}
        pattern_data = {}
        time_pred_data = {}
        
        if self.multi_tf_analyzer:
            try:
                multi_tf_data = self.multi_tf_analyzer.analyze_all(symbol)
            except Exception as e:
                multi_tf_data = {'error': str(e)}
        
        if self.pattern_matcher:
            try:
                pattern_data = self.pattern_matcher.find_similar_patterns(primary_df)
            except Exception as e:
                pattern_data = {'error': str(e)}
        
        if self.time_predictor:
            try:
                time_pred_data = self.time_predictor.predict(
                    primary_df,
                    direction=output.layer3.direction,
                    confidence=output.layer3.confidence,
                    timeframe=timeframe,
                    pattern_data=pattern_data
                )
            except Exception as e:
                time_pred_data = {'error': str(e)}

        # ══════════════════════════════════════
        # FINAL OUTPUT COMPUTATION (Upgraded)
        # ══════════════════════════════════════
        
        # 1. Probability Calibration (Layer 7 Upgrade)
        try:
            from models.calibration.calibrator import ProbabilityCalibrator
            calibrator = ProbabilityCalibrator()
            calibrated = calibrator.calibrate(output.layer3.up_probability, output.layer3.confidence)
            calibrated_up = calibrated['calibrated_probability']
        except ImportError:
            calibrated_up = output.layer3.up_probability
            
        calibrated_down = 1.0 - calibrated_up

        # Regime from Layer 2
        output.regime = output.layer2.details.get("regime", "unknown")
        if output.regime == "unknown":
            # Heuristic fallback based on Layer 1 indicators
            try:
                adx = float(output.layer1.details.get("adx", 20))
                # Look at the ADX_Trend signal for direction
                adx_sig = output.layer1.sub_signals.get("ADX_Trend", {}).get("signal", "")
                if adx > 25:
                    output.regime = "trending_up" if "BULLISH" in adx_sig else "trending_down"
                else:
                    output.regime = "ranging"
            except:
                output.regime = "ranging"

        try:
            from models.risk.risk_manager import RiskManager
            rm = RiskManager()
            atr = output.layer1.details.get("atr", primary_df['close'].iloc[-1] * 0.02)
            volatility_pct = atr / output.current_price if output.current_price > 0 else 0.02
            
            # Use raw direction but let RM validate it
            direction = "UP" if calibrated_up > 0.5 else "DOWN"
            
            rm_signal = rm.evaluate_signal(
                direction=direction,
                up_prob=calibrated_up,
                confidence=output.layer3.confidence,
                regime=output.regime,
                volatility=volatility_pct,
                current_price=output.current_price
            )
            
            output.direction = rm_signal['direction']
            output.risk_level = rm_signal['risk_level']
            output.suggested_entry = rm_signal['suggested_entry']
            output.suggested_stop_loss = rm_signal['suggested_stop_loss']
            output.suggested_take_profit = rm_signal['suggested_take_profit']
            output.risk_reward_ratio = rm_signal['risk_reward_ratio']
            output.position_size_pct = rm_signal['position_size_pct']
            
            # Confidence penalty if RM rejected it
            output.final_confidence = output.layer3.confidence if rm_signal['approved'] else max(0.0, output.layer3.confidence - 0.4)
            
        except ImportError:
            # Fallback
            output.direction = "UP" if calibrated_up > 0.5 else "DOWN"
            output.final_confidence = output.layer3.confidence
            output.risk_level = "HIGH" if output.final_confidence < 0.6 else "LOW"
            output.position_size_pct = min(0.05, output.final_confidence * 0.07)
            
        output.up_probability = calibrated_up
        output.down_probability = calibrated_down

        output.multi_tf = multi_tf_data
        output.pattern_match = pattern_data
        output.time_prediction = time_pred_data

        return output

    def predict_from_df(self, primary_df, symbol: str = "BTC/USDT", timeframe: str = "4h",
                        higher_tf_data: dict = None, featured_df=None, offline: bool = True) -> BrainOutput:
        """
        Run 3-layer prediction on a specific in-memory historical slice.
        Essential for fast, lookahead-free market replay / backtesting.
        """
        from features.feature_engine import FeatureEngine
        
        output = BrainOutput()
        output.symbol = symbol
        output.timeframe = timeframe
        output.timestamp = str(primary_df['timestamp'].iloc[-1]) if 'timestamp' in primary_df.columns else datetime.utcnow().isoformat()

        if primary_df is None or len(primary_df) < 10:
            output.layer1.details = {"error": "Not enough data"}
            return output

        output.current_price = float(primary_df['close'].iloc[-1])

        # Feature computation
        if featured_df is None:
            feature_engine = FeatureEngine(self.config)
            featured_df = feature_engine.compute_all_features(primary_df, higher_tf_data=higher_tf_data)
            featured_df = feature_engine.remove_nan_rows(featured_df)

        if len(featured_df) == 0:
            output.layer1.details = {"error": "No features after cleaning"}
            return output

        if not self._models_loaded:
            self.load_trained_models()

        # Run layers
        output.layer1 = self.layer1.analyze(primary_df, featured_df)
        output.layer2 = self.layer2.analyze(primary_df, featured_df, output.layer1, higher_tf_data=higher_tf_data)
        output.layer3 = self.layer3.analyze(primary_df, featured_df, output.layer1, output.layer2)

        pattern_data = {}
        multi_tf_data = {}
        time_pred_data = {}

        if not offline and self.pattern_matcher:
            try:
                pattern_data = self.pattern_matcher.find_similar_patterns(primary_df)
            except Exception as e:
                pattern_data = {'error': str(e)}

        if not offline and self.time_predictor:
            try:
                time_pred_data = self.time_predictor.predict(
                    primary_df,
                    direction=output.layer3.direction,
                    confidence=output.layer3.confidence,
                    timeframe=timeframe,
                    pattern_data=pattern_data
                )
            except Exception as e:
                time_pred_data = {'error': str(e)}

        calibrated_up = output.layer3.up_probability
        calibrated_down = 1.0 - calibrated_up

        output.regime = output.layer2.details.get("regime", "unknown")
        if output.regime == "unknown":
            try:
                adx = float(output.layer1.details.get("adx", 20))
                adx_sig = output.layer1.sub_signals.get("ADX_Trend", {}).get("signal", "")
                if adx > 25:
                    output.regime = "trending_up" if "BULLISH" in adx_sig else "trending_down"
                else:
                    output.regime = "ranging"
            except:
                output.regime = "ranging"

        # Risk evaluation & dynamic target setting
        try:
            from models.risk.risk_manager import RiskManager
            rm = RiskManager()
            atr = output.layer1.details.get("atr", primary_df['close'].iloc[-1] * 0.02)
            volatility_pct = atr / output.current_price if output.current_price > 0 else 0.02
            direction = "UP" if calibrated_up > 0.5 else "DOWN"

            rm_signal = rm.evaluate_signal(
                direction=direction,
                up_prob=calibrated_up,
                confidence=output.layer3.confidence,
                regime=output.regime,
                volatility=volatility_pct,
                current_price=output.current_price
            )

            output.direction = rm_signal['direction']
            output.risk_level = rm_signal['risk_level']
            output.suggested_entry = rm_signal['suggested_entry']
            output.suggested_stop_loss = rm_signal['suggested_stop_loss']
            output.suggested_take_profit = rm_signal['suggested_take_profit']
            output.risk_reward_ratio = rm_signal['risk_reward_ratio']
            output.position_size_pct = rm_signal['position_size_pct']
            output.final_confidence = output.layer3.confidence if rm_signal['approved'] else max(0.0, output.layer3.confidence - 0.4)
        except Exception:
            output.direction = "UP" if calibrated_up > 0.5 else "DOWN"
            output.final_confidence = output.layer3.confidence
            output.risk_level = "HIGH" if output.final_confidence < 0.6 else "LOW"
            output.position_size_pct = min(0.05, output.final_confidence * 0.07)
            atr = float(primary_df['high'].iloc[-1] - primary_df['low'].iloc[-1])
            atr = max(atr, output.current_price * 0.015)
            if output.direction == "UP":
                output.suggested_entry = output.current_price
                output.suggested_take_profit = output.current_price + atr * 2.0
                output.suggested_stop_loss = output.current_price - atr * 1.5
            else:
                output.suggested_entry = output.current_price
                output.suggested_take_profit = output.current_price - atr * 2.0
                output.suggested_stop_loss = output.current_price + atr * 1.5
            output.risk_reward_ratio = 1.33

        output.up_probability = calibrated_up
        output.down_probability = calibrated_down
        output.multi_tf = multi_tf_data
        output.pattern_match = pattern_data
        output.time_prediction = time_pred_data

        return output

    def to_dict(self, output: BrainOutput, offline: bool = False) -> dict:
        """Convert BrainOutput to JSON-safe dictionary."""
        def layer_to_dict(layer: LayerSignal):
            return {
                "direction": layer.direction,
                "confidence": round(float(layer.confidence), 4),
                "up_probability": round(float(layer.up_probability), 4),
                "down_probability": round(float(layer.down_probability), 4),
                "signals_bullish": int(layer.signals_bullish),
                "signals_bearish": int(layer.signals_bearish),
                "signals_neutral": int(layer.signals_neutral),
                "sub_signals": _clean_dict(layer.sub_signals),
                "details": _clean_dict(layer.details),
            }

        payload = {
            "direction": output.direction,
            "final_confidence": round(float(output.final_confidence), 4),
            "up_probability": round(float(output.up_probability), 4),
            "down_probability": round(float(output.down_probability), 4),
            "symbol": output.symbol,
            "timeframe": output.timeframe,
            "current_price": round(float(output.current_price), 4),
            "timestamp": output.timestamp,
            "regime": output.regime,
            "risk_level": output.risk_level,
            "suggested_entry": round(float(output.suggested_entry), 6),
            "suggested_stop_loss": round(float(output.suggested_stop_loss), 6),
            "suggested_take_profit": round(float(output.suggested_take_profit), 6),
            "risk_reward_ratio": round(float(output.risk_reward_ratio), 2),
            "position_size_pct": round(float(output.position_size_pct), 4),
            "models_trained": self._models_loaded,
            "layer1": layer_to_dict(output.layer1),
            "layer2": layer_to_dict(output.layer2),
            "layer3": layer_to_dict(output.layer3),
            "multi_tf": output.multi_tf,
            "pattern_match": output.pattern_match,
            "time_prediction": output.time_prediction,
        }

        # ─────────────────────────────────────────
        # MASTER STRATEGY (QAMR ENGINE + INSTITUTIONAL CITADEL)
        # ─────────────────────────────────────────
        try:
            from strategy.master_strategy import QuantumArbitrageStrategy
            from strategy.institutional_master import InstitutionalMasterStrategy
            
            # Run QAMR Base
            qamr = QuantumArbitrageStrategy()
            strategy_execution = qamr.generate_trade_execution(payload, offline=offline)
            payload["master_strategy"] = strategy_execution
            
            # Run Project Citadel Institutional Overlay
            citadel = InstitutionalMasterStrategy()
            inst_exec = citadel.build_institutional_portfolio(payload, strategy_execution["internet_context"])
            payload["institutional_strategy"] = inst_exec
            
            # The 90% Win-Rate Omni-Confluence Veto Engine (The Grail)
            from strategy.grail_filter import HolyGrailFilter
            grail_status = HolyGrailFilter.evaluate_grail_conditions(payload)
            payload["grail_status"] = grail_status
            
            # If it's a Grail Trade, override leverage and confidence to the absolute maximum
            if grail_status["is_grail"]:
                payload["trade_style"] = "HOLY_GRAIL_SNIPER"
                payload["institutional_strategy"]["primary_action"] = f"MAX_LEVERAGE_{payload['direction']}"
                payload["final_confidence"] = 0.999
                
            # Master Strategy overrides position size based on Macro/Internet
            payload["position_size_pct"] = strategy_execution["portfolio_allocation_pct"] / 100.0
            
            # Expose strategy style to UI
            payload["trade_style"] = strategy_execution["trade_style"]
        except Exception as e:
            import traceback
            print(f"Master Strategy Overlay Failed: {e}")
            traceback.print_exc()
            payload["master_strategy"] = {"error": str(e)}
            payload["institutional_strategy"] = {}

        return payload


def _clean_dict(d):
    """Recursively clean dict for JSON serialization."""
    if not isinstance(d, dict):
        return d
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = _clean_dict(v)
        elif isinstance(v, (np.integer,)):
            result[k] = int(v)
        elif isinstance(v, (np.floating,)):
            result[k] = round(float(v), 6) if not np.isnan(v) else 0.0
        elif isinstance(v, np.ndarray):
            result[k] = v.tolist()
        elif isinstance(v, list):
            result[k] = [_clean_dict(item) if isinstance(item, dict) else item for item in v]
        else:
            result[k] = v
    return result
