# 📁 analysis/metrics/risk.py 
"""
module_registry.yaml'da risk metriklerini güncelleyin:

risk:
  liquidation_clusters: "Basitleştirilmiş liquidation cluster tespiti, liquidation_risk"
  cascade_risk: "Basitleştirilmiş cascade risk ölçümü, liquidation_risk"
  sr_impact: "Basitleştirilmiş support/resistance impact, support_resistance"
  forced_selling: "Basitleştirilmiş forced selling tespiti, forced_selling"
  liquidity_gaps: "Basitleştirilmiş liquidity gap tespiti, liquidity_analysis"
  futures_liq_risk: "Basitleştirilmiş futures liquidation risk, liquidation_risk"
  liquidation_cascade: "Basitleştirilmiş liquidation cascade tespiti, liquidation_risk"
  market_stress: "Basitleştirilmiş market stress göstergesi, market_stress"
"""
# ============================================================
# metrics/risk.py  (PURE ATOMIC MODULE - STANDARD TEMPLATE)
# ============================================================

import numpy as np
import pandas as pd
from typing import Dict, Any, List

# ==================== MODULE CONFIG ====================
_MODULE_CONFIG = {
    "data_model": "pandas",
    "execution_type": "sync",
    "category": "risk"
}

# ==================== PURE FUNCTIONS ====================
def liquidation_clusters(price_series: pd.Series, window: int = 20) -> pd.Series:
    """Pure liquidation cluster detection"""
    if len(price_series) < window:
        return pd.Series([0.0] * len(price_series), index=price_series.index)

    returns = price_series.pct_change().fillna(0)
    volatility = returns.rolling(window=window).std().fillna(0.01)

    large_drops = (returns < -0.03).astype(float)
    vol_spike = (volatility > volatility.quantile(0.8)).astype(float)

    cluster_intensity = (large_drops * 0.7 + vol_spike * 0.3) * np.abs(returns)
    return pd.Series(np.clip(cluster_intensity, 0, 1), index=price_series.index)


def cascade_risk(price_series: pd.Series, lookback: int = 30) -> pd.Series:
    """Pure cascade risk indicator"""
    if len(price_series) < lookback:
        return pd.Series([0.0] * len(price_series), index=price_series.index)

    rolling_max = price_series.rolling(window=lookback, min_periods=1).max()
    drawdown = (price_series / rolling_max - 1).fillna(0)

    drawdown_change = drawdown.diff().fillna(0)
    volatility = price_series.pct_change().rolling(window=10).std().fillna(0.01)

    cascade_prob = np.clip(np.abs(drawdown) * np.abs(drawdown_change) * volatility * 100, 0, 1)
    return cascade_prob


def sr_impact(price_series: pd.Series, window: int = 20) -> pd.Series:
    """Pure support/resistance impact"""
    if len(price_series) < window:
        return pd.Series(index=price_series.index, dtype=float)

    resistance = price_series.rolling(window=window).max()
    support = price_series.rolling(window=window).min()

    dist_to_resistance = (resistance - price_series) / price_series
    dist_to_support = (price_series - support) / price_series

    proximity_score = 1 / (np.abs(dist_to_resistance) + np.abs(dist_to_support) + 1e-6)
    return proximity_score.fillna(0.0)


def forced_selling(price_series: pd.Series, window: int = 10) -> pd.Series:
    """Pure forced selling detector"""
    if len(price_series) < window:
        return pd.Series(index=price_series.index, dtype=float)

    returns = price_series.pct_change().fillna(0)
    volatility = returns.rolling(window=window).std()
    negative_returns = (returns < 0).astype(float)

    forced_selling_pressure = negative_returns * volatility * np.abs(returns)
    return forced_selling_pressure.fillna(0.0)


def liquidity_gaps(price_series: pd.Series) -> pd.Series:
    """Pure liquidity gaps"""
    returns = price_series.pct_change().fillna(0)
    large_jumps = (np.abs(returns) > 0.02).astype(float)
    return large_jumps.fillna(0.0)


def futures_liq_risk(price_series: pd.Series) -> pd.Series:
    """Pure futures liquidation risk"""
    returns = price_series.pct_change().fillna(0)

    short_vol = returns.rolling(window=5).std()
    long_vol = returns.rolling(window=20).std()

    vol_ratio = short_vol / (long_vol + 1e-6)
    return np.clip(vol_ratio - 1, 0, 2).fillna(0.0)


def liquidation_cascade(price_series: pd.Series) -> pd.Series:
    """Pure liquidation cascade intensity"""
    returns = price_series.pct_change().fillna(0)

    momentum_5 = returns.rolling(window=5).mean()
    momentum_10 = returns.rolling(window=10).mean()
    momentum_20 = returns.rolling(window=20).mean()

    all_negative = (momentum_5 < 0) & (momentum_10 < 0) & (momentum_20 < 0)
    cascade_intensity = all_negative.astype(float) * np.abs(momentum_5)

    return cascade_intensity.fillna(0.0)


def market_stress(price_series: pd.Series) -> pd.Series:
    """Pure market stress"""
    returns = price_series.pct_change().fillna(0)

    vol_5 = returns.rolling(window=5).std()
    vol_20 = returns.rolling(window=20).std()

    vol_spike = vol_5 / (vol_20 + 1e-6)
    large_moves = (np.abs(returns) > 0.015).astype(float)

    stress_index = np.clip(vol_spike * large_moves, 0, 3)
    return stress_index.fillna(0.0)


# ==================== MODULE REGISTRY ====================
_METRICS: Dict[str, Any] = {
    "liquidation_clusters": liquidation_clusters,
    "cascade_risk": cascade_risk,
    "sr_impact": sr_impact,
    "forced_selling": forced_selling,
    "liquidity_gaps": liquidity_gaps,
    "futures_liq_risk": futures_liq_risk,
    "liquidation_cascade": liquidation_cascade,
    "market_stress": market_stress,
}

def get_metrics() -> List[str]:
    return list(_METRICS.keys())

def get_function(metric_name: str):
    return _METRICS.get(metric_name)

def get_module_config() -> Dict[str, Any]:
    return _MODULE_CONFIG.copy()
