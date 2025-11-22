"""
analysis/metrics/classical.py
Standard template for classical technical indicators module
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Union, List

# ==================== MODULE CONFIG ====================
_MODULE_CONFIG = {
    "data_model": "pandas",      # pandas, numpy, polars
    "execution_type": "sync",    # sync, async
    "category": "technical"      # technical, regime, risk, etc.
}

# ==================== PURE FUNCTIONS ====================

# ==========================================================
# === Trend & Moving averages ==============================
# ==========================================================

# classical.py'de EMA fonksiyonunu güncelle
#def ema(data: pd.Series, period: int = 14, **kwargs) -> pd.Series:
#    return data.ewm(span=period, adjust=False).mean()  # ✅ Tüm seriyi döndürür


def ema(data: pd.Series, period: int = 14, **kwargs) -> pd.Series:
    """TÜM EMA SERİSİNİ döndürdüğünden emin ol"""
    print(f"📈 EMA called with data: len={len(data)}, type={type(data)}")
    result = data.ewm(span=period, adjust=False).mean()
    print(f"📈 EMA returning: len={len(result)}, type={type(result)}")
    return result

def sma(data: pd.Series, period: int = 14, **kwargs) -> pd.Series:
    """Simple Moving Average"""
    return data.rolling(window=period, min_periods=1).mean()

def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9, **kwargs) -> pd.DataFrame:
    """Moving Average Convergence Divergence"""
    ema_fast = data.ewm(span=fast, adjust=False).mean()
    ema_slow = data.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return pd.DataFrame({
        "macd_line": macd_line,
        "signal_line": signal_line,
        "histogram": hist
    })

def rsi(data: pd.Series, period: int = 14, **kwargs) -> pd.Series:
    """Relative Strength Index"""
    delta = data.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))

def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14, **kwargs) -> pd.Series:
    """Average Directional Index"""
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    atr = tr.rolling(period, min_periods=1).mean()
    plus_di = 100 * (plus_dm.rolling(period, min_periods=1).mean() / (atr + 1e-10))
    minus_di = 100 * (minus_dm.rolling(period, min_periods=1).mean() / (atr + 1e-10))
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)) * 100
    return dx.rolling(period, min_periods=1).mean()

def stochastic_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14, **kwargs) -> pd.Series:
    """Stochastic Oscillator"""
    lowest_low = low.rolling(window=period).min()
    highest_high = high.rolling(window=period).max()
    return ((close - lowest_low) / (highest_high - lowest_low + 1e-10)) * 100

def roc(data: pd.Series, period: int = 1, **kwargs) -> pd.Series:
    """Rate of Change - Price change percentage over period"""
    return data.pct_change(periods=period) * 100

# ==========================================================
# === Volatility metrics ===================================
# ==========================================================

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14, **kwargs) -> pd.Series:
    """Average True Range"""
    tr = pd.concat([
        (high - low),
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=1).mean()

def bollinger_bands(data: pd.Series, period: int = 20, std_factor: float = 2.0, **kwargs) -> pd.DataFrame:
    """Bollinger Bands"""
    sma_val = data.rolling(window=period, min_periods=1).mean()
    std = data.rolling(window=period, min_periods=1).std()
    upper = sma_val + std_factor * std
    lower = sma_val - std_factor * std
    bandwidth = (upper - lower) / (sma_val + 1e-10)
    return pd.DataFrame({
        "middle": sma_val,
        "upper": upper,
        "lower": lower,
        "bandwidth": bandwidth
    })

def historical_volatility(data: pd.Series, window: int = 30, annualize: bool = True, **kwargs) -> pd.Series:
    """Annualized Historical Volatility"""
    log_returns = np.log(data / data.shift(1))
    volatility = log_returns.rolling(window=window).std()
    if annualize:
        volatility = volatility * np.sqrt(252)  # Annualization
    return volatility

# ==========================================================
# === Risk metrics =========================================
# ==========================================================

def value_at_risk(data: pd.Series, confidence: float = 0.95, **kwargs) -> pd.Series:
    """Value at Risk (percentile-based)"""
    returns = data.pct_change().dropna()
    if len(returns) == 0:
        return pd.Series([np.nan], index=data.index[-1:], name="value_at_risk")
    val = np.percentile(returns, (1 - confidence) * 100)
    return pd.Series([val], index=data.index[-1:], name="value_at_risk")

def conditional_value_at_risk(data: pd.Series, confidence: float = 0.95, **kwargs) -> pd.Series:
    """Conditional Value at Risk (expected shortfall)"""
    returns = data.pct_change().dropna()
    if len(returns) == 0:
        return pd.Series([np.nan], index=data.index[-1:], name="conditional_value_at_risk")
    var_val = np.percentile(returns, (1 - confidence) * 100)
    cvar = returns[returns <= var_val].mean()
    return pd.Series([cvar], index=data.index[-1:], name="conditional_value_at_risk")

def max_drawdown(data: pd.Series, **kwargs) -> pd.Series:
    """Maximum Drawdown"""
    roll_max = data.cummax()
    drawdown = (data - roll_max) / (roll_max + 1e-10)
    return pd.Series([drawdown.min()], index=data.index[-1:], name="max_drawdown")

# ==========================================================
# === Open Interest & Market structure =====================
# ==========================================================

def oi_growth_rate(oi_series: pd.Series, period: int = 7, **kwargs) -> pd.Series:
    """Open Interest Growth Rate"""
    return oi_series.pct_change(periods=period).fillna(0)

def oi_price_correlation(oi_series: pd.Series, price_series: pd.Series, window: int = 14, **kwargs) -> pd.Series:
    """Rolling correlation between Open Interest and Price"""
    return oi_series.rolling(window=window, min_periods=1).corr(price_series)

# ==========================================================
# === Correlation metrics ==================================
# ==========================================================

def spearman_corr(series_x: pd.Series, series_y: pd.Series, **kwargs) -> pd.Series:
    """Spearman rank correlation coefficient"""
    aligned_x, aligned_y = series_x.align(series_y, join='inner')
    
    if len(aligned_x) > 0:
        window = min(20, len(aligned_x))
        corr = aligned_x.rolling(window=window).corr(aligned_y)
        return corr
    else:
        return pd.Series([np.nan], index=series_x.index[:1])

def cross_correlation(series_x: pd.Series, series_y: pd.Series, max_lag: int = 10, **kwargs) -> pd.Series:
    """Cross-correlation between two series with various lags"""
    from scipy.signal import correlate

    aligned_x, aligned_y = series_x.align(series_y, join='inner')

    if len(aligned_x) == 0:
        return pd.Series([], dtype=float)

    # Normalize series
    x_normalized = (aligned_x - aligned_x.mean()) / (aligned_x.std() + 1e-10)
    y_normalized = (aligned_y - aligned_y.mean()) / (aligned_y.std() + 1e-10)

    # Calculate cross-correlation
    correlation = correlate(x_normalized, y_normalized, mode='full')
    lags = np.arange(-len(aligned_x) + 1, len(aligned_x))

    valid_indices = (lags >= -max_lag) & (lags <= max_lag)
    valid_correlations = correlation[valid_indices]
    valid_lags = lags[valid_indices]

    if len(valid_correlations) > 0:
        max_corr_idx = np.argmax(np.abs(valid_correlations))
        best_lag = valid_lags[max_corr_idx]
        best_corr = valid_correlations[max_corr_idx]

        return pd.Series({
            'best_correlation': best_corr,
            'best_lag': best_lag
        })
    else:
        return pd.Series([np.nan], index=['cross_correlation'])

# ==========================================================
# === Futures metrics ======================================
# ==========================================================

def futures_roc(futures_series: pd.Series, period: int = 1, **kwargs) -> pd.Series:
    """Futures Price Change - Simple roc for futures contracts"""
    return futures_series.pct_change(periods=period) * 100

# ==================== MODULE REGISTRY ====================
_METRICS = {
    "adx": adx,
    "atr": atr,
    "bollinger_bands": bollinger_bands,
    "cross_correlation": cross_correlation,
    "conditional_value_at_risk": conditional_value_at_risk,
    "ema": ema,
    "futures_roc": futures_roc,
    "historical_volatility": historical_volatility,
    "macd": macd,
    "max_drawdown": max_drawdown,
    "oi_growth_rate": oi_growth_rate,
    "oi_price_correlation": oi_price_correlation,
    "roc": roc,
    "rsi": rsi,
    "sma": sma,
    "spearman_corr": spearman_corr,
    "stochastic_oscillator": stochastic_oscillator,
    "value_at_risk": value_at_risk
}

def get_metrics() -> List[str]:
    """Composite engine için metric listesi"""
    return list(_METRICS.keys())

def get_function(metric_name: str):
    """Composite engine için fonksiyon döndür"""
    return _METRICS.get(metric_name)

def get_module_config() -> Dict[str, Any]:
    """Module-level configuration"""
    return _MODULE_CONFIG.copy()