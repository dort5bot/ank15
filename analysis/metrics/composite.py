# -*- coding: utf-8 -*-
"""
analysis/metrics/composite.py
Unified Composite Engine (v2025.11) - OPTIMIZED VERSION
- Full module integration (atomic, composite, macro)
- Async + Sync hybrid execution
- Memory-safe (WeakRefs, shared executor, embedded config)
- YAML-free embedded configuration system
"""

import asyncio
import atexit
import concurrent.futures
import functools
import importlib
import inspect
import logging
import os
import numpy as np
np.seterr(invalid="ignore", divide="ignore")

import pandas as pd

from collections import OrderedDict
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Union
from weakref import WeakValueDictionary

try:
    import polars as pl
except ImportError:
    pl = None

logger = logging.getLogger("metrics.composite")

# ==========================================================
# === EMBEDDED CONFIGURATION ===============================
# ==========================================================

ATOMIC_INDICATORS = {
    "classical": [
        "ema", "sma", "macd", "rsi", "adx", "stochastic_oscillator", 
        "roc", "atr", "bollinger_bands", "value_at_risk",
        "conditional_value_at_risk", "max_drawdown", "oi_growth_rate",
        "oi_price_correlation", "spearman_corr", "cross_correlation", "futures_roc"
    ],
    "advanced": [
        "kalman_filter_trend", "wavelet_transform", "hilbert_transform_slope",
        "hilbert_transform_amplitude", "fractal_dimension_index_fdi", 
        "shannon_entropy", "permutation_entropy", "sample_entropy",
        "granger_causality", "phase_shift_index"
    ],
    "volatility": [
        "historical_volatility", "bollinger_width", "garch_1_1", "hurst_exponent",
        "entropy_index", "variance_ratio_test", "range_expansion_index"
    ],
    "sentiment": [
        "funding_rate", "funding_premium", "oi_trend"
    ],
    "microstructure": [
        "ofi", "cvd", "microprice_deviation", "market_impact", "depth_elasticity",
        "taker_dominance_ratio", "liquidity_density"
    ],
    "onchain": [
        "etf_net_flow", "exchange_netflow", "stablecoin_flow", "net_realized_pl",
        "realized_cap", "nupl", "exchange_whale_ratio", "mvrv_zscore", "sopr"
    ],
    "regime": [
        "advance_decline_line", "volume_leadership", "performance_dispersion"
    ],
    "risk": [
        "liquidation_clusters", "cascade_risk", "sr_impact", "forced_selling",
        "liquidity_gaps", "futures_liq_risk", "liquidation_cascade", "market_stress"
    ]
}
    
METRIC_COMPOSITES = {
    "trend_momentum_composite": {
        "depends_on": ["ema", "macd", "rsi", "adx", "roc", "stochastic_oscillator"],
        "formula": "0.25*ema + 0.25*macd + 0.20*rsi + 0.10*adx + 0.10*roc + 0.10*stochastic_oscillator",
        "category": "trend_momentum",
        "interpretation": "Momentum + trend strength combined."
    },
    "volatility_composite": {
        "depends_on": ["atr", "historical_volatility", "garch_1_1", "hurst_exponent", "entropy_index"],
        "formula": "0.30*atr + 0.25*historical_volatility + 0.20*garch_1_1 + 0.15*hurst_exponent + 0.10*entropy_index",
        "category": "volatility",
        "interpretation": "Multi-dimensional volatility composite."
    },
    "volatility_momentum_composite": {
        "depends_on": ["roc", "adx", "historical_volatility", "atr"],
        "formula": "0.35*roc + 0.25*adx - 0.25*historical_volatility - 0.15*atr",
        "category": "volatility_momentum",
        "interpretation": "Momentum hızını volatiliteye göre normalize eder."
    },
    "regime_composite": {
        "depends_on": ["advance_decline_line", "volume_leadership", "performance_dispersion"],
        "formula": "0.45*tanh(advance_decline_line/1000) + 0.35*tanh(volume_leadership*10) + 0.20*tanh(performance_dispersion*100)",
        "category": "regime",
        "interpretation": "Breadth + leadership + dispersion composite."
    },
    "risk_composite": {
        "depends_on": ["liquidation_clusters", "liquidity_gaps", "cascade_risk", "forced_selling"],
        "formula": "0.3*liquidation_clusters + 0.25*liquidity_gaps + 0.25*cascade_risk + 0.20*forced_selling",
        "category": "risk",
        "interpretation": "Systemic risk composite from derivatives & liquidity."
    },
    "liquidity_composite": {
        "depends_on": ["market_impact", "depth_elasticity", "liquidity_density"],
        "formula": "0.4*market_impact + 0.3*depth_elasticity + 0.3*liquidity_density",
        "category": "microstructure",
        "interpretation": "Liquidity conditions composite."
    },
    "liquidity_risk_composite": {
        "depends_on": ["liquidity_density", "market_impact", "liquidity_gaps", "cascade_risk"],
        "formula": "0.3*liquidity_density - 0.3*market_impact - 0.2*liquidity_gaps - 0.2*cascade_risk",
        "category": "liquidity_risk",
        "interpretation": "Derinlik ve akış verilerini kullanarak piyasa kırılganlığını ölçer."
    },
    "entropy_fractal_composite": {
        "depends_on": ["entropy_index", "fractal_dimension_index_fdi", "hurst_exponent", "variance_ratio_test"],
        "formula": "0.35*entropy_index + 0.25*fractal_dimension_index_fdi - 0.25*hurst_exponent - 0.15*variance_ratio_test",
        "category": "complexity",
        "interpretation": "Fiyat serisinin kaotiklik, fraktal yapı ve rassallık seviyesini ölçer."
    },
    "order_flow_stress_composite": {
        "depends_on": ["ofi", "cvd", "taker_dominance_ratio", "microprice_deviation"],
        "formula": "0.35*ofi + 0.25*cvd + 0.25*taker_dominance_ratio + 0.15*microprice_deviation",
        "category": "order_flow",
        "interpretation": "Alıcı-satıcı baskısı ve mikro fiyat dengesizliğini birleştirir."
    },
    "flow_dynamics_composite": {
        "depends_on": ["etf_net_flow", "exchange_netflow", "stablecoin_flow"],
        "formula": "0.4*etf_net_flow - 0.3*exchange_netflow + 0.3*stablecoin_flow",
        "category": "capital_flow",
        "interpretation": "ETF, borsa ve stablecoin akışlarını netleştirir."
    },
    "sentiment_composite": {
        "depends_on": ["funding_rate", "funding_premium", "oi_trend"],
        "formula": "0.35*funding_rate + 0.25*funding_premium + 0.40*oi_trend",
        "category": "sentiment",
        "interpretation": "Vadeli işlem fonlama koşulları ve OI eğilimlerini tek skor altında birleştirir."
    }
}

MACRO_COMPOSITES = {
    "core_macro": {
        "depends_on": ["trend_momentum_composite", "volatility_composite", "regime_composite", "risk_composite"],
        "formula": "0.35*trend_momentum_composite + 0.25*volatility_composite + 0.25*regime_composite + 0.15*risk_composite",
        "category": "market_core",
        "output": "core_score",
        "interpretation": "Aggregated core market composite."
    },
    "comprehensive_macro": {
        "depends_on": ["trend_momentum_composite", "volatility_composite", "regime_composite", "risk_composite", "liquidity_composite"],
        "formula": "0.27*trend_momentum_composite + 0.20*volatility_composite + 0.20*regime_composite + 0.18*risk_composite + 0.15*liquidity_composite",
        "category": "market_extended",
        "output": "comprehensive_score",
        "interpretation": "Expanded market health composite."
    },
    "complexity_macro": {
        "depends_on": ["entropy_fractal_composite", "volatility_composite"],
        "formula": "0.6*entropy_fractal_composite + 0.4*volatility_composite",
        "category": "complexity_macro",
        "output": "complexity_score",
        "interpretation": "Volatilite ve entropi-fraktal birleşimini kullanarak piyasa düzenliliğini veya kaotikliği ölçer."
    },
    "market_sentiment_macro": {
        "depends_on": ["sentiment_composite", "flow_dynamics_composite"],
        "formula": "0.55*sentiment_composite + 0.45*flow_dynamics_composite",
        "category": "sentiment_macro",
        "output": "sentiment_score",
        "interpretation": "Fonlama, OI eğilimi ve sermaye akışlarını birleştirerek piyasa duyarlılığının yönünü ölçer."
    },
    "microstructure_macro": {
        "depends_on": ["liquidity_composite", "liquidity_risk_composite", "order_flow_stress_composite"],
        "formula": "0.4*liquidity_composite + 0.35*liquidity_risk_composite + 0.25*order_flow_stress_composite",
        "category": "microstructure_macro",
        "output": "microstructure_score",
        "interpretation": "Likidite, emir baskısı ve mikro risk göstergelerini tek skor olarak birleştirir."
    }
}

# bunlar binanceApi ile alınamaz
FALLBACK_METRICS = [
    'advance_decline_line','cascade_risk','etf_net_flow','exchange_netflow',
    'exchange_whale_ratio','market_stress','mvrv_zscore','net_realized_pl',
    'nupl','performance_dispersion','realized_cap','sopr','sr_impact',
    'stablecoin_flow','volume_leadership','Z'
]


MICROSTRUCTURE_METRICS = {
    "ofi", "cvd", "microprice_deviation", "market_impact",
    "depth_elasticity", "taker_dominance_ratio", "liquidity_density"
}

MICROSTRUCTURE_PARAMS = {
    "ofi": ["bid_price", "bid_size", "ask_price", "ask_size"],
    "cvd": ["buy_volume", "sell_volume"],
    "microprice_deviation": ["best_bid", "best_ask", "bid_size", "ask_size"],
    "market_impact": ["trade_volume", "price_series"],
    "depth_elasticity": ["depth_price", "depth_volume"],
    "taker_dominance_ratio": ["taker_buy_volume", "taker_sell_volume"],
    "liquidity_density": ["depth_volume"]
}

# ==========================================================
# === UTILITY FUNCTIONS ====================================
# ==========================================================

def to_numpy(x: Any) -> np.ndarray:
    """Convert any data type to numpy array."""
    if x is None:
        return np.array([], dtype=float)
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, (pd.Series, list, tuple)):
        return np.asarray(x, dtype=float)
    if pl and isinstance(x, pl.Series):
        return x.to_numpy(zero_copy_only=True)
    try:
        return np.asarray(x, dtype=float)
    except Exception:
        return np.array([], dtype=float)

def extract_scalar(value: Any) -> float:
    """Extract scalar value safely from various data types."""
    try:
        if isinstance(value, pd.DataFrame):
            if not value.empty:
                return float(value.iloc[:, -1].iloc[-1])
        elif isinstance(value, pd.Series):
            if len(value) > 0:
                return float(value.iloc[-1])
        elif isinstance(value, np.ndarray):
            if value.size > 0:
                return float(value.flat[-1])
        elif pl and isinstance(value, pl.Series):
            return float(value[-1])
        elif np.isscalar(value):
            return float(value)
    except Exception as e:
        logger.debug(f"extract_scalar error: {e}")
    return np.nan

def normalize_score(metric_name: str, score: float) -> float:
    """Intelligent normalization based on metric type."""
    if np.isnan(score):
        return 0.0
    name = metric_name.lower()
    
    # Volatility metrics
    if "garch" in name:
        return np.tanh(score * 20.0)
    elif "atr" in name:
        return np.tanh(score / 1000.0)
    elif "historical_volatility" in name:
        return np.tanh(score * 10.0)
    elif "hurst" in name:
        return (score - 0.5) * 2.0
    elif "entropy" in name:
        return np.tanh(score / 5.0)
    
    # Classical metrics
    elif any(key in name for key in ["ema", "sma", "price"]):
        return np.tanh(score / 1000.0)
    elif "macd" in name:
        return np.tanh(score / 50.0)
    elif "rsi" in name:
        return (score - 50.0) / 50.0
    elif "stochastic" in name:
        return (score - 50.0) / 50.0
    elif "adx" in name:
        return score / 100.0
    elif "roc" in name:
        return np.tanh(score / 10.0)
    else:
        return np.tanh(score / 50.0)

def create_fallback_metric(name: str) -> Callable:
    """Create fallback function for missing metrics."""
    def fallback(*args, **kwargs):
        logger.warning(f"Fallback metric used: {name}")
        return 0.0
    return fallback

# ==========================================================
# === CENTRALIZED CACHE MANAGER ============================
# ==========================================================

class CacheManager:
    """Centralized cache management - thread-safe & memory-efficient."""
    
    def __init__(self):
        self.metric_cache = WeakValueDictionary()
        self.module_cache = {}
        self.formula_cache = lru_cache(maxsize=256)(lambda f: compile(f, "<formula>", "eval"))
        self._initialize_fallbacks()
    
    def _initialize_fallbacks(self):
        """Initialize fallback metrics."""
        for metric in FALLBACK_METRICS:
            if metric not in self.metric_cache:
                self.metric_cache[metric] = create_fallback_metric(metric)
                logger.debug(f"Fallback metric registered: {metric}")

# ==========================================================
# === STANDARDIZATION ENGINE ===============================
# ==========================================================

class StandardizationEngine:
    """Input/Output standardization and data conversion."""
    
    @staticmethod
    def standardize_input(data: Any, expected_type: str = "pandas", fillna: bool = True) -> Any:
        """Unified input standardization."""
        if isinstance(data, dict):
            return {k: StandardizationEngine._convert_data(v, expected_type) for k, v in data.items()}
        else:
            standardized = StandardizationEngine._convert_data(data, expected_type)
        
        if fillna:
            standardized = StandardizationEngine._fill_missing(standardized)
        return standardized
    
    @staticmethod
    def standardize_output(data: Any, expected_type: str = "pandas") -> Any:
        """Unified output standardization."""
        if isinstance(data, (pd.Series, pd.DataFrame)):
            return data
        if isinstance(data, np.ndarray):
            return pd.Series(data)
        if pl and isinstance(data, pl.Series):
            return data.to_pandas()
        return pd.Series([data])
    
    @staticmethod
    def _convert_data(data: Any, target_type: str) -> Any:
        """Data type conversion."""
        if target_type == "pandas":
            return StandardizationEngine._as_pandas(data)
        elif target_type == "numpy":
            return StandardizationEngine._as_numpy(data)
        elif target_type == "polars":
            return StandardizationEngine._as_polars(data)
        return data
    
    @staticmethod
    def _as_pandas(data: Any) -> Any:
        if isinstance(data, (pd.Series, pd.DataFrame)):
            return data
        elif isinstance(data, np.ndarray):
            return pd.Series(data.flatten()) if data.ndim > 1 else pd.Series(data)
        elif isinstance(data, list):
            return pd.Series(data)
        elif pl and isinstance(data, pl.Series):
            return data.to_pandas()
        return data
    
    @staticmethod
    def _as_numpy(data: Any) -> np.ndarray:
        if isinstance(data, np.ndarray):
            return data
        if isinstance(data, (pd.Series, list, tuple)):
            return np.asarray(data, dtype=float)
        if pl and isinstance(data, pl.Series):
            return data.to_numpy()
        return np.asarray([data], dtype=float)
    
    @staticmethod
    def _as_polars(data: Any) -> Any:
        if pl is None:
            raise ImportError("polars not installed")
        if isinstance(data, pl.Series):
            return data
        if isinstance(data, pd.Series):
            return pl.Series(data)
        if isinstance(data, np.ndarray):
            return pl.Series(data.tolist())
        return pl.Series([data])
    
    @staticmethod
    def _fill_missing(data: Any) -> Any:
        """Enhanced NaN handling."""
        if isinstance(data, dict):
            return {k: StandardizationEngine._ffill_series(v) for k, v in data.items()}
        else:
            return StandardizationEngine._ffill_series(data)
    
    @staticmethod
    def _ffill_series(series: Any) -> Any:
        """Forward fill with type checking."""
        if series is None or isinstance(series, dict):
            return series
            
        if hasattr(series, 'ffill'):  # Pandas
            return series.ffill().bfill()
        elif hasattr(series, 'fill_null'):  # Polars
            return series.fill_null(strategy="forward").fill_null(strategy="backward")
        elif isinstance(series, np.ndarray):
            try:
                mask = np.isnan(series)
                indices = np.where(~mask, np.arange(len(series)), 0)
                np.maximum.accumulate(indices, out=indices)
                return series[indices]
            except (TypeError, ValueError):
                return series
        else:
            return series

def metric_standard(input_type: str = "pandas", output_type: str = "pandas", 
                   min_periods: int = 1, fillna: bool = True) -> Callable:
    """Decorator for metric standardization."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(data: Any, *args, **kwargs) -> Any:
            std_engine = StandardizationEngine()
            standardized_data = std_engine.standardize_input(data, input_type, fillna)
            
            if hasattr(standardized_data, '__len__') and len(standardized_data) < min_periods:
                return np.nan
            
            result = func(standardized_data, *args, **kwargs)
            return std_engine.standardize_output(result, output_type)
        return wrapper
    return decorator

# ==========================================================
# === COMPOSITE ENGINE =====================================
# ==========================================================

class CompositeEngine:
    """Main composite calculation engine."""
    
    _executor = None
    
    def __init__(self, base_package: str = "analysis.metrics"):
        self.base_package = base_package
        self.standardizer = StandardizationEngine()
        self.cache = CacheManager()
        
        # Embedded configuration
        self.atomic_indicators = ATOMIC_INDICATORS
        self.metric_composites = METRIC_COMPOSITES
        self.macro_composites = MACRO_COMPOSITES
        
        # Metric mapping and runtime state
        self.metric_to_module = self._build_metric_map()
        self.runtime_results: OrderedDict[str, float] = OrderedDict()
        self.max_runtime_results = 1024

    def _build_metric_map(self) -> Dict[str, str]:
        """Build reverse lookup: metric_name -> module_name."""
        mapping = {}
        for module_name, metrics in self.atomic_indicators.items():
            for metric in metrics:
                mapping[metric] = module_name
        return mapping

    @staticmethod
    def create_constant_func(value: float) -> Callable:
        """GC-safe constant function factory."""
        def const_func(*_) -> float:
            return value
        return const_func

    def _set_runtime(self, key: str, value: float) -> None:
        """Store result in runtime cache."""
        if len(self.runtime_results) >= self.max_runtime_results:
            self.runtime_results.popitem(last=False)
        self.runtime_results[key] = value
        self.cache.metric_cache[key] = self.create_constant_func(value)

    # ------------------------------------------------------
    # --- Executor Management ------------------------------
    # ------------------------------------------------------
    
    @classmethod
    def _get_executor(cls) -> concurrent.futures.ThreadPoolExecutor:
        if cls._executor is None:
            cls._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=os.cpu_count() or 4
            )
            atexit.register(cls._executor.shutdown, wait=False)
        return cls._executor

    # ------------------------------------------------------
    # --- Metric Resolution --------------------------------
    # ------------------------------------------------------
    
    def _find_raw_metric(self, name: str) -> Optional[Callable]:
        """Find metric function in modules."""
        module_name = self.metric_to_module.get(name)
        if not module_name:
            return None
        
        try:
            module = self.cache.module_cache.get(module_name)
            if module is None:
                module = importlib.import_module(f"{self.base_package}.{module_name}")
                self.cache.module_cache[module_name] = module
            return getattr(module, name, None)
        except ImportError:
            return None

    def resolve_metric(self, name: str) -> Optional[Callable]:
        """Resolve metric function with caching."""
        # 1. Runtime cache
        if name in self.runtime_results:
            return self.create_constant_func(self.runtime_results[name])
            
        # 2. Metric cache
        if name in self.cache.metric_cache:
            return self.cache.metric_cache[name]
            
        # 3. Raw function lookup
        raw_func = self._find_raw_metric(name)
        if raw_func:
            func = self._create_standardized_metric(raw_func, name)
            self.cache.metric_cache[name] = func
            return func
        
        # 4. Optimized module lookup
        module_name = self.metric_to_module.get(name)
        if module_name:
            try:
                module = self.cache.module_cache.get(module_name)
                if module is None:
                    module = importlib.import_module(f"{self.base_package}.{module_name}")
                    self.cache.module_cache[module_name] = module
                
                func = getattr(module, name, None)
                if callable(func):
                    self.cache.metric_cache[name] = func
                    return func
            except ImportError as e:
                logger.debug(f"Module {module_name} not found for {name}: {e}")
        
        # 5. Fallback
        if name in FALLBACK_METRICS:
            fallback_func = create_fallback_metric(name)
            self.cache.metric_cache[name] = fallback_func
            logger.warning(f"Using fallback for missing metric: {name}")
            return fallback_func
        
        logger.warning(f"Metric not found: {name}")
        return None
  
    def _create_standardized_metric(self, raw_func: Callable, metric_name: str) -> Callable:
            """Create standardized version of raw metric function."""
            # MİKRO YAPI METRİKLERİ İÇİN STANDARDİZASYON YAPMA - doğrudan dön
            if metric_name in MICROSTRUCTURE_METRICS:
                return raw_func
            
            def standardized_metric(data: Any, *args, **kwargs) -> Any:
                prepared_data = self.standardizer.standardize_input(data, expected_type="pandas")

                try:
                    # Align OHLC data
                    if isinstance(prepared_data, dict) and all(k in prepared_data for k in ["high", "low", "close"]):
                        for k in ["high", "low", "close"]:
                            prepared_data[k] = pd.Series(prepared_data[k]).reset_index(drop=True)

                    # Check function signature
                    sig = inspect.signature(raw_func)
                    param_count = len(sig.parameters)

                    # OHLC metrics
                    if isinstance(prepared_data, dict) and all(k in prepared_data for k in ["high", "low", "close"]):
                        if param_count >= 3:
                            result = raw_func(
                                prepared_data["high"],
                                prepared_data["low"],
                                prepared_data["close"],
                                *args, **kwargs
                            )
                        else:
                            result = raw_func(prepared_data["close"], *args, **kwargs)

                    # Dict type metrics
                    elif isinstance(prepared_data, dict):
                        result = raw_func(prepared_data, *args, **kwargs)

                    # Simple series
                    else:
                        result = raw_func(prepared_data, *args, **kwargs)

                    return self.standardizer.standardize_output(result, expected_type="pandas")

                except Exception as e:
                    logger.error(f"Metric {metric_name} failed: {e}")
                    return np.nan

            return standardized_metric
        
    
    
    # ------------------------------------------------------
    # --- Input Preparation --------------------------------
    # ------------------------------------------------------
        
    def prepare_input(self, data: Any, metric_name: str) -> Any:
        """Prepare input data for metric calculation - OPTIMIZED VERSION"""
        name = metric_name.lower()
        
        
        # Mevcut OHLC dict ise direkt dön
        if isinstance(data, dict) and all(k in data for k in ["high", "low", "close"]):
            return data
        
        arr = to_numpy(data)
        
        # OHLC metrikleri için sentetik data oluştur
        if name in ["adx", "atr", "stochastic_oscillator", "bollinger_bands"]:
            if len(arr) == 0:
                return {"high": arr, "low": arr, "close": arr}
            
            volatility = np.std(arr) * 0.01 if len(arr) > 1 else 0.002
            return {
                "high": arr * (1 + volatility),
                "low": arr * (1 - volatility), 
                "close": arr
            }
        
        return arr
        
 
    # ------------------------------------------------------
    # --- Async Execution ----------------------------------
    # ------------------------------------------------------
    
    async def call_metric_async(self, func: Callable, *args, **kwargs) -> Any:
        """Execute metric function asynchronously."""
        try:
            if inspect.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
                if inspect.iscoroutine(result):
                    result = await result
            return result
        except Exception as e:
            logger.error(f"Metric {getattr(func, '__name__', 'unknown')} failed: {e}")
            return np.nan



    # ------------------------------------------------------
    # --- Core Computation ---------------------------------
    # ------------------------------------------------------
        
    async def compute_metric(self, name: str, data: Any) -> Any:
        """Compute single metric - OPTIMIZED VERSION"""
        logger.debug(f"compute_metric START: {name}")
        
        func = self.resolve_metric(name)
        if func is None:
            return np.nan

        try:
            
            
            # SENTIMENT METRİKLERİ
            if name == "funding_rate":
                if isinstance(data, dict) and 'funding_rate_series' in data:
                    return await self.call_metric_async(func, data['funding_rate_series'])
            
            elif name == "funding_premium":
                if isinstance(data, dict) and all(k in data for k in ['futures_price', 'spot_price']):
                    return await self.call_metric_async(func, data['futures_price'], data['spot_price'])
            
            elif name == "oi_trend":
                if isinstance(data, dict) and 'open_interest_series' in data:
                    return await self.call_metric_async(func, data['open_interest_series'])
                    
            
            # MİKRO YAPI METRİKLERİ - doğrudan parametre yönlendirme
            if name in MICROSTRUCTURE_METRICS:
                params = self._extract_metric_params(name, data)
                raw_result = await self.call_metric_async(func, **params)
            else:
                # STANDART METRİKLER - mevcut mantık
                prepared_data = self.prepare_input(data, name)
                
                if (isinstance(prepared_data, dict) and 
                    all(k in prepared_data for k in ["high", "low", "close"])):
                    
                    raw_result = await self.call_metric_async(
                        func, 
                        pd.Series(prepared_data["high"].flatten()),
                        pd.Series(prepared_data["low"].flatten()), 
                        pd.Series(prepared_data["close"].flatten())
                    )
                else:
                    standardized_data = to_numpy(prepared_data)
                    raw_result = await self.call_metric_async(func, standardized_data)

            return self._optimize_and_standardize(name, raw_result)

        except Exception as e:
            logger.error(f"compute_metric failed for {name}: {e}")
            return np.nan

    def _extract_metric_params(self, metric_name: str, data: Any) -> Dict[str, Any]:
        """Metrik için gerekli parametreleri otomatik çıkar"""
        required_params = MICROSTRUCTURE_PARAMS.get(metric_name, [])
        params = {}
        
        for param in required_params:
            if isinstance(data, dict) and param in data:
                params[param] = data[param]
            else:
                # Parametre yoksa varsayılan değer üret
                params[param] = self._generate_default_param(param, data)
        
        return params

    def _generate_default_param(self, param_name: str, base_data: Any) -> np.ndarray:
        """Eksik parametreler için akıllı varsayılan değerler"""
        base_array = to_numpy(base_data)
        
        param_generators = {
            "bid_price": lambda: base_array * 0.999,
            "ask_price": lambda: base_array * 1.001,
            "best_bid": lambda: base_array * 0.999,
            "best_ask": lambda: base_array * 1.001,
            "bid_size": lambda: np.ones_like(base_array) * 100,
            "ask_size": lambda: np.ones_like(base_array) * 100,
            "buy_volume": lambda: np.ones_like(base_array) * 50,
            "sell_volume": lambda: np.ones_like(base_array) * 50,
            "trade_volume": lambda: np.ones_like(base_array) * 100,
            "depth_volume": lambda: np.ones_like(base_array) * 200,
            "depth_price": lambda: base_array * 1.005,
            "price_series": lambda: base_array,
            "taker_buy_volume": lambda: np.ones_like(base_array) * 30,
            "taker_sell_volume": lambda: np.ones_like(base_array) * 30,
        }
        
        return param_generators.get(param_name, lambda: base_array)()

    def _optimize_and_standardize(self, name: str, raw_value: Any) -> Any:
        """Optimizasyon ve standardizasyon"""
        optimized = self._optimize_for_composite(name, raw_value)
        return self.standardizer.standardize_output(optimized)

  

    def _optimize_for_composite(self, name: str, raw_value: Any) -> Any:
        """Optimize metric output for composite calculation."""
        logger.debug(f"Optimizing {name}: {type(raw_value)} -> scalar")
        
        # Convert scalar to Series
        if isinstance(raw_value, (np.number, float)):
            raw_value = pd.Series([raw_value])
        
        # Special metric processing
        if name == "macd" and isinstance(raw_value, pd.DataFrame):
            return raw_value["histogram"].iloc[-1] if "histogram" in raw_value.columns else 0.0
        
        # Price-based metrics to momentum
        if name in ["ema", "sma"] and isinstance(raw_value, (pd.Series, pd.DataFrame)):
            return self._calculate_momentum(raw_value, name)
        
        # Extract scalar
        return extract_scalar(raw_value)

    def _calculate_momentum(self, data: Any, name: str) -> float:
        """Calculate momentum for price-based metrics."""
        try:
            if len(data) <= 1:
                return 0.0
                
            if isinstance(data, pd.Series):
                return ((data.iloc[-1] - data.iloc[-2]) / data.iloc[-2]) * 100
            else:  # DataFrame
                col_data = data.iloc[:, -1]
                return ((col_data.iloc[-1] - col_data.iloc[-2]) / col_data.iloc[-2]) * 100
        except Exception as e:
            logger.debug(f"Momentum calculation failed for {name}: {e}")
            return 0.0

    async def compute_metrics(self, metrics: List[str], data: Any) -> Dict[str, Any]:
        """Compute multiple metrics in parallel."""
        if not metrics:
            return {}
            
        sem = asyncio.Semaphore(os.cpu_count() or 4)

        async def worker(metric: str) -> tuple[str, Any]:
            async with sem:
                val = await self.compute_metric(metric, data)
                return metric, val

        results = await asyncio.gather(*[worker(m) for m in metrics])
        return dict(results)

    async def compute_composite(self, composite_name: str, data: Any) -> float:
        """Compute composite metric."""
        logger.debug(f"START compute_composite: {composite_name}")
        
        try:
            cfg = self.metric_composites.get(composite_name) or self.macro_composites.get(composite_name)
            
            if not cfg:
                logger.error(f"Composite config not found: {composite_name}")
                return np.nan

            depends = cfg.get("depends_on", [])
            formula = cfg.get("formula", "")
            
            logger.debug(f"Config loaded: depends={depends}, formula={formula}")

            if not depends:
                logger.error(f"No dependencies for: {composite_name}")
                return np.nan

            # Compute dependencies
            logger.debug(f"Computing dependencies: {depends}")
            results = await self.compute_metrics(depends, data)
            logger.debug(f"Dependencies computed: {list(results.keys())}")

            # Extract scalars
            scores = {}
            for k, v in results.items():
                try:
                    scores[k] = extract_scalar(v)
                except Exception as e:
                    logger.error(f"extract_scalar failed for {k}: {e}")
                    scores[k] = np.nan

            logger.debug(f"Final scores: {scores}")
            
            # Normalize scores
            normed = {}
            for k, s in scores.items():
                try:
                    normed[k] = normalize_score(k, s)
                except Exception as e:
                    logger.error(f"normalize_score failed for {k}: {e}")
                    normed[k] = 0.0

            logger.debug(f"Normalized scores: {normed}")

            # No formula - use average
            if not formula:
                vals = [v for v in normed.values() if np.isfinite(v)]
                result = float(np.nanmean(vals)) if vals else np.nan
                return result

            # Evaluate formula
            try:
                code = self.cache.formula_cache(formula)
                safe_env = {
                    "np": np,
                    "tanh": np.tanh,
                    "exp": np.exp,
                    "log": np.log,
                    **normed,
                }
                
                val = eval(code, {"__builtins__": {}}, safe_env)
                
                # Cache and return result
                self._set_runtime(composite_name, val)
                return float(np.clip(val, -1.0, 1.0))
                
            except Exception as e:
                logger.error(f"Formula evaluation failed: {e}")
                return np.nan

        except Exception as e:
            logger.error(f"CRITICAL ERROR in compute_composite: {e}")
            return np.nan
    
    async def compute(self, metrics: List[str], data: Any = None, user_id: Optional[int] = None) -> Dict[str, Any]:
        """Main computation entry point."""
        composite_results = {}
        for metric in metrics:
            if metric in self.metric_composites or metric in self.macro_composites:
                score = await self.compute_composite(metric, data)
                composite_results[metric] = score
            else:
                # Atomic metric
                score = await self.compute_metric(metric, data)
                composite_results[metric] = score
        
        return composite_results

    # ------------------------------------------------------
    # --- Info Methods -------------------------------------
    # ------------------------------------------------------
    
    def get_available_metrics(self) -> List[str]:
        """Get all available atomic metrics."""
        all_metrics = []
        for metrics in self.atomic_indicators.values():
            all_metrics.extend(metrics)
        return sorted(all_metrics)
    
    def get_available_composites(self) -> List[str]:
        """Get all available composites."""
        return sorted(list(self.metric_composites.keys()) + list(self.macro_composites.keys()))

# ==========================================================
# === HIGH-LEVEL HELPER ====================================
# ==========================================================

async def compute_all(engine: CompositeEngine, data: Any) -> Dict[str, Any]:
    """Compute all defined composites."""
    composite_names = list(engine.metric_composites.keys()) + list(engine.macro_composites.keys())
    
    sem = asyncio.Semaphore(os.cpu_count() or 4)
    
    async def worker(name: str) -> tuple[str, float]:
        async with sem:
            val = await engine.compute_composite(name, data)
            return name, extract_scalar(val)

    results = await asyncio.gather(*[worker(name) for name in composite_names])
    return dict(results)