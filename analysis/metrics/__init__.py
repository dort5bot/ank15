# analysis/metrics/__init__.py
from .composite import CompositeEngine
from .standard import MetricStandard, metric_standard

# 📁 metrics/__init__.py - ACİL PATCH
import numpy as np
import pandas as pd

def safe_data_conversion(data):
    """Universal data conversion for metric inputs"""
    if data is None:
        return np.array([], dtype=float)
    
    # Handle pandas
    if hasattr(data, 'iloc'):
        return data.values if hasattr(data, 'values') else np.array(data)
    
    # Handle lists/tuples
    if isinstance(data, (list, tuple)):
        return np.array(data, dtype=float)
    
    # Handle dicts (for microstructure)
    if isinstance(data, dict):
        return {k: safe_data_conversion(v) for k, v in data.items()}
    
    return np.array([data], dtype=float)
    
__all__ = ['CompositeEngine', 'MetricStandard', 'metric_standard']