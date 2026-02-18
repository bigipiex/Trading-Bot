import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple

@dataclass
class RegimeMetrics:
    trend_strength: float
    normalized_deviation: float
    volatility: float
    regime_label: int # 0=Neutral, 1=Trending, -1=Mean Reversion (Wait, labels? Let's clarify)
    # User said: 
    # Trending (1 or -1 direction?)
    # Mean Reversion (0?)
    # Neutral (-99?)
    # Let's map: 
    # 1: Trending (Momentum Strategy)
    # 0: Mean Reversion (MR Strategy)
    # -1: Neutral (No Trade)

class RegimeDetector:
    """
    Detects market regime based on heuristics.
    
    Metrics:
    1. TrendStrength: Slope of rolling linear regression of Price vs Time.
       Normalized by Price to be % change per step? Or just raw slope?
       Let's use annualized slope / current price = % drift.
       
    2. NormalizedDeviation: |P - mu| / sigma
       (Z-score relative to recent moving average)
       
    3. Volatility: Rolling std dev of returns.
    """
    def __init__(self, 
                 window_size: int = 50,
                 threshold_trend: float = 0.0005, # Slope threshold (e.g. 0.05% per step)
                 threshold_dev: float = 2.0):     # Z-score threshold for MR
        self.window_size = window_size
        self.threshold_trend = threshold_trend
        self.threshold_dev = threshold_dev
        
    def detect(self, prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns array of regime labels for the whole history.
        Output: (regimes, slopes, norm_dev)
        """
        n = len(prices)
        regimes = np.full(n, -1, dtype=int) # Default Neutral
        
        # We need rolling metrics.
        # Use pandas for efficiency
        s_prices = pd.Series(prices)
        
        # 1. Rolling Mean & Volatility (for Normalized Deviation)
        # We use same window for simplicity, or maybe shorter for deviation?
        rolling_mean = s_prices.rolling(window=self.window_size).mean()
        rolling_std = s_prices.rolling(window=self.window_size).std()
        
        # Normalized Deviation = |P - MA| / Std
        norm_dev = np.abs(s_prices - rolling_mean) / (rolling_std + 1e-6)
        norm_dev = norm_dev.fillna(0).values
        
        # 2. Trend Strength (Rolling Slope)
        slopes = np.zeros(n)
        
        # Precompute X (0..W-1)
        x = np.arange(self.window_size)
        x_mean = x.mean()
        x_var = ((x - x_mean)**2).sum()
        
        # Loop for slope (vectorize later if needed)
        # Only start from window_size
        for i in range(self.window_size, n):
            y = prices[i-self.window_size : i]
            y_mean = y.mean()
            numerator = ((x - x_mean) * (y - y_mean)).sum()
            slope = numerator / x_var
            
            # Normalize slope by price to get % change per step
            slopes[i] = slope / y[-1]
            
        # 3. Classify
        for i in range(self.window_size, n):
            slope = slopes[i]
            dev = norm_dev[i]
            
            if abs(slope) > self.threshold_trend:
                regimes[i] = 1 # Momentum
            elif dev > self.threshold_dev:
                regimes[i] = 0 # Mean Reversion
            else:
                regimes[i] = -1 # Neutral
                
        return regimes, slopes, norm_dev
