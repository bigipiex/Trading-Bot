import numpy as np
import torch
import pandas as pd
from typing import Union, List, Optional

class StateTransformer:
    """
    Transforms raw market data (Price, Volume) into the rigorous state vector X(t)
    required for the dynamical system.
    
    State Vector X(t) components:
    1. Log Returns: r_t = log(P_t / P_{t-1})
    2. Volatility: Rolling standard deviation of returns
    3. Momentum: Normalized rate of change
    4. Market Regime: Latent variable (placeholder for now)
    """
    
    def __init__(self, vol_window: int = 20, momentum_window: int = 10):
        self.vol_window = vol_window
        self.momentum_window = momentum_window
        self.mean_stats = {}
        self.std_stats = {}
        
    def transform(self, prices: Union[np.ndarray, pd.Series]) -> torch.Tensor:
        """
        Takes raw prices and returns the state tensor X(t).
        Shape: (Time, Features)
        """
        if isinstance(prices, list):
            prices = np.array(prices)
        if isinstance(prices, np.ndarray):
            prices = pd.Series(prices)
            
        # 1. Log Returns
        # Using log returns makes the data more stationary and additive
        log_returns = np.log(prices / prices.shift(1)).fillna(0)
        
        # 2. Rolling Volatility (annualized proxy or simple std)
        # We add a small epsilon to avoid division by zero
        volatility = log_returns.rolling(window=self.vol_window).std().bfill() + 1e-6
        
        # 3. Momentum (Z-score of price relative to moving average)
        ma = prices.rolling(window=self.momentum_window).mean()
        momentum = (prices - ma) / (prices.rolling(window=self.momentum_window).std() + 1e-6)
        momentum = momentum.fillna(0)
        
        # 4. Normalized Log Prices (Detrended)
        # Useful for mean-reversion models
        log_prices = np.log(prices)
        
        # 5. Equilibrium (Mu) - Initial Estimate using Long Window SMA
        # We append this as a feature. The model will then evolve it dynamically.
        # But wait, transform() is static. 
        # We can just initialize Mu = SMA(Price)
        mu_window = self.momentum_window * 5
        mu = log_prices.rolling(window=mu_window).mean().bfill()
        
        # Combine into a DataFrame first
        features_df = pd.DataFrame({
            'log_returns': log_returns,
            'volatility': volatility,
            'momentum': momentum,
            'log_price': log_prices,
            'mu': mu
        })
        
        # Convert to Tensor
        return torch.tensor(features_df.values, dtype=torch.float32)
    
    def normalize(self, x: torch.Tensor, fit: bool = False) -> torch.Tensor:
        """
        Z-score normalization.
        If fit=True, computes statistics from x.
        If fit=False, uses stored statistics.
        """
        if fit:
            self.mean_stats = x.mean(dim=0)
            self.std_stats = x.std(dim=0) + 1e-6
            
        return (x - self.mean_stats) / self.std_stats
    
    def inverse_transform(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Reverts normalization."""
        return x_norm * self.std_stats + self.mean_stats
