import numpy as np
import torch
from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class OUParams:
    """Parameters for the Ornstein-Uhlenbeck process."""
    mu: float  # Long-term mean
    theta: float  # Mean reversion speed
    sigma: float  # Volatility
    dt: float  # Time step

class SyntheticGenerator:
    """
    Generates synthetic financial data using stochastic differential equations.
    Focus is on Ornstein-Uhlenbeck (OU) processes which exhibit mean-reversion,
    a key property in pair trading and market making.
    """
    
    @staticmethod
    def ornstein_uhlenbeck_process(
        params: OUParams,
        n_steps: int,
        start_price: float = 100.0,
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate an Ornstein-Uhlenbeck process:
        dX_t = theta * (mu - X_t) * dt + sigma * dW_t
        
        Args:
            params: OUParams object containing process parameters
            n_steps: Number of simulation steps
            start_price: Initial value
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (time_array, price_array)
        """
        if seed is not None:
            np.random.seed(seed)
            
        t = np.linspace(0, params.dt * n_steps, n_steps)
        x = np.zeros(n_steps)
        x[0] = start_price
        
        # Pre-compute square root of dt for Brownian motion
        sqrt_dt = np.sqrt(params.dt)
        
        for i in range(1, n_steps):
            # Euler-Maruyama method
            dx = params.theta * (params.mu - x[i-1]) * params.dt + \
                 params.sigma * np.random.normal(0, 1) * sqrt_dt
            x[i] = x[i-1] + dx
            
        return t, x

    @staticmethod
    def geometric_brownian_motion(
        mu: float,
        sigma: float,
        dt: float,
        n_steps: int,
        start_price: float = 100.0,
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate Geometric Brownian Motion (standard stock price model):
        dS_t = mu * S_t * dt + sigma * S_t * dW_t
        """
        if seed is not None:
            np.random.seed(seed)
            
        t = np.linspace(0, dt * n_steps, n_steps)
        # Exact solution: S_t = S_0 * exp((mu - 0.5*sigma^2)t + sigma*W_t)
        
        # Generate Brownian Motion path
        dW = np.random.normal(0, np.sqrt(dt), size=n_steps)
        # dW[0] = 0 # No, random walk starts at 0, dW is incremental
        
        # Calculate W_t
        W = np.concatenate([[0], np.cumsum(dW[:-1])])
        
        # Calculate drift and diffusion components
        # Note: drift is time-dependent term
        drift = (mu - 0.5 * sigma**2) * t
        diffusion = sigma * W
        
        prices = start_price * np.exp(drift + diffusion)
        
        return t, prices

    @staticmethod
    def regime_switching_process(
        n_steps: int,
        switch_prob: float = 0.005,
        dt: float = 0.01,
        start_price: float = 100.0,
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates a price path that switches between Mean Reversion (OU) and Trending (GBM).
        
        Returns:
            t: time array
            prices: price array
            regimes: array of regime labels (0=OU, 1=Trend Up, -1=Trend Down)
        """
        if seed is not None:
            np.random.seed(seed)
            
        t = np.linspace(0, dt * n_steps, n_steps)
        prices = np.zeros(n_steps)
        regimes = np.zeros(n_steps)
        prices[0] = start_price
        
        # Initial regime
        current_regime = 0 # 0=OU
        
        # Parameters
        # OU parameters: High theta for clear reversion, sigma scaled to price
        # If Price ~ 100, sigma=1.0 is 1% volatility per unit time? 
        # No, sigma*dW is additive. 
        ou_theta = 2.0
        ou_sigma = 1.0 
        ou_mu = start_price
        
        # GBM parameters
        gbm_mu = 0.10 # 10% drift
        gbm_sigma = 0.10 # 10% vol
        
        current_price = start_price
        
        sqrt_dt = np.sqrt(dt)
        
        for i in range(1, n_steps):
            # Check for switch
            if np.random.random() < switch_prob:
                # Switch logic:
                # If OU (0) -> go to Trend (1 or -1)
                # If Trend (1/-1) -> go to OU (0)
                if current_regime == 0:
                    current_regime = np.random.choice([1, -1])
                    # No parameter update needed for GBM, it's relative
                else:
                    current_regime = 0
                    # IMPORTANT: Reset OU mean to current price to simulate "new equilibrium"
                    ou_mu = current_price
            
            # Simulate step based on regime
            if current_regime == 0: # OU
                # dP = theta*(mu - P)*dt + sigma*dW
                dx = ou_theta * (ou_mu - current_price) * dt + \
                     ou_sigma * np.random.normal(0, 1) * sqrt_dt
                current_price += dx
                
            elif current_regime == 1: # Trend Up
                # dS = mu*S*dt + sigma*S*dW
                dS = gbm_mu * current_price * dt + \
                     gbm_sigma * current_price * np.random.normal(0, 1) * sqrt_dt
                current_price += dS
                
            elif current_regime == -1: # Trend Down
                dS = -gbm_mu * current_price * dt + \
                     gbm_sigma * current_price * np.random.normal(0, 1) * sqrt_dt
                current_price += dS
                
            prices[i] = current_price
            regimes[i] = current_regime
            
        return t, prices, regimes

    @staticmethod
    def generate_training_batch(
        batch_size: int,
        seq_len: int,
        params: OUParams
    ) -> torch.Tensor:
        """
        Generates a batch of synthetic data for training.
        Returns tensor of shape (batch_size, seq_len, 1)
        """
        batch_data = []
        for _ in range(batch_size):
            _, x = SyntheticGenerator.ornstein_uhlenbeck_process(
                params, 
                seq_len, 
                start_price=np.random.uniform(params.mu - 10, params.mu + 10)
            )
            batch_data.append(x)
            
        return torch.tensor(np.array(batch_data), dtype=torch.float32).unsqueeze(-1)
