import unittest
import torch
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import HybridDynamicalSystem
from src.control import ControlAffineSystem
from src.strategy import SimpleStrategy
from src.synthetic import SyntheticGenerator, OUParams

class SignalAnalyzer(unittest.TestCase):
    
    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        
    def test_signal_quality(self):
        """
        Analyze if the raw model prediction (dP/dt) has correlation with future returns.
        """
        # 1. Generate Mean Reversion Data
        params = OUParams(mu=0.0, theta=0.5, sigma=0.1, dt=0.01)
        _, prices = SyntheticGenerator.ornstein_uhlenbeck_process(params, n_steps=500, start_price=100.0)
        
        prices_tensor = torch.tensor(prices, dtype=torch.float32)
        log_prices = torch.log(prices_tensor)
        
        # Features [LogRet, Vol, Mom, LogPrice, Mu]
        time_len = len(prices)
        market_data = torch.zeros(time_len, 1, 5, dtype=torch.float32)
        market_data[:, 0, 3] = log_prices
        market_data[1:, 0, 0] = log_prices[1:] - log_prices[:-1]
        market_data[:, 0, 1] = 0.01
        market_data[:, 0, 2] = log_prices - 0.0 # Deviation
        market_data[:, 0, 4] = np.log(100.0) # True Mu
        
        # 2. Setup Perfect Model (Cheating with correct params)
        model = HybridDynamicalSystem(feature_dim=5, epsilon=0.0)
        with torch.no_grad():
            model.physics.raw_kappa.data = torch.tensor([0.5], dtype=torch.float32)
            model.physics.beta.data = torch.tensor([0.0], dtype=torch.float32) # No momentum coupling
            
        control_system = ControlAffineSystem(model)
        strategy = SimpleStrategy(control_system, scaling_factor=100.0)
        
        # 3. Compute Predictions vs Realized Returns
        predictions = []
        realized_returns = []
        
        # We need to manually inject the correct Mu (log(100)) into the state
        # Because we constructed market_data manually.
        # market_data[:, 0, 4] is Mu.
        
        # Wait, the strategy uses model.physics.mu?
        # No, the model uses x[:, 4] as Mu.
        # But we also initialized model.physics.mu parameter?
        # Let's check model.py forward():
        # d_log_price = self.kappa * (mu - log_price) + self.beta * momentum
        # It uses the input x[:, 4] as mu!
        # And we set market_data[:, 0, 4] = log(100).
        
        # So prediction dP/dt = kappa * (log(100) - log(P))
        
        for t in range(time_len - 1):
            # Construct state (Batch, 6) -> 5 Market + 1 Inventory
            # market_data[t] is (1, 5)
            # inventory is (1, 1)
            
            # Important: We must use the market_data as constructed
            current_obs = market_data[t]
            
            # Augment with inventory
            inv = torch.zeros(1, 1)
            state = torch.cat([current_obs, inv], dim=1) 
            
            # Predict
            # Strategy computes u ~ dP/dt. 
            # We want raw dP/dt prediction.
            # Strategy calls system(0, state, u=0)
            
            # Let's call system directly to be sure
            dx_dt = control_system(0.0, state, torch.zeros(1, 1))
            pred_dp = dx_dt[0, 3].item() # dP/dt
            
            # Realized Return (Next Price - Current Price)
            # market_data[t+1] is next observation
            real_ret = market_data[t+1, 0, 3].item() - market_data[t, 0, 3].item()
            
            predictions.append(pred_dp)
            realized_returns.append(real_ret)
            
        # 4. Correlation Analysis
        pred_tensor = torch.tensor(predictions)
        real_tensor = torch.tensor(realized_returns)
        
        correlation = torch.corrcoef(torch.stack([pred_tensor, real_tensor]))[0, 1].item()
        
        print(f"\nSignal Correlation (Pred vs Real): {correlation:.4f}")
        
        # In Mean Reversion:
        # If P > Mu, Model predicts dP < 0.
        # Realized dP should be < 0 on average.
        # So Correlation should be POSITIVE (Model predicts down, Price goes down).
        
        # If P > Mu, P-Mu > 0.
        # dP/dt = -kappa * (P-Mu) < 0.
        # So model output is correct.
        
        self.assertGreater(correlation, 0.1)
        print("Signal Test Passed: Raw model has predictive edge.")

if __name__ == '__main__':
    unittest.main()
