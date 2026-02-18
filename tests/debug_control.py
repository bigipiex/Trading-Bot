import unittest
import torch
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import HybridDynamicalSystem
from src.control import ControlAffineSystem
from src.objective import ProfitFunctional, ControlConfig
from src.mpc import MPCSolver
from src.synthetic import SyntheticGenerator, OUParams

class ControlDebugger(unittest.TestCase):
    
    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        
    def run_control_sweep(self, horizon_steps: int, neural_on: bool = False):
        """
        Runs the MPC on a synthetic Mean Reversion (OU) signal.
        Returns:
            corr_momentum: Correlation(u, dP/dt)
            corr_reversion: Correlation(u, P - mu)
        """
        # 1. Generate Synthetic OU Data (Mean Reversion)
        # Mu=0, Theta=0.5
        params = OUParams(mu=0.0, theta=0.5, sigma=0.1, dt=0.01)
        # Generate longer sequence
        _, prices = SyntheticGenerator.ornstein_uhlenbeck_process(params, n_steps=200, start_price=100.0)
        
        prices_tensor = torch.tensor(prices, dtype=torch.float32)
        log_prices = torch.log(prices_tensor)
        
        # State construction
        time_len = len(prices)
        # (Time, Batch=1, Feat=4)
        market_data = torch.zeros(time_len, 1, 4, dtype=torch.float32)
        market_data[:, 0, 3] = log_prices 
        market_data[1:, 0, 0] = log_prices[1:] - log_prices[:-1] # dP/dt
        market_data[:, 0, 1] = 0.01 
        market_data[:, 0, 2] = log_prices - 0.0 # Deviation from mean
        
        # 2. Setup System
        epsilon = 0.1 if neural_on else 0.0
        market_model = HybridDynamicalSystem(feature_dim=4, epsilon=epsilon)
        
        # Set parameters to MATCH the OU process
        with torch.no_grad():
            market_model.physics.raw_kappa.data = torch.tensor([0.5], dtype=torch.float32)
            market_model.physics.mu.data = torch.tensor([np.log(100.0)], dtype=torch.float32)
            
        control_system = ControlAffineSystem(market_model)
        
        # 3. Setup MPC
        # High cost to encourage smooth control
        obj_config = ControlConfig(lambda_risk=0.1, lambda_cost=0.1, dt=0.01)
        objective = ProfitFunctional(obj_config)
        
        solver = MPCSolver(control_system, objective, horizon_steps=horizon_steps)
        solver.dt = 0.01
        
        # 4. Run Step-by-Step (Closed Loop Simulation)
        u_history = []
        p_dev_history = [] # P - mu
        dp_history = []    # dP/dt
        
        # Initial state
        state = torch.cat([market_data[0], torch.zeros(1, 1)], dim=1)
        
        # Run for 50 steps
        for i in range(50):
            # Observe current state from data (Open Loop Model predictive control usually uses current observation)
            current_obs = market_data[i]
            # Update state with observation
            state[:, :4] = current_obs
            
            # Solve
            u_seq, _ = solver.solve(state, iterations=20, lr=0.1)
            u_t = u_seq[0].detach()
            
            u_history.append(u_t.item())
            p_dev_history.append(current_obs[0, 2].item()) # Deviation
            dp_history.append(current_obs[0, 0].item())    # dP/dt
            
        # 5. Compute Correlations
        u_tensor = torch.tensor(u_history)
        p_dev_tensor = torch.tensor(p_dev_history)
        dp_tensor = torch.tensor(dp_history)
        
        # Normalize
        if u_tensor.std() == 0: return 0.0, 0.0
        
        corr_mom = torch.corrcoef(torch.stack([u_tensor, dp_tensor]))[0, 1].item()
        corr_rev = torch.corrcoef(torch.stack([u_tensor, p_dev_tensor]))[0, 1].item()
        
        return corr_mom, corr_rev

    def test_horizon_sweep(self):
        """Analyze effect of horizon length on control behavior."""
        print("\n--- Horizon Sweep Diagnostics ---")
        print(f"{'Horizon':<10} {'Neural':<10} {'Corr(u, dP)':<15} {'Corr(u, P-mu)':<15}")
        
        horizons = [5, 10, 20, 50]
        
        for h in horizons:
            # Test WITHOUT Neural Net (Pure Physics)
            cm_phy, cr_phy = self.run_control_sweep(h, neural_on=False)
            print(f"{h:<10} {'OFF':<10} {cm_phy:<15.4f} {cr_phy:<15.4f}")
            
            # Test WITH Neural Net
            cm_nn, cr_nn = self.run_control_sweep(h, neural_on=True)
            print(f"{h:<10} {'ON':<10} {cm_nn:<15.4f} {cr_nn:<15.4f}")
            
        print("-" * 60)
        
        # Logic Check:
        # For Mean Reversion, we want NEGATIVE correlation with (P - mu).
        # i.e., If P > mu (High), u should be negative (Sell).
        # If P < mu (Low), u should be positive (Buy).
        
        # We also likely want NEGATIVE correlation with dP/dt IF dP/dt is moving away from mean.
        # But dP/dt alone is ambiguous. P-mu is the true signal.
        
        # If corr_rev is positive, we are MOMENTUM TRADING (Buying high P).
        # If corr_rev is negative, we are MEAN REVERSION TRADING (Selling high P).

if __name__ == '__main__':
    unittest.main()
