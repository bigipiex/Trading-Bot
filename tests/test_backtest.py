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
from src.risk import RiskManager, RiskConfig
from src.backtest import BacktestEngine
from src.synthetic import SyntheticGenerator, OUParams
from src.strategy import SimpleStrategy

class TestBacktest(unittest.TestCase):
    
    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        
    def test_closed_loop_mean_reversion(self):
        """
        Backtest on Mean-Reverting Synthetic Data.
        Strategy should make money.
        """
        # 1. Generate Synthetic OU Data (Mean Reversion)
        # Mu=0, Theta=0.5
        params = OUParams(mu=0.0, theta=0.5, sigma=0.1, dt=0.01)
        # Increase steps to allow reversion to play out
        _, prices = SyntheticGenerator.ornstein_uhlenbeck_process(params, n_steps=1000, start_price=100.0)
        
        # Create Feature Tensor: [LogRet, Vol, Mom, LogPrice, Mu]
        prices_tensor = torch.tensor(prices, dtype=torch.float32)
        log_prices = torch.log(prices_tensor)
        
        # Simple feature construction
        # Need batch dim: (Time, Batch, Feat)
        time_len = len(prices)
        market_data = torch.zeros(time_len, 1, 5, dtype=torch.float32)
        market_data[:, 0, 3] = log_prices # LogPrice
        market_data[1:, 0, 0] = log_prices[1:] - log_prices[:-1] # LogRet
        market_data[:, 0, 1] = 0.01 # Vol (constant low)
        market_data[:, 0, 2] = log_prices - 0.0 # Momentum (Proxy for deviation from mean 0)
        
        # Initialize Mu feature with the true mean (0.0 in log returns space? No, log(100))
        # For OU around 100, log(100) is mean.
        market_data[:, 0, 4] = np.log(100.0) 
        
        # 2. Setup System
        # We assume model knows the physics (Mean Reversion)
        market_model = HybridDynamicalSystem(feature_dim=5, epsilon=0.0)
        # Manually set parameters to match data roughly
        with torch.no_grad():
            market_model.physics.raw_kappa.data = torch.tensor([0.5], dtype=torch.float32) # Theta
            market_model.physics.raw_gamma.data = torch.tensor([0.1], dtype=torch.float32) # Gamma (Mu adaptation)
            
        control_system = ControlAffineSystem(market_model)
        
        # 3. Setup Strategy & Risk
        risk_config = RiskConfig(base_inventory_limit=5.0)
        risk_manager = RiskManager(risk_config)
        
        # Switch to SimpleStrategy with Long Horizon (N=50)
        # Reduce scaling factor to avoid blowing up on noise
        strategy = SimpleStrategy(
            control_system, 
            risk_manager=risk_manager, 
            scaling_factor=1.0,
            prediction_horizon=50
        )
        
        # Wrap Strategy in a mock MPC interface so we can reuse BacktestEngine?
        # BacktestEngine expects an MPCSolver which has a .solve() method.
        # Let's mock it.
        class StrategyWrapper:
            def __init__(self, strat, sys):
                self.strat = strat
                self.system = sys
            def solve(self, state, **kwargs):
                # Returns u_seq, loss
                # u_seq shape: (Horizon, Batch, 1)
                # We just need the first action u_0
                u_0 = self.strat.get_action(state)
                # Create dummy sequence
                u_seq = u_0.unsqueeze(0).repeat(10, 1, 1)
                return u_seq, 0.0
                
        solver_wrapper = StrategyWrapper(strategy, control_system)
        
        # 4. Run Backtest
        # Initial state from data
        initial_state = torch.cat([market_data[0], torch.zeros(1, 1)], dim=1) # Add Inventory=0
        
        engine = BacktestEngine(
            solver_wrapper, 
            initial_state, 
            risk_manager, 
            transaction_cost_bps=1.0, 
            dt=0.01
        )
        
        # Run
        print("Running Backtest with SimpleStrategy (N=50)...")
        engine.run(steps=time_len - 1, market_data=market_data)
        
        metrics = engine.get_metrics()
        print("\nBacktest Metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
            
        # 5. Validation
        # Should be profitable
        self.assertGreater(metrics["Total PnL"], 0.0)
        
        print("Backtest Test Passed: Profitable on mean-reversion.")

if __name__ == '__main__':
    unittest.main()
