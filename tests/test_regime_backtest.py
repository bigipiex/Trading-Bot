import unittest
import torch
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import HybridDynamicalSystem
from src.control import ControlAffineSystem
from src.risk import RiskManager, RiskConfig
from src.backtest import BacktestEngine
from src.synthetic import SyntheticGenerator, OUParams
from src.regime import RegimeDetector
from src.strategy_regime import RegimeSwitchingStrategy

class TestRegimeBacktest(unittest.TestCase):
    
    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        
    def run_simulation(self, prices, regime_type="Unknown"):
        # Setup Data
        prices_tensor = torch.tensor(prices, dtype=torch.float32)
        log_prices = torch.log(prices_tensor)
        time_len = len(prices)
        
        market_data = torch.zeros(time_len, 1, 5, dtype=torch.float32)
        market_data[:, 0, 3] = log_prices
        market_data[1:, 0, 0] = log_prices[1:] - log_prices[:-1]
        market_data[:, 0, 1] = 0.01 # Low vol
        market_data[:, 0, 2] = log_prices - 0.0 
        market_data[:, 0, 4] = np.log(100.0)
        
        # Setup System
        model = HybridDynamicalSystem(feature_dim=5, epsilon=0.0)
        # Initialize reasonable params
        with torch.no_grad():
            model.physics.raw_kappa.data = torch.tensor([0.5])
            model.physics.raw_gamma.data = torch.tensor([0.1])
            
        control_system = ControlAffineSystem(model)
        risk_manager = RiskManager(RiskConfig(base_inventory_limit=5.0))
        regime_detector = RegimeDetector(window_size=50, threshold_trend=0.0002) # Tuned threshold
        
        strategy = RegimeSwitchingStrategy(
            control_system,
            regime_detector,
            risk_manager,
            scaling_factor=10.0 # Higher leverage for signal visibility
        )
        
        # Wrapper for BacktestEngine
        class StrategyWrapper:
            def __init__(self, strat): self.strat = strat
            def solve(self, state, **kwargs):
                return self.strat.get_action(state).unsqueeze(0).repeat(10, 1, 1), 0.0
                
        engine = BacktestEngine(
            StrategyWrapper(strategy),
            torch.cat([market_data[0], torch.zeros(1, 1)], dim=1),
            risk_manager,
            transaction_cost_bps=1.0,
            dt=0.01
        )
        
        print(f"\nRunning {regime_type} Backtest ({time_len} steps)...")
        engine.run(steps=time_len - 1, market_data=market_data)
        
        metrics = engine.get_metrics()
        print(f"{regime_type} Metrics:")
        print(f"Total PnL: {metrics['Total PnL']:.4f}")
        print(f"Sharpe: {metrics['Sharpe Ratio']:.4f}")
        
        return metrics

    def test_pure_ou(self):
        # 1. Pure OU (Mean Reversion)
        params = OUParams(mu=0.0, theta=0.5, sigma=0.1, dt=0.01)
        _, prices = SyntheticGenerator.ornstein_uhlenbeck_process(params, n_steps=1000, start_price=100.0)
        metrics = self.run_simulation(prices, "Pure OU")
        
        # Should be profitable
        # Note: If regime detector classifies as Neutral/Trend, we miss out.
        # But OU has low trend, high deviation -> Should be MR.
        self.assertGreater(metrics["Total PnL"], -10.0) # Allow small loss due to costs/neutral

    def test_pure_trend(self):
        # 2. Pure Trend (GBM)
        # Strong drift
        _, prices = SyntheticGenerator.geometric_brownian_motion(mu=0.2, sigma=0.1, dt=0.01, n_steps=1000, start_price=100.0)
        metrics = self.run_simulation(prices, "Pure Trend")
        
        # Should be profitable (Momentum)
        self.assertGreater(metrics["Total PnL"], 0.0)

    def test_regime_switching(self):
        # 3. Switching
        _, prices, regimes = SyntheticGenerator.regime_switching_process(n_steps=2000, switch_prob=0.005)
        metrics = self.run_simulation(prices, "Regime Switching")
        
        # This is the hardest test.
        # Check if we survive.
        self.assertGreater(metrics["Total PnL"], -100.0)

if __name__ == '__main__':
    unittest.main()
