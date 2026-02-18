import unittest
import torch
import numpy as np
import pandas as pd
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.synthetic import SyntheticGenerator, OUParams
from src.risk import RiskManager, RiskConfig
from src.backtest import BacktestEngine
from src.edge_analysis import DirectSignalStrategy
from src.regime import RegimeDetector
from src.economic_analysis import StressTester

class TailRiskValidation(unittest.TestCase):
    
    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        
    def _run_scenario(self, scenario_name, risk_config=None):
        stress = StressTester()
        # Mock StressTester to use custom RiskConfig
        # StressTester._run_strategy uses default RiskConfig
        # We need to override it.
        
        # Let's run manually here.
        if scenario_name == "FlashCrash":
            _, prices = SyntheticGenerator.ornstein_uhlenbeck_process(
                OUParams(mu=100, theta=0.5, sigma=1.0, dt=0.01), n_steps=1000, seed=42
            )
            prices[500:] *= 0.90 # 10% drop
        else:
            _, prices, _ = SyntheticGenerator.regime_switching_process(2000, seed=42)
            
        prices_tensor = torch.tensor(prices, dtype=torch.float32)
        log_prices = torch.log(prices_tensor)
        market_data = torch.zeros(len(prices), 1, 5, dtype=torch.float32)
        market_data[:, 0, 3] = log_prices
        market_data[1:, 0, 0] = log_prices[1:] - log_prices[:-1]
        market_data[:, 0, 1] = 0.01
        market_data[:, 0, 4] = np.log(100.0)
        
        rd = RegimeDetector(window_size=50)
        
        if risk_config is None:
            # Baseline (No Tail Protection)
            # Default config has some tail protection? 
            # RiskConfig defaults: vol_sigma=3.0, jump_sigma=4.0.
            # To test baseline, we need to disable it (set thresholds high).
            risk_config = RiskConfig(
                vol_sigma_threshold=100.0, 
                jump_sigma_threshold=100.0,
                drawdown_limit=1.0
            )
            
        rm = RiskManager(risk_config)
        strat = DirectSignalStrategy(rd, rm, w_trend=0.0, w_meanrev=-2.0)
        
        class StratWrap:
            def __init__(self, s): self.strat = s
            def solve(self, state, **kwargs):
                return self.strat.get_action(state).unsqueeze(0).repeat(10, 1, 1), 0.0
                
        engine = BacktestEngine(
            StratWrap(strat),
            torch.cat([market_data[0], torch.zeros(1, 1)], dim=1),
            rm,
            dt=0.01
        )
        
        engine.run(steps=len(prices)-1, market_data=market_data)
        return engine.get_metrics()

    def test_1_flash_crash_protection(self):
        print("\n=== 1. Flash Crash Protection Check ===")
        
        # Baseline (No Protection)
        base_metrics = self._run_scenario("FlashCrash", risk_config=None) # Disable
        print(f"Baseline: MaxDD {base_metrics['Max Drawdown']:.2f}, PnL {base_metrics['Total PnL']:.2f}")
        
        # Protected (Default Tail Params)
        # vol_sigma=3.0, jump_sigma=4.0
        prot_config = RiskConfig(
            vol_sigma_threshold=3.0,
            jump_sigma_threshold=4.0, # 4 sigma return
            jump_freeze_steps=20
        )
        prot_metrics = self._run_scenario("FlashCrash", risk_config=prot_config)
        print(f"Protected: MaxDD {prot_metrics['Max Drawdown']:.2f}, PnL {prot_metrics['Total PnL']:.2f}")
        
        # Verify improvement
        # MaxDD should be significantly lower
        # Baseline MaxDD was ~111.
        self.assertLess(prot_metrics['Max Drawdown'], base_metrics['Max Drawdown'] * 0.8)

    def test_2_normal_regime_impact(self):
        print("\n=== 2. Normal Regime Impact Check ===")
        
        # Baseline
        base_metrics = self._run_scenario("Normal", risk_config=None)
        print(f"Baseline: Sharpe {base_metrics['Sharpe Ratio']:.2f}, PnL {base_metrics['Total PnL']:.2f}")
        
        # Protected
        prot_config = RiskConfig(
            vol_sigma_threshold=3.0,
            jump_sigma_threshold=4.0
        )
        prot_metrics = self._run_scenario("Normal", risk_config=prot_config)
        print(f"Protected: Sharpe {prot_metrics['Sharpe Ratio']:.2f}, PnL {prot_metrics['Total PnL']:.2f}")
        
        # Verify degradation is minimal (<10%)
        # Sharpe might even improve if it filters noise?
        # Degradation: (Base - Prot) / Base
        deg = (base_metrics['Sharpe Ratio'] - prot_metrics['Sharpe Ratio']) / base_metrics['Sharpe Ratio']
        print(f"Sharpe Degradation: {deg*100:.1f}%")
        
        self.assertLess(deg, 0.10) # Less than 10% degradation

    def test_3_monte_carlo_stress(self):
        print("\n=== 3. Monte Carlo Flash Stress (10 runs) ===")
        # Run 10 random flash crashes with varying magnitudes
        
        results = []
        for i in range(10):
            # Generate Flash Crash
            _, prices = SyntheticGenerator.ornstein_uhlenbeck_process(
                OUParams(mu=100, theta=0.5, sigma=1.0, dt=0.01), n_steps=1000, seed=i
            )
            # Random drop 5% to 15%
            drop = np.random.uniform(0.85, 0.95)
            prices[500:] *= drop
            
            # Run Protected
            prot_config = RiskConfig(vol_sigma_threshold=3.0, jump_sigma_threshold=4.0)
            
            # Setup manually again... code dup but robust
            prices_tensor = torch.tensor(prices, dtype=torch.float32)
            log_prices = torch.log(prices_tensor)
            market_data = torch.zeros(len(prices), 1, 5, dtype=torch.float32)
            market_data[:, 0, 3] = log_prices
            market_data[1:, 0, 0] = log_prices[1:] - log_prices[:-1]
            market_data[:, 0, 1] = 0.01
            market_data[:, 0, 4] = np.log(100.0)
            
            rd = RegimeDetector(window_size=50)
            rm = RiskManager(prot_config)
            strat = DirectSignalStrategy(rd, rm, w_trend=0.0, w_meanrev=-2.0)
            
            class StratWrap:
                def __init__(self, s): self.strat = s
                def solve(self, state, **kwargs):
                    return self.strat.get_action(state).unsqueeze(0).repeat(10, 1, 1), 0.0
                    
            engine = BacktestEngine(StratWrap(strat), torch.cat([market_data[0], torch.zeros(1, 1)], dim=1), rm, dt=0.01)
            engine.run(steps=len(prices)-1, market_data=market_data)
            m = engine.get_metrics()
            
            results.append(m['Max Drawdown'])
            
        avg_dd = np.mean(results)
        print(f"Avg MaxDD under stress: {avg_dd:.2f}")
        print(f"Worst MaxDD: {np.max(results):.2f}")
        
        # Ensure survival (MaxDD < 0.5? Or < 1.0 relative to capital?
        # Note: Backtest PnL is absolute.
        # If Base Capital is ~1000 (implied by positions), DD of 100 is 10%.
        # Baseline had DD 111.
        # Protected should be much less.
        
        self.assertLess(avg_dd, 100.0) # Arbitrary safe limit based on baseline failure

if __name__ == '__main__':
    unittest.main()
