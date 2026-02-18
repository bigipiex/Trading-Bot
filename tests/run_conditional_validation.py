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
from src.strategy_conditional import RegimeConditionalStrategy
from src.regime import RegimeDetector

class ConditionalValidation(unittest.TestCase):
    
    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        
    def _run_sim(self, scenario):
        if scenario == "OU":
            _, prices = SyntheticGenerator.ornstein_uhlenbeck_process(
                OUParams(mu=100, theta=0.5, sigma=1.0, dt=0.01), n_steps=1000, seed=42
            )
        elif scenario == "Trend":
            _, prices = SyntheticGenerator.geometric_brownian_motion(0.2, 0.1, 0.01, 1000, 100.0, seed=42)
        elif scenario == "Switching":
            _, prices, _ = SyntheticGenerator.regime_switching_process(2000, seed=42)
        elif scenario == "FlashCrash":
            _, prices = SyntheticGenerator.ornstein_uhlenbeck_process(
                OUParams(mu=100, theta=0.5, sigma=1.0, dt=0.01), n_steps=1000, seed=42
            )
            prices[500:] *= 0.90
            
        prices_tensor = torch.tensor(prices, dtype=torch.float32)
        log_prices = torch.log(prices_tensor)
        market_data = torch.zeros(len(prices), 1, 5, dtype=torch.float32)
        market_data[:, 0, 3] = log_prices
        market_data[1:, 0, 0] = log_prices[1:] - log_prices[:-1]
        market_data[:, 0, 1] = 0.01
        market_data[:, 0, 4] = np.log(100.0)
        
        rd = RegimeDetector(window_size=50)
        # Use Calibrated Risk Config
        risk_config = RiskConfig(
            vol_sigma_threshold=4.0,
            jump_sigma_threshold=5.0,
            drawdown_limit=0.15,
            max_drawdown_limit=0.30
        )
        rm = RiskManager(risk_config)
        
        # Conditional Strategy
        # Trend Regime: Focus on Trend (w_trend > 0), ignore MR (w_meanrev ~ 0)
        # MR Regime: Focus on MR (w_meanrev > 0), ignore Trend
        # Note: w_meanrev > 0 for standard MR logic (Buy low).
        strat = RegimeConditionalStrategy(
            rd, rm,
            w_trend_T=1.0, w_meanrev_T=0.0,
            w_trend_MR=0.0, w_meanrev_MR=2.0, # Strong MR in MR regime
            sigmoid_alpha=1000.0
        )
        
        class StratWrap:
            def __init__(self, s): self.strat = s
            def solve(self, state, **kwargs):
                return self.strat.get_action(state).unsqueeze(0).repeat(10, 1, 1), 0.0
                
        engine = BacktestEngine(StratWrap(strat), torch.cat([market_data[0], torch.zeros(1, 1)], dim=1), rm, dt=0.01)
        engine.run(steps=len(prices)-1, market_data=market_data)
        return engine.get_metrics()

    def test_all_regimes(self):
        print("\n=== Regime-Conditional Strategy Validation ===")
        
        # 1. OU (Mean Reversion)
        ou_met = self._run_sim("OU")
        print(f"OU Regime: Sharpe {ou_met['Sharpe Ratio']:.2f}, PnL {ou_met['Total PnL']:.2f}")
        
        # 2. Trend
        tr_met = self._run_sim("Trend")
        print(f"Trend Regime: Sharpe {tr_met['Sharpe Ratio']:.2f}, PnL {tr_met['Total PnL']:.2f}")
        
        # 3. Switching
        sw_met = self._run_sim("Switching")
        print(f"Switching Regime: Sharpe {sw_met['Sharpe Ratio']:.2f}, PnL {sw_met['Total PnL']:.2f}")
        
        # 4. Flash Crash
        fc_met = self._run_sim("FlashCrash")
        print(f"Flash Crash: MaxDD {fc_met['Max Drawdown']:.2f}, PnL {fc_met['Total PnL']:.2f}")
        
        # Assertions
        # Goal: Positive Sharpe in all normal regimes
        self.assertGreater(ou_met['Sharpe Ratio'], 0.0)
        self.assertGreater(tr_met['Sharpe Ratio'], 0.0)
        self.assertGreater(sw_met['Sharpe Ratio'], 0.0)
        
        # Crash Protection
        self.assertLess(fc_met['Max Drawdown'], 5.0)

if __name__ == '__main__':
    unittest.main()
