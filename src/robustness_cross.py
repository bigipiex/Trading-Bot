import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from src.synthetic import SyntheticGenerator, OUParams
from src.edge_analysis import DirectSignalStrategy
from src.backtest import BacktestEngine
from src.regime import RegimeDetector
from src.risk import RiskManager, RiskConfig
from src.control import ControlAffineSystem
from src.model import HybridDynamicalSystem

from typing import Dict, Any, List

class CrossPathRobustness:
    def __init__(self, n_paths: int = 100):
        self.n_paths = n_paths
        
    def run(self) -> Dict[str, pd.DataFrame]:
        results = {
            "OU": self._simulate_batch("OU"),
            "Trend": self._simulate_batch("Trend"),
            "Switching": self._simulate_batch("Switching")
        }
        return results
        
    def _simulate_batch(self, regime_type: str) -> pd.DataFrame:
        stats = []
        print(f"Simulating {self.n_paths} {regime_type} paths...")
        
        for i in tqdm(range(self.n_paths)):
            # 1. Generate Data
            if regime_type == "OU":
                params = OUParams(mu=100.0, theta=0.5, sigma=1.0, dt=0.01)
                _, prices = SyntheticGenerator.ornstein_uhlenbeck_process(params, 1000, seed=i)
            elif regime_type == "Trend":
                _, prices = SyntheticGenerator.geometric_brownian_motion(mu=0.1, sigma=0.1, dt=0.01, n_steps=1000, seed=i)
            else:
                _, prices, _ = SyntheticGenerator.regime_switching_process(1000, switch_prob=0.01, seed=i)
                
            # 2. Setup System (Mock Control System for Strategy init)
            # Strategy needs a system? Actually DirectSignalStrategy doesn't use system!
            # It just uses RegimeDetector and RiskManager.
            # But we might need to mock ControlAffineSystem if Strategy expects it in signature?
            # DirectSignalStrategy init: (regime_detector, risk_manager, ...)
            # It DOES NOT require system. Good.
            
            rd = RegimeDetector(window_size=50)
            rm = RiskManager(RiskConfig(base_inventory_limit=5.0))
            strat = DirectSignalStrategy(rd, rm, w_trend=1.0, w_meanrev=1.0, sigmoid_alpha=1000.0) # set alpha
            
            # Wrapper for BacktestEngine
            class StratWrap:
                def __init__(self, s): self.strat = s
                def solve(self, state, **kwargs):
                    return self.strat.get_action(state).unsqueeze(0).repeat(10, 1, 1), 0.0
            
            # 3. Prepare Data Tensor
            prices_tensor = torch.tensor(prices, dtype=torch.float32)
            log_prices = torch.log(prices_tensor)
            market_data = torch.zeros(len(prices), 1, 5, dtype=torch.float32)
            market_data[:, 0, 3] = log_prices
            market_data[1:, 0, 0] = log_prices[1:] - log_prices[:-1]
            market_data[:, 0, 1] = 0.01
            market_data[:, 0, 4] = np.log(100.0)
            
            # 4. Run Backtest
            engine = BacktestEngine(
                StratWrap(strat),
                torch.cat([market_data[0], torch.zeros(1, 1)], dim=1),
                rm,
                dt=0.01
            )
            
            # Suppress tqdm?
            engine.run(steps=len(prices)-1, market_data=market_data)
            metrics = engine.get_metrics()
            
            stats.append({
                "Sharpe": metrics["Sharpe Ratio"],
                "MaxDD": metrics["Max Drawdown"],
                "Turnover": metrics["Turnover"],
                "PnL": metrics["Total PnL"]
            })
            
        return pd.DataFrame(stats)
