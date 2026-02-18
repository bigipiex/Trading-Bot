import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from src.synthetic import SyntheticGenerator
from src.edge_analysis import DirectSignalStrategy
from src.regime import RegimeDetector
from src.risk import RiskManager, RiskConfig
from src.backtest import BacktestEngine

class SensitivitySweeper:
    def __init__(self):
        pass
        
    def run_sweep(self) -> pd.DataFrame:
        # Parameter Grids
        w_trends = [-2.0, 0.0, 2.0]
        w_mrs = [-2.0, 0.0, 2.0]
        alphas = [1.0, 10.0, 100.0]
        
        # Base Data (Switching)
        _, prices, _ = SyntheticGenerator.regime_switching_process(1000, seed=42)
        prices_tensor = torch.tensor(prices, dtype=torch.float32)
        log_prices = torch.log(prices_tensor)
        market_data = torch.zeros(len(prices), 1, 5, dtype=torch.float32)
        market_data[:, 0, 3] = log_prices
        market_data[1:, 0, 0] = log_prices[1:] - log_prices[:-1]
        market_data[:, 0, 1] = 0.01
        market_data[:, 0, 4] = np.log(100.0)
        
        results = []
        print(f"Running Sensitivity Sweep ({len(w_trends)*len(w_mrs)*len(alphas)} combos)...")
        
        for wt in w_trends:
            for wm in w_mrs:
                for a in alphas:
                    rd = RegimeDetector(window_size=50)
                    rm = RiskManager(RiskConfig(base_inventory_limit=5.0))
                    
                    # Set alpha as attribute
                    strat = DirectSignalStrategy(rd, rm, w_trend=wt, w_meanrev=wm)
                    strat.sigmoid_alpha = a
                    
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
                    metrics = engine.get_metrics()
                    
                    results.append({
                        "w_trend": wt,
                        "w_meanrev": wm,
                        "alpha": a,
                        "Sharpe": metrics["Sharpe Ratio"],
                        "PnL": metrics["Total PnL"]
                    })
                    
        return pd.DataFrame(results)
