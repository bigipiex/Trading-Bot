import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from typing import Dict, Any, List
from src.synthetic import SyntheticGenerator, OUParams
from src.validation import WalkForwardValidator

class RobustnessSweeper:
    """
    Performs Monte Carlo parameter perturbation to assess strategy stability.
    """
    def __init__(self, n_simulations: int = 100, perturbation_scale: float = 0.2):
        self.n_simulations = n_simulations
        self.scale = perturbation_scale
        
    def run_sweep(self) -> pd.DataFrame:
        results = []
        
        # Base Data (Fixed for all runs to isolate parameter effect? Or perturb data too?)
        # "Run parameter perturbation". Usually implies fixed data.
        # Let's generate a standard regime switching dataset.
        _, prices, _ = SyntheticGenerator.regime_switching_process(n_steps=2000, seed=42)
        prices_tensor = torch.tensor(prices, dtype=torch.float32)
        log_prices = torch.log(prices_tensor)
        time_len = len(prices)
        
        market_data = torch.zeros(time_len, 1, 5, dtype=torch.float32)
        market_data[:, 0, 3] = log_prices
        market_data[1:, 0, 0] = log_prices[1:] - log_prices[:-1]
        market_data[:, 0, 1] = 0.01 
        market_data[:, 0, 2] = log_prices - 0.0
        market_data[:, 0, 4] = np.log(100.0)
        
        print(f"Running {self.n_simulations} Robustness Simulations...")
        
        for i in tqdm(range(self.n_simulations)):
            # Perturb Parameters
            # We can't easily perturb the internal logic of WalkForwardValidator 
            # without exposing config.
            # Let's just run a single backtest with perturbed params for speed,
            # or WFV if required. "Run parameter perturbation".
            # Let's run a single full backtest (no WFV) with perturbed strategy params.
            
            # Base params
            window = 50
            threshold = 0.0005
            scaling = 5.0
            alpha = 1000.0
            
            # Perturb
            p_window = int(window * np.random.uniform(1 - self.scale, 1 + self.scale))
            p_threshold = threshold * np.random.uniform(1 - self.scale, 1 + self.scale)
            p_scaling = scaling * np.random.uniform(1 - self.scale, 1 + self.scale)
            p_alpha = alpha * np.random.uniform(1 - self.scale, 1 + self.scale)
            
            # Run Single Backtest (Not WFV, to save time, unless WFV is strictly required for Sweep)
            # "Run parameter perturbation... Report Distribution of Sharpe".
            # We reuse the components.
            
            from src.model import HybridDynamicalSystem
            from src.control import ControlAffineSystem
            from src.regime import RegimeDetector
            from src.risk import RiskManager, RiskConfig
            from src.strategy_regime import RegimeSwitchingStrategy
            from src.backtest import BacktestEngine
            
            # Setup
            model = HybridDynamicalSystem(feature_dim=5, epsilon=0.0)
            with torch.no_grad():
                model.physics.raw_kappa.data = torch.tensor([0.5])
                model.physics.raw_gamma.data = torch.tensor([0.1])
            
            sys = ControlAffineSystem(model)
            # Perturbed Regime Detector
            rd = RegimeDetector(window_size=p_window, threshold_trend=p_threshold)
            rm = RiskManager(RiskConfig(base_inventory_limit=5.0))
            
            # Perturbed Strategy
            strat = RegimeSwitchingStrategy(sys, rd, rm, scaling_factor=p_scaling, sigmoid_alpha=p_alpha)
            
            # Wrapper
            class StratWrap:
                def __init__(self, s): self.strat = s
                def solve(self, state, **kwargs):
                    return self.strat.get_action(state).unsqueeze(0).repeat(10, 1, 1), 0.0
            
            # Run
            engine = BacktestEngine(
                StratWrap(strat),
                torch.cat([market_data[0], torch.zeros(1, 1)], dim=1),
                rm,
                dt=0.01
            )
            
            # Suppress tqdm in engine
            # Actually we can't easily suppress it.
            # Just run.
            # To avoid nested tqdm, we could pass verbose=False if supported.
            # It's not.
            
            engine.run(steps=len(market_data)-1, market_data=market_data)
            metrics = engine.get_metrics()
            
            results.append({
                "sharpe": metrics["Sharpe Ratio"],
                "pnl": metrics["Total PnL"],
                "window": p_window,
                "threshold": p_threshold,
                "scaling": p_scaling,
                "alpha": p_alpha
            })
            
        return pd.DataFrame(results)
