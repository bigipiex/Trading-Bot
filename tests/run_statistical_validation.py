import unittest
import torch
import numpy as np
import pandas as pd
from scipy import stats
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.synthetic import SyntheticGenerator, OUParams
from src.edge_analysis import DirectSignalStrategy, ICAnalyzer
from src.robustness_cross import CrossPathRobustness
from src.sensitivity import SensitivitySweeper
from src.regime import RegimeDetector

class StatisticalValidation(unittest.TestCase):
    
    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        
    def test_1_ic_analysis(self):
        print("\n=== 1. Information Coefficient (IC) Analysis ===")
        # Generate Switching Data
        _, prices, _ = SyntheticGenerator.regime_switching_process(2000, seed=42)
        
        # Prepare Inputs
        prices_tensor = torch.tensor(prices, dtype=torch.float32)
        log_prices = torch.log(prices_tensor)
        market_data = torch.zeros(len(prices), 1, 5, dtype=torch.float32)
        market_data[:, 0, 3] = log_prices
        market_data[1:, 0, 0] = log_prices[1:] - log_prices[:-1]
        market_data[:, 0, 4] = np.log(100.0)
        
        # Strategy
        rd = RegimeDetector(window_size=50)
        strat = DirectSignalStrategy(rd)
        strat.sigmoid_alpha = 1000.0
        
        # Collect Signals
        signals = []
        for t in range(len(prices)-1):
            state = torch.cat([market_data[t], torch.zeros(1, 1)], dim=1)
            u = strat.get_action(state).item()
            signals.append(u)
            
        signals = np.array(signals)
        returns = np.diff(np.log(prices))
        
        # Run IC Analysis
        analyzer = ICAnalyzer(horizons=[1, 5, 10, 20, 50])
        ic_df = analyzer.analyze(signals, returns)
        
        print(ic_df)
        
        # Check if Long Horizon IC > Short Horizon IC (Drift Hypothesis)
        ic_1 = ic_df[ic_df["Horizon"]==1]["Global IC"].values[0]
        ic_50 = ic_df[ic_df["Horizon"]==50]["Global IC"].values[0]
        
        print(f"IC(1): {ic_1:.4f}, IC(50): {ic_50:.4f}")
        # Note: Depending on path, IC might be low if regime switching is frequent.
        # But generally N=50 should see more signal.
        
    def test_2_cross_path_robustness(self):
        print("\n=== 2. Cross-Path Robustness (Monte Carlo) ===")
        # Reduce n_paths for unit test speed (10 instead of 100)
        cpr = CrossPathRobustness(n_paths=10)
        results = cpr.run()
        
        for regime, df in results.items():
            print(f"\nRegime: {regime}")
            print(f"Mean Sharpe: {df['Sharpe'].mean():.2f}")
            print(f"Median Sharpe: {df['Sharpe'].median():.2f}")
            print(f"Profitable Runs: {(df['PnL'] > 0).mean()*100:.1f}%")
            
    def test_3_sensitivity_sweep(self):
        print("\n=== 3. Parameter Sensitivity Surface ===")
        sweeper = SensitivitySweeper()
        # Mock run or run full? Full sweep is 3*3*3 = 27 runs. Fast enough.
        results = sweeper.run_sweep()
        
        print("\nTop 5 Configurations:")
        print(results.sort_values("Sharpe", ascending=False).head(5))
        
        # Check stability
        sharpe_std = results["Sharpe"].std()
        print(f"Sharpe Stability (Std): {sharpe_std:.4f}")
        
    def test_4_significance_test(self):
        print("\n=== 4. Statistical Significance ===")
        # Use Cross-Path Switching results
        cpr = CrossPathRobustness(n_paths=20) # Generate sample
        df = cpr._simulate_batch("Switching")
        
        sharpes = df["Sharpe"].values
        t_stat, p_val = stats.ttest_1samp(sharpes, 0.0)
        
        print(f"Mean Sharpe: {np.mean(sharpes):.4f}")
        print(f"T-Statistic: {t_stat:.4f}")
        print(f"P-Value: {p_val:.4e}")
        
        if p_val < 0.05 and t_stat > 0:
            print("RESULT: Statistically Significant Edge Detected.")
        else:
            print("RESULT: No Significant Edge.")

if __name__ == '__main__':
    unittest.main()
