import unittest
import torch
import numpy as np
import pandas as pd
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.synthetic import SyntheticGenerator, OUParams
from src.economic_analysis import CapitalSimulator, StressTester, CapacityAnalyzer
from src.backtest import BacktestEngine
from src.regime import RegimeDetector
from src.edge_analysis import DirectSignalStrategy
from src.risk import RiskManager, RiskConfig

class EconomicValidation(unittest.TestCase):
    
    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        
    def test_1_capital_growth_and_leverage(self):
        print("\n=== 1. Capital Growth & Leverage Analysis ===")
        # Generate Returns (Base Strategy)
        _, prices, _ = SyntheticGenerator.regime_switching_process(2000, seed=42)
        # Use simple returns simulation or run full backtest?
        # Run backtest to get PnL series
        # Then feed to CapitalSimulator
        
        # Setup Backtest
        prices_tensor = torch.tensor(prices, dtype=torch.float32)
        log_prices = torch.log(prices_tensor)
        market_data = torch.zeros(len(prices), 1, 5, dtype=torch.float32)
        market_data[:, 0, 3] = log_prices
        market_data[1:, 0, 0] = log_prices[1:] - log_prices[:-1]
        market_data[:, 0, 1] = 0.01
        market_data[:, 0, 4] = np.log(100.0)
        
        rd = RegimeDetector(window_size=50)
        rm = RiskManager(RiskConfig(base_inventory_limit=5.0))
        strat = DirectSignalStrategy(rd, rm, w_trend=0.0, w_meanrev=-2.0, sigmoid_alpha=1000.0)
        
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
        
        # Extract PnL (Dollar PnL)
        # Convert to % Returns assuming Base Capital
        pnl_series = pd.Series(engine.history["pnl"])
        base_capital = 10000.0
        pct_returns = pnl_series / base_capital
        
        # Simulate Leverage Levels
        sim = CapitalSimulator(initial_capital=base_capital)
        leverages = [0.5, 1.0, 2.0, 3.0]
        
        for lev in leverages:
            res = sim.simulate(pct_returns, leverage=lev)
            print(f"Leverage {lev}x: CAGR {res['CAGR']*100:.1f}%, MaxDD {res['MaxDD']*100:.1f}%, Calmar {res['Calmar']:.2f}")
            
        # Kelly Optimization
        kelly = sim.optimize_kelly(pct_returns)
        print(f"Optimal Kelly Fraction: {kelly:.2f}")
        
    def test_2_structural_break(self):
        print("\n=== 2. Structural Break & Stress Test ===")
        stress = StressTester()
        
        scenarios = ["FlashCrash", "TrendReversal"]
        for sc in scenarios:
            res = stress.run_structural_break(sc)
            print(f"Scenario {sc}: Sharpe {res['Sharpe Ratio']:.2f}, MaxDD {res['Max Drawdown']:.2f}")
            
    def test_3_edge_decay(self):
        print("\n=== 3. Edge Decay Simulation ===")
        # Simulate gradual loss of alpha
        # Half-life 200 steps
        # We need to inject this into the backtest loop.
        # Implemented in StressTester via wrapper.
        
        stress = StressTester()
        res = stress.run_edge_decay(decay_halflife=200)
        print(f"Decay (HL=200): Sharpe {res['Sharpe Ratio']:.2f}, PnL {res['Total PnL']:.2f}")
        
        # Check if Sharpe degraded compared to no decay (Base ~ 1.0)
        # With decay it should be lower.
        
    def test_4_liquidity_capacity(self):
        print("\n=== 4. Liquidity & Capacity Analysis ===")
        cap = CapacityAnalyzer()
        # Test volumes: Low to High
        vols = [1e3, 1e4, 1e5, 1e6] # 1e3 is very low liquidity
        res = cap.estimate_capacity(daily_volumes=vols)
        
        print(res)
        
        # Find capacity limit (Sharpe < 0)
        limit_vol = res[res["Sharpe"] < 0]["DailyVolume"].max()
        if pd.isna(limit_vol):
            print("Capacity Limit: > 1e6")
        else:
            print(f"Capacity Limit (Min Volume): ~{limit_vol:.0f}")
            
    def test_5_regime_sensitivity(self):
        print("\n=== 5. Regime Error Sensitivity ===")
        stress = StressTester()
        probs = [0.0, 0.1, 0.2, 0.3]
        
        for p in probs:
            res = stress.run_regime_error(error_prob=p)
            print(f"Error Prob {p*100:.0f}%: Sharpe {res['Sharpe Ratio']:.2f}")

    def test_6_bootstrap_survival(self):
        print("\n=== 6. Bootstrap Survival Probability ===")
        # Use returns from Test 1
        # Generate new returns for simplicity
        _, prices, _ = SyntheticGenerator.regime_switching_process(2000, seed=42)
        # Assume mean 0.05% per step, std 1% per step
        returns = np.diff(np.log(prices))
        # Add strategy edge (sharpe ~ 1.0 means mean ~ std/sqrt(252)? No.
        # Sharpe = mean/std * sqrt(252).
        # If Sharpe=1, mean = std/16.
        # Let's use the actual backtest returns from test 1 if possible.
        # Re-run quickly.
        
        prices_tensor = torch.tensor(prices, dtype=torch.float32)
        log_prices = torch.log(prices_tensor)
        market_data = torch.zeros(len(prices), 1, 5, dtype=torch.float32)
        market_data[:, 0, 3] = log_prices
        market_data[1:, 0, 0] = log_prices[1:] - log_prices[:-1]
        market_data[:, 0, 1] = 0.01
        market_data[:, 0, 4] = np.log(100.0)
        
        rd = RegimeDetector(window_size=50)
        rm = RiskManager(RiskConfig(base_inventory_limit=5.0))
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
        pnl = np.array(engine.history["pnl"])
        
        # Bootstrap
        n_boot = 1000
        sharpes = []
        for _ in range(n_boot):
            sample = np.random.choice(pnl, size=len(pnl), replace=True)
            if sample.std() > 0:
                s = (sample.mean() / sample.std()) * np.sqrt(1/0.01) # Annualize? dt=0.01 implies 100 steps=1.
                # If 1 year = 252 days, 1 day = ?
                # Just use relative Sharpe.
                sharpes.append(s)
                
        sharpes = np.array(sharpes)
        prob_pos = (sharpes > 0).mean()
        print(f"Bootstrap Prob(Sharpe > 0): {prob_pos*100:.1f}%")
        print(f"Mean Bootstrap Sharpe: {sharpes.mean():.2f}")

if __name__ == '__main__':
    unittest.main()
