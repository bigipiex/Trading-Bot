import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Any
from tqdm import tqdm

from src.synthetic import SyntheticGenerator, OUParams
from src.edge_analysis import DirectSignalStrategy
from src.regime import RegimeDetector
from src.risk import RiskManager, RiskConfig
from src.backtest import BacktestEngine
from src.control import ControlAffineSystem
from src.model import HybridDynamicalSystem

class CapitalSimulator:
    """
    Simulates geometric capital growth and calculates risk metrics.
    """
    def __init__(self, 
                 initial_capital: float = 10000.0,
                 risk_free_rate: float = 0.02):
        self.initial_capital = initial_capital
        self.rf = risk_free_rate
        
    def simulate(self, returns: pd.Series, leverage: float = 1.0) -> Dict[str, Any]:
        """
        returns: Series of PnL (or % returns? BacktestEngine gives PnL in dollars)
                 If inputs are dollar PnL, we need to know the base capital at each step to compute % return.
                 Or we can just simulate Equity(t) = Equity(t-1) + PnL(t) * leverage?
                 No, Leverage multiplies the POSITION size.
                 If we already ran the backtest with a certain position size, we can scale the PnL.
                 
                 Let's assume input 'returns' is percentage returns of the STRATEGY (unlevered).
                 Equity(t) = Equity(t-1) * (1 + ret * leverage)
        """
        # Equity Curve
        equity = [self.initial_capital]
        for r in returns:
            # Geometric growth:
            # New Capital = Old Capital * (1 + r * L)
            # Clip loss to -100% to avoid negative equity math issues (ruin)
            r_levered = max(r * leverage, -1.0)
            equity.append(equity[-1] * (1 + r_levered))
            
        equity_curve = pd.Series(equity)
        
        # Metrics
        total_ret = (equity_curve.iloc[-1] / self.initial_capital) - 1.0
        n_years = len(returns) / 252.0 # Assume daily returns? Or sub-daily? 
        # Backtest is usually minute or tick.
        # Let's assumes steps. If dt=0.01 and we have N steps.
        # We need to know "Time" in years.
        # Let's pass n_years or freq.
        # For synthetic data, T=1000 steps with dt=0.01 is T=10 time units.
        # Let's assume T=1 year = 1.0 time unit? No, usually T=1 year in finance models.
        # Synthetic dt=0.01 usually implies 100 steps = 1 time unit.
        
        # Drawdown
        peaks = equity_curve.cummax()
        drawdowns = (peaks - equity_curve) / peaks
        max_dd = drawdowns.max()
        
        # Time Under Water
        is_underwater = drawdowns > 0
        tuw = is_underwater.mean() # % of time
        
        # Calmar
        cagr = (equity_curve.iloc[-1] / self.initial_capital) ** (1 / max(n_years, 0.1)) - 1 if n_years > 0 else 0.0
        calmar = cagr / max_dd if max_dd > 0 else 0.0
        
        # Ulcer Index
        ulcer = np.sqrt((drawdowns**2).mean())
        
        return {
            "Leverage": leverage,
            "CAGR": cagr,
            "MaxDD": max_dd,
            "Calmar": calmar,
            "TUW": tuw,
            "Ulcer": ulcer,
            "FinalEquity": equity_curve.iloc[-1],
            "Ruin": equity_curve.iloc[-1] < self.initial_capital * 0.1
        }

    def optimize_kelly(self, returns: pd.Series) -> float:
        # Simple Kelly: Mean / Variance (for Gaussian)
        # Or numerical optimization of log growth.
        mu = returns.mean()
        var = returns.var()
        if var == 0: return 0.0
        kelly = mu / var
        # Half-Kelly is safer
        return kelly * 0.5

class StressTester:
    """
    Generates stress scenarios and measures resilience.
    """
    def run_structural_break(self, scenario: str = "FlashCrash") -> Dict:
        # Generate data based on scenario
        if scenario == "FlashCrash":
            # OU process with a sudden 5-sigma drop in the middle
            _, prices = SyntheticGenerator.ornstein_uhlenbeck_process(
                OUParams(mu=100, theta=0.5, sigma=1.0, dt=0.01), n_steps=1000, seed=42
            )
            # Inject crash at t=500
            # 5 sigma drop. Sigma is vol over dt?
            # Drop by 10% instant.
            prices[500:] *= 0.90 
        elif scenario == "TrendReversal":
            # Trend Up then Trend Down
            t1, p1 = SyntheticGenerator.geometric_brownian_motion(0.2, 0.1, 0.01, 500, 100.0, seed=42)
            t2, p2 = SyntheticGenerator.geometric_brownian_motion(-0.2, 0.1, 0.01, 500, p1[-1], seed=43)
            prices = np.concatenate([p1, p2])
        else:
            # Default OU
            _, prices = SyntheticGenerator.ornstein_uhlenbeck_process(
                OUParams(mu=100, theta=0.5, sigma=1.0, dt=0.01), n_steps=1000
            )
            
        # Run Backtest
        metrics = self._run_strategy(prices)
        return metrics

    def run_edge_decay(self, decay_halflife: int = 200) -> Dict:
        # We need to inject decay into the STRATEGY, not the data.
        # This requires a wrapper around the strategy or a flag.
        # Let's handle it in _run_strategy by passing a decay callback or wrapper.
        
        # Generate Standard Data
        _, prices, _ = SyntheticGenerator.regime_switching_process(2000, seed=42)
        
        metrics = self._run_strategy(prices, decay_halflife=decay_halflife)
        return metrics

    def run_regime_error(self, error_prob: float = 0.1) -> Dict:
        _, prices, _ = SyntheticGenerator.regime_switching_process(2000, seed=42)
        metrics = self._run_strategy(prices, regime_error_prob=error_prob)
        return metrics

    def _run_strategy(self, prices: np.ndarray, 
                      decay_halflife: int = None,
                      regime_error_prob: float = 0.0) -> Dict:
        
        # Setup Data
        prices_tensor = torch.tensor(prices, dtype=torch.float32)
        log_prices = torch.log(prices_tensor)
        market_data = torch.zeros(len(prices), 1, 5, dtype=torch.float32)
        market_data[:, 0, 3] = log_prices
        market_data[1:, 0, 0] = log_prices[1:] - log_prices[:-1]
        market_data[:, 0, 1] = 0.01
        market_data[:, 0, 4] = np.log(100.0)
        
        # Setup Strategy
        rd = RegimeDetector(window_size=50)
        rm = RiskManager(RiskConfig(base_inventory_limit=5.0))
        strat = DirectSignalStrategy(rd, rm, w_trend=0.0, w_meanrev=-2.0, sigmoid_alpha=1000.0)
        
        # Strategy Wrapper to inject faults
        class FaultyStratWrapper:
            def __init__(self, s, decay, error_p):
                self.strat = s
                self.decay = decay
                self.error_p = error_p
                self.step_count = 0
                
            def solve(self, state, **kwargs):
                self.step_count += 1
                
                # Get base action
                # We need to hack the regime detector if we want to inject regime error
                # But regime detector is inside strategy.
                # We can inject error into the signal output?
                # "Flip regime label randomly". 
                # DirectSignalStrategy uses `get_signal` internally.
                # We can't easily intercept without subclassing.
                # Let's assume we just add noise to the output control `u` 
                # to simulate "wrong decision"? No, that's different.
                # If regime flips, Trend becomes MR or vice versa.
                # Signal sign might flip.
                
                # Let's assume for this test we proceed with normal strategy
                # but scale output for decay.
                
                u_raw = self.strat.get_action(state)
                
                # Apply Decay
                if self.decay:
                    decay_factor = np.exp(-np.log(2) * self.step_count / self.decay)
                    u_raw *= decay_factor
                    
                # Apply Regime Error (Random Flip of sign?)
                # Regime error means we use Trend logic in MR regime or vice versa.
                # This is hard to simulate perfectly from outside.
                # Approximation: With prob p, multiply u by -1?
                # (Buying instead of selling).
                if np.random.random() < self.error_p:
                    u_raw *= -1.0
                    
                return u_raw.unsqueeze(0).repeat(10, 1, 1), 0.0
                
        engine = BacktestEngine(
            FaultyStratWrapper(strat, decay_halflife, regime_error_prob),
            torch.cat([market_data[0], torch.zeros(1, 1)], dim=1),
            rm,
            dt=0.01
        )
        
        engine.run(steps=len(prices)-1, market_data=market_data)
        return engine.get_metrics()

class CapacityAnalyzer:
    def estimate_capacity(self, daily_volumes: List[float] = [1e4, 1e5, 1e6, 1e7]) -> pd.DataFrame:
        results = []
        # Base Data
        _, prices, _ = SyntheticGenerator.regime_switching_process(1000, seed=42)
        
        for vol in daily_volumes:
            # Run Backtest with this volume limit
            # Setup...
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

            # Use new Market Impact Coeff
            # impact_coeff ~ 0.1 to 1.0 usually.
            engine = BacktestEngine(
                StratWrap(strat),
                torch.cat([market_data[0], torch.zeros(1, 1)], dim=1),
                rm,
                dt=0.01,
                daily_volume=vol,
                market_impact_coeff=0.5 # Significant impact
            )
            
            engine.run(steps=len(prices)-1, market_data=market_data)
            metrics = engine.get_metrics()
            
            results.append({
                "DailyVolume": vol,
                "Sharpe": metrics["Sharpe Ratio"],
                "PnL": metrics["Total PnL"],
                "Cost": metrics["Total PnL"] - metrics.get("Gross PnL", metrics["Total PnL"]) # Need gross to calc cost?
                # engine history has 'cost'
            })
            
        return pd.DataFrame(results)
