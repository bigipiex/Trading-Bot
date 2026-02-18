import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from tqdm import tqdm

from src.mpc import MPCSolver
from src.risk import RiskManager
from src.control import ControlAffineSystem
from src.solver import rk4_step

class BacktestEngine:
    """
    Simulates the closed-loop system:
    1. Observe State X_t
    2. Optimize u_t* (MPC)
    3. Project u_t (Risk)
    4. Execute u_t (Simulate Next Step)
    5. Log Metrics
    """
    def __init__(self, 
                 mpc_solver: MPCSolver, 
                 initial_state: torch.Tensor,
                 risk_manager: RiskManager = None,
                 transaction_cost_bps: float = 10.0, # Higher fee for crypto/swing (10bps)
                 slippage_bps: float = 5.0,         # Higher base slippage
                 vol_slippage_coeff: float = 1.0,   
                 market_impact_coeff: float = 0.1,  # Impact matters more for larger positions
                 daily_volume: float = 1e8,         # Crypto daily vol (e.g. 100M)
                 dt: float = 4/24):                 # Default 4H candles
        self.mpc = mpc_solver
        self.state = initial_state.clone() # (Batch, StateDim+Inventory)
        self.risk_manager = risk_manager
        
        # Costs
        self.tc = transaction_cost_bps * 1e-4
        self.slippage = slippage_bps * 1e-4
        self.vol_slippage_coeff = vol_slippage_coeff
        self.market_impact_coeff = market_impact_coeff
        self.daily_volume = daily_volume
        self.dt = dt
        
        # Logs
        self.history = {
            "time": [],
            "price": [],
            "inventory": [],
            "control": [],
            "pnl": [],
            "equity": [],
            "cost": [],
            "volatility": [],
            "energy": [] # Stability metric
        }
        self.current_equity = 0.0
        
    def step(self, t: float, external_forcing: torch.Tensor = None):
        """
        Advances the simulation by one step.
        
        Args:
            t: Current time
            external_forcing: Optional external market movement (e.g., real price data drift)
                              If None, system evolves purely by internal dynamics (self-driving).
                              Usually we want to feed real market data drift here.
        """
        # 1. Observe State
        current_inv = self.state[:, -1:]
        current_price = self.state[:, 3:4] # LogPrice
        current_vol = self.state[:, 1:2]
        
        # 2. Run MPC
        # We optimize over horizon from current state
        # Note: In a real backtest, we might only run MPC every N steps or re-plan every step.
        # Here we re-plan every step (Receding Horizon Control).
        u_seq, _ = self.mpc.solve(self.state, iterations=10, lr=0.1) # Fewer iters for speed?
        u_optimal = u_seq[0] # Take first action
        
        # 3. Apply Risk Constraints (Projection)
        if self.risk_manager:
            # Need to pass extra args for Tail Risk
            # current_return (from last step? or just 0 if not tracked here)
            # We can approximate current_return as log price diff from last step.
            # But BacktestEngine doesn't store prev price easily in step().
            # Let's use history if available.
            
            curr_ret = 0.0
            if len(self.history["price"]) > 1:
                curr_ret = np.log(self.history["price"][-1]) - np.log(self.history["price"][-2])
            
            # current_equity is self.current_equity (normalized? RiskManager expects normalized peak=1.0?)
            # RiskManager tracks peak internally based on inputs.
            # We should pass the actual equity value, RiskManager will normalize relative to its own seen peak.
            # Wait, RiskManager.peak_equity init is 1.0. 
            # If our equity starts at 0 (long/short), this might be weird.
            # Let's assume CapitalSimulator handles the "Growth" aspect.
            # BacktestEngine tracks "Accumulated PnL".
            # If we treat this as "Return on Capital", we need base capital.
            # Let's pass self.current_equity assuming it's accumulated PnL.
            # RiskManager logic: dd = (peak - current) / peak.
            # If peak=100, current=90, dd=0.1.
            # If peak=0, current=-10... division by zero.
            # BacktestEngine usually assumes purely PnL accumulation starting from 0.
            # To use Drawdown Control properly, we should probably simulate "Total Equity" starting from Base Capital.
            # But we don't have Base Capital here.
            # Let's disable Drawdown Control in BacktestEngine unless we hack a base capital.
            # Or pass a mock value.
            # Ideally, we update BacktestEngine to accept initial_capital.
            
            # For now, pass 1.0 + current_equity (assuming 1.0 is base)
            # This is a hack but works for relative drawdown if base is 1.0 unit.
            
            u_executed = self.risk_manager.project_controls(
                u_optimal, current_inv, current_vol, self.dt,
                current_return=curr_ret,
                current_equity=1.0 + self.current_equity
            )
        else:
            u_executed = u_optimal
            
        # 4. Simulate Execution & Next State
        # We use the system dynamics to evolve the state
        # But wait! In a backtest on historical data, the MARKET state (Price, Vol) comes from data.
        # The INVENTORY state comes from our actions.
        # If we use the model to evolve market state, we are hallucinating a future.
        # We should use REAL data for market state update if available.
        
        if external_forcing is not None:
            # Hybrid Update:
            # Inventory evolves by our action: I_next = I + u * dt
            # Market evolves by DATA: X_next = Data_next
            # But we need to maintain consistency.
            
            # Let's assume external_forcing IS the next market state X_{t+1} (excluding inventory)
            # state: (Batch, 5) -> 4 market + 1 inventory
            next_market_state = external_forcing
            
            # Update Inventory
            next_inventory = current_inv + u_executed * self.dt
            
            # Combine
            self.state = torch.cat([next_market_state, next_inventory], dim=1)
            
            # Calculate realized price change from Data
            # forcing is LogPrice
            next_price = next_market_state[:, 3:4]
            price_change = torch.exp(next_price) - torch.exp(current_price) # Real $ change
            
        else:
            # Pure Simulation (Hallucination Mode / Synthetic Test)
            def dynamics_closure(time, state):
                return self.mpc.system(time, state, u_executed)
            
            self.state = rk4_step(dynamics_closure, t, self.state, self.dt)
            
            # Price change from model
            next_price = self.state[:, 3:4]
            price_change = torch.exp(next_price) - torch.exp(current_price)

        # 5. Calculate PnL
        # Mark-to-Market PnL: Inventory * (Price_{t+1} - Price_t)
        # We use real dollar price change
        # Assuming Inventory is in units of asset
        pnl_gross = current_inv * price_change
        
        # Transaction Costs: |u| * Price * (Fee + Slippage)
        # u is rate per second. Volume = |u| * dt
        trade_volume = torch.abs(u_executed) * self.dt
        asset_price = torch.exp(current_price)
        
        # Volatility-Adjusted Slippage
        # total_slippage = base_slippage + vol_coeff * volatility
        current_vol_val = current_vol.item()
        
        # Market Impact (Square Root Law)
        # Impact ~ sigma * sqrt(OrderSize / DailyVolume)
        # OrderSize = trade_volume (units)
        # We need a coefficient to scale it properly.
        # Let's assume market_impact_coeff handles the scaling.
        impact_slippage = 0.0
        if self.market_impact_coeff > 0 and trade_volume.item() > 0:
            impact_slippage = self.market_impact_coeff * current_vol_val * \
                              torch.sqrt(trade_volume / (self.daily_volume + 1e-6))
        
        total_slippage = self.slippage + \
                         self.vol_slippage_coeff * current_vol_val * 1e-4 + \
                         impact_slippage
        
        cost = trade_volume * asset_price * (self.tc + total_slippage)
        
        pnl_net = pnl_gross - cost
        
        self.current_equity += pnl_net.item()
        
        # 6. Log Metrics
        self.history["time"].append(t)
        self.history["price"].append(asset_price.item())
        self.history["inventory"].append(current_inv.item())
        self.history["control"].append(u_executed.item())
        self.history["pnl"].append(pnl_net.item())
        self.history["equity"].append(self.current_equity)
        self.history["cost"].append(cost.item())
        self.history["volatility"].append(current_vol.item())
        
        # Energy Metric: alpha * P^2 + beta * I^2 (using log price deviation from 0?)
        energy = 0.5 * (current_price.item()**2) + 0.5 * (current_inv.item()**2)
        self.history["energy"].append(energy)

    def run(self, steps: int, market_data: torch.Tensor = None):
        """
        Runs the backtest loop.
        market_data: Tensor (Steps, Batch, MarketStateDim)
        """
        for i in tqdm(range(steps)):
            t = i * self.dt
            # market_data[i] has shape (Batch, MarketStateDim)
            # If market_data is provided, it dictates the NEXT state.
            # But step() updates self.state to match this.
            # Ideally, market_data[i] is the observation at time t_i.
            # So we pass market_data[i+1] as the forcing for the next step?
            # No, let's say market_data[i] is the observation we see NOW at step i.
            # Wait, step() logic:
            # 1. Observe (using self.state which was set in previous step)
            # 2. Act
            # 3. Evolve to next state (using forcing)
            # So forcing should be market_data[i+1]
            
            forcing = None
            if market_data is not None:
                if i + 1 < len(market_data):
                    forcing = market_data[i+1]
                else:
                    break # End of data
            
            self.step(t, forcing)
            
    def get_metrics(self) -> Dict:
        """Computes summary statistics."""
        df = pd.DataFrame(self.history)
        
        total_pnl = df["equity"].iloc[-1]
        returns = df["pnl"]
        sharpe = (returns.mean() / returns.std()) * np.sqrt(1/self.dt) if returns.std() > 0 else 0.0
        
        # Max Drawdown
        peaks = df["equity"].cummax()
        drawdowns = (peaks - df["equity"]) / (peaks.abs() + 1e-6) # Relative to peak equity? No, usually relative to capital.
        # If starting capital is 0 (long/short), DD is absolute.
        max_dd = (peaks - df["equity"]).max()
        
        # Turnover: sum(|u|*dt) / steps
        turnover = np.abs(df["control"]).mean() * self.dt
        
        # Control Efficiency: Corr(u, price_change)
        # Shift price to align with action? u_t causes PnL at t+1
        price_diff = df["price"].diff().fillna(0)
        control_eff = df["control"].corr(price_diff)
        
        # Constraint Activation Frequency
        # Assume base limit 1.0
        limit = 1.0 # Simplify
        constraints_hit = (df["inventory"].abs() > 0.99 * limit).mean()
        
        return {
            "Total PnL": total_pnl,
            "Sharpe Ratio": sharpe,
            "Max Drawdown": max_dd,
            "Turnover": turnover,
            "Control Efficiency": control_eff,
            "Constraint Freq": constraints_hit,
            "Final Energy": df["energy"].iloc[-1]
        }
