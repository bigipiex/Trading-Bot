import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field
from typing import List

@dataclass
class RiskConfig:
    max_leverage: float = 2.0
    base_inventory_limit: float = 1.0
    volatility_target: float = 0.05
    drawdown_limit: float = 0.10       # 10% Start scaling down
    max_drawdown_limit: float = 0.25   # 25% Stop trading
    barrier_weight: float = 100.0
    
    # Tail Risk Params
    vol_circuit_window: int = 50
    vol_sigma_threshold: float = 3.0   # 3-sigma event triggers breaker
    vol_sigmoid_tau: float = 0.1       # Smoothness of breaker
    
    jump_sigma_threshold: float = 4.0  # 4-sigma return triggers freeze
    jump_freeze_steps: int = 20        # Steps to freeze
    
class RiskManager(nn.Module):
    """
    Enforces risk constraints on the trading system.
    Includes Tail Risk Protection.
    """
    def __init__(self, config: RiskConfig):
        super().__init__()
        self.config = config
        
        # State for Tail Risk
        self.vol_history: List[float] = []
        self.return_history: List[float] = []
        self.freeze_counter: int = 0
        self.peak_equity: float = 1.0 # Normalized
        
    def update_state(self, current_vol: float, current_return: float, current_equity: float):
        """Updates internal history for risk metrics."""
        self.vol_history.append(current_vol)
        if len(self.vol_history) > self.config.vol_circuit_window * 2:
            self.vol_history.pop(0)
            
        self.return_history.append(current_return)
        if len(self.return_history) > self.config.vol_circuit_window * 2:
            self.return_history.pop(0)
            
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
            
        # Decrement freeze
        if self.freeze_counter > 0:
            self.freeze_counter -= 1
            
    def get_tail_risk_scale(self, current_vol: float, current_return: float, current_equity: float) -> float:
        """
        Computes the scaling factor (0.0 to 1.0) for the position size based on:
        1. Volatility Circuit Breaker (MAD-based)
        2. Jump Filter
        3. Drawdown Throttle (Convex)
        """
        # 0. Update state
        self.update_state(current_vol, current_return, current_equity)
        
        scale = 1.0
        
        # 1. Volatility Circuit Breaker (Robust MAD)
        if len(self.vol_history) >= self.config.vol_circuit_window:
            hist = np.array(self.vol_history[:-1])
            if len(hist) > 2:
                # Use Median and MAD instead of Mean/Std
                vol_median = np.median(hist)
                vol_mad = np.median(np.abs(hist - vol_median))
                # Standardize MAD to sigma equivalent (approx 1.4826 * MAD for normal)
                vol_sigma_equiv = 1.4826 * vol_mad
                
                vol_threshold = vol_median + self.config.vol_sigma_threshold * vol_sigma_equiv
                
                # Sigmoid scaling
                x = (vol_threshold - current_vol) / self.config.vol_sigmoid_tau
                x = max(min(x, 20.0), -20.0)
                vol_scale = 1.0 / (1.0 + np.exp(-x))
                
                # Log activation frequency (can't easily log to file here without passing logger)
                # But we can store it in self.stats if we had one.
                
                scale *= vol_scale
                
        # 2. Jump Filter (Hard Freeze with Ramp)
        if abs(current_return) > self.config.jump_sigma_threshold * current_vol:
            self.freeze_counter = self.config.jump_freeze_steps
            
        if self.freeze_counter > 0:
            ramp = (self.config.jump_freeze_steps - self.freeze_counter) / self.config.jump_freeze_steps
            scale *= (ramp ** 2)
            
        # 3. Drawdown-Based Risk Throttle (Convex)
        dd = (self.peak_equity - current_equity) / self.peak_equity
        
        if dd > self.config.drawdown_limit:
            # Convex ramp: 1 - sigmoid((dd - dd_start)/tau)
            # We want scale=1 at dd_limit, scale=0 at max_dd_limit.
            # Sigmoid is centered at 0.5.
            # Let's map [dd_limit, max_dd_limit] to [-3, 3] in sigmoid space?
            # Or use simple convex power function:
            # Scale = ( (MaxDD - DD) / (MaxDD - Limit) ) ^ 2
            
            denom = self.config.max_drawdown_limit - self.config.drawdown_limit
            if denom > 0:
                rel_pos = (self.config.max_drawdown_limit - dd) / denom
                rel_pos = max(0.0, min(1.0, rel_pos))
                throttle = rel_pos ** 2 # Convex
                scale *= throttle
            else:
                scale = 0.0
            
        return scale

    def get_volatility_adjusted_limit(self, current_vol: torch.Tensor) -> torch.Tensor:
        """
        Computes dynamic inventory limit based on inverse volatility.
        I_max = Base * (Target / Vol)
        """
        # Avoid division by zero
        safe_vol = torch.clamp(current_vol, min=1e-6)
        
        # Scale factor: If vol is high, limit decreases.
        # If vol is low, limit increases (up to a cap?).
        # Let's cap the multiplier at 2.0 to avoid excessive leverage in calm markets.
        vol_scaler = torch.clamp(self.config.volatility_target / safe_vol, max=2.0)
        
        return self.config.base_inventory_limit * vol_scaler
        
    def barrier_penalty(self, trajectory: torch.Tensor, volatility: torch.Tensor) -> torch.Tensor:
        """
        Computes soft constraint penalty for MPC objective.
        trajectory: (Time, Batch, State+Inventory)
        """
        inventory = trajectory[:, :, -1:]
        
        # Compute limit for each time step (assuming vol is constant over horizon or passed as seq)
        # Here we simplify: use current vol for the whole horizon?
        # Ideally vol is part of the state trajectory (index 1).
        traj_vol = trajectory[:, :, 1:2] # Volatility feature
        
        i_max = self.get_volatility_adjusted_limit(traj_vol)
        
        # Barrier: ReLU(|I| - I_max)^2
        violation = torch.relu(torch.abs(inventory) - i_max)
        return self.config.barrier_weight * torch.sum(violation**2)
        
    def check_drawdown(self, equity_curve: torch.Tensor) -> bool:
        """
        Circuit breaker: Returns True if trading should STOP.
        Deprecated: Logic moved to get_tail_risk_scale
        """
        peak = torch.max(equity_curve)
        current = equity_curve[-1]
        dd = (peak - current) / peak
        return dd > self.config.max_drawdown_limit
        
    def project_controls(self, u: torch.Tensor, current_inventory: torch.Tensor, 
                        volatility: torch.Tensor, dt: float,
                        current_return: float = 0.0, current_equity: float = 1.0) -> torch.Tensor:
        """
        Projects control action u onto the feasible set to ensure next inventory
        stays within limits.
        
        NOW INCLUDES TAIL RISK SCALING.
        """
        # Calculate Tail Risk Scale
        # We need float values for state update
        vol_val = volatility.item() if isinstance(volatility, torch.Tensor) else volatility
        # current_return and current_equity passed from BacktestEngine?
        # Yes, we need to update signature in BacktestEngine.
        
        scale = self.get_tail_risk_scale(vol_val, current_return, current_equity)
        
        # Apply scaling to u directly?
        # Or to the LIMITS?
        # If we scale u, we slow down trading.
        # If we scale limits, we force liquidation (if current_inv > scaled_limit).
        # "I_final = I_mpc * leverage_scale" implies scaling the POSITION.
        # So we should scale the TARGET inventory.
        # u drives I towards target.
        # If we project u such that I_next is within SCALED limits.
        
        # Get base limit
        i_max_base = self.get_volatility_adjusted_limit(volatility)
        
        # Apply Tail Scale to Limit
        i_max_scaled = i_max_base * scale
        
        # Force liquidation if current inventory exceeds new limit
        # This will happen naturally if we clamp u such that I_next is inside i_max_scaled.
        
        # Bounds for u
        u_upper = (i_max_scaled - current_inventory) / dt
        u_lower = (-i_max_scaled - current_inventory) / dt
        
        # Also clamp u by explicit rate limit (liquidity constraint)
        u_rate_limit = self.config.base_inventory_limit * 5.0 
        
        u_clamped = torch.clamp(u, min=u_lower, max=u_upper)
        u_clamped = torch.clamp(u_clamped, min=-u_rate_limit, max=u_rate_limit)
        
        return u_clamped
