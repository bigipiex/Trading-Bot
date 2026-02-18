import torch
import torch.nn as nn
import numpy as np
from src.control import ControlAffineSystem
from src.risk import RiskManager
from src.regime import RegimeDetector
from src.solver import odeint_rk4

class RegimeSwitchingStrategy(nn.Module):
    """
    Switching strategy that blends Mean Reversion and Momentum signals
    based on detected market regime using Soft Switching.
    
    u_t = w_MR(t) * u_MR + w_MOM(t) * u_MOM
    
    w_MOM = sigmoid(alpha * TrendStrength)
    w_MR = 1 - w_MOM
    """
    def __init__(self, 
                 system: ControlAffineSystem, 
                 regime_detector: RegimeDetector,
                 risk_manager: RiskManager = None,
                 scaling_factor: float = 1.0,
                 sigmoid_alpha: float = 1000.0): # Controls sharpness of switch
        super().__init__()
        self.system = system
        self.regime_detector = regime_detector
        self.risk_manager = risk_manager
        self.scaling_factor = scaling_factor
        self.sigmoid_alpha = sigmoid_alpha
        self.dt = 0.01
        
        # Internal state to track history for regime detection
        # We need a buffer of prices
        self.price_history = []
        self.window_size = regime_detector.window_size
        
    def update_history(self, price: float):
        self.price_history.append(price)
        if len(self.price_history) > self.window_size * 2:
            self.price_history.pop(0)
            
    def get_action(self, state: torch.Tensor) -> torch.Tensor:
        """
        Computes u_t based on regime.
        state: (Batch, Features)
        We assume Batch=1 for backtesting strategy usually.
        """
        # 1. Update History & Detect Regime
        current_log_price = state[0, 3].item()
        current_price = np.exp(current_log_price)
        self.update_history(current_price)
        
        trend_strength = 0.0
        
        if len(self.price_history) >= self.window_size:
            # Run detection on history
            prices_array = np.array(self.price_history)
            _, slopes, _ = self.regime_detector.detect(prices_array)
            trend_strength = slopes[-1]
            
        # 2. Compute Signals
        vol = state[:, 1:2]
        safe_vol = torch.clamp(vol, min=1e-6)
        
        # Signal A: Mean Reversion (using Model Prediction)
        # Predict 50 steps ahead
        horizon = 50
        u_zero = torch.zeros(state.shape[0], 1, device=state.device)
        
        def dynamics_closure(time, s):
            return self.system(time, s, u_zero)
            
        t_span = torch.tensor([0, horizon * self.dt], device=state.device)
        traj = odeint_rk4(dynamics_closure, state, t_span)
        final_state = traj[-1]
        expected_return = final_state[:, 3:4] - state[:, 3:4]
        
        u_mr = self.scaling_factor * expected_return / safe_vol
        
        # Signal B: Momentum (using Trend Strength)
        # u_mom ~ trend / vol
        # Note: We don't need sign() anymore if we want smooth transition?
        # But u_mom direction is determined by sign of trend.
        # So yes, u_mom is proportional to trend.
        u_mom = torch.tensor([[trend_strength]], dtype=torch.float32) * self.scaling_factor * 100.0 / safe_vol
        # Scaled up because trend_strength is usually small (e.g. 0.0001)
        
        # 3. Soft Switching
        # w_mom = sigmoid(alpha * trend)
        # If trend > 0, w_mom > 0.5. 
        # Wait, if trend is positive, we want Momentum (Buy).
        # If trend is negative, we want Momentum (Sell).
        # And if trend is near zero, we want Mean Reversion.
        # So we need w_mom based on MAGNITUDE of trend?
        # "w_trend = sigmoid(alpha * S)" where S is "normalized trend strength".
        # If S is |trend|, then large trend -> w_trend=1 (Pure Momentum).
        # Small trend -> w_trend=0.5 (Mixed) -> No, sigmoid(0)=0.5.
        # We want w_trend -> 0 when S -> 0.
        # So we should use sigmoid(alpha * (|trend| - threshold))?
        # Or just S = |trend|.
        # Let's use a shifted sigmoid to center transition around a "threshold".
        # Or simpler: w_mom = |tanh(alpha * trend)| ?
        # tanh(0) = 0. tanh(large) = 1.
        # This seems better for "Magnitude of Trend activates Momentum".
        
        # User specified: w_trend = sigmoid(alpha * S)
        # Let's assume S = |trend| - threshold.
        # Threshold is roughly 0.0002 from previous tuning.
        threshold = 0.0002
        S = abs(trend_strength) - threshold
        w_mom = torch.sigmoid(torch.tensor(self.sigmoid_alpha * S))
        w_mr = 1.0 - w_mom
        
        u_raw = w_mom * u_mom + w_mr * u_mr
            
        # 4. Risk Projection
        if self.risk_manager:
            current_inv = state[:, -1:]
            u_executed = self.risk_manager.project_controls(
                u_raw, current_inv, vol, self.dt
            )
        else:
            u_executed = u_raw
            
        return u_executed
