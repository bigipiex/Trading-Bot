import torch
import torch.nn as nn
from src.control import ControlAffineSystem
from src.risk import RiskManager
from src.solver import odeint_rk4

class SimpleStrategy(nn.Module):
    """
    Implements a simple volatility-scaled position sizing strategy
    based on the model's prediction.
    
    Logic:
    1. Compute predicted return for next N steps: E[P_{t+N} - P_t]
    2. Size position: u = k * E[Return] / Volatility
    3. Apply Risk Projection (Hard Constraints)
    """
    def __init__(self, 
                 system: ControlAffineSystem, 
                 risk_manager: RiskManager = None,
                 scaling_factor: float = 1.0,
                 prediction_horizon: int = 1):
        super().__init__()
        self.system = system
        self.risk_manager = risk_manager
        self.scaling_factor = scaling_factor
        self.prediction_horizon = prediction_horizon
        self.dt = 0.01
        
    def get_action(self, state: torch.Tensor) -> torch.Tensor:
        """
        Computes the control action u_t given current state X_t.
        """
        batch_size = state.shape[0]
        u_zero = torch.zeros(batch_size, 1, device=state.device)
        
        # 1. Predict Expected Return
        if self.prediction_horizon <= 1:
            # Instantaneous Drift
            dx_dt_aug = self.system(0.0, state, u_zero)
            expected_return = dx_dt_aug[:, 3:4] # dP/dt
        else:
            # Multi-Step Rollout (Integration)
            # We assume u=0 for prediction (passive drift)
            def dynamics_closure(time, s):
                # ControlAffineSystem expects (t, x_aug, u)
                return self.system(time, s, u_zero)

            # Integrate
            t_span = torch.tensor([0, self.prediction_horizon * self.dt], device=state.device)
            # odeint returns [x0, xN]
            traj = odeint_rk4(dynamics_closure, state, t_span)
            final_state = traj[-1]
            
            # Predicted Price Change (Log Price is index 3)
            # state is [LogRet, Vol, Mom, LogPrice, Mu, Inventory]
            expected_return = final_state[:, 3:4] - state[:, 3:4]
            
            # Normalize return by horizon to keep scale comparable to dP/dt?
            # Or just use total return.
            # Strategy scaling factor 'k' handles magnitude.
            # Let's use total return.
        
        # 2. Extract Volatility (index 1)
        vol = state[:, 1:2]
        safe_vol = torch.clamp(vol, min=1e-6)
        
        # 3. Simple Sizing Rule
        # u ~ ExpectedReturn / Vol
        raw_u = self.scaling_factor * expected_return / safe_vol
        
        # 4. Risk Projection (Hard Constraints)
        if self.risk_manager:
            current_inv = state[:, -1:]
            u_executed = self.risk_manager.project_controls(
                raw_u, current_inv, vol, self.dt
            )
        else:
            u_executed = raw_u
            
        return u_executed
