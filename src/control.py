import torch
import torch.nn as nn
from src.model import HybridDynamicalSystem

class ControlAffineSystem(nn.Module):
    """
    Wraps the learned market dynamics f(X) into a control-affine system:
    dX/dt = f(X, theta) + B * u(t)
    
    Here, u(t) represents the TRADING RATE (shares/contracts per second).
    The state X must now include 'Inventory' or 'Position' as a variable
    to track the effect of u(t).
    
    If X = [LogRet, Vol, Mom, LogPrice], we append Inventory I.
    dI/dt = u(t)
    
    Market Impact:
    We can also model price impact:
    dP/dt = f_price(X) + lambda_impact * u(t)
    """
    def __init__(self, market_model: HybridDynamicalSystem, impact_factor: float = 0.0):
        super().__init__()
        self.market_model = market_model
        # Impact factor (linear price impact)
        self.impact_factor = nn.Parameter(torch.tensor([impact_factor]), requires_grad=False)
        
    def forward(self, t: float, x_aug: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: time
            x_aug: Augmented state [MarketState (5), Inventory (1)]
            u: Control input (trading rate) shape (Batch, 1)
            
        Returns:
            dx_aug/dt
        """
        # Split state
        market_state = x_aug[:, :-1] # First 5 dims
        inventory = x_aug[:, -1:]    # Last dim
        
        # 1. Uncontrolled Market Dynamics
        # dx_m/dt = f(X_m)
        dx_market = self.market_model(t, market_state)
        
        # 2. Add Control Effects (Market Impact)
        # Price is at index 3 of market_state
        # dP/dt_total = dP/dt_model + impact * u
        # We add impact to the price derivative
        dx_market[:, 3:4] += self.impact_factor * u
        
        # 3. Inventory Dynamics
        # dI/dt = u
        d_inventory = u
        
        # Combine
        return torch.cat([dx_market, d_inventory], dim=1)
