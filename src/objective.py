import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class ControlConfig:
    lambda_risk: float = 0.5  # Risk aversion (quadratic inventory cost)
    lambda_cost: float = 0.01 # Transaction cost (slippage)
    horizon: float = 1.0
    dt: float = 0.01

class ProfitFunctional(nn.Module):
    """
    Computes the NEGATIVE expected utility (to be minimized).
    J = Integral [ - (u(t) * dP/dt) + lambda_risk * Inventory(t)^2 + lambda_cost * u(t)^2 ] dt
    
    1. Profit Rate: u(t) * dP/dt (Actually, P * dI/dt + I * dP/dt... careful here).
       Standard PnL rate is: Inventory(t) * dP/dt.
       Or if we model cash explicitly: dCash/dt = -P * u.
       
       Let's use the standard "Mark-to-Market" PnL rate:
       d(Wealth)/dt = Inventory(t) * dP(t)/dt
    
    2. Risk: Quadratic penalty on inventory (avoids large positions).
       lambda_risk * Inventory(t)^2
       
    3. Transaction Cost: Quadratic penalty on trading rate (avoids churn).
       lambda_cost * u(t)^2
    """
    def __init__(self, config: ControlConfig):
        super().__init__()
        self.config = config
        
    def forward(self, 
                trajectory: torch.Tensor, 
                controls: torch.Tensor,
                price_changes: torch.Tensor) -> dict:
        """
        Args:
            trajectory: (Time, Batch, State+Inventory)
            controls: (Time, Batch, 1) - u(t)
            price_changes: (Time, Batch, 1) - dP/dt (from model)
            
        Returns:
            Total Cost (Scalar) to minimize.
        """
        # Extract Inventory (last dimension)
        inventory = trajectory[:, :, -1:]
        
        # 1. PnL Term (Maximize) -> Minimize Negative
        # PnL Rate = Inventory * PriceChange
        pnl_rate = inventory * price_changes
        total_pnl = torch.sum(pnl_rate) * self.config.dt
        
        # 2. Risk Term (Minimize)
        risk_cost = self.config.lambda_risk * torch.sum(inventory**2) * self.config.dt
        
        # 3. Transaction Cost (Minimize)
        trans_cost = self.config.lambda_cost * torch.sum(controls**2) * self.config.dt
        
        # Total Objective: Maximize (PnL - Risk - Cost)
        # Minimize -(PnL - Risk - Cost) = -PnL + Risk + Cost
        total_objective = -total_pnl + risk_cost + trans_cost
        
        return {
            "total_objective": total_objective,
            "pnl": total_pnl,
            "risk": risk_cost,
            "cost": trans_cost
        }
