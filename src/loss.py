import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class LossConfig:
    lambda_mse: float = 1.0
    lambda_vol: float = 0.1
    lambda_drawdown: float = 0.5
    lambda_l2: float = 0.01

class TrajectoryLoss(nn.Module):
    """
    Computes the time-accumulated loss functional for a continuous-time system.
    J(theta) = Integral [ (P_real - P_model)^2 + lambda_vol * Vol + lambda_dd * Drawdown ] dt
    """
    def __init__(self, config: LossConfig):
        super().__init__()
        self.config = config
        
    def forward(self, 
                predicted_trajectory: torch.Tensor, 
                target_trajectory: torch.Tensor, 
                model_params: list = None) -> dict:
        """
        Args:
            predicted_trajectory: Tensor (Time, Batch, Features)
            target_trajectory: Tensor (Time, Batch, Features)
            model_params: List of model parameters for L2 regularization
            
        Returns:
            Dictionary containing total loss and individual components
        """
        # 1. Trajectory MSE (Integral approximation)
        # We assume uniform time steps dt, so mean over time dimension is proportional to integral
        mse_loss = torch.mean((predicted_trajectory - target_trajectory)**2)
        
        # 2. Volatility Penalty
        # Volatility is estimated as the standard deviation of changes along the trajectory
        # Diff along time dimension: (Time-1, Batch, Features)
        diffs = predicted_trajectory[1:] - predicted_trajectory[:-1]
        volatility = torch.std(diffs, dim=0).mean() # Mean over batch and features
        
        # 3. Drawdown Penalty (Max Drawdown over the horizon)
        # Calculate running max
        # We focus on the Price feature (assumed index 3 based on StateTransformer)
        # If Features < 4, we take the last feature
        price_idx = -1 
        prices = predicted_trajectory[:, :, price_idx]
        
        # Avoid inplace operations for running max
        # running_max[t] = torch.max(...) is inplace assignment to the tensor slice? No, it's setitem.
        # But let's build a list instead to be safe for autograd.
        
        running_max_list = [prices[0]]
        for t in range(1, prices.shape[0]):
            current_max = torch.max(running_max_list[-1], prices[t])
            running_max_list.append(current_max)
            
        running_max = torch.stack(running_max_list)
            
        drawdowns = (running_max - prices) / (torch.abs(running_max) + 1e-6)
        max_drawdown = torch.max(drawdowns, dim=0)[0].mean() # Mean max DD over batch
        
        # 4. L2 Regularization
        l2_reg = torch.tensor(0.0, device=predicted_trajectory.device)
        if model_params is not None:
            for param in model_params:
                l2_reg += torch.norm(param)**2
                
        # Total Loss
        total_loss = (self.config.lambda_mse * mse_loss +
                      self.config.lambda_vol * volatility +
                      self.config.lambda_drawdown * max_drawdown +
                      self.config.lambda_l2 * l2_reg)
                      
        return {
            "total_loss": total_loss,
            "mse": mse_loss,
            "volatility": volatility,
            "drawdown": max_drawdown,
            "l2_reg": l2_reg
        }
