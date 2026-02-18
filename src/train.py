import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Callable, Tuple, List, Dict
import numpy as np

from src.model import HybridDynamicalSystem
from src.solver import odeint_rk4
from src.loss import TrajectoryLoss, LossConfig

class SystemIdentification:
    """
    Optimization loop for the nonlinear dynamical system.
    This class handles the time-unrolled optimization (BPTT),
    stability monitoring, and diagnostics.
    """
    def __init__(self, 
                 model: HybridDynamicalSystem, 
                 optimizer: optim.Optimizer,
                 loss_fn: TrajectoryLoss,
                 scheduler: optim.lr_scheduler._LRScheduler = None,
                 clip_grad_norm: float = 1.0):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.clip_grad_norm = clip_grad_norm
        self.logs = []
        
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Runs one epoch of training over time-series trajectories.
        
        Args:
            dataloader: Returns batches of shape (Batch, Time, Features)
            
        Returns:
            Dictionary of epoch metrics
        """
        self.model.train()
        epoch_loss = 0.0
        n_batches = 0
        
        for batch_idx, trajectories in enumerate(dataloader):
            # Shape: (Batch, Time, Features) -> (Time, Batch, Features) for ODE solver
            target_trajectory = trajectories.permute(1, 0, 2)
            time_steps = target_trajectory.shape[0]
            batch_size = target_trajectory.shape[1]
            
            # Initial state x0: First time step of the batch
            x0 = target_trajectory[0]
            
            # Time span: Assuming uniform dt=0.01 for now (can be passed in batch)
            t_span = torch.linspace(0, (time_steps-1)*0.01, time_steps, device=x0.device)
            
            # 1. Forward Pass (ODE Integration)
            def dynamics_func(t, x):
                return self.model(t, x)
                
            predicted_trajectory = odeint_rk4(dynamics_func, x0, t_span)
            
            # 2. Compute Loss
            # Pass model parameters for L2 regularization
            loss_dict = self.loss_fn(predicted_trajectory, target_trajectory, 
                                    list(self.model.parameters()))
            loss = loss_dict["total_loss"]
            
            # 3. Backward Pass (BPTT through ODE Solver)
            self.optimizer.zero_grad()
            loss.backward()
            
            # 4. Gradient Clipping & Monitoring
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            
            # 5. Parameter Update
            self.optimizer.step()
            
            # Accumulate metrics
            epoch_loss += loss.item()
            n_batches += 1
            
            # Log stability metrics for this batch
            self._log_batch_metrics(batch_idx, loss_dict, grad_norm, predicted_trajectory)
            
        if self.scheduler:
            self.scheduler.step()
            
        return {"avg_loss": epoch_loss / n_batches}
        
    def _log_batch_metrics(self, batch_idx: int, loss_dict: dict, grad_norm: float, trajectory: torch.Tensor):
        """Logs detailed diagnostics for stability monitoring."""
        with torch.no_grad():
            kappa = self.model.physics.kappa.item()
            beta = self.model.physics.beta.item()
            max_val = torch.max(torch.abs(trajectory)).item()
            
            log_entry = {
                "batch": batch_idx,
                "loss": loss_dict["total_loss"].item(),
                "mse": loss_dict["mse"].item(),
                "grad_norm": grad_norm.item(),
                "kappa": kappa,
                "beta": beta,
                "max_traj": max_val
            }
            self.logs.append(log_entry)
            
            # Check for instability
            if max_val > 1e4:
                print(f"WARNING: Trajectory explosion detected in batch {batch_idx}! Max val: {max_val:.2f}")
                
            if grad_norm > self.clip_grad_norm * 2:
                print(f"WARNING: Large gradient norm in batch {batch_idx}: {grad_norm:.2f}")

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluates the model on validation data."""
        self.model.eval()
        total_mse = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for trajectories in dataloader:
                target_trajectory = trajectories.permute(1, 0, 2)
                x0 = target_trajectory[0]
                time_steps = target_trajectory.shape[0]
                t_span = torch.linspace(0, (time_steps-1)*0.01, time_steps, device=x0.device)
                
                def dynamics_func(t, x):
                    return self.model(t, x)
                
                predicted_trajectory = odeint_rk4(dynamics_func, x0, t_span)
                loss_dict = self.loss_fn(predicted_trajectory, target_trajectory, list(self.model.parameters()))
                total_mse += loss_dict["mse"].item()
                n_batches += 1
                
        return {"val_mse": total_mse / n_batches}
