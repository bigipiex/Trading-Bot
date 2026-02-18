import torch
import torch.nn as nn
from typing import Optional
from src.control import ControlAffineSystem
from src.objective import ProfitFunctional
from src.solver import rk4_step
from src.risk import RiskManager

class MPCSolver(nn.Module):
    """
    Model Predictive Control Solver.
    
    Given a learned dynamical model f(X, theta), it optimizes a sequence of control actions u(t)
    to maximize profit over a horizon T.
    
    Implementation:
    - We treat the control sequence u_t as learnable parameters for this specific instance.
    - We run gradient descent on u_t directly (shooting method).
    """
    def __init__(self, 
                 system: ControlAffineSystem, 
                 objective: ProfitFunctional, 
                 risk_manager: Optional[RiskManager] = None,
                 horizon_steps: int = 10):
        super().__init__()
        self.system = system
        self.objective = objective
        self.risk_manager = risk_manager
        self.horizon_steps = horizon_steps
        self.dt = 0.01 # Should match objective config
        
    def solve(self, x0: torch.Tensor, iterations: int = 100, lr: float = 0.1):
        """
        Solves the optimal control problem for a given initial state x0.
        
        Args:
            x0: Initial state (Batch, StateDim). Assumes Inventory=0 usually.
            
        Returns:
            optimal_controls: (Time, Batch, 1)
            final_loss: Scalar
        """
        batch_size = x0.shape[0]
        
        # Initialize controls as learnable parameters
        # u(t) for t=0...T-1
        u_seq = nn.Parameter(torch.zeros(self.horizon_steps, batch_size, 1, requires_grad=True))
        
        optimizer = torch.optim.Adam([u_seq], lr=lr)
        
        final_loss = 0.0
        
        for i in range(iterations):
            optimizer.zero_grad()
            
            # 1. Rollout Trajectory with current u_seq
            # trajectory stores states x_0, x_1, ..., x_T
            trajectory = [x0]
            x = x0
            
            # Store price changes for PnL calculation
            price_changes = []
            
            for t in range(self.horizon_steps):
                u_t = u_seq[t] # (B, 1)
                
                # We need a closure that binds u_t for the RK4 step
                # The closure signature must be (time, state) -> dstate/dt
                # BUT u_t changes at each step t.
                # Inside the RK4 step, time advances by 0.5*dt, but u is held constant (Zero-Order Hold)
                
                def dynamics_closure(time, state):
                    return self.system(time, state, u_t)
                
                # Perform RK4 step
                x_next = rk4_step(dynamics_closure, t * self.dt, x, self.dt)
                
                # Compute dP/dt for the objective function (at the start of the interval)
                # We can just call system once
                dx_dt_start = self.system(t * self.dt, x, u_t)
                
                # Store
                trajectory.append(x_next)
                
                # Price change rate (approx) at start of interval
                # Price is index 3 in Market State (which is first 4 dims of x)
                # So index 3 in x_aug
                dp = dx_dt_start[:, 3:4]
                price_changes.append(dp)
                
                x = x_next
            
            # Convert lists to tensors
            # trajectory has T+1 elements: x_0 ... x_T
            traj_tensor = torch.stack(trajectory) 
            
            # We align trajectory with controls for cost calculation.
            # Controls u_0 ... u_{T-1} apply to states x_0 ... x_{T-1}
            traj_aligned = traj_tensor[:-1] 
            
            # price_changes has T elements: dp_0 ... dp_{T-1}
            price_changes_tensor = torch.stack(price_changes)
            
            # 2. Compute Loss
            # We pass u_seq directly (Time, Batch, 1)
            loss_dict = self.objective(traj_aligned, u_seq, price_changes_tensor)
            loss = loss_dict["total_objective"]
            
            # Add Soft Constraint (Barrier Penalty) if Risk Manager is present
            if self.risk_manager is not None:
                # Volatility is feature index 1
                vol = traj_aligned[:, :, 1:2]
                penalty = self.risk_manager.barrier_penalty(traj_aligned, vol)
                loss += penalty
            
            # 3. Optimize
            loss.backward()
            optimizer.step()
            
            final_loss = loss.item()
            
        # Post-Optimization: Hard Constraint Projection
        # We must project u_seq to ensure I_t stays within bounds.
        # This is done sequentially forward in time because I_{t+1} depends on u_t.
        if self.risk_manager is not None:
            with torch.no_grad():
                current_inv = x0[:, -1:]
                projected_u = []
                
                for t in range(self.horizon_steps):
                    u_t = u_seq[t]
                    
                    # Estimate vol at this step (using initial state vol for simplicity, or re-rollout)
                    # Ideally we re-rollout, but for projection we can approximate with x0 vol?
                    # No, vol changes. Let's assume constant vol for projection safety or use last known.
                    # Better: Re-rollout dynamics with projection.
                    
                    # For now, let's just project based on current inventory state
                    # assuming Vol stays roughly same as x0 for the limit calculation
                    # (This is a simplification for the projection step)
                    vol = x0[:, 1:2] 
                    
                    u_proj = self.risk_manager.project_controls(u_t, current_inv, vol, self.dt)
                    projected_u.append(u_proj)
                    
                    # Update inventory
                    current_inv = current_inv + u_proj * self.dt
                
                u_seq.data = torch.stack(projected_u)
            
        return u_seq.detach(), final_loss
