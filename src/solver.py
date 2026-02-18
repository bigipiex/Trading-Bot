import torch
import torch.nn as nn
from typing import Callable, Tuple

def rk4_step(
    func: Callable[[float, torch.Tensor], torch.Tensor],
    t: float,
    x: torch.Tensor,
    dt: float
) -> torch.Tensor:
    """
    Explicit Runge-Kutta 4th Order Step.
    
    Args:
        func: The function f(t, x) defining dx/dt.
        t: Current time.
        x: Current state tensor (batch_size, state_dim).
        dt: Time step size.
        
    Returns:
        x_next: State at t + dt.
    """
    k1 = func(t, x)
    k2 = func(t + 0.5 * dt, x + 0.5 * dt * k1)
    k3 = func(t + 0.5 * dt, x + 0.5 * dt * k2)
    k4 = func(t + dt, x + dt * k3)
    
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

def odeint_rk4(
    func: Callable[[float, torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    t_span: torch.Tensor,
) -> torch.Tensor:
    """
    Integrates the system over t_span using RK4.
    Preserves the computational graph for backpropagation.
    """
    trajectory = [x0]
    x = x0
    
    # Iterate through time steps
    for i in range(len(t_span) - 1):
        t_current = t_span[i]
        t_next = t_span[i+1]
        dt = t_next - t_current
        
        # Perform RK4 step
        x = rk4_step(func, t_current, x, dt)
        trajectory.append(x)
        
    return torch.stack(trajectory)
