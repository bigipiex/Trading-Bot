import torch
import unittest
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import HybridDynamicalSystem
from src.solver import odeint_rk4

class TestDynamics(unittest.TestCase):
    
    def test_rk4_solver(self):
        """Test RK4 solver on a simple exponential decay."""
        # dy/dt = -k * y
        k = 0.5
        y0 = torch.tensor([[1.0]])
        t_span = torch.linspace(0, 5, 100)
        
        def func(t, y):
            return -k * y
            
        trajectory = odeint_rk4(func, y0, t_span)
        
        # Analytical solution: y(t) = y0 * exp(-k * t)
        # Ensure y_true has shape (Time, Batch, Feat) = (100, 1, 1)
        y_true = y0 * torch.exp(-k * t_span).view(-1, 1, 1)
        
        # Check max error
        error = torch.max(torch.abs(trajectory - y_true))
        print(f"RK4 Max Error: {error.item():.6f}")
        self.assertLess(error.item(), 1e-4)

    def test_physics_constraints(self):
        """Test if physics parameters respect constraints (e.g., kappa > 0)."""
        model = HybridDynamicalSystem(feature_dim=4)
        
        # Initialize with negative raw value
        with torch.no_grad():
            model.physics.raw_kappa.data = torch.tensor([-10.0])
            
        # Check if property is positive (softplus)
        self.assertTrue(model.physics.kappa > 0)
        print(f"Kappa: {model.physics.kappa.item():.4f} (Raw: {model.physics.raw_kappa.item()})")

    def test_gradient_flow(self):
        """Test if gradients propagate through the ODE solver."""
        model = HybridDynamicalSystem(feature_dim=4)
        x0 = torch.randn(1, 4, requires_grad=True)
        t_span = torch.linspace(0, 1, 10)
        
        def func(t, x):
            return model(t, x)
            
        trajectory = odeint_rk4(func, x0, t_span)
        loss = torch.sum(trajectory ** 2)
        loss.backward()
        
        # Check gradients exist
        self.assertIsNotNone(x0.grad)
        self.assertIsNotNone(model.physics.raw_kappa.grad)
        self.assertIsNotNone(model.neural.net[0].weight.grad)
        
        grad_norm = torch.norm(model.physics.raw_kappa.grad)
        print(f"Gradient Norm (Kappa): {grad_norm.item():.6f}")
        self.assertGreater(grad_norm.item(), 0.0)

    def test_hybrid_stability(self):
        """Test if the hybrid model produces stable trajectories."""
        model = HybridDynamicalSystem(feature_dim=4, epsilon=0.1)
        x0 = torch.randn(10, 4) # Batch of 10
        t_span = torch.linspace(0, 10, 100) # Long horizon
        
        def func(t, x):
            return model(t, x)
            
        trajectory = odeint_rk4(func, x0, t_span)
        
        # Check for NaNs or Infinity
        self.assertFalse(torch.isnan(trajectory).any())
        self.assertFalse(torch.isinf(trajectory).any())
        
        # Check if values explode (simple bound check)
        max_val = torch.max(torch.abs(trajectory))
        print(f"Max Trajectory Value: {max_val.item():.2f}")
        self.assertLess(max_val.item(), 1e5) # Should not explode

if __name__ == '__main__':
    unittest.main()
