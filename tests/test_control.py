import unittest
import torch
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import HybridDynamicalSystem
from src.control import ControlAffineSystem
from src.objective import ProfitFunctional, ControlConfig
from src.mpc import MPCSolver

class TestControl(unittest.TestCase):
    
    def setUp(self):
        torch.manual_seed(42)
        
    def test_profit_maximization_sine_wave(self):
        """
        Test if the controller can maximize profit on a predictable sine wave price.
        P(t) = sin(t)
        dP/dt = cos(t)
        
        Strategy:
        - When dP/dt > 0 (Price rising), Inventory should be positive.
        - When dP/dt < 0 (Price falling), Inventory should be negative.
        """
        # 1. Setup Mock Model
        # We cheat and make the model output the exact derivative of a sine wave
        # regardless of input.
        class MockModel(torch.nn.Module):
            def forward(self, t, x):
                # x shape: (Batch, 4)
                # We want dLogPrice/dt (index 3) to be cos(t)
                # Other derivatives 0
                batch_size = x.shape[0]
                out = torch.zeros_like(x)
                # dP/dt = cos(t)
                # Note: t is a float in forward, but we need to broadcast it
                out[:, 3] = torch.cos(torch.tensor(t))
                return out
                
        market_model = MockModel()
        control_system = ControlAffineSystem(market_model, impact_factor=0.0)
        
        # 2. Setup Objective
        # Low risk aversion to encourage trading
        config = ControlConfig(lambda_risk=0.1, lambda_cost=0.0, horizon=10, dt=0.1)
        objective = ProfitFunctional(config)
        
        solver = MPCSolver(control_system, objective, horizon_steps=20)
        solver.dt = 0.1 # Sync dt
        
        # 3. Solve
        # Initial state: Inventory 0, Price 0
        x0 = torch.zeros(1, 5) # 4 market + 1 inventory
        
        # Optimize
        u_opt, loss = solver.solve(x0, iterations=100, lr=0.1)
        
        # 4. Verify
        # Check correlation between u(t) and dP/dt
        # u(t) is rate of change of inventory.
        # Inventory should track price direction.
        # Wait, optimal control for dP/dt > 0 is u > 0 to build inventory?
        # Yes.
        
        # Let's check the inventory trajectory
        trajectory = []
        x = x0
        for t in range(20):
            def func(time, state):
                return control_system(time, state, u_opt[t])
            # Simple Euler for check
            # Note: func(t, x) uses u_opt[t] from closure
            dx = func(t*0.1, x) 
            x = x + dx * 0.1
            trajectory.append(x)
        traj = torch.stack(trajectory)
        inventory = traj[:, 0, 4]
        
        # Price derivative over time 0 to 2.0 (20 steps * 0.1)
        time_steps = torch.linspace(0, 1.9, 20)
        dp_dt = torch.cos(time_steps)
        
        # Inventory should be positively correlated with dP/dt?
        # If dP/dt is positive, we want long inventory.
        # So Inventory ~ dP/dt.
        
        correlation = torch.corrcoef(torch.stack([inventory, dp_dt]))[0, 1]
        print(f"Inventory vs dP/dt Correlation: {correlation.item():.4f}")
        
        # u(t) is dI/dt.
        # If I ~ cos(t), then u ~ -sin(t)?
        # Or does u just want to be positive when dP/dt is positive?
        # If dP/dt > 0, we want max inventory. To get max inventory, u should be large positive.
        
        self.assertGreater(correlation.item(), 0.5)
        print("Control Test Passed: Controller builds inventory when price rises.")

if __name__ == '__main__':
    unittest.main()
