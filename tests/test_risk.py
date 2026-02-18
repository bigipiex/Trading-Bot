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
from src.risk import RiskManager, RiskConfig

class TestRisk(unittest.TestCase):
    
    def setUp(self):
        torch.manual_seed(42)
        
    def test_risk_constraints(self):
        """
        Verify that RiskManager enforces constraints under high volatility.
        """
        # 1. Setup Mock System
        class MockModel(torch.nn.Module):
            def forward(self, t, x):
                # Always predict price UP to encourage buying
                out = torch.zeros_like(x)
                out[:, 3] = 1.0 # dP/dt = 1.0 (Price rising)
                return out
                
        market_model = MockModel()
        control_system = ControlAffineSystem(market_model)
        
        # 2. Setup Risk Manager with TIGHT limits
        # Base limit = 1.0
        # If vol = 0.1 (high), limit = 1.0 * (0.05 / 0.1) = 0.5
        risk_config = RiskConfig(
            base_inventory_limit=1.0,
            volatility_target=0.05,
            barrier_weight=100.0
        )
        risk_manager = RiskManager(risk_config)
        
        # 3. Setup MPC
        # High profit incentive to try and violate limits
        control_config = ControlConfig(lambda_risk=0.0, lambda_cost=0.0, dt=0.1)
        objective = ProfitFunctional(control_config)
        
        solver = MPCSolver(
            control_system, 
            objective, 
            risk_manager=risk_manager,
            horizon_steps=10
        )
        solver.dt = 0.1
        
        # 4. Test Case: High Volatility
        # State: [LogRet, Vol=0.1, Mom, LogPrice, Inventory=0]
        x0 = torch.zeros(1, 5)
        x0[:, 1] = 0.1 # High Volatility (double the target 0.05)
        
        # Expected Limit: 1.0 * (0.05/0.1) = 0.5
        expected_limit = 0.5
        
        # Solve
        u_opt, _ = solver.solve(x0, iterations=50, lr=0.1)
        
        # 5. Verify Inventory Trajectory
        # Calculate resulting inventory
        # Initial inventory (Batch, 1) -> (1, 1)
        curr_inv = x0[:, 4:5] 
        inv_traj = [curr_inv]
        
        for t in range(10):
            # u_opt is (Batch, 1)
            u_t = u_opt[t]
            curr_inv = curr_inv + u_t * 0.1
            inv_traj.append(curr_inv)
            
        inv_tensor = torch.stack(inv_traj)
        max_inv = torch.max(torch.abs(inv_tensor))
        
        print(f"Max Inventory Reached: {max_inv.item():.4f}")
        print(f"Expected Limit: {expected_limit:.4f}")
        
        # Check if constraints were respected (approx)
        # Due to projection simplification, it might be slightly off if vol changes,
        # but here vol is constant 0.1 in our MockModel (MockModel returns 0 derivative for vol).
        # So it should be exact.
        
        # Allow small numerical error margin
        self.assertLessEqual(max_inv.item(), expected_limit + 1e-3)
        print("Risk Test Passed: Constraints respected under high volatility.")

if __name__ == '__main__':
    unittest.main()
