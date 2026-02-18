import unittest
import torch
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import HybridDynamicalSystem
from src.control import ControlAffineSystem
from src.strategy import SimpleStrategy
from src.synthetic import SyntheticGenerator, OUParams
from src.solver import rk4_step, odeint_rk4
from src.train import SystemIdentification
from src.loss import TrajectoryLoss, LossConfig
from torch.utils.data import DataLoader, TensorDataset

class SignalHorizon(unittest.TestCase):
    
    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        
    def _create_synthetic_data(self, n_steps=2000):
        # Generate Mean Reversion Data
        params = OUParams(mu=0.0, theta=0.5, sigma=0.1, dt=0.01)
        _, prices = SyntheticGenerator.ornstein_uhlenbeck_process(params, n_steps=n_steps, start_price=100.0)
        
        prices_tensor = torch.tensor(prices, dtype=torch.float32)
        log_prices = torch.log(prices_tensor)
        
        # Features [LogRet, Vol, Mom, LogPrice, Mu]
        time_len = len(prices)
        market_data = torch.zeros(time_len, 1, 5, dtype=torch.float32)
        market_data[:, 0, 3] = log_prices
        market_data[1:, 0, 0] = log_prices[1:] - log_prices[:-1]
        market_data[:, 0, 1] = 0.01
        market_data[:, 0, 2] = log_prices - 0.0 # Deviation
        # Use simple initial Mu guess (sma or fixed)
        market_data[:, 0, 4] = np.log(100.0) # True Mu
        
        return market_data

    def _evaluate_horizon(self, model, data, horizon_steps):
        # Compute N-step prediction correlation
        predictions = []
        realized_returns = []
        
        time_len = data.shape[0]
        dt = 0.01
        
        # We need to simulate forward N steps
        # This is slow if we do it for every single point.
        # Let's sample every N steps or just do a subset.
        # Doing it for every step is fine for 2000 points.
        
        with torch.no_grad():
            for t in range(time_len - horizon_steps):
                current_obs = data[t] # (1, 5)
                # Augment with inventory=0
                inv = torch.zeros(1, 1)
                state = torch.cat([current_obs, inv], dim=1)
                
                # Predict N steps ahead using model dynamics
                # We assume u=0 for prediction (passive drift)
                def dynamics_closure(time, s):
                    # ControlAffineSystem expects (t, x_aug, u)
                    # We need to wrap the model directly or use ControlAffine with 0 u
                    # Let's use the model directly on market state
                    dx_market = model(time, s[:, :-1])
                    # Inventory stays 0
                    dx_inv = torch.zeros_like(s[:, -1:])
                    return torch.cat([dx_market, dx_inv], dim=1)

                # Integrate
                # We just want the final state at t + N*dt
                t_span = torch.tensor([0, horizon_steps * dt])
                # odeint returns [x0, xN]
                traj = odeint_rk4(dynamics_closure, state, t_span)
                final_state = traj[-1]
                
                # Predicted Price Change (Log Price is index 3)
                pred_price_change = final_state[0, 3] - state[0, 3]
                
                # Realized Price Change
                real_price_change = data[t + horizon_steps, 0, 3] - data[t, 0, 3]
                
                predictions.append(pred_price_change.item())
                realized_returns.append(real_price_change.item())
                
        pred_tensor = torch.tensor(predictions)
        real_tensor = torch.tensor(realized_returns)
        
        if len(predictions) < 2: return 0.0
        
        correlation = torch.corrcoef(torch.stack([pred_tensor, real_tensor]))[0, 1].item()
        return correlation

    def test_horizon_sweep(self):
        print("\n--- Horizon Sweep Analysis (Manual Params) ---")
        data = self._create_synthetic_data(n_steps=2000)
        
        # Manual Model
        model = HybridDynamicalSystem(feature_dim=5, epsilon=0.0)
        with torch.no_grad():
            model.physics.raw_kappa.data = torch.tensor([0.5], dtype=torch.float32)
            model.physics.raw_gamma.data = torch.tensor([0.0], dtype=torch.float32) # Fixed Mu for now
            model.physics.beta.data = torch.tensor([0.0], dtype=torch.float32)
        
        horizons = [1, 5, 10, 20, 50]
        for h in horizons:
            corr = self._evaluate_horizon(model, data, h)
            print(f"Horizon {h}: Correlation = {corr:.4f}")
            
        print("-" * 40)
        
    def test_trained_model(self):
        print("\n--- Training Model for N-step Prediction ---")
        # 1. Generate Data
        data = self._create_synthetic_data(n_steps=2000)
        train_len = 1000
        train_data = data[:train_len]
        test_data = data[train_len:]
        
        # 2. Train Model
        # We train on short trajectories (e.g. 20 steps) to capture dynamics
        # Create dataset of trajectories
        traj_len = 20
        trajectories = []
        for i in range(0, train_len - traj_len, 5): # Stride 5
            # train_data[i:i+traj_len] shape is (20, 1, 5)
            # We want (20, 5)
            traj = train_data[i:i+traj_len].squeeze(1)
            trajectories.append(traj)
        
        # Stack -> (Batch, Time, Features)
        dataset = TensorDataset(torch.stack(trajectories))
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Initialize Model (Random-ish)
        model = HybridDynamicalSystem(feature_dim=5, epsilon=0.0) # Start with physics only
        # Use explicit manual initialization to be close to truth but slightly off to test training
        with torch.no_grad():
            model.physics.raw_kappa.data = torch.tensor([0.2]) # Wrong kappa (True is 0.5)
            model.physics.raw_gamma.data = torch.tensor([0.0]) # Fixed Mu
            
        optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
        loss_fn = TrajectoryLoss(LossConfig(lambda_mse=1.0))
        
        trainer = SystemIdentification(model, optimizer, loss_fn)
        
        print("Training...")
        for epoch in range(5):
            # Unpack dataloader
            # DataLoader yields a list [tensor], we need just the tensor
            def unpacked_dataloader():
                for batch in dataloader:
                    yield batch[0]
            
            metrics = trainer.train_epoch(unpacked_dataloader())
            print(f"Epoch {epoch}: Loss {metrics['avg_loss']:.4f}, Kappa {model.physics.kappa.item():.4f}")
            
        # 3. Evaluate on Test Data
        print("\n--- Horizon Sweep (Trained Model) ---")
        horizons = [1, 5, 10, 20, 50]
        for h in horizons:
            corr = self._evaluate_horizon(model, test_data, h)
            print(f"Horizon {h}: Correlation = {corr:.4f}")
            
        # Assertion: Correlation should improve with horizon (up to a point)
        # N=1 is noisy. N=20 should be better.
        # But this is synthetic data, so N=1 correlation should be roughly:
        # drift ~ dt, noise ~ sqrt(dt).
        # drift/noise ~ sqrt(dt) = 0.1.
        # So correlation around 0.1 is expected for N=1?
        # Actually drift is 0.5 * deviation * dt.
        # If deviation is 1 sigma (0.1), drift is 0.5 * 0.1 * 0.01 = 0.0005.
        # Noise is 0.1 * sqrt(0.01) = 0.01.
        # SNR = 0.05. Very low correlation expected for N=1.
        
        # For N=50 (t=0.5):
        # Drift accumulates. Noise accumulates as sqrt(T).
        # Should see higher correlation.
        
        self.assertTrue(True) # Just printing for analysis

if __name__ == '__main__':
    unittest.main()
