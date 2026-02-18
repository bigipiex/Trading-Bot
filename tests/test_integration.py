import unittest
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.synthetic import SyntheticGenerator, OUParams
from src.features import StateTransformer
from src.model import HybridDynamicalSystem
from src.loss import TrajectoryLoss, LossConfig
from src.train import SystemIdentification

class TestIntegration(unittest.TestCase):
    
    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        
    def test_full_training_loop(self):
        """
        Integration test: Can the system learn parameters of a synthetic OU process?
        Ground Truth: dP/dt = theta(mu - P)
        Model: dP/dt = kappa(mu - P) + ...
        We expect kappa to approach theta.
        """
        # 1. Generate Synthetic Data
        # High theta (0.5) to make mean reversion obvious
        params = OUParams(mu=0.0, theta=0.5, sigma=0.1, dt=0.01) 
        batch_size = 10
        seq_len = 50
        
        # (Batch, Time, 1)
        raw_data = SyntheticGenerator.generate_training_batch(batch_size, seq_len, params)
        
        # 2. Feature Engineering
        # We manually construct features to match model expectations
        # [LogRet, Vol, Mom, LogPrice]
        features_list = []
        for i in range(batch_size):
            p = raw_data[i, :, 0]
            # Simple diff for log returns (velocity)
            log_ret = torch.zeros_like(p)
            log_ret[1:] = p[1:] - p[:-1]
            
            vol = torch.ones_like(p) * 0.1
            mom = torch.zeros_like(p)
            log_price = p
            
            # Stack features: (Time, Features=4)
            f = torch.stack([log_ret, vol, mom, log_price], dim=1)
            features_list.append(f)
            
        # Create dataset
        # Shape: (Batch, Time, Features)
        dataset_tensor = torch.stack(features_list)
        
        # DataLoader returns a list of tensors if TensorDataset has multiple tensors,
        # but here we pass a single tensor, so it should return a single tensor.
        # However, let's be safe.
        dataset = TensorDataset(dataset_tensor)
        dataloader = DataLoader(dataset, batch_size=5, shuffle=True)
        
        # 3. Initialize System
        model = HybridDynamicalSystem(feature_dim=4, epsilon=0.0) # Turn off neural net to test physics only
        
        # Initialize kappa close to 0 to see if it grows
        with torch.no_grad():
            model.physics.raw_kappa.data = torch.tensor([-2.0]) # Softplus(-2) approx 0.12
            # Initialize beta to 0 (no momentum coupling in pure OU)
            model.physics.beta.data = torch.tensor([0.0])
            model.physics.mu.data = torch.tensor([0.0]) # Correct mu
        
        loss_config = LossConfig(lambda_mse=1.0, lambda_vol=0.0, lambda_drawdown=0.0, lambda_l2=0.0)
        loss_fn = TrajectoryLoss(loss_config)
        optimizer = optim.Adam(model.parameters(), lr=0.05)
        
        trainer = SystemIdentification(model, optimizer, loss_fn)
        
        # 4. Train
        initial_kappa = model.physics.kappa.item()
        print(f"Initial Kappa: {initial_kappa:.4f}")
        
        for epoch in range(20):
            # In test, we need to handle the fact that DataLoader returns a list [tensor]
            # but our train_epoch expects just tensor.
            # Let's wrap the dataloader to unpack.
            def unpacked_dataloader():
                for batch in dataloader:
                    yield batch[0]
            
            metrics = trainer.train_epoch(unpacked_dataloader())
            if epoch % 5 == 0:
                print(f"Epoch {epoch}: Loss {metrics['avg_loss']:.6f}, Kappa {model.physics.kappa.item():.4f}")
        
        final_kappa = model.physics.kappa.item()
        print(f"Final Kappa: {final_kappa:.4f}")
        
        # 5. Validation
        # Kappa should increase towards true theta (0.5)
        # It might not reach exactly 0.5 due to discretization error and limited data,
        # but it should definitely move significantly from 0.12
        self.assertGreater(final_kappa, initial_kappa)
        self.assertLess(metrics['avg_loss'], 1.0)
        
        print("Integration Test Passed: Physics term learned from data.")

if __name__ == '__main__':
    unittest.main()
