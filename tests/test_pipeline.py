import unittest
import numpy as np
import torch
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.synthetic import SyntheticGenerator, OUParams
from src.features import StateTransformer

class TestPipeline(unittest.TestCase):
    
    def test_synthetic_data_generation(self):
        """Test if OU process generates valid data."""
        params = OUParams(mu=100.0, theta=0.5, sigma=2.0, dt=0.01)
        t, x = SyntheticGenerator.ornstein_uhlenbeck_process(params, n_steps=1000)
        
        self.assertEqual(len(t), 1000)
        self.assertEqual(len(x), 1000)
        self.assertTrue(np.all(np.isfinite(x)))
        print(f"Mean: {np.mean(x):.2f}, Std: {np.std(x):.2f}")
        
    def test_state_transformer(self):
        """Test feature engineering pipeline."""
        # Generate synthetic data
        params = OUParams(mu=100.0, theta=0.5, sigma=2.0, dt=0.01)
        _, prices = SyntheticGenerator.ornstein_uhlenbeck_process(params, n_steps=1000)
        
        transformer = StateTransformer(vol_window=20)
        features = transformer.transform(prices)
        
        # Check shape: (Time, Features=4)
        self.assertEqual(features.shape[0], 1000)
        self.assertEqual(features.shape[1], 4)
        
        # Check for NaNs
        self.assertFalse(torch.any(torch.isnan(features)), "Features contain NaNs")
        
        # Check normalization
        features_norm = transformer.normalize(features, fit=True)
        # Note: We relax the tolerance slightly because we add 1e-6 to std in implementation
        self.assertTrue(torch.allclose(features_norm.mean(dim=0), torch.zeros(4), atol=1e-4))
        self.assertTrue(torch.allclose(features_norm.std(dim=0), torch.ones(4), atol=1e-2))
        
        print("Feature Transformer Test Passed")

if __name__ == '__main__':
    unittest.main()
