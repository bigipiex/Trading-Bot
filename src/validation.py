import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from tqdm import tqdm
import copy

from src.model import HybridDynamicalSystem
from src.control import ControlAffineSystem
from src.risk import RiskManager, RiskConfig
from src.regime import RegimeDetector
from src.strategy_regime import RegimeSwitchingStrategy
from src.backtest import BacktestEngine
from src.train import SystemIdentification
from src.loss import TrajectoryLoss, LossConfig
from torch.utils.data import DataLoader, TensorDataset

class WalkForwardValidator:
    """
    Implements rigorous Walk-Forward Validation.
    
    Splits data into rolling windows:
    [Train | Val | Test] -> Shift -> [Train | Val | Test]
    
    Ensures NO lookahead bias by locking parameters before Test window.
    """
    def __init__(self, 
                 train_window: int = 500,
                 test_window: int = 100,
                 stride: int = 100):
        self.train_window = train_window
        self.test_window = test_window
        self.stride = stride
        self.results = []
        
    def validate(self, market_data: torch.Tensor) -> pd.DataFrame:
        """
        Runs the WFV process.
        market_data: (TotalTime, 1, Features)
        """
        total_steps = market_data.shape[0]
        # Start point such that we have enough history for train
        start_idx = 0
        
        fold = 0
        while start_idx + self.train_window + self.test_window <= total_steps:
            # 1. Define Splits
            train_end = start_idx + self.train_window
            test_end = train_end + self.test_window
            
            # Slice Data (Strict Separation)
            train_data = market_data[start_idx : train_end]
            test_data = market_data[train_end : test_end]
            
            # 2. Calibrate/Train on TRAIN set
            # Initialize fresh model to avoid carry-over bias?
            # Or fine-tune? "No parameter reuse across folds" -> Fresh model.
            model, strategy = self._calibrate(train_data)
            
            # 3. Test on TEST set
            # Important: The backtest engine needs the state at start of test.
            # State is the last state of train?
            # Actually, BacktestEngine updates state step-by-step.
            # We initialize it with the first frame of test_data.
            # BUT: Inventory starts at 0 for each fold? Or carries over?
            # "No parameter reuse". Usually WFV implies evaluating the strategy's performance
            # as if we traded continuously. But if we reset models, we might reset inventory.
            # Standard WFV aggregates PnL curves.
            # Let's assume inventory reset for simplicity of metric calculation per fold.
            
            metrics = self._run_backtest(strategy, test_data)
            
            self.results.append({
                "fold": fold,
                "train_start": start_idx,
                "train_end": train_end,
                "test_start": train_end,
                "test_end": test_end,
                "sharpe": metrics["Sharpe Ratio"],
                "pnl": metrics["Total PnL"],
                "turnover": metrics["Turnover"],
                "kappa": model.physics.kappa.item(),
                "gamma": model.physics.gamma.item()
            })
            
            print(f"Fold {fold}: Sharpe {metrics['Sharpe Ratio']:.2f}, PnL {metrics['Total PnL']:.2f}")
            
            # Shift
            start_idx += self.stride
            fold += 1
            
        return pd.DataFrame(self.results)
        
    def _calibrate(self, train_data: torch.Tensor) -> Tuple[HybridDynamicalSystem, RegimeSwitchingStrategy]:
        """
        Train model and setup strategy on historical data.
        """
        # 1. Train Dynamics Model
        # Create dataset of trajectories
        traj_len = 20
        trajectories = []
        # train_data shape (T, 1, F)
        data_squeeze = train_data.squeeze(1)
        for i in range(0, len(data_squeeze) - traj_len, 5):
            trajectories.append(data_squeeze[i:i+traj_len])
            
        if len(trajectories) == 0:
            # Fallback if window too small
            trajectories.append(data_squeeze)
            
        dataset = TensorDataset(torch.stack(trajectories))
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        feature_dim = train_data.shape[2]
        model = HybridDynamicalSystem(feature_dim=feature_dim, epsilon=0.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
        loss_fn = TrajectoryLoss(LossConfig(lambda_mse=1.0))
        trainer = SystemIdentification(model, optimizer, loss_fn)
        
        # Train loop
        for _ in range(5):
            def unpacked_dataloader():
                for batch in dataloader:
                    yield batch[0]
            trainer.train_epoch(unpacked_dataloader())
            
        # 2. Setup Components
        control_system = ControlAffineSystem(model)
        
        # Calibrate Regime Detector?
        # We use fixed heuristics for now as per "Keep It Simple", 
        # but we could tune thresholds here.
        regime_detector = RegimeDetector(window_size=50)
        
        risk_manager = RiskManager(RiskConfig(base_inventory_limit=5.0))
        
        strategy = RegimeSwitchingStrategy(
            control_system,
            regime_detector,
            risk_manager,
            scaling_factor=5.0, # Fixed scaling for now
            sigmoid_alpha=1000.0
        )
        
        return model, strategy
        
    def _run_backtest(self, strategy: RegimeSwitchingStrategy, test_data: torch.Tensor) -> Dict:
        # Wrapper
        class StrategyWrapper:
            def __init__(self, strat): self.strat = strat
            def solve(self, state, **kwargs):
                return self.strat.get_action(state).unsqueeze(0).repeat(10, 1, 1), 0.0
        
        # Initial state: First frame of test data + 0 inventory
        initial_state = torch.cat([test_data[0], torch.zeros(1, 1)], dim=1)
        
        engine = BacktestEngine(
            StrategyWrapper(strategy),
            initial_state,
            strategy.risk_manager,
            transaction_cost_bps=1.0,
            dt=0.01
        )
        
        # Run
        # Note: We pass test_data. 
        # BacktestEngine.run iterates through it.
        engine.run(steps=len(test_data) - 1, market_data=test_data)
        
        return engine.get_metrics()
