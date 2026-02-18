import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple
from src.regime import RegimeDetector
from src.risk import RiskManager

class RegimeConditionalStrategy(nn.Module):
    """
    Direct Signal Strategy with Regime-Conditional Parameters.
    
    Regimes:
    - Trend (T): Uses weights (w_trend_T, w_meanrev_T) -> Usually (High, Low)
    - MeanRev (MR): Uses weights (w_trend_MR, w_meanrev_MR) -> Usually (Low, High)
    
    Signal blending:
    u = p_trend * Signal(weights_T) + p_mr * Signal(weights_MR)
    """
    def __init__(self, 
                 regime_detector: RegimeDetector,
                 risk_manager: RiskManager = None,
                 # Regime T weights (Momentum focus)
                 w_trend_T: float = 1.0,
                 w_meanrev_T: float = 0.0,
                 # Regime MR weights (Reversion focus)
                 w_trend_MR: float = 0.0,
                 w_meanrev_MR: float = 1.0,
                 # Config
                 target_vol: float = 0.1,
                 scaling_factor: float = 1.0,
                 sigmoid_alpha: float = 1000.0,
                 # Swing Trading Config
                 prediction_horizon: int = 1,
                 signal_smoothing: float = 0.1, # EMA alpha (1.0 = no smoothing)
                 signal_deadband: float = 0.0,
                 min_holding_period: int = 0):
        super().__init__()
        self.regime_detector = regime_detector
        self.risk_manager = risk_manager
        
        self.w_trend_T = w_trend_T
        self.w_meanrev_T = w_meanrev_T
        self.w_trend_MR = w_trend_MR
        self.w_meanrev_MR = w_meanrev_MR
        
        self.target_vol = target_vol
        self.scaling_factor = scaling_factor
        self.sigmoid_alpha = sigmoid_alpha
        self.dt = 0.01
        
        self.price_history = []
        self.window_size = regime_detector.window_size

        # Swing State
        self.prediction_horizon = prediction_horizon
        self.signal_smoothing = signal_smoothing
        self.signal_deadband = signal_deadband
        self.min_holding_period = min_holding_period
        
        self.smoothed_signal = 0.0
        self.last_trade_step = -min_holding_period - 1
        self.current_step = 0
        self.last_u = None
        
    def update_history(self, price: float):
        self.price_history.append(price)
        if len(self.price_history) > self.window_size * 2:
            self.price_history.pop(0)
            
    def get_signal_components(self, state: torch.Tensor) -> Tuple[float, float, float]:
        """
        Returns (sig_trend, sig_mr, trend_strength)
        """
        current_log_price = state[0, 3].item()
        current_price = np.exp(current_log_price)
        
        if state.shape[1] > 4:
            mu_log = state[0, 4].item()
            mu = np.exp(mu_log)
        else:
            mu = current_price
            
        self.update_history(current_price)
        
        trend_strength = 0.0
        
        if len(self.price_history) >= self.window_size:
            prices_array = np.array(self.price_history)
            _, slopes, _ = self.regime_detector.detect(prices_array)
            trend_strength = slopes[-1]
            
        # Raw Signals
        # Trend: sign(slope)
        sig_trend = np.sign(trend_strength) if abs(trend_strength) > 1e-8 else 0.0
        
        # Mean Rev: -(logP - logMu)
        # Note: In pure OU test, we want to BUY when P < Mu.
        # logP < logMu => -(logP - logMu) > 0 => Buy.
        # So w_meanrev should be POSITIVE for standard MR.
        # Previous tests used -2.0 because they likely flipped the signal definition or had momentum bias.
        sig_mr = -(current_log_price - np.log(mu))
        
        return sig_trend, sig_mr, trend_strength

    def get_action(self, state: torch.Tensor) -> torch.Tensor:
        self.current_step += 1
        
        # Check Holding Period
        if self.last_u is not None and (self.current_step - self.last_trade_step < self.min_holding_period):
            return self.last_u
        
        sig_trend, sig_mr, trend_strength = self.get_signal_components(state)
        
        # Soft Regime Probability
        # p_trend = sigmoid(alpha * (|slope| - threshold))
        threshold = 0.0002
        S = abs(trend_strength) - threshold
        p_trend = torch.sigmoid(torch.tensor(self.sigmoid_alpha * S)).item()
        p_mr = 1.0 - p_trend
        
        # Compute Signal for T Regime
        raw_T = self.w_trend_T * sig_trend + self.w_meanrev_T * (sig_mr * 10.0) # Scaling MR
        
        # Compute Signal for MR Regime
        raw_MR = self.w_trend_MR * sig_trend + self.w_meanrev_MR * (sig_mr * 10.0)
        
        # Blend
        final_signal = p_trend * raw_T + p_mr * raw_MR
        
        # Signal Smoothing (EMA)
        if self.current_step == 1:
            self.smoothed_signal = final_signal
        else:
            self.smoothed_signal = (1.0 - self.signal_smoothing) * self.smoothed_signal + \
                                   self.signal_smoothing * final_signal
                                   
        # Deadband
        if abs(self.smoothed_signal) < self.signal_deadband:
            output_signal = 0.0
        else:
            output_signal = self.smoothed_signal
            
        # Holding Period Logic: If within lock period, do not update output_signal?
        # But we don't know what the previous output was if we zeroed it or not.
        # Let's ignore holding period inside Strategy for now, rely on smoothing to reduce turnover.
        # User asked for "Structure Turnover Reduction", smoothing is key.
        
        # Volatility Scaling
        vol = state[0, 1].item()
        safe_vol = max(vol, 1e-6)
        
        # Position
        u_val = self.scaling_factor * output_signal / safe_vol
        u_tensor = torch.tensor([[u_val]], dtype=torch.float32)
        
        # Update state if signal changed significantly?
        # For now, just update last_u every time we calculate fresh
        self.last_u = u_tensor
        self.last_trade_step = self.current_step
        
        # Risk Projection (Tail Risk Aware)
        if self.risk_manager:
            current_inv = state[:, -1:]
            
            # Need return and equity for Tail Risk
            # Mock or estimate? BacktestEngine should pass them if using updated engine.
            # But get_action usually just takes state.
            # We can't easily get external context here without passing it.
            # Let's assume standard projection first.
            # Tail risk is handled inside RiskManager.project_controls
            # But it needs `current_return` arg.
            # If we don't pass it, it uses default 0.0 (no jump check).
            # This is fine for strategy logic; BacktestEngine calls project_controls with args.
            
            # Wait, BacktestEngine calls risk_manager.project_controls DIRECTLY on u_optimal.
            # Strategy returns u_optimal (unconstrained).
            # So we just return u_tensor here.
            pass
            
        return u_tensor
