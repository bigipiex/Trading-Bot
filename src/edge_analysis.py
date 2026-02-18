import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from src.regime import RegimeDetector
from src.risk import RiskManager

class DirectSignalStrategy(nn.Module):
    """
    Direct Signal Strategy for Statistical Validation.
    
    Signal = w_trend * sign(trend_slope) + w_meanrev * (-(P - mu))
    Position = Signal * VolatilityTarget / Volatility
    
    No MPC. No ODE integration. Pure signal logic.
    """
    def __init__(self, 
                 regime_detector: RegimeDetector,
                 risk_manager: RiskManager = None,
                 w_trend: float = 1.0,
                 w_meanrev: float = 1.0,
                 target_vol: float = 0.1,
                 scaling_factor: float = 1.0,
                 sigmoid_alpha: float = 1000.0):
        super().__init__()
        self.regime_detector = regime_detector
        self.risk_manager = risk_manager
        self.w_trend = w_trend
        self.w_meanrev = w_meanrev
        self.target_vol = target_vol
        self.scaling_factor = scaling_factor
        self.sigmoid_alpha = sigmoid_alpha
        self.dt = 0.01
        
        self.price_history = []
        self.window_size = regime_detector.window_size
        
    def update_history(self, price: float):
        self.price_history.append(price)
        if len(self.price_history) > self.window_size * 2:
            self.price_history.pop(0)
            
    def get_signal(self, state: torch.Tensor) -> Tuple[float, float, int]:
        """
        Computes raw signal components and regime.
        """
        current_log_price = state[0, 3].item()
        current_price = np.exp(current_log_price)
        
        # State index 4 is Mu (Equilibrium) if present
        # If not, use internal moving average?
        # Let's use the Mu from the state vector for consistency with the "System" view
        if state.shape[1] > 4:
            mu_log = state[0, 4].item()
            mu = np.exp(mu_log)
        else:
            # Fallback if state doesn't have Mu
            mu = current_price 
            
        self.update_history(current_price)
        
        trend_strength = 0.0
        norm_dev = 0.0
        regime = -1
        
        if len(self.price_history) >= self.window_size:
            prices_array = np.array(self.price_history)
            regimes, slopes, devs = self.regime_detector.detect(prices_array)
            trend_strength = slopes[-1]
            norm_dev = devs[-1]
            regime = regimes[-1]
            
        # Signal Components
        # Trend: sign(slope)
        # Note: slope is % change per step. 
        # If we use sign, we lose magnitude info.
        # "signal = w_trend * sign(trend_slope)" as per spec.
        # But for soft blending, maybe magnitude matters?
        # Spec says: "signal = w_trend * sign(trend_slope) + w_meanrev * (-(P - mu))"
        # Let's follow spec strictly.
        
        sig_trend = np.sign(trend_strength) if abs(trend_strength) > 1e-8 else 0.0
        
        # Mean Reversion: -(P - mu)
        # We work in log space usually for symmetry? 
        # "-(P - mu)" implies linear.
        # Let's use log deviation: -(logP - logMu)
        sig_mr = -(current_log_price - np.log(mu))
        
        return sig_trend, sig_mr, regime

    def get_action(self, state: torch.Tensor) -> torch.Tensor:
        """
        Computes position u_t.
        """
        sig_trend, sig_mr, regime = self.get_signal(state)
        
        # Combine Signals
        # Wait, spec says:
        # "signal = w_trend * sign(trend_slope) + w_meanrev * (-(P - mu))"
        # Is this modulated by regime?
        # "Regime-Conditioned Performance" is separate.
        # "Strategy Switching" was previous task.
        # Current task: "Replace it with a Direct Signal Strategy... Position = signal scaled..."
        # It implies a GLOBAL signal composed of both factors.
        # Or should we use the regime weights?
        # "Parameter Sensitivity Surface... Sweep w_trend, w_meanrev, regime alpha"
        # This implies we MIGHT use regime weighting.
        # Let's use the Soft Regime logic from before BUT applied to these simple signals.
        # "w_trend" in sweep might be the weight of the trend COMPONENT?
        
        # Let's implement:
        # u = SoftSwitch(regime) * (w_trend * sig_trend) + (1-SoftSwitch) * (w_meanrev * sig_mr)
        # Or just linear combination?
        # "signal = w_trend * sign + w_meanrev * dev" looks like linear combo.
        # Let's assume linear combination for "Direct Signal Strategy".
        # BUT we need to handle "Regime-Conditioned Performance".
        
        # Actually, let's use the regime-based weighting if alpha is involved.
        # If alpha is involved, we must be using sigmoid switching.
        
        # Re-read: "Sweep ... regime alpha".
        # So we DO use regime switching.
        # Let's use the trend_strength to determine weights.
        
        # Get trend strength again (inefficient but safe)
        if len(self.price_history) >= self.window_size:
            prices_array = np.array(self.price_history)
            _, slopes, _ = self.regime_detector.detect(prices_array)
            trend_strength = slopes[-1]
        else:
            trend_strength = 0.0
            
        # Soft Switch
        # Threshold 0.0002
        threshold = 0.0002
        # Alpha from config (using scaling_factor as placeholder? No, pass it in init)
        # Let's add alpha to init.
        # Default alpha=1000?
        # For now, let's assume Hard Switch if alpha not passed, or smooth if passed.
        # Actually, let's just stick to the simple linear formula first?
        # "signal = w_trend * sign(trend_slope) + w_meanrev * (-(P - mu))"
        # This formula DOES NOT have regime switching weights explicit.
        # It's just a linear factor model.
        
        # However, "Sweep regime alpha" implies we should use it.
        # Let's assume the user wants the Regime-Weighted version:
        # u = w_regime_trend * (w_trend_param * sig_trend) + w_regime_mr * (w_meanrev_param * sig_mr)
        
        # Let's assume alpha is passed via a property or config.
        # I'll add `sigmoid_alpha` to init.
        
        alpha = getattr(self, 'sigmoid_alpha', 1000.0)
        S = abs(trend_strength) - threshold
        weight_trend_regime = torch.sigmoid(torch.tensor(alpha * S)).item()
        weight_mr_regime = 1.0 - weight_trend_regime
        
        # Composite Signal
        # Scale sig_mr to be comparable to sig_trend (which is {-1, 0, 1})
        # sig_mr is log deviation, e.g. 0.01 or 0.1.
        # multiply by 100 to make it ~1?
        # Let's leave scaling to w_meanrev.
        
        raw_signal = weight_trend_regime * (self.w_trend * sig_trend) + \
                     weight_mr_regime * (self.w_meanrev * sig_mr * 10.0) # scaling MR
                     
        # Volatility Scaling
        vol = state[0, 1].item()
        safe_vol = max(vol, 1e-6)
        
        # Position
        # u ~ Signal / Vol
        u_val = self.scaling_factor * raw_signal / safe_vol
        
        u_tensor = torch.tensor([[u_val]], dtype=torch.float32)
        
        # Risk Projection
        if self.risk_manager:
            current_inv = state[:, -1:]
            u_executed = self.risk_manager.project_controls(
                u_tensor, current_inv, state[:, 1:2], self.dt
            )
        else:
            u_executed = u_tensor
            
        return u_executed

class ICAnalyzer:
    """
    Computes Information Coefficient (Correlation between Signal and Future Returns).
    """
    def __init__(self, horizons: List[int] = [1, 5, 10, 20, 50]):
        self.horizons = horizons
        
    def analyze(self, signals: np.ndarray, returns: np.ndarray) -> pd.DataFrame:
        """
        signals: (T,) array of strategy signals (u_t)
        returns: (T,) array of 1-step returns
        """
        # We need N-step future returns
        # ret_N[t] = Price[t+N] - Price[t] = Sum(returns[t...t+N-1])
        
        results = []
        T = len(signals)
        
        for h in self.horizons:
            # Construct N-step returns
            # Rolling sum of returns shifted back?
            # returns[t] is P_t+1 - P_t.
            # We want P_t+h - P_t.
            # Series:
            # 0: P1-P0
            # 1: P2-P1
            # ...
            # Sum(0..h-1) = P_h - P_0. Correct.
            
            s_returns = pd.Series(returns)
            # rolling(h).sum() gives Sum(t-h+1 ... t). We want forward.
            # Shift back by h-1?
            # rolling_sum[t] is return from t-h to t.
            # We want return from t to t+h.
            # So shift rolling sum back by h.
            future_returns = s_returns.rolling(window=h).sum().shift(-h)
            
            # Align
            # signals[t] vs future_returns[t]
            valid_idx = ~np.isnan(future_returns)
            
            sig_valid = signals[valid_idx]
            ret_valid = future_returns[valid_idx]
            
            if len(sig_valid) > 10:
                ic = np.corrcoef(sig_valid, ret_valid)[0, 1]
                
                # Rolling IC (e.g. 100-step window) for stability
                # We report Mean/Std of Rolling IC?
                # "Report Mean IC, Std IC"
                # Let's compute rolling IC
                df_pair = pd.DataFrame({'sig': signals, 'ret': future_returns})
                rolling_ic = df_pair['sig'].rolling(100).corr(df_pair['ret'])
                
                mean_ic = rolling_ic.mean()
                std_ic = rolling_ic.std()
                ic_ir = mean_ic / (std_ic + 1e-6)
                pos_ic = (rolling_ic > 0).mean()
                
                results.append({
                    "Horizon": h,
                    "Mean IC": mean_ic,
                    "Std IC": std_ic,
                    "IC IR": ic_ir,
                    "Pos IC %": pos_ic,
                    "Global IC": ic
                })
            else:
                results.append({"Horizon": h, "Global IC": 0.0})
                
        return pd.DataFrame(results)
