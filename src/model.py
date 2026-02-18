import torch
import torch.nn as nn
import torch.nn.functional as F

class PhysicsDrift(nn.Module):
    """
    Implements the interpretable physics-based component of the dynamics.
    
    Structure:
    - Assumes the state vector X includes [LogReturns, Volatility, Momentum, LogPrice, Mu]
    - Mu is the dynamic equilibrium level (moving average).
    
    Equation for LogPrice (P):
    dP/dt = kappa * (Mu - P) + beta * Momentum
    
    Equation for Equilibrium (Mu):
    dMu/dt = gamma * (P - Mu)  (Exponential Moving Average dynamics)
    
    Equation for Momentum (M):
    dM/dt = -decay * M
    """
    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Trainable parameters
        # Kappa: Mean reversion speed to Mu
        self.raw_kappa = nn.Parameter(torch.tensor([0.5])) 
        
        # Gamma: Adaptation rate of Mu (how fast equilibrium updates)
        # Should be slower than Kappa usually.
        self.raw_gamma = nn.Parameter(torch.tensor([0.05]))
        
        # Beta: Coupling coefficient for Momentum -> Price
        self.beta = nn.Parameter(torch.tensor([0.1]))
        
        # Decay for momentum
        self.raw_decay = nn.Parameter(torch.tensor([0.5]))
        
    @property
    def kappa(self):
        return F.softplus(self.raw_kappa)
        
    @property
    def gamma(self):
        return F.softplus(self.raw_gamma)
        
    @property
    def decay(self):
        return F.softplus(self.raw_decay)

    def forward(self, t: float, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the physics-based drift.
        x shape: (batch_size, feature_dim)
        Features: [0:LogRet, 1:Vol, 2:Mom, 3:LogPrice, 4:Mu]
        """
        # Extract components
        log_ret = x[:, 0:1]
        vol = x[:, 1:2]
        momentum = x[:, 2:3]
        log_price = x[:, 3:4]
        mu = x[:, 4:5]
        
        # 1. Price Dynamics: dP/dt
        # d(LogPrice)/dt = kappa * (Mu - LogPrice) + beta * Momentum
        d_log_price = self.kappa * (mu - log_price) + self.beta * momentum
        
        # 2. Equilibrium Dynamics: dMu/dt
        # Mu follows Price with lag (EMA)
        d_mu = self.gamma * (log_price - mu)
        
        # 3. Momentum Dynamics: dM/dt
        d_momentum = -self.decay * momentum
        
        # 4. Volatility Dynamics: dV/dt
        d_vol = -0.1 * (vol - 1.0)
        
        # 5. Log Returns: d(LogRet)/dt
        d_log_ret = -0.5 * log_ret
        
        # Stack derivatives
        dx_dt = torch.cat([d_log_ret, d_vol, d_momentum, d_log_price, d_mu], dim=1)
        
        return dx_dt

class NeuralCorrection(nn.Module):
    """
    Residual neural network to capture non-linearities and complex interactions
    not modeled by the simple physics term.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim), # +1 for time t
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Initialize weights to be small
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, t: float, x: torch.Tensor) -> torch.Tensor:
        # Augment x with time t
        t_tensor = torch.full((x.shape[0], 1), t, device=x.device, dtype=x.dtype)
        aug_x = torch.cat([x, t_tensor], dim=1)
        return self.net(aug_x)

class HybridDynamicalSystem(nn.Module):
    """
    Combines Physics-based drift and Neural residual correction.
    dX/dt = f_physics(X, theta_phy) + epsilon * f_neural(X, theta_nn)
    """
    def __init__(self, feature_dim: int, epsilon: float = 0.1):
        super().__init__()
        self.physics = PhysicsDrift(feature_dim)
        self.neural = NeuralCorrection(feature_dim)
        self.epsilon = epsilon
        
    def forward(self, t: float, x: torch.Tensor) -> torch.Tensor:
        f_phy = self.physics(t, x)
        f_nn = self.neural(t, x)
        return f_phy + self.epsilon * f_nn
