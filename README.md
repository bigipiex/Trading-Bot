# Neural-Physics Hybrid Trading Bot

A research platform for **System Identification** and **Optimal Control** in financial markets.
This system learns differential equations ($dX/dt$) governing market dynamics and uses Model Predictive Control (MPC) to execute trades.

## 🏗 Architecture

### 1. Data Layer (`src/features.py`)
- Transforms raw price/volume into a state vector $X(t)$.
- **State**: `[LogReturns, Volatility, Momentum, LogPrice, Equilibrium(Mu)]`.
- **Normalization**: Rolling Z-scores and log-space transformations.

### 2. Model Layer (`src/model.py`)
- **Hybrid Neural ODE**: $dX/dt = f_{\text{physics}}(X) + \epsilon \cdot f_{\text{neural}}(X)$.
- **Physics Term**: Interpretable Ornstein-Uhlenbeck dynamics.
  - $dP/dt = \kappa (\mu - P) + \beta \cdot \text{Momentum}$
  - $d\mu/dt = \gamma (P - \mu)$ (Adaptive Equilibrium)
- **Neural Term**: Small residual network to capture non-linearities.

### 3. Control Layer (`src/mpc.py` & `src/strategy.py`)
- **MPC**: Optimizes trajectory $u_{t:t+T}$ to maximize profit functional $J(u)$.
- **SimpleStrategy**: Signal-based sizing $u_t \propto E[R_{t+N}] / \sigma_t$.
- **Horizon**: Critical parameter. Longer horizons ($N=50$) recover signal from noise.

### 4. Risk Layer (`src/risk.py`)
- **Hard Constraints**: $|I(t)| \le I_{\text{max}}$.
- **Volatility Scaling**: $I_{\text{max}} \propto 1/\sigma_t$.
- **Projection**: Clamps control actions post-optimization to guarantee safety.

## 🚀 Quick Start

### 1. Installation
```bash
pip install torch numpy pandas tqdm matplotlib
```

### 2. Run Tests
Validate all components:
```bash
# Core Pipeline
python tests/test_pipeline.py

# Dynamics & Solver
python tests/test_dynamics.py

# System Identification (Training)
python tests/test_integration.py

# Control Logic
python tests/test_control.py

# Backtest
python tests/test_backtest.py
```

### 3. Signal Analysis
Analyze Signal-to-Noise Ratio (SNR) across horizons:
```bash
python tests/test_signal_horizon.py
```
*Expected Result*: Correlation improves as Horizon $N$ increases (e.g., from 0.03 at $N=1$ to 0.35 at $N=50$).

## 🧠 Usage Guide

### Training the Model
```python
from src.train import SystemIdentification
from src.model import HybridDynamicalSystem

# Initialize
model = HybridDynamicalSystem(feature_dim=5)
trainer = SystemIdentification(model, optimizer, loss_fn)

# Train on historical trajectories
trainer.train_epoch(dataloader)
```

### Running a Backtest
```python
from src.backtest import BacktestEngine
from src.strategy import SimpleStrategy

# Setup Strategy
strategy = SimpleStrategy(control_system, risk_manager, prediction_horizon=50)

# Run Simulation
engine = BacktestEngine(strategy, initial_state)
engine.run(steps=1000, market_data=data)

# Analyze
metrics = engine.get_metrics()
print(metrics)
```

## ⚠️ Scientific Notes
1.  **Low SNR**: Financial data has extremely low signal-to-noise ratio at $N=1$. Models must be trained and evaluated on longer horizons ($N \ge 20$).
2.  **Drift vs Reversion**: Mean-reversion strategies lose money if the asset trends (drifts) away from the assumed mean. The Adaptive Equilibrium ($\gamma$) parameter is crucial.
3.  **Closed-Loop Stability**: The system includes energy metrics to monitor stability. If $\kappa$ (mean reversion speed) is too high, the system may oscillate.

## 📂 File Structure
- `src/`: Source code.
- `tests/`: Unit and integration tests.
- `data/`: Place your CSV files here.
- `docs/`: Design documents.
