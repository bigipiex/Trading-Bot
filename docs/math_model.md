# Mathematical Modeling Approach

This document defines the mathematical framework for the "Gradient-Based Nonlinear Dynamical Trading System".

## 1. State-Space Formulation

The core idea is to model the evolution of the market price $P(t)$ and potentially other latent variables (like volatility $\sigma(t)$ or sentiment $S(t)$) as a continuous-time dynamical system.

We define the state vector $X(t) \in \mathbb{R}^n$ as:
$$ X(t) = \begin{bmatrix} P(t) \\ \sigma(t) \\ S(t) \\ \dots \end{bmatrix} $$

The dynamics are governed by a system of differential equations:
$$ \frac{dX}{dt} = f(X(t), t, \theta) + \epsilon(t) $$

where:
- $f$: A nonlinear vector-valued function representing the deterministic drift and interactions.
- $\theta$: A vector of trainable parameters (e.g., weights in a Neural ODE or coefficients in a physical model).
- $\epsilon(t)$: Stochastic noise (representing market microstructure noise), often modeled as a Wiener process $dW_t$.

### Example Specification of $f$
Instead of a black-box RNN, we can structure $f$ to respect financial constraints:
$$ \frac{dP}{dt} = \mu(S, \theta) P + \sigma(S, \theta) P \cdot \text{noise} $$
$$ \frac{dS}{dt} = -\kappa (S - \bar{S}) + \text{forcing}(P) $$
Here, $\mu$ is the drift, $\sigma$ is volatility, and $S$ is a mean-reverting sentiment factor.

## 2. Optimization Strategy: The Cost Functional $J(\theta)$

Our goal is to find the optimal parameters $\theta^*$ that minimize a comprehensive cost functional over a time horizon $T$:

$$ J(\theta) = \int_{t_0}^{T} L(X(t), \hat{X}(t), u(t)) dt + \Omega(\theta) $$

Where:
1.  **Prediction Error**: $L_{pred} = (P_{real}(t) - P_{model}(t))^2$
2.  **Risk Penalty**: $L_{risk} = \lambda_1 \cdot \text{Drawdown}(P_{model}) + \lambda_2 \cdot \text{Volatility}(P_{model})$
3.  **Regularization**: $\Omega(\theta) = \lambda_3 \|\theta\|_2^2$ (L2 norm to prevent overfitting).

The optimization problem is:
$$ \theta^* = \arg\min_{\theta} J(\theta) $$

## 3. Gradient Computation

We use **Automatic Differentiation (AD)** provided by PyTorch to compute the gradient of the scalar cost $J$ with respect to the vector parameters $\theta$:

$$ \nabla_\theta J = \frac{\partial J}{\partial \theta} $$

### Computational Graph
1.  **Forward Pass**: Integrate the system $dX/dt = f(X, t, \theta)$ from $t_0$ to $T$ using a numerical solver (e.g., Runge-Kutta 4 or Euler method implemented in PyTorch). This generates a trajectory of states $\hat{X}_{t_0:T}$.
2.  **Loss Calculation**: Compute $J(\theta)$ based on the trajectory and real data.
3.  **Backward Pass**: PyTorch's Autograd engine traverses the graph backwards from $J$ to $\theta$, applying the chain rule at each step (including through the ODE solver steps). This is effectively **Backpropagation Through Time (BPTT)** for continuous systems.

### Optimization Algorithm
We use standard gradient-based optimizers:
-   **Adam (Adaptive Moment Estimation)**: Good default for non-convex landscapes.
-   **SGD with Momentum**: Sometimes generalizes better.

$$ \theta_{k+1} = \theta_k - \eta \cdot \nabla_\theta J(\theta_k) $$

## 4. Risk Modeling (Mathematical)

Risk is integrated directly into the optimization or as a hard constraint.

### Volatility-Adjusted Position Sizing (Inverse Volatility)
Let $\sigma_t$ be the estimated volatility. The target position size $u_t$ is:
$$ u_t \propto \frac{1}{\sigma_t} $$
This ensures constant risk exposure regardless of market regime.

### Drawdown Constraint
We can penalize maximum drawdown in the loss function:
$$ \text{DD}(t) = \max_{\tau \in [0, t]} (P(\tau) - P(t)) $$
$$ J_{DD} = \int \max(0, \text{DD}(t) - \text{Threshold})^2 dt $$

## 5. Research Extensions

### Neural Ordinary Differential Equations (Neural ODEs)
Instead of a fixed functional form, let $f$ be a neural network:
$$ \frac{dX}{dt} = \text{NeuralNet}(X(t), t; \theta) $$
This allows the model to learn arbitrary complex dynamics from data while maintaining the continuous-time inductive bias.

### Stochastic Differential Equations (SDEs)
Explicitly modeling the noise term:
$$ dX_t = f(X_t, t)dt + g(X_t, t)dW_t $$
This requires SDE solvers (available in `torchsde`) and allows for probabilistic forecasts (distributions of possible future paths).
