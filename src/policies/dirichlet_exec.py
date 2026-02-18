"""
Dirichlet Policy Execution Utilities

This module provides evaluation-time wrappers for Dirichlet-based portfolio policies.
These utilities improve deployment characteristics without requiring retraining.

Key Features:
- Deterministic weight extraction (mean, mode)
- Controlled stochastic sampling (scaled concentration)
- Regime-aware gating with smoothing
- Cash tilting for risk management
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any


def dirichlet_mean(alpha: np.ndarray) -> np.ndarray:
    """
    Compute mean weights from Dirichlet concentration parameters.
    
    For Dirichlet(α), the mean is: E[w] = α / sum(α)
    
    Args:
        alpha: Concentration parameters, shape (n_assets,)
        
    Returns:
        Mean weights, shape (n_assets,), sums to 1.0
    """
    alpha = np.asarray(alpha, dtype=np.float64)
    alpha = np.clip(alpha, 1e-12, None)  # Avoid division by zero
    return alpha / alpha.sum()


def dirichlet_mode_boundary_aware(alpha: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Compute Dirichlet mode (MAP estimate) with boundary handling.
    
    The mode exists in the interior only when all α > 1.
    When some α ≤ 1, the mode sits on the boundary (some weights → 0).
    
    Args:
        alpha: Concentration parameters, shape (n_assets,)
        eps: Small value for numerical stability
        
    Returns:
        Mode weights, shape (n_assets,), sums to 1.0
    """
    alpha = np.asarray(alpha, dtype=np.float64)
    alpha = np.clip(alpha, eps, None)
    
    # Identify active set: indices where α > 1
    active = alpha > 1.0
    
    if not np.any(active):
        # All α ≤ 1: No unique interior mode
        # Tie-break: one-hot on argmax
        weights = np.zeros_like(alpha)
        weights[np.argmax(alpha)] = 1.0
        return weights
    
    # Compute mode for active set
    weights = np.zeros_like(alpha)
    alpha_shifted = alpha - 1.0
    alpha_shifted = np.maximum(alpha_shifted, 0.0)  # Zero out inactive
    
    mode_sum = alpha_shifted.sum()
    if mode_sum > eps:
        weights = alpha_shifted / mode_sum
    else:
        # Fallback to uniform on active set
        weights[active] = 1.0 / np.sum(active)
    
    return weights


def controlled_stochastic_sample(
    alpha: np.ndarray,
    s: float,
    rng: Optional[np.random.Generator] = None,
    eps: float = 1e-12
) -> np.ndarray:
    """
    Sample from scaled Dirichlet: Dirichlet(s·α)
    
    Scaling by s > 1:
    - Preserves mean: E[w] = α/sum(α)
    - Reduces variance: Var ∝ 1/(s·α₀ + 1)
    - Lower turnover for larger s
    
    Recommended scales:
    - s=2: Modest variance reduction
    - s=5: Balanced (recommended)
    - s=10: Very tight around mean
    
    Args:
        alpha: Concentration parameters, shape (n_assets,)
        s: Scaling factor (s ≥ 1)
        rng: Random number generator (creates new if None)
        eps: Minimum alpha value for numerical stability
        
    Returns:
        Sampled weights, shape (n_assets,), sums to 1.0
    """
    alpha = np.asarray(alpha, dtype=np.float64)
    alpha = np.clip(alpha, eps, None)
    
    if rng is None:
        rng = np.random.default_rng()
    
    scaled_alpha = s * alpha
    return rng.dirichlet(scaled_alpha)


def apply_cash_tilt(
    weights: np.ndarray,
    cash_idx: int,
    target_cash: float
) -> np.ndarray:
    """
    Apply defensive cash allocation by setting cash weight to target
    and renormalizing remaining assets.
    
    Args:
        weights: Current weights, shape (n_assets,)
        cash_idx: Index of cash asset (typically last)
        target_cash: Target cash weight (0.0 to 1.0)
        
    Returns:
        Adjusted weights with target cash allocation
    """
    weights = weights.copy()
    target_cash = float(np.clip(target_cash, 0.0, 1.0))
    
    # Get non-cash indices
    rest_indices = np.arange(len(weights)) != cash_idx
    
    # Set cash to target
    weights[cash_idx] = target_cash
    
    # Renormalize rest to sum to (1 - target_cash)
    rest_sum = weights[rest_indices].sum()
    if rest_sum > 1e-12:
        weights[rest_indices] *= (1.0 - target_cash) / rest_sum
    else:
        # If rest is ~0, distribute uniformly
        n_rest = np.sum(rest_indices)
        if n_rest > 0:
            weights[rest_indices] = (1.0 - target_cash) / n_rest
    
    return weights


def smooth_weights(
    w_prev: np.ndarray,
    w_new: np.ndarray,
    eta: float
) -> np.ndarray:
    """
    Apply exponentially weighted moving average (EWMA) smoothing.
    
    Formula: w_t = (1-η)·w_{t-1} + η·w̃_t
    
    Lower η → slower rebalancing → lower turnover
    
    Recommended values:
    - η = 0.05: Very slow (risk-off)
    - η = 0.10: Moderate (neutral)
    - η = 0.20: Responsive (risk-on)
    
    Args:
        w_prev: Previous weights, shape (n_assets,)
        w_new: New target weights, shape (n_assets,)
        eta: Smoothing parameter (0 to 1)
        
    Returns:
        Smoothed weights, shape (n_assets,)
    """
    eta = float(np.clip(eta, 0.0, 1.0))
    return (1.0 - eta) * w_prev + eta * w_new


def gated_deterministic_step(
    alpha: np.ndarray,
    w_prev: np.ndarray,
    regime_signals: Tuple[float, float, float],
    params: Dict[str, Any]
) -> Tuple[np.ndarray, str]:
    """
    Execute one step of gated deterministic policy.
    
    Applies regime-aware smoothing and optional cash tilting based on
    market conditions (volatility, drawdown, trend).
    
    Args:
        alpha: Dirichlet concentration parameters from policy
        w_prev: Previous weights
        regime_signals: Tuple of (volatility, drawdown, trend)
        params: Configuration dict with keys:
            - vol_hi: High volatility threshold
            - dd_hi: High drawdown threshold
            - trend_hi: Trend threshold (default: 0.0)
            - base_method: 'mean' or 'map' (default: 'mean')
            - use_cash_tilt: Enable cash tilting in risk-off (default: True)
            - cash_idx: Index of cash asset
            - risk_off_cash: Target cash in risk-off (default: 0.30)
            - eta_risk_off: Smoothing in risk-off (default: 0.05)
            - eta_neutral: Smoothing in neutral (default: 0.10)
            - eta_risk_on: Smoothing in risk-on (default: 0.20)
            
    Returns:
        Tuple of (new_weights, regime_name)
    """
    vol, dd, trend = regime_signals
    
    # Determine regime
    vol_hi = params.get('vol_hi', 0.20)
    dd_hi = params.get('dd_hi', 0.15)
    trend_hi = params.get('trend_hi', 0.0)
    
    if (dd > dd_hi) or (vol > vol_hi):
        regime = 'risk_off'
    elif (trend > trend_hi) and (vol <= vol_hi):
        regime = 'risk_on'
    else:
        regime = 'neutral'
    
    # Convert alpha to deterministic weights
    base_method = params.get('base_method', 'mean')
    if base_method == 'mean':
        w = dirichlet_mean(alpha)
    elif base_method == 'map':
        w = dirichlet_mode_boundary_aware(alpha)
    else:
        w = dirichlet_mean(alpha)
    
    # Apply cash tilt in risk-off
    if regime == 'risk_off' and params.get('use_cash_tilt', True):
        cash_idx = params.get('cash_idx', len(w) - 1)
        target_cash = params.get('risk_off_cash', 0.30)
        w = apply_cash_tilt(w, cash_idx, target_cash)
        eta = params.get('eta_risk_off', 0.05)
    elif regime == 'risk_on':
        eta = params.get('eta_risk_on', 0.20)
    else:  # neutral
        eta = params.get('eta_neutral', 0.10)
    
    # Apply EWMA smoothing
    w = smooth_weights(w_prev, w, eta)
    
    # Ensure valid weights
    w = np.clip(w, 0.0, None)
    w = w / w.sum()
    
    return w, regime
