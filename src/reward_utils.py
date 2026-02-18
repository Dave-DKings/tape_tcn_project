"""
Terminal Aggregate Performance Enhancement (TAPE) Reward System

This module implements the sophisticated two-stage reward system that aligns
agent learning with holistic, episode-level financial goals.

Stage 1: Step-Level Risk Filtering
- Immediate penalties for large sudden losses
- Promotes baseline risk aversion

Stage 2: Terminal Aggregate Performance Enhancement (TAPE)
- Episode-level "report card" across multiple financial metrics
- Retrospectively scales all episode rewards based on aggregate performance
- Uses monotonic, skewed, and truncated utility functions

Key Metrics Evaluated:
- Sharpe Ratio: Risk-adjusted returns
- Sortino Ratio: Downside risk-adjusted returns
- Maximum Drawdown: Worst peak-to-trough decline
- Turnover: Portfolio trading frequency
- Skewness: Return distribution shape

Author: AI Assistant
Date: October 2, 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from scipy import stats

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_sharpe_ratio_dsr(returns: np.ndarray, trading_days_per_year: int = 252) -> float:
    """
    Calculate annualized Sharpe ratio specifically for Differential Sharpe Ratio (DSR) computation.
    
    This version is optimized for incremental, window-based calculations and includes
    robust handling of edge cases critical for DSR reward signals.
    
    Args:
        returns: Array of periodic (e.g., daily) returns
        trading_days_per_year: Trading days for annualization (default: 252)
    
    Returns:
        float: Annualized Sharpe ratio (0.0 if calculation not possible)
    
    Notes:
        - Returns 0.0 if fewer than 2 samples (cannot calculate std)
        - Returns 0.0 if std is near zero (flat returns)
        - Assumes risk-free rate is 0 for simplicity in DSR context
    """
    # Ensure there are enough data points
    if len(returns) < 2:
        return 0.0
    
    mean_return = np.mean(returns)
    std_dev = np.std(returns, ddof=1)  # Use sample std (unbiased estimator)
    
    # CRITICAL: Handle zero or near-zero standard deviation
    if std_dev is None or std_dev < 1e-8:
        # Flat returns → no risk → Sharpe is technically undefined
        # Return 0.0 to avoid numerical instability in DSR
        return 0.0
    
    # Calculate annualized Sharpe (assuming rf=0 for DSR simplicity)
    sharpe_ratio = (mean_return / std_dev) * np.sqrt(trading_days_per_year)
    
    # Handle potential NaN/Inf (defensive programming)
    if np.isnan(sharpe_ratio) or np.isinf(sharpe_ratio):
        return 0.0
    
    return float(sharpe_ratio)


def calculate_portfolio_entropy(weights: np.ndarray, epsilon: float = 1e-8) -> float:
    """
    Calculate the entropy of a portfolio allocation to measure diversification.
    
    Higher entropy = more diversified portfolio
    Lower entropy = more concentrated portfolio
    
    Args:
        weights: Portfolio weights (including cash), should sum to 1.0
        epsilon: Small value to prevent log(0)
        
    Returns:
        Entropy value (0 to log(N) where N is number of assets)
        
    Example:
        Equal weights [0.2, 0.2, 0.2, 0.2, 0.2] → High entropy (~1.61)
        Concentrated [0.96, 0.01, 0.01, 0.01, 0.01] → Low entropy (~0.24)
    """
    # Clip weights to avoid log(0)
    weights_clipped = np.clip(weights, epsilon, 1.0)
    
    # Normalize to ensure sum = 1.0 after clipping
    weights_clipped = weights_clipped / np.sum(weights_clipped)
    
    # Calculate Shannon entropy: H = -Σ(p_i * log(p_i))
    entropy = -np.sum(weights_clipped * np.log(weights_clipped))
    
    return entropy


def calculate_sharpe_ratio(returns: np.ndarray, 
                          risk_free_rate: float = 0.02, 
                          trading_days_per_year: int = 252) -> float:
    """
    Calculate annualized Sharpe Ratio.
    
    Sharpe Ratio = (Mean Return - Risk Free Rate) / Std Dev of Returns
    
    Args:
        returns: Array of period returns
        risk_free_rate: Annual risk-free rate
        trading_days_per_year: Trading days per year for annualization
        
    Returns:
        Annualized Sharpe Ratio
    """
    if len(returns) == 0:
        return 0.0
    
    # Convert annual risk-free rate to daily
    daily_rf = (1 + risk_free_rate) ** (1 / trading_days_per_year) - 1
    
    # Calculate excess returns
    excess_returns = returns - daily_rf
    
    # Calculate mean and std
    mean_excess = np.mean(excess_returns)
    std_returns = np.std(returns, ddof=1)
    
    if std_returns == 0:
        return 0.0
    
    # Annualize Sharpe Ratio
    sharpe = (mean_excess / std_returns) * np.sqrt(trading_days_per_year)
    
    return float(sharpe)


def calculate_sortino_ratio(returns: np.ndarray,
                           risk_free_rate: float = 0.02,
                           trading_days_per_year: int = 252) -> float:
    """
    Calculate annualized Sortino Ratio (focuses on downside volatility).
    
    Sortino Ratio = (Mean Return - Risk Free Rate) / Downside Deviation
    
    Args:
        returns: Array of period returns
        risk_free_rate: Annual risk-free rate
        trading_days_per_year: Trading days per year
        
    Returns:
        Annualized Sortino Ratio
    """
    if len(returns) == 0:
        return 0.0
    
    # Convert annual risk-free rate to daily
    daily_rf = (1 + risk_free_rate) ** (1 / trading_days_per_year) - 1
    
    # Calculate excess returns
    excess_returns = returns - daily_rf
    
    # Calculate downside deviation (only negative returns)
    downside_returns = returns[returns < 0]
    
    if len(downside_returns) == 0:
        downside_std = 0.0
    else:
        downside_std = np.std(downside_returns, ddof=1)
    
    if downside_std == 0:
        return 0.0
    
    # Annualize Sortino Ratio
    mean_excess = np.mean(excess_returns)
    sortino = (mean_excess / downside_std) * np.sqrt(trading_days_per_year)
    
    return float(sortino)


def calculate_maximum_drawdown(portfolio_values: np.ndarray) -> float:
    """
    Calculate Maximum Drawdown (MDD).
    
    MDD = Max(Peak - Trough) / Peak
    
    Args:
        portfolio_values: Array of portfolio values over time
        
    Returns:
        Maximum Drawdown as a positive percentage (0.0 to 1.0)
    """
    if len(portfolio_values) == 0:
        return 0.0
    
    # Calculate running maximum
    running_max = np.maximum.accumulate(portfolio_values)
    
    # Calculate drawdown at each point
    drawdowns = (running_max - portfolio_values) / running_max
    
    # Maximum drawdown
    max_dd = np.max(drawdowns)
    
    return float(max_dd)


def calculate_turnover(weight_changes: List[np.ndarray]) -> float:
    """
    Calculate portfolio turnover.
    
    Turnover = Average of sum of absolute weight changes per period
    
    Args:
        weight_changes: List of weight change arrays for each rebalancing
        
    Returns:
        Average turnover per period
    """
    if len(weight_changes) == 0:
        return 0.0
    
    # Calculate turnover for each period
    turnovers = [np.sum(np.abs(change)) for change in weight_changes]
    
    # Return average turnover
    avg_turnover = np.mean(turnovers)
    
    return float(avg_turnover)


def calculate_skewness(returns: np.ndarray) -> float:
    """
    Calculate return distribution skewness.
    
    Positive skew = right tail (more extreme positive returns)
    Negative skew = left tail (more extreme negative returns)
    
    Args:
        returns: Array of returns
        
    Returns:
        Skewness of return distribution
    """
    if len(returns) < 3:
        return 0.0
    
    skew = stats.skew(returns)
    
    return float(skew)


def calculate_volatility(returns: np.ndarray,
                        trading_days_per_year: int = 252) -> float:
    """
    Calculate annualized volatility.
    
    Args:
        returns: Array of returns
        trading_days_per_year: Trading days for annualization
        
    Returns:
        Annualized volatility
    """
    if len(returns) == 0:
        return 0.0
    
    vol = np.std(returns, ddof=1) * np.sqrt(trading_days_per_year)
    
    return float(vol)


def calculate_win_rate(returns: np.ndarray) -> float:
    """
    Calculate win rate (percentage of positive returns).
    
    Args:
        returns: Array of returns
        
    Returns:
        Win rate as decimal (0.0 to 1.0)
    """
    if len(returns) == 0:
        return 0.0
    
    win_rate = np.sum(returns > 0) / len(returns)
    
    return float(win_rate)


def calculate_total_return(portfolio_values: np.ndarray) -> float:
    """
    Calculate total return over the episode.
    
    Args:
        portfolio_values: Array of portfolio values
        
    Returns:
        Total return as decimal
    """
    if len(portfolio_values) == 0:
        return 0.0
    
    total_ret = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
    
    return float(total_ret)


def calculate_calmar_ratio(total_return: float, max_drawdown_abs: float) -> float:
    """Calmar ratio = total return / max drawdown (absolute)."""
    if max_drawdown_abs <= 0:
        return 0.0
    return float(total_return / max_drawdown_abs)


def calculate_omega_ratio(returns: np.ndarray, threshold: float = 0.0) -> float:
    """
    Compute Omega ratio: ratio of gains to losses relative to a threshold.
    """
    if len(returns) == 0:
        return 0.0
    gains = np.maximum(returns - threshold, 0.0)
    losses = np.maximum(threshold - returns, 0.0)
    losses_sum = np.sum(losses)
    if losses_sum == 0:
        return float('inf')
    omega = np.sum(gains) / losses_sum
    return float(omega)


def calculate_ulcer_index(drawdowns: np.ndarray) -> float:
    """
    Ulcer Index = sqrt(mean(square(drawdown%))).
    """
    if drawdowns.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(drawdowns))))


def calculate_cvar(returns: np.ndarray, alpha: float = 0.05) -> float:
    """
    Conditional Value at Risk (CVaR) at level alpha (default 5%).
    Returns expected loss in worst alpha tail (negative value).
    """
    if len(returns) == 0:
        return 0.0
    sorted_returns = np.sort(returns)
    tail_count = max(1, int(np.floor(alpha * len(sorted_returns))))
    tail_losses = sorted_returns[:tail_count]
    return float(np.mean(tail_losses))


def calculate_episode_metrics(portfolio_values: np.ndarray,
                             returns: np.ndarray,
                             weight_changes: List[np.ndarray],
                             risk_free_rate: float = 0.02,
                             trading_days_per_year: int = 252) -> Dict[str, float]:
    """
    Calculate all episode-level performance metrics.
    
    Args:
        portfolio_values: Array of portfolio values over episode
        returns: Array of portfolio returns over episode
        weight_changes: List of weight changes for turnover calculation
        risk_free_rate: Annual risk-free rate
        trading_days_per_year: Trading days per year
        
    Returns:
        Dictionary of metric names to values
    """
    metrics = {}
    
    # Calculate each metric
    metrics['total_return'] = calculate_total_return(portfolio_values)
    metrics['sharpe_ratio'] = calculate_sharpe_ratio(returns, risk_free_rate, trading_days_per_year)
    metrics['sortino_ratio'] = calculate_sortino_ratio(returns, risk_free_rate, trading_days_per_year)
    if len(portfolio_values) > 0:
        running_max = np.maximum.accumulate(portfolio_values)
        drawdown_series = (running_max - portfolio_values) / np.where(running_max == 0, 1.0, running_max)
    else:
        drawdown_series = np.array([])

    max_drawdown = calculate_maximum_drawdown(portfolio_values)
    metrics['max_drawdown'] = -float(max_drawdown)
    metrics['max_drawdown_abs'] = float(max_drawdown)
    metrics['turnover'] = calculate_turnover(weight_changes)
    metrics['skewness'] = calculate_skewness(returns)
    metrics['volatility'] = calculate_volatility(returns, trading_days_per_year)
    metrics['win_rate'] = calculate_win_rate(returns)
    metrics['calmar_ratio'] = calculate_calmar_ratio(metrics['total_return'], metrics['max_drawdown_abs'])
    metrics['omega_ratio'] = calculate_omega_ratio(returns, threshold=0.0)
    metrics['ulcer_index'] = calculate_ulcer_index(drawdown_series)
    metrics['cvar_5pct'] = calculate_cvar(returns, alpha=0.05)
    
    return metrics


def asymmetric_sigmoid_utility(x: float,
                                mu: float,
                                k_minus: float,
                                k_plus: float,
                                a: float,
                                b: float,
                                direction: str = 'increasing') -> float:
    """
    Asymmetric sigmoid utility function for TAPE scoring.
    
    Uses a logistic sigmoid with different steepness below vs above the target:
    - k_minus controls penalty steepness for underperformance
    - k_plus controls reward slope for overperformance
    
    For 'increasing' metrics (Sharpe, Sortino, Skew):
        U(x) = 1 / (1 + exp(-k * (x - mu)))
        Higher x → higher utility
        
    For 'decreasing' metrics (MDD, Turnover):
        U(x) = 1 / (1 + exp(+k * (x - mu)))
        Lower x → higher utility
    
    Key advantages over Gaussian:
    - Monotonic: never penalizes overperformance
    - Always provides gradient (no vanishing signal)
    - Naturally bounded [0, 1]
    
    Args:
        x: Metric value
        mu: Target/midpoint value (50% utility)
        k_minus: Steepness below target (higher = sharper penalty)
        k_plus: Steepness above target (lower = gentler saturation)
        a: Lower truncation bound
        b: Upper truncation bound
        direction: 'increasing' (higher is better) or 'decreasing' (lower is better)
        
    Returns:
        Utility score between 0 and 1
    """
    # Truncate x to bounds
    x_trunc = np.clip(x, a, b)
    
    # Select steepness based on which side of mu we're on
    if direction == 'decreasing':
        # For decreasing metrics (e.g. MDD, Turnover):
        # Higher values = worse (e.g. turnover 0.80 is worse than 0.60)
        # Lower values = better (e.g. MDD -0.05 is better than -0.25)
        # ABOVE mu: bad territory → steep penalty (k_plus)
        # BELOW mu: good territory → gentle reward (k_minus)
        if x_trunc > mu:
            k = k_plus   # Above target → steep penalty
        else:
            k = k_minus  # Below target → gentle reward
        z = -k * (x_trunc - mu)  # Flip sigmoid direction
    else:
        # For increasing metrics (Sharpe, Sortino, Skew):
        # Higher values = better
        # BELOW mu: bad territory → steep penalty (k_minus)
        # ABOVE mu: good territory → gentle reward (k_plus)
        if x_trunc < mu:
            k = k_minus  # Below target → steep penalty
        else:
            k = k_plus   # Above target → gentle reward
        z = k * (x_trunc - mu)
    
    # Clip exponent for numerical stability
    z = np.clip(z, -20.0, 20.0)
    
    utility = 1.0 / (1.0 + np.exp(-z))
    
    return float(utility)


def skewed_utility_function(x: float,
                            mu: float,
                            sigma_sq_minus: float,
                            sigma_sq_plus: float,
                            a: float,
                            b: float) -> float:
    """
    [LEGACY] Gaussian skewed utility function - kept for backward compatibility.
    
    Prefer asymmetric_sigmoid_utility() for new code.
    
    U(x) = exp(-((x - mu)^2) / (2 * sigma^2))
    Where sigma^2 = sigma_sq_minus if x < mu, else sigma_sq_plus
    """
    x_trunc = np.clip(x, a, b)
    
    if x_trunc < mu:
        sigma_sq = sigma_sq_minus
    else:
        sigma_sq = sigma_sq_plus
    
    if sigma_sq <= 0:
        return 1.0 if np.abs(x_trunc - mu) < 1e-6 else 0.0
    
    exponent = -((x_trunc - mu) ** 2) / (2 * sigma_sq)
    utility = np.exp(exponent)
    
    return float(utility)


def calculate_tape_score(metrics: Dict[str, float],
                         profile: Dict[str, Any]) -> float:
    """
    Calculate Terminal Aggregate Performance Enhancement (TAPE) score.
    
    This is the core of Stage 2 reward shaping. It evaluates the agent's
    holistic performance across multiple financial metrics using the
    active utility profile.
    
    TAPE Score = Σ(weight_i * utility_i(metric_i))
    
    Where:
    - weight_i: Importance of metric i in the profile
    - utility_i: Skewed utility function for metric i
    - metric_i: Actual metric value achieved
    
    Args:
        metrics: Dictionary of calculated episode metrics
        profile: Active utility profile with targets and preferences
        
    Returns:
        TAPE score (typically 0 to 1, weighted average of utilities)
    """
    # Extract profile parameters
    mu = profile['mu']
    weights = profile['weights']
    a_bounds = profile['a_bounds']
    b_bounds = profile['b_bounds']
    metrics_order = profile['metrics_order']
    directions = profile.get('directions', ['increasing'] * len(metrics_order))
    
    # Detect profile format: new (k_minus/k_plus) vs legacy (sigma_sq)
    use_sigmoid = 'k_minus' in profile and 'k_plus' in profile
    
    if use_sigmoid:
        k_minus = profile['k_minus']
        k_plus = profile['k_plus']
    else:
        # Legacy Gaussian format
        sigma_sq_minus = profile['sigma_sq_minus']
        sigma_sq_plus = profile['sigma_sq_plus']
    
    # Ensure weights sum to 1
    weights_normalized = weights / np.sum(weights)
    
    # Calculate utility for each metric
    utilities = []
    for i, metric_name in enumerate(metrics_order):
        # Map metric name to actual metric key
        metric_key_map = {
            'sharpe': 'sharpe_ratio',
            'sortino': 'sortino_ratio',
            'mdd': 'max_drawdown',
            'turnover': 'turnover',
            'skew': 'skewness'
        }
        
        metric_key = metric_key_map.get(metric_name, metric_name)
        metric_value = metrics.get(metric_key, 0.0)
        
        if use_sigmoid:
            # New asymmetric sigmoid utility
            utility = asymmetric_sigmoid_utility(
                x=metric_value,
                mu=mu[i],
                k_minus=k_minus[i],
                k_plus=k_plus[i],
                a=a_bounds[i],
                b=b_bounds[i],
                direction=directions[i]
            )
        else:
            # Legacy Gaussian utility (backward compatibility)
            utility = skewed_utility_function(
                x=metric_value,
                mu=mu[i],
                sigma_sq_minus=sigma_sq_minus[i],
                sigma_sq_plus=sigma_sq_plus[i],
                a=a_bounds[i],
                b=b_bounds[i]
            )
        
        utilities.append(utility)
    
    # Calculate weighted average utility
    utilities = np.array(utilities)
    tape_score = np.sum(weights_normalized * utilities)
    
    return float(tape_score)


def apply_tape_scaling(episode_rewards: List[float],
                       tape_score: float,
                       scaling_method: str = 'multiplicative',
                       min_scale: float = 0.5,
                       max_scale: float = 2.0) -> List[float]:
    """
    Apply TAPE score to scale episode rewards retrospectively.
    
    This is the key mechanism that aligns step-level rewards with
    episode-level performance goals.
    
    Args:
        episode_rewards: List of raw step-level rewards
        tape_score: Calculated TAPE score (0 to 1)
        scaling_method: 'multiplicative' or 'additive'
        min_scale: Minimum scaling factor
        max_scale: Maximum scaling factor
        
    Returns:
        List of scaled rewards
    """
    if len(episode_rewards) == 0:
        return []
    
    # Map TAPE score (0 to 1) to scaling range
    scale_range = max_scale - min_scale
    scale_factor = min_scale + (tape_score * scale_range)
    
    if scaling_method == 'multiplicative':
        # Multiply each reward by scale factor
        scaled_rewards = [r * scale_factor for r in episode_rewards]
    elif scaling_method == 'additive':
        # Add bonus based on scale factor
        bonus = (scale_factor - 1.0) * np.mean(np.abs(episode_rewards))
        scaled_rewards = [r + bonus for r in episode_rewards]
    else:
        logger.warning(f"Unknown scaling method: {scaling_method}. Using multiplicative.")
        scaled_rewards = [r * scale_factor for r in episode_rewards]
    
    return scaled_rewards


def step_level_risk_filter(current_balance: float,
                          previous_balance: float,
                          loss_threshold: float = 0.05,
                          penalty_multiplier: float = 2.0) -> float:
    """
    Stage 1: Step-level risk filtering for large sudden losses.
    
    This provides an immediate penalty for large drawdowns to promote
    baseline risk aversion before the TAPE system is applied.
    
    Args:
        current_balance: Portfolio balance after step
        previous_balance: Portfolio balance before step
        loss_threshold: Threshold for "large" loss (e.g., 0.05 = 5%)
        penalty_multiplier: How much to amplify the penalty
        
    Returns:
        Penalty value (0 if no large loss, negative if large loss)
    """
    # Calculate step return
    if previous_balance <= 0:
        return 0.0
    
    step_return = (current_balance - previous_balance) / previous_balance
    
    # Check if loss exceeds threshold
    if step_return < -loss_threshold:
        # Apply penalty proportional to loss magnitude
        excess_loss = abs(step_return) - loss_threshold
        penalty = -excess_loss * penalty_multiplier
        return float(penalty)
    
    return 0.0


def calculate_rolling_performance(episode_history: List[Dict[str, Any]],
                                  window: int = 20) -> Dict[str, float]:
    """
    Calculate rolling performance metrics over recent episodes.
    
    This is used by the profile manager to trigger performance-based
    profile switching.
    
    Args:
        episode_history: List of episode info dictionaries
        window: Number of episodes to include in rolling calculation
        
    Returns:
        Dictionary of rolling performance metrics
    """
    if len(episode_history) == 0:
        return {
            'rolling_sharpe': 0.0,
            'rolling_sortino': 0.0,
            'rolling_mdd': 0.0,
            'rolling_turnover': 0.0,
            'rolling_total_return': 0.0,
            'rolling_volatility': 0.0,
            'rolling_win_rate': 0.0
        }
    
    # Get recent episodes
    recent_episodes = episode_history[-window:]
    
    # Extract metrics from each episode
    all_returns = []
    all_values = []
    all_turnovers = []
    
    for ep in recent_episodes:
        if 'returns' in ep:
            all_returns.extend(ep['returns'])
        if 'portfolio_values' in ep:
            if len(all_values) == 0:
                all_values = ep['portfolio_values'].copy()
            else:
                # Chain portfolio values across episodes
                multiplier = all_values[-1] / ep['portfolio_values'][0]
                scaled_values = [v * multiplier for v in ep['portfolio_values'][1:]]
                all_values.extend(scaled_values)
        if 'turnover' in ep:
            all_turnovers.append(ep['turnover'])
    
    # Calculate rolling metrics
    rolling_metrics = {}
    
    if len(all_returns) > 0:
        rolling_metrics['rolling_sharpe'] = calculate_sharpe_ratio(np.array(all_returns))
        rolling_metrics['rolling_sortino'] = calculate_sortino_ratio(np.array(all_returns))
        rolling_metrics['rolling_volatility'] = calculate_volatility(np.array(all_returns))
        rolling_metrics['rolling_win_rate'] = calculate_win_rate(np.array(all_returns))
    else:
        rolling_metrics['rolling_sharpe'] = 0.0
        rolling_metrics['rolling_sortino'] = 0.0
        rolling_metrics['rolling_volatility'] = 0.0
        rolling_metrics['rolling_win_rate'] = 0.0
    
    if len(all_values) > 0:
        rolling_metrics['rolling_mdd'] = calculate_maximum_drawdown(np.array(all_values))
        rolling_metrics['rolling_total_return'] = calculate_total_return(np.array(all_values))
    else:
        rolling_metrics['rolling_mdd'] = 0.0
        rolling_metrics['rolling_total_return'] = 0.0
    
    if len(all_turnovers) > 0:
        rolling_metrics['rolling_turnover'] = np.mean(all_turnovers)
    else:
        rolling_metrics['rolling_turnover'] = 0.0
    
    return rolling_metrics


# ============================================================================
# TESTING AND VALIDATION FUNCTIONS
# ============================================================================

def test_utility_function():
    """Test the asymmetric sigmoid utility and compare with legacy Gaussian."""
    print("\n=== Testing Asymmetric Sigmoid Utility Function ===")
    
    # Test parameters (Sharpe ratio example)
    mu = 1.0
    k_minus = 4.0   # Steep penalty below target
    k_plus = 1.0    # Gentle saturation above target
    a = -2.0
    b = 3.0
    
    # Legacy Gaussian params for comparison
    sigma_sq_minus = 0.09
    sigma_sq_plus = 0.25
    
    test_values = [-0.5, 0.0, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5]
    
    print(f"  {'x':>6s}  {'Sigmoid':>10s}  {'Gaussian':>10s}  {'Improvement':>12s}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*12}")
    for x in test_values:
        sigmoid = asymmetric_sigmoid_utility(x, mu, k_minus, k_plus, a, b, direction='increasing')
        gaussian = skewed_utility_function(x, mu, sigma_sq_minus, sigma_sq_plus, a, b)
        ratio = sigmoid / max(gaussian, 1e-6)
        print(f"  {x:6.2f}  {sigmoid:10.4f}  {gaussian:10.4f}  {ratio:10.1f}×")
    
    # Test MDD as increasing metric (stored as negative: less-negative = better)
    print("\n  --- MDD (stored as negative, 'increasing': less drawdown = better) ---")
    mu_mdd = -0.15
    k_minus_mdd = 5.0  # steep penalty below target (deeper drawdown)
    k_plus_mdd = 1.0   # gentle above target (less drawdown, diminishing returns)
    mdd_values = [-0.05, -0.10, -0.15, -0.20, -0.25]
    for x in mdd_values:
        u = asymmetric_sigmoid_utility(x, mu_mdd, k_minus_mdd, k_plus_mdd, -0.30, 0.0, direction='increasing')
        print(f"  MDD={x:6.2f} -> Utility={u:.4f}")


def test_tape_score():
    """Test TAPE score calculation with example metrics."""
    print("\n=== Testing TAPE Score Calculation ===")
    
    # Import config to get profile
    try:
        from .config import PROFILE_BALANCED_GROWTH
    except ImportError:
        from config import PROFILE_BALANCED_GROWTH
    
    # Example metrics
    metrics = {
        'sharpe_ratio': 1.2,
        'sortino_ratio': 1.5,
        'max_drawdown': 0.12,
        'turnover': 3.5,
        'skewness': 0.05
    }
    
    tape_score = calculate_tape_score(metrics, PROFILE_BALANCED_GROWTH)
    print(f"  Metrics: {metrics}")
    print(f"  TAPE Score: {tape_score:.4f}")


if __name__ == "__main__":
    # Run tests
    test_utility_function()
    test_tape_score()
    
    print("\n=== Reward Utils Module Tests Complete ===")
