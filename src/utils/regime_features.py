"""
Regime Feature Calculation Utilities

Provides market regime detection features for gated policy execution:
- Volatility (rolling standard deviation)
- Drawdown (current from peak)
- Trend (moving average differential)
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple


def compute_rolling_volatility(
    returns: np.ndarray,
    window: int = 20,
    annualize: bool = True
) -> float:
    """
    Compute rolling volatility from recent returns.
    
    Args:
        returns: Array of returns (most recent at end)
        window: Rolling window size (default: 20 days)
        annualize: If True, annualize using sqrt(252)
        
    Returns:
        Current volatility estimate
    """
    if len(returns) < window:
        window = max(2, len(returns))
    
    recent_returns = returns[-window:]
    vol = np.std(recent_returns, ddof=1)
    
    if annualize:
        vol *= np.sqrt(252)
    
    return float(vol)


def compute_drawdown(portfolio_values: np.ndarray) -> float:
    """
    Compute current drawdown from peak.
    
    Drawdown = 1 - (current_value / peak_value)
    
    Args:
        portfolio_values: Array of portfolio values
        
    Returns:
        Current drawdown (0.0 to 1.0)
    """
    if len(portfolio_values) == 0:
        return 0.0
    
    peak = np.maximum.accumulate(portfolio_values)[-1]
    current = portfolio_values[-1]
    
    if peak > 0:
        dd = 1.0 - (current / peak)
        return float(np.clip(dd, 0.0, 1.0))
    else:
        return 0.0


def compute_trend(
    prices: np.ndarray,
    short_window: int = 20,
    long_window: int = 100
) -> float:
    """
    Compute trend indicator as difference between short and long MA.
    
    Positive trend → short MA > long MA (uptrend)
    Negative trend → short MA < long MA (downtrend)
    
    Args:
        prices: Array of prices (or portfolio values)
        short_window: Short moving average window
        long_window: Long moving average window
        
    Returns:
        Trend score (positive = uptrend, negative = downtrend)
    """
    if len(prices) < long_window:
        # Not enough data, use what we have
        short_window = min(short_window, len(prices))
        long_window = min(long_window, len(prices))
    
    if len(prices) < 2:
        return 0.0
    
    ma_short = np.mean(prices[-short_window:])
    ma_long = np.mean(prices[-long_window:])
    
    # Normalize by current price to make scale-invariant
    current_price = prices[-1]
    if current_price > 0:
        trend = (ma_short - ma_long) / current_price
    else:
        trend = 0.0
    
    return float(trend)


def compute_regime(
    vol: float,
    drawdown: float,
    trend: float,
    vol_hi: float = 0.20,
    dd_hi: float = 0.15,
    trend_hi: float = 0.0
) -> str:
    """
    Determine market regime based on volatility, drawdown, and trend.
    
    Regime logic:
    - risk_off: High volatility OR high drawdown
    - risk_on: Positive trend AND normal volatility
    - neutral: Otherwise
    
    Args:
        vol: Current volatility (annualized)
        drawdown: Current drawdown (0 to 1)
        trend: Current trend score
        vol_hi: High volatility threshold
        dd_hi: High drawdown threshold
        trend_hi: Trend threshold (usually 0.0)
        
    Returns:
        Regime name: 'risk_off', 'neutral', or 'risk_on'
    """
    if (drawdown > dd_hi) or (vol > vol_hi):
        return 'risk_off'
    elif (trend > trend_hi) and (vol <= vol_hi):
        return 'risk_on'
    else:
        return 'neutral'


def compute_all_regime_signals(
    portfolio_values: np.ndarray,
    returns: np.ndarray,
    vol_window: int = 20,
    short_trend: int = 20,
    long_trend: int = 100
) -> Tuple[float, float, float]:
    """
    Compute all regime signals in one call.
    
    Convenience function that computes volatility, drawdown, and trend.
    
    Args:
        portfolio_values: Portfolio value history
        returns: Return history
        vol_window: Window for volatility calculation
        short_trend: Short MA window for trend
        long_trend: Long MA window for trend
        
    Returns:
        Tuple of (volatility, drawdown, trend)
    """
    vol = compute_rolling_volatility(returns, window=vol_window)
    dd = compute_drawdown(portfolio_values)
    trend = compute_trend(portfolio_values, short_window=short_trend, long_window=long_trend)
    
    return vol, dd, trend
