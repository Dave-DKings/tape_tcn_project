"""
Backtesting utilities extracted from `tcn_architecture_analysis.ipynb`.

The functions here match the notebook implementations so cells can simply
`from src.notebook_helpers.backtests import ...` and keep identical outputs.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd


def calculate_metrics_from_trajectory(
    portfolio_values: List[float],
    initial_value: float,
) -> Dict[str, float]:
    """Compute core performance metrics from a portfolio value trajectory."""
    pv_array = np.array(portfolio_values)

    total_return = (pv_array[-1] / initial_value - 1) * 100
    daily_returns = np.diff(pv_array) / pv_array[:-1]

    if len(daily_returns) > 0 and np.std(daily_returns) > 0:
        sharpe_ratio = np.sqrt(252) * np.mean(daily_returns) / np.std(daily_returns)
    else:
        sharpe_ratio = 0.0

    cummax = np.maximum.accumulate(pv_array)
    drawdown = (pv_array - cummax) / cummax * 100
    max_drawdown = float(np.min(drawdown))

    win_rate = (
        np.sum(daily_returns > 0) / len(daily_returns) * 100 if len(daily_returns) > 0 else 0.0
    )
    daily_volatility = float(np.std(daily_returns) * 100) if len(daily_returns) > 0 else 0.0

    return {
        "total_return": float(total_return),
        "sharpe_ratio": float(sharpe_ratio),
        "max_drawdown": max_drawdown,
        "win_rate": float(win_rate),
        "daily_volatility": daily_volatility,
        "num_days": len(pv_array) - 1,
    }


def run_backtest_deterministic(
    model,
    env,
    *,
    env_type: str = "finrl",
    stock_dim: int = 5,
    random_seed: int = 42,
    evaluation_mode: str = 'mode',  # ADDED: Match training eval mode
):
    """
    Execute a single deterministic backtest episode and return trajectory + metrics.
    
    Args:
        model: Trained agent model (must have .predict() method)
        env: Environment to run backtest in
        env_type: Type of environment ("finrl" or other)
        stock_dim: Number of assets (for FinRL environments)
        random_seed: Seed for reproducibility
        evaluation_mode: Dirichlet evaluation strategy ('mean', 'mode', 'mean_plus_noise')
            - 'mode': Recommended - shows true learned policy (sparse allocations)
            - 'mean': More diversified (can differ from training behavior)
            - 'mean_plus_noise': Mean with small noise (default in old code)
    
    Returns:
        tuple: (portfolio_values, metrics)
    """
    if env_type == "finrl":
        obs = env.reset()
        done = False
        actual_env = env.envs[0]
        portfolio_values = [actual_env.asset_memory[-1]]

        while not done:
            # Pass evaluation_mode to ensure consistency with training
            action, _ = model.predict(obs, deterministic=True, evaluation_mode=evaluation_mode)
            obs, reward, done, info = env.step(action)
            done = done[0]

            if not done:
                portfolio_values.append(actual_env.asset_memory[-1])
            else:
                terminal_obs = info[0].get("terminal_observation")
                if terminal_obs is not None:
                    final_value = terminal_obs[0] + np.sum(
                        np.array(terminal_obs[(stock_dim + 1) : (stock_dim * 2 + 1)])
                        * np.array(terminal_obs[1 : (stock_dim + 1)])
                    )
                else:
                    final_value = actual_env.asset_memory[-1]
                portfolio_values.append(final_value)
    else:
        obs, info = env.reset(seed=random_seed)
        done = False
        portfolio_values = [env.portfolio_value]

        while not done:
            # Pass evaluation_mode to ensure consistency with training
            action, _ = model.predict(obs, deterministic=True, evaluation_mode=evaluation_mode)
            obs, reward, done, truncated, info = env.step(action)
            done = done or truncated
            portfolio_values.append(env.portfolio_value)

    initial_value = portfolio_values[0]
    metrics = calculate_metrics_from_trajectory(portfolio_values, initial_value)
    return portfolio_values, metrics


def run_backtest_stochastic(
    model,
    env,
    *,
    num_episodes: int = 100,
    env_type: str = "finrl",
    stock_dim: int = 5,
    random_seed: int = 42,
):
    """
    Monte Carlo evaluation with stochastic policy sampling.
    """
    all_metrics: List[Dict[str, float]] = []
    all_portfolio_values: List[List[float]] = []

    for episode_idx in range(num_episodes):
        if env_type == "finrl":
            obs = env.reset()
            done = False
            actual_env = env.envs[0]
            portfolio_values = [actual_env.asset_memory[-1]]

            while not done:
                action, _ = model.predict(obs, deterministic=False)
                obs, reward, done, info = env.step(action)
                done = done[0]

                if not done:
                    portfolio_values.append(actual_env.asset_memory[-1])
                else:
                    terminal_obs = info[0].get("terminal_observation")
                    if terminal_obs is not None:
                        final_value = terminal_obs[0] + np.sum(
                            np.array(terminal_obs[(stock_dim + 1) : (stock_dim * 2 + 1)])
                            * np.array(terminal_obs[1 : (stock_dim + 1)])
                        )
                    else:
                        final_value = actual_env.asset_memory[-1]
                    portfolio_values.append(final_value)
        else:
            episode_seed = random_seed + episode_idx
            obs, info = env.reset(seed=episode_seed)
            done = False
            portfolio_values = [env.portfolio_value]

            while not done:
                action, _ = model.predict(obs, deterministic=False)
                obs, reward, done, truncated, info = env.step(action)
                done = done or truncated
                portfolio_values.append(env.portfolio_value)

        initial_value = portfolio_values[0]
        metrics = calculate_metrics_from_trajectory(portfolio_values, initial_value)
        all_metrics.append(metrics)
        all_portfolio_values.append(portfolio_values)

    aggregated_metrics = {
        "mean_return": float(np.mean([m["total_return"] for m in all_metrics])),
        "std_return": float(np.std([m["total_return"] for m in all_metrics])),
        "mean_sharpe": float(np.mean([m["sharpe_ratio"] for m in all_metrics])),
        "std_sharpe": float(np.std([m["sharpe_ratio"] for m in all_metrics])),
        "mean_drawdown": float(np.mean([m["max_drawdown"] for m in all_metrics])),
        "mean_win_rate": float(np.mean([m["win_rate"] for m in all_metrics])),
        "num_episodes": num_episodes,
    }

    return all_portfolio_values, aggregated_metrics


def run_backtest_walk_forward(
    model,
    test_df: pd.DataFrame,
    *,
    env_creator_func: Callable[[pd.DataFrame], object],
    num_windows: int = 5,
    window_overlap: float = 0.5,
    env_type: str = "finrl",
    stock_dim: int = 5,
    random_seed: int = 42,
    evaluation_mode: str = 'mode',  # ADDED: Match training eval mode
):
    """
    Walk-forward analysis across overlapping time windows.
    
    Args:
        evaluation_mode: Dirichlet evaluation strategy (passed to run_backtest_deterministic)
    """
    date_col = "date" if env_type == "finrl" else "Date"
    unique_dates = sorted(test_df[date_col].unique())
    total_days = len(unique_dates)

    window_size = int(total_days / (num_windows * (1 - window_overlap) + window_overlap))
    step_size = max(int(window_size * (1 - window_overlap)), 1)

    all_portfolio_values: List[List[float]] = []
    window_metrics: List[Dict[str, float]] = []

    for window_idx in range(num_windows):
        start_idx = window_idx * step_size
        end_idx = start_idx + window_size
        if end_idx > total_days:
            break

        window_dates = unique_dates[start_idx:end_idx]
        window_df = test_df[test_df[date_col].isin(window_dates)].copy()

        env = env_creator_func(window_df)
        # Pass evaluation_mode through to the backtest function
        portfolio_values, metrics = run_backtest_deterministic(
            model,
            env,
            env_type=env_type,
            stock_dim=stock_dim,
            random_seed=random_seed,
            evaluation_mode=evaluation_mode,  # ADDED
        )

        metrics.update(
            {
                "window_start": window_dates[0],
                "window_end": window_dates[-1],
                "window_idx": window_idx,
            }
        )
        all_portfolio_values.append(portfolio_values)
        window_metrics.append(metrics)

    return all_portfolio_values, window_metrics


def describe_backtest_helpers() -> None:
    """Print a concise overview mirroring the notebook cell footer."""
    print("✅ State-of-the-art evaluation framework loaded!")
    print("   Available functions:")
    print("   - calculate_metrics_from_trajectory(): Compute metrics from portfolio trajectory")
    print("   - run_backtest_deterministic(): Single deterministic backtest (standard)")
    print("   - run_backtest_stochastic(): Monte Carlo evaluation with stochastic policy")
    print("   - run_backtest_walk_forward(): Walk-forward analysis across time windows")
    print("   🎲 Pass `random_seed` for custom env reproducibility")
    print("   📌 FinRL envs remain deterministic by design")


__all__ = [
    "calculate_metrics_from_trajectory",
    "run_backtest_deterministic",
    "run_backtest_stochastic",
    "run_backtest_walk_forward",
    "describe_backtest_helpers",
]
