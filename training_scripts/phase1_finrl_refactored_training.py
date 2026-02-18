#!/usr/bin/env python3
"""
Phase 1 Training with FinRL-Refactored Environment

This script uses the validated FinRL-refactored environment that achieved:
- 54.22% return on test set (67% of benchmark)
- 0.806 Sharpe ratio (94% of benchmark)
- Complete episode execution (no premature termination)
- Stable training (no NaN errors)

Key improvements over old environment:
- Simple reward = portfolio return * 100
- No training wheels or milestone rewards
- No premature termination from balance thresholds
- Direct return calculation from Close prices
- Proven FinRL design patterns

Usage:
    python phase1_finrl_refactored_training.py [--episodes EPISODES] [--timesteps TIMESTEPS]
"""

import sys
import os
sys.path.insert(0, 'src')

import numpy as np
import json
import argparse
from datetime import datetime
from typing import Dict, List
import logging

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from config import get_active_config
from data_utils import DataProcessor
from environment_tape_rl import PortfolioEnvTAPE
from reproducibility_helper import set_all_seeds

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class ProgressCallback(BaseCallback):
    """
    Custom callback for tracking training progress.
    """
    
    def __init__(self, check_freq: int = 2048, verbose: int = 1):
        super(ProgressCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        # Check if we're at a rollout boundary
        if self.n_calls % self.check_freq == 0:
            # Log progress
            if self.verbose > 0:
                logger.info(f"Timesteps: {self.num_timesteps:,} | "
                           f"Episodes: {len(self.episode_rewards)}")
        
        return True


def print_header():
    """Print training header."""
    print("=" * 80)
    print("PHASE 1 TRAINING - FINRL-REFACTORED ENVIRONMENT")
    print("=" * 80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("✅ Using validated FinRL-refactored environment:")
    print("   - Reward: Portfolio return * 100")
    print("   - Returns: Calculated from Close prices")
    print("   - Termination: Data exhausted only")
    print("   - Action norm: Softmax")
    print("   - NO training wheels, NO milestone rewards")
    print()


def train_agent(config: Dict, args) -> Dict:
    """
    Train the PPO agent using FinRL-refactored environment.
    
    Args:
        config: Configuration dictionary
        args: Command line arguments
        
    Returns:
        Training results dictionary
    """
    print("=" * 80)
    print("TRAINING SETUP")
    print("=" * 80)
    print()
    
    # Load and prepare data
    print("📊 Loading and preparing data...")
    processor = DataProcessor(config)
    processed_df, scalers = processor.prepare_features_phase1()
    
    print(f"✅ Data loaded: {len(processed_df):,} rows")
    print(f"   Date range: {processed_df['Date'].min()} to {processed_df['Date'].max()}")
    print()
    
    # Split data (80/20 for train/validation)
    print("✂️  Splitting data...")
    unique_dates = sorted(processed_df['Date'].unique())
    train_date_count = int(len(unique_dates) * 0.8)
    
    train_dates = unique_dates[:train_date_count]
    val_dates = unique_dates[train_date_count:]
    
    train_df = processed_df[processed_df['Date'].isin(train_dates)].copy()
    val_df = processed_df[processed_df['Date'].isin(val_dates)].copy()
    
    print(f"✅ Training: {len(train_dates):,} dates ({len(train_df):,} rows)")
    print(f"✅ Validation: {len(val_dates):,} dates ({len(val_df):,} rows)")
    print()
    
    # Create environment
    print("🏗️  Creating FinRL-refactored environment...")
    env_train = PortfolioEnvTAPE(
        config=config,
        data_processor=processor,
        processed_data=train_df,
        mode='train',
        start_idx=0,
        end_idx=None,
        episode_length_limit=config.get('environment_params', {}).get('max_steps_per_episode')
    )
    
    # Wrap for Stable-Baselines3
    env_train = DummyVecEnv([lambda: env_train])
    print("✅ Environment created")
    print()
    
    # Create PPO agent
    n_steps = 2048
    ent_coef = 0.005
    learning_rate = 0.00005
    batch_size = 128
    gamma = 0.99

    print("🤖 Creating PPO agent...")
    print("   Hyperparameters:")
    print(f"   - n_steps: {n_steps}")
    print(f"   - ent_coef: {ent_coef}")
    print(f"   - learning_rate: {learning_rate}")
    print(f"   - batch_size: {batch_size}")
    print(f"   - gamma: {gamma}")
    print()
    
    model = PPO(
        "MlpPolicy",
        env_train,
        seed=args.seed,
        n_steps=n_steps,
        ent_coef=ent_coef,
        learning_rate=learning_rate,
        batch_size=batch_size,
        gamma=gamma,
        verbose=1,
        tensorboard_log="./phase1_finrl_refactored_tensorboard/"
    )
    
    # Train
    print("=" * 80)
    print("TRAINING")
    print("=" * 80)
    print()
    print(f"🚀 Training for {args.timesteps:,} timesteps...")
    print()
    
    callback = ProgressCallback(check_freq=2048, verbose=1)
    
    training_start = datetime.now()
    model.learn(total_timesteps=args.timesteps, callback=callback)
    training_end = datetime.now()
    
    training_duration = (training_end - training_start).total_seconds()
    
    print()
    print(f"✅ Training complete!")
    print(f"   Duration: {training_duration / 60:.1f} minutes")
    print()
    
    # Save model
    model_path = f"./phase1_finrl_refactored_model_{args.timesteps // 1000}k"
    model.save(model_path)
    print(f"💾 Model saved to: {model_path}")
    print()
    
    # Validate on validation set
    print("=" * 80)
    print("VALIDATION")
    print("=" * 80)
    print()
    print("📈 Validating on validation set...")
    
    env_val = PortfolioEnvTAPE(
        config=config,
        data_processor=processor,
        processed_data=val_df,
        mode='test',
        start_idx=0,
        end_idx=None,
        episode_length_limit=None
    )
    
    obs = env_val.reset(seed=args.seed)[0]
    done = False
    step_count = 0
    portfolio_values = [env_val.initial_balance]
    
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env_val.step(action)
        
        portfolio_values.append(info.get('portfolio_value', env_val.portfolio_value))
        step_count += 1
        
        if done or truncated:
            break
    
    print(f"✅ Validation complete ({step_count} steps)")
    print()
    
    # Calculate metrics
    final_value = env_val.portfolio_value
    initial_value = env_val.initial_balance
    total_return = (final_value - initial_value) / initial_value
    annualized_return = (1 + total_return) ** (252 / len(val_dates)) - 1
    
    # Calculate returns
    returns = []
    for i in range(1, len(portfolio_values)):
        ret = (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]
        returns.append(ret)
    
    returns_array = np.array(returns)
    sharpe_ratio = (252 ** 0.5) * returns_array.mean() / (returns_array.std() + 1e-10)
    
    # Sortino ratio
    downside_returns = returns_array[returns_array < 0]
    sortino_ratio = (252 ** 0.5) * returns_array.mean() / (downside_returns.std() + 1e-10) if len(downside_returns) > 0 else 0
    
    # Max drawdown
    cumulative = np.array(portfolio_values)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Volatility
    volatility = returns_array.std() * (252 ** 0.5)
    
    # Win rate
    wins = (returns_array > 0).sum()
    win_rate = wins / len(returns_array) if len(returns_array) > 0 else 0
    
    # Print results
    print("=" * 80)
    print("VALIDATION RESULTS")
    print("=" * 80)
    print()
    print("📊 Performance Metrics:")
    print(f"   Initial Capital:        ${initial_value:,.2f}")
    print(f"   Final Portfolio Value:  ${final_value:,.2f}")
    print(f"   Total Return:           {total_return * 100:.2f}%")
    print(f"   Annualized Return:      {annualized_return * 100:.2f}%")
    print()
    print("📈 Risk-Adjusted Metrics:")
    print(f"   Sharpe Ratio:           {sharpe_ratio:.3f}")
    print(f"   Sortino Ratio:          {sortino_ratio:.3f}")
    print(f"   Maximum Drawdown:       {max_drawdown * 100:.2f}%")
    print(f"   Volatility (Annual):    {volatility * 100:.2f}%")
    print()
    print("🎯 Trading Statistics:")
    print(f"   Win Rate:               {win_rate * 100:.2f}%")
    print(f"   Total Trading Days:     {len(val_dates)}")
    print(f"   Average Daily Return:   {returns_array.mean() * 100:.4f}%")
    print()
    
    # Save results
    results = {
        "configuration": {
            "environment": "FinRL-Refactored",
            "training_timesteps": args.timesteps,
            "training_duration_seconds": training_duration,
            "initial_capital": float(initial_value),
            "assets": config['ASSET_TICKERS']
        },
        "validation_performance": {
            "final_portfolio_value": float(final_value),
            "total_return": float(total_return),
            "annualized_return": float(annualized_return),
            "sharpe_ratio": float(sharpe_ratio),
            "sortino_ratio": float(sortino_ratio),
            "max_drawdown": float(max_drawdown),
            "volatility": float(volatility),
            "win_rate": float(win_rate),
            "trading_days": len(val_dates)
        }
    }
    
    results_path = f'phase1_finrl_refactored_results_{args.timesteps // 1000}k.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Results saved to: {results_path}")
    print()
    
    return results


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Phase 1 Training with FinRL-Refactored Environment')
    parser.add_argument('--timesteps', type=int, default=100000,
                       help='Total training timesteps (default: 100000)')
    parser.add_argument('--config', type=str, default='phase1',
                       help='Config name to use (default: phase1)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Set all random seeds for reproducibility
    set_all_seeds(args.seed)
    logger.info(f"🎲 Random seed set to: {args.seed}")
    
    print_header()
    
    # Load config
    config = get_active_config(args.config)
    
    print(f"📋 Configuration: {args.config}")
    print(f"   Assets: {', '.join(config['ASSET_TICKERS'])}")
    print(f"   Initial Balance: ${config['INITIAL_BALANCE']:,.2f}")
    print(f"   Transaction Cost: {config['TRANSACTION_COST_RATE'] * 100:.2f}%")
    print(f"   Training Timesteps: {args.timesteps:,}")
    print(f"   Random Seed: {args.seed}")
    print()
    
    # Train
    results = train_agent(config, args)
    
    # Final summary
    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print()
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("✅ Summary:")
    val_perf = results['validation_performance']
    print(f"   Total Return:     {val_perf['total_return'] * 100:.2f}%")
    print(f"   Sharpe Ratio:     {val_perf['sharpe_ratio']:.3f}")
    print(f"   Max Drawdown:     {val_perf['max_drawdown'] * 100:.2f}%")
    print()
    
    # Comparison with benchmark
    benchmark_return = 0.8098  # From finrl_simple_benchmark
    pct_of_benchmark = (val_perf['total_return'] / benchmark_return) * 100
    
    print("📊 vs Benchmark (80.98% return):")
    print(f"   Achievement:      {pct_of_benchmark:.1f}% of benchmark")
    print()
    
    if pct_of_benchmark >= 80:
        print("🎉 EXCELLENT: Achieves ≥80% of benchmark!")
    elif pct_of_benchmark >= 65:
        print("✅ SUCCESS: Achieves ≥65% of benchmark!")
    else:
        print("⚠️  Below target. Consider more training or hyperparameter tuning.")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
