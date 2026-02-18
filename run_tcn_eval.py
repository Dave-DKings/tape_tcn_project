#!/usr/bin/env python
"""
Evaluate TCN RL Agent on Test Set and Save Daily Portfolio Values

This script loads the trained TCN model and runs a deterministic evaluation
on the test set to generate daily portfolio values for benchmark comparison.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

# Configure TensorFlow to use CPU only (avoid CUDA issues)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

# Import project modules
from src.data_utils import DataProcessor
from src.agents.ppo_agent_tf import PPOAgentTF
from src.environment_tape_rl import PortfolioEnvTAPE
from src.config import get_active_config
from src.reproducibility_helper import set_all_seeds
from src.notebook_helpers.tcn_phase1 import split_dataset_by_date

# Configuration
CHECKPOINT_PATH = 'tcn_results/exp6_tape_ep167'
CHECKPOINT_EPISODE = 167
RANDOM_SEED = 42


def run_tcn_evaluation(checkpoint_path=None, checkpoint_episode=None):
    """Run TCN agent evaluation and save daily portfolio values."""
    
    # Use defaults if not provided
    checkpoint_path = checkpoint_path or CHECKPOINT_PATH
    checkpoint_episode = checkpoint_episode or CHECKPOINT_EPISODE
    
    print("="*80)
    print("TCN RL AGENT EVALUATION FOR BENCHMARK COMPARISON")
    print("="*80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Episode: {checkpoint_episode}")
    print(f"Random Seed: {RANDOM_SEED}\n")
    
    # Set reproducibility
    set_all_seeds(RANDOM_SEED, deterministic=True)
    
    # Load configuration
    print("Loading configuration...")
    config = get_active_config("phase1")
    
    # Ensure TCN architecture is set
    config['agent_params']['actor_critic_type'] = 'TCN'
    config['agent_params']['use_covariance'] = True
    config['agent_params']['evaluation_mode'] = 'mode'  # Deterministic
    
    print(f"✅ Config loaded (architecture: {config['agent_params']['actor_critic_type']})\n")
    
    # Load and process data
    # prepare_features_phase1 handles loading, splitting, and normalization
    # It returns (dataframe, scalers) not the phase1_data dict structure
    # We need to use the notebook helper to get the proper structure
    dp = DataProcessor(config)
    
    print("Loading and processing data...")
    df_processed, scalers = dp.prepare_features_phase1()
    
    # Split into train/test using the configured split date
    split_date = config.get('TRAIN_TEST_SPLIT_DATE', '2019-12-31')
    phase1_data = split_dataset_by_date(df_processed, split_date)
    
    print(f"✅ Data loaded:")
    print(f"   Train shape: {phase1_data['train']['features'].shape}")
    print(f"   Test shape: {phase1_data['test']['features'].shape}\n")
    
    # Create test environment
    print("Creating test environment...")
    test_env = PortfolioEnvTAPE(
        phase1_data['test']['features'],
        phase1_data['test']['prices'],
        config,
        phase='test',
        initial_capital=config['initial_capital']
    )
    print(f"✅ Environment created\n")
    
    # Create agent
    print("Initializing TCN agent...")
    agent = PPOAgentTF(
        state_dim=test_env.observation_space.shape,
        action_dim=test_env.action_space.shape[0],
        config_dict=config['agent_params']
    )
    print(f"✅ Agent initialized\n")
    
    # Load checkpoint weights
    print(f"Loading checkpoint weights from {checkpoint_path}...")
    try:
        agent.load_weights(checkpoint_path)
        print(f"✅ Weights loaded successfully\n")
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        print(f"   Checkpoint path: {checkpoint_path}")
        return
    
    # Run deterministic evaluation
    print("Running deterministic evaluation on test set...")
    obs, _ = test_env.reset()
    done = False
    truncated = False
    step_count = 0
    
    # Track portfolio values for each day
    portfolio_values = []
    dates = []
    
    while not (done or truncated):
        # Get deterministic action (mode of Dirichlet)
        action, _ = agent.act(
            np.expand_dims(obs, axis=0),
            training=False
        )
        action = action[0]  # Remove batch dimension
        
        # Step environment
        obs, reward, done, truncated, info = test_env.step(action)
        
        # Record portfolio value and date
        portfolio_values.append(info['portfolio_value'])
        dates.append(info.get('date', phase1_data['test']['prices'].index[step_count]))
        
        step_count += 1
        
        if step_count % 100 == 0:
            print(f"   Step {step_count}/{len(phase1_data['test']['prices'])}")
    
    print(f"✅ Evaluation complete ({step_count} steps)\n")
    
    # Create results DataFrame
    result_df = pd.DataFrame({
        'Date': dates,
        'Portfolio Value': portfolio_values
    })
    
    # Ensure dates are datetime
    result_df['Date'] = pd.to_datetime(result_df['Date'])
    
    # Save results
    output_dir = 'tcn_results/benchmarks'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'tcn_rl_agent_daily_values.csv')
    
    result_df.to_csv(output_path, index=False)
    
    # Print summary
    print("="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"Start Date: {dates[0]}")
    print(f"End Date: {dates[-1]}")
    print(f"Days: {len(dates)}")
    print(f"Initial Value: ${config['initial_capital']:,.2f}")
    print(f"Final Value: ${portfolio_values[-1]:,.2f}")
    total_return = (portfolio_values[-1] / config['initial_capital'] - 1) * 100
    print(f"Total Return: {total_return:.2f}%")
    print(f"\n✅ Results saved to: {output_path}")
    print("="*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate TCN RL Agent')
    parser.add_argument('--checkpoint', type=str, default=CHECKPOINT_PATH,
                        help='Path to model checkpoint')
    parser.add_argument('--episode', type=int, default=CHECKPOINT_EPISODE,
                        help='Checkpoint episode number')
    
    args = parser.parse_args()
    
    run_tcn_evaluation(
        checkpoint_path=args.checkpoint,
        checkpoint_episode=args.episode
    )
