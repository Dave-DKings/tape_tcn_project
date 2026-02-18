"""
Enhanced Evaluation for TCN Model with Multiple Execution Modes

This script evaluates the trained TCN model using:
1. Deterministic mean (baseline)
2. Gated deterministic (regime-aware with smoothing)
3. Controlled stochastic (s=2, 5, 10)

All evaluations use the same test data and transaction costs for fair comparison.
"""

import sys
sys.path.append('c:/Users/Owner/new_project/adaptive_portfolio_rl')

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# Import existing evaluation infrastructure
from src.notebook_helpers.tcn_phase1 import (
    create_experiment6_result_stub,
    prepare_phase1_dataset
)
from src.config import BASE_CONFIG_TCN_PHASE1
from src.policies import (
    dirichlet_mean,
    controlled_stochastic_sample,
    gated_deterministic_step
)
from src.utils.regime_features import compute_all_regime_signals


def evaluate_with_mode(
    agent,
    env,
    mode: str,
    s_value: float = None,
    regime_params: Dict = None,
    num_episodes: int = 1,
    random_seed: int = 42
) -> Dict:
    """
    Evaluate agent with specified execution mode.
    
    Args:
        agent: Trained PPO agent
        env: Test environment
        mode: 'deterministic_mean', 'gated_deterministic', 'controlled_stochastic'
        s_value: Scaling factor for controlled stochastic (required if mode='controlled_stochastic')
        regime_params: Parameters for gated deterministic
        num_episodes: Number of episodes to run
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary with evaluation results
    """
    rng = np.random.default_rng(random_seed)
    
    results = {
        'mode': mode,
        's_value': s_value,
        'episodes': [],
        'alphas': [],
        'weights': [],
        'actions': [],
        'regimes': [] if mode == 'gated_deterministic' else None
    }
    
    for ep in range(num_episodes):
        obs, info = env.reset(seed=random_seed + ep)
        done = False
        step_count = 0
        
        episode_alphas = []
        episode_weights = []
        episode_actions = []
        episode_regimes = []
        
        # Initialize previous weights for smoothing
        w_prev = None
        
        # Regime tracking
        portfolio_values = [env.portfolio_value]
        returns = []
        
        while not done:
            # Get alpha from policy (ensure we're using internal alpha output)
            # The agent's actor network outputs alphas directly
            obs_tensor = obs.reshape(1, -1)
            alphas = agent.actor(obs_tensor).numpy().flatten()
            
            # Apply execution mode
            if mode == 'deterministic_mean':
                weights = dirichlet_mean(alphas)
                regime = None
                
            elif mode == 'gated_deterministic':
                if w_prev is None:
                    w_prev = dirichlet_mean(alphas)
                
                # Compute regime signals
                if len(returns) > 0:
                    vol, dd, trend = compute_all_regime_signals(
                        np.array(portfolio_values),
                        np.array(returns)
                    )
                else:
                    vol, dd, trend = 0.0, 0.0, 0.0
                
                weights, regime = gated_deterministic_step(
                    alphas, w_prev, (vol, dd, trend), regime_params
                )
                w_prev = weights.copy()
                episode_regimes.append(regime)
                
            elif mode == 'controlled_stochastic':
                weights = controlled_stochastic_sample(alphas, s_value, rng)
                regime = None
                
            else:
                raise ValueError(f"Unknown mode: {mode}")
            
            # Store data
            episode_alphas.append(alphas)
            episode_weights.append(weights)
            episode_actions.append(weights)  # Actions are the final weights
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(weights)
            done = terminated or truncated
            step_count += 1
            
            # Track for regime calculation
            portfolio_values.append(env.portfolio_value)
            if len(portfolio_values) > 1:
                ret = (portfolio_values[-1] / portfolio_values[-2]) - 1.0
                returns.append(ret)
        
        # Store episode results
        results['episodes'].append({
            'return': info.get('total_return_pct', 0.0) / 100.0,
            'sharpe': info.get('sharpe_ratio', 0.0),
            'max_drawdown': info.get('max_drawdown', 0.0),
            'volatility': info.get('volatility', 0.0),
            'win_rate': info.get('win_rate', 0.0),
            'final_value': info.get('final_balance', 0.0),
            'steps': step_count,
            'dates': env.dates.copy() if hasattr(env, 'dates') else None
        })
        results['alphas'].append(np.array(episode_alphas))
        results['weights'].append(np.array(episode_weights))
        results['actions'].append(np.array(episode_actions))
        
        if mode == 'gated_deterministic':
            results['regimes'].append(episode_regimes)
    
    return results


def calculate_turnover_from_weights(weights: np.ndarray) -> float:
    """Calculate average daily turnover from weight history."""
    if len(weights) <= 1:
        return 0.0
    
    turnovers = []
    for i in range(1, len(weights)):
        turnover = np.sum(np.abs(weights[i] - weights[i-1]))
        turnovers.append(turnover)
    
    return np.mean(turnovers)


def run_comprehensive_evaluation(
    checkpoint_path: str = "tcn_results/exp6_tape_ep136_actor.weights.h5",
    output_dir: str = "eval_results"
):
    """
    Run comprehensive evaluation comparing all execution modes.
    
    Args:
        checkpoint_path: Path to trained actor weights
        output_dir: Directory to save results
    """
    print("=" * 80)
    print("COMPREHENSIVE TCN EVALUATION - Multiple Execution Modes")
    print("=" * 80)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Load data and create environment
    print("\n📊 Loading data...")
    config = BASE_CONFIG_TCN_PHASE1.copy()
    dataset = prepare_phase1_dataset(config, force_download=False)
    
    # Load trained agent
    print(f"\n🤖 Loading trained model from: {checkpoint_path}")
    exp_stub = create_experiment6_result_stub(
        random_seed=42,
        checkpoint_path_override=checkpoint_path
    )
    agent = exp_stub.agent
    
    # Create test environment (deterministic)
    from src.environment_tape_rl import PortfolioEnvTAPE
    env_test = PortfolioEnvTAPE(
        config=config,
        data_processor=dataset.data_processor,
        processed_data=dataset.test_df,
        mode='test',
        action_normalization='none',  # We'll handle normalization ourselves
        random_start=False
    )
    
    # Define regime parameters for gated deterministic
    regime_params = {
        'vol_hi': 0.20,  # High volatility threshold (20% annualized)
        'dd_hi': 0.15,   # High drawdown threshold (15%)
        'trend_hi': 0.0,  # Neutral trend threshold
        'base_method': 'mean',
        'use_cash_tilt': True,
        'cash_idx': len(env_test.current_weights) - 1,
        'risk_off_cash': 0.30,
        'eta_risk_off': 0.05,
        'eta_neutral': 0.10,
        'eta_risk_on': 0.20
    }
    
    # Run evaluations
    all_results = {}
    
    print("\n" + "=" * 80)
    print("EVALUATION 1: Deterministic Mean (Baseline)")
    print("=" * 80)
    all_results['deterministic_mean'] = evaluate_with_mode(
        agent, env_test, 'deterministic_mean'
    )
    
    print("\n" + "=" * 80)
    print("EVALUATION 2: Gated Deterministic (Regime-Aware)")
    print("=" * 80)
    all_results['gated_deterministic'] = evaluate_with_mode(
        agent, env_test, 'gated_deterministic', regime_params=regime_params
    )
    
    print("\n" + "=" * 80)
    print("EVALUATION 3: Controlled Stochastic (s=2)")
    print("=" * 80)
    all_results['controlled_s2'] = evaluate_with_mode(
        agent, env_test, 'controlled_stochastic', s_value=2.0
    )
    
    print("\n" + "=" * 80)
    print("EVALUATION 4: Controlled Stochastic (s=5)")
    print("=" * 80)
    all_results['controlled_s5'] = evaluate_with_mode(
        agent, env_test, 'controlled_stochastic', s_value=5.0
    )
    
    print("\n" + "=" * 80)
    print("EVALUATION 5: Controlled Stochastic (s=10)")
    print("=" * 80)
    all_results['controlled_s10'] = evaluate_with_mode(
        agent, env_test, 'controlled_stochastic', s_value=10.0
    )
    
    # Compile comparison table
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    
    comparison_data = []
    for name, results in all_results.items():
        ep = results['episodes'][0]
        weights = results['weights'][0]
        turnover = calculate_turnover_from_weights(weights)
        
        comparison_data.append({
            'Mode': name,
            'Return (%)': ep['return'] * 100,
            'Sharpe Ratio': ep['sharpe'],
            'Max DD (%)': ep['max_drawdown'] * 100,
            'Volatility (%)': ep['volatility'] * 100,
            'Win Rate (%)': ep['win_rate'] * 100,
            'Avg Turnover (%)': turnover * 100
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    print("\n", df_comparison.to_string(index=False))
    
    # Save results
    csv_path = output_path / "execution_mode_comparison.csv"
    df_comparison.to_csv(csv_path, index=False)
    print(f"\n✅ Results saved to: {csv_path}")
    
    # Save detailed results
    import pickle
    pickle_path = output_path / "detailed_results.pkl"
    with open(pickle_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"✅ Detailed results saved to: {pickle_path}")
    
    return all_results, df_comparison


if __name__ == "__main__":
    results, comparison = run_comprehensive_evaluation()
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
