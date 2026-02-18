"""
Notebook Helper: Alpha Diagnostics and Evaluation Tools

Add this file to your notebook imports for easy alpha monitoring during
training and evaluation.

Usage:
    from src.utils.notebook_helpers import (
        diagnose_alpha_distribution,
        log_alpha_stats,
        evaluate_with_diagnostics
    )
"""

import tensorflow as tf
import numpy as np
import logging

logger = logging.getLogger(__name__)


def diagnose_alpha_distribution(agent, state, verbose=True):
    """
    Check if TCN is learning to discriminate between assets.
    
    Args:
        agent: PPOAgentTF instance
        state: Current state (numpy array)
        verbose: If True, print detailed diagnostics
    
    Returns:
        dict: Dictionary with alpha statistics and implied weights
    """
    # Prepare state
    if agent.is_sequential:
        seq = agent._build_sequence(state)
        state_tensor = tf.constant([seq], dtype=tf.float32)
    else:
        state_tensor = tf.constant([state], dtype=tf.float32)
    
    # Get alpha parameters
    alpha = agent.actor(state_tensor, training=False)[0].numpy()
    
    # Calculate statistics
    stats = {
        'alpha_values': alpha,
        'alpha_mean': alpha.mean(),
        'alpha_std': alpha.std(),
        'alpha_min': alpha.min(),
        'alpha_max': alpha.max(),
        'alpha_ratio': alpha.max() / alpha.min() if alpha.min() > 0 else np.inf,
        'implied_weights': alpha / alpha.sum()
    }
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"📊 ALPHA DISTRIBUTION DIAGNOSTICS")
        print(f"{'='*80}")
        print(f"   Alpha values: {alpha}")
        print(f"   Mean: {stats['alpha_mean']:.3f}")
        print(f"   Std: {stats['alpha_std']:.3f}")
        print(f"   Min: {stats['alpha_min']:.3f}, Max: {stats['alpha_max']:.3f}")
        print(f"   Ratio (max/min): {stats['alpha_ratio']:.2f}x")
        
        # Interpret
        if stats['alpha_std'] < 0.3:
            print(f"\n   ⚠️ WARNING: Alphas too uniform! TCN not learning.")
            print(f"   → Check if gradients are flowing back to actor")
            print(f"   → Consider deeper TCN or lower entropy coefficient")
        elif stats['alpha_std'] < 1.0:
            print(f"\n   ⚡ Moderate differentiation. May need more training.")
        else:
            print(f"\n   ✅ Good differentiation! TCN learning asset preferences.")
        
        # Display implied weights
        print(f"\n   Implied mean weights: {stats['implied_weights']}")
        print(f"   Weight std: {stats['implied_weights'].std():.3f}")
        
        if stats['implied_weights'].std() < 0.05:
            print(f"   ⚠️ Near-uniform weights despite varied alphas!")
        
        print(f"{'='*80}\n")
    
    return stats


def log_alpha_stats(stats, update_num, logger=None):
    """
    Log alpha statistics during training.
    
    Args:
        stats: Dictionary returned from agent.update()
        update_num: Current update number
        logger: Logger instance (uses print if None)
    """
    alpha_mean = stats.get('alpha_mean', 0)
    alpha_std = stats.get('alpha_std', 0)
    alpha_min = stats.get('alpha_min', 0)
    alpha_max = stats.get('alpha_max', 0)
    
    msg = (
        f"Update {update_num:4d} | "
        f"Alpha - mean:{alpha_mean:.2f} "
        f"std:{alpha_std:.2f} "
        f"range:[{alpha_min:.2f}, {alpha_max:.2f}]"
    )
    
    if logger:
        logger.info(msg)
    else:
        print(msg)
    
    # Warning if alpha stuck
    if update_num > 500 and alpha_std < 0.3:
        warning = (
            "⚠️ Alpha std still < 0.3 after 500 updates. "
            "TCN may not be learning. Consider:"
            "\n  1. Checking PPO ratio (should be 0.95-1.05)"
            "\n  2. Lowering entropy further (0.005)"
            "\n  3. Increasing batch size (128)"
        )
        if logger:
            logger.warning(warning)
        else:
            print(f"\n{warning}\n")


def evaluate_with_diagnostics(env, agent, num_episodes=1, evaluation_mode='mode'):
    """
    Run evaluation episodes with comprehensive diagnostics.
    
    Args:
        env: Environment instance
        agent: PPOAgentTF instance  
        num_episodes: Number of episodes to run
        evaluation_mode: 'mode', 'mean', or 'mean_plus_noise'
    
    Returns:
        dict: Comprehensive evaluation results
    """
    results = {
        'episode_returns': [],
        'episode_lengths': [],
        'sharpe_ratios': [],
        'max_drawdowns': [],
        'alpha_stats_history': [],
        'weight_stats_history': []
    }
    
    for ep in range(num_episodes):
        state, _ = env.reset()
        episode_return = 0
        episode_length = 0
        episode_alphas = []
        episode_weights = []
        done = False
        
        while not done:
            # Get alpha diagnostics
            alpha_stats = diagnose_alpha_distribution(agent, state, verbose=False)
            episode_alphas.append(alpha_stats['alpha_values'])
            
            # Get action
            action, log_prob, value = agent.get_action_and_value(
                state, 
                deterministic=True,
                evaluation_mode=evaluation_mode
            )
            episode_weights.append(action.numpy())
            
            # Step environment
            state, reward, terminated, truncated, info = env.step(action.numpy())
            episode_return += reward
            episode_length += 1
            done = terminated or truncated
        
        # Calculate episode statistics
        episode_alphas = np.array(episode_alphas)
        episode_weights = np.array(episode_weights)
        
        results['episode_returns'].append(episode_return)
        results['episode_lengths'].append(episode_length)
        
        # Calculate Sharpe (from info if available)
        if 'sharpe_ratio' in info:
            results['sharpe_ratios'].append(info['sharpe_ratio'])
        
        # Calculate max drawdown (from info if available)
        if 'max_drawdown' in info:
            results['max_drawdowns'].append(info['max_drawdown'])
        
        # Alpha diversity stats
        alpha_std_per_step = episode_alphas.std(axis=1)
        results['alpha_stats_history'].append({
            'mean': alpha_std_per_step.mean(),
            'min': alpha_std_per_step.min(),
            'max': alpha_std_per_step.max()
        })
        
        # Weight diversity stats
        weight_std_per_step = episode_weights.std(axis=1)
        results['weight_stats_history'].append({
            'mean': weight_std_per_step.mean(),
            'min': weight_std_per_step.min(),
            'max': weight_std_per_step.max()
        })
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"EVALUATION SUMMARY ({num_episodes} episode(s))")
    print(f"{'='*80}")
    print(f"Episode Returns: mean={np.mean(results['episode_returns']):.2f}, "
          f"std={np.std(results['episode_returns']):.2f}")
    print(f"Episode Lengths: mean={np.mean(results['episode_lengths']):.1f}")
    
    if results['sharpe_ratios']:
        print(f"Sharpe Ratios: mean={np.mean(results['sharpe_ratios']):.3f}")
    if results['max_drawdowns']:
        print(f"Max Drawdowns: mean={np.mean(results['max_drawdowns'])*100:.2f}%")
    
    avg_alpha_std = np.mean([s['mean'] for s in results['alpha_stats_history']])
    avg_weight_std = np.mean([s['mean'] for s in results['weight_stats_history']])
    
    print(f"\nAlpha Std (episode avg): {avg_alpha_std:.3f}")
    print(f"Weight Std (episode avg): {avg_weight_std:.3f}")
    
    if avg_alpha_std < 0.3:
        print(f"\n⚠️ WARNING: Low alpha std indicates TCN not learning")
    elif avg_alpha_std > 0.5:
        print(f"\n✅ Good alpha diversity - TCN is learning!")
    
    if avg_weight_std < 0.05:
        print(f"⚠️ WARNING: Near-uniform weights despite alphas")
    elif avg_weight_std > 0.08:
        print(f"✅ Good weight diversity - avoiding uniform allocation!")
    
    print(f"{'='*80}\n")
    
    return results


# Convenience function for quick checks
def quick_alpha_check(agent, env, num_samples=5):
    """
    Quick alpha check on random states.
    
    Args:
        agent: PPOAgentTF instance
        env: Environment instance
        num_samples: Number of random states to check
    """
    print(f"\n{'='*80}")
    print(f"QUICK ALPHA CHECK ({num_samples} random states)")
    print(f"{'='*80}")
    
    alpha_stds = []
    for i in range(num_samples):
        state, _ = env.reset()
        stats = diagnose_alpha_distribution(agent, state, verbose=False)
        alpha_stds.append(stats['alpha_std'])
        print(f"Sample {i+1}: alpha_std={stats['alpha_std']:.3f}, "
              f"weight_std={stats['implied_weights'].std():.3f}")
    
    mean_alpha_std = np.mean(alpha_stds)
    print(f"\nMean alpha_std: {mean_alpha_std:.3f}")
    
    if mean_alpha_std < 0.3:
        print("⚠️ Alphas too uniform - TCN may not be learning")
    elif mean_alpha_std > 0.5:
        print("✅ Good alpha diversity!")
    else:
        print("⚡ Moderate diversity - more training may help")
    
    print(f"{'='*80}\n")
    
    return np.mean(alpha_stds)
