# Alpha Distribution Diagnostic Tool
"""
Add this to your notebook AFTER creating the agent to monitor alpha distributions
during training and evaluation.
"""

import tensorflow as tf
import numpy as np

def diagnose_alpha_distribution(agent, state, num_samples=1):
    """
    Check if TCN is learning to discriminate between assets.
    
    Args:
        agent: PPOAgentTF instance
        state: Current state (numpy array)
        num_samples: Number of samples to inspect (default 1)
    
    Returns:
        alpha: Alpha values from the actor
        mean_weights: Implied mean weights from Dirichlet
    """
    import tensorflow as tf
    import numpy as np
    
    # Prepare state
    if agent.is_sequential:
        seq = agent._build_sequence(state)
        state_tensor = tf.constant([seq], dtype=tf.float32)
    else:
        state_tensor = tf.constant([state], dtype=tf.float32)
    
    # Get alpha parameters
    alpha = agent.actor(state_tensor, training=False)[0].numpy()
    
    print(f"\n{'='*80}")
    print(f"📊 ALPHA DISTRIBUTION DIAGNOSTICS")
    print(f"{'='*80}")
    print(f"   Alpha values: {alpha}")
    print(f"   Mean: {alpha.mean():.3f}")
    print(f"   Std: {alpha.std():.3f}")
    print(f"   Min: {alpha.min():.3f}, Max: {alpha.max():.3f}")
    print(f"   Ratio (max/min): {alpha.max() / alpha.min():.2f}x")
    
    # Interpret
    if alpha.std() < 0.3:
        print(f"\n   ⚠️ WARNING: Alphas too uniform! TCN not learning.")
        print(f"   → Check if gradients are flowing back to actor")
        print(f"   → Consider deeper TCN or lower entropy coefficient")
    elif alpha.std() < 1.0:
        print(f"\n   ⚡ Moderate differentiation. May need more training.")
    else:
        print(f"\n   ✅ Good differentiation! TCN learning asset preferences.")
    
    # Calculate implied mean weights (Dirichlet mean)
    mean_weights = alpha / alpha.sum()
    print(f"\n   Implied mean weights: {mean_weights}")
    print(f"   Weight std: {mean_weights.std():.3f}")
    
    if mean_weights.std() < 0.05:
        print(f"   ⚠️ Near-uniform weights despite varied alphas!")
    
    print(f"{'='*80}\n")
    
    return alpha, mean_weights


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

# 1. DURING EVALUATION (one-time check)
"""
# After loading trained model:
state = env.reset()[0]
alpha, weights_pred = diagnose_alpha_distribution(agent, state)
"""

# 2. DURING TRAINING (periodic monitoring) 
"""
# Add to your training loop (after every N updates):
if update_num % 10 == 0:
    # Sample one state from current trajectory
    sample_state = states_batch[0]
    
    if agent.is_sequential:
        seq = agent._build_sequence(sample_state)
        state_tensor = tf.constant([seq], dtype=tf.float32)
    else:
        state_tensor = tf.constant([sample_state], dtype=tf.float32)
    
    sample_alpha = agent.actor(state_tensor, training=False)[0].numpy()
    
    logger.info(
        f"Update {update_num} - Alpha stats: "
        f"mean={sample_alpha.mean():.2f}, "
        f"std={sample_alpha.std():.2f}, "
        f"range=[{sample_alpha.min():.2f}, {sample_alpha.max():.2f}]"
    )
"""

# 3. FULL EPISODE ANALYSIS
"""
# Analyze alpha evolution throughout an episode:
def analyze_episode_alphas(env, agent, num_steps=None):
    state, _ = env.reset()
    alpha_history = []
    weight_history = []
    
    step = 0
    done = False
    while not done:
        # Get alpha for current state
        if agent.is_sequential:
            seq = agent._build_sequence(state)
            state_tensor = tf.constant([seq], dtype=tf.float32)
        else:
            state_tensor = tf.constant([state], dtype=tf.float32)
        
        alpha = agent.actor(state_tensor, training=False)[0].numpy()
        alpha_history.append(alpha)
        
        # Get action (weights)
        action, log_prob, value = agent.get_action_and_value(
            state, deterministic=True, evaluation_mode='mode'
        )
        weight_history.append(action.numpy()) 
        
        # Step environment
        state, reward, terminated, truncated, info = env.step(action.numpy())
        done = terminated or truncated
        
        step += 1
        if num_steps and step >= num_steps:
            break
    
    alpha_history = np.array(alpha_history)
    weight_history = np.array(weight_history)
    
    print(f"\\n{'='*80}")
    print(f"EPISODE ALPHA ANALYSIS ({step} steps)")
    print(f"{'='*80}")
    print(f"Alpha std across episode:")
    print(f"  Min: {alpha_history.std(axis=1).min():.3f}")
    print(f"  Max: {alpha_history.std(axis=1).max():.3f}")
    print(f"  Mean: {alpha_history.std(axis=1).mean():.3f}")
    print(f"\\nWeight std across episode:")
    print(f"  Min: {weight_history.std(axis=1).min():.3f}")
    print(f"  Max: {weight_history.std(axis=1).max():.3f}")
    print(f"  Mean: {weight_history.std(axis=1).mean():.3f}")
    print(f"{'='*80}\\n")
    
    return alpha_history, weight_history

# Usage:
# alpha_hist, weight_hist = analyze_episode_alphas(env, agent, num_steps=50)
"""
