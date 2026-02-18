"""
Test script for the PPO Agent with Dirichlet distribution.

This script tests the complete agent implementation including:
1. ActorNetwork and CriticNetwork functionality
2. PPOAgentTF action sampling and value estimation
3. Dirichlet distribution properties
4. PPO update mechanism
5. Integration with portfolio environment mock
"""

import sys
import os
import numpy as np
import logging

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_networks():
    """Test the ActorNetwork and CriticNetwork directly."""
    print("=" * 80)
    print("TESTING ACTOR-CRITIC NETWORKS")
    print("=" * 80)
    
    try:
        import tensorflow as tf
        from src.agents.actor_critic_tf import ActorNetwork, CriticNetwork, create_actor_critic_networks
        
        # Test parameters
        batch_size = 4
        state_dim = 8
        num_assets = 5
        
        print(f"Creating networks with state_dim={state_dim}, num_assets={num_assets}")
        
        # Create networks
        actor, critic = create_actor_critic_networks(
            input_dim=state_dim,
            num_assets=num_assets,
            actor_hidden_dims=[64, 64],
            critic_hidden_dims=[64, 64]
        )
        
        # Create dummy state
        dummy_state = tf.random.normal((batch_size, state_dim))
        print(f"Input state shape: {dummy_state.shape}")
        
        # Test actor
        alpha = actor(dummy_state)
        print(f"✅ Actor forward pass successful")
        print(f"   Alpha shape: {alpha.shape}")
        print(f"   Alpha values (sample): {alpha[0].numpy()}")
        print(f"   All alpha > 0: {tf.reduce_all(alpha > 0).numpy()}")
        
        # Test critic
        value = critic(dummy_state)
        print(f"✅ Critic forward pass successful")
        print(f"   Value shape: {value.shape}")
        print(f"   Value estimates: {value[:, 0].numpy()}")
        
        # Test Dirichlet distribution
        import tensorflow_probability as tfp
        tfd = tfp.distributions
        
        dirichlet = tfd.Dirichlet(alpha)
        sample = dirichlet.sample()
        log_prob = dirichlet.log_prob(sample)
        entropy = dirichlet.entropy()
        
        print(f"✅ Dirichlet distribution successful")
        print(f"   Sample shape: {sample.shape}")
        print(f"   Sample weights (should sum to 1): {sample[0].numpy()}")
        print(f"   Sample sums: {tf.reduce_sum(sample, axis=1).numpy()}")
        print(f"   Log probabilities: {log_prob.numpy()}")
        print(f"   Entropies: {entropy.numpy()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Network test failed: {e}")
        return False

def test_ppo_agent():
    """Test the PPOAgentTF class."""
    print("\n" + "=" * 80)
    print("TESTING PPO AGENT")
    print("=" * 80)
    
    try:
        import tensorflow as tf
        from src.agents.ppo_agent_tf import PPOAgentTF
        
        # Test parameters
        state_dim = 8
        num_assets = 5
        
        print(f"Creating PPO agent with state_dim={state_dim}, num_assets={num_assets}")
        
        # Create agent
        agent = PPOAgentTF(
            state_dim=state_dim,
            num_assets=num_assets,
            actor_hidden_dims=[64, 64],
            critic_hidden_dims=[64, 64],
            actor_lr=3e-3,
            critic_lr=3e-3
        )
        
        print(f"✅ PPO agent created successfully")
        
        # Test single state action sampling
        dummy_state = tf.random.normal((state_dim,))
        action, log_prob, value = agent.get_action_and_value(dummy_state)
        
        print(f"✅ Action sampling successful")
        print(f"   State shape: {dummy_state.shape}")
        print(f"   Action shape: {action.shape}")
        print(f"   Action weights: {action.numpy()}")
        print(f"   Action sum: {tf.reduce_sum(action).numpy():.6f}")
        print(f"   Log probability: {log_prob.numpy():.6f}")
        print(f"   Value estimate: {value.numpy():.6f}")
        
        # Test deterministic action
        det_action, det_log_prob, det_value = agent.get_action_and_value(dummy_state, deterministic=True)
        print(f"✅ Deterministic action successful")
        det_sum = float(tf.reduce_sum(det_action).numpy())
        print(f"   Deterministic action sum: {det_sum:.6f}")
        actions_different = not bool(tf.reduce_all(tf.abs(action - det_action) < 1e-6).numpy())
        print(f"   Actions are different: {actions_different}")
        
        # Test batch action sampling
        batch_states = tf.random.normal((4, state_dim))
        batch_actions, batch_log_probs, batch_values = agent.get_action_and_value(batch_states)
        
        print(f"✅ Batch action sampling successful")
        print(f"   Batch states shape: {batch_states.shape}")
        print(f"   Batch actions shape: {batch_actions.shape}")
        print(f"   Batch action sums: {tf.reduce_sum(batch_actions, axis=1).numpy()}")
        
        return True, agent
        
    except Exception as e:
        print(f"❌ PPO agent test failed: {e}")
        return False, None

def test_ppo_training():
    """Test the PPO training mechanism."""
    print("\n" + "=" * 80)
    print("TESTING PPO TRAINING")
    print("=" * 80)
    
    try:
        import tensorflow as tf
        from src.agents.ppo_agent_tf import PPOAgentTF
        
        # Create agent
        state_dim = 8
        num_assets = 5
        
        agent = PPOAgentTF(
            state_dim=state_dim,
            num_assets=num_assets,
            actor_hidden_dims=[32, 32],  # Smaller for faster testing
            critic_hidden_dims=[32, 32]
        )
        
        print(f"✅ Agent created for training test")
        
        # Simulate a short episode
        num_steps = 10
        
        for step in range(num_steps):
            # Random state
            state = tf.random.normal((state_dim,))
            
            # Get action and value
            action, log_prob, value = agent.get_action_and_value(state)
            
            # Simulate reward (random for testing)
            reward = tf.random.normal(()) * 0.01  # Small random return
            
            # Random done flag
            done = step == num_steps - 1
            
            # Store transition
            agent.store_transition(state, action, log_prob, reward, value, done)
        
        print(f"✅ Stored {num_steps} transitions")
        
        # Test update
        print("Running PPO update...")
        stats = agent.update(num_epochs=2, batch_size=5)  # Small values for testing
        
        print(f"✅ PPO update successful")
        print(f"   Actor loss: {stats.get('actor_loss', 'N/A')}")
        print(f"   Critic loss: {stats.get('critic_loss', 'N/A')}")
        print(f"   Policy loss: {stats.get('policy_loss', 'N/A')}")
        print(f"   Entropy: {stats.get('entropy', 'N/A')}")
        print(f"   Mean advantage: {stats.get('mean_advantage', 'N/A')}")
        print(f"   Mean return: {stats.get('mean_return', 'N/A')}")
        
        # Test that memory was cleared
        assert len(agent.memory['states']) == 0, "Memory should be cleared after update"
        print(f"✅ Memory cleared after update")
        
        return True
        
    except Exception as e:
        print(f"❌ PPO training test failed: {e}")
        return False

def test_portfolio_constraints():
    """Test that the agent respects portfolio constraints."""
    print("\n" + "=" * 80)
    print("TESTING PORTFOLIO CONSTRAINTS")
    print("=" * 80)
    
    try:
        import tensorflow as tf
        from src.agents.ppo_agent_tf import PPOAgentTF
        
        # Create agent
        state_dim = 8
        num_assets = 5
        
        agent = PPOAgentTF(
            state_dim=state_dim,
            num_assets=num_assets
        )
        
        # Test multiple random states
        num_tests = 100
        all_weights_positive = True
        all_weights_sum_to_one = True
        tolerance = 1e-5
        
        for i in range(num_tests):
            state = tf.random.normal((state_dim,))
            action, _, _ = agent.get_action_and_value(state)
            
            # Check constraints
            weights = action.numpy()
            
            # All weights should be positive
            if np.any(weights < 0):
                all_weights_positive = False
                print(f"   ❌ Negative weights found: {weights}")
            
            # Weights should sum to 1
            weight_sum = np.sum(weights)
            if abs(weight_sum - 1.0) > tolerance:
                all_weights_sum_to_one = False
                print(f"   ❌ Weights don't sum to 1: {weight_sum}")
        
        print(f"✅ Tested {num_tests} random states")
        print(f"   All weights positive: {all_weights_positive}")
        print(f"   All weights sum to 1: {all_weights_sum_to_one}")
        
        if all_weights_positive and all_weights_sum_to_one:
            print(f"✅ Portfolio constraints satisfied!")
            return True
        else:
            print(f"❌ Portfolio constraints violated!")
            return False
            
    except Exception as e:
        print(f"❌ Portfolio constraints test failed: {e}")
        return False

def test_configuration_compatibility():
    """Test compatibility with the project configuration."""
    print("\n" + "=" * 80)
    print("TESTING CONFIGURATION COMPATIBILITY")
    print("=" * 80)
    
    try:
        import tensorflow as tf
        from src.config import get_active_config
        from src.agents.ppo_agent_tf import PPOAgentTF
        
        # Get Phase 1 configuration
        config = get_active_config('phase1')
        
        print(f"✅ Loaded Phase 1 configuration")
        print(f"   Number of assets: {config['NUM_ASSETS']}")
        print(f"   Agent type: {config['agent_params']['actor_critic_type']}")
        print(f"   Actor hidden dims: {config['agent_params']['actor_hidden_dims']}")
        print(f"   Critic hidden dims: {config['agent_params']['critic_hidden_dims']}")
        
        # Extract parameters
        agent_params = config['agent_params']
        ppo_params = agent_params['ppo_params']
        
        # Calculate state dimension (placeholder - would come from environment)
        # For Phase 1: num_features_per_asset * num_assets + previous_weights
        # Count technical indicator outputs
        num_features_per_asset = 0
        for ti_config in config['feature_params']['technical_indicators']:
            num_features_per_asset += len(ti_config['output_cols'])
        
        # Add log returns
        if config['feature_params']['include_log_returns']:
            num_features_per_asset += 1
        
        state_dim = num_features_per_asset * config['NUM_ASSETS'] + (config['NUM_ASSETS'] + 1)
        
        print(f"   Estimated state dimension: {state_dim}")
        
        # Create agent with config parameters
        agent = PPOAgentTF(
            state_dim=state_dim,
            num_assets=config['NUM_ASSETS'],
            actor_hidden_dims=agent_params['actor_hidden_dims'],
            critic_hidden_dims=agent_params['critic_hidden_dims'],
            actor_lr=ppo_params['actor_lr'],
            critic_lr=ppo_params['critic_lr'],
            gamma=ppo_params['gamma'],
            gae_lambda=ppo_params['gae_lambda'],
            policy_clip=ppo_params['policy_clip'],
            entropy_coef=ppo_params['entropy_coef'],
            vf_coef=ppo_params['vf_coef'],
            max_grad_norm=ppo_params['max_grad_norm']
        )
        
        print(f"✅ Agent created with configuration parameters")
        
        # Test basic functionality
        dummy_state = tf.random.normal((state_dim,))
        action, log_prob, value = agent.get_action_and_value(dummy_state)
        
        print(f"✅ Agent functionality test passed")
        print(f"   Action shape: {action.shape}")
        print(f"   Expected action dimensions: {config['NUM_ASSETS'] + 1}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration compatibility test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 STARTING PPO AGENT COMPREHENSIVE TESTS")
    print("=" * 100)
    
    tests = [
        ("Actor-Critic Networks", test_networks),
        ("PPO Agent Basic", test_ppo_agent),
        ("PPO Training", test_ppo_training),
        ("Portfolio Constraints", test_portfolio_constraints),
        ("Configuration Compatibility", test_configuration_compatibility)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            if isinstance(result, tuple):
                result = result[0]  # Take first element if tuple returned
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 100)
    print("TEST SUMMARY")
    print("=" * 100)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<30} {status}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! PPO Agent is ready for Day 4 completion.")
    else:
        print("⚠️ Some tests failed. Please review the implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = main()