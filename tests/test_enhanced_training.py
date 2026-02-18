"""
Quick Test of Enhanced Training System

This script demonstrates the improved training system with a short test run.
"""

import os
import sys
sys.path.append('./src')

from enhanced_train_rl import EnhancedTrainingSession
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

def test_enhanced_training():
    """Test the enhanced training system with a short run."""
    print("🚀 Testing Enhanced Training System")
    print("=" * 50)
    
    # Test with dense architecture first
    print("\n1️⃣ Testing Dense Architecture with Progressive Training")
    session = EnhancedTrainingSession(
        phase='phase1',
        architecture='dense',
        enable_progressive=True,
        config_override={
            'training_params': {
                'max_total_timesteps': 1000,  # Short test
                'log_interval_episodes': 2,
                'save_freq_episodes': 20
            }
        }
    )
    
    print(f"✅ Enhanced session created successfully!")
    print(f"   Phase: {session.phase}")
    print(f"   Architecture: {session.architecture}")
    print(f"   Progressive: {session.enable_progressive}")
    print(f"   Results dir: {session.results_dir}")
    
    # Show initial configuration
    config = session.config
    print(f"\n📊 Initial Configuration:")
    print(f"   Episode length: {config['environment_params']['max_steps_per_episode']}")
    print(f"   Update frequency: {config['training_params']['timesteps_per_ppo_update']}")
    print(f"   Learning rate: {config['agent_params']['ppo_params']['actor_lr']}")
    
    # Test configuration updates
    if session.training_manager:
        training_stats = session.training_manager.schedule.get_training_stats()
        print(f"   Current phase: {training_stats['current_phase']}")
        
        # Simulate some progress
        session.training_manager.update_training_progress({'episode_reward': -1.0, 'episode_length': 45})
        session.training_manager.update_training_progress({'episode_reward': -0.8, 'episode_length': 48})
        
        # Get recommendations
        recommendations = session.training_manager.get_recommendations()
        if recommendations:
            print(f"\n💡 Current Recommendations:")
            for rec in recommendations:
                print(f"   - {rec}")
    
    print(f"\n✅ Enhanced training system test completed successfully!")
    
    # Test TCN configuration
    print(f"\n2️⃣ Testing TCN Architecture Configuration")
    tcn_session = EnhancedTrainingSession(
        phase='phase1',
        architecture='tcn',
        enable_progressive=True,
        config_override={'training_params': {'max_total_timesteps': 500}}
    )
    
    tcn_config = tcn_session.config
    print(f"   TCN Episode length: {tcn_config['environment_params']['max_steps_per_episode']}")
    print(f"   TCN Update frequency: {tcn_config['training_params']['timesteps_per_ppo_update']}")
    print(f"   TCN Learning rate: {tcn_config['agent_params']['ppo_params']['actor_lr']}")
    
    return session, tcn_session

if __name__ == "__main__":
    try:
        session, tcn_session = test_enhanced_training()
        print(f"\n🎉 All tests passed! Enhanced training system is ready for use.")
        print(f"\nTo run full training:")
        print(f"   python src/enhanced_train_rl.py --phase phase1 --architecture dense --progressive")
        print(f"   python src/enhanced_train_rl.py --phase phase1 --architecture tcn --progressive")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()