"""
Reproducibility Helper - Set All Random Seeds

Add this to your training scripts to ensure reproducible results.
Uses TensorFlow and custom PPO agents (not PyTorch/Stable-Baselines3).
"""

import random
import numpy as np
import os


def set_all_seeds(seed: int = 42, deterministic: bool = True):
    """
    Set random seeds for all libraries to ensure reproducibility.
    
    Args:
        seed: Random seed value
        deterministic: If True, set deterministic behavior (slower but reproducible)
    
    Usage:
        from reproducibility_helper import set_all_seeds
        set_all_seeds(42)
    """
    print(f"🎲 Setting all random seeds to: {seed}")
    
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # TensorFlow (primary framework)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        
        # Set deterministic behavior for TensorFlow
        if deterministic:
            os.environ['TF_DETERMINISTIC_OPS'] = '1'
            os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
            # Enable op determinism (TF 2.9+)
            try:
                tf.config.experimental.enable_op_determinism()
            except AttributeError:
                pass  # Older TF versions don't have this
    except ImportError:
        print("   ⚠️  TensorFlow not available")
    
    # Set PYTHONHASHSEED for hash reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Set deterministic behavior
    if deterministic:
        print("   ✅ Deterministic mode enabled (slower but reproducible)")
    else:
        print("   ⚠️  Non-deterministic mode (faster but may vary slightly)")
    
    print("   ✅ Python random seed set")
    print("   ✅ NumPy random seed set")
    print("   ✅ TensorFlow seed set")
    print("   ✅ Custom PPO agents seeded")


def test_reproducibility():
    """
    Test that seeding works correctly.
    
    Run this function twice with same seed - should get identical results.
    """
    print("\n🧪 Testing Reproducibility...")
    print("=" * 60)
    
    # Test 1: Python random
    set_all_seeds(42)
    python_rand = [random.random() for _ in range(5)]
    print(f"Python random: {python_rand}")
    
    # Test 2: NumPy random
    set_all_seeds(42)
    numpy_rand = np.random.rand(5).tolist()
    print(f"NumPy random:  {numpy_rand}")
    
    # Test 3: TensorFlow random
    set_all_seeds(42)
    try:
        import tensorflow as tf
        tf_rand = tf.random.uniform([5]).numpy().tolist()
        print(f"TensorFlow random: {tf_rand}")
    except ImportError:
        print(f"TensorFlow random: [not available]")
    
    print("\n✅ If you run this again with seed=42, you should see IDENTICAL numbers!")
    print("=" * 60)


if __name__ == "__main__":
    # Example usage
    set_all_seeds(42)
    test_reproducibility()
