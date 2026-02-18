
import numpy as np
import tensorflow as tf
import os
import sys

# Add source to path
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

from src.agents.actor_critic_tf import TCNAttentionActor, TCNAttentionCritic
from src.config import get_active_config


def debug_attention_init():
    """
    Instantiate TCNAttentionActor and check attention weights.
    """
    print("Debugging TCNAttention Initialization...")
    
    # 1. Config
    config = get_active_config('phase1')
    agent_cfg = config.get('agent_params', {})
    input_dim = int(agent_cfg.get('debug_input_dim', 16))  # fallback for standalone debug
    num_actions = int(config.get('NUM_ASSETS', len(config.get('ASSET_TICKERS', [])))) + 1
    batch_size = int(agent_cfg.get('debug_batch_size', 4))
    timesteps = int(agent_cfg.get('sequence_length', 60))
    tcn_filters = agent_cfg.get('tcn_filters', [64, 64])
    kernel_size = int(agent_cfg.get('tcn_kernel_size', 3))
    dilations = agent_cfg.get('tcn_dilations', [1, 2])
    attention_heads = int(agent_cfg.get('attention_heads', 4))
    attention_dim = int(agent_cfg.get('attention_dim', 64))
    dropout = float(agent_cfg.get('attention_dropout', agent_cfg.get('tcn_dropout', 0.1)))

    print("   Runtime config:")
    print(f"   - input_dim(debug): {input_dim}")
    print(f"   - num_actions: {num_actions}")
    print(f"   - batch_size(debug): {batch_size}")
    print(f"   - timesteps(sequence_length): {timesteps}")
    print(f"   - tcn_filters: {tcn_filters}")
    print(f"   - tcn_kernel_size: {kernel_size}")
    print(f"   - tcn_dilations: {dilations}")
    print(f"   - attention_heads: {attention_heads}")
    print(f"   - attention_dim: {attention_dim}")
    print(f"   - dropout: {dropout}")
    
    # 2. Instantiate Actor
    actor = TCNAttentionActor(
        input_dim=input_dim,
        num_actions=num_actions,
        tcn_filters=tcn_filters,
        kernel_size=kernel_size,
        dilations=dilations,
        attention_heads=attention_heads,
        attention_dim=attention_dim,
        dropout=dropout,
        name="debug_actor"
    )
    
    # 3. Build with dummy input
    dummy_state = tf.random.normal((batch_size, timesteps, input_dim))
    print(f"   Input shape: {dummy_state.shape}")
    
    # Run forward pass
    alpha = actor(dummy_state)
    print(f"   Output alpha shape: {alpha.shape}")
    
    # 4. Inspect Attention Weights
    print("\nInspecting Attention Layer Weights:")
    att_layer = actor.attention
    
    if hasattr(att_layer, 'wq'):
        wq = att_layer.wq.kernel.numpy()
        print(f"   WQ stats: mean={wq.mean():.6f}, std={wq.std():.6f}, range=[{wq.min():.6f}, {wq.max():.6f}]")
        
        # Check if they are non-zero (initialized)
        if np.allclose(wq, 0):
            print("   WARNING: WQ weights are all zero!")
        else:
            print("   WQ weights look initialized.")
    else:
        print("   Error: Could not find 'wq' layer in attention block.")


    # 5. Check Attention Output Magnitude vs Residual
    print("\nSignal Magnitude Check:")
    
    # We need to access intermediate outputs. 
    # Since we can't easily hook, we'll manually run the layers in the script to see values.
    x = dummy_state
    for block in actor.tcn_blocks:
        x = block(x)
    tcn_out = x
    
    # Projection
    x_proj = actor.projection(x)
    
    # Attention Output (before residual addition inside the layer)
    # Note: MultiHeadSelfAttention applies residual internally: output = layer_norm(x + attention_output)
    # We want to see 'attention_output' magnitude.
    # To do this, we'll call the internal sub-layers of attention if possible, 
    # OR just compare the output of attention block vs input.
    
    att_block_out = actor.attention(x_proj)
    
    # Approximate Residual check: The MultiHeadSelfAttention layer does: Norm(x + Attn(x))
    # If Attn(x) is near zero, then att_block_out ~= Norm(x).
    
    # Let's peek at the internal attention weights (Softmax)
    # We can rebuild the attention step manually to inspect.
    
    q = actor.attention.wq(x_proj)
    k = actor.attention.wk(x_proj)
    v = actor.attention.wv(x_proj)
    
    print(f"   TCN Output Mean stats: {tcn_out.numpy().mean():.4f} +/- {tcn_out.numpy().std():.4f}")
    print(f"   Projected Input to Attn: {x_proj.numpy().mean():.4f} +/- {x_proj.numpy().std():.4f}")
    
    # Check if Q, K, V are collapsing
    print(f"   Q Stats: {q.numpy().mean():.4f} +/- {q.numpy().std():.4f}")
    
    # Check simple difference
    # If attention is doing work, output should be different from input (modulo normalization)
    diff = tf.reduce_mean(tf.abs(att_block_out - x_proj)) # crude check because of LayerNorm
    print(f"   Attention Block Output delta from input: {diff.numpy():.4f}")
    
    if diff.numpy() < 0.01:
        print("   WARNING: Attention block output is very similar to input (or purely Norm). Attention might be weak.")
    else:
        print("   Attention block is modifying the signal significantly.")


    # 5. Check Attention Output (is it doing anything?)
    # We can't easily hook into the layer without a custom call, but we can check if gradients flow (simulated)
    print("\nGradient Check (Simulation):")
    with tf.GradientTape() as tape:
        tape.watch(dummy_state)
        # Force re-call to capture graph
        out = actor(dummy_state)
        loss = tf.reduce_sum(out)
    


    # Corrected Gradient Calculation (Single Pass)
    grads = tape.gradient(loss, actor.trainable_variables)

    
    print(f"\n trainable_variables count: {len(actor.trainable_variables)}")
    # Print ALL to verify names
    for v in actor.trainable_variables:
        print(f"   - {v.name}")
    
    # Find attention gradients - broaden search
    att_grads = []
    att_vars = []
    for g, v in zip(grads, actor.trainable_variables):
        if 'attention' in v.name or 'wq' in v.name or 'wk' in v.name or 'wv' in v.name:
            att_vars.append(v.name)
            att_grads.append(g)
    
    print(f"\nIdentified {len(att_vars)} attention variables: {att_vars}")


    
    if not att_grads:
        print("   Error: No gradients found for attention layer variables!")
    else:
        grad_norms = [np.mean(np.abs(g.numpy())) for g in att_grads if g is not None]
        avg_grad = np.mean(grad_norms) if grad_norms else 0.0
        print(f"   Attention Gradient Magnitude (Avg): {avg_grad:.8f}")
        if avg_grad < 1e-9:
            print("   WARNING: Vanishing gradients in attention layer!")
        else:
            print("   Gradients are flowing to attention layer.")


if __name__ == "__main__":
    debug_attention_init()
