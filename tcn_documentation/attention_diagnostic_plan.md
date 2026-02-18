# TCN-Attention Diagnostic Plan

## Goal
Investigate why TCN and TCN-Attention variants yield identical performance, suggesting the attention mechanism might be ineffective or bypassed.

## Diagnosis Steps

### 1. Verify Configuration
- Confirm `use_attention` is actually `True` in the runtime config.
- Check `tcn_filters` vs `attention_dim`. If `attention_dim` is too small compared to TCN output, it might be a bottleneck.
- Check `attention_heads`.

### 2. Inspect Model Weights (Post-Initialization)
- Load the "identical" model (TCN-Attention).
- Print summary to ensure `attention` layer exists and has parameters.
- Check if attention weights (`wq`, `wk`, `wv`) are moving from initialization.

### 3. Runtime Attention Weight Check
- **Action**: Create a small debug script or notebook cell.
- **Steps**:
    - Instantiate `TCNAttentionActor`.
    - Pass a dummy input `(1, 60, 16)`.
    - Extract attention weights from the `MultiHeadSelfAttention` layer.
    - Check if they are uniform (1/N) or distinct. Uniform weights imply the mechanism is not learning importance.

### 4. Gradient Flow (Optional)
- Train for 1 step and check gradients for the attention layer. If 0, the layer is detached.

## Proposed Patch
If the attention layer is working but weak:
1.  **Increase `attention_dim`**: Match `tcn_filters[-1]` (e.g., 64 -> 128).
2.  **Remove Projection**: If dimensions match, remove the bottleneck projection layer to preserve information.
3.  **Add Residual Connection**: Ensure the attention block has a strong residual connection (it does in the code: `output = self.layernorm(x + output)`).

## Action
Create a `debug_attention.py` script to inspect a saved model or fresh instance.
