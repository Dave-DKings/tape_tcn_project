"""
Refactored Actor-Critic Networks with Architecture Factory Pattern

This module implements a TCN-focused architecture system supporting:
- TCN (Temporal Convolutional Network) for sequence processing
- TCN+Attention for advanced temporal feature extraction
- TCN+Fusion for hierarchical asset/context fusion

All architectures output Dirichlet distribution parameters for portfolio weights.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model # type: ignore
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from src import config
from src.config import PHASE1_CONFIG

# Extract default values from config to use as fallbacks
_DEFAULT_AGENT_PARAMS = PHASE1_CONFIG.get('agent_params', {})
_DEFAULT_TCN_FILTERS = _DEFAULT_AGENT_PARAMS.get('tcn_filters', [64, 128, 256])
_DEFAULT_TCN_DILATIONS = _DEFAULT_AGENT_PARAMS.get('tcn_dilations', [1, 2, 4, 8, 16])
_DEFAULT_TCN_KERNEL_SIZE = _DEFAULT_AGENT_PARAMS.get('tcn_kernel_size', 3)
_DEFAULT_TCN_DROPOUT = _DEFAULT_AGENT_PARAMS.get('tcn_dropout', 0.2)
_DEFAULT_ACTOR_HIDDEN_DIMS = _DEFAULT_AGENT_PARAMS.get('actor_hidden_dims', [768, 512, 256, 128])
_DEFAULT_CRITIC_HIDDEN_DIMS = _DEFAULT_AGENT_PARAMS.get('critic_hidden_dims', [768, 512, 256, 128])
_DEFAULT_ATTENTION_HEADS = _DEFAULT_AGENT_PARAMS.get('attention_heads', 4)
_DEFAULT_ATTENTION_DIM = _DEFAULT_AGENT_PARAMS.get('attention_dim', 64)
_DEFAULT_ATTENTION_DROPOUT = _DEFAULT_AGENT_PARAMS.get('attention_dropout', 0.1)
_DEFAULT_FUSION_EMBED_DIM = _DEFAULT_AGENT_PARAMS.get('fusion_embed_dim', 128)
_DEFAULT_FUSION_HEADS = _DEFAULT_AGENT_PARAMS.get('fusion_attention_heads', 4)
_DEFAULT_FUSION_DROPOUT = _DEFAULT_AGENT_PARAMS.get('fusion_dropout', 0.1)
_DEFAULT_FUSION_ALPHA_HEAD_HIDDEN_DIMS = _DEFAULT_AGENT_PARAMS.get('fusion_alpha_head_hidden_dims', [])
_DEFAULT_FUSION_ALPHA_HEAD_DROPOUT = _DEFAULT_AGENT_PARAMS.get('fusion_alpha_head_dropout', _DEFAULT_FUSION_DROPOUT)
_DEFAULT_FUSION_CROSS_ASSET_MIXER_ENABLED = bool(_DEFAULT_AGENT_PARAMS.get('fusion_cross_asset_mixer_enabled', False))
_DEFAULT_FUSION_CROSS_ASSET_MIXER_LAYERS = int(_DEFAULT_AGENT_PARAMS.get('fusion_cross_asset_mixer_layers', 1))
_DEFAULT_FUSION_CROSS_ASSET_MIXER_EXPANSION = float(_DEFAULT_AGENT_PARAMS.get('fusion_cross_asset_mixer_expansion', 2.0))
_DEFAULT_FUSION_CROSS_ASSET_MIXER_DROPOUT = _DEFAULT_AGENT_PARAMS.get('fusion_cross_asset_mixer_dropout', _DEFAULT_FUSION_DROPOUT)
_DEFAULT_ALPHA_ACTIVATION = _DEFAULT_AGENT_PARAMS.get('dirichlet_alpha_activation', 'elu')
_DEFAULT_EXP_CLIP = tuple(_DEFAULT_AGENT_PARAMS.get('dirichlet_exp_clip', (-5.0, 3.0)))


def _to_tensor_with_cast(value: Any, dtype: Optional[tf.dtypes.DType] = None) -> tf.Tensor:
    """Convert to tensor and cast dtype explicitly (safe with mixed precision tensors)."""
    tensor = tf.convert_to_tensor(value)
    if dtype is not None:
        target_dtype = tf.dtypes.as_dtype(dtype)
        if tensor.dtype != target_dtype:
            tensor = tf.cast(tensor, target_dtype)
    return tensor


# ============================================================================
# ATTENTION MECHANISM
# ============================================================================

class MultiHeadSelfAttention(layers.Layer):
    """
    Multi-head self-attention mechanism for sequential data.
    
    Can be applied after TCN layers to learn important temporal relationships.
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, name: str = "attention"):
        """
        Args:
            d_model: Dimension of the model (must be divisible by num_heads)
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(MultiHeadSelfAttention, self).__init__(name=name)
        
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        
        self.wq = layers.Dense(d_model, name=f'{name}_wq')
        self.wk = layers.Dense(d_model, name=f'{name}_wk')
        self.wv = layers.Dense(d_model, name=f'{name}_wv')
        
        self.dense = layers.Dense(d_model, name=f'{name}_output')
        self.dropout = layers.Dropout(dropout)
        self.layernorm = layers.LayerNormalization(epsilon=1e-6)
    
    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth)."""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, x, training=None):
        """
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        batch_size = tf.shape(x)[0]
        
        # Linear projections
        q = self.wq(x)  # (batch, seq_len, d_model)
        k = self.wk(x)
        v = self.wv(x)
        
        # Split into multiple heads
        q = self.split_heads(q, batch_size)  # (batch, num_heads, seq_len, depth)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        # Scaled dot-product attention
        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (batch, num_heads, seq_len, seq_len)
        
        # Scale
        # Keep scale dtype aligned with attention logits for mixed precision safety.
        dk = tf.cast(tf.shape(k)[-1], matmul_qk.dtype)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        # Softmax
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        attention_weights = self.dropout(attention_weights, training=training)
        
        # Apply attention to values
        attention_output = tf.matmul(attention_weights, v)  # (batch, num_heads, seq_len, depth)
        
        # Concatenate heads
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention_output, (batch_size, -1, self.d_model))
        
        # Final linear projection
        output = self.dense(concat_attention)
        output = self.dropout(output, training=training)
        
        # Residual connection and layer norm
        output = self.layernorm(x + output)
        
        return output


class CrossAssetMixerBlock(layers.Layer):
    """
    Lightweight cross-asset mixer block:
    pre-norm self-attention + feed-forward residual.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        expansion: float = 2.0,
        dropout: float = 0.1,
        name: str = "cross_asset_mixer",
    ):
        super(CrossAssetMixerBlock, self).__init__(name=name)
        if d_model % num_heads != 0:
            num_heads = 1
        hidden_dim = max(d_model, int(round(d_model * float(expansion))))

        self.norm_attn = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_norm_attn")
        self.attn = layers.MultiHeadAttention(
            num_heads=int(num_heads),
            key_dim=max(1, d_model // int(num_heads)),
            dropout=dropout,
            name=f"{name}_attn",
        )
        self.attn_dropout = layers.Dropout(dropout, name=f"{name}_attn_dropout")

        self.norm_ffn = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_norm_ffn")
        self.ffn_dense1 = layers.Dense(hidden_dim, activation="gelu", name=f"{name}_ffn_dense1")
        self.ffn_dropout1 = layers.Dropout(dropout, name=f"{name}_ffn_dropout1")
        self.ffn_dense2 = layers.Dense(d_model, activation=None, name=f"{name}_ffn_dense2")
        self.ffn_dropout2 = layers.Dropout(dropout, name=f"{name}_ffn_dropout2")

    def call(self, x, training=None):
        attn_in = self.norm_attn(x)
        attn_out = self.attn(attn_in, attn_in, training=training)
        attn_out = self.attn_dropout(attn_out, training=training)
        x = x + attn_out

        ffn_in = self.norm_ffn(x)
        ffn_out = self.ffn_dense1(ffn_in)
        ffn_out = self.ffn_dropout1(ffn_out, training=training)
        ffn_out = self.ffn_dense2(ffn_out)
        ffn_out = self.ffn_dropout2(ffn_out, training=training)
        return x + ffn_out


# ============================================================================
# TCN (TEMPORAL CONVOLUTIONAL NETWORK) BLOCK
# ============================================================================

class TCNBlock(layers.Layer):
    """
    Temporal Convolutional Network block with dilated causal convolutions.
    
    TCNs are efficient alternatives to RNNs for sequence modeling, offering:
    - Parallel processing (unlike sequential RNNs)
    - Large receptive fields through dilations
    - Stable gradients
    """
    
    def __init__(self, 
                 filters: int, 
                 kernel_size: int = None, 
                 dilation_rate: int = 1,
                 dropout: float = None,
                 name: str = "tcn_block"):
        """
        Args:
            filters: Number of convolutional filters
            kernel_size: Size of the convolutional kernel
            dilation_rate: Dilation rate for the convolution
            dropout: Dropout rate
        """
        super(TCNBlock, self).__init__(name=name)
        
        # Apply config defaults
        if kernel_size is None:
            kernel_size = _DEFAULT_TCN_KERNEL_SIZE
        if dropout is None:
            dropout = _DEFAULT_TCN_DROPOUT
        
        self.conv1 = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding='causal',
            activation='relu',
            name=f'{name}_conv1'
        )
        self.dropout1 = layers.Dropout(dropout)
        
        self.conv2 = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding='causal',
            activation='relu',
            name=f'{name}_conv2'
        )
        self.dropout2 = layers.Dropout(dropout)
        
        self.downsample = None
        self.relu = layers.Activation('relu')
    
    def build(self, input_shape):
        """Build the layer (create downsample if needed)."""
        if input_shape[-1] != self.conv2.filters:
            self.downsample = layers.Conv1D(
                filters=self.conv2.filters,
                kernel_size=1,
                padding='same',
                name=f'{self.name}_downsample'
            )
    
    def call(self, x, training=None):
        """
        Args:
            x: Input tensor of shape (batch, seq_len, features)
            
        Returns:
            Output tensor of shape (batch, seq_len, filters)
        """
        # Residual connection
        residual = x
        
        # First convolution
        out = self.conv1(x)
        out = self.dropout1(out, training=training)
        
        # Second convolution
        out = self.conv2(out)
        out = self.dropout2(out, training=training)
        
        # Downsample residual if needed
        if self.downsample is not None:
            residual = self.downsample(residual)
        
        # Add residual and apply activation
        out = self.relu(out + residual)
        
        return out


def _flatten_structured_sequence_input(state: Any) -> tf.Tensor:
    """
    Accept either flat sequence tensor or structured dict {"asset": ..., "context": ...}
    and return a flat sequence tensor (batch, timesteps, features).
    """
    if not isinstance(state, dict):
        return _to_tensor_with_cast(state, tf.float32)

    asset = state.get("asset")
    if asset is None:
        raise ValueError("Structured state must include key 'asset'.")

    asset = _to_tensor_with_cast(asset, tf.float32)
    if asset.shape.rank == 3:
        asset = tf.expand_dims(asset, axis=0)
    if asset.shape.rank != 4:
        raise ValueError(
            f"Structured asset input must have rank 3 or 4, got shape {asset.shape}"
        )

    batch = tf.shape(asset)[0]
    timesteps = tf.shape(asset)[1]
    asset_flat = tf.reshape(asset, (batch, timesteps, -1))

    context = state.get("context")
    if context is None:
        return asset_flat

    context = _to_tensor_with_cast(context, asset_flat.dtype)
    if context.shape.rank == 1:
        context = tf.expand_dims(tf.expand_dims(context, axis=0), axis=0)
    elif context.shape.rank == 2:
        context = tf.expand_dims(context, axis=1)
    elif context.shape.rank == 3:
        pass
    else:
        raise ValueError(
            f"Structured context input must have rank 1, 2, or 3, got shape {context.shape}"
        )

    pad_time = tf.maximum(0, timesteps - tf.shape(context)[1])
    context = tf.pad(context, [[0, 0], [0, pad_time], [0, 0]])
    context = context[:, :timesteps, :]

    return tf.concat([asset_flat, context], axis=-1)


# ============================================================================
# ACTOR NETWORKS
# ============================================================================

class DirichletActor(Model):
    """
    Base class that manages the adaptive Dirichlet epsilon schedule.
    """

    def __init__(
        self,
        *,
        name: str,
        epsilon_start: float = 0.5,
        epsilon_min: float = 0.1,
        alpha_activation: str = None,
        exp_clip: Tuple[float, float] = None,
        logit_temperature: float = 1.0,  # New parameter
        alpha_cap: float = None,         # New parameter
        adaptive_temperature_enabled: bool = False,
        adaptive_temperature_base: float = 1.0,
        adaptive_temperature_slope: float = 0.0,
        adaptive_temperature_min: float = 0.8,
        adaptive_temperature_max: float = 2.5,
        **kwargs,
    ):
        super(DirichletActor, self).__init__(name=name, **kwargs)
        self._epsilon_max_value = float(epsilon_start)
        self._epsilon_min_value = float(epsilon_min)
        # Apply config defaults if not explicitly provided
        if alpha_activation is None:
            alpha_activation = _DEFAULT_ALPHA_ACTIVATION
        if exp_clip is None:
            exp_clip = _DEFAULT_EXP_CLIP
        self._alpha_activation = alpha_activation.lower().strip()
        self._exp_clip = exp_clip
        self._version_flag = "v2_updated" 
        
        # Dirichlet Controls
        self._logit_temperature = float(logit_temperature) if logit_temperature else 1.0
        self._alpha_cap = float(alpha_cap) if alpha_cap else None
        self._adaptive_temperature_enabled = bool(adaptive_temperature_enabled)
        self._adaptive_temperature_base = float(max(adaptive_temperature_base, 1e-6))
        self._adaptive_temperature_slope = float(max(adaptive_temperature_slope, 0.0))
        temp_min = float(max(adaptive_temperature_min, 1e-6))
        temp_max = float(max(adaptive_temperature_max, temp_min))
        self._adaptive_temperature_min = temp_min
        self._adaptive_temperature_max = temp_max

        self._dirichlet_epsilon = tf.Variable(
            float(epsilon_start),
            trainable=False,
            dtype=tf.float32,
            name=f"{name}_epsilon",
        )

    def epsilon_value(self) -> tf.Tensor:
        """Return the current epsilon tensor (clamped to >0)."""
        return tf.maximum(self._dirichlet_epsilon, 1e-6)

    def update_dirichlet_epsilon(
        self,
        progress: float,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ) -> None:
        """
        Update epsilon based on normalized training progress (0 → 1).
        """
        min_val = float(self._epsilon_min_value if min_value is None else min_value)
        max_val = float(self._epsilon_max_value if max_value is None else max_value)
        progress_tensor = tf.convert_to_tensor(float(progress), dtype=tf.float32)
        progress_tensor = tf.clip_by_value(progress_tensor, 0.0, 1.0)
        min_tensor = tf.constant(min_val, dtype=tf.float32)
        max_tensor = tf.constant(max_val, dtype=tf.float32)
        new_value = max_tensor * (1.0 - progress_tensor)
        new_value = tf.clip_by_value(new_value, min_tensor, max_tensor)
        
        # Also decay temperature if desired (optional future feature)
        self._dirichlet_epsilon.assign(new_value)

    def reset_dirichlet_epsilon(self) -> None:
        """Return epsilon to its starting value."""
        self._dirichlet_epsilon.assign(self._epsilon_max_value)

    def _compute_alpha(self, logits: tf.Tensor) -> tf.Tensor:
        """
        Apply the selected activation to produce Dirichlet concentration parameters.
        Includes temperature scaling and optional capping.
        """
        logits = tf.convert_to_tensor(logits)
        eps = tf.cast(self.epsilon_value(), logits.dtype)
        activation = self._alpha_activation
        
        # 1. Apply Temperature Scaling (flatten logits before activation)
        if self._adaptive_temperature_enabled:
            abs_logits = tf.stop_gradient(tf.abs(logits))
            temp = (
                tf.cast(self._adaptive_temperature_base, logits.dtype)
                + tf.cast(self._adaptive_temperature_slope, logits.dtype) * abs_logits
            )
            temp = tf.clip_by_value(
                temp,
                tf.cast(self._adaptive_temperature_min, logits.dtype),
                tf.cast(self._adaptive_temperature_max, logits.dtype),
            )
            scaled_logits = logits / tf.maximum(temp, tf.cast(1e-6, logits.dtype))
        elif abs(self._logit_temperature - 1.0) > 1e-6:
            scaled_logits = logits / self._logit_temperature
        else:
            scaled_logits = logits

        # 2. Apply Activation (Shifted for Positivity)
        if activation == "elu":
            # ELU + 1 ensures strictly positive range (-1 -> 0)
            alpha = tf.nn.elu(scaled_logits) + 1.0 + eps
        elif activation == "softplus_shift":
            alpha = tf.nn.softplus(scaled_logits - 1.0) + 1.0 + eps
        elif activation == "swish":
            # Swish can be negative, so we must shift it too? 
            # Or assume user knows risks. Let's add +1 for safety similar to ELU.
            alpha = tf.nn.swish(scaled_logits) + 1.0 + eps
        elif activation == "mish":
            alpha = scaled_logits * tf.nn.tanh(tf.nn.softplus(scaled_logits)) + 1.0 + eps
        elif activation == "exp_clip":
            low, high = self._exp_clip
            alpha = tf.exp(tf.clip_by_value(scaled_logits, low, high)) + eps
        else:
            # Default / legacy: softplus + adaptive epsilon
            alpha = tf.nn.softplus(scaled_logits) + eps

        # 3. Apply Alpha Cap (Safety Ceiling)
        if self._alpha_cap is not None:
            alpha_cap = tf.cast(self._alpha_cap, alpha.dtype)
            alpha = tf.minimum(alpha, alpha_cap)

        # Ensure strictly positive
        return tf.maximum(alpha, tf.cast(1e-6, alpha.dtype))


class TCNActor(DirichletActor):
    """
    Temporal Convolutional Network Actor.
    
    Uses dilated causal convolutions for efficient sequence processing.
    
    Input shape: (batch, timesteps, features)
    Output shape: (batch, num_actions) - Dirichlet alphas
    """
    
    def __init__(
        self,
        input_dim: int,
        num_actions: int,
        tcn_filters: List[int] = None,
        kernel_size: int = None,
        dilations: List[int] = None,
        dropout: float = None,
        name: str = "tcn_actor",
        epsilon_start: float = 0.5,
        epsilon_min: float = 0.1,
        alpha_activation: str = None,
        exp_clip: Tuple[float, float] = None,
        logit_temperature: float = None,
        alpha_cap: float = None,
        adaptive_temperature_enabled: bool = False,
        adaptive_temperature_base: float = 1.0,
        adaptive_temperature_slope: float = 0.0,
        adaptive_temperature_min: float = 0.8,
        adaptive_temperature_max: float = 2.5,
    ):
        # Apply config defaults
        if tcn_filters is None:
            tcn_filters = _DEFAULT_TCN_FILTERS
        if kernel_size is None:
            kernel_size = _DEFAULT_TCN_KERNEL_SIZE
        if dilations is None:
            dilations = _DEFAULT_TCN_DILATIONS
        if dropout is None:
            dropout = _DEFAULT_TCN_DROPOUT
        
        super(TCNActor, self).__init__(
            name=name,
            epsilon_start=epsilon_start,
            epsilon_min=epsilon_min,
            alpha_activation=alpha_activation,
            exp_clip=exp_clip,
            logit_temperature=logit_temperature,
            alpha_cap=alpha_cap,
            adaptive_temperature_enabled=adaptive_temperature_enabled,
            adaptive_temperature_base=adaptive_temperature_base,
            adaptive_temperature_slope=adaptive_temperature_slope,
            adaptive_temperature_min=adaptive_temperature_min,
            adaptive_temperature_max=adaptive_temperature_max,
        )
        
        self.input_dim = input_dim
        self.num_actions = num_actions
        
        # Build TCN blocks
        self.tcn_blocks = []
        for i, (filters, dilation) in enumerate(zip(tcn_filters, dilations)):
            self.tcn_blocks.append(
                TCNBlock(
                    filters=filters,
                    kernel_size=kernel_size,
                    dilation_rate=dilation,
                    dropout=dropout,
                    name=f'{name}_tcn_{i}'
                )
            )
        
        # Global pooling
        self.global_pool = layers.GlobalAveragePooling1D()
        
        # Output layer
        self.output_layer = layers.Dense(
            num_actions,
            activation=None,
            kernel_initializer='orthogonal',
            bias_initializer=tf.keras.initializers.RandomNormal(mean=0.5, stddev=0.3),  # 🔥 FIX: Initialize away from zero
            name=f'{name}_output'
        )
        
    def call(self, state, training=None):
        """
        Args:
            state: (batch, timesteps, features)
            
        Returns:
            alpha: (batch, num_actions)
        """
        x = _flatten_structured_sequence_input(state)
        
        # TCN processing
        for block in self.tcn_blocks:
            x = block(x, training=training)
        
        # x is now (batch, timesteps, tcn_filters[-1])
        
        # Aggregate sequence
        x = self.global_pool(x)  # (batch, tcn_filters[-1])
        
        # Output
        logits = self.output_layer(x, training=training)
        return self._compute_alpha(logits)


class TCNAttentionActor(DirichletActor):
    """
    TCN + Multi-Head Self-Attention Actor.
    
    Combines TCN's efficient convolutions with attention mechanism.
    
    Input shape: (batch, timesteps, features)
    Output shape: (batch, num_actions) - Dirichlet alphas
    """
    
    def __init__(
        self,
        input_dim: int,
        num_actions: int,
        tcn_filters: List[int] = None,
        kernel_size: int = None,
        dilations: List[int] = None,
        attention_heads: int = None,
        attention_dim: int = None,
        dropout: float = None,
        name: str = "tcn_attention_actor",
        epsilon_start: float = 0.5,
        epsilon_min: float = 0.1,
        alpha_activation: str = None,
        exp_clip: Tuple[float, float] = None,
        logit_temperature: float = None,
        alpha_cap: float = None,
        adaptive_temperature_enabled: bool = False,
        adaptive_temperature_base: float = 1.0,
        adaptive_temperature_slope: float = 0.0,
        adaptive_temperature_min: float = 0.8,
        adaptive_temperature_max: float = 2.5,
    ):
        # Apply config defaults
        if tcn_filters is None:
            tcn_filters = _DEFAULT_TCN_FILTERS
        if kernel_size is None:
            kernel_size = _DEFAULT_TCN_KERNEL_SIZE
        if dilations is None:
            dilations = _DEFAULT_TCN_DILATIONS
        if attention_heads is None:
            attention_heads = _DEFAULT_ATTENTION_HEADS
        if attention_dim is None:
            attention_dim = _DEFAULT_ATTENTION_DIM
        if dropout is None:
            dropout = _DEFAULT_TCN_DROPOUT
        
        super(TCNAttentionActor, self).__init__(
            name=name,
            epsilon_start=epsilon_start,
            epsilon_min=epsilon_min,
            alpha_activation=alpha_activation,
            exp_clip=exp_clip,
            logit_temperature=logit_temperature,
            alpha_cap=alpha_cap,
            adaptive_temperature_enabled=adaptive_temperature_enabled,
            adaptive_temperature_base=adaptive_temperature_base,
            adaptive_temperature_slope=adaptive_temperature_slope,
            adaptive_temperature_min=adaptive_temperature_min,
            adaptive_temperature_max=adaptive_temperature_max,
        )
        
        self.input_dim = input_dim
        self.num_actions = num_actions
        
        # TCN blocks
        self.tcn_blocks = []
        for i, (filters, dilation) in enumerate(zip(tcn_filters, dilations)):
            self.tcn_blocks.append(
                TCNBlock(
                    filters=filters,
                    kernel_size=kernel_size,
                    dilation_rate=dilation,
                    dropout=dropout,
                    name=f'{name}_tcn_{i}'
                )
            )
        
        # Project to attention dimension
        self.projection = layers.Dense(attention_dim, name=f'{name}_projection')
        
        # Attention
        self.attention = MultiHeadSelfAttention(
            d_model=attention_dim,
            num_heads=attention_heads,
            dropout=dropout,
            name=f'{name}_attention'
        )
        
        # Global pooling
        self.global_pool = layers.GlobalAveragePooling1D()
        
        # Output layer
        self.output_layer = layers.Dense(
            num_actions,
            activation=None,
            kernel_initializer='orthogonal',
            bias_initializer=tf.keras.initializers.Constant(0.5),  # 🔥 FIX: Initialize away from zero
            name=f'{name}_output'
        )
        
    def call(self, state, training=None):
        """
        Args:
            state: (batch, timesteps, features)
            
        Returns:
            alpha: (batch, num_actions)
        """
        x = _flatten_structured_sequence_input(state)
        
        # TCN processing
        for block in self.tcn_blocks:
            x = block(x, training=training)
        
        # Project to attention dimension
        x = self.projection(x)  # (batch, timesteps, attention_dim)
        
        # Apply attention
        x = self.attention(x, training=training)  # (batch, timesteps, attention_dim)
        
        # Aggregate sequence
        x = self.global_pool(x)  # (batch, attention_dim)
        
        # Output
        logits = self.output_layer(x, training=training)
        return self._compute_alpha(logits)


class TCNFusionActor(DirichletActor):
    """
    Hierarchical fusion actor:
    - Per-asset temporal encoding (shared TCN encoder)
    - Cross-asset attention
    - Global context branch
    - Learnable gated fusion

    Input shape: (batch, timesteps, features)
    Output shape: (batch, num_actions) - Dirichlet alphas
    """

    def __init__(
        self,
        input_dim: int,
        num_actions: int,
        tcn_filters: List[int] = None,
        kernel_size: int = None,
        dilations: List[int] = None,
        dropout: float = None,
        num_assets: int = None,
        asset_feature_dim: int = None,
        global_feature_dim: int = None,
        fusion_embed_dim: int = None,
        fusion_attention_heads: int = None,
        fusion_dropout: float = None,
        fusion_cross_asset_mixer_enabled: Optional[bool] = None,
        fusion_cross_asset_mixer_layers: Optional[int] = None,
        fusion_cross_asset_mixer_expansion: Optional[float] = None,
        fusion_cross_asset_mixer_dropout: Optional[float] = None,
        fusion_alpha_head_hidden_dims: Optional[List[int]] = None,
        fusion_alpha_head_dropout: Optional[float] = None,
        name: str = "tcn_fusion_actor",
        epsilon_start: float = 0.5,
        epsilon_min: float = 0.1,
        alpha_activation: str = None,
        exp_clip: Tuple[float, float] = None,
        logit_temperature: float = None,
        alpha_cap: float = None,
        adaptive_temperature_enabled: bool = False,
        adaptive_temperature_base: float = 1.0,
        adaptive_temperature_slope: float = 0.0,
        adaptive_temperature_min: float = 0.8,
        adaptive_temperature_max: float = 2.5,
    ):
        if tcn_filters is None:
            tcn_filters = _DEFAULT_TCN_FILTERS
        if kernel_size is None:
            kernel_size = _DEFAULT_TCN_KERNEL_SIZE
        if dilations is None:
            dilations = _DEFAULT_TCN_DILATIONS
        if dropout is None:
            dropout = _DEFAULT_TCN_DROPOUT
        if fusion_embed_dim is None:
            fusion_embed_dim = _DEFAULT_FUSION_EMBED_DIM
        if fusion_attention_heads is None:
            fusion_attention_heads = _DEFAULT_FUSION_HEADS
        if fusion_dropout is None:
            fusion_dropout = _DEFAULT_FUSION_DROPOUT
        if fusion_cross_asset_mixer_enabled is None:
            fusion_cross_asset_mixer_enabled = _DEFAULT_FUSION_CROSS_ASSET_MIXER_ENABLED
        if fusion_cross_asset_mixer_layers is None:
            fusion_cross_asset_mixer_layers = _DEFAULT_FUSION_CROSS_ASSET_MIXER_LAYERS
        if fusion_cross_asset_mixer_expansion is None:
            fusion_cross_asset_mixer_expansion = _DEFAULT_FUSION_CROSS_ASSET_MIXER_EXPANSION
        if fusion_cross_asset_mixer_dropout is None:
            fusion_cross_asset_mixer_dropout = _DEFAULT_FUSION_CROSS_ASSET_MIXER_DROPOUT
        if fusion_alpha_head_hidden_dims is None:
            fusion_alpha_head_hidden_dims = _DEFAULT_FUSION_ALPHA_HEAD_HIDDEN_DIMS
        if fusion_alpha_head_dropout is None:
            fusion_alpha_head_dropout = _DEFAULT_FUSION_ALPHA_HEAD_DROPOUT

        super(TCNFusionActor, self).__init__(
            name=name,
            epsilon_start=epsilon_start,
            epsilon_min=epsilon_min,
            alpha_activation=alpha_activation,
            exp_clip=exp_clip,
            logit_temperature=logit_temperature,
            alpha_cap=alpha_cap,
            adaptive_temperature_enabled=adaptive_temperature_enabled,
            adaptive_temperature_base=adaptive_temperature_base,
            adaptive_temperature_slope=adaptive_temperature_slope,
            adaptive_temperature_min=adaptive_temperature_min,
            adaptive_temperature_max=adaptive_temperature_max,
        )

        self.input_dim = int(input_dim)
        self.num_actions = int(num_actions)
        self.num_assets = int(num_assets) if num_assets is not None else max(1, self.num_actions - 1)
        if asset_feature_dim is not None and int(asset_feature_dim) > 0:
            self.per_asset_dim = int(asset_feature_dim)
        else:
            self.per_asset_dim = int(np.ceil(self.input_dim / max(1, self.num_assets)))
        self.local_flat_dim = self.per_asset_dim * self.num_assets
        if global_feature_dim is not None:
            self.global_feature_dim = max(0, int(global_feature_dim))
        else:
            self.global_feature_dim = max(0, int(self.input_dim) - self.local_flat_dim)
        self.expected_input_dim = self.local_flat_dim + self.global_feature_dim
        self.fusion_embed_dim = int(fusion_embed_dim)

        if self.fusion_embed_dim % fusion_attention_heads != 0:
            fusion_attention_heads = 1
        self.fusion_attention_heads = int(fusion_attention_heads)

        self.asset_tcn_blocks = []
        for i, (filters, dilation) in enumerate(zip(tcn_filters, dilations)):
            self.asset_tcn_blocks.append(
                TCNBlock(
                    filters=filters,
                    kernel_size=kernel_size,
                    dilation_rate=dilation,
                    dropout=dropout,
                    name=f"{name}_asset_tcn_{i}",
                )
            )

        self.asset_time_pool = layers.GlobalAveragePooling1D()
        self.asset_projection = layers.Dense(self.fusion_embed_dim, activation="relu", name=f"{name}_asset_projection")
        self.asset_attention = MultiHeadSelfAttention(
            d_model=self.fusion_embed_dim,
            num_heads=self.fusion_attention_heads,
            dropout=fusion_dropout,
            name=f"{name}_asset_attention",
        )
        self.asset_mixer_blocks: List[CrossAssetMixerBlock] = []
        self.cross_asset_mixer_enabled = bool(fusion_cross_asset_mixer_enabled)
        if self.cross_asset_mixer_enabled:
            mixer_layers = max(1, int(fusion_cross_asset_mixer_layers))
            mixer_dropout = float(fusion_cross_asset_mixer_dropout)
            mixer_expansion = float(fusion_cross_asset_mixer_expansion)
            for i in range(mixer_layers):
                self.asset_mixer_blocks.append(
                    CrossAssetMixerBlock(
                        d_model=self.fusion_embed_dim,
                        num_heads=self.fusion_attention_heads,
                        expansion=mixer_expansion,
                        dropout=mixer_dropout,
                        name=f"{name}_asset_mixer_{i}",
                    )
                )
        self.asset_pool = layers.GlobalAveragePooling1D()

        self.global_time_pool = layers.GlobalAveragePooling1D()
        self.global_projection = layers.Dense(self.fusion_embed_dim, activation="relu", name=f"{name}_global_projection")
        self.global_dropout = layers.Dropout(fusion_dropout)

        self.gate_layer = layers.Dense(self.fusion_embed_dim, activation="sigmoid", name=f"{name}_gate")
        sanitized_alpha_head_dims = [int(x) for x in (fusion_alpha_head_hidden_dims or []) if int(x) > 0]
        self.use_richer_alpha_head = len(sanitized_alpha_head_dims) > 0
        self.alpha_pre_norm = None
        self.alpha_head_blocks: List[Tuple[layers.Dense, layers.Dropout]] = []
        if self.use_richer_alpha_head:
            self.alpha_pre_norm = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_alpha_pre_norm")
            for i, hidden_units in enumerate(sanitized_alpha_head_dims):
                self.alpha_head_blocks.append(
                    (
                        layers.Dense(hidden_units, activation="gelu", name=f"{name}_alpha_head_{i}"),
                        layers.Dropout(float(fusion_alpha_head_dropout), name=f"{name}_alpha_dropout_{i}"),
                    )
                )
        self.output_layer = layers.Dense(
            self.num_actions,
            activation=None,
            kernel_initializer="orthogonal",
            bias_initializer=tf.keras.initializers.Constant(0.5),
            name=f"{name}_output",
        )

    def _align_feature_dim(self, x: tf.Tensor) -> tf.Tensor:
        """Pad/slice dynamic feature width so local/context split stays valid."""
        current_dim = tf.shape(x)[-1]
        pad_dim = tf.maximum(0, self.expected_input_dim - current_dim)
        x = tf.pad(x, [[0, 0], [0, 0], [0, pad_dim]])
        return x[:, :, : self.expected_input_dim]

    def _align_asset_tensor(self, asset_tensor: tf.Tensor) -> tf.Tensor:
        """Pad/slice asset and feature axes to deterministic fusion dimensions."""
        x_assets = _to_tensor_with_cast(asset_tensor, tf.float32)
        if x_assets.shape.rank != 4:
            raise ValueError(
                f"TCNFusionActor expects asset tensor rank=4 (batch,time,assets,features), got shape {x_assets.shape}"
            )

        pad_assets = tf.maximum(0, self.num_assets - tf.shape(x_assets)[2])
        x_assets = tf.pad(x_assets, [[0, 0], [0, 0], [0, pad_assets], [0, 0]])
        x_assets = x_assets[:, :, : self.num_assets, :]

        pad_feat = tf.maximum(0, self.per_asset_dim - tf.shape(x_assets)[-1])
        x_assets = tf.pad(x_assets, [[0, 0], [0, 0], [0, 0], [0, pad_feat]])
        x_assets = x_assets[:, :, :, : self.per_asset_dim]
        return x_assets

    def _align_context_tensor(
        self,
        context_tensor: tf.Tensor,
        *,
        batch: tf.Tensor,
        timesteps: tf.Tensor,
        fallback: tf.Tensor,
    ) -> tf.Tensor:
        """Pad/slice context tensor to (batch, timesteps, global_feature_dim)."""
        target_dim = int(self.global_feature_dim)
        if target_dim <= 0:
            return fallback

        if context_tensor is None:
            return tf.zeros((batch, timesteps, target_dim), dtype=fallback.dtype)

        context = _to_tensor_with_cast(context_tensor, fallback.dtype)
        if context.shape.rank == 2:
            context = tf.expand_dims(context, axis=1)
        elif context.shape.rank != 3:
            raise ValueError(
                f"TCNFusionActor expects context rank=2 or 3, got shape {context.shape}"
            )

        pad_time = tf.maximum(0, timesteps - tf.shape(context)[1])
        context = tf.pad(context, [[0, 0], [0, pad_time], [0, 0]])
        context = context[:, :timesteps, :]

        pad_dim = tf.maximum(0, target_dim - tf.shape(context)[-1])
        context = tf.pad(context, [[0, 0], [0, 0], [0, pad_dim]])
        context = context[:, :, :target_dim]
        return context

    def call(self, state, training=None):
        if isinstance(state, dict):
            structured_assets = self._align_asset_tensor(state.get("asset"))
            batch = tf.shape(structured_assets)[0]
            timesteps = tf.shape(structured_assets)[1]
            x_assets = tf.transpose(structured_assets, perm=[0, 2, 1, 3])
            x_assets = tf.reshape(x_assets, (-1, timesteps, self.per_asset_dim))

            context_seq = self._align_context_tensor(
                state.get("context"),
                batch=batch,
                timesteps=timesteps,
                fallback=tf.zeros((batch, timesteps, self.per_asset_dim), dtype=structured_assets.dtype),
            )
        else:
            x = self._align_feature_dim(state)
            batch = tf.shape(x)[0]
            timesteps = tf.shape(x)[1]
            x_local = x[:, :, : self.local_flat_dim]

            # (batch, timesteps, local_flat_dim) -> (batch*num_assets, timesteps, per_asset_dim)
            x_assets = tf.reshape(x_local, (batch, timesteps, self.num_assets, self.per_asset_dim))
            x_assets = tf.transpose(x_assets, perm=[0, 2, 1, 3])
            x_assets = tf.reshape(x_assets, (-1, timesteps, self.per_asset_dim))

            x_context = x[:, :, self.local_flat_dim:]
            if self.global_feature_dim <= 0:
                context_seq = x_local
            else:
                context_seq = self._align_context_tensor(
                    x_context,
                    batch=batch,
                    timesteps=timesteps,
                    fallback=x_local,
                )

        for block in self.asset_tcn_blocks:
            x_assets = block(x_assets, training=training)

        x_assets = self.asset_time_pool(x_assets)
        x_assets = self.asset_projection(x_assets)
        x_assets = tf.reshape(x_assets, (batch, self.num_assets, self.fusion_embed_dim))
        x_assets = self.asset_attention(x_assets, training=training)
        for mixer_block in self.asset_mixer_blocks:
            x_assets = mixer_block(x_assets, training=training)
        asset_context = self.asset_pool(x_assets)

        global_context = self.global_time_pool(context_seq)
        global_context = self.global_projection(global_context)
        global_context = self.global_dropout(global_context, training=training)

        gate = self.gate_layer(tf.concat([asset_context, global_context], axis=-1))
        fused = gate * asset_context + (1.0 - gate) * global_context

        if self.use_richer_alpha_head and self.alpha_pre_norm is not None:
            alpha_features = self.alpha_pre_norm(fused)
            for dense_layer, dropout_layer in self.alpha_head_blocks:
                alpha_features = dense_layer(alpha_features)
                alpha_features = dropout_layer(alpha_features, training=training)
        else:
            alpha_features = fused

        logits = self.output_layer(alpha_features, training=training)
        return self._compute_alpha(logits)


# ============================================================================
# CRITIC NETWORKS
# ============================================================================

class TCNCritic(Model):
    """
    TCN-based Critic.
    
    Input shape: (batch, timesteps, features)
    Output shape: (batch, 1) - State value
    """
    
    def __init__(self,
                 input_dim: int,
                 tcn_filters: List[int] = None,
                 kernel_size: int = None,
                 dilations: List[int] = None,
                 dropout: float = None,
                 name: str = "tcn_critic"):
        super(TCNCritic, self).__init__(name=name)
        
        # Apply config defaults
        if tcn_filters is None:
            tcn_filters = _DEFAULT_TCN_FILTERS
        if kernel_size is None:
            kernel_size = _DEFAULT_TCN_KERNEL_SIZE
        if dilations is None:
            dilations = _DEFAULT_TCN_DILATIONS
        if dropout is None:
            dropout = _DEFAULT_TCN_DROPOUT
        
        self.input_dim = input_dim
        
        # TCN blocks
        self.tcn_blocks = []
        for i, (filters, dilation) in enumerate(zip(tcn_filters, dilations)):
            self.tcn_blocks.append(
                TCNBlock(
                    filters=filters,
                    kernel_size=kernel_size,
                    dilation_rate=dilation,
                    dropout=dropout,
                    name=f'{name}_tcn_{i}'
                )
            )
        
        # Global pooling
        self.global_pool = layers.GlobalAveragePooling1D()
        
        # Output layer
        self.output_layer = layers.Dense(
            1,
            activation=None,
            kernel_initializer='orthogonal',
            name=f'{name}_output'
        )
    
    def call(self, state, training=None):
        """
        Args:
            state: (batch, timesteps, features)
            
        Returns:
            value: (batch, 1)
        """
        x = _flatten_structured_sequence_input(state)
        
        for block in self.tcn_blocks:
            x = block(x, training=training)
        
        x = self.global_pool(x)
        value = self.output_layer(x, training=training)
        
        return value


class TCNAttentionCritic(Model):
    """
    TCN + Attention Critic.
    
    Input shape: (batch, timesteps, features)
    Output shape: (batch, 1) - State value
    """
    
    def __init__(self,
                 input_dim: int,
                 tcn_filters: List[int] = None,
                 kernel_size: int = None,
                 dilations: List[int] = None,
                 attention_heads: int = None,
                 attention_dim: int = None,
                 dropout: float = None,
                 name: str = "tcn_attention_critic"):
        super(TCNAttentionCritic, self).__init__(name=name)
        
        # Apply config defaults
        if tcn_filters is None:
            tcn_filters = _DEFAULT_TCN_FILTERS
        if kernel_size is None:
            kernel_size = _DEFAULT_TCN_KERNEL_SIZE
        if dilations is None:
            dilations = _DEFAULT_TCN_DILATIONS
        if attention_heads is None:
            attention_heads = _DEFAULT_ATTENTION_HEADS
        if attention_dim is None:
            attention_dim = _DEFAULT_ATTENTION_DIM
        if dropout is None:
            dropout = _DEFAULT_TCN_DROPOUT
        
        self.input_dim = input_dim
        
        # TCN blocks
        self.tcn_blocks = []
        for i, (filters, dilation) in enumerate(zip(tcn_filters, dilations)):
            self.tcn_blocks.append(
                TCNBlock(
                    filters=filters,
                    kernel_size=kernel_size,
                    dilation_rate=dilation,
                    dropout=dropout,
                    name=f'{name}_tcn_{i}'
                )
            )
        
        # Projection
        self.projection = layers.Dense(attention_dim, name=f'{name}_projection')
        
        # Attention
        self.attention = MultiHeadSelfAttention(
            d_model=attention_dim,
            num_heads=attention_heads,
            dropout=dropout,
            name=f'{name}_attention'
        )
        
        # Global pooling
        self.global_pool = layers.GlobalAveragePooling1D()
        
        # Output layer
        self.output_layer = layers.Dense(
            1,
            activation=None,
            kernel_initializer='orthogonal',
            name=f'{name}_output'
        )
    
    def call(self, state, training=None):
        """
        Args:
            state: (batch, timesteps, features)
            
        Returns:
            value: (batch, 1)
        """
        x = _flatten_structured_sequence_input(state)
        
        for block in self.tcn_blocks:
            x = block(x, training=training)
        
        x = self.projection(x)
        x = self.attention(x, training=training)
        x = self.global_pool(x)
        
        value = self.output_layer(x, training=training)
        
        return value


class TCNFusionCritic(Model):
    """
    Critic counterpart of TCNFusionActor with shared fusion strategy.
    """

    def __init__(
        self,
        input_dim: int,
        tcn_filters: List[int] = None,
        kernel_size: int = None,
        dilations: List[int] = None,
        dropout: float = None,
        num_assets: int = None,
        asset_feature_dim: int = None,
        global_feature_dim: int = None,
        fusion_embed_dim: int = None,
        fusion_attention_heads: int = None,
        fusion_dropout: float = None,
        fusion_cross_asset_mixer_enabled: Optional[bool] = None,
        fusion_cross_asset_mixer_layers: Optional[int] = None,
        fusion_cross_asset_mixer_expansion: Optional[float] = None,
        fusion_cross_asset_mixer_dropout: Optional[float] = None,
        name: str = "tcn_fusion_critic",
    ):
        super(TCNFusionCritic, self).__init__(name=name)

        if tcn_filters is None:
            tcn_filters = _DEFAULT_TCN_FILTERS
        if kernel_size is None:
            kernel_size = _DEFAULT_TCN_KERNEL_SIZE
        if dilations is None:
            dilations = _DEFAULT_TCN_DILATIONS
        if dropout is None:
            dropout = _DEFAULT_TCN_DROPOUT
        if fusion_embed_dim is None:
            fusion_embed_dim = _DEFAULT_FUSION_EMBED_DIM
        if fusion_attention_heads is None:
            fusion_attention_heads = _DEFAULT_FUSION_HEADS
        if fusion_dropout is None:
            fusion_dropout = _DEFAULT_FUSION_DROPOUT
        if fusion_cross_asset_mixer_enabled is None:
            fusion_cross_asset_mixer_enabled = _DEFAULT_FUSION_CROSS_ASSET_MIXER_ENABLED
        if fusion_cross_asset_mixer_layers is None:
            fusion_cross_asset_mixer_layers = _DEFAULT_FUSION_CROSS_ASSET_MIXER_LAYERS
        if fusion_cross_asset_mixer_expansion is None:
            fusion_cross_asset_mixer_expansion = _DEFAULT_FUSION_CROSS_ASSET_MIXER_EXPANSION
        if fusion_cross_asset_mixer_dropout is None:
            fusion_cross_asset_mixer_dropout = _DEFAULT_FUSION_CROSS_ASSET_MIXER_DROPOUT

        self.input_dim = int(input_dim)
        self.num_assets = int(num_assets) if num_assets is not None else 5
        if asset_feature_dim is not None and int(asset_feature_dim) > 0:
            self.per_asset_dim = int(asset_feature_dim)
        else:
            self.per_asset_dim = int(np.ceil(self.input_dim / max(1, self.num_assets)))
        self.local_flat_dim = self.per_asset_dim * self.num_assets
        if global_feature_dim is not None:
            self.global_feature_dim = max(0, int(global_feature_dim))
        else:
            self.global_feature_dim = max(0, int(self.input_dim) - self.local_flat_dim)
        self.expected_input_dim = self.local_flat_dim + self.global_feature_dim
        self.fusion_embed_dim = int(fusion_embed_dim)

        if self.fusion_embed_dim % fusion_attention_heads != 0:
            fusion_attention_heads = 1
        self.fusion_attention_heads = int(fusion_attention_heads)

        self.asset_tcn_blocks = []
        for i, (filters, dilation) in enumerate(zip(tcn_filters, dilations)):
            self.asset_tcn_blocks.append(
                TCNBlock(
                    filters=filters,
                    kernel_size=kernel_size,
                    dilation_rate=dilation,
                    dropout=dropout,
                    name=f"{name}_asset_tcn_{i}",
                )
            )

        self.asset_time_pool = layers.GlobalAveragePooling1D()
        self.asset_projection = layers.Dense(self.fusion_embed_dim, activation="relu", name=f"{name}_asset_projection")
        self.asset_attention = MultiHeadSelfAttention(
            d_model=self.fusion_embed_dim,
            num_heads=self.fusion_attention_heads,
            dropout=fusion_dropout,
            name=f"{name}_asset_attention",
        )
        self.asset_mixer_blocks: List[CrossAssetMixerBlock] = []
        self.cross_asset_mixer_enabled = bool(fusion_cross_asset_mixer_enabled)
        if self.cross_asset_mixer_enabled:
            mixer_layers = max(1, int(fusion_cross_asset_mixer_layers))
            mixer_dropout = float(fusion_cross_asset_mixer_dropout)
            mixer_expansion = float(fusion_cross_asset_mixer_expansion)
            for i in range(mixer_layers):
                self.asset_mixer_blocks.append(
                    CrossAssetMixerBlock(
                        d_model=self.fusion_embed_dim,
                        num_heads=self.fusion_attention_heads,
                        expansion=mixer_expansion,
                        dropout=mixer_dropout,
                        name=f"{name}_asset_mixer_{i}",
                    )
                )
        self.asset_pool = layers.GlobalAveragePooling1D()

        self.global_time_pool = layers.GlobalAveragePooling1D()
        self.global_projection = layers.Dense(self.fusion_embed_dim, activation="relu", name=f"{name}_global_projection")
        self.global_dropout = layers.Dropout(fusion_dropout)
        self.gate_layer = layers.Dense(self.fusion_embed_dim, activation="sigmoid", name=f"{name}_gate")
        self.output_layer = layers.Dense(1, activation=None, kernel_initializer="orthogonal", name=f"{name}_output")

    def _align_feature_dim(self, x: tf.Tensor) -> tf.Tensor:
        current_dim = tf.shape(x)[-1]
        pad_dim = tf.maximum(0, self.expected_input_dim - current_dim)
        x = tf.pad(x, [[0, 0], [0, 0], [0, pad_dim]])
        return x[:, :, : self.expected_input_dim]

    def _align_asset_tensor(self, asset_tensor: tf.Tensor) -> tf.Tensor:
        x_assets = _to_tensor_with_cast(asset_tensor, tf.float32)
        if x_assets.shape.rank != 4:
            raise ValueError(
                f"TCNFusionCritic expects asset tensor rank=4 (batch,time,assets,features), got shape {x_assets.shape}"
            )
        pad_assets = tf.maximum(0, self.num_assets - tf.shape(x_assets)[2])
        x_assets = tf.pad(x_assets, [[0, 0], [0, 0], [0, pad_assets], [0, 0]])
        x_assets = x_assets[:, :, : self.num_assets, :]

        pad_feat = tf.maximum(0, self.per_asset_dim - tf.shape(x_assets)[-1])
        x_assets = tf.pad(x_assets, [[0, 0], [0, 0], [0, 0], [0, pad_feat]])
        x_assets = x_assets[:, :, :, : self.per_asset_dim]
        return x_assets

    def _align_context_tensor(
        self,
        context_tensor: tf.Tensor,
        *,
        batch: tf.Tensor,
        timesteps: tf.Tensor,
        fallback: tf.Tensor,
    ) -> tf.Tensor:
        target_dim = int(self.global_feature_dim)
        if target_dim <= 0:
            return fallback

        if context_tensor is None:
            return tf.zeros((batch, timesteps, target_dim), dtype=fallback.dtype)

        context = _to_tensor_with_cast(context_tensor, fallback.dtype)
        if context.shape.rank == 2:
            context = tf.expand_dims(context, axis=1)
        elif context.shape.rank != 3:
            raise ValueError(
                f"TCNFusionCritic expects context rank=2 or 3, got shape {context.shape}"
            )

        pad_time = tf.maximum(0, timesteps - tf.shape(context)[1])
        context = tf.pad(context, [[0, 0], [0, pad_time], [0, 0]])
        context = context[:, :timesteps, :]

        pad_dim = tf.maximum(0, target_dim - tf.shape(context)[-1])
        context = tf.pad(context, [[0, 0], [0, 0], [0, pad_dim]])
        context = context[:, :, :target_dim]
        return context

    def call(self, state, training=None):
        if isinstance(state, dict):
            structured_assets = self._align_asset_tensor(state.get("asset"))
            batch = tf.shape(structured_assets)[0]
            timesteps = tf.shape(structured_assets)[1]
            x_assets = tf.transpose(structured_assets, perm=[0, 2, 1, 3])
            x_assets = tf.reshape(x_assets, (-1, timesteps, self.per_asset_dim))

            context_seq = self._align_context_tensor(
                state.get("context"),
                batch=batch,
                timesteps=timesteps,
                fallback=tf.zeros((batch, timesteps, self.per_asset_dim), dtype=structured_assets.dtype),
            )
        else:
            x = self._align_feature_dim(state)
            batch = tf.shape(x)[0]
            timesteps = tf.shape(x)[1]
            x_local = x[:, :, : self.local_flat_dim]

            x_assets = tf.reshape(x_local, (batch, timesteps, self.num_assets, self.per_asset_dim))
            x_assets = tf.transpose(x_assets, perm=[0, 2, 1, 3])
            x_assets = tf.reshape(x_assets, (-1, timesteps, self.per_asset_dim))

            x_context = x[:, :, self.local_flat_dim:]
            if self.global_feature_dim <= 0:
                context_seq = x_local
            else:
                context_seq = self._align_context_tensor(
                    x_context,
                    batch=batch,
                    timesteps=timesteps,
                    fallback=x_local,
                )

        for block in self.asset_tcn_blocks:
            x_assets = block(x_assets, training=training)

        x_assets = self.asset_time_pool(x_assets)
        x_assets = self.asset_projection(x_assets)
        x_assets = tf.reshape(x_assets, (batch, self.num_assets, self.fusion_embed_dim))
        x_assets = self.asset_attention(x_assets, training=training)
        for mixer_block in self.asset_mixer_blocks:
            x_assets = mixer_block(x_assets, training=training)
        asset_context = self.asset_pool(x_assets)

        global_context = self.global_time_pool(context_seq)
        global_context = self.global_projection(global_context)
        global_context = self.global_dropout(global_context, training=training)

        gate = self.gate_layer(tf.concat([asset_context, global_context], axis=-1))
        fused = gate * asset_context + (1.0 - gate) * global_context
        return self.output_layer(fused, training=training)


# ============================================================================
# ARCHITECTURE FACTORY
# ============================================================================

def _resolve_dirichlet_epsilon_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract Dirichlet-related defaults from config."""
    epsilon_cfg = config.get("dirichlet_epsilon") or {}
    epsilon_start = float(epsilon_cfg.get("max", epsilon_cfg.get("start", 0.5)))
    epsilon_min = float(epsilon_cfg.get("min", 0.1))
    alpha_activation = str(config.get("dirichlet_alpha_activation", _DEFAULT_ALPHA_ACTIVATION))
    exp_clip_cfg = config.get("dirichlet_exp_clip", (-5.0, 3.0))
    try:
        exp_clip = (float(exp_clip_cfg[0]), float(exp_clip_cfg[1]))
    except Exception:
        exp_clip = (-5.0, 3.0)
    return {
        "epsilon_start": epsilon_start,
        "epsilon_min": epsilon_min,
        "alpha_activation": alpha_activation,
        "exp_clip": exp_clip,
        # New parameters
        "logit_temperature": float(config.get("dirichlet_logit_temperature", 1.0)),
        "alpha_cap": float(config.get("dirichlet_alpha_cap", 100.0)) if "dirichlet_alpha_cap" in config else None,
        # Optional adaptive temperature controller (disabled by default).
        "adaptive_temperature_enabled": bool(config.get("dirichlet_adaptive_temperature_enabled", False)),
        "adaptive_temperature_base": float(
            config.get("dirichlet_adaptive_temperature_base", config.get("dirichlet_logit_temperature", 1.0))
        ),
        "adaptive_temperature_slope": float(config.get("dirichlet_adaptive_temperature_slope", 0.0)),
        "adaptive_temperature_min": float(config.get("dirichlet_adaptive_temperature_min", 0.8)),
        "adaptive_temperature_max": float(config.get("dirichlet_adaptive_temperature_max", 2.5)),
    }


def create_actor_critic(architecture: str,
                       input_dim: int,
                       num_actions: int,
                       config: dict) -> Tuple[Model, Model]:
    """
    Factory function to create Actor and Critic networks based on architecture type.
    
    Args:
        architecture: One of ['TCN', 'TCN_ATTENTION', 'TCN_FUSION']
        input_dim: Input dimension for sequential models
        num_actions: Number of actions (assets + cash)
        config: Configuration dictionary with architecture-specific parameters
        
    Returns:
        Tuple of (actor_network, critic_network)
    """
    arch_upper = architecture.upper()
    epsilon_kwargs = _resolve_dirichlet_epsilon_kwargs(config)
    state_layout = config.get("state_layout", {}) if isinstance(config.get("state_layout", {}), dict) else {}
    resolved_asset_feature_dim = state_layout.get("asset_feature_dim", config.get("asset_feature_dim"))
    resolved_global_feature_dim = state_layout.get("global_feature_dim", config.get("global_feature_dim"))
    if arch_upper == 'TCN':
        if config.get('use_fusion', False):
            resolved_num_assets = int(config.get('num_assets', max(1, num_actions - 1)))
            actor = TCNFusionActor(
                input_dim=input_dim,
                num_actions=num_actions,
                tcn_filters=config.get('tcn_filters', _DEFAULT_TCN_FILTERS),
                kernel_size=config.get('tcn_kernel_size', _DEFAULT_TCN_KERNEL_SIZE),
                dilations=config.get('tcn_dilations', _DEFAULT_TCN_DILATIONS),
                dropout=config.get('tcn_dropout', _DEFAULT_TCN_DROPOUT),
                num_assets=resolved_num_assets,
                asset_feature_dim=resolved_asset_feature_dim,
                global_feature_dim=resolved_global_feature_dim,
                fusion_embed_dim=config.get('fusion_embed_dim', _DEFAULT_FUSION_EMBED_DIM),
                fusion_attention_heads=config.get('fusion_attention_heads', _DEFAULT_FUSION_HEADS),
                fusion_dropout=config.get('fusion_dropout', _DEFAULT_FUSION_DROPOUT),
                fusion_cross_asset_mixer_enabled=config.get('fusion_cross_asset_mixer_enabled', _DEFAULT_FUSION_CROSS_ASSET_MIXER_ENABLED),
                fusion_cross_asset_mixer_layers=config.get('fusion_cross_asset_mixer_layers', _DEFAULT_FUSION_CROSS_ASSET_MIXER_LAYERS),
                fusion_cross_asset_mixer_expansion=config.get('fusion_cross_asset_mixer_expansion', _DEFAULT_FUSION_CROSS_ASSET_MIXER_EXPANSION),
                fusion_cross_asset_mixer_dropout=config.get('fusion_cross_asset_mixer_dropout', _DEFAULT_FUSION_CROSS_ASSET_MIXER_DROPOUT),
                fusion_alpha_head_hidden_dims=config.get('fusion_alpha_head_hidden_dims', _DEFAULT_FUSION_ALPHA_HEAD_HIDDEN_DIMS),
                fusion_alpha_head_dropout=config.get('fusion_alpha_head_dropout', _DEFAULT_FUSION_ALPHA_HEAD_DROPOUT),
                **epsilon_kwargs,
            )
            critic = TCNFusionCritic(
                input_dim=input_dim,
                tcn_filters=config.get('tcn_filters', _DEFAULT_TCN_FILTERS),
                kernel_size=config.get('tcn_kernel_size', _DEFAULT_TCN_KERNEL_SIZE),
                dilations=config.get('tcn_dilations', _DEFAULT_TCN_DILATIONS),
                dropout=config.get('tcn_dropout', _DEFAULT_TCN_DROPOUT),
                num_assets=resolved_num_assets,
                asset_feature_dim=resolved_asset_feature_dim,
                global_feature_dim=resolved_global_feature_dim,
                fusion_embed_dim=config.get('fusion_embed_dim', _DEFAULT_FUSION_EMBED_DIM),
                fusion_attention_heads=config.get('fusion_attention_heads', _DEFAULT_FUSION_HEADS),
                fusion_dropout=config.get('fusion_dropout', _DEFAULT_FUSION_DROPOUT),
                fusion_cross_asset_mixer_enabled=config.get('fusion_cross_asset_mixer_enabled', _DEFAULT_FUSION_CROSS_ASSET_MIXER_ENABLED),
                fusion_cross_asset_mixer_layers=config.get('fusion_cross_asset_mixer_layers', _DEFAULT_FUSION_CROSS_ASSET_MIXER_LAYERS),
                fusion_cross_asset_mixer_expansion=config.get('fusion_cross_asset_mixer_expansion', _DEFAULT_FUSION_CROSS_ASSET_MIXER_EXPANSION),
                fusion_cross_asset_mixer_dropout=config.get('fusion_cross_asset_mixer_dropout', _DEFAULT_FUSION_CROSS_ASSET_MIXER_DROPOUT),
            )
        elif config.get('use_attention', False):
            actor = TCNAttentionActor(
                input_dim=input_dim,
                num_actions=num_actions,
                tcn_filters=config.get('tcn_filters', _DEFAULT_TCN_FILTERS),
                kernel_size=config.get('tcn_kernel_size', _DEFAULT_TCN_KERNEL_SIZE),
                dilations=config.get('tcn_dilations', _DEFAULT_TCN_DILATIONS),
                attention_heads=config.get('attention_heads', _DEFAULT_ATTENTION_HEADS),
                attention_dim=config.get('attention_dim', _DEFAULT_ATTENTION_DIM),
                dropout=config.get('tcn_dropout', _DEFAULT_TCN_DROPOUT),
                **epsilon_kwargs,
            )
            critic = TCNAttentionCritic(
                input_dim=input_dim,
                tcn_filters=config.get('tcn_filters', _DEFAULT_TCN_FILTERS),
                kernel_size=config.get('tcn_kernel_size', _DEFAULT_TCN_KERNEL_SIZE),
                dilations=config.get('tcn_dilations', _DEFAULT_TCN_DILATIONS),
                attention_heads=config.get('attention_heads', _DEFAULT_ATTENTION_HEADS),
                attention_dim=config.get('attention_dim', _DEFAULT_ATTENTION_DIM),
                dropout=config.get('tcn_dropout', _DEFAULT_TCN_DROPOUT)
            )
        else:
            actor = TCNActor(
                input_dim=input_dim,
                num_actions=num_actions,
                tcn_filters=config.get('tcn_filters', _DEFAULT_TCN_FILTERS),
                kernel_size=config.get('tcn_kernel_size', _DEFAULT_TCN_KERNEL_SIZE),
                dilations=config.get('tcn_dilations', _DEFAULT_TCN_DILATIONS),
                dropout=config.get('tcn_dropout', _DEFAULT_TCN_DROPOUT),
                **epsilon_kwargs,
            )
            critic = TCNCritic(
                input_dim=input_dim,
                tcn_filters=config.get('tcn_filters', _DEFAULT_TCN_FILTERS),
                kernel_size=config.get('tcn_kernel_size', _DEFAULT_TCN_KERNEL_SIZE),
                dilations=config.get('tcn_dilations', _DEFAULT_TCN_DILATIONS),
                dropout=config.get('tcn_dropout', _DEFAULT_TCN_DROPOUT)
            )

    elif arch_upper == 'TCN_FUSION':
        resolved_num_assets = int(config.get('num_assets', max(1, num_actions - 1)))
        actor = TCNFusionActor(
            input_dim=input_dim,
            num_actions=num_actions,
            tcn_filters=config.get('tcn_filters', _DEFAULT_TCN_FILTERS),
            kernel_size=config.get('tcn_kernel_size', _DEFAULT_TCN_KERNEL_SIZE),
            dilations=config.get('tcn_dilations', _DEFAULT_TCN_DILATIONS),
            dropout=config.get('tcn_dropout', _DEFAULT_TCN_DROPOUT),
            num_assets=resolved_num_assets,
            asset_feature_dim=resolved_asset_feature_dim,
            global_feature_dim=resolved_global_feature_dim,
            fusion_embed_dim=config.get('fusion_embed_dim', _DEFAULT_FUSION_EMBED_DIM),
            fusion_attention_heads=config.get('fusion_attention_heads', _DEFAULT_FUSION_HEADS),
            fusion_dropout=config.get('fusion_dropout', _DEFAULT_FUSION_DROPOUT),
            fusion_cross_asset_mixer_enabled=config.get('fusion_cross_asset_mixer_enabled', _DEFAULT_FUSION_CROSS_ASSET_MIXER_ENABLED),
            fusion_cross_asset_mixer_layers=config.get('fusion_cross_asset_mixer_layers', _DEFAULT_FUSION_CROSS_ASSET_MIXER_LAYERS),
            fusion_cross_asset_mixer_expansion=config.get('fusion_cross_asset_mixer_expansion', _DEFAULT_FUSION_CROSS_ASSET_MIXER_EXPANSION),
            fusion_cross_asset_mixer_dropout=config.get('fusion_cross_asset_mixer_dropout', _DEFAULT_FUSION_CROSS_ASSET_MIXER_DROPOUT),
            fusion_alpha_head_hidden_dims=config.get('fusion_alpha_head_hidden_dims', _DEFAULT_FUSION_ALPHA_HEAD_HIDDEN_DIMS),
            fusion_alpha_head_dropout=config.get('fusion_alpha_head_dropout', _DEFAULT_FUSION_ALPHA_HEAD_DROPOUT),
            **epsilon_kwargs,
        )
        critic = TCNFusionCritic(
            input_dim=input_dim,
            tcn_filters=config.get('tcn_filters', _DEFAULT_TCN_FILTERS),
            kernel_size=config.get('tcn_kernel_size', _DEFAULT_TCN_KERNEL_SIZE),
            dilations=config.get('tcn_dilations', _DEFAULT_TCN_DILATIONS),
            dropout=config.get('tcn_dropout', _DEFAULT_TCN_DROPOUT),
            num_assets=resolved_num_assets,
            asset_feature_dim=resolved_asset_feature_dim,
            global_feature_dim=resolved_global_feature_dim,
            fusion_embed_dim=config.get('fusion_embed_dim', _DEFAULT_FUSION_EMBED_DIM),
            fusion_attention_heads=config.get('fusion_attention_heads', _DEFAULT_FUSION_HEADS),
            fusion_dropout=config.get('fusion_dropout', _DEFAULT_FUSION_DROPOUT),
            fusion_cross_asset_mixer_enabled=config.get('fusion_cross_asset_mixer_enabled', _DEFAULT_FUSION_CROSS_ASSET_MIXER_ENABLED),
            fusion_cross_asset_mixer_layers=config.get('fusion_cross_asset_mixer_layers', _DEFAULT_FUSION_CROSS_ASSET_MIXER_LAYERS),
            fusion_cross_asset_mixer_expansion=config.get('fusion_cross_asset_mixer_expansion', _DEFAULT_FUSION_CROSS_ASSET_MIXER_EXPANSION),
            fusion_cross_asset_mixer_dropout=config.get('fusion_cross_asset_mixer_dropout', _DEFAULT_FUSION_CROSS_ASSET_MIXER_DROPOUT),
        )

    elif arch_upper == 'TCN_ATTENTION':
        actor = TCNAttentionActor(
            input_dim=input_dim,
            num_actions=num_actions,
            tcn_filters=config.get('tcn_filters', _DEFAULT_TCN_FILTERS),
            kernel_size=config.get('tcn_kernel_size', _DEFAULT_TCN_KERNEL_SIZE),
            dilations=config.get('tcn_dilations', _DEFAULT_TCN_DILATIONS),
            attention_heads=config.get('attention_heads', _DEFAULT_ATTENTION_HEADS),
            attention_dim=config.get('attention_dim', _DEFAULT_ATTENTION_DIM),
            dropout=config.get('tcn_dropout', _DEFAULT_TCN_DROPOUT),
            **epsilon_kwargs,
        )
        critic = TCNAttentionCritic(
            input_dim=input_dim,
            tcn_filters=config.get('tcn_filters', _DEFAULT_TCN_FILTERS),
            kernel_size=config.get('tcn_kernel_size', _DEFAULT_TCN_KERNEL_SIZE),
            dilations=config.get('tcn_dilations', _DEFAULT_TCN_DILATIONS),
            attention_heads=config.get('attention_heads', _DEFAULT_ATTENTION_HEADS),
            attention_dim=config.get('attention_dim', _DEFAULT_ATTENTION_DIM),
            dropout=config.get('tcn_dropout', _DEFAULT_TCN_DROPOUT)
        )
    
    else:
        raise ValueError(
            f"Unknown architecture: {architecture}. "
            f"Must be one of: TCN, TCN_ATTENTION, TCN_FUSION"
        )
    
    return actor, critic
