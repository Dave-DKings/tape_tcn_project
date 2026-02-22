"""
PPO Agent Implementation for Portfolio Optimization

This module implements the Proximal Policy Optimization (PPO) algorithm
specifically designed for portfolio optimization using Dirichlet distributions.

The agent uses:
- Dirichlet distribution for action space (portfolio weights)
- Generalized Advantage Estimation (GAE) for advantage calculation
- Clipped surrogate objective for stable policy updates
- Separate actor and critic networks
"""

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from collections import deque
import logging
import sys
import os
import math

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.actor_critic_tf import create_actor_critic
from config import is_sequential_architecture

# Set up logging
logger = logging.getLogger(__name__)

tfd = tfp.distributions


def _to_tensor_with_cast(value, dtype=None):
    """Convert to tensor and cast dtype explicitly (safe with mixed precision tensors)."""
    tensor = tf.convert_to_tensor(value)
    if dtype is not None:
        target_dtype = tf.dtypes.as_dtype(dtype)
        if tensor.dtype != target_dtype:
            tensor = tf.cast(tensor, target_dtype)
    return tensor


class RunningMeanStd:
    """Track running mean and variance for streaming normalization."""

    def __init__(self, epsilon: float = 1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon

    def update(self, x):
        """Update running statistics with a new batch."""
        x = np.asarray(x, dtype=np.float64)
        if x.size == 0:
            return

        batch_mean = np.mean(x)
        batch_var = np.var(x)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / total_count
        new_var = m2 / total_count

        self.mean = float(new_mean)
        self.var = float(max(new_var, 1e-8))
        self.count = float(total_count)

    @property
    def std(self):
        """Return the running standard deviation."""
        return math.sqrt(self.var + 1e-8)

    def normalize(self, x):
        """Normalize array using current running statistics."""
        return (np.asarray(x) - self.mean) / (self.std)


class PPOAgentTF:
    """
    PPO Agent for Portfolio Optimization using Dirichlet distributions.
    
    This agent is specifically designed for portfolio optimization where:
    1. Actions are portfolio weights that must sum to 1
    2. All weights must be non-negative
    3. The action space is naturally constrained to the probability simplex
    
    The Dirichlet distribution handles these constraints naturally while
    allowing for exploration and smooth policy updates.
    """
    
    def __init__(self, 
                 state_dim,
                 num_assets,
                 config,
                 name="PPOAgent"):
        """
        Initialize the PPO agent with architecture-agnostic design.
        
        Args:
            state_dim (int): Dimension of the state space (features)
            num_assets (int): Number of assets (excluding cash)
            config (dict): Configuration dictionary with all agent parameters
            name (str): Name of the agent
        """
        
        self.state_dim = state_dim
        self.num_assets = num_assets
        self.num_actions = num_assets + 1  # assets + cash
        self.name = name
        
        # Extract architecture type
        self.architecture = config.get('actor_critic_type', 'TCN')
        self.is_sequential = is_sequential_architecture(self.architecture)
        
        # Sequential model parameters
        if self.is_sequential:
            self.sequence_length = config.get('sequence_length', 30)
            logger.info(f"Sequential architecture: {self.architecture}, sequence_length={self.sequence_length}")
            self.state_history = deque(maxlen=self.sequence_length)
        else:
            self.sequence_length = None
            self.state_history = None
            logger.info(f"Non-sequential architecture: {self.architecture}")
        self._latest_sequence = None

        # Optional structured layout metadata for fusion architectures.
        state_layout = config.get("state_layout", {}) if isinstance(config.get("state_layout", {}), dict) else {}
        self.state_layout = state_layout
        self.asset_feature_dim = int(state_layout.get("asset_feature_dim", 0) or 0)
        self.global_feature_dim = int(state_layout.get("global_feature_dim", 0) or 0)
        self.local_flat_dim = int(state_layout.get("local_flat_dim", self.num_assets * max(self.asset_feature_dim, 0)) or 0)
        self.structured_observation = bool(state_layout.get("structured_observation", False))
        self.uses_structured_state_inputs = bool(
            self.is_sequential
            and self.structured_observation
            and self.asset_feature_dim > 0
        )
        # Backward-compatible alias kept for notebook conditionals.
        self.uses_structured_fusion_inputs = self.uses_structured_state_inputs

        # Dirichlet exploration annealing defaults
        epsilon_cfg = config.get("dirichlet_epsilon") or {}
        self.dirichlet_epsilon_max = float(epsilon_cfg.get("max", epsilon_cfg.get("start", 0.5)))
        self.dirichlet_epsilon_min = float(epsilon_cfg.get("min", 0.1))
        self._dirichlet_progress = 0.0
        training_cfg = config.get("training_params")
        training_timesteps = 0
        if isinstance(training_cfg, dict):
            training_timesteps = training_cfg.get("max_total_timesteps", 0)
        self.max_total_timesteps = int(config.get("max_total_timesteps", training_timesteps or 0))
        self._global_step = 0
        
        # PPO hyperparameters
        ppo_params = config.get('ppo_params', {})
        self.gamma = ppo_params.get('gamma', 0.99)
        self.gae_lambda = ppo_params.get('gae_lambda', 0.9)
        self.policy_clip = ppo_params.get('policy_clip', 0.2)
        self.entropy_coef = ppo_params.get('entropy_coef', 0.01)
        self.vf_coef = ppo_params.get('vf_coef', 0.5)
        self.max_grad_norm = ppo_params.get('max_grad_norm', 0.5)
        self.value_clip_range = ppo_params.get('value_clip', 0.2)
        self.target_kl = float(ppo_params.get('target_kl', 0.03))
        self.kl_stop_multiplier = float(ppo_params.get('kl_stop_multiplier', 1.5))
        self.minibatches_before_kl_stop = int(ppo_params.get('minibatches_before_kl_stop', 2))

        # Optional risk-aware actor auxiliary losses (disabled by default).
        # These are additive regularizers intended to improve risk-adjusted robustness.
        self.use_risk_aux_loss = bool(ppo_params.get('use_risk_aux_loss', False))
        self.risk_aux_return_feature_index = int(ppo_params.get('risk_aux_return_feature_index', 0))
        self.risk_aux_cash_return = float(ppo_params.get('risk_aux_cash_return', 0.0))
        self.risk_aux_sharpe_coef = float(ppo_params.get('risk_aux_sharpe_coef', 0.0))
        self.risk_aux_mvo_coef = float(ppo_params.get('risk_aux_mvo_coef', 0.0))
        self.risk_aux_mvo_cov_ridge = float(ppo_params.get('risk_aux_mvo_cov_ridge', 1e-3))
        self.risk_aux_mvo_long_only = bool(ppo_params.get('risk_aux_mvo_long_only', True))
        self.risk_aux_mvo_risky_budget = float(
            np.clip(ppo_params.get('risk_aux_mvo_risky_budget', 0.95), 0.0, 1.0)
        )
        
        # Create networks using architecture factory
        logger.info(f"Creating {self.architecture} actor-critic networks...")
        self.actor, self.critic = create_actor_critic(
            architecture=self.architecture,
            input_dim=state_dim,
            num_actions=self.num_actions,
            config=config
        )
        
        # Create optimizers
        actor_lr = ppo_params.get('actor_lr', 3e-3)
        critic_lr = ppo_params.get('critic_lr', 3e-3)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)
        self._current_actor_lr = float(actor_lr)
        
        # Memory for storing trajectory data
        self.memory = {
            'states': [],
            'actions': [],
            'log_probs': [],
            'rewards': [],
            'values': [],
            'dones': []
        }
        
        logger.info(f"Initialized {name}")
        logger.info(f"  State dim: {state_dim}, Num assets: {num_assets}, Actions: {self.num_actions}")
        logger.info(f"  Architecture: {self.architecture} (Sequential: {self.is_sequential})")
        if self.uses_structured_state_inputs:
            logger.info(
                "  Structured state reshape enabled: assets=%d, asset_feature_dim=%d, global_feature_dim=%d",
                self.num_assets,
                self.asset_feature_dim,
                self.global_feature_dim,
            )
        logger.info(
            f"  PPO params: γ={self.gamma}, λ={self.gae_lambda}, clip={self.policy_clip}, "
            f"value_clip={self.value_clip_range}, target_kl={self.target_kl:.4f}"
        )
        logger.info(
            "  Risk aux: enabled=%s, sharpe_coef=%.4f, mvo_coef=%.4f, return_feature_idx=%d",
            self.use_risk_aux_loss,
            self.risk_aux_sharpe_coef,
            self.risk_aux_mvo_coef,
            self.risk_aux_return_feature_index,
        )
        logger.info(
            f"  Learning rates (init): actor={self.get_actor_lr():.6f}, critic={self.get_critic_lr():.6f}"
        )
        logger.info(f"  Networks created: {self.actor.name}, {self.critic.name}")

        # Initialize epsilon schedule at maximum exploration
        self.set_dirichlet_progress(0.0)

        # Running statistics for critic target normalization
        self.returns_rms = RunningMeanStd()
        self._returns_mean = 0.0
        self._returns_std = 1.0
        # Running statistics for reward normalization
        self.reward_rms = RunningMeanStd()
        self._reward_mean = 0.0
        self._reward_std = 1.0
        # Debug/diagnostic verbosity toggle
        self.debug_prints = bool(config.get('debug_prints', False))

    def get_actor_lr(self) -> float:
        """Return the actor optimizer learning rate as a python float."""
        try:
            return float(tf.keras.backend.get_value(self.actor_optimizer.learning_rate))
        except Exception:
            return float(self._current_actor_lr)

    def set_actor_lr(self, new_lr: float) -> None:
        """Update the actor optimizer learning rate in-place."""
        new_lr = float(new_lr)
        if hasattr(self.actor_optimizer.learning_rate, "assign"):
            self.actor_optimizer.learning_rate.assign(new_lr)
        else:
            self.actor_optimizer.learning_rate = new_lr
        self._current_actor_lr = new_lr

    def get_critic_lr(self) -> float:
        """Return the critic optimizer learning rate as a python float."""
        try:
            return float(tf.keras.backend.get_value(self.critic_optimizer.learning_rate))
        except Exception:
            return float(self.critic_optimizer.learning_rate)

    def set_dirichlet_progress(self, progress: float) -> None:
        """Adjust the actor's Dirichlet epsilon according to normalized progress."""
        if not hasattr(self.actor, "update_dirichlet_epsilon"):
            return
        if np.isnan(progress):
            progress = 0.0
        progress = float(np.clip(progress, 0.0, 1.0))
        self._dirichlet_progress = progress
        self.actor.update_dirichlet_epsilon(
            progress,
            self.dirichlet_epsilon_min,
            self.dirichlet_epsilon_max,
        )

    def _split_flat_state_array(self, state_array):
        """
        Split flat states into per-asset and global-context tensors using state_layout metadata.
        """
        if self.asset_feature_dim <= 0:
            raise ValueError("asset_feature_dim must be > 0 for structured fusion input.")

        arr = np.asarray(state_array, dtype=np.float32)
        if arr.ndim < 1:
            raise ValueError(f"Unsupported state array shape for structured split: {arr.shape}")

        expected_dim = self.local_flat_dim + max(0, self.global_feature_dim)
        current_dim = int(arr.shape[-1])
        if current_dim < expected_dim:
            pad_width = [(0, 0)] * arr.ndim
            pad_width[-1] = (0, expected_dim - current_dim)
            arr = np.pad(arr, pad_width, mode="constant", constant_values=0.0)
        elif current_dim > expected_dim:
            arr = arr[..., :expected_dim]

        local = arr[..., :self.local_flat_dim]
        local_shape = local.shape[:-1] + (self.num_assets, self.asset_feature_dim)
        local = np.reshape(local, local_shape)

        if self.global_feature_dim > 0:
            context = arr[..., self.local_flat_dim:self.local_flat_dim + self.global_feature_dim]
        else:
            context = np.zeros(arr.shape[:-1] + (0,), dtype=np.float32)

        return local.astype(np.float32), context.astype(np.float32)

    def _structured_state_to_tensor_input(self, state):
        """
        Convert flat or dict state into fusion-ready tensor dict.
        """
        if isinstance(state, dict):
            asset = state.get("asset")
            context = state.get("context")
            if asset is None:
                raise ValueError("Structured state dict must contain key 'asset'.")
            asset = _to_tensor_with_cast(asset, tf.float32)
            context = _to_tensor_with_cast(context, tf.float32) if context is not None else None
        else:
            if isinstance(state, tf.Tensor):
                state = state.numpy()
            asset_np, context_np = self._split_flat_state_array(state)
            asset = _to_tensor_with_cast(asset_np, tf.float32)
            context = _to_tensor_with_cast(context_np, tf.float32)

        return {"asset": asset, "context": context}

    def _convert_states_for_network(self, states):
        """
        Convert stored rollout states into actor/critic network inputs.
        """
        if not self.uses_structured_state_inputs:
            return tf.constant(states, dtype=tf.float32)

        asset_np, context_np = self._split_flat_state_array(states)
        return {
            "asset": tf.constant(asset_np, dtype=tf.float32),
            "context": tf.constant(context_np, dtype=tf.float32),
        }
    
    def prepare_state_input(self, state):
        """
        Prepare state tensor with architecture-specific handling.
        
        Args:
            state: Input state (numpy array or tensor)
                - Sequential: (timesteps, features) or (batch, timesteps, features)
        
        Returns:
            Tuple of (prepared_state, needs_squeeze)
            - prepared_state: TensorFlow tensor with batch dimension
            - needs_squeeze: Boolean indicating if output should be squeezed
        """
        if self.is_sequential and self.uses_structured_state_inputs:
            structured = self._structured_state_to_tensor_input(state)
            asset = structured["asset"]
            context = structured["context"]

            if len(asset.shape) == 3:
                # (timesteps, assets, features) -> (1, timesteps, assets, features)
                asset = tf.expand_dims(asset, axis=0)
                if context is None:
                    context = tf.zeros((1, tf.shape(asset)[1], self.global_feature_dim), dtype=asset.dtype)
                elif len(context.shape) == 2:
                    context = tf.expand_dims(context, axis=0)
                elif len(context.shape) == 1:
                    context = tf.expand_dims(tf.expand_dims(context, axis=0), axis=0)
                return {"asset": asset, "context": context}, True

            if len(asset.shape) == 4:
                # Already batched
                if context is None:
                    batch = tf.shape(asset)[0]
                    steps = tf.shape(asset)[1]
                    context = tf.zeros((batch, steps, self.global_feature_dim), dtype=asset.dtype)
                elif len(context.shape) == 2:
                    context = tf.expand_dims(context, axis=1)
                return {"asset": asset, "context": context}, False

            raise ValueError(
                f"Structured sequential state expects asset rank 3 or 4, got shape {asset.shape}"
            )
        
        # Convert to tensor if needed
        if not isinstance(state, tf.Tensor):
            state = tf.constant(state, dtype=tf.float32)
        
        current_ndim = len(state.shape)
        
        if self.is_sequential:
            # Sequential models expect: (batch, timesteps, features)
            if current_ndim == 2:
                # (timesteps, features) → (1, timesteps, features)
                return tf.expand_dims(state, axis=0), True  # needs_squeeze
            elif current_ndim == 3:
                # Already batched: (batch, timesteps, features)
                return state, False
            else:
                raise ValueError(
                    f"Sequential architecture expects 2D or 3D input, got {current_ndim}D with shape {state.shape}"
                )
        else:
            # Non-sequential expects: (batch, features)
            if current_ndim == 1:
                # (features,) → (1, features)
                return tf.expand_dims(state, axis=0), True  # needs_squeeze
            elif current_ndim == 2:
                # Already batched: (batch, features)
                return state, False
            else:
                raise ValueError(
                    f"Non-sequential architecture expects 1D or 2D input, got {current_ndim}D with shape {state.shape}"
                )
    
    def reset_state_history(self):
        """Reset the state history (e.g., at the start of a new episode)."""
        if self.state_history is not None:
            self.state_history.clear()
        self._latest_sequence = None

    def _build_sequence(self, state):
        """
        Build a temporal sequence for sequential architectures.
        
        Args:
            state: Current state vector or pre-built sequence.
        
        Returns:
            np.ndarray of shape (sequence_length, state_dim)
        """
        if not self.is_sequential:
            raise RuntimeError("Attempted to build sequence for non-sequential architecture.")
        
        if isinstance(state, tf.Tensor):
            state = state.numpy()
        state = np.asarray(state, dtype=np.float32)
        
        if state.ndim == 2:
            sequence = state
        elif state.ndim == 1:
            self.state_history.append(state)
            sequence = np.array(list(self.state_history), dtype=np.float32)
        else:
            raise ValueError(f"Unsupported state shape for sequential architecture: {state.shape}")
        
        if sequence.shape[0] > self.sequence_length:
            sequence = sequence[-self.sequence_length:]
        elif sequence.shape[0] < self.sequence_length:
            pad_len = self.sequence_length - sequence.shape[0]
            if sequence.shape[0] > 0:
                pad_value = sequence[0:1]
            else:
                pad_value = np.zeros((1, self.state_dim), dtype=np.float32)
            padding = np.repeat(pad_value, pad_len, axis=0)
            sequence = np.vstack([padding, sequence])
        
        # Refresh history to match the returned sequence
        self.state_history.clear()
        for row in sequence:
            self.state_history.append(row)
        
        return sequence
    
    def get_action_and_value(self, state, deterministic=False, stochastic=False, evaluation_mode='mean_plus_noise'):
        """
        Get action and value estimate for a given state.
        
        Args:
            state: Current state (various shapes supported)
            deterministic: If True, use deterministic evaluation strategy
            stochastic: If True, force stochastic sampling (overrides deterministic)
            evaluation_mode: Strategy for deterministic evaluation ('mean', 'mode', 'mean_plus_noise')
            
        Returns:
            Tuple of (action, log_prob, value)
        """
        # Prepare state input and check if we need to squeeze output
        if self.is_sequential:
            sequence = self._build_sequence(state)
            self._latest_sequence = np.array(sequence, copy=True)
            state_input, needs_squeeze = self.prepare_state_input(sequence)
        else:
            self._latest_sequence = None
            state_input, needs_squeeze = self.prepare_state_input(state)
        
        # Get alpha parameters from actor
        alpha = self.actor(state_input, training=False)
        alpha = _to_tensor_with_cast(alpha, tf.float32)
        
        # 🔥 CRITICAL FIX: Ensure alpha > 0 for Dirichlet distribution
        # This must happen in BOTH get_action_and_value() and _actor_loss()
        # because get_action_and_value() is called during rollout collection
        # BEFORE any loss calculation occurs
        alpha = tf.maximum(alpha, tf.constant(1e-6, dtype=alpha.dtype))
        
        # Create Dirichlet distribution
        dirichlet = tfd.Dirichlet(alpha)
        
        # ✅ FIX: stochastic parameter forces sampling even when not training
        if stochastic:
            # Force stochastic sampling for Monte Carlo evaluation
            action = dirichlet.sample()
        elif deterministic:
            # Deterministic evaluation strategies
            if evaluation_mode == 'mean':
                # Use the mean of the Dirichlet distribution
                # For Dirichlet, mean = alpha / sum(alpha)
                action = dirichlet.mean()
                
            elif evaluation_mode == 'mode':
                # Mode for Dirichlet:
                # If alpha > 1: (alpha - 1) / (sum(alpha) - K)
                # If alpha <= 1: Mode is at vertices (argmax)
                
                # Check if all alpha > 1 (per sample)
                min_alpha = tf.reduce_min(alpha, axis=-1, keepdims=True)
                use_formula = min_alpha > 1.0
                
                # Formula for alpha > 1
                sum_alpha = tf.reduce_sum(alpha, axis=-1, keepdims=True)
                k = tf.cast(tf.shape(alpha)[-1], alpha.dtype)
                mode_formula = (alpha - 1.0) / (sum_alpha - k)
                
                # Vertex for alpha <= 1 (argmax)
                max_indices = tf.argmax(alpha, axis=-1)
                mode_vertex = tf.one_hot(
                    max_indices,
                    depth=tf.shape(alpha)[-1],
                    dtype=alpha.dtype,
                )
                
                # Select based on validity
                action = tf.where(use_formula, mode_formula, mode_vertex)
                
            elif evaluation_mode == 'mean_plus_noise':
                # Mean + small noise (epsilon) to break symmetry/flatness
                # Useful when mean is too conservative/flat
                mean_val = dirichlet.mean()
                noise = tf.random.normal(shape=tf.shape(mean_val), mean=0.0, stddev=0.005) # Small epsilon
                action = mean_val + noise
                # Re-normalize and clip
                action = tf.maximum(action, 1e-6)
                action = action / tf.reduce_sum(action, axis=-1, keepdims=True)
                
            else:
                # Default fallback
                action = dirichlet.mean()
                
        else:
            # Sample from the distribution
            action = dirichlet.sample()
        
        # Calculate log probability (then clip to avoid numerical blow-ups)
        log_prob = dirichlet.log_prob(action)
        
        # Get value estimate
        value = self.critic(state_input, training=False)
        value = _to_tensor_with_cast(value, tf.float32)
        
        # Squeeze batch dimension if needed
        if needs_squeeze:
            action = tf.squeeze(action, 0)
            log_prob = tf.squeeze(log_prob, 0)
            value = tf.squeeze(value, 0)
        
        return action, log_prob, value
    
    def store_transition(self, state, action, log_prob, reward, value, done):
        """
        Store a transition in memory with shape normalization.
        
        CRITICAL: Parameter order is log_prob BEFORE reward!
        
        Args:
            state: Current state
            action: Action taken  
            log_prob: Log probability of action (BEFORE reward!)
            reward: Reward received (AFTER log_prob!)
            value: Value estimate
            done: Whether episode is done
        """
        # Normalize shapes for storage
        if self.is_sequential:
            if self._latest_sequence is not None:
                state_to_store = np.array(self._latest_sequence, copy=True)
                self._latest_sequence = None
            else:
                state_to_store = self._build_sequence(state)
        else:
            state_to_store = state

        state = self._normalize_state_shape(state_to_store)
        action = self._normalize_action_shape(action)
        
        # Validate types
        assert isinstance(log_prob, (int, float, np.number, tf.Tensor)), \
            f"log_prob must be numeric, got {type(log_prob)}"
        assert isinstance(reward, (int, float, np.number)), \
            f"reward must be numeric, got {type(reward)}"
        assert isinstance(value, (int, float, np.number, tf.Tensor)), \
            f"value must be numeric, got {type(value)}"
        assert isinstance(done, (bool, np.bool_)), \
            f"done must be boolean, got {type(done)}"
        
        # Convert to Python native types
        if isinstance(log_prob, tf.Tensor):
            log_prob = float(log_prob.numpy())
        if isinstance(value, tf.Tensor):
            value = float(value.numpy())
        
        # Store in memory
        self.memory['states'].append(state)
        self.memory['actions'].append(action)
        self.memory['log_probs'].append(float(log_prob))
        self.memory['rewards'].append(float(reward))
        self.memory['values'].append(float(value))
        self.memory['dones'].append(bool(done))

        self._global_step += 1
        if self.max_total_timesteps > 0:
            self.set_dirichlet_progress(self._global_step / self.max_total_timesteps)
        
        if done:
            self.reset_state_history()
    
    def _normalize_state_shape(self, state):
        """
        Ensure state has correct shape for storage.
        
        Args:
            state: State tensor or array
            
        Returns:
            Normalized state as numpy array
        """
        if isinstance(state, tf.Tensor):
            state = state.numpy()
        
        if self.is_sequential:
            # Sequential: should be (timesteps, features)
            if state.ndim == 3 and state.shape[0] == 1:
                # Remove batch dimension: (1, timesteps, features) → (timesteps, features)
                state = state.squeeze(0)
            elif state.ndim != 2:
                raise ValueError(
                    f"Invalid state shape for sequential architecture: {state.shape}. "
                    f"Expected (timesteps, features) or (1, timesteps, features)"
                )
        else:
            # Non-sequential: should be (features,)
            if state.ndim == 2 and state.shape[0] == 1:
                # Remove batch dimension: (1, features) → (features,)
                state = state.squeeze(0)
            elif state.ndim != 1:
                raise ValueError(
                    f"Invalid state shape for non-sequential architecture: {state.shape}. "
                    f"Expected (features,) or (1, features)"
                )
        
        return state.astype(np.float32)
    
    def _normalize_action_shape(self, action):
        """
        Ensure action has correct shape for storage.
        
        Args:
            action: Action tensor or array
            
        Returns:
            Normalized action as numpy array
        """
        if isinstance(action, tf.Tensor):
            action = action.numpy()
        
        # Actions should always be (num_actions,)
        if action.ndim == 2 and action.shape[0] == 1:
            # Remove batch dimension: (1, num_actions) → (num_actions,)
            action = action.squeeze(0)
        elif action.ndim != 1:
            raise ValueError(f"Invalid action shape: {action.shape}. Expected (num_actions,) or (1, num_actions)")
        
        assert action.shape[0] == self.num_actions, \
            f"Action size mismatch: got {action.shape[0]}, expected {self.num_actions}"
        
        return action.astype(np.float32)
    
    def clear_memory(self):
        """Clear the memory buffer."""
        for key in self.memory:
            # Handle both lists and numpy arrays
            if isinstance(self.memory[key], list):
                self.memory[key].clear()
            else:
                # For numpy arrays or other types, reinitialize as empty list
                self.memory[key] = []
    
    def compute_gae(self, rewards, values, dones, next_value=0.0):
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: List of rewards
            values: List of value estimates
            dones: List of done flags
            next_value: Value estimate for the next state (for bootstrapping)
            
        Returns:
            tuple: (advantages, returns) as numpy arrays
        """
        advantages = []
        gae = 0
        
        # Convert to numpy for easier manipulation
        rewards = np.array(rewards)
        values = np.array(values)
        dones = np.array(dones)
        
        # Add next_value to values for bootstrapping
        values_with_next = np.append(values, next_value)
        
        # Compute GAE backwards
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = next_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value = values_with_next[t + 1]
            
            # TD error
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            
            # GAE
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages.insert(0, gae)
        
        advantages = np.array(advantages)
        returns = advantages + values
        
        # DIAGNOSTIC: Check TD errors to understand advantage computation
        td_errors = []
        for t in range(len(rewards)):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_val = next_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_val = values_with_next[t + 1]
            delta = rewards[t] + self.gamma * next_val * next_non_terminal - values[t]
            td_errors.append(delta)
        td_errors = np.array(td_errors)
        
        # Print TD error statistics (only if called from update, not every forward pass)
        # We'll check this by seeing if we're computing for multiple timesteps
        if self.debug_prints and len(rewards) > 10:
            print(f"\n🎯 TD ERROR DIAGNOSTICS (GAE computation):")
            print(f"   TD errors: min={td_errors.min():.6f}, max={td_errors.max():.6f}")
            print(f"   TD error mean: {np.mean(td_errors):.6f}, std: {np.std(td_errors):.6f}")
            print(f"   Abs TD error mean: {np.mean(np.abs(td_errors)):.6f}")
            if np.std(td_errors) < 0.01:
                print(f"   ⚠️  WARNING: TD errors have very low variance! Critic may be overfitting.")
        
        return advantages, returns
    
    def _extract_asset_return_proxy(self, states):
        """
        Extract a per-asset return proxy from structured states.

        Uses the latest timestep and configured feature index from the per-asset tensor:
          states['asset'] shape: (batch, timesteps, num_assets, asset_feature_dim)
        """
        if not isinstance(states, dict):
            return None

        asset = states.get("asset")
        if asset is None:
            return None

        asset = _to_tensor_with_cast(asset, tf.float32)
        if asset.shape.rank != 4:
            return None
        if asset.shape[-1] == 0:
            return None

        latest = asset[:, -1, :, :]  # (batch, num_assets, feature_dim)
        feature_dim = tf.shape(latest)[-1]
        safe_idx = tf.clip_by_value(
            tf.constant(self.risk_aux_return_feature_index, dtype=tf.int32),
            0,
            tf.maximum(feature_dim - 1, 0),
        )
        return tf.gather(latest, safe_idx, axis=-1)  # (batch, num_assets)

    def _compute_mvo_target_weights(self, asset_returns: tf.Tensor):
        """
        Compute a simple long-only MVO target from batch return proxies.

        Returns:
            target_risky_weights: (num_assets,)
            target_cash_weight: scalar
        """
        asset_returns = _to_tensor_with_cast(asset_returns, tf.float32)
        n_obs = tf.cast(tf.shape(asset_returns)[0], tf.float32)
        n_assets = tf.shape(asset_returns)[1]

        mu = tf.reduce_mean(asset_returns, axis=0)  # (num_assets,)
        centered = asset_returns - mu
        denom = tf.maximum(n_obs - 1.0, 1.0)
        cov = tf.matmul(centered, centered, transpose_a=True) / denom

        ridge = tf.cast(self.risk_aux_mvo_cov_ridge, tf.float32)
        cov_reg = cov + ridge * tf.eye(n_assets, dtype=tf.float32)
        inv_cov = tf.linalg.pinv(cov_reg)

        raw = tf.linalg.matvec(inv_cov, mu)
        if self.risk_aux_mvo_long_only:
            raw = tf.nn.relu(raw)

        eps = tf.constant(1e-8, dtype=tf.float32)
        raw_sum = tf.reduce_sum(raw)
        equal = tf.ones_like(raw) / tf.cast(n_assets, tf.float32)
        normalized = tf.where(raw_sum > eps, raw / (raw_sum + eps), equal)

        risky_budget = tf.constant(self.risk_aux_mvo_risky_budget, dtype=tf.float32)
        target_risky = normalized * risky_budget
        target_cash = 1.0 - risky_budget
        return target_risky, target_cash

    def _compute_risk_aux_loss(self, states, alpha):
        """
        Compute optional risk-aware actor auxiliaries.

        Components:
          - Sharpe surrogate: maximize batch Sharpe of actor-implied one-step proxy returns.
          - MVO regularizer: pull actor risky weights toward a long-only MVO target.
        """
        zero = tf.constant(0.0, dtype=tf.float32)
        if not self.use_risk_aux_loss:
            return zero, zero, zero, zero

        asset_returns = self._extract_asset_return_proxy(states)
        if asset_returns is None:
            return zero, zero, zero, zero

        alpha = _to_tensor_with_cast(alpha, tf.float32)
        weights = alpha / tf.maximum(tf.reduce_sum(alpha, axis=-1, keepdims=True), 1e-8)
        risky_weights = weights[:, :self.num_assets]
        cash_weights = weights[:, self.num_assets]

        sharpe_proxy = tf.constant(0.0, dtype=tf.float32)
        sharpe_loss = tf.constant(0.0, dtype=tf.float32)
        if self.risk_aux_sharpe_coef > 0.0:
            cash_ret = tf.constant(self.risk_aux_cash_return, dtype=tf.float32)
            portfolio_proxy_returns = tf.reduce_sum(risky_weights * asset_returns, axis=-1) + cash_weights * cash_ret
            mu_p = tf.reduce_mean(portfolio_proxy_returns)
            sigma_p = tf.math.reduce_std(portfolio_proxy_returns)
            sharpe_proxy = mu_p / (sigma_p + 1e-6)
            sharpe_loss = -tf.constant(self.risk_aux_sharpe_coef, dtype=tf.float32) * sharpe_proxy

        mvo_loss = tf.constant(0.0, dtype=tf.float32)
        if self.risk_aux_mvo_coef > 0.0:
            target_risky, target_cash = self._compute_mvo_target_weights(asset_returns)
            risky_mse = tf.reduce_mean(tf.square(risky_weights - target_risky[tf.newaxis, :]))
            cash_mse = tf.reduce_mean(tf.square(cash_weights - target_cash))
            mvo_loss = tf.constant(self.risk_aux_mvo_coef, dtype=tf.float32) * (risky_mse + cash_mse)

        total_aux = sharpe_loss + mvo_loss
        return total_aux, sharpe_proxy, sharpe_loss, mvo_loss

    # @tf.function  # DISABLED: Causes weight caching issues with PPO ratio stuck at 1.0
    def _actor_loss(self, states, actions, log_probs_old, advantages):
        """
        Compute the actor loss (PPO clipped objective + entropy bonus).
        
        Args:
            states: Batch of states
            actions: Batch of actions
            log_probs_old: Old log probabilities
            advantages: Advantage estimates
            
        Returns:
            tuple with PPO losses/diagnostics + optional risk-aware auxiliaries
        """
        # Get current policy distribution
        alpha = self.actor(states, training=True)
        alpha = _to_tensor_with_cast(alpha, tf.float32)
        
        # CRITICAL FIX: Ensure alpha > 0 for Dirichlet distribution
        # Dirichlet requires strictly positive parameters
        alpha = tf.maximum(alpha, tf.constant(1e-6, dtype=alpha.dtype))
        
        dirichlet = tfd.Dirichlet(alpha)
        
        # Current log probabilities
        log_probs_new = dirichlet.log_prob(actions)
        
        # Stabilize PPO ratio by clipping the log-probability delta
        log_prob_delta_raw = log_probs_new - log_probs_old
        log_prob_delta_raw = tf.clip_by_value(log_prob_delta_raw, -10.0, 10.0)
        ratio_unclipped = tf.exp(log_prob_delta_raw)
        
        lower_clip = tf.math.log(tf.maximum(1.0 - self.policy_clip, 1e-3))
        upper_clip = tf.math.log(1.0 + self.policy_clip)
        log_prob_delta = tf.clip_by_value(log_prob_delta_raw, lower_clip, upper_clip)
        ratio = tf.exp(log_prob_delta)
        
        # DIAGNOSTIC: Compute ratio statistics before clipping to understand instability
        ratio_mean = tf.reduce_mean(ratio_unclipped)
        ratio_std = tf.math.reduce_std(ratio_unclipped)
        
        # Clipped surrogate objective
        surr1 = ratio * advantages
        surr2 = tf.clip_by_value(ratio, 1.0 - self.policy_clip, 1.0 + self.policy_clip) * advantages
        policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
        
        # Entropy bonus for exploration
        entropy = tf.reduce_mean(dirichlet.entropy())
        entropy_loss = -self.entropy_coef * entropy
        
        risk_aux_total, sharpe_aux_proxy, sharpe_aux_loss, mvo_aux_loss = self._compute_risk_aux_loss(states, alpha)
        total_loss = policy_loss + entropy_loss + risk_aux_total
        
        approx_kl = tf.reduce_mean(log_probs_old - log_probs_new)
        clip_mask = tf.cast(tf.abs(ratio_unclipped - 1.0) > self.policy_clip, tf.float32)
        clip_fraction = tf.reduce_mean(clip_mask)
        
        return (
            total_loss,
            policy_loss,
            entropy_loss,
            entropy,
            ratio_mean,
            ratio_std,
            approx_kl,
            clip_fraction,
            risk_aux_total,
            sharpe_aux_proxy,
            sharpe_aux_loss,
            mvo_aux_loss,
        )
    
    @tf.function(reduce_retracing=True)
    def _critic_loss(self, states, returns, returns_mean, returns_std, old_values=None):
        """
        Compute the critic loss (value function MSE).
        
        Args:
            states: Batch of states
            returns: Target returns
            returns_mean: Running mean of returns
            returns_std: Running std of returns
            old_values: Value predictions collected during rollout (for value clipping)
            
        Returns:
            critic_loss: MSE loss between predicted and target values
        """
        values = self.critic(states, training=True)
        values = _to_tensor_with_cast(values, tf.float32)
        values = tf.squeeze(values, -1)  # Remove last dimension

        returns_mean = _to_tensor_with_cast(returns_mean, tf.float32)
        returns_std = _to_tensor_with_cast(returns_std, tf.float32)
        returns_std = tf.maximum(returns_std, 1e-6)

        returns = _to_tensor_with_cast(returns, tf.float32)
        returns_centered = returns - returns_mean
        returns_norm = returns_centered / returns_std

        values_centered = values - returns_mean
        values_norm = values_centered / returns_std

        clip_fraction = tf.constant(0.0, dtype=tf.float32)
        if (
            self.value_clip_range is not None
            and self.value_clip_range > 0.0
            and old_values is not None
        ):
            old_values = _to_tensor_with_cast(old_values, tf.float32)
            old_values = tf.reshape(old_values, tf.shape(values))
            values_clipped = old_values + tf.clip_by_value(
                values - old_values,
                -self.value_clip_range,
                self.value_clip_range
            )
            values_clipped_norm = (values_clipped - returns_mean) / returns_std
            clip_fraction = tf.reduce_mean(
                tf.cast(tf.abs(values - old_values) > self.value_clip_range, tf.float32)
            )

            loss_unclipped = tf.square(returns_norm - values_norm)
            loss_clipped = tf.square(returns_norm - values_clipped_norm)
            loss = tf.reduce_mean(tf.maximum(loss_unclipped, loss_clipped))
        else:
            # MSE loss on centered values
            loss = tf.reduce_mean(tf.square(returns_norm - values_norm))
        
        return loss, clip_fraction
    
    def update(self, num_epochs=10, batch_size=64):
        """
        Update the actor and critic networks using PPO.
        
        This is the baseline version that uses direct log-return rewards
        without the TAPE reward scaling system.
        
        Args:
            num_epochs: Number of optimization epochs
            batch_size: Batch size for updates
            
        Returns:
            dict: Training statistics
        """
        if len(self.memory['states']) == 0:
            logger.warning("No data in memory for update")
            return {}
        
        # Convert memory to numpy arrays
        states_np = np.array(self.memory['states'])
        actions = np.array(self.memory['actions'])
        log_probs_old = np.array(self.memory['log_probs'])
        rewards = np.array(self.memory['rewards'])
        values = np.array(self.memory['values'])
        dones = np.array(self.memory['dones'])
        
        if self.debug_prints:
            # DIAGNOSTIC: Check reward and value variance
            print(f"\n📊 REWARD & VALUE DIAGNOSTICS:")
            print(f"   Rewards: min={rewards.min():.4f}, max={rewards.max():.4f}")
            print(f"   Reward mean: {np.mean(rewards):.4f}, std: {np.std(rewards):.4f}")
            print(f"   Values: min={values.min():.4f}, max={values.max():.4f}")
            print(f"   Value mean: {np.mean(values):.4f}, std: {np.std(values):.4f}")

        # Apply running z-score normalization to rewards
        raw_rewards = rewards.copy()
        # Use previous running stats to normalize current batch
        prev_mean = self._reward_mean
        prev_std = max(self._reward_std, 1e-1)
        rewards = (raw_rewards - prev_mean) / prev_std
        rewards = np.clip(rewards, -5.0, 5.0)

        # Update running statistics AFTER normalization so the next batch sees updated stats
        self.reward_rms.update(raw_rewards)
        self._reward_mean = float(self.reward_rms.mean)
        self._reward_std = float(max(self.reward_rms.std, 1e-1))

        if self.debug_prints:
            print(f"   Reward running mean: {self._reward_mean:.6f}, std: {self._reward_std:.6f}")
            print(
                f"   Normalized rewards: min={rewards.min():.6f}, max={rewards.max():.6f}, "
                f"mean={np.mean(rewards):.6f}, std={np.std(rewards):.6f}"
            )
        
        # Compute advantages and returns using GAE
        values_old = values.copy()
        advantages, returns = self.compute_gae(rewards, values_old, dones)
        
        # Store raw advantages for diagnostics BEFORE normalization
        raw_advantages = advantages.copy()
        
        # DIAGNOSTIC: Check advantage variance BEFORE normalization
        raw_adv_mean = np.mean(raw_advantages)
        raw_adv_std = np.std(raw_advantages)
        if self.debug_prints:
            print(f"\n🔍 ADVANTAGE DIAGNOSTICS:")
            print(f"   Raw advantages: min={raw_advantages.min():.6f}, max={raw_advantages.max():.6f}")
            print(f"   Raw adv mean: {raw_adv_mean:.6f}, std: {raw_adv_std:.6f}")
            if raw_adv_std < 0.01:
                print(f"   ⚠️  WARNING: Very low std! Normalized advantages will be near zero!")
        
        # Normalize advantages
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        # DIAGNOSTIC: Check advantages AFTER normalization
        if self.debug_prints:
            print(f"   Normalized advantages: min={advantages.min():.6f}, max={advantages.max():.6f}")
            print(f"   Normalized adv mean: {np.mean(advantages):.6f}, std: {np.std(advantages):.6f}")

        # Update running statistics for critic targets before tensor conversion
        self.returns_rms.update(returns)
        self._returns_mean = float(self.returns_rms.mean)
        self._returns_std = float(max(self.returns_rms.std, 1e-6))
        
        # Convert to tensors
        returns_np = returns.copy()

        states = self._convert_states_for_network(states_np)
        actions = tf.constant(actions, dtype=tf.float32)
        log_probs_old = tf.constant(log_probs_old, dtype=tf.float32)
        advantages = tf.constant(advantages, dtype=tf.float32)
        returns = tf.constant(returns, dtype=tf.float32)
        old_values_tf = tf.constant(values_old, dtype=tf.float32)
        # Keep these as tensors so tf.function does not retrace on changing Python scalars.
        returns_mean_tf = tf.constant(self._returns_mean, dtype=tf.float32)
        returns_std_tf = tf.constant(self._returns_std, dtype=tf.float32)
        
        # Training statistics
        stats = {
            'actor_loss': 0.0,
            'critic_loss': 0.0,
            'critic_loss_scaled': 0.0,
            'risk_aux_total': 0.0,
            'risk_aux_sharpe_proxy': 0.0,
            'risk_aux_sharpe_loss': 0.0,
            'risk_aux_mvo_loss': 0.0,
            'policy_loss': 0.0,
            'entropy_loss': 0.0,
            'entropy': 0.0,
            'mean_advantage': float(tf.reduce_mean(advantages)),
            'mean_return': float(tf.reduce_mean(returns)),
            'mean_reward_raw': float(np.mean(raw_rewards)),
            'reward_running_mean': self._reward_mean,
            'reward_running_std': self._reward_std,
            # Diagnostic statistics (raw advantages before normalization)
            'adv_min': float(np.min(raw_advantages)),
            'adv_max': float(np.max(raw_advantages)),
            'adv_mean': float(np.mean(raw_advantages)),
            'adv_std': float(np.std(raw_advantages)),
            'actor_grad_norm': 0.0,
            'critic_grad_norm': 0.0,
            'alpha_min': 0.0,
            'alpha_max': 0.0,
            'alpha_mean': 0.0,
            'alpha_std': 0.0,  # Track alpha diversity for TCN learning
            # NEW: PPO ratio statistics
            'ratio_mean': 0.0,
            'ratio_std': 0.0,
            'approx_kl': 0.0,
            'clip_fraction': 0.0,
            'value_clip_fraction': 0.0,
            # Running statistics for critic normalization
            'returns_running_mean': self._returns_mean,
            'returns_running_std': self._returns_std,
            'num_grad_updates': 0,
            'explained_variance': 0.0,
            'early_stop_kl_triggered': 0.0,
            'early_stop_kl': 0.0,
            'early_stop_epoch': -1.0,
        }
        
        # Multiple epochs of optimization
        if isinstance(states, dict):
            dataset_size = int(states["asset"].shape[0])
        else:
            dataset_size = int(states.shape[0])
        early_stop = False
        for epoch in range(num_epochs):
            if early_stop:
                break
            # Shuffle data
            indices = tf.random.shuffle(tf.range(dataset_size))
            
            # Mini-batch updates
            for start_idx in range(0, dataset_size, batch_size):
                end_idx = min(start_idx + batch_size, dataset_size)
                batch_indices = indices[start_idx:end_idx]
                
                if isinstance(states, dict):
                    batch_states = {
                        "asset": tf.gather(states["asset"], batch_indices),
                        "context": tf.gather(states["context"], batch_indices),
                    }
                else:
                    batch_states = tf.gather(states, batch_indices)
                batch_actions = tf.gather(actions, batch_indices)
                batch_log_probs_old = tf.gather(log_probs_old, batch_indices)
                batch_advantages = tf.gather(advantages, batch_indices)
                batch_returns = tf.gather(returns, batch_indices)
                batch_old_values = tf.gather(old_values_tf, batch_indices)
                
                # Update actor
                with tf.GradientTape() as tape:
                    (
                        actor_loss,
                        policy_loss,
                        entropy_loss,
                        entropy,
                        ratio_mean,
                        ratio_std,
                        approx_kl,
                        clip_fraction,
                        risk_aux_total,
                        sharpe_aux_proxy,
                        sharpe_aux_loss,
                        mvo_aux_loss,
                    ) = self._actor_loss(
                        batch_states, batch_actions, batch_log_probs_old, batch_advantages
                    )
                
                actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
                
                # Compute gradient norm BEFORE clipping for diagnostics
                actor_grad_norm = tf.linalg.global_norm(actor_grads).numpy()
                
                if self.max_grad_norm > 0:
                    actor_grads, _ = tf.clip_by_global_norm(actor_grads, self.max_grad_norm)
                self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
                
                # Update critic
                with tf.GradientTape() as tape:
                    critic_loss_raw, value_clip_fraction = self._critic_loss(
                        batch_states,
                        batch_returns,
                        returns_mean_tf,
                        returns_std_tf,
                        batch_old_values
                    )
                    # Apply configurable value-function coefficient to critic optimization.
                    # This was previously ignored because actor/critic are updated separately.
                    critic_loss = critic_loss_raw * self.vf_coef
                
                critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
                
                # Compute gradient norm BEFORE clipping for diagnostics
                critic_grad_norm = tf.linalg.global_norm(critic_grads).numpy()
                
                if self.max_grad_norm > 0:
                    critic_grads, _ = tf.clip_by_global_norm(critic_grads, self.max_grad_norm)
                self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
                
                # Get alpha statistics from current batch
                alpha_batch = self.actor(batch_states, training=False).numpy()
                
                # Accumulate statistics
                stats['actor_loss'] += float(actor_loss)
                stats['critic_loss'] += float(critic_loss_raw)
                stats['critic_loss_scaled'] += float(critic_loss)
                stats['risk_aux_total'] += float(risk_aux_total)
                stats['risk_aux_sharpe_proxy'] += float(sharpe_aux_proxy)
                stats['risk_aux_sharpe_loss'] += float(sharpe_aux_loss)
                stats['risk_aux_mvo_loss'] += float(mvo_aux_loss)
                stats['policy_loss'] += float(policy_loss)
                stats['entropy_loss'] += float(entropy_loss)
                stats['entropy'] += float(entropy)
                stats['actor_grad_norm'] += float(actor_grad_norm)
                stats['critic_grad_norm'] += float(critic_grad_norm)
                stats['alpha_min'] += float(np.min(alpha_batch))
                stats['alpha_max'] += float(np.max(alpha_batch))
                stats['alpha_mean'] += float(np.mean(alpha_batch))
                stats['alpha_std'] += float(np.std(alpha_batch))  # Track alpha diversity
                stats['ratio_mean'] += float(ratio_mean)
                stats['ratio_std'] += float(ratio_std)
                stats['approx_kl'] += float(approx_kl)
                stats['clip_fraction'] += float(clip_fraction)
                stats['value_clip_fraction'] += float(value_clip_fraction)
                stats['num_grad_updates'] += 1                
                # CRITICAL: Detect NaN/Inf early to prevent cascade failures
                if tf.math.is_nan(actor_loss) or tf.math.is_inf(actor_loss):
                    logger.error(f"❌ CRITICAL: NaN/Inf detected in actor_loss! Training unstable.")
                    logger.error(f"   Policy loss: {policy_loss:.6f}, Entropy loss: {entropy_loss:.6f}")
                    # Return current stats to allow graceful handling
                    break
                
                if tf.math.is_nan(critic_loss_raw) or tf.math.is_inf(critic_loss_raw):
                    logger.error(f"❌ CRITICAL: NaN/Inf detected in critic_loss! Training unstable.")
                    break

                # Early-stop PPO update when KL drift is too high (stability guard).
                approx_kl_value = float(approx_kl)
                if (
                    self.target_kl > 0.0
                    and stats['num_grad_updates'] >= max(self.minibatches_before_kl_stop, 1)
                    and approx_kl_value > (self.target_kl * self.kl_stop_multiplier)
                ):
                    stats['early_stop_kl_triggered'] = 1.0
                    stats['early_stop_kl'] = approx_kl_value
                    stats['early_stop_epoch'] = float(epoch)
                    logger.warning(
                        "⚠️ PPO early-stop: approx_kl %.6f exceeded threshold %.6f (target_kl %.6f × %.2f)",
                        approx_kl_value,
                        self.target_kl * self.kl_stop_multiplier,
                        self.target_kl,
                        self.kl_stop_multiplier,
                    )
                    early_stop = True
                    break
        
        # Average statistics over all updates
        num_updates = stats['num_grad_updates']
        if num_updates > 0:
            for key in ['actor_loss', 'critic_loss', 'critic_loss_scaled',
                       'risk_aux_total', 'risk_aux_sharpe_proxy', 'risk_aux_sharpe_loss', 'risk_aux_mvo_loss',
                       'policy_loss', 'entropy_loss', 'entropy',
                       'actor_grad_norm', 'critic_grad_norm', 'alpha_min', 'alpha_max', 'alpha_mean', 'alpha_std',
                       'ratio_mean', 'ratio_std', 'approx_kl', 'clip_fraction', 'value_clip_fraction']:
                stats[key] /= num_updates
        
        # Remove temporary counter
        del stats['num_grad_updates']
        
        # Clear memory after update
        self.clear_memory()
        
        values_post = self.critic(states, training=False)
        returns_np = np.asarray(returns_np, dtype=np.float32)
        values_post = tf.squeeze(values_post, -1).numpy()
        returns_var = np.var(returns_np)
        if returns_var > 1e-8:
            stats['explained_variance'] = float(
                1.0 - np.var(returns_np - values_post) / (returns_var + 1e-8)
            )
        else:
            stats['explained_variance'] = 0.0

        logger.debug(f"PPO Update completed: {num_updates} mini-batch updates over {num_epochs} epochs")
        
        return stats
    
    def save_models(self, filepath_prefix):
        """
        Save the actor and critic networks.
        
        Args:
            filepath_prefix: Prefix for the saved model files
        """
        actor_path = f"{filepath_prefix}_actor.weights.h5"
        critic_path = f"{filepath_prefix}_critic.weights.h5"
        
        self.actor.save_weights(actor_path)
        self.critic.save_weights(critic_path)
        
        logger.info(f"Models saved to {actor_path} and {critic_path}")
    
    def load_models(self, filepath_prefix):
        """
        Load the actor and critic networks.
        
        Args:
            filepath_prefix: Prefix for the saved model files
        """
        actor_path = f"{filepath_prefix}_actor.weights.h5"
        critic_path = f"{filepath_prefix}_critic.weights.h5"
        
        # Create dummy forward pass to build the models
        if self.is_sequential:
            if self.uses_structured_state_inputs:
                dummy_state = {
                    "asset": tf.zeros(
                        (1, int(self.sequence_length), int(self.num_assets), int(self.asset_feature_dim)),
                        dtype=tf.float32,
                    ),
                    "context": tf.zeros(
                        (1, int(self.sequence_length), int(max(self.global_feature_dim, 0))),
                        dtype=tf.float32,
                    ),
                }
            else:
                dummy_state = tf.random.normal((1, int(self.sequence_length), self.state_dim))
        else:
            dummy_state = tf.random.normal((1, self.state_dim))
        _ = self.actor(dummy_state)
        _ = self.critic(dummy_state)
        
        # Load weights
        self.actor.load_weights(actor_path)
        self.critic.load_weights(critic_path)
        
        logger.info(f"Models loaded from {actor_path} and {critic_path}")
    
    @property
    def memory_size(self):
        """Return the current size of memory buffer."""
        return len(self.memory['states'])
    
    def update_last_episode_rewards(self, scaled_rewards):
        """
        Update rewards in memory for the last episode (for TAPE system).
        
        Args:
            scaled_rewards: List of scaled rewards to replace the last N rewards
        """
        episode_length = len(scaled_rewards)
        if episode_length > len(self.memory['rewards']):
            logger.warning(f"Scaled rewards length ({episode_length}) exceeds memory size ({len(self.memory['rewards'])})")
            return
        
        # Replace the last N rewards
        for i, scaled_reward in enumerate(scaled_rewards):
            idx = -(episode_length - i)
            self.memory['rewards'][idx] = scaled_reward
    
    def save(self, directory):
        """
        Save agent models to directory.
        
        Args:
            directory: Directory path to save models
        """
        import os
        os.makedirs(directory, exist_ok=True)
        
        # Save actor and critic weights
        actor_path = os.path.join(directory, 'actor_weights.h5')
        critic_path = os.path.join(directory, 'critic_weights.h5')
        
        self.actor.save_weights(actor_path)
        self.critic.save_weights(critic_path)
        
        logger.info(f"Models saved to {directory}")
    
    def load(self, directory):
        """
        Load agent models from directory.
        
        Args:
            directory: Directory path to load models from
        """
        import os
        
        actor_path = os.path.join(directory, 'actor_weights.h5')
        critic_path = os.path.join(directory, 'critic_weights.h5')
        
        if os.path.exists(actor_path) and os.path.exists(critic_path):
            self.actor.load_weights(actor_path)
            self.critic.load_weights(critic_path)
            logger.info(f"Models loaded from {directory}")
        else:
            logger.error(f"Model weights not found in {directory}")
            raise FileNotFoundError(f"Model weights not found in {directory}")
    
    def get_action(self, state, training=True, evaluation_mode='mean_plus_noise'):
        """
        Get action for compatibility with train_rl.py.
        Same as get_action_and_value but with training flag.
        
        Args:
            state: Current state
            training: Whether in training mode (True) or evaluation mode (False)
            evaluation_mode: Strategy for deterministic evaluation
            
        Returns:
            tuple: (action, log_prob, value)
        """
        return self.get_action_and_value(state, deterministic=not training, evaluation_mode=evaluation_mode)
    
    def predict(self, observation, deterministic=False, evaluation_mode='mode'):
        """
        Compatible interface for notebook backtests and stable-baselines-like usage.
        
        Args:
            observation: State observation (can be batched or single)
            deterministic: If True, use deterministic evaluation (no stochastic sampling)
            evaluation_mode: Strategy for deterministic evaluation
                - 'mean': Use Dirichlet mean (alpha / sum(alpha))
                - 'mode': Use Dirichlet mode (peak of distribution) - RECOMMENDED
                - 'mean_plus_noise': Mean with small Gaussian noise
        
        Returns:
            tuple: (action, state) where state is None (for compatibility)
        """
        # Call get_action with appropriate parameters
        action, _, _ = self.get_action(
            observation,
            training=not deterministic,
            evaluation_mode=evaluation_mode
        )
        
        # Return (action, state) tuple for API compatibility
        # State is None because we're using feedforward policies (not RNN)
        return action, None
