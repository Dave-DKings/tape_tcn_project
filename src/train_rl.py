"""
Training Script for Portfolio RL Agent with Multi-Architecture Support

This script implements the complete training pipeline for the PPO agent with support
for 5 different neural network architectures:
- TCN: Dense feedforward network (baseline)
- TCN: Recurrent network for temporal dependencies
- TCN_ATTENTION: TCN with multi-head self-attention
- TCN: Temporal Convolutional Network (efficient)
- TCN_ATTENTION: TCN with attention mechanism

Features:
- Architecture-agnostic training loop
- Automatic sequence building for sequential models
- Comprehensive logging and checkpointing
- Performance visualization
- Configurable via config.py

Usage:
    python src/train_rl_CLEAN.py --phase phase1 --architecture TCN
    python src/train_rl_CLEAN.py --phase phase1 --architecture TCN

Author: AI Assistant (Refactored)
Date: October 2, 2025
"""

import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt
import json
import copy
from typing import Dict, Any, Tuple, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import get_active_config, validate_agent_params, is_sequential_architecture
from config import PROFILE_BALANCED_GROWTH, ALL_PROFILES_LIST
from data_utils import DataProcessor
from environment_tape_rl import PortfolioEnvTAPE as PortfolioEnvTF
from agents.ppo_agent_tf import PPOAgentTF
from reward_utils import calculate_episode_metrics, calculate_tape_score, apply_tape_scaling

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrainingSession:
    """
    Manages a complete training session for the portfolio RL agent.
    
    Supports all 5 architectures through the refactored factory pattern:
    - TCN (baseline)
    - TCN (temporal)
    - TCN_ATTENTION (enhanced temporal)
    - TCN (efficient convolutions)
    - TCN_ATTENTION (advanced features)
    """
    
    def __init__(self, phase: str = 'phase1', architecture: str = 'TCN', 
                 config_override: Dict[str, Any] = None):
        """
        Initialize training session.
        
        Args:
            phase: Training phase ('phase1' for baseline)
            architecture: Neural network architecture (TCN, TCN_ATTENTION, TCN_FUSION)
            config_override: Optional config overrides
        """
        self.phase = phase
        self.architecture = architecture.upper()
        
        # Load configuration
        self.config = get_active_config(phase)
        
        # Set architecture in config
        if 'agent_params' not in self.config:
            self.config['agent_params'] = {}
        self.config['agent_params']['actor_critic_type'] = self.architecture
        
        # Apply any config overrides
        if config_override:
            self._deep_update(self.config, config_override)
        
        # Validate configuration
        validate_agent_params(self.config['agent_params'])
        
        # Create results directory
        self.results_dir = self._create_results_directory()
        
        # Training state
        self.episode = 0
        self.total_timesteps = 0
        # ENHANCEMENT: Split stats for separate CSV files
        self.episode_stats_list = []  # For episodes.csv
        self.update_stats_list = []   # For updates.csv
        self.best_performance = -np.inf
        self.training_env: Optional[PortfolioEnvTF] = None
        self.active_feature_manifest_path: Optional[str] = None
        self._last_episode_horizon: Optional[int] = None
        self.tape_terminal_scalar_value: Optional[float] = None
        self.tape_clip_value: Optional[float] = None

        training_params = self.config.get('training_params', {})
        self.episode_log_interval = int(training_params.get(
            'episode_log_interval',
            training_params.get('log_interval_episodes', 10)
        ))
        self.use_episode_length_curriculum = self.config.get('training_params', {}).get(
            'use_episode_length_curriculum', True
        )
        self.data_processor: Optional[DataProcessor] = None
        self.training_df: Optional[pd.DataFrame] = None
        
        rare_params = self.config.get('training_params', {}).get('rare_checkpoint_params', {})
        self.rare_enabled = rare_params.get('enable', False)
        self.rare_min_sharpe = rare_params.get('min_sharpe', 1.6)
        self.rare_min_sortino = rare_params.get('min_sortino')
        self.rare_max_mdd = rare_params.get('max_mdd')
        self.rare_max_turnover = rare_params.get('max_turnover')
        self.rare_top_n = int(rare_params.get('top_n', 5))
        self.rare_records: List[Dict[str, Any]] = []
        self.rare_save_dir: Optional[str] = None
        if self.rare_enabled:
            self.rare_save_dir = os.path.join(self.results_dir, 'rare_models')
            os.makedirs(self.rare_save_dir, exist_ok=True)
        
        # Initialize Profile Manager and TAPE system (if Phase 2)
        env_params = self.config.get('environment_params', {})
        self.reward_type = env_params.get('reward_type', 'simple')
        
        if self.reward_type == 'advanced_tape':
            logger.info("TAPE Reward System Enabled")
            self.default_profile = PROFILE_BALANCED_GROWTH
        else:
            logger.info("Simple Reward System (Phase 1)")
            self.default_profile = PROFILE_BALANCED_GROWTH
        
        logger.info("=" * 80)
        logger.info(f"TRAINING SESSION INITIALIZED")
        logger.info(f"Phase: {phase}")
        logger.info(f"Architecture: {self.architecture}")
        logger.info(f"Sequential Model: {is_sequential_architecture(self.architecture)}")
        logger.info(f"Reward System: {self.reward_type}")
        logger.info(f"Results Directory: {self.results_dir}")
        logger.info("=" * 80)
    
    def _log_curriculum_schedule(self, curriculum_dict: Dict[int, float], metric_name: str) -> None:
        """Dynamically log curriculum schedule from config."""
        if not curriculum_dict:
            return
        
        sorted_thresholds = sorted(curriculum_dict.keys())
        logger.info(f"📚 {metric_name} Curriculum:")
        
        for i, threshold in enumerate(sorted_thresholds):
            value = curriculum_dict[threshold]
            if i < len(sorted_thresholds) - 1:
                next_threshold = sorted_thresholds[i + 1]
                logger.info(f"   {threshold:,}-{next_threshold:,} steps: {metric_name.lower()}={value:.2f}")
            else:
                logger.info(f"   {threshold:,}+ steps: {metric_name.lower()}={value:.2f} (final)")
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict) -> None:
        """Deep update nested dictionary."""
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def _create_results_directory(self) -> str:
        """Create unique results directory with architecture name."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        base_path = self.config.get('training_params', {}).get('results_path')
        if not base_path:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            base_path = os.path.join(project_root, 'results')
        
        # Create directory with architecture name
        dir_name = f"{self.phase}_{self.architecture.lower()}_{timestamp}"
        results_dir = os.path.join(base_path, dir_name)
        os.makedirs(results_dir, exist_ok=True)
        
        logger.info(f"Created results directory: {results_dir}")
        return results_dir
    
    def prepare_data(self) -> Tuple[DataProcessor, pd.DataFrame]:
        """
        Load and prepare training data.
        
        STATE-OF-THE-ART FIX #4: Includes volatility regime labeling for curriculum learning
        
        Returns:
            tuple: (data_processor, processed_dataframe)
        """
        logger.info("=" * 80)
        logger.info("PREPARING TRAINING DATA")
        logger.info("=" * 80)
        
        # Initialize data processor
        data_processor = DataProcessor(self.config)
        
        # Run the feature pipeline based on phase
        if self.phase == 'phase1':
            processed_df, scalers = data_processor.prepare_features_phase1()
        elif self.phase == 'phase2':
            processed_df, scalers = data_processor.prepare_features_phase2()
        else:
            raise ValueError(f"Unknown phase: {self.phase}")
        
        # Store scalers for later use
        data_processor.scalers = scalers
        
        # STATE-OF-THE-ART FIX #4: Calculate volatility regimes for curriculum learning
        training_params = self.config.get('training_params', {})
        if training_params.get('use_curriculum_learning', False):
            logger.info("=" * 80)
            logger.info("🎓 CALCULATING VOLATILITY REGIMES FOR CURRICULUM LEARNING")
            logger.info("=" * 80)
            
            from data_utils import calculate_volatility_regimes
            processed_df = calculate_volatility_regimes(processed_df, window=30)
            
            # Store for environment access
            data_processor.data = processed_df
            
            # Log regime distribution
            if 'volatility_regime' in processed_df.columns:
                regime_counts = processed_df.groupby('volatility_regime')['Date'].nunique()
                total_dates = processed_df['Date'].nunique()
                logger.info(f"Regime distribution across {total_dates} unique dates:")
                for regime, count in regime_counts.items():
                    pct = 100 * count / total_dates
                    logger.info(f"  {regime:12s}: {count:4d} dates ({pct:5.1f}%)")
            logger.info("=" * 80)
        
        logger.info(f"✅ Data preparation completed")
        logger.info(f"   Shape: {processed_df.shape}")
        logger.info(f"   Date range: {processed_df['Date'].min()} to {processed_df['Date'].max()}")
        logger.info(f"   Assets: {data_processor.asset_tickers}")
        logger.info("=" * 80)
        
        self.data_processor = data_processor
        self.training_df = processed_df
        return data_processor, processed_df
    
    def create_environment(self, data_processor: DataProcessor, 
                          processed_df: pd.DataFrame) -> PortfolioEnvTF:
        """
        Create portfolio environment.
        
        Args:
            data_processor: Data processor instance
            processed_df: Processed dataframe
            
        Returns:
            Portfolio environment
        """
        logger.info("=" * 80)
        logger.info("CREATING PORTFOLIO ENVIRONMENT")
        logger.info("=" * 80)
        env_params = self.config.get('environment_params', {})
        mode = env_params.get('mode', 'train')
        reward_type = env_params.get('reward_type', 'simple').lower()
        reward_system = 'tape' if reward_type in ('advanced_tape', 'tape', 'tape_three_component') else 'simple'
        tape_profile = env_params.get('tape_profile', PROFILE_BALANCED_GROWTH)
        tape_scalar = env_params.get('tape_terminal_scalar', 10.0)
        tape_clip = env_params.get('tape_terminal_clip', 8.0)
        dsr_window = env_params.get('dsr_window', 60)
        dsr_scalar = env_params.get('dsr_scalar', 7.0)
        target_turnover = env_params.get('target_turnover', 0.70)
        turnover_penalty_scalar = env_params.get('turnover_penalty_scalar', 2.5)
        turnover_target_band = env_params.get('turnover_target_band', 0.20)
        gamma = env_params.get('tape_gamma', 0.99)
        episode_limit = env_params.get('max_steps_per_episode')
        action_norm = env_params.get('action_normalization', 'softmax')
        exclude_covariance = env_params.get('exclude_covariance', False)
        random_start = env_params.get('random_start', mode == 'train')

        env = PortfolioEnvTF(
            config=self.config,
            data_processor=data_processor,
            processed_data=processed_df,
            mode=mode,
            start_idx=env_params.get('start_idx', 0),
            end_idx=env_params.get('end_idx'),
            action_normalization=action_norm,
            exclude_covariance=exclude_covariance,
            random_start=random_start,
            episode_length_limit=episode_limit,
            reward_system=reward_system,
            tape_profile=tape_profile if reward_system == 'tape' else None,
            tape_terminal_scalar=tape_scalar if reward_system == 'tape' else 1.0,
            tape_terminal_clip=tape_clip,
            dsr_window=dsr_window,
            dsr_scalar=dsr_scalar,
            target_turnover=target_turnover,
            turnover_penalty_scalar=turnover_penalty_scalar,
            turnover_target_band=turnover_target_band,
            gamma=gamma,
            enable_base_reward=env_params.get('enable_base_reward', True)
        )

        self.training_env = env
        self.tape_terminal_scalar_value = getattr(env, 'tape_terminal_scalar', None)
        self.tape_clip_value = getattr(env, 'tape_terminal_clip', None)

        if hasattr(env, "save_active_feature_manifest"):
            manifest_path = os.path.join(self.results_dir, "active_feature_manifest.json")
            env.save_active_feature_manifest(manifest_path)
            self.active_feature_manifest_path = manifest_path
            logger.info("   Active feature manifest: %s", manifest_path)

        logger.info("   Action space: %s", env.action_space.shape)
        logger.info("   Episode horizon (current): %s", "full dataset" if env.episode_length_limit is None else env.episode_length_limit)
        logger.info("   Reward system: %s", reward_system.upper())
        logger.info("=" * 80)

        return env

    def _determine_episode_limit(self, env: PortfolioEnvTF) -> Optional[int]:
        """
        Curriculum for episode length: 1 year → 2 years → full history.
        """
        if not self.use_episode_length_curriculum:
            return getattr(env, 'total_days', None)

        total_days = getattr(env, 'total_days', None)
        if not total_days:
            return None

        if self.total_timesteps < 20_000:
            return min(252, total_days)
        if self.total_timesteps < 60_000:
            return min(504, total_days)
        return total_days

    def _apply_turnover_curriculum(self, env: PortfolioEnvTF) -> None:
        """
        Adjust turnover incentives based on the active curriculum horizon.
        """
        if not hasattr(env, 'set_turnover_parameters'):
            return
        
        if not self.use_episode_length_curriculum:
            env.set_turnover_parameters(scalar=3.0, band=0.18)
            return
        
        total_days = getattr(env, 'total_days', None)
        horizon = env.episode_length_limit
        
        try:
            if horizon is None or (total_days and horizon >= total_days):
                # Full dataset horizon
                env.set_turnover_parameters(scalar=3.0, band=0.18)
            elif horizon <= 252:
                # First curriculum stage
                env.set_turnover_parameters(scalar=2.5, band=0.20)
            elif horizon <= 504:
                # Medium curriculum stage
                env.set_turnover_parameters(scalar=3.0, band=0.12)
            else:
                # Fallback for any intermediate horizons
                env.set_turnover_parameters(scalar=3.0, band=0.18)
        except Exception as exc:
            logger.debug("Turnover curriculum update skipped: %s", exc)
    
    def _maybe_save_rare_checkpoint(self, agent: PPOAgentTF, metrics: Dict[str, float]) -> None:
        """
        Persist rare/high-quality checkpoints based on configurable thresholds.
        """
        if not self.rare_enabled or not metrics:
            return

        sharpe = metrics.get('sharpe_ratio')
        sortino = metrics.get('sortino_ratio')
        mdd = metrics.get('max_drawdown_abs')
        turnover = metrics.get('turnover')

        if sharpe is None or mdd is None:
            return

        if sharpe < self.rare_min_sharpe:
            return
        if self.rare_min_sortino is not None and (sortino is None or sortino < self.rare_min_sortino):
            return
        if self.rare_max_mdd is not None and mdd > self.rare_max_mdd:
            return
        if self.rare_max_turnover is not None and turnover is not None and turnover > self.rare_max_turnover:
            return
        if any(record['episode'] == self.episode for record in self.rare_records):
            return

        # Evaluate ranking score (higher Sharpe, higher Sortino, lower MDD)
        score = (
            float(sharpe),
            float(sortino) if sortino is not None else -np.inf,
            -float(mdd)
        )

        if self.rare_top_n > 0 and len(self.rare_records) >= self.rare_top_n:
            worst = min(self.rare_records, key=lambda r: r['score'])
            if score <= worst['score']:
                return

        if not self.rare_save_dir:
            self.rare_save_dir = os.path.join(self.results_dir, 'rare_models')
            os.makedirs(self.rare_save_dir, exist_ok=True)

        prefix = os.path.join(
            self.rare_save_dir,
            f"ep{self.episode:04d}_sh{sharpe:.3f}_dd{mdd * 100:.1f}"
        )
        agent.save_models(prefix)

        actor_path = f"{prefix}_actor.weights.h5"
        critic_path = f"{prefix}_critic.weights.h5"

        logger.info(
            "💎 Rare checkpoint saved | Episode %d | Sharpe=%.3f | Sortino=%.3f | MDD=%.2f%% | Turnover=%.3f",
            self.episode,
            sharpe,
            sortino if sortino is not None else float('nan'),
            mdd * 100,
            turnover if turnover is not None else float('nan')
        )

        record = {
            'episode': self.episode,
            'score': score,
            'sharpe': sharpe,
            'sortino': sortino,
            'mdd': mdd,
            'turnover': turnover,
            'paths': [actor_path, critic_path]
        }
        self.rare_records.append(record)

        if self.rare_top_n > 0 and len(self.rare_records) > self.rare_top_n:
            # Keep highest scoring entries, drop the rest
            self.rare_records.sort(key=lambda r: r['score'], reverse=True)
            while len(self.rare_records) > self.rare_top_n:
                worst = self.rare_records.pop()
                for path in worst['paths']:
                    try:
                        os.remove(path)
                    except FileNotFoundError:
                        pass
    
    def create_agent(self, env: PortfolioEnvTF) -> PPOAgentTF:
        """
        Create PPO agent with specified architecture.
        
        Args:
            env: Portfolio environment
            
        Returns:
            PPO agent
        """
        logger.info("=" * 80)
        logger.info("CREATING PPO AGENT")
        logger.info("=" * 80)
        
        # Get state dimension (features)
        state_dim = env.observation_space.shape[-1]
        agent_config = copy.deepcopy(self.config['agent_params'])
        agent_config['num_assets'] = env.num_assets
        if hasattr(env, "get_observation_layout"):
            try:
                state_layout = env.get_observation_layout()
                if isinstance(state_layout, dict) and state_layout:
                    agent_config['state_layout'] = state_layout
            except Exception as exc:
                logger.debug("State layout metadata unavailable: %s", exc)
        
        # Create agent using refactored config-based initialization
        agent = PPOAgentTF(
            state_dim=state_dim,
            num_assets=env.num_assets,
            config=agent_config,
            name=f"PPOAgent_{self.architecture}"
        )
        
        logger.info(f"✅ Agent created successfully")
        logger.info(f"   Architecture: {agent.architecture}")
        logger.info(f"   Sequential model: {agent.is_sequential}")
        logger.info(f"   State dimension: {state_dim}")
        logger.info(f"   Number of assets: {env.num_assets}")
        logger.info(f"   Action dimension: {env.num_assets + 1} (assets + cash)")
        if 'state_layout' in agent_config:
            layout = agent_config['state_layout']
            logger.info(
                "   State layout: structured=%s | local=%d x %d | global=%d",
                bool(layout.get('structured_observation', False)),
                int(layout.get('num_assets', env.num_assets)),
                int(layout.get('asset_feature_dim', 0)),
                int(layout.get('global_feature_dim', 0)),
            )
        logger.info("=" * 80)
        
        return agent
    
    def run_episode(self, env: PortfolioEnvTF, agent: PPOAgentTF, volatility_regime: Optional[str] = None, training: bool = True) -> Dict[str, float]:
        """
        Run a single training or evaluation episode.
        
        STATE-OF-THE-ART FIX #4: Supports curriculum learning via volatility regime
        
        Args:
            env: Portfolio environment
            agent: PPO agent
            volatility_regime: Optional regime for curriculum learning
            training: If True, run in training mode (stochastic); else evaluation (deterministic)
            
        Returns:
            dict: Episode statistics including TAPE score if applicable
        """
        # Get active profile for this episode (if using TAPE)
        current_profile = None
        if self.reward_type == 'advanced_tape':
            current_profile = self.default_profile
            if training: # Only log profile usage during training to avoid spam
                logger.info(f"Episode {self.episode}: Using profile '{current_profile['name']}'")

        if self.use_episode_length_curriculum and hasattr(env, 'set_episode_length_limit'):
            new_limit = self._determine_episode_limit(env)
            if new_limit != self._last_episode_horizon:
                env.set_episode_length_limit(new_limit)
                self._last_episode_horizon = new_limit
            if self.reward_type == 'advanced_tape':
                self._apply_turnover_curriculum(env)
        
        # STATE-OF-THE-ART FIX #4: Reset with curriculum regime
        reset_options = {}
        if volatility_regime:
            reset_options['volatility_regime'] = volatility_regime
        
        upcoming_episode = self.episode + 1
        should_log_episode = (upcoming_episode == 1) or (
            self.episode_log_interval > 0 and upcoming_episode % self.episode_log_interval == 0
        )

        state, info = env.reset(options=reset_options if reset_options else None)

        if hasattr(env, 'set_episode_logging'):
            try:
                env.set_episode_logging(should_log_episode)
            except Exception:
                pass
        done = False
        truncated = False
        
        total_reward = 0.0
        episode_length = 0
        step_info = {}
        
        # Get evaluation mode from config (default to 'mean_plus_noise' as requested)
        eval_mode = self.config['agent_params'].get('evaluation_mode', 'mean_plus_noise')
        
        while not done and not truncated:
            # Get action from agent (with correct shapes)
            # Pass training flag and evaluation_mode
            action, log_prob, value = agent.get_action(state, training=training, evaluation_mode=eval_mode)
            
            # Environment step
            next_state, reward, done, truncated, step_info = env.step(action)
            
            # CRITICAL FIX: Correct parameter order - log_prob BEFORE reward
            agent.store_transition(state, action, log_prob, reward, value, done)
            
            # Update counters
            total_reward += reward
            episode_length += 1
            self.total_timesteps += 1
            
            # Move to next state
            state = next_state
            
            if done or truncated:
                break
        
        # Reconstruct episode metrics regardless of reward type
        portfolio_history = np.array(env.portfolio_history)
        if len(portfolio_history) > 1:
            returns = np.diff(portfolio_history) / portfolio_history[:-1]
        else:
            returns = np.array([])
        
        weight_changes = []
        for i in range(1, len(env.weights_history)):
            weight_change = np.abs(env.weights_history[i] - env.weights_history[i-1])
            weight_changes.append(weight_change)
        
        metrics = calculate_episode_metrics(
            portfolio_values=portfolio_history,
            returns=returns,
            weight_changes=weight_changes,
            risk_free_rate=0.02,
            trading_days_per_year=252
        )
        metrics['return_skew'] = metrics.get('skewness', metrics.get('return_skew', 0.0))

        tape_score = None
        if step_info.get('apply_tape', False) and current_profile:
            tape_score = calculate_tape_score(metrics, current_profile)

            # Scale rewards in agent's memory
            original_rewards = list(agent.memory['rewards'])
            scaled_rewards = apply_tape_scaling(
                episode_rewards=original_rewards,
                tape_score=tape_score,
                scaling_method='multiplicative',
                min_scale=0.5,
                max_scale=2.0
            )
            
            # Update agent's reward memory using the update method
            agent.update_last_episode_rewards(scaled_rewards)

            mean_before = float(np.mean(original_rewards)) if len(original_rewards) > 0 else 0.0
            mean_after = float(np.mean(scaled_rewards)) if len(scaled_rewards) > 0 else 0.0
            if should_log_episode:
                logger.info(
                    "Episode %d TAPE scaling | Profile=%s | Score=%.4f | Sharpe=%.3f | Sortino=%.3f | "
                    "MDD=%.2f%% | Turnover=%.3f | Skew=%.3f | mean_reward %.4f → %.4f",
                    self.episode,
                    current_profile['name'],
                    tape_score,
                    metrics.get('sharpe_ratio', 0.0),
                    metrics.get('sortino_ratio', 0.0),
                    metrics.get('max_drawdown_abs', abs(metrics.get('max_drawdown', 0.0))) * 100,
                    metrics.get('turnover', 0.0),
                    metrics.get('return_skew', metrics.get('skewness', 0.0)),
                    mean_before,
                    mean_after
                )
        
        if self.reward_type == 'advanced_tape':
            self._maybe_save_rare_checkpoint(agent, metrics)
        
        episode_stats = {
            'episode_reward': float(total_reward),
            'episode_length': episode_length,
            'final_portfolio_value': float(step_info.get('portfolio_value', 0)),
            'total_return': float(step_info.get('total_return', 0))
        }
        episode_stats.update({
            'episode_return_pct': float(metrics.get('total_return', 0.0) * 100),
            'episode_sharpe': float(metrics.get('sharpe_ratio', 0.0)),
            'episode_sortino': float(metrics.get('sortino_ratio', 0.0)),
            'episode_max_dd': float(metrics.get('max_drawdown_abs', 0.0) * 100),
            'episode_volatility': float(metrics.get('volatility', 0.0)),
            'episode_turnover': float(metrics.get('turnover', 0.0) * 100),
            'episode_return_skew': float(metrics.get('return_skew', 0.0))
        })

        if self.reward_type == 'advanced_tape':
            tape_bonus_val = step_info.get('tape_bonus')
            tape_bonus_raw_val = step_info.get('tape_bonus_raw')
            episode_stats['tape_bonus'] = float(tape_bonus_val) if tape_bonus_val is not None else 0.0
            if tape_bonus_raw_val is not None:
                episode_stats['tape_bonus_raw'] = float(tape_bonus_raw_val)
            else:
                episode_stats['tape_bonus_raw'] = episode_stats['tape_bonus']

        if tape_score is not None:
            episode_stats['tape_score'] = float(tape_score)
        if current_profile:
            episode_stats['profile_name'] = current_profile['name']
        elif 'profile_name' not in episode_stats:
            episode_stats['profile_name'] = None
        
        return episode_stats
    
    def should_update_agent(self) -> bool:
        """Check if it's time to update the agent."""
        timesteps_per_update = self.config['training_params']['timesteps_per_ppo_update']
        return self.total_timesteps % timesteps_per_update == 0
    
    def update_agent(self, agent: PPOAgentTF) -> Dict[str, float]:
        """
        Update the agent using collected experience.
        
        Args:
            agent: PPO agent
            
        Returns:
            dict: Training statistics
        """
        ppo_params = self.config['agent_params']['ppo_params']
        
        update_stats = agent.update(
            num_epochs=ppo_params['num_ppo_epochs'],
            batch_size=ppo_params['batch_size_ppo']
        )
        
    def log_progress(self, episode_stats: Dict[str, float], update_stats: Dict[str, float] = None):
        """
        Log training progress.

        Args:
            episode_stats: Statistics from the episode
            update_stats: Statistics from agent update (if any)
        """
        log_interval = self.config['training_params']['log_interval_episodes']
        
        # ENHANCEMENT: Store episode stats separately (for episodes.csv)
        episode_entry = {
            'episode': self.episode,
            'timesteps': self.total_timesteps,
            # Episode metrics with defaults to avoid placeholder zeros
            'episode_reward': episode_stats.get('episode_reward', 0.0),
            'episode_length': episode_stats.get('episode_length', 0),
            'final_portfolio_value': episode_stats.get('final_portfolio_value', self.config.get('INITIAL_BALANCE', 100000)),
            'total_return': episode_stats.get('total_return', 0.0),
            'episode_return_pct': episode_stats.get('episode_return_pct', 0.0),
            'episode_sharpe': episode_stats.get('episode_sharpe', 0.0),
            'episode_sortino': episode_stats.get('episode_sortino', 0.0),
            'episode_max_dd': episode_stats.get('episode_max_dd', 0.0),
            'episode_volatility': episode_stats.get('episode_volatility', 0.0),
            'episode_turnover': episode_stats.get('episode_turnover', 0.0),
            'episode_return_skew': episode_stats.get('episode_return_skew', 0.0),
            # TAPE metrics (if available)
            'tape_score': episode_stats.get('tape_score', np.nan),
            'tape_bonus': episode_stats.get('tape_bonus', np.nan),
            'tape_bonus_raw': episode_stats.get('tape_bonus_raw', np.nan),
            'profile_name': episode_stats.get('profile_name', '')
        }
        self.episode_stats_list.append(episode_entry)
        
        # ENHANCEMENT: Store update stats separately (for updates.csv) - only when update occurs
        if update_stats:
            update_entry = {
                'episode': self.episode,
                'timesteps': self.total_timesteps,
                # RL training metrics
                'actor_loss': update_stats.get('actor_loss', 0.0),
                'critic_loss': update_stats.get('critic_loss', 0.0),
                'mean_adv': update_stats.get('mean_advantage', 0.0),
                'entropy': update_stats.get('entropy', 0.0),
                'kl_divergence': update_stats.get('approx_kl', np.nan),
                'ppo_clip_ratio': update_stats.get('clip_fraction', np.nan),
                'value_clip_ratio': update_stats.get('value_clip_fraction', np.nan),
                'explained_variance': update_stats.get('explained_variance', np.nan),
                # Additional industry-standard metrics
                'policy_gradient_norm': update_stats.get('actor_grad_norm', 0.0),
                'advantage_mean': update_stats.get('adv_mean', 0.0),
                'advantage_std': update_stats.get('adv_std', 0.0),
            }
            self.update_stats_list.append(update_entry)
        
        # Log to console
        if self.episode % log_interval == 0:
            log_msg = (f"Episode {self.episode:4d} | "
                      f"Reward: {episode_stats['episode_reward']:8.4f} | "
                      f"Length: {episode_stats['episode_length']:3d} | "
                      f"Portfolio: ${episode_stats['final_portfolio_value']:10.2f} | "
                      f"Return: {episode_stats['total_return']:7.2%}")
            
            # Add TAPE info if available
            if 'tape_score' in episode_stats:
                log_msg += f" | TAPE: {episode_stats['tape_score']:.4f} | Profile: {episode_stats['profile_name']}"
            elif 'profile_name' in episode_stats:
                log_msg += f" | Profile: {episode_stats['profile_name']}"
            
            logger.info(log_msg)
            
            if update_stats:
                logger.info(f"         Update | "
                           f"Actor: {update_stats.get('actor_loss', 0):8.4f} | "
                           f"Critic: {update_stats.get('critic_loss', 0):8.4f} | "
                           f"Entropy: {update_stats.get('entropy', 0):8.4f}")
    
    def save_checkpoint(self, agent: PPOAgentTF, episode_stats: Dict[str, float]):
        """
        Save model checkpoint if performance improved.
        
        Args:
            agent: PPO agent
            episode_stats: Latest episode statistics
        """
        current_performance = episode_stats['episode_reward']
        
        # Save best model
        if current_performance > self.best_performance:
            self.best_performance = current_performance
            best_model_dir = os.path.join(self.results_dir, 'best_model')
            os.makedirs(best_model_dir, exist_ok=True)
            
            agent.actor.save_weights(os.path.join(best_model_dir, 'actor.weights.h5'))
            agent.critic.save_weights(os.path.join(best_model_dir, 'critic.weights.h5'))
            
            logger.info(f"💾 New best model saved! Reward: {current_performance:.4f}")
        
        # Save periodic checkpoint
        save_freq = self.config['training_params']['save_freq_episodes']
        if self.episode % save_freq == 0:
            checkpoint_dir = os.path.join(self.results_dir, f'checkpoint_episode_{self.episode}')
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            agent.actor.save_weights(os.path.join(checkpoint_dir, 'actor.weights.h5'))
            agent.critic.save_weights(os.path.join(checkpoint_dir, 'critic.weights.h5'))
            
            logger.info(f"💾 Checkpoint saved at episode {self.episode}")

        tape_score = episode_stats.get('tape_score')
        tape_bonus = episode_stats.get('tape_bonus')
        tape_bonus_raw = episode_stats.get('tape_bonus_raw')
        if self.reward_type == 'advanced_tape' and tape_score is not None:
            scalar = float(self.tape_terminal_scalar_value or 1.0)
            clip_limit = float(self.tape_clip_value) if self.tape_clip_value is not None else None

            raw_bonus = (
                float(tape_bonus_raw)
                if tape_bonus_raw is not None
                else float(tape_score) * scalar
            )

            clipped_bonus = (
                float(tape_bonus)
                if tape_bonus is not None
                else (
                    max(-clip_limit, min(raw_bonus, clip_limit))
                    if clip_limit is not None
                    else raw_bonus
                )
            )

            if clip_limit is not None:
                clip_threshold = clip_limit / scalar
                if abs(tape_score) >= clip_threshold - 1e-6:
                    self._save_tape_checkpoint(
                        agent,
                        tape_score=tape_score,
                        tape_bonus=clipped_bonus,
                        tape_bonus_raw=raw_bonus,
                        reason="clip"
                    )

            raw_threshold = 8.0
            if abs(raw_bonus) >= raw_threshold - 1e-6:
                self._save_tape_checkpoint(
                    agent,
                    tape_score=tape_score,
                    tape_bonus=clipped_bonus,
                    tape_bonus_raw=raw_bonus,
                    reason="bonus"
                )

    def _save_tape_checkpoint(
        self,
        agent: PPOAgentTF,
        tape_score: float,
        tape_bonus: float,
        tape_bonus_raw: Optional[float],
        reason: str = "clip"
    ) -> None:
        """
        Persist a snapshot whenever the TAPE objective meets a checkpoint condition.
        """
        dir_name = f'{reason}_episode_{self.episode:04d}'
        save_dir = os.path.join(self.results_dir, dir_name)
        actor_path = os.path.join(save_dir, 'actor.weights.h5')

        # Skip if we already saved this episode
        if os.path.exists(actor_path):
            return

        os.makedirs(save_dir, exist_ok=True)
        agent.actor.save_weights(actor_path)
        agent.critic.save_weights(os.path.join(save_dir, 'critic.weights.h5'))

        metadata = {
            "episode": self.episode,
            "tape_score": float(tape_score),
            "tape_bonus": float(tape_bonus),
            "tape_bonus_raw": float(tape_bonus_raw) if tape_bonus_raw is not None else None,
            "tape_terminal_scalar": float(self.tape_terminal_scalar_value),
            "tape_clip_value": float(self.tape_clip_value),
            "total_timesteps": self.total_timesteps,
            "timestamp": datetime.now().isoformat(),
            "reason": reason
        }
        with open(os.path.join(save_dir, 'metadata.json'), 'w') as metadata_file:
            json.dump(metadata, metadata_file, indent=2)

        logger.info(
            "💾 TAPE episode snapshot saved (%s, episode %d, score %.4f, bonus %.2f)",
            reason,
            self.episode,
            tape_score,
            tape_bonus
        )

    def save_training_results(self):
        """Save training statistics, plots, and configuration."""
        logger.info("=" * 80)
        logger.info("SAVING TRAINING RESULTS")
        logger.info("=" * 80)
        
        # ENHANCEMENT: Save split CSV files
        # 1. Episodes CSV
        if self.episode_stats_list:
            episodes_df = pd.DataFrame(self.episode_stats_list)
            episodes_file = os.path.join(self.results_dir, 'episodes.csv')
            episodes_df.to_csv(episodes_file, index=False)
            logger.info(f"✅ Episodes stats saved: {episodes_file}")
            
        # 2. Updates CSV
        if self.update_stats_list:
            updates_df = pd.DataFrame(self.update_stats_list)
            updates_file = os.path.join(self.results_dir, 'updates.csv')
            updates_df.to_csv(updates_file, index=False)
            logger.info(f"✅ Updates stats saved: {updates_file}")
            
        # Save training stats as JSON (legacy/backup)
        stats_file = os.path.join(self.results_dir, 'training_stats.json')
        with open(stats_file, 'w') as f:
            json.dump(self.training_stats, f, indent=2)
        logger.info(f"✅ Training stats saved: {stats_file}")
        
        # Create training plots
        self.create_training_plots()
        
        # Save configuration
        config_file = os.path.join(self.results_dir, 'training_config.json')
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=2, default=str)
        logger.info(f"✅ Configuration saved: {config_file}")
        
        # Save metadata
        metadata = {
            'phase': self.phase,
            'architecture': self.architecture,
            'total_episodes': self.episode,
            'total_timesteps': self.total_timesteps,
            'best_performance': float(self.best_performance),
            'results_directory': self.results_dir,
            'training_date': datetime.now().isoformat(),
            'active_feature_manifest_path': self.active_feature_manifest_path
        }
        metadata_file = os.path.join(self.results_dir, 'training_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"✅ Metadata saved: {metadata_file}")
        
        if self.rare_enabled and self.rare_records:
            summary_dir = self.rare_save_dir or os.path.join(self.results_dir, 'rare_models')
            os.makedirs(summary_dir, exist_ok=True)
            summary_path = os.path.join(summary_dir, 'rare_models_summary.json')
            summary_payload = [
                {
                    'episode': record['episode'],
                    'sharpe': record['sharpe'],
                    'sortino': record['sortino'],
                    'max_drawdown_pct': record['mdd'] * 100 if record['mdd'] is not None else None,
                    'turnover': record['turnover'],
                    'paths': record['paths']
                }
                for record in sorted(self.rare_records, key=lambda r: r['score'], reverse=True)
            ]
            with open(summary_path, 'w') as f:
                json.dump(summary_payload, f, indent=2)
            logger.info("✅ Rare checkpoint summary saved: %s", summary_path)
        
        logger.info("=" * 80)
    
    def create_training_plots(self):
        """Create and save comprehensive training progress plots."""
        if not self.training_stats:
            logger.warning("No training stats to plot")
            return
        
        df_stats = pd.DataFrame(self.training_stats)
        
        # Create figure with 6 subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Training Progress - {self.architecture} Architecture', fontsize=16, fontweight='bold')
        
        # 1. Episode rewards
        axes[0, 0].plot(df_stats['episode'], df_stats['episode_reward'], alpha=0.6, label='Rewards')
        axes[0, 0].plot(df_stats['episode'], df_stats['episode_reward'].rolling(10).mean(), 
                       linewidth=2, label='MA(10)')
        axes[0, 0].set_title('Episode Rewards', fontweight='bold')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Portfolio value
        axes[0, 1].plot(df_stats['episode'], df_stats['final_portfolio_value'], color='green')
        axes[0, 1].axhline(y=100000, color='red', linestyle='--', alpha=0.5, label='Initial Capital')
        axes[0, 1].set_title('Final Portfolio Value', fontweight='bold')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Portfolio Value ($)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Total returns
        if 'total_return' in df_stats.columns:
            axes[0, 2].plot(df_stats['episode'], df_stats['total_return'] * 100, color='purple')
            axes[0, 2].axhline(y=0, color='black', linestyle='--', alpha=0.3)
            axes[0, 2].set_title('Total Returns', fontweight='bold')
            axes[0, 2].set_xlabel('Episode')
            axes[0, 2].set_ylabel('Return (%)')
            axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Episode lengths
        axes[1, 0].plot(df_stats['episode'], df_stats['episode_length'], color='orange')
        axes[1, 0].set_title('Episode Lengths', fontweight='bold')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Steps')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Actor loss (if available)
        if 'actor_loss' in df_stats.columns:
            actor_losses = df_stats['actor_loss'].dropna()
            if len(actor_losses) > 0:
                axes[1, 1].plot(actor_losses.index, actor_losses, color='blue', alpha=0.6)
                axes[1, 1].plot(actor_losses.index, actor_losses.rolling(5).mean(), 
                               linewidth=2, color='darkblue', label='MA(5)')
                axes[1, 1].set_title('Actor Loss', fontweight='bold')
                axes[1, 1].set_xlabel('Update Step')
                axes[1, 1].set_ylabel('Loss')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Critic loss (if available)
        if 'critic_loss' in df_stats.columns:
            critic_losses = df_stats['critic_loss'].dropna()
            if len(critic_losses) > 0:
                axes[1, 2].plot(critic_losses.index, critic_losses, color='red', alpha=0.6)
                axes[1, 2].plot(critic_losses.index, critic_losses.rolling(5).mean(), 
                               linewidth=2, color='darkred', label='MA(5)')
                axes[1, 2].set_title('Critic Loss', fontweight='bold')
                axes[1, 2].set_xlabel('Update Step')
                axes[1, 2].set_ylabel('Loss')
                axes[1, 2].legend()
                axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = os.path.join(self.results_dir, 'training_progress.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Training plots saved: {plot_file}")
    
    def run_training(self):
        """
        Execute the complete training pipeline.
        """
        logger.info("=" * 80)
        logger.info(f"🚀 STARTING TRAINING SESSION - {self.phase.upper()}")
        logger.info(f"🏗️  Architecture: {self.architecture}")
        logger.info("=" * 80)
        
        start_time = datetime.now()
        
        try:
            # 1. Prepare data
            data_processor, processed_df = self.prepare_data()
            
            # 2. Create environment
            env = self.create_environment(data_processor, processed_df)
            
            # 3. Create agent
            agent = self.create_agent(env)
            
            # 4. Training loop with STATE-OF-THE-ART FIX #4: Curriculum Learning
            max_timesteps = self.config['training_params']['max_total_timesteps']
            
            # Setup curriculum learning phases
            training_params = self.config['training_params']
            use_curriculum = training_params.get('use_curriculum_learning', False)
            curriculum_phases = training_params.get('curriculum_phases', [])
            
            if use_curriculum and curriculum_phases:
                logger.info("=" * 80)
                logger.info("🎓 CURRICULUM LEARNING ENABLED")
                logger.info("=" * 80)
                for phase in curriculum_phases:
                    logger.info(f"  Phase: {phase['name']:12s} - {phase['timesteps_fraction']*100:.0f}% of training")
                logger.info("=" * 80)
            
            logger.info("=" * 80)
            logger.info(f"🎯 STARTING TRAINING LOOP")
            logger.info(f"   Max timesteps: {max_timesteps:,}")
            logger.info(f"   Update frequency: {self.config['training_params']['timesteps_per_ppo_update']}")
            logger.info("=" * 80)
            
            while self.total_timesteps < max_timesteps:
                self.episode += 1
                
                # STATE-OF-THE-ART FIX #4: Determine current curriculum phase
                current_regime = None
                if use_curriculum and curriculum_phases:
                    progress = self.total_timesteps / max_timesteps
                    cumulative = 0.0
                    for phase in curriculum_phases:
                        cumulative += phase['timesteps_fraction']
                        if progress < cumulative:
                            current_regime = phase['name']
                            break
                    if current_regime is None:  # Safety fallback
                        current_regime = curriculum_phases[-1]['name']
                    
                    # Log phase transitions
                    if not hasattr(self, '_last_regime') or self._last_regime != current_regime:
                        logger.info("=" * 80)
                        logger.info(f"🎓 CURRICULUM PHASE CHANGE: Now training on '{current_regime}' data")
                        logger.info(f"   Progress: {progress*100:.1f}% ({self.total_timesteps:,}/{max_timesteps:,} timesteps)")
                        logger.info("=" * 80)
                        self._last_regime = current_regime
                
                # Run episode with curriculum regime
                episode_stats = self.run_episode(env, agent, volatility_regime=current_regime)
                
                # Update agent if enough timesteps collected
                update_stats = None
                if self.should_update_agent():
                    update_stats = self.update_agent(agent)
                
                # Log progress
                self.log_progress(episode_stats, update_stats)
                
                # Save checkpoints
                self.save_checkpoint(agent, episode_stats)
                
                # Check if training should stop
                if self.total_timesteps >= max_timesteps:
                    logger.info(f"✅ Reached maximum timesteps ({self.total_timesteps:,})")
                    break
            
            # 5. Save final model
            logger.info("=" * 80)
            logger.info("SAVING FINAL MODEL")
            logger.info("=" * 80)
            
            final_model_dir = os.path.join(self.results_dir, 'final_model')
            os.makedirs(final_model_dir, exist_ok=True)
            agent.actor.save_weights(os.path.join(final_model_dir, 'actor.weights.h5'))
            agent.critic.save_weights(os.path.join(final_model_dir, 'critic.weights.h5'))
            logger.info(f"✅ Final model saved: {final_model_dir}")
            
            # 6. Save training results
            self.save_training_results()
            
            # 7. Training summary
            end_time = datetime.now()
            training_duration = end_time - start_time
            
            logger.info("=" * 80)
            logger.info("🎉 TRAINING COMPLETED SUCCESSFULLY!")
            logger.info("=" * 80)
            logger.info(f"⏱️  Duration: {training_duration}")
            logger.info(f"📈 Episodes: {self.episode}")
            logger.info(f"🔢 Total timesteps: {self.total_timesteps:,}")
            logger.info(f"🏆 Best performance: {self.best_performance:.4f}")
            logger.info(f"💾 Results directory: {self.results_dir}")
            logger.info("=" * 80)
            
        except KeyboardInterrupt:
            logger.warning("=" * 80)
            logger.warning("⚠️  Training interrupted by user!")
            logger.warning("=" * 80)
            logger.info("Saving current progress...")
            self.save_training_results()
            
        except Exception as e:
            logger.error("=" * 80)
            logger.error(f"❌ Training failed with error: {e}")
            logger.error("=" * 80)
            raise


def main():
    """Main training function with command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Train Portfolio RL Agent with Multi-Architecture Support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with TCN architecture (baseline)
  python src/train_rl_CLEAN.py --phase phase1 --architecture TCN
  
  # Train with TCN architecture
  python src/train_rl_CLEAN.py --phase phase1 --architecture TCN
  
  # Train with TCN+Attention architecture
  python src/train_rl_CLEAN.py --phase phase1 --architecture TCN_ATTENTION
  
  # Custom timesteps
  python src/train_rl_CLEAN.py --phase phase1 --architecture TCN --timesteps 50000

Available Architectures:
  - TCN: Dense feedforward network (fastest, baseline)
  - TCN: Recurrent network for temporal dependencies
  - TCN_ATTENTION: TCN with multi-head self-attention
  - TCN: Temporal Convolutional Network (efficient)
  - TCN_ATTENTION: TCN with attention mechanism (most advanced)
        """
    )
    
    parser.add_argument('--phase', type=str, default='phase1', 
                       choices=['phase1', 'phase2'],
                       help='Training phase (default: phase1)')
    
    parser.add_argument('--architecture', type=str, default='TCN',
                       choices=['TCN', 'TCN_ATTENTION', 'TCN_FUSION'],
                       help='Neural network architecture (default: TCN)')
    
    parser.add_argument('--timesteps', type=int, default=None,
                       help='Override max timesteps')
    
    parser.add_argument('--results-dir', type=str, default=None,
                       help='Override results directory')
    
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Prepare config overrides
    config_override = {}
    if args.timesteps:
        config_override['training_params'] = {'max_total_timesteps': args.timesteps}
    if args.results_dir:
        config_override['training_params'] = config_override.get('training_params', {})
        config_override['training_params']['results_path'] = args.results_dir
    
    # Display banner
    print("=" * 80)
    print("    ADAPTIVE PORTFOLIO RL - TRAINING PIPELINE")
    print("    Multi-Architecture Support (TCN, TCN, TCN)")
    print("=" * 80)
    print(f"Phase: {args.phase}")
    print(f"Architecture: {args.architecture}")
    if args.timesteps:
        print(f"Max timesteps: {args.timesteps:,}")
    print("=" * 80)
    print()
    
    # Run training
    try:
        session = TrainingSession(
            phase=args.phase, 
            architecture=args.architecture,
            config_override=config_override
        )
        session.run_training()
    except Exception as e:
        logger.error(f"Training session failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
