"""
Portfolio Management Environment - TAPE RL Version

TAPE RL ENVIRONMENT
=====================================

This environment has been designed for robust portfolio RL under the TAPE framework
with a focus on stable training dynamics, realistic trading mechanics, and risk-aware rewards
on our exact dataset, proving that the data is good and PPO can work - IF the
environment is designed correctly.

KEY DESIGN CHOICES:
=====================================

1. **REWARD FUNCTION** (Design Pattern: Lines ~210)
   - OLD: Complex punitive system with penalties, clipping, log-space conversions
   - NEW: Simple portfolio value as reward (self.reward = new_portfolio_value)
   - RATIONALE: Uses raw portfolio-value progression for clear objective alignment.

2. **TERMINATION LOGIC** (Design Pattern: Lines ~128)
   - OLD: Aggressive balance threshold terminations (60% drawdown = episode end)
   - NEW: Only terminate when data exhausted (self.terminal = self.day >= len(data) - 1)
   - RATIONALE: Episodes run full available horizon to preserve temporal learning signal.

3. **PORTFOLIO RETURN CALCULATION** (Design Pattern: Lines ~195-200)
   - OLD: Log-space arithmetic with multiple safeguards and caps
   - NEW: Simple linear returns: portfolio_value * (1 + weighted_return)
   - RATIONALE: Uses straightforward portfolio math with stable scaling assumptions.

4. **ACTION NORMALIZATION** (Design Pattern: Lines ~172-176)
   - OLD: Clipping then division normalization
   - NEW: Softmax normalization (numerically stable, always sums to 1)
   - RATIONALE: Uses softmax for stable simplex-constrained allocations.

5. **TRANSACTION COSTS** (Maintained from our implementation)
   - Kept our turnover-based transaction cost model
   - Applied as simple subtraction from portfolio value
   - RATIONALE: Transaction costs are modeled explicitly with a turnover-based formulation

6. **NO TRAINING WHEELS, NO MILESTONE REWARDS**
   - Removed all artificial scaffolding (grace periods, survival bonuses, milestones)
   - RATIONALE: The environment avoids artificial scaffolding to preserve clean credit assignment.

EXPECTED OUTCOME:
================
With these changes, our PPO agent should achieve performance comparable to the
project benchmark (~80% return, ~0.85 Sharpe) when trained on the same data.

Author: AI Assistant (Claude)
Date: October 7, 2025
Benchmark Reference: internal project benchmark
"""

import logging
import json
import os
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any, Optional, List, Set
from collections import deque
import gymnasium as gym
from gymnasium import spaces

# Import TAPE reward utilities
from src.reward_utils import (
    calculate_episode_metrics,
    calculate_tape_score,
    calculate_sharpe_ratio_dsr,  # New: For DSR daily reward
    step_level_risk_filter  # Keep for reference, but will be replaced by DSR
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PortfolioEnvTAPE(gym.Env):
    """
    Portfolio Management Environment - TAPE RL Design
    
    This environment implements a robust TAPE portfolio RL environment design
    that achieved 80.98% return on our benchmark test.
    
    Key Design Principles (TAPE):
    - Simple reward = portfolio value (no complex penalties)
    - Minimal termination conditions (only when data exhausted)
    - Clean portfolio math (no log-space gymnastics)
    - Softmax action normalization (numerical stability)
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, 
                 config: Dict,
                 data_processor,
                 processed_data: pd.DataFrame,
                 mode: str = 'train',
                 start_idx: int = 0,
                 end_idx: Optional[int] = None,
                 action_normalization: str = 'softmax',
                 exclude_covariance: bool = False,
                 random_start: bool = False,
                 episode_length_limit: Optional[int] = None,
                 terminal_reward_metric: str = 'return',
                 reward_system: str = 'simple',
                 tape_profile: Optional[Dict] = None,
                 tape_terminal_scalar: float = 1000.0,
                 tape_terminal_clip: Optional[float] = 5.0,
                 dsr_window: int = 60,
                 dsr_scalar: float = 5.0,
                 target_turnover: float = 0.76,  # Target turnover per day as decimal (0.76 = 76%)
                 turnover_penalty_scalar: float = 5.0,
                 turnover_target_band: float = 0.20,
                 gamma: float = 0.99,
                 enable_base_reward: bool = True,
                 drawdown_constraint: Optional[Dict[str, Any]] = None):
        """
        Initialize the TAPE-style Portfolio Environment with optional TAPE reward system.
        
        Args:
            config: Configuration dictionary
            data_processor: DataProcessor with fitted scalers
            processed_data: DataFrame with features and returns
            mode: 'train' or 'test'
            start_idx: Starting index
            end_idx: Ending index (None = use all)
            action_normalization: 'softmax', 'dirichlet', or 'none' (default: 'softmax')
                - 'softmax': Apply softmax normalization (treats actions as logits)
                - 'dirichlet': Transform actions to alpha, sample from Dirichlet
                - 'none': Use actions directly as weights (agent must output normalized weights)
            exclude_covariance: If True, exclude covariance eigenvalue features (default: False)
            random_start: If True, enable random starting days even in test mode (default: False)
            terminal_reward_metric: Method for calculating terminal reward (default: 'return')
                - 'return': Based on final portfolio return (original method)
                - 'sharpe': Based on annualized Sharpe ratio of episode returns
                - Only used when reward_system='simple'
            reward_system: Reward system to use (default: 'simple')
                - 'simple': Direct portfolio return × 100 (current system)
                - 'tape': Three-Component Risk-Aligned Reward System with PBRS + Terminal TAPE
            tape_profile: TAPE utility profile dictionary (required if reward_system='tape')
                - Must contain: mu, weights, a_bounds, b_bounds, metrics_order
                - Plus either k_minus/k_plus (sigmoid) or sigma_sq_minus/sigma_sq_plus (Gaussian)
                - Example: PROFILE_BALANCED_GROWTH from config.py
            tape_terminal_scalar: Multiplier for TAPE terminal bonus (default: 1000.0)
                - Only used when reward_system='tape'
                - Final bonus = tape_score * tape_terminal_scalar
                - Moderate scalar acts as final "nudge" rather than overwhelming signal
            tape_terminal_clip: Clip bounds for the terminal bonus (default: ±5.0)
                - Set to None to disable clipping.
            dsr_window: Window size for Differential Sharpe Ratio calculation (default: 60)
                - Number of recent returns to use for DSR computation
                - Only used when reward_system='tape'
            dsr_scalar: Multiplier for DSR daily reward (default: 7.0)
                - Scales the differential Sharpe to meaningful reward magnitude
            drawdown_constraint: Optional dict enabling dual drawdown controller with keys:
                enabled (bool), target, tolerance, dual_learning_rate, lambda_init, lambda_max,
                penalty_coef/base_coef. When provided, applies architecture-sensitive penalty.
                - Only used when reward_system='tape'
            target_turnover: Turnover ceiling per step as DECIMAL (default: 0.76)
                - Acts as a CEILING: no penalty below, proportional penalty above
                - Value is in decimal: 0.02 = 2% daily turnover
                - Transaction costs in base return already discourage excess trading;
                  this is a safety rail for extreme churning
                - Only used when reward_system='tape'
            turnover_penalty_scalar: Multiplier for excess turnover penalty (default: 5.0)
                - Penalty = (excess / ceiling) * scalar
                - Only applied when turnover exceeds ceiling
                - Only used when reward_system='tape'
            turnover_target_band: DEPRECATED - no longer used (was for symmetric proximity)
            episode_length_limit: Optional hard cap on episode length (default: None)
                - When provided, episode terminates after `limit` steps even if data remains.
            gamma: Discount factor for PBRS (Potential-Based Reward Shaping) (default: 0.99)
                - Used in DSR formula: R_shaped = R + gamma * P(s') - P(s)
                - Only used when reward_system='tape'
        """
        super().__init__()
        
        # Store configuration
        self.config = config
        self.data_processor = data_processor
        self.mode = mode
        self.random_start = random_start  # New parameter
        self.episode_length_limit = int(episode_length_limit) if episode_length_limit is not None else None
        self.episode_step_count = 0
        
        # Reward system configuration
        self.reward_system = reward_system.lower()
        if self.reward_system not in ['simple', 'tape']:
            raise ValueError(f"reward_system must be 'simple' or 'tape', got '{reward_system}'")
        
        # TAPE profile validation
        if self.reward_system == 'tape':
            if tape_profile is None:
                raise ValueError("tape_profile must be provided when reward_system='tape'")
            
            # Validate profile structure - accept both new sigmoid and legacy Gaussian formats
            required_keys = ['mu', 'weights', 
                           'a_bounds', 'b_bounds', 'metrics_order']
            # Must have either k_minus/k_plus (sigmoid) or sigma_sq_minus/sigma_sq_plus (Gaussian)
            has_sigmoid = 'k_minus' in tape_profile and 'k_plus' in tape_profile
            has_gaussian = 'sigma_sq_minus' in tape_profile and 'sigma_sq_plus' in tape_profile
            if not has_sigmoid and not has_gaussian:
                raise ValueError("tape_profile must contain either 'k_minus'/'k_plus' (sigmoid) "
                               "or 'sigma_sq_minus'/'sigma_sq_plus' (Gaussian)")
            missing_keys = [k for k in required_keys if k not in tape_profile]
            if missing_keys:
                raise ValueError(f"tape_profile missing required keys: {missing_keys}")
            
            self.tape_profile = tape_profile
            self.tape_terminal_scalar = tape_terminal_scalar
            self.tape_terminal_clip = tape_terminal_clip
            self.dsr_window = dsr_window
            self.dsr_scalar = dsr_scalar
            self.target_turnover = target_turnover
            self.turnover_penalty_scalar = turnover_penalty_scalar
            self.turnover_target_band = float(max(0.0, turnover_target_band))
            self.gamma = gamma
            # Toggle for Component 1 (Base Reward)
            self.enable_base_reward = enable_base_reward
            # Clip terminal bonus to keep reward scale aligned with daily rewards
            if self.tape_terminal_clip is not None:
                self.tape_terminal_clip = float(self.tape_terminal_clip)

            logger.info(f"🎯 TAPE Three-Component Reward System enabled")
            logger.info(f"   Profile: {tape_profile.get('name', 'Custom')}")
            logger.info(f"   Component 1: Base Reward (Net Return)")
            logger.info(f"   Component 2: DSR/PBRS (window={dsr_window}, scalar={dsr_scalar}, gamma={gamma})")
            logger.info(
                f"   Component 3: Turnover Ceiling (max={target_turnover}, "
                f"penalty_scalar={turnover_penalty_scalar})"
            )
            logger.info(f"   Terminal: TAPE Score × {tape_terminal_scalar}")
            logger.info(f"   Component 1 enabled: {self.enable_base_reward}")
        else:
            self.tape_profile = None
            self.tape_terminal_scalar = None
            self.dsr_window = None
            self.dsr_scalar = None
            self.target_turnover = None
            self.turnover_penalty_scalar = None
            self.turnover_target_band = None
            self.gamma = None
            self.tape_terminal_clip = None
        
        # Terminal reward metric (only used for 'simple' reward system)
        self.terminal_reward_metric = terminal_reward_metric.lower()
        if self.terminal_reward_metric not in ['return', 'sharpe']:
            raise ValueError(f"terminal_reward_metric must be 'return' or 'sharpe', got '{terminal_reward_metric}'")
        
        # Feature control
        self.exclude_covariance = exclude_covariance
        self.feature_phase = self._resolve_feature_phase()
        self.requested_feature_columns: List[str] = []
        self.active_feature_columns: List[str] = []
        self.missing_feature_columns: List[str] = []
        self.excluded_covariance_columns: List[str] = []
        self.asset_feature_columns: List[str] = []
        self.global_feature_columns: List[str] = []
        self.asset_feature_count_per_asset: int = 0
        self.global_feature_count: int = 0
        self.asset_feature_matrix: Optional[np.ndarray] = None
        self.context_feature_matrix: Optional[np.ndarray] = None
        
        # Action normalization method
        self.action_normalization = action_normalization.lower()
        if self.action_normalization not in ['softmax', 'dirichlet', 'none']:
            raise ValueError(f"action_normalization must be 'softmax', 'dirichlet', or 'none', got '{action_normalization}'")
        
        # Extract key parameters from config
        self.num_assets = config.get('NUM_ASSETS', 5)
        self.initial_balance = config.get('INITIAL_BALANCE', 100000.0)
        env_params = config.get('environment_params', {}) if isinstance(config.get('environment_params', {}), dict) else {}
        self.structured_observation = bool(env_params.get('structured_observation', True))
        self.transaction_cost_rate = float(
            env_params.get(
                'transaction_cost_pct',
                config.get('transaction_cost_pct', config.get('TRANSACTION_COST_RATE', 0.0001))
            )
        )

        # Action-realization alignment and concentration controls
        self.concentration_penalty_scalar = float(env_params.get('concentration_penalty_scalar', 0.0))
        self.concentration_target_hhi = float(env_params.get('concentration_target_hhi', 1.0 / max(self.num_assets, 1)))
        self.top_weight_penalty_scalar = float(env_params.get('top_weight_penalty_scalar', 0.0))
        self.target_top_weight = float(env_params.get('target_top_weight', 1.0))
        self.action_realization_penalty_scalar = float(env_params.get('action_realization_penalty_scalar', 0.0))
        self.penalty_budget_ratio = float(env_params.get('penalty_budget_ratio', 2.0))
        self.intra_step_tape_delta_enabled = bool(env_params.get('intra_step_tape_delta_enabled', False))
        self.intra_step_tape_delta_window = int(env_params.get('intra_step_tape_delta_window', 60))
        self.intra_step_tape_delta_min_history = int(env_params.get('intra_step_tape_delta_min_history', 20))
        self.intra_step_tape_delta_beta = float(env_params.get('intra_step_tape_delta_beta', 0.10))
        self.intra_step_tape_delta_clip = float(env_params.get('intra_step_tape_delta_clip', 0.20))
        self._last_intra_step_tape_potential = 0.0
        self.last_intra_step_tape_potential = None
        self.last_intra_step_tape_delta_reward = 0.0

        dd_regime_cfg = env_params.get('dd_regime_scaling', {}) if isinstance(env_params.get('dd_regime_scaling', {}), dict) else {}
        self.dd_regime_scaling_enabled = bool(dd_regime_cfg.get('enabled', False))
        self.dd_regime_vol_window = int(dd_regime_cfg.get('vol_window', 21))
        self.dd_regime_low_vol_threshold = float(dd_regime_cfg.get('low_vol_threshold', 0.12))
        self.dd_regime_high_vol_threshold = float(dd_regime_cfg.get('high_vol_threshold', 0.25))
        self.dd_regime_low_mult = float(dd_regime_cfg.get('low_mult', 0.9))
        self.dd_regime_mid_mult = float(dd_regime_cfg.get('mid_mult', 1.0))
        self.dd_regime_high_mult = float(dd_regime_cfg.get('high_mult', 1.3))
        
        # Dirichlet concentration parameter (only used if action_normalization='dirichlet')
        self.dirichlet_alpha_scale = config.get('DIRICHLET_ALPHA_SCALE', 1.0)
        
        # Stable ticker order used for all per-day alignment.
        processor_tickers = list(getattr(self.data_processor, "asset_tickers", []) or [])
        config_tickers = list(config.get("ASSET_TICKERS", []) or [])
        resolved_tickers = processor_tickers or config_tickers
        if not resolved_tickers and 'Ticker' in processed_data.columns:
            resolved_tickers = sorted(str(t) for t in processed_data['Ticker'].dropna().unique().tolist())
        self.ordered_tickers = [str(t) for t in resolved_tickers][:self.num_assets]

        # Process data
        self.processed_data = processed_data
        self.start_idx = start_idx
        self.end_idx = end_idx if end_idx is not None else len(processed_data)
        
        # Extract dates (for multi-asset data in long format)
        if 'Date' in processed_data.columns:
            self.dates = sorted(processed_data['Date'].unique())
            self.dates = self.dates[start_idx:self.end_idx]
            self.total_days = len(self.dates)
        else:
            raise ValueError("processed_data must have a 'Date' column")
        
        # Build return matrix: (days, assets)
        self._build_return_matrix()
        
        # Build feature matrix: (days, features)
        self._build_feature_matrix()
        
        # Define action space: continuous weights for N assets + cash
        # Design Pattern: Box(low=0, high=1, shape=(n_assets,))
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.num_assets + 1,),  # +1 for cash
            dtype=np.float32
        )
        
        # Define observation space
        self.num_features = self.feature_matrix.shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_features,),
            dtype=np.float32
        )
        
        # Initialize episode state
        self.day = 0
        self.portfolio_value = self.initial_balance
        self.current_weights = self._build_initial_weights()
        
        # Memory for tracking
        self.portfolio_history = [self.initial_balance]
        self.return_history = [0.0]
        self.weights_history = [self.current_weights.copy()]
        self.date_history = [self.dates[0]]
        self.action_realization_l1_history = []
        self.concentration_hhi_history = []
        self.top_weight_history = []
        self.concentration_penalty_sum = 0.0
        self.action_realization_penalty_sum = 0.0
        
        # 🎯 TAPE System: Episode buffers for DSR + Terminal metrics
        if self.reward_system == 'tape':
            # DSR (Component 2): Differential Sharpe Ratio buffers for PBRS
            self.dsr_history = deque(maxlen=self.dsr_window)  # Rolling window of returns
            self.last_sharpe = 0.0  # Track previous Sharpe for PBRS calculation
            
            # Terminal TAPE: Episode-level metrics
            self.episode_portfolio_values = [self.initial_balance]
            self.episode_return_history = []  # For TAPE metrics
            self.episode_weight_changes = []  # For turnover calculation
            
            # Component 3: Per-step turnover target
            # FIX: target_turnover is ALREADY per-step in decimal format (not cumulative episode total)
            # Config values are in decimal: 0.76 = 76% per day, 0.5 = 50% per day
            # No conversion needed - use directly as per-step target
            self.target_turnover_per_step = self.target_turnover if self.target_turnover else 0.0

        # Drawdown dual-controller state
        self.drawdown_constraint_enabled = False
        self.drawdown_target = 0.0
        self.drawdown_tolerance = 0.0
        self.drawdown_dual_lr = 0.0
        self.drawdown_lambda = 0.0
        self.drawdown_lambda_init = 0.0
        self.drawdown_lambda_max = 0.0
        self.drawdown_penalty_coef = 0.0
        self.drawdown_penalty_sum = 0.0
        self.drawdown_excess_accumulator = 0.0
        self.running_peak = self.initial_balance
        self.current_drawdown = 0.0
        self.drawdown_lambda_floor = 0.0
        self.drawdown_trigger_boundary = 0.0
        self.drawdown_lambda_peak = 0.0
        self.drawdown_triggered = False
        self.drawdown_penalty_reference = "target"
        self.drawdown_cooling_rate = 1.0
        self.drawdown_lambda_carry_decay = 0.7
        self._drawdown_has_episode_history = False

        if drawdown_constraint and drawdown_constraint.get("enabled", False):
            self.drawdown_constraint_enabled = True
            self.drawdown_target = float(drawdown_constraint.get("target", 0.2))
            self.drawdown_tolerance = float(drawdown_constraint.get("tolerance", 0.02))
            self.drawdown_trigger_boundary = max(0.0, self.drawdown_target + self.drawdown_tolerance)
            self.drawdown_dual_lr = float(drawdown_constraint.get("dual_learning_rate", 0.1))
            self.drawdown_lambda_init = float(drawdown_constraint.get("lambda_init", 0.0))
            self.drawdown_lambda = self.drawdown_lambda_init
            self.drawdown_lambda_max = float(drawdown_constraint.get("lambda_max", 5.0))
            self.drawdown_lambda_floor = float(drawdown_constraint.get("lambda_floor", 0.0))
            self.drawdown_penalty_reference = str(
                drawdown_constraint.get("penalty_reference", "target")
            ).strip().lower()
            self.drawdown_cooling_rate = float(drawdown_constraint.get("cooling_rate", 1.0))
            self.drawdown_lambda_carry_decay = float(drawdown_constraint.get("lambda_carry_decay", 0.7))
            penalty_key = "penalty_coef" if "penalty_coef" in drawdown_constraint else "base_coef"
            self.drawdown_penalty_coef = float(drawdown_constraint.get(penalty_key, 1.0))
            self.drawdown_lambda_peak = self.drawdown_lambda
        
        logger.info(f"TAPE Portfolio Environment initialized:")
        logger.info(f"  Mode: {mode}")
        logger.info(f"  Assets: {self.num_assets}")
        logger.info(f"  Days: {self.total_days}")
        logger.info(f"  Features: {self.num_features}")
        logger.info(f"  Initial Balance: ${self.initial_balance:,.2f}")
        logger.info(f"  Reward System: {self.reward_system.upper()}")
        if self.episode_length_limit is not None:
            logger.info(f"  Episode Step Limit: {self.episode_length_limit}")

    def _build_initial_weights(self) -> np.ndarray:
        """Build initial portfolio weights using configuration-driven mode."""
        env_params = self.config.get("environment_params", {})
        training_params = self.config.get("training_params", {})
        mode = env_params.get(
            "initial_allocation_mode",
            training_params.get("initial_allocation_mode", "equal_with_cash"),
        )

        if mode == "equal_assets_with_min_cash":
            min_cash = float(training_params.get("min_cash_position", 0.05))
            cash_seed = float(env_params.get("initial_cash_position", min_cash))
            cash_weight = float(np.clip(cash_seed, min_cash, 0.95))
            risky_weight = (1.0 - cash_weight) / float(self.num_assets)
            weights = np.concatenate(
                [np.full(self.num_assets, risky_weight, dtype=np.float32), np.array([cash_weight], dtype=np.float32)]
            )
            return (weights / np.sum(weights)).astype(np.float32)

        # Default behavior: equal split across risky assets + cash.
        return np.full(self.num_assets + 1, 1.0 / (self.num_assets + 1), dtype=np.float32)
    
    def _build_return_matrix(self):
        """
        Build matrix of asset returns: shape (days, assets)
        
        Calculate returns from raw Close prices, not normalized features.
        CRITICAL: Do NOT use 'LogReturn_1d' from processed_data because
        it's been z-score normalized and exp(z-score) creates astronomical values!
        """
        return_matrix = []
        
        # Build price matrix first
        price_matrix = []
        for date_idx, date in enumerate(self.dates):
            day_data = self.processed_data[self.processed_data['Date'] == date]
            day_data = self._align_day_data_by_ticker(day_data, date=date)
            
            if 'Close' in day_data.columns:
                prices = pd.to_numeric(day_data['Close'], errors='coerce').to_numpy(dtype=np.float64, copy=True)
            else:
                prices = np.full(self.num_assets, np.nan, dtype=np.float64)

            if date_idx > 0:
                prev_prices = np.asarray(price_matrix[-1], dtype=np.float64)
                prices = np.where(np.isnan(prices), prev_prices, prices)
            prices = np.where(np.isnan(prices), 1.0, prices)
            prices = np.where(prices > 0.0, prices, 1.0)
            
            price_matrix.append(prices)
        
        price_matrix = np.array(price_matrix, dtype=np.float32)
        
        # Calculate simple returns from prices
        # return[t] = (price[t] - price[t-1]) / price[t-1]
        for i in range(len(price_matrix)):
            if i == 0:
                # First day: no previous price, use 0 return
                returns = np.zeros(self.num_assets)
            else:
                prev_prices = price_matrix[i-1]
                curr_prices = price_matrix[i]
                
                # Avoid division by zero
                prev_prices = np.where(prev_prices > 0, prev_prices, 1.0)
                
                # Simple return formula
                returns = (curr_prices - prev_prices) / prev_prices
                
                # Clip extreme returns to prevent numerical issues
                returns = np.clip(returns, -0.5, 0.5)  # Max ±50% daily return
            
            return_matrix.append(returns)
        
        self.return_matrix = np.array(return_matrix, dtype=np.float32)
        logger.info(f"Return matrix built: shape {self.return_matrix.shape}")

    def _resolve_feature_phase(self) -> str:
        """
        Resolve the feature phase for observation construction.
        """
        env_params = self.config.get("environment_params", {})
        if isinstance(env_params, dict):
            explicit_phase = str(env_params.get("feature_phase", "")).strip().lower()
            if explicit_phase in ("phase1", "phase2"):
                return explicit_phase

        phase_name = str(self.config.get("phase_name", "")).strip().lower()
        if "phase2" in phase_name:
            return "phase2"
        return "phase1"

    def _sort_day_data_by_ticker(self, day_data: pd.DataFrame) -> pd.DataFrame:
        """
        Enforce stable per-day ticker ordering to keep feature/return alignment deterministic.
        """
        return self._align_day_data_by_ticker(day_data)

    def _align_day_data_by_ticker(self, day_data: pd.DataFrame, date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        """
        Reindex each day to the full ticker list so asset slots stay deterministic even when rows are missing.
        """
        if day_data is None:
            day_data = pd.DataFrame(columns=["Ticker"])

        day_work = day_data.copy()
        if "Ticker" not in day_work.columns:
            day_work["Ticker"] = np.nan
        day_work["Ticker"] = day_work["Ticker"].astype(str)

        ordered_tickers = list(self.ordered_tickers)
        if not ordered_tickers:
            ordered_tickers = sorted(day_work["Ticker"].dropna().unique().tolist())
        ordered_tickers = ordered_tickers[:self.num_assets]
        if len(ordered_tickers) < self.num_assets:
            start_idx = len(ordered_tickers)
            ordered_tickers.extend([f"__MISSING_TICKER_{i}__" for i in range(start_idx, self.num_assets)])

        if not ordered_tickers:
            return day_work

        day_aligned = (
            day_work
            .drop_duplicates(subset=["Ticker"], keep="last")
            .set_index("Ticker")
            .reindex(ordered_tickers)
            .reset_index()
            .rename(columns={"index": "Ticker"})
        )

        if date is not None:
            day_aligned["Date"] = date

        return day_aligned

    def _infer_global_feature_columns(self, feature_cols: List[str]) -> List[str]:
        """
        Identify features that should appear once per timestep as global context.
        """
        env_params = self.config.get("environment_params", {}) if isinstance(self.config.get("environment_params", {}), dict) else {}
        explicit_global = list(env_params.get("global_feature_columns", []) or [])
        explicit_prefixes = list(env_params.get("global_feature_prefixes", []) or [])
        if not explicit_prefixes:
            explicit_prefixes = [
                "Covariance_Eigenvalue_",
                "YieldCurve_",
                "EFFR_",
                "SOFR_",
                "FEDFUNDS_",
                "DGS",
                "T10Y",
                "TIPS",
                "BreakevenInf",
                "IG_Credit_",
                "HY_Credit_",
                "MOVE_",
                "VIX_",
                "FedBalanceSheet_",
                "ON_RRP_",
                "CPI_",
                "PPI_",
                "UNRATE_",
                "PAYEMS_",
                "INDPRO_",
                "ISM_",
            ]

        explicit_set: Set[str] = set()
        for col in feature_cols:
            if col in explicit_global:
                explicit_set.add(col)
                continue
            for prefix in explicit_prefixes:
                if col.startswith(prefix):
                    explicit_set.add(col)
                    break

        inferred_set: Set[str] = set(explicit_set)
        if "Date" in self.processed_data.columns and "Ticker" in self.processed_data.columns:
            grouped = self.processed_data.groupby("Date")
            for col in feature_cols:
                if col in inferred_set or col not in self.processed_data.columns:
                    continue
                try:
                    nunique_per_date = grouped[col].nunique(dropna=True)
                except Exception:
                    continue
                if len(nunique_per_date) == 0:
                    continue
                if int(nunique_per_date.max()) <= 1:
                    inferred_set.add(col)

        return [col for col in feature_cols if col in inferred_set]
    
    def _build_feature_matrix(self):
        """
        Build matrix of features: shape (days, features)
        
        For each date, extract all normalized features for all assets and flatten.
        """
        feature_matrix = []
        
        # Get feature columns using the data processor's filtering logic
        # This respects the config's feature selection settings (disable_features, etc.)
        requested_cols = self.data_processor.get_feature_columns(phase=self.feature_phase)
        feature_cols = list(requested_cols)
        
        # Optionally exclude covariance features
        if self.exclude_covariance:
            covariance_features = ['Covariance_Eigenvalue_0', 'Covariance_Eigenvalue_1', 'Covariance_Eigenvalue_2']
            self.excluded_covariance_columns = [col for col in covariance_features if col in feature_cols]
            feature_cols = [col for col in feature_cols if col not in covariance_features]
        else:
            self.excluded_covariance_columns = []
        
        # Only keep features that actually exist in the processed_data
        self.requested_feature_columns = list(feature_cols)
        self.missing_feature_columns = [col for col in feature_cols if col not in self.processed_data.columns]
        feature_cols = [col for col in feature_cols if col in self.processed_data.columns]
        self.active_feature_columns = list(feature_cols)

        global_cols = self._infer_global_feature_columns(feature_cols)
        asset_cols = [col for col in feature_cols if col not in set(global_cols)]
        if not asset_cols:
            # Safety fallback: never allow empty per-asset channel.
            asset_cols = list(feature_cols)
            global_cols = []

        self.asset_feature_columns = list(asset_cols)
        self.global_feature_columns = list(global_cols)
        self.asset_feature_count_per_asset = len(self.asset_feature_columns)
        self.global_feature_count = len(self.global_feature_columns)

        asset_tensor_matrix = []
        context_matrix = []
        
        for date in self.dates:
            day_data = self.processed_data[self.processed_data['Date'] == date]
            day_data = self._align_day_data_by_ticker(day_data, date=date)

            if self.asset_feature_columns:
                asset_frame = day_data[self.asset_feature_columns].apply(pd.to_numeric, errors='coerce')
                asset_values = asset_frame.to_numpy(dtype=np.float32, copy=True)
            else:
                asset_values = np.zeros((self.num_assets, 0), dtype=np.float32)
            asset_values = np.nan_to_num(asset_values, nan=0.0, posinf=1.0, neginf=-1.0)
            asset_tensor_matrix.append(asset_values)

            if self.global_feature_columns:
                global_frame = day_data[self.global_feature_columns].apply(pd.to_numeric, errors='coerce')
                global_raw = global_frame.to_numpy(dtype=np.float32, copy=True)
                with np.errstate(invalid="ignore"):
                    global_values = np.nanmean(global_raw, axis=0)
                global_values = np.nan_to_num(global_values, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)
            else:
                global_values = np.zeros((0,), dtype=np.float32)
            context_matrix.append(global_values)

            if self.structured_observation:
                day_features = np.concatenate([asset_values.reshape(-1), global_values], axis=0)
            else:
                legacy_frame = day_data[feature_cols].apply(pd.to_numeric, errors='coerce')
                legacy_values = legacy_frame.to_numpy(dtype=np.float32, copy=True)
                legacy_values = np.nan_to_num(legacy_values, nan=0.0, posinf=1.0, neginf=-1.0)
                day_features = legacy_values.reshape(-1)

            feature_matrix.append(day_features)
        
        self.feature_matrix = np.array(feature_matrix, dtype=np.float32)
        self.asset_feature_matrix = np.array(asset_tensor_matrix, dtype=np.float32)
        self.context_feature_matrix = np.array(context_matrix, dtype=np.float32)
        
        # Check for inf/nan BEFORE cleaning
        nan_count = np.isnan(self.feature_matrix).sum()
        inf_count = np.isinf(self.feature_matrix).sum()
        if nan_count > 0 or inf_count > 0:
            logger.warning(f"⚠️  Feature matrix has {nan_count} NaN and {inf_count} inf values BEFORE cleaning")
        
        # Handle any NaN values
        self.feature_matrix = np.nan_to_num(self.feature_matrix, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Verify after cleaning
        nan_after = np.isnan(self.feature_matrix).sum()
        inf_after = np.isinf(self.feature_matrix).sum()
        if nan_after > 0 or inf_after > 0:
            logger.error(f"❌ Feature matrix STILL has {nan_after} NaN and {inf_after} inf values AFTER cleaning!")
        
        logger.info(f"Feature matrix built: shape {self.feature_matrix.shape}")
        if self.structured_observation:
            logger.info(
                "  Observation layout: local=%d x %d + global=%d => %d",
                self.num_assets,
                self.asset_feature_count_per_asset,
                self.global_feature_count,
                (self.num_assets * self.asset_feature_count_per_asset) + self.global_feature_count,
            )
        else:
            logger.info(f"  Expected features per day: {len(feature_cols) * self.num_assets}")
        logger.info(f"  Feature phase: {self.feature_phase} | Active features: {len(self.active_feature_columns)}")
        logger.info(f"  Min/Max values: {self.feature_matrix.min():.4f} / {self.feature_matrix.max():.4f}")

    def get_observation_layout(self) -> Dict[str, Any]:
        """
        Return deterministic state layout metadata for agent-side reshaping.
        """
        local_flat_dim = int(self.num_assets * self.asset_feature_count_per_asset)
        return {
            "structured_observation": bool(self.structured_observation),
            "num_assets": int(self.num_assets),
            "asset_feature_dim": int(self.asset_feature_count_per_asset),
            "global_feature_dim": int(self.global_feature_count),
            "local_flat_dim": local_flat_dim,
            "total_observation_dim": int(self.num_features),
            "asset_feature_columns": list(self.asset_feature_columns),
            "global_feature_columns": list(self.global_feature_columns),
            "active_feature_columns": list(self.active_feature_columns),
        }

    def get_active_feature_manifest(self) -> Dict[str, Any]:
        """
        Return a machine-readable manifest of the active observation features.
        """
        active_count_total = len(self.active_feature_columns)
        return {
            "feature_phase": self.feature_phase,
            "exclude_covariance": bool(self.exclude_covariance),
            "num_assets": int(self.num_assets),
            "requested_feature_count": len(self.requested_feature_columns),
            "active_feature_count_total": active_count_total,
            "active_feature_count_per_asset": int(self.asset_feature_count_per_asset),
            "asset_feature_count_per_asset": int(self.asset_feature_count_per_asset),
            "global_feature_count": int(self.global_feature_count),
            "structured_observation": bool(self.structured_observation),
            "flattened_observation_dim": int(self.num_features),
            "excluded_covariance_columns": list(self.excluded_covariance_columns),
            "missing_requested_columns": list(self.missing_feature_columns),
            "asset_feature_columns": list(self.asset_feature_columns),
            "global_feature_columns": list(self.global_feature_columns),
            "active_feature_columns": list(self.active_feature_columns),
        }

    def save_active_feature_manifest(self, output_path: str) -> None:
        """
        Persist active feature manifest for reproducibility/auditability.
        """
        directory = os.path.dirname(output_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(self.get_active_feature_manifest(), handle, indent=2, default=str)
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment to initial state.
        
        🔧 CRITICAL FIX #4: Proper seed + random_start interaction
        - Always initialize RNG (with seed if provided, or create new one)
        - If mode='train' OR random_start=True: Use RNG to pick random starting day
        - Otherwise: Start from day 0 (deterministic evaluation)
        
        Key insight: Seed controls the RNG state, random_start controls whether to use it!
        
        Design Pattern (Lines ~244-259):
        - Reset day counter
        - Reset portfolio value
        - Reset weights to equal allocation
        - Clear history
        
        Returns:
            Tuple of (observation, info)
        """
        # 🎯 STEP 1: Initialize RNG (ALWAYS, whether seed provided or not)
        if seed is not None:
            self.np_random = np.random.RandomState(seed)
        elif not hasattr(self, 'np_random'):
            self.np_random = np.random.RandomState()
        
        # � STEP 2: Determine starting day based on mode and random_start flag
        if self.mode == 'train' or self.random_start:
            # Random starting point using the initialized RNG.
            # Reserve enough room for the active episode limit so curriculum horizons
            # are respected; fall back to 252 when no explicit limit is set.
            reserve_horizon = int(self.episode_length_limit) if self.episode_length_limit is not None else 252
            reserve_horizon = max(1, reserve_horizon)
            max_start = max(0, len(self.dates) - reserve_horizon)
            self.day = self.np_random.randint(0, max_start + 1) if max_start > 0 else 0
        else:
            # Deterministic: always start from day 0 (for evaluation/testing)
            self.day = 0

        # Track episode start index so logs reflect true episode span
        self.episode_start_day = int(self.day)
        
        self.portfolio_value = self.initial_balance
        
        self.current_weights = self._build_initial_weights()
        
        # Reset memory
        self.portfolio_history = [self.initial_balance]
        self.return_history = [0.0]
        self.weights_history = [self.current_weights.copy()]
        self.date_history = [self.dates[self.day]]
        self.action_realization_l1_history = []
        self.concentration_hhi_history = []
        self.top_weight_history = []
        self.concentration_penalty_sum = 0.0
        self.action_realization_penalty_sum = 0.0
        self.episode_step_count = 0
        
        # 🎯 TAPE System: Reset all three-component buffers
        if self.reward_system == 'tape':
            # Component 2 (DSR/PBRS): Reset rolling window and last Sharpe
            self.dsr_history.clear()
            self.last_sharpe = 0.0
            
            # Terminal TAPE: Reset episode-level tracking
            self.episode_portfolio_values = [self.initial_balance]
            self.episode_return_history = []
            self.episode_weight_changes = []
            self._last_intra_step_tape_potential = 0.0
            self.last_intra_step_tape_potential = None
            self.last_intra_step_tape_delta_reward = 0.0

        self._reset_drawdown_controller_state()
        
        # Get initial observation
        observation = self._get_observation()
        
        info = {
            'portfolio_value': self.portfolio_value,
            'day': self.day,
            'date': self.dates[self.day]
        }

        return observation, info

    def _reset_drawdown_controller_state(self) -> None:
        """Reset dual-controller stats at episode start."""
        self.running_peak = self.initial_balance
        if self.drawdown_constraint_enabled:
            if self._drawdown_has_episode_history:
                decayed_lambda = self.drawdown_lambda * self.drawdown_lambda_carry_decay
                self.drawdown_lambda = max(
                    self.drawdown_lambda_floor,
                    min(decayed_lambda, self.drawdown_lambda_max),
                )
            else:
                self.drawdown_lambda = max(self.drawdown_lambda_init, self.drawdown_lambda_floor)
                self._drawdown_has_episode_history = True
        else:
            self.drawdown_lambda = 0.0
            self._drawdown_has_episode_history = False
        self.drawdown_lambda_peak = self.drawdown_lambda
        self.drawdown_penalty_sum = 0.0
        self.drawdown_excess_accumulator = 0.0
        self.current_drawdown = 0.0
        self.drawdown_triggered = False
        if self.drawdown_constraint_enabled:
            self.drawdown_trigger_boundary = max(0.0, self.drawdown_target + self.drawdown_tolerance)

    def set_episode_length_limit(self, limit: Optional[int]) -> None:
        """Set or clear the maximum number of steps per episode."""
        self.episode_length_limit = int(limit) if limit is not None else None

    def _compute_intra_step_tape_potential(self) -> Optional[float]:
        """
        Compute a rolling potential proxy used for intra-step TAPE delta shaping.
        """
        if (
            self.reward_system != 'tape'
            or not self.intra_step_tape_delta_enabled
            or self.tape_profile is None
        ):
            return None

        history_len = len(self.episode_return_history)
        min_history = max(2, int(self.intra_step_tape_delta_min_history))
        if history_len < min_history:
            return None

        window = max(2, int(self.intra_step_tape_delta_window))
        use_len = min(window, history_len)
        values_len = use_len + 1
        if len(self.episode_portfolio_values) < values_len:
            return None

        returns_window = np.array(self.episode_return_history[-use_len:], dtype=np.float32)
        values_window = np.array(self.episode_portfolio_values[-values_len:], dtype=np.float32)
        weight_changes_window = self.episode_weight_changes[-use_len:]
        metrics = calculate_episode_metrics(
            portfolio_values=values_window,
            returns=returns_window,
            weight_changes=weight_changes_window,
            risk_free_rate=0.02,
            trading_days_per_year=252,
        )
        potential = calculate_tape_score(metrics=metrics, profile=self.tape_profile)
        potential = np.nan_to_num(float(potential), nan=0.0, posinf=1.0, neginf=0.0)
        return float(np.clip(potential, 0.0, 1.0))
    
    def _get_reward(self, portfolio_return: float, transaction_cost: float,
                    old_portfolio_value: float, actual_turnover_this_step: float) -> float:
        """
        Calculate step-level reward based on the configured reward system.
        
        For TAPE system, implements three-component reward:
        1. Base Reward: Net return (portfolio_return - transaction_cost_fraction)
        2. DSR/PBRS Component: Potential-based reward shaping using Differential Sharpe
        3. Turnover Ceiling: Penalty for exceeding maximum acceptable turnover (prevents churning)
        
        Args:
            portfolio_return: Gross portfolio return (before transaction costs)
            transaction_cost: Absolute transaction cost in dollars
            old_portfolio_value: Portfolio value before this step
            actual_turnover_this_step: Sum of absolute weight changes
            
        Returns:
            float: Reward for this step
        """
        
        # --- SYSTEM 1: Simple Reward ---
        if self.reward_system == 'simple':
            reward = float(portfolio_return * 100.0)
            return np.nan_to_num(reward, nan=0.0, posinf=10.0, neginf=-10.0)
        
        # --- SYSTEM 2: TAPE Three-Component Reward ---
        elif self.reward_system == 'tape':
            
            # ═══════════════════════════════════════════════════════════
            # COMPONENT 1: Base Reward (Net Return)
            # ═══════════════════════════════════════════════════════════
            # Net return already reflects transaction costs (embedded in portfolio_return)
            if self.enable_base_reward:
                base_reward = float(portfolio_return * 100.0)
                base_reward = np.nan_to_num(base_reward, nan=0.0, posinf=10.0, neginf=-10.0)
            else:
                # Component 1 disabled: Agent learns purely from DSR + Turnover
                base_reward = 0.0
            
            # ═══════════════════════════════════════════════════════════
            # COMPONENT 2: DSR / PBRS Component (Differential Sharpe)
            # ═══════════════════════════════════════════════════════════
            # Add gross return to rolling window (use gross for Sharpe calculation)
            self.dsr_history.append(portfolio_return)
            dsr_component = 0.0
            
            if len(self.dsr_history) >= self.dsr_window:
                # Buffer is full - calculate current Sharpe ratio
                returns_arr = np.array(self.dsr_history)
                current_sharpe = calculate_sharpe_ratio_dsr(returns_arr, trading_days_per_year=252)
                
                # PBRS formula: R_shaped = R + gamma * P(s') - P(s)
                # Where P(s) is the potential function (Sharpe ratio)
                differential_sharpe = (self.gamma * current_sharpe) - self.last_sharpe
                self.last_sharpe = self.gamma * current_sharpe  # Update for next step
                
                # Scale differential Sharpe to meaningful reward magnitude
                dsr_component = differential_sharpe * self.dsr_scalar
                dsr_component = np.nan_to_num(dsr_component, nan=0.0, posinf=10.0, neginf=-10.0)
            
            # ═══════════════════════════════════════════════════════════
            # COMPONENT 3: Turnover Soft Ceiling (One-Sided Penalty)
            # ═══════════════════════════════════════════════════════════
            # Industry-aligned: only penalize EXCESS turnover above a ceiling.
            # Never reward trading — transaction costs in the base return already
            # make the agent want to trade less. This is a safety rail for churning.
            
            if self.target_turnover_per_step > 0:
                if actual_turnover_this_step > self.target_turnover_per_step:
                    # Proportional penalty for exceeding ceiling
                    excess_ratio = (actual_turnover_this_step - self.target_turnover_per_step) / max(self.target_turnover_per_step, 1e-8)
                    turnover_reward = -excess_ratio * self.turnover_penalty_scalar
                else:
                    # Under ceiling → no reward, no penalty
                    turnover_reward = 0.0
            else:
                turnover_reward = 0.0
            
            # ═══════════════════════════════════════════════════════════
            # FINAL DAILY REWARD
            # ═══════════════════════════════════════════════════════════
            # Combine: Base + DSR Guidance + Turnover Proximity Reward
            final_step_reward = base_reward + dsr_component + turnover_reward

            # Optional: small rolling potential-delta shaping for denser learning signal.
            intra_step_delta_reward = 0.0
            if self.intra_step_tape_delta_enabled:
                potential_now = self._compute_intra_step_tape_potential()
                if potential_now is not None:
                    potential_delta = potential_now - self._last_intra_step_tape_potential
                    intra_step_delta_reward = potential_delta * self.intra_step_tape_delta_beta
                    clip_bound = abs(self.intra_step_tape_delta_clip)
                    if clip_bound > 0.0:
                        intra_step_delta_reward = float(np.clip(intra_step_delta_reward, -clip_bound, clip_bound))
                    self._last_intra_step_tape_potential = float(potential_now)
                    self.last_intra_step_tape_potential = float(potential_now)
                else:
                    self.last_intra_step_tape_potential = None
            self.last_intra_step_tape_delta_reward = float(intra_step_delta_reward)
            final_step_reward += intra_step_delta_reward
            
            # Clip to prevent extreme values (wider range for strong turnover signal)
            final_step_reward = np.clip(final_step_reward, -150.0, 150.0)
            return np.nan_to_num(final_step_reward, nan=0.0, posinf=150.0, neginf=-150.0)
        
        else:
            raise ValueError(f"Unknown reward_system: {self.reward_system}")
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        TAPE ENVIRONMENT IMPLEMENTATION:
        ==============================
        
        1. **Termination Check** (Reference Line ~128):
           terminal = self.day >= len(self.df.index.unique()) - 1
           
        2. **Action Normalization** (Reference Lines ~172-176):
           weights = softmax_normalization(actions)
           
        3. **Portfolio Return Calculation** (Reference Lines ~195-200):
           portfolio_return = sum((price_new / price_old - 1) * weights)
           new_portfolio_value = portfolio_value * (1 + portfolio_return)
           
        4. **Reward Assignment** (Reference Line ~210):
           self.reward = new_portfolio_value
           
        Args:
            action: Array of target portfolio weights
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        
        # Track step count for optional episode truncation
        self.episode_step_count = getattr(self, "episode_step_count", 0) + 1

        # ═══════════════════════════════════════════════════════════════
        # STEP 1: TERMINATION CHECK (Design Pattern)
        # ═══════════════════════════════════════════════════════════════
        # Reference Line ~128: self.terminal = self.day >= len(self.df.index.unique()) - 1

        terminated = self.day >= self.total_days - 1
        limit_hit = False
        if (
            self.episode_length_limit is not None
            and self.episode_step_count >= self.episode_length_limit
        ):
            terminated = True
            limit_hit = True

        if terminated:
            # Episode complete - return final state
            observation = self._get_observation()
            
            # Calculate comprehensive episode metrics
            returns_array = np.array(self.return_history[1:]) if len(self.return_history) > 1 else np.array([0.0])
            
            # Sharpe Ratio (annualized)
            sharpe_ratio_final = 0.0
            if len(returns_array) > 0 and returns_array.std() > 0:
                sharpe_ratio_final = (252 ** 0.5) * returns_array.mean() / returns_array.std()
            
            # Sortino Ratio (downside risk only)
            sortino_ratio = 0.0
            downside_returns = returns_array[returns_array < 0]
            if len(downside_returns) > 0:
                downside_std = downside_returns.std()
                if downside_std > 0:
                    sortino_ratio = (252 ** 0.5) * returns_array.mean() / downside_std
            
            # Maximum Drawdown (MDD)
            portfolio_values = np.array(self.portfolio_history)
            running_max = np.maximum.accumulate(portfolio_values)
            drawdowns = (portfolio_values - running_max) / running_max
            max_drawdown = abs(drawdowns.min()) if len(drawdowns) > 0 else 0.0
            
            # Volatility (annualized)
            volatility = returns_array.std() * np.sqrt(252) if len(returns_array) > 0 else 0.0
            
            # Win Rate
            win_rate = (returns_array > 0).sum() / len(returns_array) if len(returns_array) > 0 else 0.0
            
            # Total Return
            total_return_pct = (self.portfolio_value / self.initial_balance - 1) * 100
            
            # Compact single-line logging format
            logger.info("=" * 120)
            termination_label = 'episode_limit' if limit_hit else 'data_exhausted'
            start_idx = int(getattr(self, 'episode_start_day', 0))
            logger.info(
                f"EPISODE COMPLETE | StartIdx: {start_idx:4d} | EndIdx: {self.day:4d} | "
                f"Steps: {self.episode_step_count:4d} | End: {termination_label:>13} | "
                f"Initial: ${self.portfolio_history[0]:>10,.2f} | "
                f"Final: ${self.portfolio_value:>10,.2f} | "
                f"Return: {total_return_pct:>7.2f}% | "
                f"Volatility: {volatility*100:>6.2f}% | "
                f"Sharpe: {sharpe_ratio_final:>6.3f} | "
                f"Sortino: {sortino_ratio:>6.3f} | "
                f"MDD: {max_drawdown*100:>6.2f}% | "
                f"WinRate: {win_rate*100:>5.1f}%"
            )
            if self.drawdown_constraint_enabled:
                avg_drawdown_excess = self.drawdown_excess_accumulator / max(1, self.episode_step_count)
                logger.info(
                    "DRAWDOWN CTRL | "
                    f"λ={self.drawdown_lambda:.3f} | "
                    f"λ_peak={self.drawdown_lambda_peak:.3f} | "
                    f"current={self.current_drawdown*100:.2f}% | "
                    f"trigger={self.drawdown_trigger_boundary*100:.2f}% | "
                    f"avg_excess={avg_drawdown_excess*100:.2f}% | "
                    f"penalty_sum={self.drawdown_penalty_sum:.4f} | "
                    f"regime_mult={self._get_drawdown_regime_multiplier():.2f} | "
                    f"triggered={self.drawdown_triggered}"
                )
            logger.info("=" * 120)
            
            # 🎯 TAPE Terminal Bonus: Holistic episode-level "tie-breaker"
            tape_score_final = None
            tape_bonus_final = None
            tape_bonus_raw_final = None
            if self.reward_system == 'tape':
                # Calculate episode-level metrics for TAPE scoring
                episode_metrics = calculate_episode_metrics(
                    portfolio_values=np.array(self.episode_portfolio_values),
                    returns=np.array(self.episode_return_history),
                    weight_changes=self.episode_weight_changes,  # Already contains turnover values
                    risk_free_rate=0.02,
                    trading_days_per_year=252
                )
                
                # Calculate TAPE score using the active profile (0 to 1)
                tape_score = calculate_tape_score(
                    metrics=episode_metrics,
                    profile=self.tape_profile
                )
                
                # Calculate terminal bonus (moderate scalar acts as final "nudge")
                terminal_bonus = tape_score * self.tape_terminal_scalar
                unclipped_bonus = terminal_bonus
                if self.tape_terminal_clip is not None:
                    terminal_bonus = float(np.clip(
                        terminal_bonus,
                        -self.tape_terminal_clip,
                        self.tape_terminal_clip
                    ))
                    if terminal_bonus != unclipped_bonus:
                        logger.info(
                            f"   Terminal bonus clipped from {unclipped_bonus:.2f} "
                            f"to {terminal_bonus:.2f} (clip ±{self.tape_terminal_clip})"
                        )
                
                # Set terminal reward (no step reward on final step, only bonus)
                reward = terminal_bonus
                tape_bonus_final = float(terminal_bonus)
                tape_bonus_raw_final = float(unclipped_bonus)
                
                logger.info(f"🎯 TAPE Terminal Bonus")
                logger.info(f"   TAPE Score: {tape_score:.4f} × {self.tape_terminal_scalar} = {terminal_bonus:.2f}")
                logger.info(f"   Metrics: Sharpe={episode_metrics.get('sharpe_ratio', 0):.3f}, "
                          f"Sortino={episode_metrics.get('sortino_ratio', 0):.3f}, "
                          f"MDD={episode_metrics.get('max_drawdown', 0)*100:.2f}%, "
                          f"Turnover={episode_metrics.get('turnover', 0)*100:.2f}%, "
                          f"Skew={episode_metrics.get('skewness', 0):.3f}")
                
                # Set info values for logging (no retrospective scaling)
                tape_score_final = tape_score
            else:
                # Simple reward system: Use terminal_reward_metric
                if self.terminal_reward_metric == 'return':
                    # Original method: Reward based on final return
                    final_return = (self.portfolio_value - self.initial_balance) / self.initial_balance
                    final_return = np.clip(final_return, -1.0, 10.0)  # Allow large positive returns
                    reward = float(final_return * 100.0)
                    reward = np.nan_to_num(reward, nan=0.0, posinf=1000.0, neginf=-100.0)
                
                elif self.terminal_reward_metric == 'sharpe':
                    # New method: Reward based on annualized Sharpe ratio
                    returns = np.array(self.return_history[1:])  # Exclude initial 0
                    
                    # Safeguard: Handle zero or negligible standard deviation
                    if len(returns) == 0 or np.std(returns, ddof=1) < 1e-6:
                        sharpe_ratio = 0.0
                    else:
                        # Calculate annualized Sharpe ratio
                        sharpe_ratio = (np.mean(returns) / np.std(returns, ddof=1)) * np.sqrt(252)
                    
                    # Scale by 50.0 for numerical stability
                    reward = float(sharpe_ratio * 50.0)
                    
                    # Apply same clipping and NaN handling as 'return' method for consistency
                    reward = np.clip(reward, -100.0, 1000.0)
                    reward = np.nan_to_num(reward, nan=0.0, posinf=1000.0, neginf=-100.0)
                
                else:
                    # Fallback to 'return' method if invalid metric specified
                    logger.warning(f"Invalid terminal_reward_metric '{self.terminal_reward_metric}', using 'return'")
                    final_return = (self.portfolio_value - self.initial_balance) / self.initial_balance
                    final_return = np.clip(final_return, -1.0, 10.0)
                    reward = float(final_return * 100.0)
                    reward = np.nan_to_num(reward, nan=0.0, posinf=1000.0, neginf=-100.0)
                
                # For simple system, set dummy value for TAPE score
                tape_score_final = None
            
            info = {
                'portfolio_value': self.portfolio_value,
                'day': self.day,
                'episode_start_day': int(getattr(self, 'episode_start_day', 0)),
                'episode_end_day': int(self.day),
                'episode_complete': True,
                'sharpe_ratio': float(sharpe_ratio_final),
                'sortino_ratio': float(sortino_ratio),
                'max_drawdown': float(max_drawdown),
                'volatility': float(volatility),
                'win_rate': float(win_rate),
                'total_return_pct': float(total_return_pct),
                'initial_balance': float(self.portfolio_history[0]),
                'final_balance': float(self.portfolio_value),
                # 🎯 TAPE System: Return score for logging (no retrospective scaling)
                'tape_score': tape_score_final,  # TAPE score 0-1 (or None if simple system)
                'tape_bonus': tape_bonus_final,
                'tape_bonus_raw': tape_bonus_raw_final,
                'intra_step_tape_potential': self.last_intra_step_tape_potential,
                'intra_step_tape_delta_reward': self.last_intra_step_tape_delta_reward,
                'termination_reason': 'episode_limit' if limit_hit else 'data_exhausted',
                'episode_length': self.episode_step_count,
                'mean_concentration_hhi': float(np.mean(self.concentration_hhi_history)) if self.concentration_hhi_history else 0.0,
                'mean_top_weight': float(np.mean(self.top_weight_history)) if self.top_weight_history else 0.0,
                'mean_action_realization_l1': float(np.mean(self.action_realization_l1_history)) if self.action_realization_l1_history else 0.0,
                'concentration_penalty_sum': float(self.concentration_penalty_sum),
                'action_realization_penalty_sum': float(self.action_realization_penalty_sum),
            }

            if self.drawdown_constraint_enabled:
                avg_drawdown_excess = self.drawdown_excess_accumulator / max(1, self.episode_step_count)
                info.update(
                    {
                        'drawdown_lambda': self.drawdown_lambda,
                        'drawdown_lambda_peak': self.drawdown_lambda_peak,
                        'drawdown_penalty_sum': self.drawdown_penalty_sum,
                        'drawdown_avg_excess': avg_drawdown_excess,
                        'drawdown_current': self.current_drawdown,
                        'drawdown_target': self.drawdown_target,
                        'drawdown_tolerance': self.drawdown_tolerance,
                        'drawdown_trigger_boundary': self.drawdown_trigger_boundary,
                    }
                )
            else:
                info.update(
                    {
                        'drawdown_lambda': 0.0,
                        'drawdown_lambda_peak': 0.0,
                        'drawdown_penalty_sum': 0.0,
                        'drawdown_avg_excess': 0.0,
                        'drawdown_current': 0.0,
                        'drawdown_target': None,
                        'drawdown_tolerance': None,
                        'drawdown_trigger_boundary': None,
                    }
                )

            return observation, reward, terminated, limit_hit, info
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 2: ACTION NORMALIZATION (Design Pattern + Dirichlet Option)
        # ═══════════════════════════════════════════════════════════════
        # Reference Lines ~172-176: weights = softmax_normalization(actions)
        
        action = np.array(action, dtype=np.float32)
        
        # Choose normalization method
        if self.action_normalization == 'softmax':
            # Softmax: Deterministic, smooth, used by TAPE
            weights = self._softmax_normalization(action)
        elif self.action_normalization == 'dirichlet':
            # Dirichlet: Stochastic sampling with exploration
            weights = self._dirichlet_normalization(action)
        elif self.action_normalization == 'none':
            # None: Use action directly (assumes agent outputs normalized weights)
            # CRITICAL: Agent must output weights that sum to 1.0!
            weights = action.copy()
            # Validate and re-normalize if needed
            weight_sum = np.sum(weights)
            if not np.isclose(weight_sum, 1.0, atol=1e-6):
                logger.warning(f"⚠️  Action sum = {weight_sum:.6f} (expected 1.0). Re-normalizing.")
                weights = weights / weight_sum
            # Ensure non-negative
            if np.any(weights < 0):
                logger.warning(f"⚠️  Negative weights detected: {weights}. Clipping to 0.")
                weights = np.maximum(weights, 0.0)
                weights = weights / np.sum(weights)  # Re-normalize after clipping
        else:
            # Fallback (should never reach here due to __init__ validation)
            weights = self._softmax_normalization(action)
        
        # 🔥 CRITICAL: Check for NaN/Inf in weights
        if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
            logger.error(f"⚠️  NaN/Inf detected in weights! Action: {action}, Weights: {weights}")
            weights = np.ones(self.num_assets + 1) / (self.num_assets + 1)  # Fallback to equal weights
            logger.error(f"   Falling back to equal weights: {weights}")
        
        proposed_weights = weights.copy()
        # ═══════════════════════════════════════════════════════════════
        # POSITION CONSTRAINTS (for realistic portfolio management)
        # ═══════════════════════════════════════════════════════════════
        # Project directly into the feasible set to reduce action-vs-realized mismatch.
        max_single_position_raw = float(self.config.get('training_params', {}).get('max_single_position', 40.0))
        max_single_position = (
            max_single_position_raw / 100.0 if max_single_position_raw > 1.0 else max_single_position_raw
        )
        min_cash_position = float(self.config.get('training_params', {}).get('min_cash_position', 0.05))

        weights = self._project_weights_to_constraints(
            weights,
            max_single_position=max_single_position,
            min_cash_position=min_cash_position,
        )

        # Track concentration and action-realization mismatch diagnostics.
        risky_weights = weights[:-1] if len(weights) > 1 else weights
        concentration_hhi = float(np.sum(np.square(risky_weights))) if len(risky_weights) else 0.0
        top_weight = float(np.max(risky_weights)) if len(risky_weights) else 0.0
        action_realization_l1 = float(np.sum(np.abs(weights - proposed_weights)))
        self.concentration_hhi_history.append(concentration_hhi)
        self.top_weight_history.append(top_weight)
        self.action_realization_l1_history.append(action_realization_l1)
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 3: SAVE CURRENT STATE (for return calculation)
        # ═══════════════════════════════════════════════════════════════
        
        last_portfolio_value = self.portfolio_value
        last_weights = self.current_weights.copy()
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 4: ADVANCE TO NEXT DAY
        # ═══════════════════════════════════════════════════════════════
        # Reference Line ~187: self.day += 1
        
        self.day += 1
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 5: CALCULATE PORTFOLIO RETURN (Design Pattern)
        # ═══════════════════════════════════════════════════════════════
        # Reference Lines ~195-200:
        # portfolio_return = sum(((data.close.values / last_day_memory.close.values) - 1) * weights)
        # new_portfolio_value = portfolio_value * (1 + portfolio_return)
        
        if self.day < len(self.return_matrix):
            # Get asset simple returns for current day (already calculated from Close prices)
            asset_simple_returns = self.return_matrix[self.day]
            
            # Cash has 0 return
            cash_return = 0.0
            all_returns = np.append(asset_simple_returns, cash_return)
            
            # TAPE Portfolio Return Formula:
            # portfolio_return = sum(asset_returns * weights)
            portfolio_return = np.sum(all_returns * weights)
            
            # 🔥 CRITICAL: Check for NaN/Inf in portfolio return
            if np.isnan(portfolio_return) or np.isinf(portfolio_return):
                logger.error(f"⚠️  NaN/Inf in portfolio_return! Day: {self.day}")
                logger.error(f"   Returns: {all_returns}")
                logger.error(f"   Weights: {weights}")
                logger.error(f"   Portfolio value: {self.portfolio_value}")
                portfolio_return = 0.0  # Fallback to 0 return
            
            # TAPE Portfolio Value Update:
            # new_portfolio_value = portfolio_value * (1 + portfolio_return)
            new_portfolio_value = self.portfolio_value * (1.0 + portfolio_return)
            
            # 🔥 CRITICAL: Check for NaN/Inf in new portfolio value
            if np.isnan(new_portfolio_value) or np.isinf(new_portfolio_value):
                logger.error(f"⚠️  NaN/Inf in new_portfolio_value after return calculation!")
                logger.error(f"   Old value: {self.portfolio_value}, Return: {portfolio_return}")
                new_portfolio_value = self.portfolio_value  # Fallback to previous value
            
        else:
            # No more data - shouldn't reach here due to termination check
            portfolio_return = 0.0
            new_portfolio_value = self.portfolio_value
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 6: APPLY TRANSACTION COSTS
        # ═══════════════════════════════════════════════════════════════
        # NOTE: TAPE doesn't explicitly model transaction costs in the
        # portfolio environment, but we keep this for realism
        
        # Calculate turnover: sum of absolute weight changes
        turnover = np.sum(np.abs(weights - last_weights))
        
        # Transaction costs = rate * portfolio_value * turnover
        transaction_costs = self.transaction_cost_rate * new_portfolio_value * turnover
        
        # Subtract transaction costs from portfolio value
        new_portfolio_value -= transaction_costs
        
        # 🔥 CRITICAL: Check for NaN/Inf after transaction costs
        if np.isnan(new_portfolio_value) or np.isinf(new_portfolio_value):
            logger.error(f"⚠️  NaN/Inf after transaction costs!")
            logger.error(f"   Transaction costs: {transaction_costs}, Turnover: {turnover}")
            new_portfolio_value = self.portfolio_value  # Fallback
        
        # Ensure portfolio value doesn't go negative or become NaN
        if np.isnan(new_portfolio_value):
            new_portfolio_value = self.initial_balance  # Reset to initial
            logger.error(f"⚠️  Portfolio value was NaN, reset to initial balance")
        else:
            new_portfolio_value = max(new_portfolio_value, 1.0)
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 7: UPDATE PORTFOLIO STATE
        # ═══════════════════════════════════════════════════════════════
        
        self.portfolio_value = new_portfolio_value
        self.current_weights = weights.copy()
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 8: CALCULATE REWARD (Design Pattern + TAPE Enhancement)
        # ═══════════════════════════════════════════════════════════════
        # Reference Line ~210: self.reward = new_portfolio_value
        #
        # For numerical stability in PPO training, we use the PORTFOLIO RETURN
        # (percentage change) scaled by 100, matching our successful benchmark.
        # This keeps rewards in a reasonable range (-100 to +100).
        #
        # TAPE Enhancement: Optionally apply Stage 1 risk filtering for large losses.
        
        portfolio_return = (new_portfolio_value - last_portfolio_value) / last_portfolio_value
        portfolio_return = np.clip(portfolio_return, -1.0, 1.0)  # Clip extreme returns
        
        # 🎯 TAPE System: Track episode-level data for terminal metrics
        if self.reward_system == 'tape':
            self.episode_portfolio_values.append(new_portfolio_value)
            self.episode_return_history.append(portfolio_return)
            self.episode_weight_changes.append(turnover)  # Store turnover for this step
        
        # Calculate reward using new three-component system
        reward = self._get_reward(
            portfolio_return=portfolio_return,
            transaction_cost=transaction_costs,
            old_portfolio_value=last_portfolio_value,
            actual_turnover_this_step=turnover
        )
        reward_before_penalties = float(reward)

        # Anti-concentration and action-realization alignment penalties
        concentration_penalty = 0.0
        if self.concentration_penalty_scalar > 0.0:
            concentration_penalty += self.concentration_penalty_scalar * max(
                0.0, concentration_hhi - self.concentration_target_hhi
            )
        if self.top_weight_penalty_scalar > 0.0:
            concentration_penalty += self.top_weight_penalty_scalar * max(
                0.0, top_weight - self.target_top_weight
            )
        action_realization_penalty = self.action_realization_penalty_scalar * action_realization_l1

        drawdown_penalty = 0.0
        avg_drawdown_excess = 0.0
        current_drawdown = 0.0
        drawdown_regime_multiplier = 1.0
        drawdown_penalty_raw = 0.0
        if self.drawdown_constraint_enabled:
            drawdown_penalty, current_drawdown, _, drawdown_regime_multiplier = self._apply_drawdown_dual_controller()
            drawdown_penalty_raw = float(drawdown_penalty)
            avg_drawdown_excess = self.drawdown_excess_accumulator / max(1, self.episode_step_count)

        total_penalty = concentration_penalty + action_realization_penalty + drawdown_penalty
        if self.penalty_budget_ratio > 0.0 and total_penalty > 0.0:
            base_signal_magnitude = max(abs(reward_before_penalties), 0.1)
            max_allowed_penalty = base_signal_magnitude * self.penalty_budget_ratio
            if total_penalty > max_allowed_penalty:
                scale_factor = max_allowed_penalty / max(total_penalty, 1e-8)
                concentration_penalty *= scale_factor
                action_realization_penalty *= scale_factor
                drawdown_penalty *= scale_factor
                if self.drawdown_constraint_enabled and drawdown_penalty_raw > 0.0:
                    self.drawdown_penalty_sum -= max(0.0, drawdown_penalty_raw - drawdown_penalty)
                total_penalty = max_allowed_penalty

        reward = reward_before_penalties - total_penalty
        reward = np.clip(reward, -150.0, 150.0)
        self.concentration_penalty_sum += concentration_penalty
        self.action_realization_penalty_sum += action_realization_penalty
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 9: SAVE TO MEMORY
        # ═══════════════════════════════════════════════════════════════
        
        self.portfolio_history.append(self.portfolio_value)
        self.return_history.append(portfolio_return)
        self.weights_history.append(self.current_weights.copy())
        if self.day < len(self.dates):
            self.date_history.append(self.dates[self.day])
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 10: GET NEXT OBSERVATION
        # ═══════════════════════════════════════════════════════════════
        
        observation = self._get_observation()
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 11: CREATE INFO DICTIONARY
        # ═══════════════════════════════════════════════════════════════
        
        # Calculate current Sharpe ratio (useful for monitoring during episode)
        current_sharpe = 0.0
        if len(self.return_history) > 1:
            returns_array = np.array(self.return_history[1:])  # Exclude initial 0
            if returns_array.std() > 0:
                current_sharpe = (252 ** 0.5) * returns_array.mean() / returns_array.std()
        
        info = {
            'portfolio_value': self.portfolio_value,
            'weights': self.current_weights.copy(),
            'portfolio_return': portfolio_return,
            'turnover': turnover,
            'transaction_costs': transaction_costs,
            'day': self.day,
            'date': self.dates[self.day] if self.day < len(self.dates) else None,
            'total_return': (self.portfolio_value / self.initial_balance) - 1.0,
            'sharpe_ratio': float(current_sharpe),  # ✅ FIX: Include current Sharpe ratio
            'episode_step': self.episode_step_count,
            'episode_length_limit': self.episode_length_limit,
            'drawdown_lambda': self.drawdown_lambda if self.drawdown_constraint_enabled else 0.0,
            'drawdown_penalty': drawdown_penalty,
            'drawdown_penalty_sum': self.drawdown_penalty_sum,
            'drawdown_avg_excess': avg_drawdown_excess,
            'drawdown_current': current_drawdown,
            'drawdown_target': self.drawdown_target if self.drawdown_constraint_enabled else None,
            'drawdown_tolerance': self.drawdown_tolerance if self.drawdown_constraint_enabled else None,
            'drawdown_trigger_boundary': self.drawdown_trigger_boundary if self.drawdown_constraint_enabled else None,
            'drawdown_lambda_peak': self.drawdown_lambda_peak if self.drawdown_constraint_enabled else 0.0,
            'drawdown_regime_multiplier': drawdown_regime_multiplier,
            'tape_score': None,
            'tape_bonus': None,
            'tape_bonus_raw': None,
            'intra_step_tape_potential': self.last_intra_step_tape_potential,
            'intra_step_tape_delta_reward': self.last_intra_step_tape_delta_reward,
            'concentration_hhi': concentration_hhi,
            'top_weight': top_weight,
            'concentration_penalty': concentration_penalty,
            'concentration_penalty_sum': self.concentration_penalty_sum,
            'action_realization_l1': action_realization_l1,
            'action_realization_penalty': action_realization_penalty,
            'action_realization_penalty_sum': self.action_realization_penalty_sum,
        }
        
        return observation, reward, terminated, False, info

    def _apply_drawdown_dual_controller(self) -> Tuple[float, float, float, float]:
        """
        Update the drawdown dual controller and return (penalty, current drawdown, excess, regime_multiplier).
        """
        if not self.drawdown_constraint_enabled:
            return 0.0, 0.0, 0.0, 1.0

        self.running_peak = max(self.running_peak, self.portfolio_value)
        if self.running_peak <= 0:
            self.running_peak = self.initial_balance

        current_drawdown = max(0.0, 1.0 - (self.portfolio_value / self.running_peak))
        tolerance_boundary = self.drawdown_trigger_boundary

        if current_drawdown > tolerance_boundary:
            overshoot = current_drawdown - tolerance_boundary
            adjustment = overshoot * self.drawdown_dual_lr
            self.drawdown_lambda = min(self.drawdown_lambda + adjustment, self.drawdown_lambda_max)
            self.drawdown_triggered = True
        else:
            cooling = (
                (tolerance_boundary - current_drawdown)
                * self.drawdown_dual_lr
                * max(self.drawdown_cooling_rate, 0.0)
            )
            if cooling > 0:
                self.drawdown_lambda = max(self.drawdown_lambda - cooling, 0.0)
        if self.drawdown_lambda_floor > 0.0:
            self.drawdown_lambda = max(self.drawdown_lambda, self.drawdown_lambda_floor)

        reference = self.drawdown_penalty_reference
        if reference in {"trigger", "trigger_boundary"}:
            excess_anchor = tolerance_boundary
        else:
            excess_anchor = self.drawdown_target
        excess = max(0.0, current_drawdown - excess_anchor)
        drawdown_regime_multiplier = self._get_drawdown_regime_multiplier()
        penalty = (
            self.drawdown_penalty_coef
            * self.drawdown_lambda
            * excess
            * drawdown_regime_multiplier
        )
        self.drawdown_penalty_sum += penalty
        self.drawdown_excess_accumulator += excess
        self.current_drawdown = current_drawdown
        self.drawdown_lambda_peak = max(self.drawdown_lambda_peak, self.drawdown_lambda)

        return penalty, current_drawdown, excess, drawdown_regime_multiplier
    
    def _get_drawdown_regime_multiplier(self) -> float:
        """
        Compute regime-aware multiplier for drawdown penalties using recent realized volatility.
        """
        if not self.dd_regime_scaling_enabled:
            return 1.0

        returns = np.array(self.return_history[1:], dtype=np.float64)
        if returns.size == 0:
            return self.dd_regime_mid_mult

        window = max(1, min(self.dd_regime_vol_window, returns.size))
        recent = returns[-window:]
        realized_vol_ann = float(np.std(recent) * np.sqrt(252.0))

        if realized_vol_ann >= self.dd_regime_high_vol_threshold:
            return self.dd_regime_high_mult
        if realized_vol_ann <= self.dd_regime_low_vol_threshold:
            return self.dd_regime_low_mult
        return self.dd_regime_mid_mult

    def _project_weights_to_constraints(
        self,
        weights: np.ndarray,
        *,
        max_single_position: float,
        min_cash_position: float,
    ) -> np.ndarray:
        """
        Project portfolio weights to a feasible region:
        - nonnegative
        - sum to 1
        - risky asset cap per asset
        - minimum cash allocation
        """
        projected = np.asarray(weights, dtype=np.float64).copy()
        if projected.ndim != 1 or projected.size != self.num_assets + 1:
            projected = np.ones(self.num_assets + 1, dtype=np.float64) / (self.num_assets + 1)

        projected = np.nan_to_num(projected, nan=0.0, posinf=0.0, neginf=0.0)
        projected = np.maximum(projected, 0.0)

        total = float(np.sum(projected))
        if total <= 1e-12:
            projected = np.ones(self.num_assets + 1, dtype=np.float64) / (self.num_assets + 1)
        else:
            projected /= total

        max_single_position = float(np.clip(max_single_position, 0.0, 1.0))
        min_cash_position = float(np.clip(min_cash_position, 0.0, 1.0))

        risky = projected[:-1].copy()
        cash_target = max(float(projected[-1]), min_cash_position)

        max_risky_sum = min(1.0 - min_cash_position, self.num_assets * max_single_position)
        desired_risky_sum = min(max(0.0, 1.0 - cash_target), max_risky_sum)

        risky_sum = float(np.sum(risky))
        if risky_sum <= 1e-12:
            risky = np.ones(self.num_assets, dtype=np.float64) / max(1, self.num_assets)
            risky_sum = float(np.sum(risky))
        risky = risky / risky_sum * desired_risky_sum

        # Capped-simplex style redistribution.
        for _ in range(self.num_assets + 2):
            capped_mask = risky > (max_single_position + 1e-12)
            if not np.any(capped_mask):
                break
            excess = float(np.sum(risky[capped_mask] - max_single_position))
            risky[capped_mask] = max_single_position

            free_mask = ~capped_mask
            free_count = int(np.sum(free_mask))
            if free_count == 0 or excess <= 1e-12:
                break

            free_total = float(np.sum(risky[free_mask]))
            if free_total <= 1e-12:
                risky[free_mask] += excess / free_count
            else:
                risky[free_mask] += excess * (risky[free_mask] / free_total)

        risky = np.clip(risky, 0.0, max_single_position)
        risky_total = float(np.sum(risky))
        if risky_total > max_risky_sum and risky_total > 1e-12:
            risky *= (max_risky_sum / risky_total)
            risky_total = float(np.sum(risky))

        cash = max(min_cash_position, 1.0 - risky_total)
        final = np.concatenate([risky, np.array([cash], dtype=np.float64)], axis=0)

        final_sum = float(np.sum(final))
        if final_sum <= 1e-12:
            final = np.ones(self.num_assets + 1, dtype=np.float64) / (self.num_assets + 1)
        else:
            final /= final_sum

        return final.astype(np.float32)

    def _softmax_normalization(self, actions: np.ndarray) -> np.ndarray:
        """
        Normalize actions using softmax (TAPE pattern).
        
        Reference Lines ~277-280:
        def softmax_normalization(self, actions):
            numerator = np.exp(actions)
            denominator = np.sum(np.exp(actions))
            softmax_output = numerator / denominator
            return softmax_output
        
        This is mathematically elegant and numerically stable:
        - Always sums to 1.0 (no need for manual normalization)
        - Handles negative actions gracefully
        - No division by zero issues
        - Differentiable (important for policy gradient)
        
        Args:
            actions: Raw action values from policy network
            
        Returns:
            Normalized weights that sum to 1.0
        """
        # Numerical stability: subtract max to prevent overflow
        actions_shifted = actions - np.max(actions)
        
        numerator = np.exp(actions_shifted)
        denominator = np.sum(numerator)
        
        if denominator > 0:
            softmax_output = numerator / denominator
        else:
            # Fallback to equal weights
            softmax_output = np.ones_like(actions) / len(actions)
        
        return softmax_output.astype(np.float32)
    
    def _dirichlet_normalization(self, actions: np.ndarray) -> np.ndarray:
        """
        Normalize actions using Dirichlet sampling (stochastic exploration).
        
        The Dirichlet distribution is a continuous multivariate probability distribution
        parameterized by a vector of positive reals (concentration parameters α).
        It naturally produces probability distributions that sum to 1.
        
        Process:
        1. Transform actions to positive concentration parameters α_i
        2. Sample from Dirichlet(α): w ~ Dir(α) where Σw_i = 1
        
        Benefits:
        - Adds stochastic exploration to portfolio weights
        - Samples are guaranteed to sum to 1.0
        - Higher α_i → more probability mass on asset i
        - Natural for portfolio allocation (think Bayesian prior)
        
        Trade-offs vs Softmax:
        - Stochastic (non-deterministic) vs deterministic
        - More exploration vs more exploitation
        - Harder to debug vs reproducible
        
        Args:
            actions: Raw action values from policy network
            
        Returns:
            Sampled weights from Dirichlet distribution that sum to 1.0
        """
        # Transform actions to positive concentration parameters
        # Method 1: Exponential (similar to softmax preparation)
        # α_i = exp(action_i) * scale
        
        # 🔥 CRITICAL: Clip actions to prevent overflow in exp()
        actions_clipped = np.clip(actions, -20.0, 20.0)  # Prevent exp(x) overflow
        actions_shifted = actions_clipped - np.max(actions_clipped)  # Numerical stability
        alpha = np.exp(actions_shifted) * self.dirichlet_alpha_scale
        
        # Ensure alpha is in valid range [0.1, 100.0]
        # Too small → degenerate distribution, Too large → numerical instability
        alpha = np.clip(alpha, 0.1, 100.0)
        
        # 🔥 CRITICAL: Validate alpha before sampling
        if np.any(np.isnan(alpha)) or np.any(np.isinf(alpha)):
            logger.error(f"⚠️  Invalid alpha for Dirichlet! Actions: {actions}")
            logger.error(f"   Alpha: {alpha}")
            weights = np.ones_like(actions) / len(actions)
            return weights.astype(np.float32)
        
        # Sample from Dirichlet distribution
        try:
            # Use seeded random if available (from gymnasium's reset)
            if hasattr(self, 'np_random'):
                weights = self.np_random.dirichlet(alpha).astype(np.float32)
            else:
                weights = np.random.dirichlet(alpha).astype(np.float32)
            
            # 🔥 CRITICAL: Validate sampled weights
            if np.any(np.isnan(weights)) or np.any(np.isinf(weights)) or not np.isclose(np.sum(weights), 1.0):
                logger.error(f"⚠️  Invalid weights from Dirichlet! Alpha: {alpha}, Weights: {weights}")
                weights = np.ones_like(actions) / len(actions)
                weights = weights.astype(np.float32)
                
        except Exception as e:
            # Fallback to equal weights if sampling fails
            logger.warning(f"Dirichlet sampling failed: {e}. Alpha: {alpha}. Using equal weights.")
            weights = np.ones_like(actions) / len(actions)
            weights = weights.astype(np.float32)
        
        return weights
    
    def _get_observation(self) -> np.ndarray:
        """
        Get the current observation (state) for the agent.
        
        Returns:
            Feature vector for current day: shape (num_features,)
        """
        if self.day >= len(self.feature_matrix):
            # Return last available observation
            observation = self.feature_matrix[-1].copy()
        else:
            observation = self.feature_matrix[self.day].copy()
        
        # Ensure float32 and handle any remaining NaN
        observation = np.array(observation, dtype=np.float32)
        
        # Check BEFORE cleaning
        if np.isnan(observation).any() or np.isinf(observation).any():
            logger.error(f"❌ Observation at day {self.day} has NaN/inf BEFORE nan_to_num!")
            logger.error(f"   NaN count: {np.isnan(observation).sum()}")
            logger.error(f"   Inf count: {np.isinf(observation).sum()}")
        
        observation = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Check AFTER cleaning
        if np.isnan(observation).any() or np.isinf(observation).any():
            logger.error(f"❌ Observation at day {self.day} STILL has NaN/inf AFTER nan_to_num!")
        
        return observation
    
    def render(self, mode='human'):
        """Render the environment state."""
        if mode == 'human':
            print(f"Day: {self.day}/{self.total_days}")
            print(f"Portfolio Value: ${self.portfolio_value:,.2f}")
            print(f"Return: {(self.portfolio_value / self.initial_balance - 1) * 100:.2f}%")
            print(f"Weights: {self.current_weights}")
    
    def close(self):
        """Clean up environment resources."""
        pass
    
    def save_portfolio_history(self) -> pd.DataFrame:
        """
        Save portfolio history to DataFrame (TAPE pattern).
        
        Returns:
            DataFrame with date, portfolio_value, daily_return columns
        """
        df = pd.DataFrame({
            'date': self.date_history,
            'portfolio_value': self.portfolio_history,
            'daily_return': self.return_history
        })
        return df
    
    def save_weights_history(self) -> pd.DataFrame:
        """
        Save weights history to DataFrame (TAPE pattern).
        
        Returns:
            DataFrame with date and weight columns for each asset
        """
        weights_array = np.array(self.weights_history)
        
        df = pd.DataFrame(weights_array)
        df.columns = [f'asset_{i}' for i in range(self.num_assets)] + ['cash']
        df['date'] = self.date_history
        
        return df


# ═══════════════════════════════════════════════════════════════════════════
# ALIAS FOR BACKWARD COMPATIBILITY
# ═══════════════════════════════════════════════════════════════════════════

# Create an alias so existing code can use PortfolioEnvTF name
PortfolioEnvTF = PortfolioEnvTAPE

logger.info("✅ TAPE Portfolio Environment loaded successfully")
logger.info("   Key changes:")
logger.info("   1. Reward = Portfolio Value (project baseline pattern)")
logger.info("   2. Termination = Data exhausted only (no balance thresholds)")
logger.info("   3. Action normalization = Softmax (numerically stable)")
logger.info("   4. Portfolio math = Simple linear (no log-space)")
logger.info("   5. No training wheels, no milestone rewards")
