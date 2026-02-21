# src/config.py
import numpy as np # type: ignore
import os
import copy

# --- GENERAL PROJECT SETTINGS ---
PROJECT_NAME = "AdaptivePortfolioRL_Prototype"
RANDOM_SEED = 42
TF_DEVICE = "/gpu:0" if len(os.environ.get("CUDA_VISIBLE_DEVICES", "")) > 0 else "/cpu:0"

# --- DATA PATHS ---
BASE_PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
BASE_DATA_PATH = os.path.join(BASE_PROJECT_PATH, 'data')

PATH_DAILY_OHLCV = os.path.join(BASE_DATA_PATH, "daily_ohlcv_assets.csv")  # Cache created on first successful download
PATH_PROCESSED_MACRO_DAILY_ALIGNED = os.path.join(BASE_DATA_PATH, "processed_daily_macro_features.csv")

# --- API KEYS ---
#ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "GETO31F8UHB3OWB3")
ALPHA_VANTAGE_API_KEY = "GETO31F8UHB3OWB3"

# --- DATA DATE RANGES ---
DATA_FETCH_START_DATE = "2003-09-02"  # Paper-aligned start
DATA_FETCH_END_DATE = "2025-09-01"  # Extended test horizon
ANALYSIS_START_DATE = "2003-09-02"
ANALYSIS_END_DATE = "2025-09-01"

# --- TRAIN/TEST SPLIT ---
# Two supported split presets:
# - Benchmark (paper-aligned train cutoff)
# - COVID stress test (test starts 2020-01-01)
TRAIN_TEST_SPLIT_DATE_BENCHMARK = "2021-09-01"     # Train: <= 2021-09-01
TRAIN_TEST_SPLIT_DATE_COVID_STRESS = "2019-12-31"  # Train: <= 2019-12-31
TRAIN_TEST_SPLIT_DATE = TRAIN_TEST_SPLIT_DATE_BENCHMARK

# --- ASSET CONFIGURATION ---
ASSET_TICKERS = ["MSFT", "GOOGL", "JPM", "JNJ", "XOM", "PG", "NEE", "LIN", "CAT", "UNH"] 
NUM_ASSETS = len(ASSET_TICKERS)
CASH_ASSET_NAME = "CASH"

# --- PORTFOLIO INITIALIZATION ---
# Options: 'equal', 'volume_weighted', 'custom'
INITIAL_WEIGHTS_METHOD = "volume_weighted"  # Use volume-based weighting
MARKET_CAP_CASH_ALLOCATION = 0.02  # 2% cash allocation for volume weighting
EQUAL_WEIGHT_CASH_ALLOCATION = 1.0 / (NUM_ASSETS + 1)  # Equal split across assets plus cash

# --- FEATURE ENGINEERING CONFIGURATION (TEMPLATES) ---
TECHNICAL_INDICATORS_CONFIG = [
    {"name": "EMA", "params": {"length": 12}, "output_cols": ["EMA_12"]},
    {"name": "EMA", "params": {"length": 26}, "output_cols": ["EMA_26"]},
    {"name": "BBANDS", "params": {"length": 20, "std": 2}, "output_cols": ["BBL_20_2.0", "BBM_20_2.0", "BBU_20_2.0"]},
    {"name": "MACD", "params": {"fast": 12, "slow": 26, "signal": 9}, "output_cols": ["MACD_12_26_9", "MACDh_12_26_9", "MACDs_12_26_9"]},
    {"name": "RSI", "params": {"length": 14}, "output_cols": ["RSI_14"]},
    # Adding the other 9 TIs from the full list for a comprehensive baseline
    {"name": "STOCH", "params": {"k": 14, "d": 3, "smooth_k": 3}, "output_cols": ["STOCHk_14_3_3", "STOCHd_14_3_3"]},
    {"name": "WILLR", "params": {"length": 14}, "output_cols": ["WILLR_14"]},
    {"name": "SMA_price", "params": {"length": 50}, "output_cols": ["SMA_50"]}, # Renamed to avoid conflict
    {"name": "ADX", "params": {"length": 14}, "output_cols": ["ADX_14", "DMP_14", "DMN_14"]},
    {"name": "ATR", "params": {"length": 14, "mamode":"ema"}, "output_cols": ["ATRr_14"]},
    {"name": "NATR", "params": {"length": 14}, "output_cols": ["NATR_14"]},
    {"name": "SMA_volume", "params": {"length": 20, "close_col_name": "Volume"}, "output_cols": ["VOL_SMA_20"]},
    {"name": "OBV", "params": {}, "output_cols": ["OBV"]},
    {"name": "MFI", "params": {"length": 14}, "output_cols": ["MFI_14"]}
]

TEMPORAL_FORECAST_PARAMS = {
    "sequence_length": 20, "epochs": 2, "batch_size": 32, "learning_rate": 0.01,  # Reduced epochs for faster testing "dropout_rate": 0.2
}

DYNAMIC_COVARIANCE_PARAMS = {
    "covariance_window_length": 60, "feature_extraction_methods": ["eigenvalues"],
    "num_eigenvalues": min(3, NUM_ASSETS)
}

ACTUARIAL_PARAMS = {
    "enabled": True,
    "severity_buckets": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    "development_horizons": [10, 20, 30, 60, 90, 120],
    "min_events_for_credibility": 5
}

FUNDAMENTAL_FEATURES_CONFIG = {
    "enabled": True,
    # CSV expected to contain columns: Date, Ticker, FCFE, Revenue, NCFO
    "data_path": os.path.join(BASE_DATA_PATH, "quarterly_fundamentals.csv"),
    "lag_quarters": 8,
    "staleness_days_normalizer": 90.0
}

# Phase-1/2 feature pruning policy:
# - remove strongly redundant trend/oscillator channels
# - remove binary beta flags (continuous beta/rank already available)
# - reduce redundant policy-rate level channels (prefer diff/zscore dynamics)
PHASE12_REDUNDANT_FEATURES_TO_DISABLE = [
    "EMA_12",
    "EMA_26",
    "BBM_20_2.0",
    "SMA_50",
    "STOCHk_14_3_3",
    "WILLR_14",
    "HighBeta_Flag",
    "LowBeta_Flag",
    "EFFR_level",
    "FEDFUNDS_level",
    # Legacy macro outputs retained in some metadata snapshots; prune explicitly.
    "ISM_MAN_PMI_level",
    "ISM_MAN_PMI_diff",
]

FEATURES_TO_DISABLE = [
    "BAMLC0A0CMEY_level",
    "BAMLC0A0CMEY_diff",
    "BAMLC0A0CMEY_zscore",
    "BAMLH0A0HYM2_level",
    "BAMLH0A0HYM2_diff",
    "BAMLH0A0HYM2_zscore",
    "DAAA_level",
    "DAAA_zscore",
    "VIX_level",
    "VIX_zscore",
    "MOVE_level",
    "MOVE_zscore",
    "UNRATE_level",
    "UNRATE_diff",
    "UNRATE_zscore",
    "PAYEMS_level",
    "PAYEMS_diff",
    "PAYEMS_yoy",
    "ICSA_level",
    "ICSA_diff",
    "INDPRO_level",
    "INDPRO_diff",
    "INDPRO_yoy",
    "CPI_level",
    "CPI_mom",
    "CPI_yoy",
    "PPI_level",
    "PPI_mom",
    "PPI_yoy",
    "FedBalanceSheet_level",
    "FedBalanceSheet_diff",
    "ON_RRP_level",
    "ON_RRP_diff",
] + PHASE12_REDUNDANT_FEATURES_TO_DISABLE

FEATURE_SELECTION_CONFIG = {
    "disable_features": True,
    "disabled_features": FEATURES_TO_DISABLE,
}

ALPHA_FEATURES_CONFIG = {
    "enabled": True,
    "cross_sectional_column": "LogReturn_1d",
    "residual_momentum_window": 21,
    "volume_percentile_window": 63,
    "reversal_window": 5,
    "vol_of_vol_window": 63,
    "beta_window": 63,
    "obv_window": 21,
    "yield_curve": {
        "long_col": "DGS10_level",
        "short_col": "DGS2_level",
    },
    "retain_market_return": False,
    "epsilon": 1e-9,
}

# Cross-Sectional Features Configuration (Asset Differentiation)
CROSS_SECTIONAL_FEATURES_CONFIG = {
    "enabled": True,
    "momentum_windows": [21, 63, 252],  # Short/medium/long-term momentum rankings
    "zscore_features": [  # Features to standardize cross-sectionally
        "LogReturn_1d",
        "RollingVolatility_21d",
        "RSI_14"
    ],
    # Keep policy surface smooth; avoid extra binary channels by default.
    "include_beta_flags": False,
    "high_beta_threshold": 1.2,
    "low_beta_threshold": 0.8,
}

FRED_API_KEY = "da9d24dd8de4f924dcbc8416e539b4ef" # User's actual FRED API Key
FRED_SERIES_CONFIG = [
    {"code": "EFFR", "name": "EFFR", "freq": "d", "calc": ["diff", "zscore"]},
    {"code": "SOFR", "name": "SOFR", "freq": "d", "calc": ["level", "diff"]},
    {"code": "FEDFUNDS", "name": "FEDFUNDS", "freq": "m", "calc": ["diff", "zscore"]},
    {"code": "DGS10", "name": "DGS10", "freq": "d", "calc": ["level", "diff", "slope"]},
    {"code": "DGS2", "name": "DGS2", "freq": "d", "calc": ["level", "diff"]},
    {"code": "T10Y2Y", "name": "T10Y2Y", "freq": "d", "calc": ["level"]},
    {"code": "DFII10", "name": "TIPS10Y", "freq": "d", "calc": ["level", "diff"]},
    {"code": "T10YIE", "name": "BreakevenInf10Y", "freq": "d", "calc": ["level", "diff"]},
    {"code": "T5YIFR", "name": "BreakevenInf5Y", "freq": "d", "calc": ["level", "diff"]},
    {"code": "WALCL", "name": "FedBalanceSheet", "freq": "w", "calc": ["level", "diff"]},
    {"code": "RRPONTSYD", "name": "ON_RRP", "freq": "d", "calc": ["level", "diff"]},
    {"code": "CPIAUCSL", "name": "CPI", "freq": "m", "calc": ["yoy", "mom"]},
    {"code": "PPIACO", "name": "PPI", "freq": "m", "calc": ["yoy", "mom"]},
    {"code": "UNRATE", "name": "UNRATE", "freq": "m", "calc": ["level", "diff", "zscore"]},
    {"code": "PAYEMS", "name": "PAYEMS", "freq": "m", "calc": ["level", "diff", "yoy"]},
    {"code": "INDPRO", "name": "INDPRO", "freq": "m", "calc": ["level", "diff", "yoy"]},
    {"code": "BAMLC0A4CBBBEY", "name": "IG_Credit", "freq": "d", "calc": ["level", "diff", "zscore"]},
    {"code": "BAMLH0A0HYM2", "name": "HY_Credit", "freq": "d", "calc": ["level", "diff", "zscore"]},
    {"code": "VIXCLS", "name": "VIX", "freq": "d", "calc": ["level", "zscore"]},
]

# Explicit global-context routing for structured observations in Phase 1/2.
PHASE12_GLOBAL_FEATURE_COLUMNS = [
    "YieldCurve_Spread",
    "YieldCurve_Inverted_Flag",
    "Regime_Breadth_Positive",
]

PHASE12_GLOBAL_FEATURE_PREFIXES = [
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
    "VIX_",
]

MACRO_DATA_CONFIG = {
    "fred_api_key": FRED_API_KEY,
    "fred_series_config": FRED_SERIES_CONFIG,
    "business_days_only": True,
    "ffill_limit": None,
}

# --- PERFORMANCE CALCULATION PARAMETERS ---
PERFORMANCE_METRICS_CONFIG = {
    "rolling_window_episodes": 20,  # Episodes for rolling performance calculation
    "risk_free_rate": 0.02,  # Annual risk-free rate for Sharpe calculation
    "trading_days_per_year": 252,
    "metrics_to_track": [
        "total_return", "sharpe_ratio", "sortino_ratio", "max_drawdown", 
        "turnover", "skewness", "volatility", "win_rate"
    ]
}

# --- MARKET REGIME DETECTION PARAMETERS ---
MARKET_REGIME_CONFIG = {
    "volatility_regimes": {
        "low": {"vix_threshold": 15.0, "rolling_vol_threshold": 0.15},
        "medium": {"vix_threshold": 25.0, "rolling_vol_threshold": 0.25},
        "high": {"vix_threshold": 35.0, "rolling_vol_threshold": 0.35}
    },
    "trend_detection": {
        "bullish_threshold": 0.05,   # 5% above long-term MA
        "bearish_threshold": -0.05,  # 5% below long-term MA
        "sideways_threshold": 0.02   # Within 2% is sideways
    },
    "recession_indicators": {
        "yield_curve_inversion_threshold": -0.1,  # 10bp inversion
        "unemployment_change_threshold": 0.5      # 0.5% increase in unemployment
    }
}

# --- TAPE REWARD & UTILITY PROFILE DEFINITIONS ---
METRICS_ORDER = ['sharpe', 'sortino', 'mdd', 'turnover', 'skew']

PROFILE_BALANCED_GROWTH = {
    "name": "BalancedGrowth",
    # Asymmetric Sigmoid: mu is the 50% utility midpoint, k controls steepness
    "mu": np.array([1.0, 1.3, -0.15, 0.60, 0.0], dtype=np.float32),
    # k_minus: steepness BELOW target (penalty for increasing, reward for decreasing)
    "k_minus": np.array([4.0, 3.0, 5.0, 1.0, 2.0], dtype=np.float32),
    # k_plus:  steepness ABOVE target (reward for increasing, penalty for decreasing)
    "k_plus":  np.array([1.0, 1.0, 1.0, 4.0, 1.0], dtype=np.float32),
    "weights":   np.array([0.30, 0.25, 0.25, 0.15, 0.05], dtype=np.float32),
    "metrics_order": METRICS_ORDER,
    # MDD stored as negative: higher = less drawdown = better → 'increasing'
    "directions": ['increasing', 'increasing', 'increasing', 'decreasing', 'increasing'],
    "a_bounds": np.array([-2.0, -1.0, -0.30, 0.0, -1.0]),
    "b_bounds": np.array([3.0, 4.0, 0.0, 0.80, 1.0]),
}
PROFILE_AGGRESSIVE_ALPHA_SEEKER = {
    "name": "AggressiveAlphaSeeker",
    "mu": np.array([1.5, 2.5, -0.25, 1.0, 0.15], dtype=np.float32),
    "k_minus": np.array([5.0, 4.0, 3.0, 0.5, 3.0], dtype=np.float32),
    "k_plus":  np.array([0.5, 0.5, 0.5, 2.0, 0.5], dtype=np.float32),
    "weights":   np.array([0.40, 0.30, 0.10, 0.05, 0.15], dtype=np.float32),
    "metrics_order": METRICS_ORDER,
    "directions": ['increasing', 'increasing', 'increasing', 'decreasing', 'increasing'],
    "a_bounds": np.array([-1.0, 0.0, -0.50, 0.0, -1.0]),
    "b_bounds": np.array([4.0, 5.0, 0.0, 2.5, 1.5]),
}
PROFILE_CAPITAL_PRESERVATION = {
    "name": "CapitalPreservation",
    "mu": np.array([0.5, 1.0, -0.08, 0.02, -0.05], dtype=np.float32),
    "k_minus": np.array([3.0, 2.5, 8.0, 0.5, 2.0], dtype=np.float32),
    "k_plus":  np.array([2.0, 2.0, 2.0, 3.0, 1.5], dtype=np.float32),
    "weights":   np.array([0.05, 0.05, 0.60, 0.20, 0.10], dtype=np.float32),
    "metrics_order": METRICS_ORDER,
    "directions": ['increasing', 'increasing', 'increasing', 'decreasing', 'increasing'],
    "a_bounds": np.array([-1.0, -0.5, -0.20, 0.0, -1.0]),
    "b_bounds": np.array([2.0, 3.0, 0.0, 1.0, 1.0]),
}
ALL_PROFILES_LIST = [PROFILE_BALANCED_GROWTH, PROFILE_AGGRESSIVE_ALPHA_SEEKER, PROFILE_CAPITAL_PRESERVATION]

# --- PHASE CONFIGURATIONS ---

# Phase 1: Baseline Enhanced Vanilla PPO
PHASE1_CONFIG = {
    "phase_name": "Phase1_Baseline_PPO",
    #================================================
    "ASSET_TICKERS": ASSET_TICKERS,
    "NUM_ASSETS": NUM_ASSETS,
    "BASE_DATA_PATH": BASE_DATA_PATH,
    "PATH_DAILY_OHLCV": PATH_DAILY_OHLCV,
    "DATA_FETCH_START_DATE": DATA_FETCH_START_DATE,
    "DATA_FETCH_END_DATE": DATA_FETCH_END_DATE,
    "ANALYSIS_START_DATE": ANALYSIS_START_DATE,
    "ANALYSIS_END_DATE": ANALYSIS_END_DATE,
    "feature_params": {
        "technical_indicators": TECHNICAL_INDICATORS_CONFIG,
        "include_log_returns": True,
        "log_return_col_name": "Daily_LogReturn",
        # Dynamic covariance enabled for portfolio correlation insights
        "dynamic_covariance": DYNAMIC_COVARIANCE_PARAMS,
        "actuarial_params": ACTUARIAL_PARAMS,
        # Advanced features still disabled for Phase 1
        "temporal_forecast": None,
        "macro_data": copy.deepcopy(MACRO_DATA_CONFIG),
        "fundamental_features": copy.deepcopy(FUNDAMENTAL_FEATURES_CONFIG),
        "regime_features": {
            "enabled": True,
            "vol_windows": {"short": 21, "long": 126},
            "trend_windows": {"short": 50, "long": 200},
            "momentum_windows": {"short": 63, "long": 252},
            "correlation_window": 60,
            "breadth_window": 21,
        },
        "feature_selection": copy.deepcopy(FEATURE_SELECTION_CONFIG),
        "alpha_features": copy.deepcopy(ALPHA_FEATURES_CONFIG),
        "cross_sectional_features": copy.deepcopy(CROSS_SECTIONAL_FEATURES_CONFIG),
    },
    #================================================
    "environment_params": {
        "initial_balance": 100000.0,
        "transaction_cost_pct": 0.001,
        "structured_observation": True,
        "global_feature_columns": copy.deepcopy(PHASE12_GLOBAL_FEATURE_COLUMNS),
        "global_feature_prefixes": copy.deepcopy(PHASE12_GLOBAL_FEATURE_PREFIXES),
        "reward_type": "advanced_tape",  # Three-component TAPE reward
        "max_steps_per_episode": None,  # Episode horizon managed dynamically during training
        "done_on_balance_threshold_pct": 0.5,  # PHASE 1: Increased from 0.2 to 0.5 for exploration
        "random_start": True,
        "initial_allocation_mode": "equal_assets_with_min_cash",
        "initial_cash_position": 0.05,
        "tape_terminal_scalar": 10.0,
        "tape_terminal_clip": 10.0,
        "tape_terminal_bonus_mode": "signed",
        "tape_terminal_baseline": 0.20,
        "tape_terminal_neutral_band_enabled": True,
        "tape_terminal_neutral_band_halfwidth": 0.02,
        "tape_terminal_gate_a_enabled": True,
        "tape_terminal_gate_a_sharpe_threshold": 0.0,
        "tape_terminal_gate_a_max_drawdown": 0.25,
        "target_turnover": 0.60,  # Relax early ceiling; tighten via curriculum as policy stabilizes
        "turnover_penalty_scalar": 2.0,
        "turnover_target_band": 0.20,
        "dsr_scalar": 2.0,  # Further reduce PBRS noise while policy is unstable
        "concentration_penalty_scalar": 2.0,
        "concentration_target_hhi": 0.14,
        "top_weight_penalty_scalar": 1.5,
        "target_top_weight": 0.22,
        "action_realization_penalty_scalar": 0.5,
        "penalty_budget_ratio": 1.25,
        "intra_step_tape_delta_enabled": True,
        "intra_step_tape_delta_window": 60,
        "intra_step_tape_delta_min_history": 20,
        "intra_step_tape_delta_beta": 0.01,
        "intra_step_tape_delta_clip": 0.20,
        "dd_regime_scaling": {
            "enabled": True,
            "vol_window": 21,
            "low_vol_threshold": 0.12,
            "high_vol_threshold": 0.25,
            "low_mult": 0.90,
            "mid_mult": 1.00,
            "high_mult": 1.35,
        },
        "drawdown_constraint": {
            "enabled": True,
            "target": 0.18,
            "penalty_coef": 1.5,
            "dual_learning_rate": 0.10,
            "lambda_init": 0.50,
            "lambda_floor": 0.0,
            "lambda_max": 5.0,
            "tolerance": -0.015,
            # Apply penalty as soon as drawdown crosses trigger boundary (target + tolerance).
            "penalty_reference": "trigger_boundary",
            # Avoid cooling lambda too aggressively below trigger.
            "cooling_rate": 0.35,
            "lambda_carry_decay": 0.7,
        }
    },
    #================================================
    "agent_params": {
        # Architecture Selection: 'TCN', 'TCN_ATTENTION', 'TCN_FUSION'
        "actor_critic_type": "TCN_FUSION",
        "use_attention": False,
        "use_fusion": True,
        
        # Network dimensions for TCN architectures
        "actor_hidden_dims": [256, 128],
        "critic_hidden_dims": [256, 128],
        
        # Sequence parameters for sequential models (TCN, TCN, etc.)
        "sequence_length": 60,  # Requested architecture update
        
        # STATE-OF-THE-ART FIX #3: Right-sized TCN dimensions (2x faster, minimal quality loss)
        # TCN-specific parameters for faster training
        
        # TCN-specific parameters
        #"tcn_filters": [64, 128, 256],
        #"tcn_kernel_size": 3,
        #"tcn_dilations": [1, 2, 4, 8, 16],
        #"tcn_dropout": 0.2,

        # IMPROVED TCN ARCHITECTURE (deeper with better receptive field)
        "tcn_filters": [64, 128, 128],  # Upgraded TCN capacity for richer temporal representation
        "tcn_kernel_size": 5,                # Requested architecture update
        "tcn_dilations": [2, 4, 8],       # Requested architecture update
        "tcn_dropout": 0.2,
        
        # Attention-specific parameters
        "attention_heads": 4,
        "attention_dim": 64,
        "attention_dropout": 0.1,

        # Fusion-specific parameters (for TCN_FUSION or TCN with use_fusion=True)
        "fusion_embed_dim": 128,
        "fusion_attention_heads": 4,
        "fusion_dropout": 0.1,

        # Dirichlet alpha activation (controls action concentration)
        "dirichlet_alpha_activation": "elu",  # Confirmed via literature (strictly positive)
        "dirichlet_exp_clip": (-5.0, 3.0),

        # Dirichlet exploration (epsilon annealing)
        "dirichlet_epsilon": {
            "max": 0.5,  # Early training: encourage exploration
            "min": 0.1,  # Late training / evaluation: sharper allocations
        },

        # Deterministic evaluation mode
        # Options: 'mean', 'mode', 'mean_plus_noise'
        "evaluation_mode": "mode",  # ✅ RECOMMENDED: Shows true learned policy
        
        # PPO Algorithm parameters
        # Stabilized PPO regime for better out-of-sample Sharpe retention.
        "ppo_params": {
            "gamma": 0.99, "gae_lambda": 0.9, "policy_clip": 0.10,
            "entropy_coef": 0.01, "vf_coef": 0.5, "num_ppo_epochs": 4,
            "batch_size_ppo": 252, "actor_lr": 0.00002, "critic_lr": 0.0003,
            "max_grad_norm": 0.5, "value_clip": 0.2, "target_kl": 0.015,
            "kl_stop_multiplier": 1.2, "minibatches_before_kl_stop": 1,
            # Optional risk-aware actor auxiliaries (default disabled).
            "use_risk_aux_loss": False,
            # Per-asset feature index used as one-step return proxy in structured state tensor.
            "risk_aux_return_feature_index": 0,
            "risk_aux_cash_return": 0.0,
            "risk_aux_sharpe_coef": 0.0,
            "risk_aux_mvo_coef": 0.0,
            "risk_aux_mvo_cov_ridge": 1e-3,
            "risk_aux_mvo_long_only": True,
            "risk_aux_mvo_risky_budget": 0.95,
        },
    },
    #================================================
    "training_params": {
        "max_total_timesteps": 150000,  # Requested architecture update
        "timesteps_per_ppo_update": 504,  # Frequent updates (matched to episode length) — archive used ~252
        "log_interval_episodes": 10,
        "update_log_interval": 20,
        "alpha_diversity_log_interval": 10,
        "alpha_diversity_warning_after_updates": 500,
        "alpha_diversity_warning_std_threshold": 0.30,
        "save_freq_episodes": 50,
        "max_episode_length": None,  # Episode length controlled by curriculum
        
        # STATE-OF-THE-ART FIX #2: Stronger Entropy Incentive
        "entropy_coefficient": 0.10,  # Diversification bonus weight (10x stronger for meaningful impact)
        
        # STATE-OF-THE-ART FIX #3: Position Size Constraints
        "max_single_position": 25.00,  # Maximum allocation to any single asset (25%)
        
        # Alternative: Curriculum schedule for max single position
        # Start strict, relax as agent learns 
        #"max_single_position_curriculum": {
        #    0: 25.0,        # First 30k steps: Learn diversification
        #    30_000: 30.0,   # 30k-60k: Allow more concentration
        #    60_000: 35.0,   # 60k-100k: Further relaxation
        #    100_000: 40.0,  # Final: Full flexibility
        #},
        "min_cash_position": 0.05,    # Minimum cash buffer (5%)
        
        # STATE-OF-THE-ART FIX #4: Curriculum Learning
        "use_curriculum_learning": True,
        "curriculum_phases": [
            {"name": "low_vol", "timesteps_fraction": 0.30},    # First 30% on low volatility
            {"name": "medium_vol", "timesteps_fraction": 0.40},  # Next 40% on medium volatility
            {"name": "all", "timesteps_fraction": 0.30}          # Final 30% on all data
        ],

        # Disable episode-length curriculum: use full dataset horizon throughout.
        "use_episode_length_curriculum": False,
        "episode_length_curriculum_schedule": [
            {"threshold": 0, "limit": 1500},
            {"threshold": 30_000, "limit": 2000},
            {"threshold": 60_000, "limit": 2500},
            {"threshold": 90_000, "limit": None},
        ],
        
        # PHASE 1: Progressive Threshold Curriculum
        "use_progressive_threshold": True,
        # Progressive threshold schedule for Phase 1 (20K episodes, gentler curve)
        "progressive_threshold_schedule": {
            0: 0.90,       # Episodes 1-1000: Very forgiving start
            1000: 0.85,    # Episodes 1001-2000: Gradual decrease
            2000: 0.80,    # Episodes 2001-3000
            3000: 0.75,    # Episodes 3001-4000
            4000: 0.70,    # Episodes 4001-5000
            5000: 0.65,    # Episodes 5001-6000
            6000: 0.60,    # Episodes 6001-7000
            7000: 0.55,    # Episodes 7001-8000
            8000: 0.50     # Episodes 8001-20000: Target threshold
        },

        # Rare checkpoint capture (high-performance snapshots)
        "rare_checkpoint_params": {
            "enable": True,
            "min_sharpe": 1.5,
            "min_sortino": 2.0,
            "max_mdd": 0.15,          # 15% drawdown
            "max_turnover": 0.80,     # 80% daily turnover
            "top_n": 5
        },
        # TAPE checkpoint trigger uses clipped terminal-bonus scale.
        # With signed mode + baseline 0.20 + scalar 10, 4.0 ~= tape_score >= 0.52.
        "tape_checkpoint_threshold": 4.0,
        # Periodic safety snapshots independent of TAPE threshold.
        "periodic_checkpoint_every_steps": 10_000,
        # Episode-level high-watermark snapshots (save on new best Sharpe >= threshold).
        "high_watermark_checkpoint_enabled": True,
        "high_watermark_sharpe_threshold": 0.5,
        # Optional per-step snapshots (very noisy; disabled by default).
        "step_sharpe_checkpoint_enabled": False,
        "step_sharpe_checkpoint_threshold": 0.5,

        # Actor LR schedule (canonical across project).
        # Starts conservative, then decays further for stability.
        "actor_lr_schedule": [
            {"threshold": 0, "lr": 0.00002},
            {"threshold": 40_000, "lr": 0.000015},
            {"threshold": 70_000, "lr": 0.00001},
        ],

        # Turnover curriculum matching 2.0 → 1.75 → 1.50 → 1.25 request
        "turnover_penalty_curriculum": {
            0: 0.75,
            30_000: 1.25,
            60_000: 1.50,
            90_000: 1.75,
            120_000: 2.00,
        },
    },
}

# Phase 2: Advanced Single-Horizon Model (3-Day Prototype)
PHASE2_CONFIG = {
    "phase_name": "Phase2_Advanced_Prototype",
    #================================================
    "ASSET_TICKERS": ASSET_TICKERS,
    "NUM_ASSETS": NUM_ASSETS,
    "BASE_DATA_PATH": BASE_DATA_PATH,
    "PATH_DAILY_OHLCV": PATH_DAILY_OHLCV,
    "DATA_FETCH_START_DATE": DATA_FETCH_START_DATE,
    "DATA_FETCH_END_DATE": DATA_FETCH_END_DATE,
    "ANALYSIS_START_DATE": ANALYSIS_START_DATE,
    "ANALYSIS_END_DATE": ANALYSIS_END_DATE,
    "feature_params": {
        "technical_indicators": TECHNICAL_INDICATORS_CONFIG,
        "include_log_returns": True,
        "log_return_col_name": "Daily_LogReturn",
        "temporal_forecast": None,  # Disabled for testing - TCN rolling forecast is too slow (4000+ models to train)
        "dynamic_covariance": DYNAMIC_COVARIANCE_PARAMS,
        "macro_data": copy.deepcopy(MACRO_DATA_CONFIG),
        "fundamental_features": copy.deepcopy(FUNDAMENTAL_FEATURES_CONFIG),
        "regime_features": {
            "enabled": True,
            "vol_windows": {"short": 21, "long": 126},
            "trend_windows": {"short": 50, "long": 200},
            "momentum_windows": {"short": 63, "long": 252},
            "correlation_window": 60,
            "breadth_window": 21,
        },
        "feature_selection": copy.deepcopy(FEATURE_SELECTION_CONFIG),
        "alpha_features": copy.deepcopy(ALPHA_FEATURES_CONFIG),
        "cross_sectional_features": copy.deepcopy(CROSS_SECTIONAL_FEATURES_CONFIG),
    },
    #================================================
    "environment_params": {
        "initial_balance": 100000.0, "transaction_cost_pct": 0.001,
        "structured_observation": True,
        "global_feature_columns": copy.deepcopy(PHASE12_GLOBAL_FEATURE_COLUMNS),
        "global_feature_prefixes": copy.deepcopy(PHASE12_GLOBAL_FEATURE_PREFIXES),
        "reward_type": "advanced_tape", # Use TAPE reward system
        "truncated_gaussian_lambda": 0.2,
        "max_steps_per_episode": 252,
        "done_on_balance_threshold_pct": 0.5,
        "initial_allocation_mode": "equal_assets_with_min_cash",
        "initial_cash_position": 0.05,
        "target_turnover": 0.60,  # Relax early ceiling; tighten via curriculum as policy stabilizes
        "tape_terminal_bonus_mode": "signed",
        "tape_terminal_baseline": 0.20,
        "tape_terminal_neutral_band_enabled": True,
        "tape_terminal_neutral_band_halfwidth": 0.02,
        "tape_terminal_gate_a_enabled": True,
        "tape_terminal_gate_a_sharpe_threshold": 0.0,
        "tape_terminal_gate_a_max_drawdown": 0.25,
        "turnover_penalty_scalar": 2.0,
        "turnover_target_band": 0.20,
        "dsr_scalar": 2.0,  # Further reduce PBRS noise while policy is unstable
        "concentration_penalty_scalar": 2.0,
        "concentration_target_hhi": 0.14,
        "top_weight_penalty_scalar": 1.5,
        "target_top_weight": 0.22,
        "action_realization_penalty_scalar": 0.5,
        "penalty_budget_ratio": 1.25,
        "intra_step_tape_delta_enabled": True,
        "intra_step_tape_delta_window": 60,
        "intra_step_tape_delta_min_history": 20,
        "intra_step_tape_delta_beta": 0.01,
        "intra_step_tape_delta_clip": 0.20,
        "dd_regime_scaling": {
            "enabled": True,
            "vol_window": 21,
            "low_vol_threshold": 0.12,
            "high_vol_threshold": 0.25,
            "low_mult": 0.90,
            "mid_mult": 1.00,
            "high_mult": 1.35,
        },
        "drawdown_constraint": {
            "enabled": True,
            "target": 0.18,
            "penalty_coef": 1.5,
            "dual_learning_rate": 0.10,
            "lambda_init": 0.50,
            "lambda_floor": 0.0,
            "lambda_max": 5.0,
            "tolerance": -0.015,
            "penalty_reference": "trigger_boundary",
            "cooling_rate": 0.35,
            "lambda_carry_decay": 0.7,
        },
        "drawdown_constraint_overrides": {
            "sequential": {
                "penalty_coef": 1.5,
                "dual_learning_rate": 0.10,
                "lambda_floor": 0.0,
                "lambda_max": 5.0,
                "tolerance": -0.015,
                "penalty_reference": "trigger_boundary",
                "cooling_rate": 0.35,
                "lambda_carry_decay": 0.7,
            }
        }
    },
    #================================================
    "agent_params": {
        # Architecture Selection: 'TCN', 'TCN_ATTENTION', 'TCN_FUSION'
        "actor_critic_type": "TCN_FUSION",
        "use_attention": False,
        "use_fusion": True,
        
        # Network dimensions for TCN architectures
        "actor_hidden_dims": [256, 128], # Larger TCN for more features
        "critic_hidden_dims": [256, 128],
        
        # Sequence parameters for sequential models (TCN, TCN, etc.)
        "sequence_length": 60,  # Requested architecture update
        
        # STATE-OF-THE-ART FIX #3: Right-sized TCN dimensions (2x faster, minimal quality loss)
        # TCN-specific parameters for faster training
        
        # TCN-specific parameters
        #"tcn_filters": [64, 128, 256],  # Modest pyramid for deeper receptive field
        #"tcn_kernel_size": 3,
        #"tcn_dilations": [1, 2, 4, 8, 16],  # Extended dilations to cover ~60–90 days
        #"tcn_dropout": 0.2,

        # IMPROVED TCN ARCHITECTURE (deeper with better receptive field)
        "tcn_filters": [64, 128, 128],  # Upgraded TCN capacity for richer temporal representation
        "tcn_kernel_size": 5,                # Requested architecture update
        "tcn_dilations": [2, 4, 8],       # Requested architecture update
        "tcn_dropout": 0.2,
        
        # Attention-specific parameters
        "attention_heads": 4,
        "attention_dim": 64,
        "attention_dropout": 0.1,

        # Fusion-specific parameters (for TCN_FUSION or TCN with use_fusion=True)
        "fusion_embed_dim": 128,
        "fusion_attention_heads": 4,
        "fusion_dropout": 0.1,

        # Dirichlet alpha activation (controls action concentration)
        "dirichlet_alpha_activation": "elu",  # ✅ RECOMMENDED: More stable than softplus
        "dirichlet_exp_clip": (-5.0, 3.0),

        # Dirichlet exploration (epsilon annealing)
        "dirichlet_epsilon": {
            "max": 0.5,
            "min": 0.1,
        },
        
        # Deterministic evaluation mode
        # Options: 'mean', 'mode', 'mean_plus_noise'
        "evaluation_mode": "mode",  # ✅ RECOMMENDED: Shows true learned policy
        
        # PPO Algorithm parameters
        "ppo_params": {
            "gamma": 0.99, "gae_lambda": 0.9, "policy_clip": 0.10,
            "entropy_coef": 0.01, "vf_coef": 0.5, "num_ppo_epochs": 4,
            "batch_size_ppo": 256, "actor_lr": 0.00002, "critic_lr": 0.0003,
            "max_grad_norm": 0.5, "value_clip": 0.2, "target_kl": 0.015,
            "kl_stop_multiplier": 1.2, "minibatches_before_kl_stop": 1,
            # Optional risk-aware actor auxiliaries (default disabled).
            "use_risk_aux_loss": False,
            "risk_aux_return_feature_index": 0,
            "risk_aux_cash_return": 0.0,
            "risk_aux_sharpe_coef": 0.0,
            "risk_aux_mvo_coef": 0.0,
            "risk_aux_mvo_cov_ridge": 1e-3,
            "risk_aux_mvo_long_only": True,
            "risk_aux_mvo_risky_budget": 0.95,
        },
    },
    #================================================
    "training_params": {
        "max_total_timesteps": 100000,
        "timesteps_per_ppo_update": 250,
        "log_interval_episodes": 10,
        "save_freq_episodes": 50,
        "max_episode_length": None,  # Use full dataset horizon

        # STATE-OF-THE-ART FIX #2: Stronger Entropy Incentive
        "entropy_coefficient": 0.10,  # Diversification bonus weight (10x stronger for meaningful impact)
        
        # STATE-OF-THE-ART FIX #3: Position Size Constraints
        "max_single_position": 25.00,  # Maximum allocation to any single asset (25%)
        "min_cash_position": 0.05,    # Minimum cash buffer (5%)

        # STATE-OF-THE-ART FIX #4: Curriculum Learning
        "use_curriculum_learning": True,
        "curriculum_phases": [
            {"name": "low_vol", "timesteps_fraction": 0.30},    # First 30% on low volatility
            {"name": "medium_vol", "timesteps_fraction": 0.40},  # Next 40% on medium volatility
            {"name": "all", "timesteps_fraction": 0.30}          # Final 30% on all data
        ],

        # Disable episode-length curriculum: use full dataset horizon throughout.
        "use_episode_length_curriculum": False,
        "episode_length_curriculum_schedule": [
            {"threshold": 0, "limit": 504},
            {"threshold": 15_000, "limit": 756},
            {"threshold": 30_000, "limit": 1_200},
            {"threshold": 45_000, "limit": 1_500},
            {"threshold": 60_000, "limit": 2_500},
            {"threshold": 75_000, "limit": None},
        ],

        # Turnover penalty schedule to match Phase 1 discipline
        "turnover_penalty_curriculum": {
            0: 0.75,
            30_000: 1.25,
            60_000: 1.50,
            90_000: 1.75,
            120_000: 2.00,
        },

        # Actor LR decay schedule (critic stays constant)
        "actor_lr_schedule": [
            {"threshold": 0, "lr": 0.00002},
            {"threshold": 40_000, "lr": 0.000015},
            {"threshold": 70_000, "lr": 0.00001},
        ],

        # Progressive reward thresholding identical to Phase 1
        "use_progressive_threshold": True,
        "progressive_threshold_schedule": {
            0: 0.90,
            1000: 0.85,
            2000: 0.80,
            3000: 0.75,
            4000: 0.70,
            5000: 0.65,
            6000: 0.60,
            7000: 0.55,
            8000: 0.50,
        },

        # Console logging cadence
        "episode_log_interval": 10,

        # Rare checkpoint capture (high-performance snapshots)
        "rare_checkpoint_params": {
            "enable": True,
            "min_sharpe": 1.5,
            "min_sortino": 2.0,
            "max_mdd": 0.15,          # 15% drawdown
            "max_turnover": 0.80,     # 80% daily turnover
            "top_n": 5
        },
        # TAPE checkpoint trigger uses clipped terminal-bonus scale.
        # With signed mode + baseline 0.20 + scalar 10, 4.0 ~= tape_score >= 0.52.
        "tape_checkpoint_threshold": 4.0,
        # Periodic safety snapshots independent of TAPE threshold.
        "periodic_checkpoint_every_steps": 10_000,
        # Episode-level high-watermark snapshots (save on new best Sharpe >= threshold).
        "high_watermark_checkpoint_enabled": True,
        "high_watermark_sharpe_threshold": 0.5,
        # Optional per-step snapshots (very noisy; disabled by default).
        "step_sharpe_checkpoint_enabled": False,
        "step_sharpe_checkpoint_threshold": 0.5,
        
        "results_path": os.path.join(BASE_PROJECT_PATH, 'results', "phase2_advanced_prototype")
    }
}

def get_active_config(phase_name: str = None):
    """Selects the active configuration based on phase_name or environment variable."""
    if phase_name is None:
        phase_name = os.getenv("RL_PORTFOLIO_PHASE", "phase1")

    if phase_name.lower() == "phase1": return PHASE1_CONFIG
    if phase_name.lower() == "phase2": return PHASE2_CONFIG
    
    print(f"Warning: Invalid phase_name '{phase_name}'. Defaulting to Phase 1 Config.")
    return PHASE1_CONFIG

def validate_profile_manager_config(config: dict) -> bool:
    """Profile manager deprecated; keep for backwards compatibility."""
    return True

def get_available_profiles():
    """Returns a list of available profile names."""
    return [profile["name"] for profile in ALL_PROFILES_LIST]

def get_profile_by_name(profile_name: str):
    """Returns the profile configuration by name."""
    for profile in ALL_PROFILES_LIST:
        if profile["name"] == profile_name:
            return profile
    raise ValueError(f"Profile '{profile_name}' not found. Available profiles: {get_available_profiles()}")

# --- ARCHITECTURE UTILITIES ---

def get_available_architectures():
    """Returns list of supported neural network architectures."""
    return ['TCN', 'TCN_ATTENTION', 'TCN_FUSION']

def is_sequential_architecture(architecture: str) -> bool:
    """Check if architecture requires sequential (3D) input."""
    architecture_upper = architecture.upper()
    sequential_archs = ['TCN', 'TCN_ATTENTION', 'TCN_FUSION', 'GRU', 'RNN']
    return any(arch in architecture_upper for arch in sequential_archs)

def validate_agent_params(agent_params: dict) -> bool:
    """Validates agent parameters for completeness and consistency."""
    
    # Check architecture type
    if "actor_critic_type" not in agent_params:
        print("Error: 'actor_critic_type' not specified in agent_params")
        return False
    
    arch_type = agent_params["actor_critic_type"]
    available = get_available_architectures()
    
    if arch_type not in available:
        print(f"Error: Invalid architecture '{arch_type}'. Available: {available}")
        return False
    
    # Check sequence length for sequential models
    if is_sequential_architecture(arch_type):
        if "sequence_length" not in agent_params or agent_params["sequence_length"] < 2:
            print(f"Error: Sequential architecture '{arch_type}' requires sequence_length >= 2")
            return False
        
        # Check architecture-specific params
        
        if 'TCN' in arch_type and "tcn_filters" not in agent_params:
            print(f"Warning: 'tcn_filters' not specified for {arch_type}, using defaults")
        
        if 'ATTENTION' in arch_type and "attention_heads" not in agent_params:
            print(f"Warning: 'attention_heads' not specified for {arch_type}, using defaults")
    
    # Check PPO params
    if "ppo_params" not in agent_params:
        print("Error: 'ppo_params' not found in agent_params")
        return False
    
    print(f"Agent parameters validation passed for architecture: {arch_type}")
    return True


# Backward-compatible alias for phase-1 config consumers
BASE_CONFIG_TCN_PHASE1 = PHASE1_CONFIG
