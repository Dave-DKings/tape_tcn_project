"""
Utilities that extract the Phase 1 TCN architecture workflow out of the
`tcn_architecture_analysis.ipynb` notebook while preserving behaviour.

Each helper mirrors the original cell order so the notebook can import these
functions, call them, and keep its existing narrative output.
"""

from __future__ import annotations

import copy
import json
import os
import re
from platform import processor
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import tensorflow as tf
import h5py

from src.agents.ppo_agent_tf import PPOAgentTF
from src.config import PROFILE_BALANCED_GROWTH, is_sequential_architecture
from src.csv_logger import CSVLogger
from src.data_utils import DataProcessor
from src.environment_tape_rl import PortfolioEnvTAPE
from src.reward_utils import calculate_episode_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _extract_turnover_metrics(metrics: Dict[str, Any]) -> Tuple[float, float]:
    """Return turnover as decimal and percentage."""
    turnover_raw = float(metrics.get("turnover", 0.0) or 0.0)
    return turnover_raw, turnover_raw * 100.0


def _json_ready(value: Any) -> Any:
    """Convert numpy-heavy objects into JSON-serializable primitives."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def _classify_feature_group(column_name: str) -> str:
    """Map a feature column name to a compact high-level group label."""
    col = str(column_name)

    if col.startswith("LogReturn_"):
        return "log_returns"
    if col.startswith(("RollingVolatility_", "DownsideSemiVar_", "RealizedSkew_", "RealizedKurtosis_")):
        return "rolling_stats"
    if col.startswith((
        "EMA_", "BBL_", "BBM_", "BBU_", "MACD_", "MACDh_", "MACDs_", "RSI_", "STOCHk_", "STOCHd_",
        "WILLR_", "SMA_", "ADX_", "DMP_", "DMN_", "ATRr_", "NATR_", "VOL_SMA_", "OBV", "MFI_"
    )):
        return "technical_indicators"
    if col.startswith("Covariance_"):
        return "covariance"
    if col.startswith("Fundamental_"):
        return "fundamental"
    if col.startswith("Regime_"):
        return "regime"
    if col.startswith("Actuarial_"):
        return "actuarial"
    if col.startswith((
        "CrossSectional_ZScore_", "Residual_Momentum_", "Volume_Percentile_", "YieldCurve_",
        "ShortTerm_Reversal_", "VolOfVol_"
    )) or col in {"Beta_to_Market", "Market_Return_1d", "OBV_Delta_Norm_21"}:
        return "alpha_quant"
    if col.startswith("MomentumRank_") or col in {
        "BetaRank", "HighBeta_Flag", "LowBeta_Flag", "VolatilityRank", "InverseVolRank"
    } or col.endswith("_ZScore"):
        return "cross_sectional"
    if col.endswith(("_level", "_diff", "_zscore", "_yoy", "_mom", "_slope")):
        return "macro"
    return "other"


def _summarize_env_manifest(env_manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Summarize one environment block from an active feature manifest."""
    active_features = list(env_manifest.get("active_feature_columns", []) or [])
    group_counts = Counter(_classify_feature_group(col) for col in active_features)
    ordered_group_counts = {k: int(group_counts[k]) for k in sorted(group_counts.keys())}

    return {
        "feature_phase": env_manifest.get("feature_phase"),
        "active_feature_count_per_asset": int(env_manifest.get("active_feature_count_per_asset", len(active_features))),
        "flattened_observation_dim": int(env_manifest.get("flattened_observation_dim", 0)),
        "excluded_covariance_count": len(env_manifest.get("excluded_covariance_columns", []) or []),
        "missing_requested_count": len(env_manifest.get("missing_requested_columns", []) or []),
        "group_counts": ordered_group_counts,
    }


def summarize_active_feature_manifest(
    manifest_path: Union[str, Path],
    *,
    print_output: bool = True,
) -> Dict[str, Any]:
    """
    Load and summarize an active feature manifest JSON.

    Supports both:
    - Single-env format (train_rl path)
    - Multi-env format with train/test blocks (notebook helper path)
    """
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict) and "active_feature_columns" in payload:
        env_blocks = {"env": payload}
        manifest_type = "single_env"
    elif isinstance(payload, dict):
        env_blocks = {
            name: block
            for name, block in payload.items()
            if isinstance(block, dict) and "active_feature_columns" in block
        }
        manifest_type = "multi_env"
    else:
        raise ValueError(f"Unsupported manifest format: {path}")

    if not env_blocks:
        raise ValueError(f"No environment blocks found in manifest: {path}")

    summary = {
        "manifest_path": str(path),
        "manifest_type": manifest_type,
        "environments": {
            name: _summarize_env_manifest(block)
            for name, block in env_blocks.items()
        },
    }

    if print_output:
        print(f"🧾 Active Feature Manifest Summary: {path}")
        print(f"   Type: {manifest_type} | Environments: {len(env_blocks)}")
        for env_name, env_summary in summary["environments"].items():
            print(
                f"   - {env_name}: phase={env_summary['feature_phase']} | "
                f"features/asset={env_summary['active_feature_count_per_asset']} | "
                f"obs_dim={env_summary['flattened_observation_dim']}"
            )
            group_line = ", ".join(
                f"{group}={count}" for group, count in env_summary["group_counts"].items()
            )
            print(f"     groups: {group_line if group_line else 'none'}")
            print(
                f"     missing_requested={env_summary['missing_requested_count']} | "
                f"excluded_covariance={env_summary['excluded_covariance_count']}"
            )

    return summary


def _prepare_drawdown_constraint(config: Dict[str, Any], architecture: str) -> Optional[Dict[str, Any]]:
    """
    Prepare drawdown constraint settings with architecture-aware overrides.
    """
    env_params = config.get("environment_params", {})
    constraint = env_params.get("drawdown_constraint")
    if not constraint or not constraint.get("enabled", False):
        return None

    prepared = copy.deepcopy(constraint)
    overrides_root = env_params.get("drawdown_constraint_overrides", {})
    if is_sequential_architecture(architecture):
        seq_overrides = overrides_root.get("sequential", {})
        for key, value in seq_overrides.items():
            prepared[key] = value
    else:
        tcn_overrides = overrides_root.get("tcn", {})
        for key, value in tcn_overrides.items():
            prepared[key] = value

    if "tolerance" in prepared:
        prepared["tolerance"] = float(prepared["tolerance"])

    return prepared


def _infer_tcn_filters_from_checkpoint(checkpoint_prefix: str) -> Optional[List[int]]:
    """
    Inspect a saved actor checkpoint to infer the TCN block filter sizes it was trained with.

    This allows evaluation stubs to rebuild the exact architecture even if the current
    configuration was changed after training (e.g., adding/removing TCN blocks).
    """
    actor_path = Path(f"{checkpoint_prefix}_actor.weights.h5")
    if not actor_path.exists():
        return None

    try:
        import h5py  # type: ignore
    except ImportError:
        print("[create_experiment6_result_stub] h5py not available; cannot infer TCN filters from checkpoint.")
        return None

    filters: List[int] = []
    try:
        with h5py.File(actor_path, "r") as handle:
            layers_group = handle.get("layers")
            if layers_group is None:
                return None

            # Legacy/simple TCN naming pattern (tcn_block, tcn_block_1, ...).
            block_idx = 0
            while True:
                block_name = "tcn_block" if block_idx == 0 else f"tcn_block_{block_idx}"
                if block_name not in layers_group:
                    break
                block_group = layers_group[block_name]
                conv1_group = block_group.get("conv1")
                if conv1_group is None:
                    break
                vars_group = conv1_group.get("vars")
                if vars_group is None or "0" not in vars_group:
                    break
                kernel_ds = vars_group["0"]
                if len(kernel_ds.shape) < 3:
                    break
                filters.append(int(kernel_ds.shape[-1]))
                block_idx += 1

            # Fusion/attention naming pattern:
            #   tcn_fusion_actor_asset_tcn_{i}_conv1
            #   tcn_actor_tcn_{i}_conv1
            if not filters:
                indexed_filters: Dict[int, int] = {}
                for layer_name in layers_group.keys():
                    match = re.search(r"_tcn_(\d+)_conv1$", str(layer_name))
                    if not match:
                        continue
                    conv_group = layers_group.get(layer_name)
                    if conv_group is None:
                        continue
                    vars_group = conv_group.get("vars")
                    if vars_group is None or "0" not in vars_group:
                        continue
                    kernel_ds = vars_group["0"]
                    if len(kernel_ds.shape) < 3:
                        continue
                    indexed_filters[int(match.group(1))] = int(kernel_ds.shape[-1])
                if indexed_filters:
                    filters = [indexed_filters[idx] for idx in sorted(indexed_filters.keys())]
    except OSError:
        return None

    return filters or None


def _infer_fusion_input_signature_from_actor_weights(actor_weights_path: Union[str, Path]) -> Optional[Dict[str, int]]:
    """
    Infer fusion actor input signature from checkpoint kernels.

    Returns:
      {
        "conv1_in_channels": int,            # asset_tcn_0_conv1 kernel input channels
        "global_projection_in_dim": int,     # global_projection kernel input dim
      }
    """
    path = Path(actor_weights_path)
    if not path.exists():
        return None

    try:
        import h5py  # type: ignore
    except ImportError:
        return None

    try:
        with h5py.File(path, "r") as handle:
            layers_group = handle.get("layers")
            if layers_group is None:
                return None

            conv1_in: Optional[int] = None
            global_in: Optional[int] = None

            for layer_name in layers_group.keys():
                name = str(layer_name)
                layer_group = layers_group.get(layer_name)
                if layer_group is None:
                    continue
                vars_group = layer_group.get("vars")
                if vars_group is None or "0" not in vars_group:
                    continue
                kernel_ds = vars_group["0"]
                kernel_shape = tuple(int(x) for x in kernel_ds.shape)

                if name.endswith("_asset_tcn_0_conv1") and len(kernel_shape) >= 3:
                    # Conv1D kernel: (kernel_size, in_channels, out_channels)
                    conv1_in = int(kernel_shape[1])
                elif name.endswith("_global_projection") and len(kernel_shape) >= 2:
                    # Dense kernel: (in_dim, out_dim)
                    global_in = int(kernel_shape[0])

            if conv1_in is None and global_in is None:
                return None
            return {
                "conv1_in_channels": int(conv1_in or 0),
                "global_projection_in_dim": int(global_in or 0),
            }
    except OSError:
        return None


@dataclass
class Phase1Dataset:
    """Container for the processed feature set used in Phase 1 analysis."""

    master_df: pd.DataFrame
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    scalers: Dict[str, object]
    train_end_date: pd.Timestamp
    test_start_date: pd.Timestamp
    covariance_columns: List[str]
    data_processor: DataProcessor


@dataclass
class Experiment6Result:
    """Bundle of outputs produced by the Experiment 6 training helper."""

    exp_idx: int
    exp_name: str
    experiment_seed: int
    architecture: str
    use_covariance: bool
    agent: Any
    agent_config: Dict[str, Any]
    env_train: PortfolioEnvTAPE
    env_test_deterministic: PortfolioEnvTAPE
    env_test_random: PortfolioEnvTAPE
    env_test_alias: PortfolioEnvTAPE
    rare_records: List[Dict[str, Any]]
    training_summary_path: str
    training_episodes_path: str
    training_custom_path: str
    training_rows: pd.DataFrame
    checkpoint_path: str
    total_timesteps: int
    total_episodes: int
    training_duration: float
    turnover_curriculum: Dict[int, float]
    actor_lr_schedule: List[Dict[str, float]]


def _get_results_root_for_architecture(
    architecture: str,
    use_attention: bool = False,
    use_fusion: bool = False,
    project_root: Optional[Union[str, Path]] = None,
) -> Path:
    """Return the base results directory for the given architecture."""
    base_dir = Path(project_root) if project_root is not None else PROJECT_ROOT
    arch_upper = architecture.upper()
    if arch_upper.startswith("TCN"):
        if use_fusion or arch_upper == "TCN_FUSION":
            return base_dir / "tcn_fusion_results"
        if use_attention or arch_upper == "TCN_ATTENTION":
            return base_dir / "tcn_att_results"
        return base_dir / "tcn_results"
    return base_dir / "results"

def _infer_attention_from_checkpoint_path(path_like: Union[str, Path]) -> bool:
    """Return True if checkpoint path suggests TCN+Attention weights."""
    path_str = str(path_like).lower()
    return "tcn_att_results" in path_str or "_attention_" in path_str


def _infer_results_root_from_actor_weights(actor_path: Path) -> Path:
    """Infer the architecture root from an actor weights file path."""
    parent = actor_path.parent
    if parent.name in {"rare_models", "logs"}:
        return parent.parent
    return parent


def _extract_episode_number_from_name(name: str, exp_idx: int) -> Optional[int]:
    """Parse episode number from names like exp6_tape_ep31_* or exp6_tape_hw_ep31_*."""
    import re

    pattern = rf"exp{int(exp_idx)}_(?:tape_)?(?:hw_)?ep(\d+)"
    match = re.search(pattern, name)
    if not match:
        return None
    try:
        return int(match.group(1))
    except (TypeError, ValueError):
        return None


def _latest_normal_checkpoint_prefix(results_root: Path, exp_idx: int) -> Optional[Path]:
    """
    Return latest normal checkpoint prefix under results_root.

    Prefers exp{idx}_tape_epNN_actor.weights.h5 in the root directory.
    Falls back to exp{idx}_final if present.
    """
    if not results_root.exists():
        return None

    candidates: List[Tuple[int, int, int, Path]] = []
    search_specs = [
        (results_root, f"exp{exp_idx}_tape_ep*_actor.weights.h5"),
        (results_root, f"exp{exp_idx}_ep*_actor.weights.h5"),
        # High-watermark style naming used by checkpoint policy and final save.
        (results_root / "high_watermark_checkpoints", f"exp{exp_idx}_tape_hw_ep*_sh*_actor.weights.h5"),
    ]
    for search_dir, pattern in search_specs:
        if not search_dir.exists():
            continue
        for actor_path in search_dir.glob(pattern):
            ep = _extract_episode_number_from_name(actor_path.name, exp_idx)
            if ep is None:
                continue
            is_tape_name = int(f"exp{exp_idx}_tape_" in actor_path.name)
            is_hw_name = int(f"exp{exp_idx}_tape_hw_ep" in actor_path.name)
            candidates.append((ep, is_hw_name, is_tape_name, actor_path))

    if candidates:
        latest_actor = max(candidates, key=lambda item: (item[0], item[1], item[2]))[3]
        return Path(str(latest_actor).replace("_actor.weights.h5", ""))

    final_prefix_tape = results_root / f"exp{exp_idx}_tape_final"
    if Path(f"{final_prefix_tape}_actor.weights.h5").exists():
        return final_prefix_tape

    final_prefix = results_root / f"exp{exp_idx}_final"
    if Path(f"{final_prefix}_actor.weights.h5").exists():
        return final_prefix
    return None


def _best_rare_actor_checkpoint(rare_dir: Path, exp_idx: int) -> Optional[Path]:
    """
    Pick best rare checkpoint actor file.

    Priority:
    1) Highest Sharpe parsed from filename `..._shX.YYY_...`
    2) Highest episode number if sharpe cannot be parsed
    """
    import re

    if not rare_dir.exists():
        return None

    actor_files = sorted(
        {
            *rare_dir.glob(f"exp{exp_idx}_tape_ep*_actor.weights.h5"),
            *rare_dir.glob(f"exp{exp_idx}_ep*_actor.weights.h5"),
            *rare_dir.glob("exp6_tape_ep*_actor.weights.h5"),
            *rare_dir.glob("exp6_ep*_actor.weights.h5"),
        }
    )
    if not actor_files:
        return None

    ranked: List[Tuple[float, int, Path]] = []
    for p in actor_files:
        ep = _extract_episode_number_from_name(p.name, exp_idx) or -1
        sh_match = re.search(r"_sh([\-0-9\.]+)", p.name)
        sharpe = float(sh_match.group(1)) if sh_match else float("-inf")
        ranked.append((sharpe, ep, p))

    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return ranked[0][2]


def _next_incremental_checkpoint_prefix(results_root: Path, exp_idx: int) -> Path:
    """
    Return next checkpoint prefix as exp{idx}_tape_epNN based on highest episode found.

    Scans both results_root and rare_models for existing exp{idx}_tape_ep* checkpoints
    (and legacy exp{idx}_ep* checkpoints for backward compatibility).
    """
    max_ep = 0
    search_dirs = [results_root, results_root / "rare_models"]
    for directory in search_dirs:
        if not directory.exists():
            continue
        for pattern in (
            f"exp{exp_idx}_tape_ep*_*.weights.h5",
            f"exp{exp_idx}_ep*_*.weights.h5",
            f"exp{exp_idx}_tape_ep*_actor.weights.h5",
            f"exp{exp_idx}_ep*_actor.weights.h5",
        ):
            for p in directory.glob(pattern):
                ep = _extract_episode_number_from_name(p.name, exp_idx)
                if ep is not None:
                    max_ep = max(max_ep, ep)

    next_ep = max_ep + 1
    return results_root / f"exp{exp_idx}_tape_ep{next_ep}"

def _extract_effective_agent_params(
    agent_config: Dict[str, Any],
    architecture: str,
    *,
    use_attention: bool,
    use_fusion: bool,
) -> Dict[str, Any]:
    """Return architecture-effective agent params for cleaner metadata auditing."""
    arch_upper = str(architecture).upper()
    if arch_upper == "TCN" and use_fusion:
        resolved_arch = "TCN_FUSION"
    elif arch_upper == "TCN" and use_attention:
        resolved_arch = "TCN_ATTENTION"
    else:
        resolved_arch = arch_upper

    effective: Dict[str, Any] = {
        "resolved_architecture": resolved_arch,
        "actor_critic_type": agent_config.get("actor_critic_type", arch_upper),
        "use_attention": bool(agent_config.get("use_attention", use_attention)),
        "use_fusion": bool(agent_config.get("use_fusion", use_fusion)),
    }

    common_keys = [
        "sequence_length",
        "state_layout",
        "asset_feature_dim",
        "global_feature_dim",
        "dirichlet_alpha_activation",
        "dirichlet_epsilon",
        "dirichlet_exp_clip",
        "dirichlet_logit_temperature",
        "dirichlet_alpha_cap",
        "logit_temperature",
        "alpha_cap",
        "num_assets",
        "max_total_timesteps",
        "evaluation_mode",
        "debug_prints",
    ]
    for key in common_keys:
        if key in agent_config:
            effective[key] = agent_config[key]

    if "ppo_params" in agent_config:
        effective["ppo_params"] = copy.deepcopy(agent_config["ppo_params"])

    if resolved_arch.startswith("TCN"):
        for key in ["actor_hidden_dims", "critic_hidden_dims"]:
            if key in agent_config:
                effective[key] = agent_config[key]

    if resolved_arch.startswith("TCN"):
        for key in ["tcn_units", "tcn_dropout"]:
            if key in agent_config:
                effective[key] = agent_config[key]
        if resolved_arch == "TCN_ATTENTION":
            for key in ["attention_heads", "attention_dim", "attention_dropout"]:
                if key in agent_config:
                    effective[key] = agent_config[key]

    if resolved_arch.startswith("TCN"):
        for key in ["tcn_filters", "tcn_kernel_size", "tcn_dilations", "tcn_dropout"]:
            if key in agent_config:
                effective[key] = agent_config[key]

        if resolved_arch == "TCN_ATTENTION":
            for key in ["attention_heads", "attention_dim", "attention_dropout"]:
                if key in agent_config:
                    effective[key] = agent_config[key]

        if resolved_arch == "TCN_FUSION":
            for key in [
                "fusion_embed_dim",
                "fusion_attention_heads",
                "fusion_dropout",
                "fusion_cross_asset_mixer_enabled",
                "fusion_cross_asset_mixer_layers",
                "fusion_cross_asset_mixer_expansion",
                "fusion_cross_asset_mixer_dropout",
                "fusion_alpha_head_hidden_dims",
                "fusion_alpha_head_dropout",
            ]:
                if key in agent_config:
                    effective[key] = agent_config[key]

    return effective


def _checkpoint_has_attention(actor_path: Union[str, Path]) -> bool:
    """Inspect an HDF5 checkpoint to determine whether attention weights exist."""
    try:
        with h5py.File(actor_path, "r") as handle:
            found = False

            def visitor(name):
                nonlocal found
                if "attention" in name.lower():
                    found = True
            handle.visit(visitor)
            return found
    except Exception:
        return False


def _infer_checkpoint_architecture_hints(
    actor_path: Union[str, Path],
    *,
    fallback_architecture: str = "TCN",
    fallback_use_attention: bool = False,
    fallback_use_fusion: bool = False,
) -> Dict[str, Any]:
    """
    Infer architecture flags directly from checkpoint contents/path.

    Returns dict with:
      - actor_critic_type
      - use_attention
      - use_fusion
      - source (how inference was made)
    """
    actor_path = Path(actor_path)
    path_l = str(actor_path).lower()

    inferred_arch = str(fallback_architecture or "TCN").upper()
    inferred_attention = bool(fallback_use_attention)
    inferred_fusion = bool(fallback_use_fusion)
    source = "fallback"

    # Fast path-based hints.
    if "tcn_fusion_results" in path_l:
        inferred_arch, inferred_attention, inferred_fusion = "TCN_FUSION", False, True
        source = "path"
    elif "tcn_att_results" in path_l:
        inferred_arch, inferred_attention, inferred_fusion = "TCN_ATTENTION", True, False
        source = "path"

    # Inspect HDF5 layer names when available.
    try:
        with h5py.File(actor_path, "r") as handle:
            layers_group = handle.get("layers")
            if layers_group is not None:
                layer_names = [str(k).lower() for k in layers_group.keys()]
                if any(name.startswith("tcn_fusion_actor") for name in layer_names):
                    inferred_arch, inferred_attention, inferred_fusion = "TCN_FUSION", False, True
                    source = "h5_layers"
                elif any(name.startswith("tcn_attention_actor") for name in layer_names):
                    inferred_arch, inferred_attention, inferred_fusion = "TCN_ATTENTION", True, False
                    source = "h5_layers"
                elif any(name.startswith("tcn_actor") for name in layer_names):
                    inferred_arch = "TCN"
                    inferred_fusion = False
                    # Keep attention flag if generic attention layers exist.
                    inferred_attention = any("attention" in name for name in layer_names)
                    source = "h5_layers"
                elif any(name.startswith("mlp_actor") for name in layer_names):
                    inferred_arch, inferred_attention, inferred_fusion = "MLP", False, False
                    source = "h5_layers"
    except Exception:
        pass

    return {
        "actor_critic_type": inferred_arch,
        "use_attention": inferred_attention,
        "use_fusion": inferred_fusion,
        "source": source,
    }


@dataclass
class Experiment6Evaluation:
    """Summary of Experiment 6 evaluation results."""

    actor_weights_path: str
    critic_weights_path: str
    deterministic_metrics: Dict[str, float]
    deterministic_portfolio: np.ndarray
    deterministic_weights: np.ndarray  # Portfolio weights over time (deterministic)
    deterministic_actions: np.ndarray  # Raw actions from policy (deterministic)
    deterministic_alphas: np.ndarray   # Alpha values from TCN (deterministic)
    stochastic_results: pd.DataFrame
    stochastic_weights: List[np.ndarray]  # List of weight arrays, one per run
    stochastic_actions: List[np.ndarray]  # List of action arrays, one per run
    stochastic_alphas: List[np.ndarray]   # List of alpha arrays, one per run
    eval_results_path: str
    checkpoint_description: str
    agent: PPOAgentTF
    env_test_deterministic: PortfolioEnvTAPE
    env_test_random: PortfolioEnvTAPE


TRAINING_FIELDNAMES: List[str] = [
    "update",
    "timestep",
    "episode",
    "elapsed_time",
    "episode_return_pct",
    "episode_sharpe",
    "episode_sortino",
    "episode_max_dd",
    "episode_volatility",
    "episode_win_rate",
    "episode_turnover",
    "episode_turnover_pct",
    "episode_return_skew",
    "episode_calmar_ratio",
    "episode_omega_ratio",
    "episode_ulcer_index",
    "episode_cvar_5pct",
    "profile_name",
    "turnover_scalar",
    "terminal_drawdown_lambda",
    "terminal_drawdown_lambda_peak",
    "terminal_drawdown_avg_excess",
    "terminal_drawdown_penalty_sum",
    "snapshot_drawdown_lambda",
    "snapshot_drawdown_lambda_peak",
    "snapshot_drawdown_current",
    "snapshot_drawdown_avg_excess",
    "snapshot_drawdown_penalty_sum",
    "snapshot_drawdown_triggered",
    "snapshot_drawdown_trigger_boundary",
    "snapshot_drawdown_target",
    "snapshot_drawdown_tolerance",
    "snapshot_intra_step_tape_potential",
    "snapshot_intra_step_tape_delta_reward",
    "drawdown_lambda",
    "tape_score",
    "tape_bonus",
    "tape_bonus_raw",
    "tape_terminal_bonus_mode",
    "tape_terminal_baseline",
    "tape_terminal_neutral_band_applied",
    "tape_terminal_neutral_band_halfwidth",
    "tape_gate_a_triggered",
    "tape_gate_a_sharpe",
    "tape_gate_a_max_drawdown_abs",
    "terminal_intra_step_tape_potential",
    "terminal_intra_step_tape_delta_reward",
    "drawdown_avg_excess",
    "drawdown_penalty_sum",
    "initial_balance",
    "final_balance",
    "next_profile_name",
    "next_profile_reason",
    "actor_loss",
    "critic_loss",
    "critic_loss_scaled",
    "risk_aux_total",
    "risk_aux_sharpe_proxy",
    "risk_aux_sharpe_loss",
    "risk_aux_mvo_loss",
    "mean_advantage",
    "policy_entropy",
    "policy_loss",
    "entropy_loss",
    "approx_kl",
    "target_kl_active",
    "kl_stop_threshold_active",
    "ra_kl_enabled",
    "ra_kl_ema_approx_kl",
    "ra_kl_error_ratio",
    "ra_kl_adjust_factor",
    "clip_fraction",
    "value_clip_fraction",
    "explained_variance",
    "actor_grad_norm",
    "critic_grad_norm",
    "alpha_min",
    "alpha_max",
    "alpha_mean",
    "ratio_mean",
    "ratio_std",
    "drawdown_lambda_peak",
    "episode_length",
    "termination_reason",
]

EVALUATION_EXTRA_FIELDNAMES: List[str] = [
    "checkpoint",
    "architecture",
    "eval_track",
    "evaluation_type",
    "run",
    "seed",
    "test_start",
    "test_end",
    "start_date",
    "market_regime",
    "days_traded",
    "trading_years",
    "final_value",
    "total_return",
    "annualized_return",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "volatility",
    "turnover",
    "turnover_pct",
    "turnover_step_mean",
    "turnover_step_p95",
    "turnover_step_max",
    "turnover_target_step",
    "turnover_exceed_rate",
    "turnover_excess_mean",
    "raw_turnover_step_mean",
    "executed_to_raw_turnover_ratio",
    "win_rate",
    "action_uniques",
    "alpha_le1_fraction",
    "argmax_alpha_uniques",
    "mean_concentration_hhi",
    "mean_top_weight",
    "mean_action_realization_l1",
    "max_action_realization_l1",
]

EVALUATION_FIELDNAMES: List[str] = TRAINING_FIELDNAMES + [
    name for name in EVALUATION_EXTRA_FIELDNAMES if name not in TRAINING_FIELDNAMES
]

STEP_DIAGNOSTIC_FIELDNAMES: List[str] = [
    "update",
    "timestep",
    "episode",
    "episode_step",
    "date",
    "elapsed_time",
    "reward_total",
    "portfolio_return_pct_points",
    "portfolio_value",
    "prev_portfolio_value",
    "l1_w_delta",
    "turnover",
    "raw_turnover",
    "executed_turnover",
    "action_execution_beta",
    "execution_smoothing_l1",
    "turnover_target",
    "turnover_scalar",
    "turnover_penalty_contrib",
    "transaction_cost_dollars",
    "tx_cost_contrib_reward_pts",
    "action_realization_l1",
    "action_realization_penalty",
    "drawdown_penalty",
]


def _build_training_metrics_row(
    metrics: Dict[str, Any],
    *,
    episode_length: int,
    initial_balance: Optional[float],
    final_balance: Optional[float],
    profile_name: Optional[str],
    turnover_scalar: Optional[float],
    elapsed_time: Optional[float] = None,
    termination_reason: Optional[str] = None,
    drawdown_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create a dictionary containing the training-style metrics columns.
    """

    row = {field: None for field in TRAINING_FIELDNAMES}
    total_return = metrics.get("total_return", 0.0)
    sharpe = metrics.get("sharpe_ratio", 0.0)
    sortino = metrics.get("sortino_ratio", 0.0)
    max_dd = metrics.get("max_drawdown_abs", 0.0)
    volatility = metrics.get("volatility", 0.0)
    win_rate = metrics.get("win_rate", 0.0)
    turnover_raw = float(metrics.get("turnover", 0.0) or 0.0)
    turnover_pct = turnover_raw * 100.0
    skew = metrics.get("return_skew", metrics.get("skewness", 0.0))

    row.update(
        {
            "elapsed_time": elapsed_time,
            "episode_return_pct": total_return * 100.0,
            "episode_sharpe": sharpe,
            "episode_sortino": sortino,
            "episode_max_dd": max_dd * 100.0,
            "episode_volatility": volatility,
            "episode_win_rate": win_rate * 100.0,
            "episode_turnover": turnover_raw,
            "episode_turnover_pct": turnover_pct,
            "episode_return_skew": skew,
            "episode_calmar_ratio": metrics.get("calmar_ratio", 0.0),
            "episode_omega_ratio": metrics.get("omega_ratio", 0.0),
            "episode_ulcer_index": metrics.get("ulcer_index", 0.0),
            "episode_cvar_5pct": metrics.get("cvar_5pct", 0.0),
            "profile_name": profile_name or "N/A",
            "turnover_scalar": turnover_scalar,
            "initial_balance": initial_balance,
            "final_balance": final_balance,
            "episode_length": episode_length,
            "termination_reason": termination_reason,
        }
    )

    if drawdown_info:
        row["terminal_drawdown_lambda"] = drawdown_info.get("drawdown_lambda")
        row["terminal_drawdown_lambda_peak"] = drawdown_info.get("drawdown_lambda_peak")
        row["terminal_drawdown_avg_excess"] = drawdown_info.get("drawdown_avg_excess")
        row["terminal_drawdown_penalty_sum"] = drawdown_info.get("drawdown_penalty_sum")
        row["drawdown_lambda"] = drawdown_info.get("drawdown_lambda", row.get("drawdown_lambda"))
        row["drawdown_lambda_peak"] = drawdown_info.get("drawdown_lambda_peak", row.get("drawdown_lambda_peak"))
        row["tape_score"] = drawdown_info.get("tape_score", row.get("tape_score"))
        row["tape_bonus"] = drawdown_info.get("tape_bonus", row.get("tape_bonus"))
        row["tape_bonus_raw"] = drawdown_info.get("tape_bonus_raw", row.get("tape_bonus_raw"))
        row["tape_terminal_bonus_mode"] = drawdown_info.get(
            "tape_terminal_bonus_mode",
            row.get("tape_terminal_bonus_mode"),
        )
        row["tape_terminal_baseline"] = drawdown_info.get(
            "tape_terminal_baseline",
            row.get("tape_terminal_baseline"),
        )
        row["tape_terminal_neutral_band_applied"] = drawdown_info.get(
            "tape_terminal_neutral_band_applied",
            row.get("tape_terminal_neutral_band_applied"),
        )
        row["tape_terminal_neutral_band_halfwidth"] = drawdown_info.get(
            "tape_terminal_neutral_band_halfwidth",
            row.get("tape_terminal_neutral_band_halfwidth"),
        )
        row["tape_gate_a_triggered"] = drawdown_info.get("tape_gate_a_triggered", row.get("tape_gate_a_triggered"))
        row["tape_gate_a_sharpe"] = drawdown_info.get("tape_gate_a_sharpe", row.get("tape_gate_a_sharpe"))
        row["tape_gate_a_max_drawdown_abs"] = drawdown_info.get(
            "tape_gate_a_max_drawdown_abs",
            row.get("tape_gate_a_max_drawdown_abs"),
        )
        row["terminal_intra_step_tape_potential"] = drawdown_info.get(
            "intra_step_tape_potential",
            row.get("terminal_intra_step_tape_potential"),
        )
        row["terminal_intra_step_tape_delta_reward"] = drawdown_info.get(
            "intra_step_tape_delta_reward",
            row.get("terminal_intra_step_tape_delta_reward"),
        )
        row["drawdown_avg_excess"] = drawdown_info.get("drawdown_avg_excess", row.get("drawdown_avg_excess"))
        row["drawdown_penalty_sum"] = drawdown_info.get("drawdown_penalty_sum", row.get("drawdown_penalty_sum"))

    return row


def create_experiment6_result_stub(
    *,
    random_seed: int,
    exp_idx: int = 6,
    exp_name: Optional[str] = None,
    architecture: str = "tcn",
    use_covariance: bool = True,
    checkpoint_path: Optional[str] = None,
    agent_config: Optional[Dict[str, Any]] = None,
    turnover_curriculum: Optional[Dict[int, float]] = None,
    actor_lr_schedule: Optional[List[Dict[str, float]]] = None,
    base_agent_params: Optional[Dict[str, Any]] = None,
    max_total_timesteps: Optional[int] = None,
    **ignored_kwargs: Any,
) -> Experiment6Result:
    """
    Rebuild the minimal Experiment 6 metadata so checkpoints can be evaluated
    without rerunning the full training loop.
    """
    # Default max_total_timesteps if not provided
    if max_total_timesteps is None:
        max_total_timesteps = 100_000
    
    architecture_upper = architecture.upper()
    resolved_exp_name = exp_name or f"{architecture_upper} Enhanced + TAPE Three-Component"
    default_agent_config: Dict[str, Any] = {
        "actor_critic_type": architecture_upper,
        "actor_hidden_dims": [256, 128],
        "critic_hidden_dims": [256, 128],
        "sequence_length": 60,
        "tcn_filters": [32, 64, 64],
        "tcn_kernel_size": 5,
        "tcn_dilations": [2, 4, 8],
        "ppo_params": {
            "gamma": 0.99,
            "policy_clip": 0.2,
            "entropy_coef": 0.01,
            "vf_coef": 0.5,
            "num_ppo_epochs": 10,
            "batch_size_ppo": 252,
            "actor_lr": 0.00005,
            "critic_lr": 0.0005,
            "max_grad_norm": 0.5,
        },
        "debug_prints": False,
        "dirichlet_epsilon": {
            "max": 0.5,
            "min": 0.1,
        },
        "dirichlet_alpha_activation": base_agent_params.get('dirichlet_alpha_activation', 'softplus') if base_agent_params else 'softplus',
        "dirichlet_exp_clip": (-5.0, 3.0),
        "dirichlet_logit_temperature": 1.0,
        "dirichlet_alpha_cap": 100.0,
        "fusion_cross_asset_mixer_enabled": False,
        "fusion_cross_asset_mixer_layers": 1,
        "fusion_cross_asset_mixer_expansion": 2.0,
        "fusion_cross_asset_mixer_dropout": 0.1,
        "fusion_alpha_head_hidden_dims": [],
        "fusion_alpha_head_dropout": 0.1,
        "max_total_timesteps": max_total_timesteps,
    }

    # Prefer explicit runtime agent_config, then base_agent_params from caller.
    if agent_config is not None:
        resolved_agent_config = copy.deepcopy(agent_config)
    elif base_agent_params is not None:
        resolved_agent_config = copy.deepcopy(base_agent_params)
    else:
        resolved_agent_config = copy.deepcopy(default_agent_config)

    # Fill any missing keys from defaults (without overriding caller-provided values).
    for key, value in default_agent_config.items():
        if key == "ppo_params":
            resolved_agent_config.setdefault("ppo_params", {})
            for param_key, param_value in value.items():
                resolved_agent_config["ppo_params"].setdefault(param_key, copy.deepcopy(param_value))
        else:
            resolved_agent_config.setdefault(key, copy.deepcopy(value))

    # Ensure max_total_timesteps exists for epsilon/annealing paths.
    resolved_agent_config["max_total_timesteps"] = int(
        resolved_agent_config.get("max_total_timesteps", max_total_timesteps) or max_total_timesteps
    )
    if ignored_kwargs:
        print(
            f"[create_experiment6_result_stub] Ignoring unsupported kwargs: {', '.join(sorted(ignored_kwargs.keys()))}"
        )
    use_attention_flag = bool(resolved_agent_config.get("use_attention", False))
    use_fusion_flag = bool(resolved_agent_config.get("use_fusion", False))
    results_root = _get_results_root_for_architecture(
        architecture_upper,
        use_attention=use_attention_flag,
        use_fusion=use_fusion_flag,
        project_root=PROJECT_ROOT,
    )

    if checkpoint_path:
        resolved_checkpoint_path = Path(checkpoint_path)
    else:
        results_root.mkdir(parents=True, exist_ok=True)
        resolved_checkpoint_path = _latest_normal_checkpoint_prefix(results_root, exp_idx)
        if resolved_checkpoint_path is None:
            resolved_checkpoint_path = results_root / f"exp{exp_idx}_final"
        actor_candidate = Path(f"{resolved_checkpoint_path}_actor.weights.h5")
        if not actor_candidate.exists() and architecture_upper.startswith("TCN"):
            fallback_roots = []
            fallback_roots.append(
                _get_results_root_for_architecture(
                    architecture_upper,
                    use_attention=not use_attention_flag,
                    use_fusion=use_fusion_flag,
                    project_root=PROJECT_ROOT,
                )
            )
            for alt_root in fallback_roots:
                alt_prefix = _latest_normal_checkpoint_prefix(alt_root, exp_idx)
                if alt_prefix is None:
                    alt_prefix = alt_root / f"exp{exp_idx}_final"
                if Path(f"{alt_prefix}_actor.weights.h5").exists():
                    resolved_checkpoint_path = alt_prefix
                    if alt_root != results_root:
                        resolved_agent_config["use_attention"] = alt_root == _get_results_root_for_architecture(
                            architecture_upper,
                            use_attention=True,
                            use_fusion=use_fusion_flag,
                            project_root=PROJECT_ROOT,
                        )
                        results_root = alt_root
                    break
    results_root.mkdir(parents=True, exist_ok=True)

    resolved_turnover_curriculum = (
        copy.deepcopy(turnover_curriculum)
        if turnover_curriculum is not None
        else {0: 2.00, 30_000: 1.75, 60_000: 1.50, 90_000: 1.25}
    )

    # Ensure architecture string stays consistent
    resolved_agent_config["actor_critic_type"] = architecture_upper

    # Auto-align TCN filter configuration with the checkpoint, if possible
    if architecture_upper.startswith("TCN"):
        if _infer_attention_from_checkpoint_path(resolved_checkpoint_path):
            resolved_agent_config["use_attention"] = True
        inferred_filters = _infer_tcn_filters_from_checkpoint(str(resolved_checkpoint_path))
        existing_filters = resolved_agent_config.get("tcn_filters")
        if inferred_filters and inferred_filters != existing_filters:
            print(
                "[create_experiment6_result_stub] "
                f"Overriding TCN filters {existing_filters} with {inferred_filters} inferred from checkpoint."
            )
            resolved_agent_config["tcn_filters"] = inferred_filters
            dilations = resolved_agent_config.get("tcn_dilations", [])
            if not dilations:
                dilations = [1]
            if len(dilations) < len(inferred_filters):
                repeats = len(inferred_filters) // len(dilations) + int(len(inferred_filters) % len(dilations) != 0)
                dilations = (dilations * repeats)[: len(inferred_filters)]
            elif len(dilations) > len(inferred_filters):
                dilations = dilations[: len(inferred_filters)]
            resolved_agent_config["tcn_dilations"] = dilations

    resolved_actor_lr_schedule = []
    schedule_source = actor_lr_schedule or [{"threshold": 0, "lr": resolved_agent_config["ppo_params"].get("actor_lr", 0.001)}]
    for entry in schedule_source:
        threshold = int(entry.get("threshold", 0))
        lr_value = float(entry.get("lr", resolved_agent_config["ppo_params"].get("actor_lr", 0.001)))
        resolved_actor_lr_schedule.append({"threshold": threshold, "lr": lr_value})
    resolved_actor_lr_schedule = sorted(resolved_actor_lr_schedule, key=lambda item: item["threshold"])

    return Experiment6Result(
        exp_idx=exp_idx,
        exp_name=resolved_exp_name,
        experiment_seed=random_seed + exp_idx * 1000,
        architecture=architecture_upper,
        use_covariance=use_covariance,
        agent=None,
        agent_config=resolved_agent_config,
        env_train=None,
        env_test_deterministic=None,
        env_test_random=None,
        env_test_alias=None,
        rare_records=[],
        training_summary_path="",
        training_episodes_path="",
        training_custom_path="",
        training_rows=pd.DataFrame(),
        checkpoint_path=str(resolved_checkpoint_path),
        total_timesteps=0,
        total_episodes=0,
        training_duration=0.0,
        turnover_curriculum=resolved_turnover_curriculum,
        actor_lr_schedule=resolved_actor_lr_schedule,
    )


def _to_int_key_dict(raw: Dict[Any, Any]) -> Dict[int, Any]:
    """Convert JSON-loaded dict keys to ints where possible."""
    converted: Dict[int, Any] = {}
    for key, value in raw.items():
        try:
            converted[int(key)] = value
        except (TypeError, ValueError):
            continue
    return converted


def load_training_metadata_into_config(
    metadata_path: Union[str, Path],
    config: Dict[str, Any],
    *,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Load a saved training metadata JSON and re-apply core settings into `config`.

    This is intended for post-restart evaluation, so the recreated evaluation env
    can match the run configuration that produced the checkpoint.
    """
    meta_path = Path(metadata_path)
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    arch_meta = metadata.get("Architecture_Settings", {}) or {}
    train_meta = metadata.get("Training_Hyperparameters", {}) or {}
    reward_meta = metadata.get("Reward_and_Environment", {}) or {}
    checkpoint_meta = metadata.get("Checkpointing", {}) or {}

    agent_params = config.setdefault("agent_params", {})
    env_params = config.setdefault("environment_params", {})
    training_params = config.setdefault("training_params", {})

    # Prefer effective snapshot because it already reflects overrides used at runtime.
    effective_agent_params = arch_meta.get("agent_params_effective")
    if isinstance(effective_agent_params, dict) and effective_agent_params:
        for key, value in effective_agent_params.items():
            agent_params[key] = copy.deepcopy(value)
    else:
        for key in ("actor_critic_type", "use_attention", "use_fusion"):
            if key in arch_meta:
                agent_params[key] = copy.deepcopy(arch_meta[key])

    if "actor_lr_schedule" in train_meta:
        schedule = train_meta.get("actor_lr_schedule", [])
        if isinstance(schedule, list):
            training_params["actor_lr_schedule"] = copy.deepcopy(schedule)
    if "turnover_penalty_curriculum" in train_meta:
        raw_curr = train_meta.get("turnover_penalty_curriculum", {})
        if isinstance(raw_curr, dict):
            training_params["turnover_penalty_curriculum"] = _to_int_key_dict(raw_curr)
    for key in (
        "max_total_timesteps",
        "timesteps_per_ppo_update",
        "timesteps_per_ppo_update_schedule",
        "batch_size_ppo",
        "batch_size_ppo_schedule",
        "action_execution_beta_curriculum",
        "evaluation_action_execution_beta",
        "evaluation_turnover_penalty_scalar",
        "ra_kl_enabled",
        "ra_kl_target_ratio",
        "ra_kl_ema_alpha",
        "ra_kl_gain",
        "ra_kl_deadband",
        "ra_kl_max_change_fraction",
        "ra_kl_min_target_kl",
        "ra_kl_max_target_kl",
        "log_step_diagnostics",
        "alpha_diversity_log_interval",
        "alpha_diversity_warning_after_updates",
        "alpha_diversity_warning_std_threshold",
    ):
        if key in train_meta:
            training_params[key] = copy.deepcopy(train_meta[key])
    if "use_episode_length_curriculum" in train_meta:
        training_params["use_episode_length_curriculum"] = bool(train_meta["use_episode_length_curriculum"])
    if "episode_length_curriculum_schedule" in train_meta:
        schedule = train_meta.get("episode_length_curriculum_schedule")
        if isinstance(schedule, list):
            training_params["episode_length_curriculum_schedule"] = copy.deepcopy(schedule)

    for key in (
        "dsr_scalar",
        "target_turnover",
        "turnover_target_band",
        "tape_terminal_scalar",
        "tape_terminal_clip",
        "tape_terminal_bonus_mode",
        "tape_terminal_baseline",
        "tape_terminal_neutral_band_enabled",
        "tape_terminal_neutral_band_halfwidth",
        "tape_terminal_gate_a_enabled",
        "tape_terminal_gate_a_sharpe_threshold",
        "tape_terminal_gate_a_max_drawdown",
        "action_execution_beta",
        "reward_credit_assignment_mode",
        "retroactive_episode_reward_scaling",
        "training_entrypoint",
    ):
        if key in reward_meta:
            env_params[key] = copy.deepcopy(reward_meta[key])

    if isinstance(reward_meta.get("drawdown_constraint"), dict):
        env_params["drawdown_constraint"] = copy.deepcopy(reward_meta["drawdown_constraint"])

    for key in (
        "deterministic_validation_checkpointing_enabled",
        "deterministic_validation_checkpointing_only",
        "deterministic_validation_eval_every_episodes",
        "deterministic_validation_mode",
        "deterministic_validation_episode_length_limit",
        "deterministic_validation_sharpe_min",
        "deterministic_validation_sharpe_min_delta",
        "deterministic_validation_seed_offset",
        "deterministic_validation_log_alpha_stats",
        "high_watermark_checkpoint_enabled",
        "high_watermark_sharpe_threshold",
        "step_sharpe_checkpoint_enabled",
        "step_sharpe_checkpoint_threshold",
        "periodic_checkpoint_every_steps",
    ):
        if key in checkpoint_meta:
            training_params[key] = copy.deepcopy(checkpoint_meta[key])

    # Keep a profile override so evaluation can align after kernel restart.
    if isinstance(reward_meta.get("tape_profile_full"), dict):
        env_params["tape_profile_override"] = copy.deepcopy(reward_meta["tape_profile_full"])

    if verbose:
        run_ctx = metadata.get("Run_Context", {}) or {}
        feature_meta = metadata.get("Feature_Groups", {}) or {}
        actuarial_cols_meta = feature_meta.get("actuarial_columns_detected", []) or []
        actuarial_missing_meta = feature_meta.get("actuarial_columns_missing_from_feature_list", []) or []
        print("✅ Applied training metadata to config")
        print(f"   Metadata: {meta_path}")
        if run_ctx.get("timestamp"):
            print(f"   Run timestamp: {run_ctx.get('timestamp')}")
        print(f"   Architecture: {agent_params.get('actor_critic_type')}")
        if all(k in agent_params for k in ("tcn_filters", "tcn_dilations")):
            print(
                "   TCN stack: "
                f"filters={agent_params.get('tcn_filters')} | "
                f"kernel={agent_params.get('tcn_kernel_size')} | "
                f"dilations={agent_params.get('tcn_dilations')} | "
                f"dropout={agent_params.get('tcn_dropout')}"
            )
        if str(agent_params.get("actor_critic_type", "")).upper().startswith("TCN_FUSION") or bool(agent_params.get("use_fusion", False)):
            print(
                "   Fusion core: "
                f"embed={agent_params.get('fusion_embed_dim')} | "
                f"heads={agent_params.get('fusion_attention_heads')} | "
                f"dropout={agent_params.get('fusion_dropout')}"
            )
            print(
                "   Fusion mixer (A4): "
                f"enabled={bool(agent_params.get('fusion_cross_asset_mixer_enabled', False))} | "
                f"layers={agent_params.get('fusion_cross_asset_mixer_layers', 1)} | "
                f"expansion={agent_params.get('fusion_cross_asset_mixer_expansion', 2.0)} | "
                f"dropout={agent_params.get('fusion_cross_asset_mixer_dropout', agent_params.get('fusion_dropout'))}"
            )
            print(
                "   Fusion alpha head (A3): "
                f"dims={agent_params.get('fusion_alpha_head_hidden_dims', [])} | "
                f"dropout={agent_params.get('fusion_alpha_head_dropout', agent_params.get('fusion_dropout'))}"
            )
        print(
            "   Dirichlet controls: "
            f"activation={agent_params.get('dirichlet_alpha_activation')} | "
            f"temperature={agent_params.get('dirichlet_logit_temperature', agent_params.get('logit_temperature', 1.0))} | "
            f"alpha_cap={agent_params.get('dirichlet_alpha_cap', agent_params.get('alpha_cap', None))} | "
            f"epsilon={agent_params.get('dirichlet_epsilon')}"
        )
        print(f"   Turnover target: {env_params.get('target_turnover')}")
        print(f"   DSR scalar: {env_params.get('dsr_scalar')}")
        if training_params.get("timesteps_per_ppo_update_schedule"):
            print("   PPO update timesteps: scheduled")
        else:
            print(f"   PPO update timesteps: {training_params.get('timesteps_per_ppo_update')}")
        print(f"   Episode length curriculum: {training_params.get('use_episode_length_curriculum')}")
        print(f"   RA-KL enabled: {bool(training_params.get('ra_kl_enabled', False))}")
        if "deterministic_validation_checkpointing_enabled" in training_params:
            print(
                "   Deterministic validation checkpointing: "
                f"{bool(training_params.get('deterministic_validation_checkpointing_enabled'))} "
                f"(mode={training_params.get('deterministic_validation_mode', 'mean')}, "
                f"every={training_params.get('deterministic_validation_eval_every_episodes', 5)})"
            )
        print(f"   Profile override loaded: {'tape_profile_override' in env_params}")
        print(
            f"   Credit assignment mode: "
            f"{env_params.get('reward_credit_assignment_mode', 'step_reward_plus_terminal_bonus')}"
        )
        print(
            f"   Retroactive episode scaling: "
            f"{bool(env_params.get('retroactive_episode_reward_scaling', False))}"
        )
        print(f"   Actuarial columns detected: {len(actuarial_cols_meta)}")
        if actuarial_missing_meta:
            print(f"   ⚠️ Actuarial columns missing from feature list: {actuarial_missing_meta}")

    return config


def _save_phase1_preparation_artifacts(
    *,
    config: Dict[str, Any],
    processor: DataProcessor,
    raw_df: pd.DataFrame,
    engineered_full_df: pd.DataFrame,
    engineered_analysis_df: pd.DataFrame,
    normalized_master_df: pd.DataFrame,
    train_norm_df: pd.DataFrame,
    test_norm_df: pd.DataFrame,
    feature_cols: List[str],
    scalers: Dict[str, Any],
    train_end_date: Optional[pd.Timestamp],
    test_start_date: Optional[pd.Timestamp],
    artifacts_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, str]:
    """
    Persist key phase-1 preparation artifacts for reproducibility and auditing.
    """
    base_path = Path(config.get("BASE_DATA_PATH", "data"))
    if artifacts_dir is None:
        configured = config.get("training_params", {}).get("phase1_artifacts_dir")
        artifacts_root = Path(configured) if configured else (base_path / "phase1_preparation_artifacts")
    else:
        artifacts_root = Path(artifacts_dir)
    artifacts_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"phase1_prep_{timestamp}"

    paths = {
        "raw_ohlcv": artifacts_root / f"{stem}_raw_ohlcv.csv",
        "feature_engineered_full": artifacts_root / f"{stem}_feature_engineered_full.csv",
        "feature_engineered_analysis_window": artifacts_root / f"{stem}_feature_engineered_analysis_window.csv",
        "feature_engineered_normalized": artifacts_root / f"{stem}_feature_engineered_normalized.csv",
        "train_normalized": artifacts_root / f"{stem}_train_normalized.csv",
        "test_normalized": artifacts_root / f"{stem}_test_normalized.csv",
        "scalers": artifacts_root / f"{stem}_scalers.joblib",
        "audit_report": artifacts_root / f"{stem}_preparation_audit.json",
    }

    raw_df.to_csv(paths["raw_ohlcv"], index=False)
    engineered_full_df.to_csv(paths["feature_engineered_full"], index=False)
    engineered_analysis_df.to_csv(paths["feature_engineered_analysis_window"], index=False)
    normalized_master_df.to_csv(paths["feature_engineered_normalized"], index=False)
    train_norm_df.to_csv(paths["train_normalized"], index=False)
    test_norm_df.to_csv(paths["test_normalized"], index=False)
    processor.save_scalers(scalers, str(paths["scalers"]))

    method_counts: Counter = Counter()
    feature_methods: Dict[str, str] = {}
    for col in feature_cols:
        spec = scalers.get(col)
        if isinstance(spec, dict):
            method = str(spec.get("method", "unknown"))
        elif spec is None:
            method = "missing"
        else:
            method = "legacy_scaler"
        method_counts[method] += 1
        feature_methods[col] = method

    binary_flag_cols = [
        col for col in feature_cols
        if col.endswith("_Flag") or col == "YieldCurve_Inverted_Flag"
    ]
    binary_flag_non_bounded = [
        col for col in binary_flag_cols if feature_methods.get(col) != "bounded"
    ]

    train_max = pd.to_datetime(train_norm_df["Date"]).max() if not train_norm_df.empty else None
    test_min = pd.to_datetime(test_norm_df["Date"]).min() if not test_norm_df.empty else None
    split_non_overlap = bool(train_max < test_min) if (train_max is not None and test_min is not None) else False

    audit_report = {
        "timestamp": timestamp,
        "train_end_date": str(pd.to_datetime(train_end_date)) if train_end_date is not None else None,
        "test_start_date": str(pd.to_datetime(test_start_date)) if test_start_date is not None else None,
        "split_check": {
            "train_max_date": str(train_max) if train_max is not None else None,
            "test_min_date": str(test_min) if test_min is not None else None,
            "non_overlap_train_before_test": split_non_overlap,
        },
        "normalization": {
            "fit_scope": "train_only",
            "scaler_count": int(len(scalers)),
            "feature_count": int(len(feature_cols)),
            "method_counts": {k: int(v) for k, v in method_counts.items()},
            "binary_flag_columns": binary_flag_cols,
            "binary_flag_columns_non_bounded": binary_flag_non_bounded,
        },
        "artifact_paths": {k: str(v) for k, v in paths.items()},
    }

    with open(paths["audit_report"], "w", encoding="utf-8") as f:
        json.dump(_json_ready(audit_report), f, indent=2)

    return {k: str(v) for k, v in paths.items()}


def configure_episode_length_curriculum(
    config: Dict,
    enable: bool,
) -> Dict:
    """
    Toggle the `use_episode_length_curriculum` flag in-place and return config.

    The notebook relied on mutating the shared configuration object; we mirror
    that behaviour but also return the config so callers can use a functional
    style if they prefer.
    """
    training_params = config.setdefault("training_params", {})
    training_params["use_episode_length_curriculum"] = bool(enable)
    return config


def prepare_phase1_dataset(
    config: Dict,
    *,
    periods: Iterable[int] = (1, 5, 10, 21),
    train_fraction: float = 0.8,
    scaler_type: str = "standard",
    force_download: bool = False,
    save_preparation_artifacts: bool = True,
    preparation_artifacts_dir: Optional[Union[str, Path]] = None,
) -> Phase1Dataset:
    """
    Run the exact Phase 1 feature-engineering pipeline and return splits.

    Steps replicated from the notebook:
    1. Instantiate `DataProcessor` with the config
    2. Load OHLCV data (optionally forcing a refresh)
    3. Calculate log returns, rolling stats, indicators, and covariance features
    4. Filter to analysis date range (ANALYSIS_START_DATE to ANALYSIS_END_DATE)
    5. Split into train/test by fixed date (TRAIN_TEST_SPLIT_DATE) or fraction
    6. Normalise features using train-only statistics
    7. Identify covariance feature columns
    """
    processor = DataProcessor(config)

    print("📊 Loading raw market data...")
    raw_df = processor.load_ohlcv_data(
        start_date=config.get("DATA_FETCH_START_DATE"),
        end_date=config.get("DATA_FETCH_END_DATE"),
        force_download=force_download,
    )
    print(f"   ✅ Raw data shape: {raw_df.shape}")
    print(f"   ✅ Date range: {raw_df['Date'].min()} → {raw_df['Date'].max()}")

    print(f"\n🔧 Computing multi-horizon log returns: {list(periods)}")
    df_with_returns = processor.calculate_log_returns(raw_df, periods=list(periods))
    print(f"   ✅ Shape after returns: {df_with_returns.shape}")

    print("\n📈 Calculating 21-day rolling statistics")
    df_with_stats = processor.calculate_return_statistics(df_with_returns, window=21)

    print("\n🧮 Computing technical indicators")
    ti_configs = config.get("feature_params", {}).get("technical_indicators", [])
    df_with_indicators = processor.calculate_technical_indicators(
        df_with_stats,
        ti_configs=ti_configs,
    )

    print("\n📊 Computing dynamic covariance features")
    master_df = processor.calculate_dynamic_covariance_features(df_with_indicators)

    print("\n🎯 Adding regime awareness features")
    master_df = processor.add_regime_features(master_df)
    
    print(f"   ✅ Master DF shape: {master_df.shape}")
    print(f"   ✅ Total features: {len(master_df.columns)}")

    # Add fundamental features (if enabled)
    print("\n📊 Integrating fundamental features (if enabled)...")
    master_df = processor.add_fundamental_features(master_df)
    fundamental_cols = [col for col in master_df.columns if col.startswith("Fundamental_")]
    print(
        f"   ✅ Fundamental columns in dataset: {len(fundamental_cols)} "
        f"(enabled={config.get('feature_params', {}).get('fundamental_features', {}).get('enabled', False)})"
    )
    if fundamental_cols:
        preview = fundamental_cols[:8]
        suffix = " ..." if len(fundamental_cols) > 8 else ""
        print(f"   🧾 Sample fundamental cols: {preview}{suffix}")
    
    # Add macroeconomic features (if enabled)
    print("\n📊 Integrating macroeconomic features (if enabled)...")
    macro_config = config.get('feature_params', {}).get('macro_data')
    if macro_config is not None:
        macro_df, macro_cols = processor._build_macro_feature_frame(
            macro_config,
            master_df[processor.date_col].min(),
            master_df[processor.date_col].max()
        )
        if macro_df is not None and macro_cols:
            master_df = master_df.merge(macro_df, on=processor.date_col, how='left')
            print(f"   ✅ Macro features added - {len(macro_cols)} columns: {macro_cols}")
        else:
            print("   ⚠️ Macro configuration provided but no features were generated.")
    else:
        print("   ⚠️ Macro features disabled (config is None).")

    # Add Alpha features (if enabled)
    print("\n📊 Integrating Alpha features (if enabled)...")
    master_df = processor.add_quant_alpha_features(master_df)

    # Add actuarial features (if enabled)
    print("\n📊 Integrating actuarial features (if enabled)...")
    master_df = processor.add_actuarial_features(master_df)
    actuarial_cols = [col for col in master_df.columns if col.startswith("Actuarial_")]
    actuarial_enabled = bool(
        config.get("feature_params", {}).get("actuarial_params", {}).get("enabled", False)
    )
    if actuarial_cols:
        non_null_counts = {
            col: int(master_df[col].notna().sum())
            for col in sorted(actuarial_cols)
        }
        print(
            f"   ✅ Actuarial columns in dataset: {len(actuarial_cols)} "
            f"(enabled={actuarial_enabled})"
        )
        print(f"   📋 Non-null counts: {non_null_counts}")
    elif actuarial_enabled:
        print("   ⚠️ Actuarial is enabled but no Actuarial_ columns were produced.")
    else:
        print("   ℹ️ Actuarial features disabled by config.")
    
    print(f"\n✅ Final master DF shape: {master_df.shape}")
    print(f"   ✅ Total features: {len(master_df.columns)}")
    engineered_full_df = master_df.copy()

    # Ensure Date column is timezone-naive for downstream processing
    date_col = processor.date_col
    if date_col in master_df.columns:
        dates = pd.to_datetime(master_df[date_col], utc=True, errors="coerce")
        try:
            master_df[date_col] = dates.dt.tz_localize(None)
        except (AttributeError, TypeError):
            master_df[date_col] = dates

    feature_cols = processor.get_feature_columns("phase1")
    available_cols = [col for col in feature_cols if col in master_df.columns]
    missing_cols = [col for col in feature_cols if col not in master_df.columns]
    selection_cfg = config.get("feature_params", {}).get("feature_selection", {})
    if selection_cfg.get("enforce_allowlist", False):
        plan_name = selection_cfg.get("feature_audit_plan_name", "feature_audit_allowlist")
        expected_total = selection_cfg.get("feature_audit_expected_total_count")
        print(f"🧭 Feature audit plan: {plan_name} (allowlist enabled)")
        print(f"   active feature count (phase1): {len(feature_cols)}")
        if expected_total is not None:
            try:
                exp = int(expected_total)
                status = "✅" if len(feature_cols) == exp else "⚠️"
                print(f"   {status} expected active features: {exp}")
            except Exception:
                pass
    if missing_cols:
        print(f"⚠️ Missing features before normalisation: {missing_cols}")

    # Filter to analysis date range BEFORE splitting
    analysis_start = config.get('ANALYSIS_START_DATE', '2003-09-02')
    analysis_end = config.get('ANALYSIS_END_DATE', '2024-09-01')
    
    print("\n" + "="*80)
    print("✂️ FILTERING TO ANALYSIS PERIOD")
    print("="*80)
    print(f"   Filtering data to: {analysis_start} → {analysis_end}")
    
    master_df = master_df[
        (master_df[date_col] >= pd.to_datetime(analysis_start)) &
        (master_df[date_col] <= pd.to_datetime(analysis_end))
    ].copy()
    engineered_analysis_df = master_df.copy()
    
    unique_dates_filtered = sorted(master_df[date_col].unique())
    print(f"   ✅ Dates after filter: {len(unique_dates_filtered)} trading days")
    print(f"   ✅ Date range: {unique_dates_filtered[0]} to {unique_dates_filtered[-1]}")
    print("="*80)

    # Check if fixed split date is specified, otherwise use percentage
    split_date = config.get('TRAIN_TEST_SPLIT_DATE')
    
    if split_date:
        # Fixed-date split
        train_df, test_df, train_end_date, test_start_date = split_dataset_by_date(
            master_df,
            date_column=date_col,
            split_date=split_date,
        )
    else:
        # Percentage-based split (backward compatibility)
        train_df, test_df, train_end_date, test_start_date = split_dataset_by_date(
            master_df,
            date_column=date_col,
            train_fraction=train_fraction,
        )

    print(f"\n🔧 NORMALISING FEATURES ({scaler_type} scaler)")
    master_df_norm, scalers = processor.normalize_features(
        master_df,
        feature_cols=feature_cols,
        train_end_date=train_end_date,
        scaler_type=scaler_type,
    )

    # Normalisation operates in-place; reflect that in split frames.
    train_df_norm = master_df_norm[master_df_norm["Date"] <= train_end_date].copy()
    test_df_norm = master_df_norm[master_df_norm["Date"] > train_end_date].copy()

    covariance_columns = identify_covariance_columns(master_df_norm.columns)

    base_path = config.get("BASE_DATA_PATH", "data")
    destination = os.path.join(base_path, "master_features_NORMALIZED.csv")
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    print(f"\n💾 Saving NORMALISED master dataframe to '{destination}'")
    master_df_norm.to_csv(destination, index=False)

    if save_preparation_artifacts:
        saved_paths = _save_phase1_preparation_artifacts(
            config=config,
            processor=processor,
            raw_df=raw_df,
            engineered_full_df=engineered_full_df,
            engineered_analysis_df=engineered_analysis_df,
            normalized_master_df=master_df_norm,
            train_norm_df=train_df_norm,
            test_norm_df=test_df_norm,
            feature_cols=feature_cols,
            scalers=scalers,
            train_end_date=train_end_date,
            test_start_date=test_start_date,
            artifacts_dir=preparation_artifacts_dir,
        )
        print("\n💾 Saved preparation artifacts:")
        print(f"   raw OHLCV: {saved_paths['raw_ohlcv']}")
        print(f"   full engineered: {saved_paths['feature_engineered_full']}")
        print(f"   analysis-window engineered: {saved_paths['feature_engineered_analysis_window']}")
        print(f"   normalized master: {saved_paths['feature_engineered_normalized']}")
        print(f"   train normalized: {saved_paths['train_normalized']}")
        print(f"   test normalized: {saved_paths['test_normalized']}")
        print(f"   scalers: {saved_paths['scalers']}")
        print(f"   audit report: {saved_paths['audit_report']}")

    return Phase1Dataset(
        master_df=master_df_norm,
        train_df=train_df_norm,
        test_df=test_df_norm,
        scalers=scalers,
        train_end_date=train_end_date,
        test_start_date=test_start_date,
        covariance_columns=covariance_columns,
        data_processor=processor,
    )


def split_dataset_by_date(
    df: pd.DataFrame,
    *,
    date_column: str = "Date",
    train_fraction: float = 0.8,
    split_date: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """
    Perform time-based train/test split.
    
    Args:
        df: DataFrame with date column
        date_column: Name of date column
        train_fraction: Fraction for training (used if split_date is None)
        split_date: Fixed date to split on (e.g., "2020-12-31"). If provided,
                   train data <= split_date, test data > split_date.
    
    Returns:
        train_df, test_df, train_end_date, test_start_date
    """
    unique_dates = sorted(pd.to_datetime(df[date_column].unique()))
    
    if split_date:
        # Fixed-date split
        split_ts = pd.to_datetime(split_date)
        train_dates = [d for d in unique_dates if d <= split_ts]
        test_dates = [d for d in unique_dates if d > split_ts]
        split_method = f"Fixed date: {split_date}"
    else:
        # Percentage-based split
        train_count = int(len(unique_dates) * train_fraction)
        train_dates = unique_dates[:train_count]
        test_dates = unique_dates[train_count:]
        split_method = f"{train_fraction*100:.0f}/{(1-train_fraction)*100:.0f} split"

    train_df = df[df[date_column].isin(train_dates)].copy()
    test_df = df[df[date_column].isin(test_dates)].copy()

    train_end_date = pd.to_datetime(train_dates[-1]) if train_dates else None
    test_start_date = pd.to_datetime(test_dates[0]) if test_dates else train_end_date

    print("=" * 80)
    print(f"✂️  TIME-BASED TRAIN/TEST SPLIT ({split_method})")
    if train_dates:
        train_years = len(train_dates) / 252
        print(f"   Train: {train_dates[0].date()} → {train_dates[-1].date()} "
              f"({len(train_dates)} days, {train_years:.1f} years, {len(train_df)} rows)")
    else:
        print("   ⚠️  No training dates available (empty dataset).")
    if test_dates:
        test_years = len(test_dates) / 252
        print(f"   Test:  {test_dates[0].date()} → {test_dates[-1].date()} "
              f"({len(test_dates)} days, {test_years:.1f} years, {len(test_df)} rows)")
    else:
        print("   ⚠️  No test dates found; entire dataset used for training.")
    print("=" * 80)

    return train_df, test_df, train_end_date, test_start_date


def identify_covariance_columns(columns: Iterable[str]) -> List[str]:
    """Replicate the notebook's heuristic for covariance feature selection."""
    return [
        col
        for col in columns
        if col.lower().startswith("covariance_eigenvalue_")
        or col.lower().startswith("covariance_")
    ]


def build_stage1_experiments(
    agents_available: Dict[str, object]
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[str]]:
    """
    Re-create the Stage 1 experiment definitions exactly as the notebook declared them.
    """
    stage1 = [
        {
            "name": "TCN Baseline",
            "stage": 1,
            "architecture": "tcn",
            "env_type": "custom",
            "use_covariance": False,
            "terminal_reward": "standard",
            "description": "Dense TCN without covariance - baseline performance",
        },
        {
            "name": "TCN Enhanced",
            "stage": 1,
            "architecture": "tcn",
            "env_type": "custom",
            "use_covariance": True,
            "terminal_reward": "standard",
            "description": "Dense TCN with covariance - measure feature impact",
        },
        {
            "name": "TCN Baseline Sharpe",
            "stage": 1,
            "architecture": "tcn",
            "env_type": "custom",
            "use_covariance": False,
            "terminal_reward": "sharpe",
            "description": "Dense TCN without covariance - Sharpe terminal reward",
        },
        {
            "name": "TCN Enhanced Sharpe",
            "stage": 1,
            "architecture": "tcn",
            "env_type": "custom",
            "use_covariance": True,
            "terminal_reward": "sharpe",
            "description": "Dense TCN with covariance - Sharpe terminal reward",
        },
    ]

    filtered = [exp for exp in stage1 if exp["architecture"] in agents_available]
    skipped = [exp["name"] for exp in stage1 if exp["architecture"] not in agents_available]
    return filtered, stage1, skipped


def build_stage2_experiments() -> List[Dict[str, object]]:
    """
    Provide the Stage 2 templates; the notebook fills in the TBD fields later.
    """
    return [
        {
            "name": "Winning Custom Agent",
            "stage": 2,
            "architecture": "TBD",
            "env_type": "custom",
            "use_covariance": "TBD",
            "description": "Best performer from Stage 1 - custom reward system",
        },
        {
            "name": "FinRL Equivalent",
            "stage": 2,
            "architecture": "TBD",
            "env_type": "finrl",
            "use_covariance": "TBD",
            "description": "Same architecture in FinRL environment - absolute $ reward",
        },
    ]


def run_experiment6_tape(
    phase1_data: Phase1Dataset,
    config: Dict[str, Any],
    *,
    random_seed: int,
    exp_idx: int = 6,
    exp_name: Optional[str] = None,
    architecture: str = "tcn",
    use_covariance: bool = True,
    profile: Optional[Dict[str, Any]] = None,
    agent_cls: Optional[Any] = None,
    csv_logger_cls: Optional[Any] = None,
    timesteps_per_update: Optional[int] = None,
    max_total_timesteps: Optional[int] = None,
) -> Experiment6Result:
    """
    Reproduce the Experiment 6 training loop (TCN + Three-Component TAPE).

    This mirrors the original notebook cell, but packages it as a reusable helper.
    Prints and file outputs remain identical so notebook narrative still applies.
    """
    profile = profile or PROFILE_BALANCED_GROWTH
    agent_cls = agent_cls or PPOAgentTF
    logger_cls = csv_logger_cls if csv_logger_cls is not None else CSVLogger

    experiment_seed = random_seed + exp_idx * 1000

    training_params = config.get("training_params", {})

    timesteps_per_update = int(
        timesteps_per_update
        if timesteps_per_update is not None
        else training_params.get("timesteps_per_ppo_update", 256)
    )
    max_total_timesteps = int(
        max_total_timesteps
        if max_total_timesteps is not None
        else training_params.get("max_total_timesteps", 100_000)
    )
    raw_timestep_schedule_cfg = training_params.get("timesteps_per_ppo_update_schedule", [])
    timestep_update_schedule: List[Dict[str, int]] = []
    if isinstance(raw_timestep_schedule_cfg, dict):
        for threshold_raw, rollout_len_raw in raw_timestep_schedule_cfg.items():
            threshold = int(threshold_raw)
            rollout_len = int(rollout_len_raw)
            if rollout_len <= 0:
                continue
            timestep_update_schedule.append(
                {"threshold": threshold, "timesteps_per_update": rollout_len}
            )
    elif isinstance(raw_timestep_schedule_cfg, list):
        for entry in raw_timestep_schedule_cfg:
            if not isinstance(entry, dict):
                continue
            threshold = int(entry.get("threshold", 0))
            rollout_len = entry.get("timesteps_per_update", entry.get("timesteps", entry.get("value")))
            if rollout_len is None:
                continue
            rollout_len = int(rollout_len)
            if rollout_len <= 0:
                continue
            timestep_update_schedule.append(
                {"threshold": threshold, "timesteps_per_update": rollout_len}
            )
    if not timestep_update_schedule:
        timestep_update_schedule = [{"threshold": 0, "timesteps_per_update": timesteps_per_update}]
    else:
        schedule_by_threshold: Dict[int, int] = {}
        for entry in sorted(timestep_update_schedule, key=lambda item: item["threshold"]):
            schedule_by_threshold[int(entry["threshold"])] = int(entry["timesteps_per_update"])
        if 0 not in schedule_by_threshold:
            schedule_by_threshold[0] = timesteps_per_update
        timestep_update_schedule = [
            {"threshold": int(threshold), "timesteps_per_update": int(value)}
            for threshold, value in sorted(schedule_by_threshold.items(), key=lambda item: item[0])
        ]

    def determine_timesteps_per_update(current_step: int) -> int:
        active_rollout = timestep_update_schedule[0]["timesteps_per_update"]
        for entry in timestep_update_schedule:
            if current_step >= entry["threshold"]:
                active_rollout = entry["timesteps_per_update"]
            else:
                break
        return max(1, int(active_rollout))

    arch_upper = architecture.upper()
    use_attention_flag = bool(config.get("agent_params", {}).get("use_attention", False))
    use_fusion_flag = bool(config.get("agent_params", {}).get("use_fusion", False))
    resolved_exp_name = exp_name or f"{arch_upper} Enhanced + TAPE Three-Component"
    
    # Set up results directory
    results_root = _get_results_root_for_architecture(
        arch_upper,
        use_attention=use_attention_flag,
        use_fusion=use_fusion_flag,
        project_root=PROJECT_ROOT,
    )
    results_root.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"EXPERIMENT {exp_idx}: {resolved_exp_name}")
    print(f"{'='*80}")
    architecture_label = arch_upper
    if arch_upper in {"TCN_FUSION"} or (arch_upper.startswith("TCN") and use_fusion_flag):
        architecture_label = "TCN + Fusion"
    elif arch_upper.startswith("TCN") and use_attention_flag:
        architecture_label = "TCN + Attention"
    print(f"Architecture: {architecture_label}")
    print(f"Results root: {results_root.resolve()}")
    print(f"Working dir: {Path.cwd().resolve()}")
    print(f"Covariance Features: {'Yes' if use_covariance else 'No'}")
    print(f"🎯 REWARD SYSTEM: TAPE (Three-Component v3)")
    print(f"   Profile: {profile.get('name', 'BALANCED_GROWTH') if isinstance(profile, dict) else 'BALANCED_GROWTH'}")
    print(f"   Daily: Base + DSR/PBRS + Turnover_Proximity")
    print(
        "   Terminal: "
        f"mode={config.get('environment_params', {}).get('tape_terminal_bonus_mode', 'signed')} | "
        f"baseline={config.get('environment_params', {}).get('tape_terminal_baseline', 0.20):.2f} | "
        f"scalar={config.get('environment_params', {}).get('tape_terminal_scalar', 10.0)} "
        f"(clipped ±{config.get('environment_params', {}).get('tape_terminal_clip', 10.0)})"
    )
    gate_cfg = config.get("environment_params", {})
    gate_enabled = bool(gate_cfg.get("tape_terminal_gate_a_enabled", False))
    gate_sharpe = float(gate_cfg.get("tape_terminal_gate_a_sharpe_threshold", 0.0))
    gate_mdd = float(gate_cfg.get("tape_terminal_gate_a_max_drawdown", 0.25))
    print(
        "   Gate A: "
        f"{'enabled' if gate_enabled else 'disabled'} "
        f"(Sharpe ≤ {gate_sharpe:.2f} "
        f"or MDD ≥ {gate_mdd*100:.1f}% -> force non-positive terminal bonus)"
    )
    gate_neutral_enabled = bool(gate_cfg.get("tape_terminal_neutral_band_enabled", True))
    gate_neutral_halfwidth = float(gate_cfg.get("tape_terminal_neutral_band_halfwidth", 0.02))
    print(
        "   Neutral Band: "
        f"{'enabled' if gate_neutral_enabled else 'disabled'} "
        f"(±{gate_neutral_halfwidth:.3f} around baseline)"
    )
    print("   🔄 Profile Manager: disabled (static profile only)")
    print(f"🎲 Experiment Seed: {experiment_seed} (Base: {random_seed}, Offset: {exp_idx * 1000})")

    experiment_train_df = phase1_data.train_df.copy()
    experiment_test_df = phase1_data.test_df.copy()
    covariance_columns = phase1_data.covariance_columns
    data_processor = phase1_data.data_processor

    print(f"✅ Features: Enhanced (includes {len(covariance_columns)} covariance eigenvalues)")
    print(f"   Eigenvalues: {covariance_columns}")
    print(f"   Train shape: {experiment_train_df.shape}")
    print(f"   Test shape: {experiment_test_df.shape}")
    actuarial_enabled_cfg = bool(
        config.get("feature_params", {}).get("actuarial_params", {}).get("enabled", False)
    )
    actuarial_columns_runtime = sorted(
        [col for col in phase1_data.master_df.columns if str(col).startswith("Actuarial_")]
    )
    actuarial_non_null_master_runtime = {
        col: int(phase1_data.master_df[col].notna().sum()) for col in actuarial_columns_runtime
    }
    if actuarial_columns_runtime:
        total_non_null_runtime = int(sum(actuarial_non_null_master_runtime.values()))
        print(
            f"   🧮 Actuarial columns: {len(actuarial_columns_runtime)} detected "
            f"(enabled={actuarial_enabled_cfg}) | total non-null={total_non_null_runtime}"
        )
        print(f"      {actuarial_non_null_master_runtime}")
    elif actuarial_enabled_cfg:
        print("   ⚠️ Actuarial is enabled but no Actuarial_ columns were detected in master_df.")
    else:
        print("   ℹ️ Actuarial features disabled by config.")

    raw_turnover_curriculum = training_params.get(
        "turnover_penalty_curriculum",
        {
            0: 2.00,
            30_000: 1.75,
            60_000: 1.50,
            90_000: 1.25,
        },
    )
    turnover_curriculum = {int(threshold): float(value) for threshold, value in raw_turnover_curriculum.items()}
    sorted_turnover_values = [
        scalar for _, scalar in sorted(turnover_curriculum.items(), key=lambda item: item[0])
    ]
    if sorted_turnover_values:
        if len(sorted_turnover_values) > 1:
            turnover_scalar_display = (
                f"{sorted_turnover_values[0]:.2f} -> " + " → ".join(f"{v:.2f}" for v in sorted_turnover_values[1:])
            )
        else:
            turnover_scalar_display = f"{sorted_turnover_values[0]:.2f}"
    else:
        turnover_scalar_display = "n/a"

    env_params = config.get("environment_params", {})
    tape_terminal_scalar = float(env_params.get("tape_terminal_scalar", 10.0))
    tape_terminal_clip = float(env_params.get("tape_terminal_clip", 10.0))
    tape_terminal_bonus_mode = str(env_params.get("tape_terminal_bonus_mode", "signed")).lower().strip()
    tape_terminal_baseline = float(env_params.get("tape_terminal_baseline", 0.20))
    tape_terminal_neutral_band_enabled = bool(env_params.get("tape_terminal_neutral_band_enabled", True))
    tape_terminal_neutral_band_halfwidth = float(env_params.get("tape_terminal_neutral_band_halfwidth", 0.02))
    tape_terminal_gate_a_enabled = bool(env_params.get("tape_terminal_gate_a_enabled", False))
    tape_terminal_gate_a_sharpe_threshold = float(env_params.get("tape_terminal_gate_a_sharpe_threshold", 0.0))
    tape_terminal_gate_a_max_drawdown = float(env_params.get("tape_terminal_gate_a_max_drawdown", 0.25))
    dsr_scalar_cfg = float(env_params.get("dsr_scalar", 7.0))
    target_turnover_cfg = float(env_params.get("target_turnover", 0.60))
    turnover_band_cfg = float(env_params.get("turnover_target_band", 0.20))
    train_action_execution_beta_default = float(
        np.clip(env_params.get("action_execution_beta", 1.0), 0.0, 1.0)
    )
    raw_action_execution_beta_curriculum = training_params.get(
        "action_execution_beta_curriculum",
        {0: train_action_execution_beta_default},
    )
    action_execution_beta_curriculum: Dict[int, float] = {}
    if isinstance(raw_action_execution_beta_curriculum, dict):
        for threshold_raw, beta_raw in raw_action_execution_beta_curriculum.items():
            action_execution_beta_curriculum[int(threshold_raw)] = float(
                np.clip(beta_raw, 0.0, 1.0)
            )
    elif isinstance(raw_action_execution_beta_curriculum, list):
        for entry in raw_action_execution_beta_curriculum:
            if not isinstance(entry, dict):
                continue
            threshold = int(entry.get("threshold", 0))
            beta_value = entry.get("beta", entry.get("value"))
            if beta_value is None:
                continue
            action_execution_beta_curriculum[threshold] = float(np.clip(beta_value, 0.0, 1.0))
    if not action_execution_beta_curriculum:
        action_execution_beta_curriculum = {0: train_action_execution_beta_default}
    if 0 not in action_execution_beta_curriculum:
        action_execution_beta_curriculum[0] = train_action_execution_beta_default
    sorted_action_execution_betas = [
        beta for _, beta in sorted(action_execution_beta_curriculum.items(), key=lambda item: item[0])
    ]
    if len(sorted_action_execution_betas) > 1:
        action_execution_beta_display = (
            f"{sorted_action_execution_betas[0]:.2f} -> "
            + " → ".join(f"{v:.2f}" for v in sorted_action_execution_betas[1:])
        )
    else:
        action_execution_beta_display = f"{sorted_action_execution_betas[0]:.2f}"
    eval_action_execution_beta = float(
        np.clip(
            training_params.get("evaluation_action_execution_beta", train_action_execution_beta_default),
            0.0,
            1.0,
        )
    )
    gamma_cfg = float(config.get("agent_params", {}).get("ppo_params", {}).get("gamma", 0.99))
    train_turnover_default = float(env_params.get("turnover_penalty_scalar", 2.0))
    eval_turnover_scalar = float(training_params.get("evaluation_turnover_penalty_scalar", train_turnover_default))

    def get_current_turnover_scalar(current_timestep: int) -> float:
        for threshold, scalar in sorted(turnover_curriculum.items(), reverse=True):
            if current_timestep >= threshold:
                return scalar
        return train_turnover_default

    def get_current_action_execution_beta(current_timestep: int) -> float:
        for threshold, beta_value in sorted(action_execution_beta_curriculum.items(), reverse=True):
            if current_timestep >= threshold:
                return float(np.clip(beta_value, 0.0, 1.0))
        return train_action_execution_beta_default

    update_log_interval = int(training_params.get("update_log_interval", 1))
    alpha_diversity_log_interval = max(1, int(training_params.get("alpha_diversity_log_interval", 10)))
    alpha_diversity_warning_after_updates = int(
        training_params.get("alpha_diversity_warning_after_updates", 500)
    )
    alpha_diversity_warning_std_threshold = float(
        training_params.get("alpha_diversity_warning_std_threshold", 0.3)
    )
    print(f"\n🏗️ Creating THREE-COMPONENT TAPE v3 environments (with curriculum)...")
    print(f"   🎯 Reward System: TAPE (Three-Component v3)")
    print(f"   📊 Profile: {profile.get('name', 'BALANCED_GROWTH') if isinstance(profile, dict) else 'BALANCED_GROWTH'}")
    print(f"   ⚙️  Component 1: Base Reward (Net Return)")
    print(f"   ⚙️  Component 2: DSR/PBRS (window=60, scalar={dsr_scalar_cfg:.2f}, gamma={gamma_cfg:.2f})")
    turnover_schedule_pretty = " → ".join(
        f"{scalar:.2f}@{threshold:,}"
        for threshold, scalar in sorted(turnover_curriculum.items(), key=lambda item: item[0])
    )
    print(
        f"   ⚙️  Component 3: Turnover Proximity "
        f"(target={target_turnover_cfg:.2f}, band=±{turnover_band_cfg:.2f}, scalar={turnover_scalar_display})"
    )
    print(f"      ↳ Schedule: {turnover_schedule_pretty}")
    action_execution_schedule_pretty = " → ".join(
        f"{beta:.2f}@{threshold:,}"
        for threshold, beta in sorted(action_execution_beta_curriculum.items(), key=lambda item: item[0])
    )
    print(
        f"   ⚙️  Component 4: Execution Inertia "
        f"(beta={action_execution_beta_display}, w_exec=(1-β)w_prev + βw_raw)"
    )
    print(f"      ↳ Schedule: {action_execution_schedule_pretty}")
    print(
        "   🎁 Terminal: "
        f"mode={tape_terminal_bonus_mode}, baseline={tape_terminal_baseline:.2f}, "
        f"scalar={tape_terminal_scalar:.1f} (clipped ±{tape_terminal_clip:.1f})"
    )
    print(
        f"   🟰 Neutral Band: {'enabled' if tape_terminal_neutral_band_enabled else 'disabled'} "
        f"(±{tape_terminal_neutral_band_halfwidth:.3f} around baseline)"
    )
    print(
        f"   🚦 Gate A: {'enabled' if tape_terminal_gate_a_enabled else 'disabled'} "
        f"(Sharpe ≤ {tape_terminal_gate_a_sharpe_threshold:.2f}, "
        f"MDD ≥ {tape_terminal_gate_a_max_drawdown*100:.1f}%)"
    )
    print("   🧠 Credit Assignment: step reward is computed at each environment step")
    print("   🧾 Episode-End Handling: terminal TAPE bonus is added at episode completion only")
    print("   ✅ Retroactive episode-wide reward rescaling: disabled in notebook helper path")

    use_episode_length_curriculum = bool(training_params.get("use_episode_length_curriculum", False))

    default_schedule = [
        {"threshold": 0, "limit": 504},
        {"threshold": 15_000, "limit": 756},
        {"threshold": 30_000, "limit": 1_200},
        {"threshold": 45_000, "limit": 1_500},
        {"threshold": 60_000, "limit": 2_500},
        {"threshold": 75_000, "limit": None},  # Unlock full dataset
    ]
    curriculum_schedule = training_params.get("episode_length_curriculum_schedule", default_schedule)

    # Sort schedule by threshold to ensure deterministic behavior
    curriculum_schedule = sorted(curriculum_schedule, key=lambda item: item["threshold"])

    def determine_episode_limit(current_timestep: int, total_days: int) -> Optional[int]:
        if not use_episode_length_curriculum:
            return None
        active_limit = curriculum_schedule[0]["limit"]
        for entry in curriculum_schedule:
            if current_timestep >= entry["threshold"]:
                active_limit = entry["limit"]
            else:
                break
        if active_limit is None:
            return None
        return min(int(active_limit), total_days)

    episode_horizon_start = determine_episode_limit(0, experiment_train_df["Date"].nunique())

    drawdown_constraint_cfg = _prepare_drawdown_constraint(config, arch_upper)
    if not drawdown_constraint_cfg or not drawdown_constraint_cfg.get("enabled", False):
        raise ValueError(
            "Drawdown constraint must be enabled for Experiment 6 runs. "
            "Check PHASE1_CONFIG['environment_params']['drawdown_constraint']."
        )
    dd_target = drawdown_constraint_cfg.get("target", 0.0)
    dd_tol = drawdown_constraint_cfg.get("tolerance", 0.0)
    boundary = dd_target + dd_tol
    print(
        f"   🔒 Drawdown dual controller (requested): target={dd_target:.2%}, tolerance={dd_tol:.2%} "
        f"(trigger boundary ≈ {boundary:.2%}), lr={drawdown_constraint_cfg.get('dual_learning_rate', 0.0):.3f}, "
        f"λ_init={drawdown_constraint_cfg.get('lambda_init', 0.0):.2f}, "
        f"λ_floor={drawdown_constraint_cfg.get('lambda_floor', 0.0):.2f}, "
        f"λ_max={drawdown_constraint_cfg.get('lambda_max', 0.0):.2f}, "
        f"penalty_coef={drawdown_constraint_cfg.get('penalty_coef', drawdown_constraint_cfg.get('base_coef', 0.0)):.2f}"
    )

    env_train = PortfolioEnvTAPE(
        config=config,
        data_processor=data_processor,
        processed_data=experiment_train_df,
        mode="train",
        action_normalization="none",
        exclude_covariance=not use_covariance,
        reward_system="tape",
        tape_profile=profile,
        tape_terminal_scalar=tape_terminal_scalar,
        tape_terminal_bonus_mode=tape_terminal_bonus_mode,
        tape_terminal_baseline=tape_terminal_baseline,
        tape_terminal_neutral_band_enabled=tape_terminal_neutral_band_enabled,
        tape_terminal_neutral_band_halfwidth=tape_terminal_neutral_band_halfwidth,
        tape_terminal_gate_a_enabled=tape_terminal_gate_a_enabled,
        tape_terminal_gate_a_sharpe_threshold=tape_terminal_gate_a_sharpe_threshold,
        tape_terminal_gate_a_max_drawdown=tape_terminal_gate_a_max_drawdown,
        dsr_window=60,
        dsr_scalar=dsr_scalar_cfg,
        target_turnover=target_turnover_cfg,
        turnover_target_band=turnover_band_cfg,
        action_execution_beta=get_current_action_execution_beta(0),
        enable_base_reward=True,
        turnover_penalty_scalar=train_turnover_default,
        gamma=gamma_cfg,
        episode_length_limit=episode_horizon_start,
        tape_terminal_clip=tape_terminal_clip,
        drawdown_constraint=copy.deepcopy(drawdown_constraint_cfg),
    )

    if episode_horizon_start is not None:
        env_train.set_episode_length_limit(episode_horizon_start)
    if not getattr(env_train, "drawdown_constraint_enabled", False):
        raise RuntimeError("Drawdown controller is not enabled on env_train despite configuration.")
    print(
        "   ✅ Drawdown controller armed in env: "
        f"target={env_train.drawdown_target:.2%}, "
        f"trigger={env_train.drawdown_trigger_boundary:.2%}, "
        f"λ_init={env_train.drawdown_lambda_init:.3f}, "
        f"λ_floor={env_train.drawdown_lambda_floor:.3f}, "
        f"λ_max={env_train.drawdown_lambda_max:.2f}, "
        f"penalty_coef={env_train.drawdown_penalty_coef:.2f}"
    )

    env_test_deterministic = PortfolioEnvTAPE(
        config=config,
        data_processor=data_processor,
        processed_data=experiment_test_df,
        mode='test',
        action_normalization='none',
        exclude_covariance=not use_covariance,
        reward_system='tape',
        tape_profile=profile,
        tape_terminal_scalar=tape_terminal_scalar,
        tape_terminal_bonus_mode=tape_terminal_bonus_mode,
        tape_terminal_baseline=tape_terminal_baseline,
        tape_terminal_neutral_band_enabled=tape_terminal_neutral_band_enabled,
        tape_terminal_neutral_band_halfwidth=tape_terminal_neutral_band_halfwidth,
        tape_terminal_gate_a_enabled=tape_terminal_gate_a_enabled,
        tape_terminal_gate_a_sharpe_threshold=tape_terminal_gate_a_sharpe_threshold,
        tape_terminal_gate_a_max_drawdown=tape_terminal_gate_a_max_drawdown,
        dsr_window=60,
        dsr_scalar=dsr_scalar_cfg,
        target_turnover=target_turnover_cfg,
        turnover_target_band=turnover_band_cfg,
        action_execution_beta=eval_action_execution_beta,
        enable_base_reward=True,
        turnover_penalty_scalar=eval_turnover_scalar,
        gamma=gamma_cfg,
        random_start=False,
        episode_length_limit=None,
        tape_terminal_clip=tape_terminal_clip,
        drawdown_constraint=copy.deepcopy(drawdown_constraint_cfg),
    )

    env_test_random = PortfolioEnvTAPE(
        config=config,
        data_processor=data_processor,
        processed_data=experiment_test_df,
        mode='test',
        action_normalization='none',
        exclude_covariance=not use_covariance,
        reward_system='tape',
        tape_profile=profile,
        tape_terminal_scalar=tape_terminal_scalar,
        tape_terminal_bonus_mode=tape_terminal_bonus_mode,
        tape_terminal_baseline=tape_terminal_baseline,
        tape_terminal_neutral_band_enabled=tape_terminal_neutral_band_enabled,
        tape_terminal_neutral_band_halfwidth=tape_terminal_neutral_band_halfwidth,
        tape_terminal_gate_a_enabled=tape_terminal_gate_a_enabled,
        tape_terminal_gate_a_sharpe_threshold=tape_terminal_gate_a_sharpe_threshold,
        tape_terminal_gate_a_max_drawdown=tape_terminal_gate_a_max_drawdown,
        dsr_window=60,
        dsr_scalar=dsr_scalar_cfg,
        target_turnover=target_turnover_cfg,
        turnover_target_band=turnover_band_cfg,
        action_execution_beta=eval_action_execution_beta,
        enable_base_reward=True,
        turnover_penalty_scalar=eval_turnover_scalar,
        gamma=gamma_cfg,
        random_start=True,
        episode_length_limit=episode_horizon_start,
        tape_terminal_clip=tape_terminal_clip,
        drawdown_constraint=copy.deepcopy(drawdown_constraint_cfg),
    )

    env_test_alias = env_test_deterministic

    print(f"✅ THREE-COMPONENT TAPE v3 Environments created:")
    print(f"   Training: {env_train.total_days} days")
    print(f"   Testing: {env_test_alias.total_days} days")

    n_features = env_train.num_features
    stock_dim = env_train.num_assets
    agent_config = copy.deepcopy(config.get("agent_params", {}))
    agent_config["actor_critic_type"] = arch_upper
    agent_config["max_total_timesteps"] = max_total_timesteps
    agent_config["num_assets"] = stock_dim
    if hasattr(env_train, "get_observation_layout"):
        try:
            state_layout = env_train.get_observation_layout()
            if isinstance(state_layout, dict) and state_layout:
                agent_config["state_layout"] = copy.deepcopy(state_layout)
        except Exception:
            pass
    agent_config.setdefault("debug_prints", False)
    agent_config.setdefault("ppo_params", {})
    agent_config["ppo_params"].setdefault("gamma", gamma_cfg)

    print(f"\n🤖 Creating {arch_upper} agent with Dirichlet distribution for Exp {exp_idx}...")
    agent = agent_cls(
        state_dim=n_features,
        num_assets=stock_dim,
        config=agent_config,
    )
    print(f"✅ Agent created: {agent.__class__.__name__}")
    print(f"   🎲 Dirichlet Distribution: ENABLED")

    actor_lr_schedule_cfg = training_params.get(
        "actor_lr_schedule",
        [
            {"threshold": 0, "lr": agent_config["ppo_params"].get("actor_lr", 0.001)},
            {"threshold": 100_000, "lr": agent_config["ppo_params"].get("actor_lr", 0.001) * 0.5},
        ],
    )
    actor_lr_schedule = sorted(
        [{"threshold": int(entry.get("threshold", 0)), "lr": float(entry.get("lr", agent_config["ppo_params"].get("actor_lr", 0.001)))} for entry in actor_lr_schedule_cfg],
        key=lambda item: item["threshold"],
    )
    ppo_params_cfg = agent_config.get("ppo_params", {})
    num_ppo_epochs = int(
        training_params.get("num_ppo_epochs", ppo_params_cfg.get("num_ppo_epochs", 10))
    )
    batch_size_ppo_base = int(
        training_params.get("batch_size_ppo", ppo_params_cfg.get("batch_size_ppo", 64))
    )
    raw_batch_size_schedule_cfg = training_params.get("batch_size_ppo_schedule", [])
    batch_size_schedule: List[Dict[str, int]] = []
    if isinstance(raw_batch_size_schedule_cfg, dict):
        for threshold_raw, batch_raw in raw_batch_size_schedule_cfg.items():
            threshold = int(threshold_raw)
            batch_value = int(batch_raw)
            if batch_value <= 0:
                continue
            batch_size_schedule.append({"threshold": threshold, "batch_size": batch_value})
    elif isinstance(raw_batch_size_schedule_cfg, list):
        for entry in raw_batch_size_schedule_cfg:
            if not isinstance(entry, dict):
                continue
            threshold = int(entry.get("threshold", 0))
            batch_value = entry.get("batch_size", entry.get("batch", entry.get("value")))
            if batch_value is None:
                continue
            batch_value = int(batch_value)
            if batch_value <= 0:
                continue
            batch_size_schedule.append({"threshold": threshold, "batch_size": batch_value})

    batch_size_auto_from_rollout = False
    if not batch_size_schedule:
        if len(timestep_update_schedule) > 1:
            batch_size_auto_from_rollout = True
            for entry in timestep_update_schedule:
                rollout_len = int(entry["timesteps_per_update"])
                derived_batch = max(32, int(round(rollout_len / 4.0)))
                batch_size_schedule.append(
                    {
                        "threshold": int(entry["threshold"]),
                        "batch_size": min(rollout_len, derived_batch),
                    }
                )
        else:
            batch_size_schedule = [{"threshold": 0, "batch_size": batch_size_ppo_base}]
    else:
        batch_by_threshold: Dict[int, int] = {}
        for entry in sorted(batch_size_schedule, key=lambda item: item["threshold"]):
            batch_by_threshold[int(entry["threshold"])] = int(entry["batch_size"])
        if 0 not in batch_by_threshold:
            batch_by_threshold[0] = batch_size_ppo_base
        batch_size_schedule = [
            {"threshold": int(threshold), "batch_size": int(value)}
            for threshold, value in sorted(batch_by_threshold.items(), key=lambda item: item[0])
        ]

    def determine_batch_size_ppo(current_step: int, current_rollout_len: int) -> int:
        active_batch = batch_size_schedule[0]["batch_size"]
        for entry in batch_size_schedule:
            if current_step >= entry["threshold"]:
                active_batch = entry["batch_size"]
            else:
                break
        return max(1, min(int(active_batch), int(current_rollout_len)))

    def determine_actor_lr(current_step: int) -> float:
        updated_lr = actor_lr_schedule[0]["lr"]
        for entry in actor_lr_schedule:
            if current_step >= entry["threshold"]:
                updated_lr = entry["lr"]
            else:
                break
        return updated_lr

    current_actor_lr = determine_actor_lr(0)
    agent.set_actor_lr(current_actor_lr)
    actor_schedule_pretty = " → ".join(
        f"{entry['lr']:.6f}@{entry['threshold']:,}" for entry in actor_lr_schedule
    )
    print(f"   🔧 Actor LR schedule: {actor_schedule_pretty}")
    print(f"   State dim: {n_features}")
    print(f"   Action dim: {stock_dim}")
    print(f"   Actor LR (configured): {agent_config['ppo_params']['actor_lr']}")
    print(f"   Actor LR (active): {agent.get_actor_lr():.6f}")
    print(f"   Critic LR (active): {agent.get_critic_lr():.6f}")
    print(
        "   🧱 TCN stack: "
        f"filters={agent_config.get('tcn_filters')} | "
        f"kernel={agent_config.get('tcn_kernel_size')} | "
        f"dilations={agent_config.get('tcn_dilations')} | "
        f"dropout={agent_config.get('tcn_dropout')}"
    )
    if arch_upper == "TCN_FUSION" or bool(agent_config.get("use_fusion", False)):
        print(
            "   🧩 Fusion core: "
            f"embed={agent_config.get('fusion_embed_dim')} | "
            f"heads={agent_config.get('fusion_attention_heads')} | "
            f"dropout={agent_config.get('fusion_dropout')}"
        )
        print(
            "   🔀 Cross-Asset Mixer (A4): "
            f"enabled={bool(agent_config.get('fusion_cross_asset_mixer_enabled', False))} | "
            f"layers={agent_config.get('fusion_cross_asset_mixer_layers', 1)} | "
            f"expansion={agent_config.get('fusion_cross_asset_mixer_expansion', 2.0)} | "
            f"dropout={agent_config.get('fusion_cross_asset_mixer_dropout', agent_config.get('fusion_dropout'))}"
        )
        print(
            "   🎯 Alpha head (A3): "
            f"dims={agent_config.get('fusion_alpha_head_hidden_dims', [])} | "
            f"dropout={agent_config.get('fusion_alpha_head_dropout', agent_config.get('fusion_dropout'))}"
        )
    print(
        "   🎛️ Dirichlet controls: "
        f"activation={agent_config.get('dirichlet_alpha_activation')} | "
        f"temperature={agent_config.get('dirichlet_logit_temperature', agent_config.get('logit_temperature', 1.0))} | "
        f"alpha_cap={agent_config.get('dirichlet_alpha_cap', agent_config.get('alpha_cap', None))} | "
        f"epsilon={agent_config.get('dirichlet_epsilon')}"
    )
    initial_rollout_len = determine_timesteps_per_update(0)
    initial_batch_size = determine_batch_size_ppo(0, initial_rollout_len)
    print(
        "   PPO update: "
        f"epochs={num_ppo_epochs}, batch_size={initial_batch_size}, "
        f"target_kl={agent.target_kl:.4f}, entropy_coef={agent.entropy_coef:.4f}"
    )
    if len(timestep_update_schedule) > 1:
        rollout_schedule_pretty = " → ".join(
            f"{entry['timesteps_per_update']}@{entry['threshold']:,}"
            for entry in timestep_update_schedule
        )
        print(f"   📐 PPO rollout schedule: {rollout_schedule_pretty}")
    if len(batch_size_schedule) > 1 or batch_size_auto_from_rollout:
        batch_schedule_pretty = " → ".join(
            f"{entry['batch_size']}@{entry['threshold']:,}" for entry in batch_size_schedule
        )
        source_note = " (auto from rollout/4)" if batch_size_auto_from_rollout else ""
        print(f"   🧺 PPO batch-size schedule: {batch_schedule_pretty}{source_note}")

    # Regime-Adaptive KL (RA-KL): adjust target_kl online using observed approx_kl.
    ra_kl_enabled = bool(training_params.get("ra_kl_enabled", False))
    ra_kl_base_target = float(max(agent.target_kl, 0.0))
    if ra_kl_enabled and ra_kl_base_target <= 0.0:
        print("   ⚠️ RA-KL disabled because initial target_kl <= 0.")
        ra_kl_enabled = False
    ra_kl_target_ratio = float(training_params.get("ra_kl_target_ratio", 1.0))
    ra_kl_ema_alpha = float(np.clip(training_params.get("ra_kl_ema_alpha", 0.25), 0.01, 1.0))
    ra_kl_gain = float(max(training_params.get("ra_kl_gain", 0.06), 0.0))
    ra_kl_deadband = float(max(training_params.get("ra_kl_deadband", 0.10), 0.0))
    ra_kl_max_change_fraction = float(
        np.clip(training_params.get("ra_kl_max_change_fraction", 0.10), 0.0, 0.95)
    )
    default_ra_kl_min = max(1e-6, ra_kl_base_target * 0.5)
    default_ra_kl_max = max(default_ra_kl_min, ra_kl_base_target * 2.0)
    ra_kl_min_target_kl = float(max(training_params.get("ra_kl_min_target_kl", default_ra_kl_min), 1e-6))
    ra_kl_max_target_kl = float(
        max(training_params.get("ra_kl_max_target_kl", default_ra_kl_max), ra_kl_min_target_kl)
    )
    if ra_kl_enabled:
        print(
            "   🧭 RA-KL controller: "
            f"enabled (ratio={ra_kl_target_ratio:.2f}, ema_alpha={ra_kl_ema_alpha:.2f}, "
            f"gain={ra_kl_gain:.3f}, deadband=±{ra_kl_deadband:.2f}, "
            f"max_change={ra_kl_max_change_fraction:.2f}, "
            f"bounds=[{ra_kl_min_target_kl:.4f}, {ra_kl_max_target_kl:.4f}])"
        )

    log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = results_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    training_log_prefix = f"Exp{exp_idx}_{arch_upper}_Enhanced_TAPE_training_{log_timestamp}"
    training_episodes_path = log_dir / f"{training_log_prefix}_episodes.csv"
    training_summary_path = log_dir / f"{training_log_prefix}_summary.csv"
    training_custom_path = log_dir / f"{training_log_prefix}_custom_summary.csv"
    step_diagnostics_path = log_dir / f"{training_log_prefix}_step_diagnostics.csv"

    training_fieldnames = TRAINING_FIELDNAMES
    step_diagnostics_enabled = bool(training_params.get("log_step_diagnostics", True))

    train_csv_logger = (
        logger_cls(training_episodes_path, fieldnames=training_fieldnames) if logger_cls else None
    )
    step_diag_csv_logger = (
        logger_cls(step_diagnostics_path, fieldnames=STEP_DIAGNOSTIC_FIELDNAMES)
        if logger_cls and step_diagnostics_enabled
        else None
    )
    training_rows: List[Dict[str, Any]] = []
    print(f"📊 Training metrics will stream to {training_episodes_path}")
    if step_diagnostics_enabled:
        print(f"🧪 Step diagnostics will stream to {step_diagnostics_path}")

    def estimate_num_updates(total_steps: int) -> int:
        simulated_step = 0
        estimated_updates = 0
        while simulated_step < total_steps:
            rollout_len = determine_timesteps_per_update(simulated_step)
            step_chunk = min(rollout_len, total_steps - simulated_step)
            simulated_step += step_chunk
            estimated_updates += 1
        return estimated_updates

    num_updates = estimate_num_updates(max_total_timesteps)

    def to_scalar(value: Any) -> Optional[float]:
        if value is None:
            return None
        if hasattr(value, "numpy"):
            value = value.numpy()
        if isinstance(value, np.ndarray):
            if value.size == 1:
                return float(value.reshape(-1)[0])
            return value.tolist()
        if isinstance(value, (np.floating, float, int)):
            return float(value)
        return value

    def compute_episode_metrics(env: PortfolioEnvTAPE) -> Dict[str, float]:
        portfolio_history = np.array(env.portfolio_history)
        if len(portfolio_history) > 1:
            returns = np.diff(portfolio_history) / portfolio_history[:-1]
        else:
            returns = np.array([])
        weight_changes = []
        for idx in range(1, len(env.weights_history)):
            weight_changes.append(np.abs(env.weights_history[idx] - env.weights_history[idx - 1]))
        metrics = calculate_episode_metrics(
            portfolio_values=portfolio_history,
            returns=returns,
            weight_changes=weight_changes,
            risk_free_rate=0.02,
            trading_days_per_year=252,
        )
        metrics["return_skew"] = metrics.get("skewness", metrics.get("return_skew", 0.0))
        return metrics

    rare_params = config.get("training_params", {}).get("rare_checkpoint_params", {})
    # Evaluation-focused checkpoint policy: disable rare checkpoint saves.
    rare_enabled = False
    rare_min_sharpe = rare_params.get("min_sharpe", 1.6)
    rare_min_sortino = rare_params.get("min_sortino")
    rare_max_mdd = rare_params.get("max_mdd")
    rare_max_turnover = rare_params.get("max_turnover")
    rare_top_n = int(rare_params.get("top_n", 5))
    rare_records: List[Dict[str, Any]] = []
    saved_checkpoint_records: List[Dict[str, Any]] = []
    rare_dir = results_root / "rare_models"
    if rare_enabled:
        rare_dir.mkdir(parents=True, exist_ok=True)

    def _rare_score_tuple(sharpe, sortino, mdd):
        return (
            sharpe if sharpe is not None else -np.inf,
            sortino if sortino is not None else -np.inf,
            -(mdd if mdd is not None else 0.0),
        )

    def maybe_save_rare_checkpoint(episode_idx: int, metrics_dict: Dict[str, float]) -> None:
        if not rare_enabled:
            return
        sharpe = metrics_dict.get("sharpe_ratio")
        sortino = metrics_dict.get("sortino_ratio")
        mdd = metrics_dict.get("max_drawdown_abs")
        turnover = metrics_dict.get("turnover")
        if sharpe is None or mdd is None:
            return
        if sharpe < rare_min_sharpe:
            return
        if rare_min_sortino is not None and (sortino is None or sortino < rare_min_sortino):
            return
        if rare_max_mdd is not None and mdd > rare_max_mdd:
            return
        if rare_max_turnover is not None and turnover is not None and turnover > rare_max_turnover:
            return
        if any(record["episode"] == episode_idx for record in rare_records):
            return
        score = _rare_score_tuple(sharpe, sortino, mdd)
        if rare_top_n > 0 and len(rare_records) >= rare_top_n:
            worst = min(rare_records, key=lambda r: r["score"])
            if score <= worst["score"]:
                return
        prefix = rare_dir / f"exp{exp_idx}_tape_ep{episode_idx}_sh{sharpe:.3f}_dd{mdd*100:.1f}"
        agent.save_models(str(prefix))
        actor_path = f"{prefix}_actor.weights.h5"
        critic_path = f"{prefix}_critic.weights.h5"
        print(f"      Rare checkpoint saved: {actor_path} (Sharpe {sharpe:.3f}, MDD {mdd*100:.2f}%)")
        record = {
            "episode": episode_idx,
            "score": score,
            "sharpe": sharpe,
            "sortino": sortino,
            "mdd": mdd,
            "turnover": turnover,
            "paths": [actor_path, critic_path],
        }
        rare_records.append(record)
        saved_checkpoint_records.append(
            {
                "type": "rare",
                "episode": int(episode_idx),
                "step": int(step) if "step" in locals() else None,
                "sharpe": float(sharpe) if sharpe is not None else None,
                "actor_path": actor_path,
                "critic_path": critic_path,
            }
        )
        if rare_top_n > 0:
            rare_records.sort(key=lambda r: r["score"], reverse=True)
            while len(rare_records) > rare_top_n:
                worst = rare_records.pop()
                for path in worst["paths"]:
                    try:
                        Path(path).unlink()
                    except FileNotFoundError:
                        pass

    def maybe_save_periodic_checkpoint(current_step: int) -> None:
        nonlocal last_periodic_checkpoint_bucket
        if periodic_checkpoint_every_steps <= 0:
            return
        current_bucket = current_step // periodic_checkpoint_every_steps
        if current_bucket <= last_periodic_checkpoint_bucket:
            return
        results_root.mkdir(parents=True, exist_ok=True)
        for bucket in range(last_periodic_checkpoint_bucket + 1, current_bucket + 1):
            step_mark = bucket * periodic_checkpoint_every_steps
            prefix = results_root / f"exp{exp_idx}_tape_step{step_mark:06d}"
            agent.save_models(str(prefix))
            print(f"      💾 Periodic checkpoint saved: {prefix}_actor.weights.h5")
            saved_checkpoint_records.append(
                {
                    "type": "periodic_step",
                    "episode": None,
                    "step": int(step_mark),
                    "sharpe": None,
                    "actor_path": f"{prefix}_actor.weights.h5",
                    "critic_path": f"{prefix}_critic.weights.h5",
                }
            )
        last_periodic_checkpoint_bucket = current_bucket

    def _format_checkpoint_metric_tag(value: float) -> str:
        sign = "p" if value >= 0 else "m"
        magnitude = f"{abs(value):.3f}".replace(".", "p")
        return f"{sign}{magnitude}"

    def maybe_save_step_sharpe_checkpoint(
        current_step: int,
        episode_idx: int,
        step_info: Dict[str, Any],
    ) -> None:
        if not step_sharpe_checkpoint_enabled:
            return
        sharpe_val = to_scalar(step_info.get("sharpe_ratio"))
        if sharpe_val is None:
            return
        sharpe_float = float(sharpe_val)
        if sharpe_float < step_sharpe_checkpoint_threshold:
            return
        step_sharpe_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        sharpe_tag = _format_checkpoint_metric_tag(sharpe_float)
        prefix = step_sharpe_checkpoint_dir / (
            f"exp{exp_idx}_step{current_step:07d}_ep{episode_idx:05d}_sh{sharpe_tag}"
        )
        agent.save_models(str(prefix))
        print(
            f"      💾 Step-Sharpe checkpoint saved: {prefix}_actor.weights.h5 "
            f"(Sharpe={sharpe_float:.3f})"
        )
        saved_checkpoint_records.append(
            {
                "type": "step_sharpe",
                "episode": int(episode_idx),
                "step": int(current_step),
                "sharpe": float(sharpe_float),
                "actor_path": f"{prefix}_actor.weights.h5",
                "critic_path": f"{prefix}_critic.weights.h5",
            }
        )

    def maybe_save_high_watermark_checkpoint(
        episode_idx: int,
        metrics_dict: Dict[str, float],
    ) -> None:
        if not high_watermark_checkpoint_enabled:
            return
        sharpe_val = to_scalar(metrics_dict.get("sharpe_ratio"))
        if sharpe_val is None:
            return
        sharpe_float = float(sharpe_val)
        if sharpe_float < high_watermark_sharpe_threshold:
            return
        high_watermark_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        sharpe_tag = _format_checkpoint_metric_tag(sharpe_float)
        prefix = high_watermark_checkpoint_dir / (
            f"exp{exp_idx}_tape_hw_ep{episode_idx:05d}_sh{sharpe_tag}"
        )
        agent.save_models(str(prefix))
        print(
            "      💾 Sharpe-threshold checkpoint saved: "
            f"{prefix}_actor.weights.h5 (Sharpe={sharpe_float:.3f})"
        )
        saved_checkpoint_records.append(
            {
                "type": "high_watermark",
                "episode": int(episode_idx),
                "step": int(step) if "step" in locals() else None,
                "sharpe": float(sharpe_float),
                "actor_path": f"{prefix}_actor.weights.h5",
                "critic_path": f"{prefix}_critic.weights.h5",
            }
        )

    deterministic_validation_checkpointing_enabled_cfg = bool(
        training_params.get("deterministic_validation_checkpointing_enabled", True)
    )
    deterministic_validation_checkpointing_only_cfg = bool(
        training_params.get("deterministic_validation_checkpointing_only", True)
    )
    deterministic_validation_eval_every_episodes_cfg = max(
        1, int(training_params.get("deterministic_validation_eval_every_episodes", 5))
    )
    deterministic_validation_mode_cfg = str(
        training_params.get("deterministic_validation_mode", "mean")
    ).strip().lower()
    if deterministic_validation_mode_cfg not in {"mean", "mode", "sample", "mean_plus_noise"}:
        deterministic_validation_mode_cfg = "mean"
    deterministic_validation_episode_length_limit_cfg = training_params.get(
        "deterministic_validation_episode_length_limit", None
    )
    if deterministic_validation_episode_length_limit_cfg is not None:
        deterministic_validation_episode_length_limit_cfg = max(
            1, int(deterministic_validation_episode_length_limit_cfg)
        )
    deterministic_validation_sharpe_min_cfg = float(
        training_params.get("deterministic_validation_sharpe_min", 0.5)
    )
    deterministic_validation_sharpe_min_delta_cfg = float(
        training_params.get("deterministic_validation_sharpe_min_delta", 0.005)
    )
    deterministic_validation_seed_offset_cfg = int(
        training_params.get("deterministic_validation_seed_offset", 10_000)
    )
    deterministic_validation_log_alpha_stats_cfg = bool(
        training_params.get("deterministic_validation_log_alpha_stats", True)
    )

    # Legacy routes disabled by default; deterministic validation is now primary selector.
    high_watermark_checkpoint_enabled_cfg = bool(training_params.get("high_watermark_checkpoint_enabled", False))
    high_watermark_sharpe_threshold_cfg = float(training_params.get("high_watermark_sharpe_threshold", 0.5))
    step_sharpe_checkpoint_enabled_cfg = bool(training_params.get("step_sharpe_checkpoint_enabled", False))
    step_sharpe_checkpoint_threshold_cfg = float(training_params.get("step_sharpe_checkpoint_threshold", 0.5))
    periodic_checkpoint_every_steps_cfg = int(training_params.get("periodic_checkpoint_every_steps", 0) or 0)
    tape_checkpoint_threshold_cfg = float(training_params.get("tape_checkpoint_threshold", 999.0))
    if deterministic_validation_checkpointing_only_cfg:
        high_watermark_checkpoint_enabled_cfg = False
        step_sharpe_checkpoint_enabled_cfg = False
        periodic_checkpoint_every_steps_cfg = 0
        tape_checkpoint_threshold_cfg = 999.0

    deterministic_validation_best_sharpe = -np.inf
    deterministic_validation_best_episode: Optional[int] = None
    deterministic_validation_best_path: Optional[str] = None
    deterministic_validation_last_metrics: Dict[str, float] = {}

    def run_deterministic_validation_metrics(episode_idx: int) -> Dict[str, float]:
        """Run deterministic policy on validation env and return episode metrics."""
        env_eval = env_test_deterministic
        state_history_backup = None
        latest_sequence_backup = None
        alpha_mean_series: List[float] = []
        alpha_std_series: List[float] = []
        alpha_range_series: List[float] = []
        alpha_argmax_history: List[int] = []
        alpha_mode_vertex_steps = 0
        alpha_diag_error: Optional[str] = None
        if getattr(agent, "state_history", None) is not None:
            state_history_backup = [np.array(x, copy=True) for x in list(agent.state_history)]
        if hasattr(agent, "_latest_sequence"):
            latest_sequence_backup = (
                np.array(agent._latest_sequence, copy=True)
                if isinstance(agent._latest_sequence, np.ndarray)
                else agent._latest_sequence
            )

        prev_eval_limit = getattr(env_eval, "episode_length_limit", None)
        try:
            agent.reset_state_history()
            if deterministic_validation_episode_length_limit_cfg is not None:
                env_eval.set_episode_length_limit(deterministic_validation_episode_length_limit_cfg)
            obs_eval, _ = env_eval.reset(seed=experiment_seed + deterministic_validation_seed_offset_cfg + int(episode_idx))
            done_eval = False
            truncated_eval = False
            while not (done_eval or truncated_eval):
                action_eval, _, _ = agent.get_action_and_value(
                    obs_eval,
                    deterministic=True,
                    evaluation_mode=deterministic_validation_mode_cfg,
                )
                if deterministic_validation_log_alpha_stats_cfg:
                    try:
                        if getattr(agent, "is_sequential", False) and getattr(agent, "_latest_sequence", None) is not None:
                            alpha_state_input, _ = agent.prepare_state_input(agent._latest_sequence)
                        else:
                            alpha_state_input, _ = agent.prepare_state_input(obs_eval)
                        alpha_eval = agent.actor(alpha_state_input, training=False)
                        alpha_eval = tf.convert_to_tensor(alpha_eval, dtype=tf.float32)
                        alpha_np = np.asarray(alpha_eval.numpy()).reshape(-1)
                        if alpha_np.size > 0:
                            alpha_mean_series.append(float(np.mean(alpha_np)))
                            alpha_std_series.append(float(np.std(alpha_np)))
                            alpha_range_series.append(float(np.max(alpha_np) - np.min(alpha_np)))
                            alpha_argmax_history.append(int(np.argmax(alpha_np)))
                            if deterministic_validation_mode_cfg == "mode" and float(np.min(alpha_np)) <= 1.0:
                                alpha_mode_vertex_steps += 1
                    except Exception as exc:
                        if alpha_diag_error is None:
                            alpha_diag_error = f"{type(exc).__name__}: {exc}"
                obs_eval, _, done_eval, truncated_eval, _ = env_eval.step(action_eval)
            metrics_eval = compute_episode_metrics(env_eval)
            if alpha_mean_series:
                metrics_eval["validation_alpha_mean"] = float(np.mean(alpha_mean_series))
                metrics_eval["validation_alpha_std"] = float(np.mean(alpha_std_series))
                metrics_eval["validation_alpha_spread"] = float(np.mean(alpha_range_series))
                metrics_eval["validation_alpha_argmax_uniques"] = int(len(set(alpha_argmax_history)))
                metrics_eval["validation_alpha_mode_vertex_fraction"] = (
                    float(alpha_mode_vertex_steps) / float(max(1, len(alpha_mean_series)))
                )
            if alpha_diag_error is not None:
                metrics_eval["validation_alpha_diag_error"] = alpha_diag_error
            return metrics_eval
        finally:
            if deterministic_validation_episode_length_limit_cfg is not None:
                env_eval.set_episode_length_limit(prev_eval_limit)
            agent.reset_state_history()
            if state_history_backup is not None and getattr(agent, "state_history", None) is not None:
                agent.state_history.clear()
                for row in state_history_backup:
                    agent.state_history.append(row)
            if hasattr(agent, "_latest_sequence"):
                agent._latest_sequence = latest_sequence_backup

    def maybe_save_deterministic_validation_checkpoint(
        episode_idx: int,
    ) -> None:
        nonlocal deterministic_validation_best_sharpe
        nonlocal deterministic_validation_best_episode
        nonlocal deterministic_validation_best_path
        nonlocal deterministic_validation_last_metrics

        if not deterministic_validation_checkpointing_enabled_cfg:
            return
        if episode_idx % deterministic_validation_eval_every_episodes_cfg != 0:
            return

        val_metrics = run_deterministic_validation_metrics(episode_idx)
        deterministic_validation_last_metrics = val_metrics
        val_sharpe = float(to_scalar(val_metrics.get("sharpe_ratio", np.nan)) or np.nan)
        val_mdd = float(to_scalar(val_metrics.get("max_drawdown_abs", np.nan)) or np.nan)
        val_ret = float(to_scalar(val_metrics.get("total_return", np.nan)) or np.nan)
        val_alpha_spread = float(to_scalar(val_metrics.get("validation_alpha_spread", np.nan)) or np.nan)
        val_alpha_std = float(to_scalar(val_metrics.get("validation_alpha_std", np.nan)) or np.nan)
        val_alpha_argmax_uniques = int(to_scalar(val_metrics.get("validation_alpha_argmax_uniques", 0)) or 0)
        val_mode_vertex_fraction = float(
            to_scalar(val_metrics.get("validation_alpha_mode_vertex_fraction", np.nan)) or np.nan
        )
        print(
            "      🧪 Deterministic validation: "
            f"Sharpe={val_sharpe:.3f} | Return={val_ret*100.0:+.2f}% | DD={val_mdd*100.0:.2f}%"
        )
        if np.isfinite(val_alpha_spread):
            print(
                "         Alpha diagnostics: "
                f"spread={val_alpha_spread:.4f} | std={val_alpha_std:.4f} | "
                f"argmax_uniques={val_alpha_argmax_uniques}"
            )
        if deterministic_validation_mode_cfg == "mode" and np.isfinite(val_mode_vertex_fraction):
            if val_mode_vertex_fraction > 0.0:
                print(
                    "         ⚠️ Mode fallback detected: "
                    f"alpha<=1 on {val_mode_vertex_fraction*100.0:.1f}% of validation steps."
                )
        if not np.isfinite(val_sharpe):
            return
        if val_sharpe < deterministic_validation_sharpe_min_cfg:
            return
        if val_sharpe <= (deterministic_validation_best_sharpe + deterministic_validation_sharpe_min_delta_cfg):
            return

        high_watermark_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        sharpe_tag = _format_checkpoint_metric_tag(val_sharpe)
        prefix = high_watermark_checkpoint_dir / (
            f"exp{exp_idx}_tape_hw_ep{episode_idx:05d}_sh{sharpe_tag}"
        )
        agent.save_models(str(prefix))
        print(
            "      💾 Deterministic-validation checkpoint saved: "
            f"{prefix}_actor.weights.h5 (val_sharpe={val_sharpe:.3f})"
        )
        deterministic_validation_best_sharpe = val_sharpe
        deterministic_validation_best_episode = int(episode_idx)
        deterministic_validation_best_path = str(prefix)
        saved_checkpoint_records.append(
            {
                "type": "deterministic_validation_high_watermark",
                "episode": int(episode_idx),
                "step": int(step) if "step" in locals() else None,
                "sharpe": float(val_sharpe),
                "validation_return": float(val_ret) if np.isfinite(val_ret) else None,
                "validation_max_drawdown_abs": float(val_mdd) if np.isfinite(val_mdd) else None,
                "validation_alpha_spread": float(val_alpha_spread) if np.isfinite(val_alpha_spread) else None,
                "validation_alpha_std": float(val_alpha_std) if np.isfinite(val_alpha_std) else None,
                "validation_alpha_argmax_uniques": int(val_alpha_argmax_uniques),
                "validation_alpha_mode_vertex_fraction": (
                    float(val_mode_vertex_fraction) if np.isfinite(val_mode_vertex_fraction) else None
                ),
                "actor_path": f"{prefix}_actor.weights.h5",
                "critic_path": f"{prefix}_critic.weights.h5",
            }
        )

    print(f"\n🎯 Starting THREE-COMPONENT TAPE v3 training (with curriculum)...")
    print(f"   Total timesteps: {max_total_timesteps:,}")
    if len(timestep_update_schedule) == 1:
        print(f"   Timesteps per update: {timestep_update_schedule[0]['timesteps_per_update']}")
    else:
        print(f"   Timesteps per update: scheduled")
        for entry in timestep_update_schedule:
            print(
                f"      {entry['threshold']:,}+ steps: timesteps_per_update={entry['timesteps_per_update']}"
            )
    print(f"   Number of updates: {num_updates}")
    if len(batch_size_schedule) == 1 and not batch_size_auto_from_rollout:
        print(f"   PPO batch_size: {batch_size_schedule[0]['batch_size']}")
    else:
        auto_note = " (auto rollout/4)" if batch_size_auto_from_rollout else ""
        print(f"   PPO batch_size: scheduled{auto_note}")
        for entry in batch_size_schedule:
            print(f"      {entry['threshold']:,}+ steps: batch_size={entry['batch_size']}")
    if use_episode_length_curriculum:
        print(f"   📚 Episode Length Curriculum:")
        for entry in curriculum_schedule:
            limit = entry.get("limit")
            limit_label = "full" if limit is None else str(limit)
            print(f"      {entry.get('threshold', 0):,}+ steps: limit={limit_label}")
    else:
        print(f"   📚 Episode Length Curriculum: disabled (full horizon)")
    print(f"   📚 Turnover Scalar Curriculum:")
    for threshold, scalar in sorted(turnover_curriculum.items(), key=lambda item: item[0]):
        print(f"      {threshold:,}+ steps: scalar={scalar:.2f}")
    print(f"   🎛️ Action Execution Beta Curriculum:")
    for threshold, beta_value in sorted(action_execution_beta_curriculum.items(), key=lambda item: item[0]):
        print(f"      {threshold:,}+ steps: beta={beta_value:.2f}")
    if deterministic_validation_checkpointing_enabled_cfg:
        val_limit_label = (
            "full"
            if deterministic_validation_episode_length_limit_cfg is None
            else str(deterministic_validation_episode_length_limit_cfg)
        )
        print(
            "   🏆 Deterministic-validation checkpoints: "
            f"enabled (every {deterministic_validation_eval_every_episodes_cfg} episodes | "
            f"mode={deterministic_validation_mode_cfg} | "
            f"min_sharpe={deterministic_validation_sharpe_min_cfg:.2f} | "
            f"min_delta={deterministic_validation_sharpe_min_delta_cfg:.3f} | "
            f"alpha_diag={bool(deterministic_validation_log_alpha_stats_cfg)} | "
            f"horizon={val_limit_label})"
        )
    else:
        print("   🏆 Deterministic-validation checkpoints: disabled")
    if deterministic_validation_checkpointing_only_cfg:
        print("   🧷 Legacy checkpoint routes: disabled (high-watermark/step/periodic/tape/rare)")
    else:
        print("   🧷 Legacy checkpoint routes: configurable")
    if deterministic_validation_checkpointing_enabled_cfg:
        print("   ✅ Checkpoint selector default: deterministic validation Sharpe improvement")
    else:
        print("   ⚠️ Checkpoint selector default: legacy high-watermark path")

    metadata_path = log_dir / f"{training_log_prefix}_metadata.json"
    feature_manifest_path = log_dir / f"{training_log_prefix}_active_feature_manifest.json"
    active_feature_manifest = {
        "train_env": env_train.get_active_feature_manifest(),
        "test_env_deterministic": env_test_deterministic.get_active_feature_manifest(),
        "test_env_random": env_test_random.get_active_feature_manifest(),
    }
    with open(feature_manifest_path, "w", encoding="utf-8") as f:
        json.dump(active_feature_manifest, f, indent=2, default=str)
    print(f"🧾 Active feature manifest saved: {feature_manifest_path.resolve()}")

    feature_params = config.get("feature_params", {})
    fundamental_cfg = feature_params.get("fundamental_features", {}) if isinstance(feature_params, dict) else {}
    macro_cfg = feature_params.get("macro_data", {}) if isinstance(feature_params, dict) else {}
    alpha_cfg = feature_params.get("alpha_features", {}) if isinstance(feature_params, dict) else {}
    cross_sectional_cfg = feature_params.get("cross_sectional_features", {}) if isinstance(feature_params, dict) else {}
    actuarial_cfg = feature_params.get("actuarial_params", {}) if isinstance(feature_params, dict) else {}
    actuarial_columns = sorted([col for col in phase1_data.master_df.columns if "Actuarial_" in col])
    actuarial_non_null_counts_master = {
        col: int(phase1_data.master_df[col].notna().sum()) for col in actuarial_columns
    }
    actuarial_non_null_counts_train = {
        col: int(phase1_data.train_df[col].notna().sum()) for col in actuarial_columns if col in phase1_data.train_df.columns
    }
    actuarial_non_null_counts_test = {
        col: int(phase1_data.test_df[col].notna().sum()) for col in actuarial_columns if col in phase1_data.test_df.columns
    }
    phase1_feature_cols_for_audit: List[str] = []
    if data_processor is not None:
        try:
            phase1_feature_cols_for_audit = list(data_processor.get_feature_columns("phase1"))
        except Exception as exc:
            print(f"⚠️ Could not collect phase1 feature list for actuarial audit: {type(exc).__name__}: {exc}")
    actuarial_columns_missing_from_feature_list = (
        sorted([col for col in actuarial_columns if col not in set(phase1_feature_cols_for_audit)])
        if phase1_feature_cols_for_audit
        else []
    )
    actuarial_columns_in_feature_list = (
        sorted([col for col in actuarial_columns if col in set(phase1_feature_cols_for_audit)])
        if phase1_feature_cols_for_audit
        else []
    )
    checkpoint_strategy = {
        "normal_checkpoint_naming": "exp{exp_idx}_tape_hw_ep{episode:05d}_sh{tag}",
        "normal_checkpoint_selection": "best_deterministic_validation_sharpe",
        "rare_checkpoint_selection": "disabled",
        "legacy_final_alias_supported": True,
        "deterministic_validation_checkpointing_enabled": bool(deterministic_validation_checkpointing_enabled_cfg),
        "deterministic_validation_checkpointing_only": bool(deterministic_validation_checkpointing_only_cfg),
        "deterministic_validation_eval_every_episodes": int(deterministic_validation_eval_every_episodes_cfg),
        "deterministic_validation_mode": deterministic_validation_mode_cfg,
        "deterministic_validation_episode_length_limit": deterministic_validation_episode_length_limit_cfg,
        "deterministic_validation_sharpe_min": float(deterministic_validation_sharpe_min_cfg),
        "deterministic_validation_sharpe_min_delta": float(deterministic_validation_sharpe_min_delta_cfg),
        "deterministic_validation_seed_offset": int(deterministic_validation_seed_offset_cfg),
        "deterministic_validation_log_alpha_stats": bool(deterministic_validation_log_alpha_stats_cfg),
        "tape_checkpoint_threshold_bonus": None,
        "periodic_checkpoint_every_steps": int(periodic_checkpoint_every_steps_cfg),
        "high_watermark_checkpoint_enabled": bool(high_watermark_checkpoint_enabled_cfg),
        "high_watermark_sharpe_threshold": float(high_watermark_sharpe_threshold_cfg),
        "high_watermark_checkpoint_subdir": "high_watermark_checkpoints",
        "high_watermark_logic": "save_on_deterministic_validation_sharpe_improvement",
        "step_sharpe_checkpoint_enabled": bool(step_sharpe_checkpoint_enabled_cfg),
        "step_sharpe_checkpoint_threshold": float(step_sharpe_checkpoint_threshold_cfg),
        "step_sharpe_checkpoint_subdir": "step_sharpe_checkpoints",
        "saved_checkpoints_for_this_run": [],
    }

    effective_agent_params = _extract_effective_agent_params(
        agent_config,
        arch_upper,
        use_attention=use_attention_flag,
        use_fusion=use_fusion_flag,
    )
    template_agent_params = copy.deepcopy(agent_config)
    resolved_architecture = str(effective_agent_params.get("resolved_architecture", arch_upper)).upper()

    unused_param_prefixes: List[str] = []
    if resolved_architecture == "TCN":
        unused_param_prefixes = ["attention_", "fusion_"]
    elif resolved_architecture == "TCN_ATTENTION":
        unused_param_prefixes = ["fusion_"]
    elif resolved_architecture == "TCN_FUSION":
        unused_param_prefixes = ["attention_"]

    architecture_unused_params = sorted(
        [
            k for k in template_agent_params.keys()
            if any(k.startswith(prefix) for prefix in unused_param_prefixes)
        ]
    )
    active_tape_profile = copy.deepcopy(profile) if isinstance(profile, dict) else copy.deepcopy(PROFILE_BALANCED_GROWTH)

    metadata = {
        "Run_Context": {
            "timestamp": log_timestamp,
            "experiment_index": exp_idx,
            "experiment_name": resolved_exp_name,
            "seed": experiment_seed,
            "train_days": int(env_train.total_days),
            "test_days": int(env_test_alias.total_days),
            "train_date_min": str(experiment_train_df["Date"].min()),
            "train_date_max": str(experiment_train_df["Date"].max()),
            "test_date_min": str(experiment_test_df["Date"].min()),
            "test_date_max": str(experiment_test_df["Date"].max()),
            "active_feature_manifest_path": str(feature_manifest_path),
            "step_diagnostics_path": str(step_diagnostics_path) if step_diagnostics_enabled else None,
        },
        "Architecture_Settings": {
            "architecture": arch_upper,
            "actor_critic_type": agent_config.get("actor_critic_type", arch_upper),
            "resolved_architecture": resolved_architecture,
            "use_attention": use_attention_flag,
            "use_fusion": use_fusion_flag,
            "use_covariance": use_covariance,
            "dirichlet_alpha_activation": agent_config.get("dirichlet_alpha_activation", "softplus"),
            "dirichlet_epsilon": copy.deepcopy(agent_config.get("dirichlet_epsilon")),
            "dirichlet_exp_clip": copy.deepcopy(agent_config.get("dirichlet_exp_clip")),
            "dirichlet_logit_temperature": copy.deepcopy(
                agent_config.get("dirichlet_logit_temperature", agent_config.get("logit_temperature"))
            ),
            "dirichlet_alpha_cap": copy.deepcopy(
                agent_config.get("dirichlet_alpha_cap", agent_config.get("alpha_cap"))
            ),
            "tcn_filters": copy.deepcopy(agent_config.get("tcn_filters")),
            "tcn_kernel_size": copy.deepcopy(agent_config.get("tcn_kernel_size")),
            "tcn_dilations": copy.deepcopy(agent_config.get("tcn_dilations")),
            "tcn_dropout": copy.deepcopy(agent_config.get("tcn_dropout")),
            "fusion_embed_dim": copy.deepcopy(agent_config.get("fusion_embed_dim")),
            "fusion_attention_heads": copy.deepcopy(agent_config.get("fusion_attention_heads")),
            "fusion_dropout": copy.deepcopy(agent_config.get("fusion_dropout")),
            "fusion_cross_asset_mixer_enabled": bool(agent_config.get("fusion_cross_asset_mixer_enabled", False)),
            "fusion_cross_asset_mixer_layers": copy.deepcopy(agent_config.get("fusion_cross_asset_mixer_layers", 1)),
            "fusion_cross_asset_mixer_expansion": copy.deepcopy(agent_config.get("fusion_cross_asset_mixer_expansion", 2.0)),
            "fusion_cross_asset_mixer_dropout": copy.deepcopy(
                agent_config.get("fusion_cross_asset_mixer_dropout", agent_config.get("fusion_dropout"))
            ),
            "fusion_alpha_head_hidden_dims": copy.deepcopy(agent_config.get("fusion_alpha_head_hidden_dims", [])),
            "fusion_alpha_head_dropout": copy.deepcopy(
                agent_config.get("fusion_alpha_head_dropout", agent_config.get("fusion_dropout"))
            ),
            "tcn_hidden_activation": agent_config.get("tcn_activation", "relu"),
            "agent_params_template": template_agent_params,
            "agent_params_effective": effective_agent_params,
            "agent_params_unused_for_resolved_architecture": architecture_unused_params,
        },
        "Feature_Groups": {
            "fundamental_features": copy.deepcopy(fundamental_cfg),
            "macro_data": copy.deepcopy(macro_cfg),
            "alpha_features": copy.deepcopy(alpha_cfg),
            "cross_sectional_features": copy.deepcopy(cross_sectional_cfg),
            "actuarial_features": copy.deepcopy(actuarial_cfg),
            "actuarial_columns_detected": actuarial_columns,
            "actuarial_columns_non_null_counts_master": actuarial_non_null_counts_master,
            "actuarial_columns_non_null_counts_train": actuarial_non_null_counts_train,
            "actuarial_columns_non_null_counts_test": actuarial_non_null_counts_test,
            "actuarial_columns_in_feature_list": actuarial_columns_in_feature_list,
            "actuarial_columns_missing_from_feature_list": actuarial_columns_missing_from_feature_list,
        },
        "Training_Hyperparameters": {
            "max_total_timesteps": max_total_timesteps,
            "timesteps_per_ppo_update": timestep_update_schedule[0]["timesteps_per_update"],
            "timesteps_per_ppo_update_schedule": timestep_update_schedule,
            "actor_lr_schedule": actor_lr_schedule,
            "num_ppo_epochs": num_ppo_epochs,
            "batch_size_ppo": batch_size_schedule[0]["batch_size"],
            "batch_size_ppo_schedule": batch_size_schedule,
            "batch_size_ppo_auto_from_rollout_schedule": batch_size_auto_from_rollout,
            "use_episode_length_curriculum": use_episode_length_curriculum,
            "episode_length_curriculum_schedule": curriculum_schedule,
            "episode_length_limit_initial": episode_horizon_start,
            "turnover_penalty_curriculum": turnover_curriculum,
            "action_execution_beta_curriculum": action_execution_beta_curriculum,
            "evaluation_action_execution_beta": eval_action_execution_beta,
            "evaluation_turnover_penalty_scalar": eval_turnover_scalar,
            "log_step_diagnostics": bool(step_diagnostics_enabled),
            "gamma": gamma_cfg,
            "ra_kl_enabled": bool(ra_kl_enabled),
            "ra_kl_target_ratio": ra_kl_target_ratio,
            "ra_kl_ema_alpha": ra_kl_ema_alpha,
            "ra_kl_gain": ra_kl_gain,
            "ra_kl_deadband": ra_kl_deadband,
            "ra_kl_max_change_fraction": ra_kl_max_change_fraction,
            "ra_kl_min_target_kl": ra_kl_min_target_kl,
            "ra_kl_max_target_kl": ra_kl_max_target_kl,
            "ra_kl_initial_target_kl": ra_kl_base_target,
        },
        "Reward_and_Environment": {
            "dsr_scalar": dsr_scalar_cfg,
            "target_turnover": target_turnover_cfg,
            "turnover_target_band": turnover_band_cfg,
            "action_execution_beta": train_action_execution_beta_default,
            "tape_terminal_scalar": tape_terminal_scalar,
            "tape_terminal_clip": tape_terminal_clip,
            "tape_terminal_bonus_mode": tape_terminal_bonus_mode,
            "tape_terminal_baseline": tape_terminal_baseline,
            "tape_terminal_neutral_band_enabled": tape_terminal_neutral_band_enabled,
            "tape_terminal_neutral_band_halfwidth": tape_terminal_neutral_band_halfwidth,
            "tape_terminal_gate_a_enabled": tape_terminal_gate_a_enabled,
            "tape_terminal_gate_a_sharpe_threshold": tape_terminal_gate_a_sharpe_threshold,
            "tape_terminal_gate_a_max_drawdown": tape_terminal_gate_a_max_drawdown,
            "drawdown_constraint": drawdown_constraint_cfg,
            "tape_profile_name": active_tape_profile.get("name", "BalancedGrowth"),
            "tape_profile_full": _json_ready(active_tape_profile),
            "reward_credit_assignment_mode": "step_reward_plus_terminal_bonus",
            "retroactive_episode_reward_scaling": False,
            "training_entrypoint": "src/notebook_helpers/tcn_phase1.py::run_experiment6_tape",
        },
        "Checkpointing": checkpoint_strategy,
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"🧾 Training metadata saved: {metadata_path.resolve()}")


    train_start = time.time()
    obs, info = env_train.reset(seed=experiment_seed)
    training_episode_count = 0
    step = 0
    done = False
    save_tape_bonus_checkpoints = not deterministic_validation_checkpointing_only_cfg
    tape_threshold = float(tape_checkpoint_threshold_cfg)
    periodic_checkpoint_every_steps = int(periodic_checkpoint_every_steps_cfg)
    high_watermark_checkpoint_enabled = bool(high_watermark_checkpoint_enabled_cfg)
    high_watermark_sharpe_threshold = float(high_watermark_sharpe_threshold_cfg)
    high_watermark_checkpoint_dir = results_root / "high_watermark_checkpoints"
    step_sharpe_checkpoint_enabled = bool(step_sharpe_checkpoint_enabled_cfg)
    step_sharpe_checkpoint_threshold = float(step_sharpe_checkpoint_threshold_cfg)
    step_sharpe_checkpoint_dir = results_root / "step_sharpe_checkpoints"
    last_periodic_checkpoint_bucket = 0

    last_tape_bonus_clipped = False
    current_turnover_scalar = get_current_turnover_scalar(0)
    current_action_execution_beta = get_current_action_execution_beta(0)

    episode_terminal_info = None
    metrics_for_update = None
    last_episode_metrics = {
        "episode_return_pct": 0.0,
        "episode_sharpe": 0.0,
        "episode_sortino": 0.0,
        "episode_max_dd": 0.0,
        "episode_volatility": 0.0,
        "episode_win_rate": 0.0,
        "episode_turnover": 0.0,
        "episode_turnover_pct": 0.0,
        "episode_return_skew": 0.0,
    }
    last_profile_name = env_train.tape_profile.get("name", "N/A")
    last_drawdown_lambda = None
    last_drawdown_lambda_peak = None
    last_tape_score = None
    last_tape_bonus = None
    last_tape_bonus_raw = None
    last_tape_terminal_bonus_mode = None
    last_tape_terminal_baseline = None
    last_tape_terminal_neutral_band_applied = False
    last_tape_terminal_neutral_band_halfwidth = None
    last_tape_gate_a_triggered = False
    last_tape_gate_a_sharpe = None
    last_tape_gate_a_max_drawdown_abs = None
    last_drawdown_avg_excess = None
    last_drawdown_penalty_sum = None
    last_terminal_intra_step_tape_potential = None
    last_terminal_intra_step_tape_delta_reward = None
    last_initial_balance = env_train.initial_balance
    last_final_balance = env_train.initial_balance
    last_next_profile_name = None
    last_next_profile_reason = None
    last_episode_length_info = None
    last_termination_reason = None
    policy_entropy_value = 0.0
    policy_loss_value = 0.0
    entropy_loss_value = 0.0
    approx_kl_value = 0.0
    clip_fraction_value = 0.0
    value_clip_fraction_value = 0.0
    explained_variance_value = 0.0
    actor_grad_norm_value = 0.0
    critic_grad_norm_value = 0.0
    alpha_min_value = 0.0
    alpha_max_value = 0.0
    alpha_mean_value = 0.0
    ratio_mean_value = 0.0
    ratio_std_value = 0.0

    current_episode_limit = episode_horizon_start if episode_horizon_start is not None else env_train.total_days
    current_timestep_rollout = determine_timesteps_per_update(0)
    current_batch_size_ppo = determine_batch_size_ppo(0, current_timestep_rollout)
    update_count = 0
    ra_kl_ema_approx_kl: Optional[float] = None
    ra_kl_last_error_ratio = 0.0
    ra_kl_last_adjust_factor = 1.0

    while step < max_total_timesteps:
        update_count += 1
        episode_terminal_info = None

        active_timestep_rollout = determine_timesteps_per_update(step)
        active_batch_size_ppo = determine_batch_size_ppo(step, active_timestep_rollout)

        if active_timestep_rollout != current_timestep_rollout:
            current_timestep_rollout = active_timestep_rollout
            print(f"\n📚 PPO ROLLOUT UPDATE at {step:,} steps:")
            print(f"   Timesteps per update: {current_timestep_rollout}")
        if active_batch_size_ppo != current_batch_size_ppo:
            current_batch_size_ppo = active_batch_size_ppo
            print(f"\n📚 PPO BATCH SIZE UPDATE at {step:,} steps:")
            print(f"   Batch size: {current_batch_size_ppo}")

        steps_this_update = min(active_timestep_rollout, max_total_timesteps - step)
        for _ in range(steps_this_update):
            action, log_prob, value = agent.get_action_and_value(obs, deterministic=False)
            prev_portfolio_value = float(getattr(env_train, "portfolio_value", np.nan))
            next_obs, reward, done, truncated, info = env_train.step(action)

            agent.store_transition(obs, action, log_prob, reward, value, done)
            obs = next_obs
            step += 1

            new_actor_lr = determine_actor_lr(step)
            if not np.isclose(new_actor_lr, current_actor_lr):
                current_actor_lr = new_actor_lr
                agent.set_actor_lr(current_actor_lr)
                print(f"   🔧 Actor learning rate adjusted to {current_actor_lr:.6f} at step {step:,}")

            if step_diag_csv_logger is not None:
                turnover_val = float(info.get("turnover", 0.0) or 0.0)
                turnover_target = float(getattr(env_train, "target_turnover_per_step", 0.0) or 0.0)
                turnover_scalar_live = float(getattr(env_train, "turnover_penalty_scalar", 0.0) or 0.0)
                if turnover_target > 0.0 and turnover_val > turnover_target:
                    excess_ratio = (turnover_val - turnover_target) / max(turnover_target, 1e-8)
                    turnover_penalty_contrib = -excess_ratio * turnover_scalar_live
                else:
                    turnover_penalty_contrib = 0.0

                tx_cost_dollars = float(info.get("transaction_costs", 0.0) or 0.0)
                tx_cost_denom = max(prev_portfolio_value, 1e-8)
                tx_cost_contrib_reward_pts = -(tx_cost_dollars / tx_cost_denom) * 100.0

                step_diag_row = {
                    "update": update_count,
                    "timestep": step,
                    "episode": training_episode_count,
                    "episode_step": int(info.get("episode_step", getattr(env_train, "episode_step_count", 0)) or 0),
                    "date": info.get("date"),
                    "elapsed_time": time.time() - train_start,
                    "reward_total": float(reward),
                    "portfolio_return_pct_points": float(info.get("portfolio_return", 0.0) or 0.0) * 100.0,
                    "portfolio_value": float(info.get("portfolio_value", getattr(env_train, "portfolio_value", np.nan))),
                    "prev_portfolio_value": float(prev_portfolio_value),
                    "l1_w_delta": turnover_val,
                    "turnover": turnover_val,
                    "raw_turnover": float(info.get("raw_turnover", turnover_val) or turnover_val),
                    "executed_turnover": float(info.get("executed_turnover", turnover_val) or turnover_val),
                    "action_execution_beta": float(
                        info.get("action_execution_beta", getattr(env_train, "action_execution_beta", np.nan))
                    ),
                    "execution_smoothing_l1": float(info.get("execution_smoothing_l1", 0.0) or 0.0),
                    "turnover_target": turnover_target,
                    "turnover_scalar": turnover_scalar_live,
                    "turnover_penalty_contrib": float(turnover_penalty_contrib),
                    "transaction_cost_dollars": tx_cost_dollars,
                    "tx_cost_contrib_reward_pts": float(tx_cost_contrib_reward_pts),
                    "action_realization_l1": float(info.get("action_realization_l1", 0.0) or 0.0),
                    "action_realization_penalty": float(info.get("action_realization_penalty", 0.0) or 0.0),
                    "drawdown_penalty": float(info.get("drawdown_penalty", 0.0) or 0.0),
                }
                step_diag_csv_logger.log(step_diag_row)

            if done or truncated:
                training_episode_count += 1
                episode_terminal_info = info.copy()

                profile_name = info.get("profile_name") or env_train.tape_profile.get("name", "N/A")
                metrics_current = compute_episode_metrics(env_train)
                metrics_for_update = metrics_current
                turnover_raw, turnover_pct = _extract_turnover_metrics(metrics_current)
                last_episode_metrics = {
                    "episode_return_pct": metrics_current.get("total_return", 0.0) * 100,
                    "episode_sharpe": metrics_current.get("sharpe_ratio", 0.0),
                    "episode_sortino": metrics_current.get("sortino_ratio", 0.0),
                    "episode_max_dd": metrics_current.get("max_drawdown_abs", 0.0) * 100,
                    "episode_volatility": metrics_current.get("volatility", 0.0),
                    "episode_win_rate": metrics_current.get("win_rate", 0.0) * 100,
                    "episode_turnover": turnover_raw,
                    "episode_turnover_pct": turnover_pct,
                    "episode_return_skew": metrics_current.get("return_skew", 0.0),
                }
                last_profile_name = profile_name

                last_drawdown_lambda = to_scalar(info.get("drawdown_lambda", last_drawdown_lambda))
                last_drawdown_lambda_peak = to_scalar(info.get("drawdown_lambda_peak", last_drawdown_lambda_peak))
                last_tape_score = to_scalar(info.get("tape_score", last_tape_score))
                last_tape_bonus = to_scalar(info.get("tape_bonus", last_tape_bonus))
                last_tape_bonus_raw = to_scalar(info.get("tape_bonus_raw", last_tape_bonus_raw))
                last_tape_terminal_bonus_mode = info.get("tape_terminal_bonus_mode", last_tape_terminal_bonus_mode)
                last_tape_terminal_baseline = to_scalar(
                    info.get("tape_terminal_baseline", last_tape_terminal_baseline)
                )
                last_tape_terminal_neutral_band_applied = bool(
                    info.get("tape_terminal_neutral_band_applied", last_tape_terminal_neutral_band_applied)
                )
                last_tape_terminal_neutral_band_halfwidth = to_scalar(
                    info.get("tape_terminal_neutral_band_halfwidth", last_tape_terminal_neutral_band_halfwidth)
                )
                last_tape_gate_a_triggered = bool(info.get("tape_gate_a_triggered", last_tape_gate_a_triggered))
                last_tape_gate_a_sharpe = to_scalar(info.get("tape_gate_a_sharpe", last_tape_gate_a_sharpe))
                last_tape_gate_a_max_drawdown_abs = to_scalar(
                    info.get("tape_gate_a_max_drawdown_abs", last_tape_gate_a_max_drawdown_abs)
                )
                last_terminal_intra_step_tape_potential = to_scalar(
                    info.get("intra_step_tape_potential", last_terminal_intra_step_tape_potential)
                )
                last_terminal_intra_step_tape_delta_reward = to_scalar(
                    info.get("intra_step_tape_delta_reward", last_terminal_intra_step_tape_delta_reward)
                )
                last_drawdown_avg_excess = to_scalar(info.get("drawdown_avg_excess", last_drawdown_avg_excess))
                last_drawdown_penalty_sum = to_scalar(info.get("drawdown_penalty_sum", last_drawdown_penalty_sum))
                last_initial_balance = to_scalar(info.get("initial_balance", last_initial_balance))
                last_final_balance = to_scalar(info.get("final_balance", last_final_balance))
                last_next_profile_name = info.get("next_profile_name", last_next_profile_name)
                last_next_profile_reason = info.get("next_profile_reason", last_next_profile_reason)
                last_episode_length_info = to_scalar(info.get("episode_length", last_episode_length_info))
                last_termination_reason = info.get("termination_reason", last_termination_reason)

                tape_score = info.get("tape_score")
                if tape_score is None:
                    print(
                        f"   ⚠️ Episode {training_episode_count}: tape_score is None; "
                        "terminal TAPE bonus metadata missing."
                    )
                if tape_score is not None:
                    env_bonus_raw = info.get("tape_bonus_raw")
                    env_bonus = info.get("tape_bonus")
                    if env_bonus_raw is not None:
                        tape_bonus_raw = float(env_bonus_raw)
                    else:
                        tape_score_f = float(tape_score)
                        if tape_terminal_bonus_mode == "centered":
                            tape_bonus_raw = float((tape_score_f - tape_terminal_baseline) * tape_terminal_scalar)
                        elif tape_terminal_bonus_mode == "signed":
                            if tape_score_f >= tape_terminal_baseline:
                                denom = max(1e-9, 1.0 - tape_terminal_baseline)
                                signed_tape = (tape_score_f - tape_terminal_baseline) / denom
                            else:
                                denom = max(1e-9, tape_terminal_baseline)
                                signed_tape = -((tape_terminal_baseline - tape_score_f) / denom)
                            tape_bonus_raw = float(np.clip(signed_tape, -1.0, 1.0) * tape_terminal_scalar)
                        else:
                            tape_bonus_raw = float(tape_score_f * tape_terminal_scalar)
                        if tape_terminal_neutral_band_enabled and abs(tape_score_f - tape_terminal_baseline) <= tape_terminal_neutral_band_halfwidth:
                            tape_bonus_raw = 0.0
                    tape_bonus_clipped = (
                        float(env_bonus)
                        if env_bonus is not None
                        else float(np.clip(tape_bonus_raw, -tape_terminal_clip, tape_terminal_clip))
                    )
                    did_clip = abs(tape_bonus_raw - tape_bonus_clipped) > 1e-12
                    neutral_band_applied = bool(info.get("tape_terminal_neutral_band_applied", False))
                    gate_triggered = bool(info.get("tape_gate_a_triggered", False))
                    print(
                        f"   🎯 Episode {training_episode_count}: TAPE Score = {tape_score:.4f} "
                        f"(bonus: {tape_bonus_raw:+.2f} → {tape_bonus_clipped:+.2f})"
                    )
                    if neutral_band_applied:
                        print(
                            "      🟰 Neutral band applied "
                            f"(±{float(info.get('tape_terminal_neutral_band_halfwidth', tape_terminal_neutral_band_halfwidth)):.3f})"
                        )
                    if gate_triggered:
                        gate_sharpe = info.get("tape_gate_a_sharpe")
                        gate_mdd = info.get("tape_gate_a_max_drawdown_abs")
                        if gate_sharpe is not None and gate_mdd is not None:
                            print(
                                "      🚦 Gate A applied: "
                                f"Sharpe={float(gate_sharpe):.3f}, MDD={float(gate_mdd)*100:.2f}%"
                            )
                        else:
                            print("      🚦 Gate A applied: forcing non-positive terminal bonus.")

                    if save_tape_bonus_checkpoints:
                        def save_tape_checkpoint(suffix: str, reason: str) -> None:
                            results_root.mkdir(parents=True, exist_ok=True)
                            prefix_path = results_root / f"exp{exp_idx}_{suffix}"
                            agent.save_models(str(prefix_path))
                            print(f"      💾 {reason} saved: {prefix_path}_actor.weights.h5")

                        if tape_bonus_clipped >= tape_threshold:
                            save_tape_checkpoint(
                                f"tape_ep{training_episode_count}",
                                "TAPE threshold checkpoint",
                            )

                        if did_clip and not last_tape_bonus_clipped:
                            save_tape_checkpoint(
                                f"tape_clip_ep{training_episode_count}",
                                "TAPE clip checkpoint",
                            )
                            last_tape_bonus_clipped = True
                        elif not did_clip:
                            last_tape_bonus_clipped = False

                if deterministic_validation_checkpointing_enabled_cfg:
                    maybe_save_deterministic_validation_checkpoint(training_episode_count)
                else:
                    maybe_save_high_watermark_checkpoint(training_episode_count, metrics_current)
                obs, info = env_train.reset()
                done = False

        new_turnover_scalar = get_current_turnover_scalar(step)
        if new_turnover_scalar != current_turnover_scalar:
            current_turnover_scalar = new_turnover_scalar
            env_train.turnover_penalty_scalar = current_turnover_scalar
            print(f"\n📚 TURNOVER CURRICULUM UPDATE at {step:,} steps:")
            print(f"   Turnover penalty scalar: {current_turnover_scalar}")

        new_action_execution_beta = get_current_action_execution_beta(step)
        if not np.isclose(new_action_execution_beta, current_action_execution_beta):
            current_action_execution_beta = new_action_execution_beta
            env_train.set_action_execution_beta(current_action_execution_beta)
            print(f"\n🎛️ EXECUTION BETA UPDATE at {step:,} steps:")
            print(
                "   action_execution_beta: "
                f"{current_action_execution_beta:.3f} "
                "(w_exec=(1-β)w_prev + βw_raw)"
            )

        if use_episode_length_curriculum:
            new_episode_limit = determine_episode_limit(step, env_train.total_days)
            if new_episode_limit != current_episode_limit:
                current_episode_limit = new_episode_limit if new_episode_limit is not None else env_train.total_days
                env_train.set_episode_length_limit(new_episode_limit)
                print(f"\n📚 EPISODE HORIZON UPDATE at {step:,} steps:")
                if new_episode_limit is None:
                    print("   Episode horizon set to full dataset")
                else:
                    print(f"   Episode horizon: {new_episode_limit} steps")

        _, _, next_value = agent.get_action_and_value(obs, deterministic=False)
        advantages, returns = agent.compute_gae(
            agent.memory["rewards"],
            agent.memory["values"],
            agent.memory["dones"],
            next_value,
        )
        agent.memory["advantages"] = advantages
        agent.memory["returns"] = returns

        update_metrics = agent.update(num_epochs=num_ppo_epochs, batch_size=current_batch_size_ppo)
        agent.clear_memory()

        actor_loss_value = update_metrics.get("actor_loss", 0.0)
        critic_loss_value = update_metrics.get("critic_loss", 0.0)
        critic_loss_scaled_value = update_metrics.get("critic_loss_scaled", critic_loss_value)
        risk_aux_total_value = update_metrics.get("risk_aux_total", 0.0)
        risk_aux_sharpe_proxy_value = update_metrics.get("risk_aux_sharpe_proxy", 0.0)
        risk_aux_sharpe_loss_value = update_metrics.get("risk_aux_sharpe_loss", 0.0)
        risk_aux_mvo_loss_value = update_metrics.get("risk_aux_mvo_loss", 0.0)
        policy_entropy_value = update_metrics.get("entropy", 0.0)
        policy_loss_value = update_metrics.get("policy_loss", 0.0)
        entropy_loss_value = update_metrics.get("entropy_loss", 0.0)
        approx_kl_value = update_metrics.get("approx_kl", 0.0)
        clip_fraction_value = update_metrics.get("clip_fraction", 0.0)
        value_clip_fraction_value = update_metrics.get("value_clip_fraction", 0.0)
        explained_variance_value = update_metrics.get("explained_variance", 0.0)
        actor_grad_norm_value = update_metrics.get("actor_grad_norm", 0.0)
        critic_grad_norm_value = update_metrics.get("critic_grad_norm", 0.0)
        alpha_min_value = update_metrics.get("alpha_min", 0.0)
        alpha_max_value = update_metrics.get("alpha_max", 0.0)
        alpha_mean_value = update_metrics.get("alpha_mean", 0.0)
        alpha_std_value = update_metrics.get("alpha_std", 0.0)  # Track alpha diversity
        ratio_mean_value = update_metrics.get("ratio_mean", 0.0)
        ratio_std_value = update_metrics.get("ratio_std", 0.0)

        if ra_kl_enabled:
            approx_kl_scalar = to_scalar(approx_kl_value)
            if approx_kl_scalar is not None and np.isfinite(approx_kl_scalar):
                approx_kl_obs = float(max(approx_kl_scalar, 0.0))
                if ra_kl_ema_approx_kl is None:
                    ra_kl_ema_approx_kl = approx_kl_obs
                else:
                    ra_kl_ema_approx_kl = (
                        (1.0 - ra_kl_ema_alpha) * ra_kl_ema_approx_kl
                        + ra_kl_ema_alpha * approx_kl_obs
                    )

                current_target_kl = float(max(agent.target_kl, 1e-8))
                kl_ratio = float(ra_kl_ema_approx_kl / current_target_kl)
                ra_kl_last_error_ratio = float(kl_ratio - ra_kl_target_ratio)
                ra_kl_last_adjust_factor = 1.0

                if abs(ra_kl_last_error_ratio) > ra_kl_deadband:
                    raw_factor = float(np.exp(-ra_kl_gain * ra_kl_last_error_ratio))
                    low = 1.0 - ra_kl_max_change_fraction
                    high = 1.0 + ra_kl_max_change_fraction
                    ra_kl_last_adjust_factor = float(np.clip(raw_factor, low, high))

                new_target_kl = float(
                    np.clip(
                        current_target_kl * ra_kl_last_adjust_factor,
                        ra_kl_min_target_kl,
                        ra_kl_max_target_kl,
                    )
                )
                if not np.isclose(new_target_kl, current_target_kl):
                    agent.target_kl = new_target_kl

        if np.isnan(actor_loss_value) or np.isinf(actor_loss_value):
            print(f"\n❌ CRITICAL ERROR: NaN/Inf detected in actor_loss at update {update_count}!")
            print(f"   Actor Loss: {actor_loss_value}")
            print(f"   Critic Loss: {critic_loss_value}")
            print(f"   🛑 Stopping training early to prevent cascade failure.")
            break

        if (update_count % update_log_interval == 0) or (step >= max_total_timesteps):
            elapsed = time.time() - train_start

            snapshot_metrics = (
                metrics_for_update if metrics_for_update is not None else compute_episode_metrics(env_train)
            )
            if snapshot_metrics:
                episode_return_pct_val = snapshot_metrics.get("total_return", 0.0) * 100.0
                episode_sharpe_val = snapshot_metrics.get("sharpe_ratio", 0.0)
                episode_sortino_val = snapshot_metrics.get("sortino_ratio", 0.0)
                episode_max_dd_val = snapshot_metrics.get("max_drawdown_abs", 0.0) * 100.0
                episode_volatility_val = snapshot_metrics.get("volatility", 0.0)
                episode_win_rate_val = snapshot_metrics.get("win_rate", 0.0) * 100.0
                episode_turnover_raw_val, episode_turnover_pct_val = _extract_turnover_metrics(snapshot_metrics)
                episode_return_skew_val = snapshot_metrics.get("return_skew", snapshot_metrics.get("skewness", 0.0))
                episode_calmar_val = snapshot_metrics.get("calmar_ratio", 0.0)
                episode_omega_val = snapshot_metrics.get("omega_ratio", 0.0)
                episode_ulcer_val = snapshot_metrics.get("ulcer_index", 0.0)
                episode_cvar_val = snapshot_metrics.get("cvar_5pct", 0.0)
            else:
                episode_return_pct_val = last_episode_metrics["episode_return_pct"]
                episode_sharpe_val = last_episode_metrics["episode_sharpe"]
                episode_sortino_val = last_episode_metrics["episode_sortino"]
                episode_max_dd_val = last_episode_metrics["episode_max_dd"]
                episode_volatility_val = last_episode_metrics["episode_volatility"]
                episode_win_rate_val = last_episode_metrics["episode_win_rate"]
                episode_turnover_raw_val = last_episode_metrics["episode_turnover"]
                episode_turnover_pct_val = last_episode_metrics["episode_turnover_pct"]
                episode_return_skew_val = last_episode_metrics["episode_return_skew"]
                episode_calmar_val = 0.0
                episode_omega_val = 0.0
                episode_ulcer_val = 0.0
                episode_cvar_val = 0.0
            metrics_for_update = None

            actor_loss_val = to_scalar(actor_loss_value)
            critic_loss_val = to_scalar(critic_loss_value)
            critic_loss_scaled_val = to_scalar(critic_loss_scaled_value)
            risk_aux_total_val = to_scalar(risk_aux_total_value)
            risk_aux_sharpe_proxy_val = to_scalar(risk_aux_sharpe_proxy_value)
            risk_aux_sharpe_loss_val = to_scalar(risk_aux_sharpe_loss_value)
            risk_aux_mvo_loss_val = to_scalar(risk_aux_mvo_loss_value)
            mean_advantage_val = to_scalar(update_metrics.get("mean_advantage", 0.0))
            policy_entropy_val = to_scalar(policy_entropy_value)
            policy_loss_val = to_scalar(policy_loss_value)
            entropy_loss_val = to_scalar(entropy_loss_value)
            approx_kl_val = to_scalar(approx_kl_value)
            clip_fraction_val = to_scalar(clip_fraction_value)
            value_clip_fraction_val = to_scalar(value_clip_fraction_value)
            explained_variance_val = to_scalar(explained_variance_value)
            actor_grad_norm_val = to_scalar(actor_grad_norm_value)
            critic_grad_norm_val = to_scalar(critic_grad_norm_value)
            alpha_min_val = to_scalar(alpha_min_value)
            alpha_max_val = to_scalar(alpha_max_value)
            alpha_mean_val = to_scalar(alpha_mean_value)
            alpha_std_val = to_scalar(alpha_std_value)  # For alpha diversity tracking
            ratio_mean_val = to_scalar(ratio_mean_value)
            ratio_std_val = to_scalar(ratio_std_value)

            # Capture live (snapshot) drawdown controller state for this update log row.
            snapshot_drawdown_lambda = to_scalar(getattr(env_train, "drawdown_lambda", None))
            snapshot_drawdown_lambda_peak = to_scalar(getattr(env_train, "drawdown_lambda_peak", None))
            snapshot_drawdown_current = to_scalar(getattr(env_train, "current_drawdown", None))
            snapshot_drawdown_avg_excess = to_scalar(
                getattr(env_train, "drawdown_excess_accumulator", 0.0) / max(1, getattr(env_train, "episode_step_count", 1))
            )
            snapshot_drawdown_penalty_sum = to_scalar(getattr(env_train, "drawdown_penalty_sum", None))
            snapshot_drawdown_triggered = bool(getattr(env_train, "drawdown_triggered", False))
            snapshot_drawdown_trigger_boundary = to_scalar(getattr(env_train, "drawdown_trigger_boundary", None))
            snapshot_drawdown_target = to_scalar(getattr(env_train, "drawdown_target", None))
            snapshot_drawdown_tolerance = to_scalar(getattr(env_train, "drawdown_tolerance", None))
            snapshot_intra_step_tape_potential = to_scalar(
                getattr(env_train, "last_intra_step_tape_potential", None)
            )
            snapshot_intra_step_tape_delta_reward = to_scalar(
                getattr(env_train, "last_intra_step_tape_delta_reward", None)
            )

            terminal_drawdown_lambda = last_drawdown_lambda
            terminal_drawdown_lambda_peak = last_drawdown_lambda_peak
            terminal_drawdown_avg_excess = last_drawdown_avg_excess
            terminal_drawdown_penalty_sum = last_drawdown_penalty_sum
            terminal_intra_step_tape_potential = last_terminal_intra_step_tape_potential
            terminal_intra_step_tape_delta_reward = last_terminal_intra_step_tape_delta_reward

            print(
                f"🔄 Update {update_count}/{num_updates} | Step {step:,}/{max_total_timesteps:,} | "
                f"Episode {training_episode_count} | Time: {elapsed:.1f}s"
            )
            print(
                f"   📊 Metrics: Return={episode_return_pct_val:+.2f}% | "
                f"Sharpe={episode_sharpe_val:.3f} | DD={episode_max_dd_val:.2f}% | "
                f"Turnover={episode_turnover_pct_val:.2f}%"
            )
            if snapshot_intra_step_tape_potential is not None:
                print(
                    "   🎚️ Intra-Step TAPE: "
                    f"potential={snapshot_intra_step_tape_potential:.4f} | "
                    f"delta_reward={snapshot_intra_step_tape_delta_reward:+.4f}"
                )
            print(f"   🎯 Profile: {last_profile_name}")
            print(
                f"   🧠 Training: actor_loss={actor_loss_val:.4f} | "
                f"critic_loss={critic_loss_val:.4f} | mean_adv={mean_advantage_val:.4f}"
            )
            print(
                f"   🧮 Loss Detail: critic_scaled={critic_loss_scaled_val:.4f} | "
                f"risk_aux_total={risk_aux_total_val:.4f} | "
                f"sharpe_proxy={risk_aux_sharpe_proxy_val:.4f} | "
                f"sharpe_loss={risk_aux_sharpe_loss_val:.4f} | "
                f"mvo_loss={risk_aux_mvo_loss_val:.4f}"
            )
            print(
                f"   ⚙️ Optimizer: actor_lr={agent.get_actor_lr():.6f} | "
                f"critic_lr={agent.get_critic_lr():.6f} | target_kl={agent.target_kl:.4f} | "
                f"rollout={current_timestep_rollout} | batch_size={current_batch_size_ppo}"
            )
            if ra_kl_enabled:
                effective_kl_threshold = float(agent.target_kl * agent.kl_stop_multiplier)
                ema_display = float(ra_kl_ema_approx_kl) if ra_kl_ema_approx_kl is not None else 0.0
                print(
                    "   🧭 RA-KL: "
                    f"ema_kl={ema_display:.5f} | "
                    f"error_ratio={ra_kl_last_error_ratio:+.3f} | "
                    f"adjust={ra_kl_last_adjust_factor:.3f} | "
                    f"stop_threshold={effective_kl_threshold:.5f}"
                )
            
            # Alpha diversity logging cadence is configurable via training_params.
            if update_count % alpha_diversity_log_interval == 0:
                print(
                    f"   🔬 Alpha Diversity: mean={alpha_mean_val:.2f} | "
                    f"std={alpha_std_val:.2f} | "
                    f"range=[{alpha_min_val:.2f}, {alpha_max_val:.2f}]"
                )
                # Warning if alpha seems stuck (TCN not learning asset discrimination).
                if (
                    update_count > alpha_diversity_warning_after_updates
                    and alpha_std_val < alpha_diversity_warning_std_threshold
                ):
                    print(
                        "   ⚠️  WARNING: "
                        f"Alpha std < {alpha_diversity_warning_std_threshold:.2f} "
                        f"after {update_count} updates. "
                        f"TCN may not be learning asset discrimination."
                    )

            if episode_terminal_info is not None:
                terminal_drawdown_lambda = to_scalar(episode_terminal_info.get("drawdown_lambda", terminal_drawdown_lambda))
                terminal_drawdown_lambda_peak = to_scalar(
                    episode_terminal_info.get("drawdown_lambda_peak", terminal_drawdown_lambda_peak)
                )
                terminal_drawdown_avg_excess = to_scalar(
                    episode_terminal_info.get("drawdown_avg_excess", terminal_drawdown_avg_excess)
                )
                terminal_drawdown_penalty_sum = to_scalar(
                    episode_terminal_info.get("drawdown_penalty_sum", terminal_drawdown_penalty_sum)
                )
                terminal_intra_step_tape_potential = to_scalar(
                    episode_terminal_info.get("intra_step_tape_potential", terminal_intra_step_tape_potential)
                )
                terminal_intra_step_tape_delta_reward = to_scalar(
                    episode_terminal_info.get("intra_step_tape_delta_reward", terminal_intra_step_tape_delta_reward)
                )
                tape_score_for_log = episode_terminal_info.get("tape_score", 0.0)
                print(
                    "   🔒 Drawdown λ "
                    f"snapshot={snapshot_drawdown_lambda:.3f} (peak {snapshot_drawdown_lambda_peak:.3f}, "
                    f"dd {snapshot_drawdown_current*100.0:.2f}% / trig {snapshot_drawdown_trigger_boundary*100.0:.2f}%) | "
                    f"terminal={terminal_drawdown_lambda:.3f} (peak {terminal_drawdown_lambda_peak:.3f}) | "
                    f"TAPE={tape_score_for_log:.4f}"
                )

            training_row = {
                "update": update_count,
                "timestep": step,
                "episode": training_episode_count,
                "elapsed_time": elapsed,
                "episode_return_pct": episode_return_pct_val,
                "episode_sharpe": episode_sharpe_val,
                "episode_sortino": episode_sortino_val,
                "episode_max_dd": episode_max_dd_val,
                "episode_volatility": episode_volatility_val,
                "episode_win_rate": episode_win_rate_val,
                "episode_turnover": episode_turnover_raw_val,
                "episode_turnover_pct": episode_turnover_pct_val,
                "episode_return_skew": episode_return_skew_val,
                "episode_calmar_ratio": episode_calmar_val,
                "episode_omega_ratio": episode_omega_val,
                "episode_ulcer_index": episode_ulcer_val,
                "episode_cvar_5pct": episode_cvar_val,
                "actor_loss": actor_loss_val,
                "critic_loss": critic_loss_val,
                "critic_loss_scaled": critic_loss_scaled_val,
                "risk_aux_total": risk_aux_total_val,
                "risk_aux_sharpe_proxy": risk_aux_sharpe_proxy_val,
                "risk_aux_sharpe_loss": risk_aux_sharpe_loss_val,
                "risk_aux_mvo_loss": risk_aux_mvo_loss_val,
                "mean_advantage": mean_advantage_val,
                "profile_name": last_profile_name,
                "turnover_scalar": current_turnover_scalar,
            }

            training_row.update(
                {
                    "policy_entropy": policy_entropy_val,
                    "policy_loss": policy_loss_val,
                    "entropy_loss": entropy_loss_val,
                    "approx_kl": approx_kl_val,
                    "target_kl_active": float(agent.target_kl),
                    "kl_stop_threshold_active": float(agent.target_kl * agent.kl_stop_multiplier),
                    "ra_kl_enabled": bool(ra_kl_enabled),
                    "ra_kl_ema_approx_kl": float(ra_kl_ema_approx_kl) if ra_kl_ema_approx_kl is not None else np.nan,
                    "ra_kl_error_ratio": float(ra_kl_last_error_ratio) if ra_kl_enabled else np.nan,
                    "ra_kl_adjust_factor": float(ra_kl_last_adjust_factor) if ra_kl_enabled else np.nan,
                    "clip_fraction": clip_fraction_val,
                    "value_clip_fraction": value_clip_fraction_val,
                    "explained_variance": explained_variance_val,
                    "actor_grad_norm": actor_grad_norm_val,
                    "critic_grad_norm": critic_grad_norm_val,
                    "alpha_min": alpha_min_val,
                    "alpha_max": alpha_max_val,
                    "alpha_mean": alpha_mean_val,
                    "ratio_mean": ratio_mean_val,
                    "ratio_std": ratio_std_val,
                    "terminal_drawdown_lambda": terminal_drawdown_lambda,
                    "terminal_drawdown_lambda_peak": terminal_drawdown_lambda_peak,
                    "terminal_drawdown_avg_excess": terminal_drawdown_avg_excess,
                    "terminal_drawdown_penalty_sum": terminal_drawdown_penalty_sum,
                    "snapshot_drawdown_lambda": snapshot_drawdown_lambda,
                    "snapshot_drawdown_lambda_peak": snapshot_drawdown_lambda_peak,
                    "snapshot_drawdown_current": snapshot_drawdown_current,
                    "snapshot_drawdown_avg_excess": snapshot_drawdown_avg_excess,
                    "snapshot_drawdown_penalty_sum": snapshot_drawdown_penalty_sum,
                    "snapshot_drawdown_triggered": snapshot_drawdown_triggered,
                    "snapshot_drawdown_trigger_boundary": snapshot_drawdown_trigger_boundary,
                    "snapshot_drawdown_target": snapshot_drawdown_target,
                    "snapshot_drawdown_tolerance": snapshot_drawdown_tolerance,
                    "snapshot_intra_step_tape_potential": snapshot_intra_step_tape_potential,
                    "snapshot_intra_step_tape_delta_reward": snapshot_intra_step_tape_delta_reward,
                    "drawdown_lambda": terminal_drawdown_lambda,
                    "drawdown_lambda_peak": terminal_drawdown_lambda_peak,
                    "tape_score": last_tape_score,
                    "tape_bonus": last_tape_bonus,
                    "tape_bonus_raw": last_tape_bonus_raw,
                    "tape_terminal_bonus_mode": last_tape_terminal_bonus_mode,
                    "tape_terminal_baseline": last_tape_terminal_baseline,
                    "tape_terminal_neutral_band_applied": last_tape_terminal_neutral_band_applied,
                    "tape_terminal_neutral_band_halfwidth": last_tape_terminal_neutral_band_halfwidth,
                    "tape_gate_a_triggered": last_tape_gate_a_triggered,
                    "tape_gate_a_sharpe": last_tape_gate_a_sharpe,
                    "tape_gate_a_max_drawdown_abs": last_tape_gate_a_max_drawdown_abs,
                    "terminal_intra_step_tape_potential": terminal_intra_step_tape_potential,
                    "terminal_intra_step_tape_delta_reward": terminal_intra_step_tape_delta_reward,
                    "drawdown_avg_excess": terminal_drawdown_avg_excess,
                    "drawdown_penalty_sum": terminal_drawdown_penalty_sum,
                    "initial_balance": last_initial_balance,
                    "final_balance": last_final_balance,
                    "next_profile_name": last_next_profile_name,
                    "next_profile_reason": last_next_profile_reason,
                    "episode_length": last_episode_length_info,
                    "termination_reason": last_termination_reason,
                }
            )

            for field in training_fieldnames:
                training_row.setdefault(field, None)

            training_rows.append(training_row)
            if train_csv_logger is not None:
                train_csv_logger.log(training_row)

    train_end = time.time()
    train_duration = train_end - train_start
    print(f"\n✅ THREE-COMPONENT TAPE v3 training completed!")
    print(f"   Total episodes: {training_episode_count}")
    print(f"   Total timesteps: {step:,}")
    print(f"   Training time: {train_duration:.2f}s ({train_duration/60:.2f}min)")

    df_training_rows = pd.DataFrame(training_rows)
    df_training_rows.to_csv(training_summary_path, index=False)
    print(f"📊 Training summary saved: {training_summary_path}")

    if train_csv_logger is not None:
        train_csv_logger.close()
    if step_diag_csv_logger is not None:
        step_diag_csv_logger.close()

    final_sharpe = float(to_scalar(last_episode_metrics.get("episode_sharpe", 0.0)) or 0.0)
    final_sharpe_tag = _format_checkpoint_metric_tag(final_sharpe)
    checkpoint_prefix_path = (
        high_watermark_checkpoint_dir
        / f"exp{exp_idx}_tape_hw_ep{int(training_episode_count):05d}_sh{final_sharpe_tag}"
    )
    checkpoint_prefix_path.parent.mkdir(parents=True, exist_ok=True)
    agent.save_models(str(checkpoint_prefix_path))
    print(f"💾 Final models saved: {checkpoint_prefix_path}_actor.weights.h5, {checkpoint_prefix_path}_critic.weights.h5")
    saved_checkpoint_records.append(
        {
            "type": "final_high_watermark_style",
            "episode": int(training_episode_count),
            "step": int(step),
            "sharpe": float(final_sharpe),
            "actor_path": f"{checkpoint_prefix_path}_actor.weights.h5",
            "critic_path": f"{checkpoint_prefix_path}_critic.weights.h5",
        }
    )

    selected_checkpoint_prefix_path = (
        Path(deterministic_validation_best_path)
        if deterministic_validation_best_path
        else checkpoint_prefix_path
    )
    if deterministic_validation_best_path:
        print(
            "🎯 Default selected checkpoint (best deterministic validation): "
            f"{selected_checkpoint_prefix_path}"
        )
        print(
            "   ↳ Selection basis: deterministic validation Sharpe "
            f"{deterministic_validation_best_sharpe:.3f} at episode {deterministic_validation_best_episode}"
        )
    else:
        print("🎯 Default selected checkpoint: final high-watermark-style checkpoint")

    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata_latest = json.load(f)
        metadata_latest.setdefault("Checkpointing", {})
        # De-duplicate by actor path while preserving insertion order.
        unique_records: List[Dict[str, Any]] = []
        seen_actor_paths: set = set()
        for rec in saved_checkpoint_records:
            actor_path = str(rec.get("actor_path", ""))
            if actor_path in seen_actor_paths:
                continue
            seen_actor_paths.add(actor_path)
            unique_records.append(rec)
        if deterministic_validation_best_episode is not None and np.isfinite(deterministic_validation_best_sharpe):
            checkpoint_description = (
                f"Best deterministic-validation checkpoint episode {deterministic_validation_best_episode} "
                f"(Val Sharpe={deterministic_validation_best_sharpe:.3f}); "
                f"final checkpoint episode {training_episode_count} (Train Sharpe={final_sharpe:.3f})"
            )
        else:
            checkpoint_description = (
                f"Final high-watermark-style checkpoint episode {training_episode_count} "
                f"(Sharpe={final_sharpe:.3f})"
            )

        metadata_latest["Checkpointing"].update(
            {
                "checkpoint_description": checkpoint_description,
                "final_actor_weights_path": f"{checkpoint_prefix_path}_actor.weights.h5",
                "final_critic_weights_path": f"{checkpoint_prefix_path}_critic.weights.h5",
                "deterministic_validation_best_episode": (
                    int(deterministic_validation_best_episode)
                    if deterministic_validation_best_episode is not None
                    else None
                ),
                "deterministic_validation_best_sharpe": (
                    float(deterministic_validation_best_sharpe)
                    if np.isfinite(deterministic_validation_best_sharpe)
                    else None
                ),
                "deterministic_validation_best_checkpoint_prefix": deterministic_validation_best_path,
                "selected_checkpoint_prefix": str(selected_checkpoint_prefix_path),
                "selected_checkpoint_source": (
                    "deterministic_validation_best"
                    if deterministic_validation_best_path is not None
                    else "final_high_watermark_style"
                ),
                "deterministic_validation_last_metrics": _json_ready(deterministic_validation_last_metrics),
                "saved_checkpoints_for_this_run": unique_records,
                "saved_checkpoints_count": int(len(unique_records)),
            }
        )
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata_latest, f, indent=2, default=str)
    except Exception as exc:
        print(f"⚠️ Could not append checkpoint details to metadata JSON: {exc}")

    return Experiment6Result(
        exp_idx=exp_idx,
        exp_name=resolved_exp_name,
        experiment_seed=experiment_seed,
        architecture=arch_upper,
        use_covariance=use_covariance,
        agent=agent,
        agent_config=agent_config,
        env_train=env_train,
        env_test_deterministic=env_test_deterministic,
        env_test_random=env_test_random,
        env_test_alias=env_test_alias,
        rare_records=rare_records,
        training_summary_path=str(training_summary_path),
        training_episodes_path=str(training_episodes_path),
        training_custom_path=str(training_custom_path),
        training_rows=df_training_rows,
        checkpoint_path=str(selected_checkpoint_prefix_path),
        total_timesteps=step,
        total_episodes=training_episode_count,
        training_duration=train_duration,
        turnover_curriculum=turnover_curriculum,
        actor_lr_schedule=actor_lr_schedule,
    )


def _classify_market_regime(date_str: str) -> str:
    """Classify market regime based on date.
    
    Args:
        date_str: Date string in format 'YYYY-MM-DD'
        
    Returns:
        String describing the market regime
    """
    from datetime import datetime
    try:
        date = datetime.strptime(date_str, '%Y-%m-%d')
    except:
        return "Unknown"
    
    # Define regime boundaries
    if date < datetime(2020, 3, 1):
        return "Pre-COVID (2020 Q1)"
    elif date < datetime(2020, 6, 1):
        return "COVID Crash (2020 Q1)"
    elif date < datetime(2021, 1, 1):
        return "COVID Recovery (2020 Q2-Q4)"
    elif date < datetime(2022, 1, 1):
        return "Post-Pandemic Rally (2021)"
    elif date < datetime(2023, 1, 1):
        return "Rate Hikes / Tech Correction (2022)"
    elif date < datetime(2024, 1, 1):
        return "Market Stabilization (2023)"
    elif date < datetime(2025, 1, 1):
        return "Continued Growth (2024)"
    else:
        return "Current Period (2025+)"


def evaluate_experiment6_checkpoint(
    experiment6: Experiment6Result,
    phase1_data: Phase1Dataset,
    config: Dict[str, Any],
    *,
    random_seed: int,
    use_final_model: bool = False,
    use_rare_model: bool = False,
    checkpoint_episode: int = 51,
    use_clip_checkpoint: bool = False,
    clip_episode: int = 28,
    model_family: Optional[str] = None,
    normal_model_strategy: str = "latest",
    rare_model_strategy: str = "best",
    num_eval_runs: int = 10,
    stochastic_episode_length_limit: int = 252,
    sample_actions: Optional[bool] = None,
    sample_actions_deterministic: Optional[bool] = None,
    sample_actions_stochastic: Optional[bool] = None,
    deterministic_eval_mode: str = "mode",
    compare_deterministic_modes: Optional[List[str]] = None,
    stochastic_eval_mode: str = "sample",
    checkpoint_path_override: Optional[str] = None,
    save_eval_logs: bool = True,
    save_eval_artifacts: bool = True,
    load_only: bool = False,
) -> Experiment6Evaluation:
    """
    Evaluate Experiment 6 checkpoints (final/rare/clip) with deterministic + stochastic tests.
    """
    exp_idx = experiment6.exp_idx
    results_root = Path(experiment6.checkpoint_path).parent
    rare_dir = results_root / "rare_models"
    data_processor = phase1_data.data_processor
    test_df = phase1_data.test_df.copy()
    training_params = config.get("training_params", {})
    env_params = config.get("environment_params", {})

    tape_terminal_scalar = float(env_params.get("tape_terminal_scalar", 10.0))
    tape_terminal_clip = float(env_params.get("tape_terminal_clip", 10.0))
    tape_terminal_bonus_mode = str(env_params.get("tape_terminal_bonus_mode", "signed")).lower().strip()
    tape_terminal_baseline = float(env_params.get("tape_terminal_baseline", 0.20))
    tape_terminal_neutral_band_enabled = bool(env_params.get("tape_terminal_neutral_band_enabled", True))
    tape_terminal_neutral_band_halfwidth = float(env_params.get("tape_terminal_neutral_band_halfwidth", 0.02))
    tape_terminal_gate_a_enabled = bool(env_params.get("tape_terminal_gate_a_enabled", False))
    tape_terminal_gate_a_sharpe_threshold = float(env_params.get("tape_terminal_gate_a_sharpe_threshold", 0.0))
    tape_terminal_gate_a_max_drawdown = float(env_params.get("tape_terminal_gate_a_max_drawdown", 0.25))
    dsr_scalar_cfg = float(env_params.get("dsr_scalar", 7.0))
    target_turnover_cfg = float(env_params.get("target_turnover", 0.60))
    turnover_band_cfg = float(env_params.get("turnover_target_band", 0.20))
    gamma_cfg = float(config.get("agent_params", {}).get("ppo_params", {}).get("gamma", 0.99))
    eval_turnover_scalar = float(
        training_params.get(
            "evaluation_turnover_penalty_scalar",
            env_params.get("turnover_penalty_scalar", 1.5),
        )
    )

    def _resolve_explicit_paths(path_override: str) -> Tuple[str, str, str]:
        """Return (actor_path, critic_path, description) for a user-provided checkpoint path or prefix."""
        path_override = path_override.strip()
        actor_candidate: Optional[str] = None
        critic_candidate: Optional[str] = None
        description = f"Custom checkpoint ({path_override})"

        if path_override.endswith("_actor.weights.h5"):
            actor_candidate = path_override
            critic_candidate = path_override.replace("_actor.weights.h5", "_critic.weights.h5")
        elif path_override.endswith("_critic.weights.h5"):
            critic_candidate = path_override
            actor_candidate = path_override.replace("_critic.weights.h5", "_actor.weights.h5")
        else:
            actor_candidate = f"{path_override}_actor.weights.h5"
            critic_candidate = f"{path_override}_critic.weights.h5"
        return actor_candidate, critic_candidate, description

    def _resolve_normal_model(strategy: str) -> Tuple[str, str, str]:
        strategy_l = (strategy or "latest").strip().lower()
        if strategy_l == "latest":
            prefix = _latest_normal_checkpoint_prefix(results_root, exp_idx)
            if prefix is None:
                prefix = Path(experiment6.checkpoint_path)
            actor = f"{prefix}_actor.weights.h5"
            critic = f"{prefix}_critic.weights.h5"
            return actor, critic, f"Normal latest ({prefix.name})"
        if strategy_l == "final":
            prefix_tape = results_root / f"exp{exp_idx}_tape_final"
            prefix_legacy = results_root / f"exp{exp_idx}_final"
            if Path(f"{prefix_tape}_actor.weights.h5").exists():
                prefix = prefix_tape
            elif Path(f"{prefix_legacy}_actor.weights.h5").exists():
                prefix = prefix_legacy
            else:
                prefix = _latest_normal_checkpoint_prefix(results_root, exp_idx) or Path(experiment6.checkpoint_path)
            return (
                f"{prefix}_actor.weights.h5",
                f"{prefix}_critic.weights.h5",
                f"Normal final ({prefix.name})",
            )
        raise ValueError("normal_model_strategy must be one of {'latest', 'final'}")

    def _resolve_rare_model(strategy: str) -> Tuple[str, str, str]:
        strategy_l = (strategy or "best").strip().lower()
        if strategy_l == "best":
            best_actor = _best_rare_actor_checkpoint(rare_dir, exp_idx)
            if best_actor is None:
                raise FileNotFoundError(f"No rare checkpoints found in {rare_dir}")
            best_critic = str(best_actor).replace("_actor.weights.h5", "_critic.weights.h5")
            return str(best_actor), best_critic, f"Rare best ({Path(best_actor).name})"
        if strategy_l == "episode":
            matching_files: List[str] = []
            for actor_path in sorted(rare_dir.glob("*_actor.weights.h5")):
                ep = _extract_episode_number_from_name(actor_path.name, exp_idx)
                if ep == int(checkpoint_episode):
                    matching_files.append(str(actor_path))
            if not matching_files:
                raise FileNotFoundError(
                    f"Rare model checkpoint not found for episode {checkpoint_episode}"
                )
            actor = matching_files[0]
            critic = actor.replace("_actor.weights.h5", "_critic.weights.h5")
            return actor, critic, f"Rare episode ({checkpoint_episode})"
        raise ValueError("rare_model_strategy must be one of {'best', 'episode'}")

    selector = (model_family or "").strip().lower()
    if checkpoint_path_override:
        actor_weights_path, critic_weights_path, checkpoint_description = _resolve_explicit_paths(
            checkpoint_path_override
        )
        print("\n" + "=" * 80)
        print(f"LOADING CUSTOM CHECKPOINT: {checkpoint_path_override}")
        print("=" * 80)
    elif selector in {"normal", "latest"}:
        actor_weights_path, critic_weights_path, checkpoint_description = _resolve_normal_model(
            normal_model_strategy
        )
        print("\n" + "=" * 80)
        print(f"LOADING NORMAL MODEL ({normal_model_strategy.upper()})")
        print("=" * 80)
        print(f"📂 {checkpoint_description}")
    elif selector == "rare":
        actor_weights_path, critic_weights_path, checkpoint_description = _resolve_rare_model(
            rare_model_strategy
        )
        print("\n" + "=" * 80)
        print(f"LOADING RARE MODEL ({rare_model_strategy.upper()})")
        print("=" * 80)
        print(f"📂 {checkpoint_description}")
    elif selector == "clip":
        checkpoint_prefix = results_root / f"exp{exp_idx}_tape_clip_ep{clip_episode}"
        actor_weights_path = f"{checkpoint_prefix}_actor.weights.h5"
        critic_weights_path = f"{checkpoint_prefix}_critic.weights.h5"
        checkpoint_description = f"TAPE clip checkpoint (episode {clip_episode})"
        print("\n" + "=" * 80)
        print("LOADING TAPE CLIP MODEL")
        print("=" * 80)
        print(f"📂 Loading {checkpoint_description}")
    elif use_final_model:
        actor_weights_path = f"{experiment6.checkpoint_path}_actor.weights.h5"
        critic_weights_path = f"{experiment6.checkpoint_path}_critic.weights.h5"
        checkpoint_description = "Final model"
        print("\n" + "=" * 80)
        print("LOADING FINAL MODEL (End of Training)")
        print("=" * 80)
        print("📂 Loading final model...")
    elif use_rare_model:
        print("\n" + "=" * 80)
        print(f"LOADING RARE MODEL: Episode {checkpoint_episode}")
        print("=" * 80)
        matching_files: List[str] = []
        for actor_path in sorted(rare_dir.glob("*_actor.weights.h5")):
            ep = _extract_episode_number_from_name(actor_path.name, exp_idx)
            if ep == int(checkpoint_episode):
                matching_files.append(str(actor_path))
        if not matching_files:
            print(f"❌ No rare model found for episode {checkpoint_episode}")
            print(f"   Searched all actor checkpoints in: {rare_dir}")
            print("\n📁 Available rare models:")
            for p in sorted(rare_dir.glob("*_actor.weights.h5")):
                print(f"   {p.name}")
            raise FileNotFoundError(f"Rare model checkpoint not found for episode {checkpoint_episode}")

        actor_weights_path = matching_files[0]
        critic_weights_path = actor_weights_path.replace("_actor.weights.h5", "_critic.weights.h5")
        checkpoint_description = f"Rare checkpoint (episode {checkpoint_episode})"
        print(f"✅ Found rare model: {actor_weights_path}")
    else:
        actor_weights_path, critic_weights_path, checkpoint_description = _resolve_normal_model("latest")
        print("\n" + "=" * 80)
        print("LOADING NORMAL MODEL (LATEST DEFAULT)")
        print("=" * 80)
        print(f"📂 {checkpoint_description}")

    path_obj = Path(actor_weights_path)
    if not path_obj.exists():
        print(f"❌ Actor weights not found: {actor_weights_path}")
        print("\n📁 Available checkpoints:")
        search_path = rare_dir if use_rare_model else results_root
        for p in sorted(search_path.glob("exp6_*.weights.h5")):
            print(f"   {p}")
    else:
        print(f"✅ Found actor weights: {actor_weights_path}")
        print(f"✅ Found critic weights: {critic_weights_path}")
        if checkpoint_path_override:
            results_root = _infer_results_root_from_actor_weights(path_obj)

    attention_needed = _infer_attention_from_checkpoint_path(actor_weights_path) or _checkpoint_has_attention(actor_weights_path)
    fusion_signature = _infer_fusion_input_signature_from_actor_weights(actor_weights_path)

    print("🏗️ Recreating evaluation environments...")
    drawdown_constraint_eval = _prepare_drawdown_constraint(config, experiment6.architecture)
    active_eval_profile = copy.deepcopy(
        getattr(getattr(experiment6, "env_train", None), "tape_profile", None)
    )
    if not active_eval_profile:
        profile_override = env_params.get("tape_profile_override")
        if isinstance(profile_override, dict) and profile_override:
            active_eval_profile = copy.deepcopy(profile_override)
    if not active_eval_profile:
        active_eval_profile = copy.deepcopy(PROFILE_BALANCED_GROWTH)

    env_test_deterministic = PortfolioEnvTAPE(
        config=config,
        data_processor=data_processor,
        processed_data=test_df.copy(),
        mode="test",
        action_normalization="none",
        exclude_covariance=not experiment6.use_covariance,
        reward_system="tape",
        tape_profile=active_eval_profile,
        tape_terminal_scalar=tape_terminal_scalar,
        tape_terminal_bonus_mode=tape_terminal_bonus_mode,
        tape_terminal_baseline=tape_terminal_baseline,
        tape_terminal_neutral_band_enabled=tape_terminal_neutral_band_enabled,
        tape_terminal_neutral_band_halfwidth=tape_terminal_neutral_band_halfwidth,
        tape_terminal_gate_a_enabled=tape_terminal_gate_a_enabled,
        tape_terminal_gate_a_sharpe_threshold=tape_terminal_gate_a_sharpe_threshold,
        tape_terminal_gate_a_max_drawdown=tape_terminal_gate_a_max_drawdown,
        dsr_window=60,
        dsr_scalar=dsr_scalar_cfg,
        target_turnover=target_turnover_cfg,
        turnover_target_band=turnover_band_cfg,
        enable_base_reward=True,
        turnover_penalty_scalar=eval_turnover_scalar,
        gamma=gamma_cfg,
        random_start=False,
        episode_length_limit=None,
        tape_terminal_clip=tape_terminal_clip,
        drawdown_constraint=copy.deepcopy(drawdown_constraint_eval),
    )

    env_test_random = PortfolioEnvTAPE(
        config=config,
        data_processor=data_processor,
        processed_data=test_df.copy(),
        mode="test",
        action_normalization="none",
        exclude_covariance=not experiment6.use_covariance,
        reward_system="tape",
        tape_profile=active_eval_profile,
        tape_terminal_scalar=tape_terminal_scalar,
        tape_terminal_bonus_mode=tape_terminal_bonus_mode,
        tape_terminal_baseline=tape_terminal_baseline,
        tape_terminal_neutral_band_enabled=tape_terminal_neutral_band_enabled,
        tape_terminal_neutral_band_halfwidth=tape_terminal_neutral_band_halfwidth,
        tape_terminal_gate_a_enabled=tape_terminal_gate_a_enabled,
        tape_terminal_gate_a_sharpe_threshold=tape_terminal_gate_a_sharpe_threshold,
        tape_terminal_gate_a_max_drawdown=tape_terminal_gate_a_max_drawdown,
        dsr_window=60,
        dsr_scalar=dsr_scalar_cfg,
        target_turnover=target_turnover_cfg,
        turnover_target_band=turnover_band_cfg,
        enable_base_reward=True,
        turnover_penalty_scalar=eval_turnover_scalar,
        gamma=gamma_cfg,
        random_start=True,
        episode_length_limit=stochastic_episode_length_limit,
        tape_terminal_clip=tape_terminal_clip,
        drawdown_constraint=copy.deepcopy(drawdown_constraint_eval),
    )

    state_dim = env_test_deterministic.observation_space.shape[0]
    stock_dim = env_test_deterministic.num_assets
    agent_config_eval = copy.deepcopy(experiment6.agent_config)
    arch_hints = _infer_checkpoint_architecture_hints(
        actor_weights_path,
        fallback_architecture=str(agent_config_eval.get("actor_critic_type", experiment6.architecture or "TCN")),
        fallback_use_attention=bool(agent_config_eval.get("use_attention", False)),
        fallback_use_fusion=bool(agent_config_eval.get("use_fusion", False)),
    )
    if (
        str(agent_config_eval.get("actor_critic_type", "")).upper() != str(arch_hints.get("actor_critic_type", "")).upper()
        or bool(agent_config_eval.get("use_attention", False)) != bool(arch_hints.get("use_attention", False))
        or bool(agent_config_eval.get("use_fusion", False)) != bool(arch_hints.get("use_fusion", False))
    ):
        print(
            "   ⚠️ Adjusting agent architecture from checkpoint hints: "
            f"{agent_config_eval.get('actor_critic_type')} -> {arch_hints['actor_critic_type']} "
            f"(attention={arch_hints['use_attention']}, fusion={arch_hints['use_fusion']}, source={arch_hints['source']})"
        )
    agent_config_eval["actor_critic_type"] = str(arch_hints["actor_critic_type"]).upper()
    agent_config_eval["use_attention"] = bool(arch_hints["use_attention"])
    agent_config_eval["use_fusion"] = bool(arch_hints["use_fusion"])
    if attention_needed and experiment6.architecture.upper().startswith("TCN"):
        agent_config_eval["use_attention"] = True
    agent_config_eval["debug_prints"] = False
    prefer_training_state_layout = bool(
        config.get("training_params", {}).get("prefer_training_state_layout_for_eval", True)
    )
    if hasattr(env_test_deterministic, "get_observation_layout"):
        try:
            eval_layout = env_test_deterministic.get_observation_layout()
            if isinstance(eval_layout, dict) and eval_layout:
                apply_env_layout = True
                existing_layout = agent_config_eval.get("state_layout")
                if (
                    prefer_training_state_layout
                    and isinstance(existing_layout, dict)
                    and bool(existing_layout)
                ):
                    # Keep training-time layout unless we explicitly infer better from checkpoint.
                    apply_env_layout = False
                if (
                    fusion_signature
                    and str(agent_config_eval.get("actor_critic_type", "")).upper().startswith("TCN_FUSION")
                ):
                    ckpt_asset_dim = int(fusion_signature.get("conv1_in_channels", 0) or 0)
                    ckpt_global_proj_in = int(fusion_signature.get("global_projection_in_dim", 0) or 0)
                    env_asset_dim = int(eval_layout.get("asset_feature_dim", 0) or 0)
                    env_global_dim = int(eval_layout.get("global_feature_dim", 0) or 0)
                    env_local_flat_dim = int(eval_layout.get("local_flat_dim", stock_dim * max(env_asset_dim, 0)) or 0)

                    layout_mismatch = (
                        ckpt_asset_dim > 0
                        and (
                            ckpt_asset_dim != env_asset_dim
                            or (
                                ckpt_global_proj_in > 0
                                and ckpt_global_proj_in not in {env_global_dim, env_local_flat_dim}
                            )
                        )
                    )

                    if layout_mismatch:
                        apply_env_layout = False
                        inferred_local_flat = stock_dim * max(ckpt_asset_dim, 0)
                        inferred_global_dim = 0 if ckpt_global_proj_in == inferred_local_flat else max(ckpt_global_proj_in, 0)
                        inferred_structured = inferred_global_dim > 0

                        if inferred_structured:
                            agent_config_eval["state_layout"] = {
                                "structured_observation": True,
                                "num_assets": int(stock_dim),
                                "asset_feature_dim": int(ckpt_asset_dim),
                                "global_feature_dim": int(inferred_global_dim),
                                "local_flat_dim": int(inferred_local_flat),
                                "total_observation_dim": int(inferred_local_flat + inferred_global_dim),
                            }
                        else:
                            # Flat fallback: rely on actor input_dim split used during training.
                            agent_config_eval.pop("state_layout", None)

                        agent_config_eval["asset_feature_dim"] = int(ckpt_asset_dim)
                        agent_config_eval["global_feature_dim"] = int(inferred_global_dim)
                        print(
                            "   ⚠️ Checkpoint/env layout mismatch detected. "
                            f"checkpoint(conv1_in={ckpt_asset_dim}, global_proj_in={ckpt_global_proj_in}) "
                            f"vs env(asset={env_asset_dim}, global={env_global_dim}). "
                            f"Using {'structured' if inferred_structured else 'flat'} checkpoint-aligned inputs."
                        )

                if apply_env_layout:
                    agent_config_eval["state_layout"] = copy.deepcopy(eval_layout)
        except Exception:
            pass

    print(
        "   🧭 Checkpoint architecture: "
        f"{agent_config_eval.get('actor_critic_type')} "
        f"(attention={bool(agent_config_eval.get('use_attention', False))}, "
        f"fusion={bool(agent_config_eval.get('use_fusion', False))}, source={arch_hints.get('source')})"
    )
    print(
        "   🧱 Eval TCN stack: "
        f"filters={agent_config_eval.get('tcn_filters')} | "
        f"kernel={agent_config_eval.get('tcn_kernel_size')} | "
        f"dilations={agent_config_eval.get('tcn_dilations')} | "
        f"dropout={agent_config_eval.get('tcn_dropout')}"
    )
    if bool(agent_config_eval.get("use_fusion", False)) or str(agent_config_eval.get("actor_critic_type", "")).upper().endswith("FUSION"):
        print(
            "   🧩 Eval fusion core: "
            f"embed={agent_config_eval.get('fusion_embed_dim')} | "
            f"heads={agent_config_eval.get('fusion_attention_heads')} | "
            f"dropout={agent_config_eval.get('fusion_dropout')}"
        )
        print(
            "   🔀 Eval mixer (A4): "
            f"enabled={bool(agent_config_eval.get('fusion_cross_asset_mixer_enabled', False))} | "
            f"layers={agent_config_eval.get('fusion_cross_asset_mixer_layers', 1)} | "
            f"expansion={agent_config_eval.get('fusion_cross_asset_mixer_expansion', 2.0)} | "
            f"dropout={agent_config_eval.get('fusion_cross_asset_mixer_dropout', agent_config_eval.get('fusion_dropout'))}"
        )
        print(
            "   🎯 Eval alpha head (A3): "
            f"dims={agent_config_eval.get('fusion_alpha_head_hidden_dims', [])} | "
            f"dropout={agent_config_eval.get('fusion_alpha_head_dropout', agent_config_eval.get('fusion_dropout'))}"
        )
    print(
        "   🎛️ Eval dirichlet: "
        f"activation={agent_config_eval.get('dirichlet_alpha_activation')} | "
        f"temperature={agent_config_eval.get('dirichlet_logit_temperature', agent_config_eval.get('logit_temperature', 1.0))} | "
        f"alpha_cap={agent_config_eval.get('dirichlet_alpha_cap', agent_config_eval.get('alpha_cap', None))} | "
        f"epsilon={agent_config_eval.get('dirichlet_epsilon')}"
    )

    agent_eval = PPOAgentTF(
        state_dim=state_dim,
        num_assets=stock_dim,
        config=agent_config_eval,
        name="PPOAgent_Exp6_Eval",
    )
    agent_eval.set_dirichlet_progress(1.0)

    print("🔧 Building models before loading weights...")
    if getattr(agent_eval, "is_sequential", False):
        seq_len = agent_config_eval.get("sequence_length", getattr(agent_eval, "sequence_length", 1) or 1)
        if getattr(agent_eval, "uses_structured_state_inputs", getattr(agent_eval, "uses_structured_fusion_inputs", False)):
            asset_dim = int(getattr(agent_eval, "asset_feature_dim", 0) or 0)
            global_dim = int(getattr(agent_eval, "global_feature_dim", 0) or 0)
            dummy_state = {
                "asset": tf.zeros((1, seq_len, stock_dim, asset_dim), dtype=tf.float32),
                "context": tf.zeros((1, seq_len, global_dim), dtype=tf.float32),
            }
        else:
            dummy_state = tf.zeros((1, seq_len, state_dim), dtype=tf.float32)
    else:
        dummy_state = tf.zeros((1, state_dim), dtype=tf.float32)
    _ = agent_eval.actor(dummy_state)
    _ = agent_eval.critic(dummy_state)
    print("   ✅ Models built successfully")

    def _load_weights_with_compat(model, weights_path: str, label: str) -> None:
        try:
            model.load_weights(weights_path)
            return
        except Exception as primary_exc:
            # Keras 2↔3 checkpoint object-path drift can sometimes be recovered via by_name.
            try:
                model.load_weights(weights_path, by_name=True, skip_mismatch=False)
                print(f"   ⚠️ {label} loaded via by_name compatibility fallback")
                return
            except (TypeError, ValueError) as kw_exc:
                # Keras 3 may reject by_name/skip_mismatch kwargs for this format.
                print(f"   ❌ {label} load failed; by_name fallback unsupported in this runtime: {kw_exc}")
                raise primary_exc
            except Exception as fallback_exc:
                print(f"   ❌ {label} load failed (primary + by_name fallback)")
                print(f"      primary: {type(primary_exc).__name__}: {primary_exc}")
                print(f"      fallback: {type(fallback_exc).__name__}: {fallback_exc}")
                raise primary_exc

    print(f"📂 Loading checkpoint weights...")
    _load_weights_with_compat(agent_eval.actor, actor_weights_path, "actor")
    _load_weights_with_compat(agent_eval.critic, critic_weights_path, "critic")
    print("   ✅ Weights loaded successfully")

    if load_only:
        print("   ✅ Load-only compatibility check passed")
        return Experiment6Evaluation(
            actor_weights_path=actor_weights_path,
            critic_weights_path=critic_weights_path,
            deterministic_metrics={},
            deterministic_portfolio=np.array([]),
            deterministic_weights=np.array([]),
            deterministic_actions=np.array([]),
            deterministic_alphas=np.array([]),
            stochastic_results=pd.DataFrame(),
            stochastic_weights=[],
            stochastic_actions=[],
            stochastic_alphas=[],
            eval_results_path="",
            checkpoint_description=checkpoint_description,
            agent=agent_eval,
            env_test_deterministic=env_test_deterministic,
            env_test_random=env_test_random,
        )

    def _normalize_mode(name: str, *, fallback: str) -> str:
        value = (name or fallback).strip().lower()
        allowed = {"mode", "mean", "sample", "mean_plus_noise"}
        if value not in allowed:
            raise ValueError(
                f"Unsupported evaluation mode '{name}'. Allowed: {sorted(allowed)}"
            )
        return value

    # Legacy compatibility mapping for old boolean flags.
    if sample_actions_deterministic is not None:
        det_modes = ["sample"] if sample_actions_deterministic else ["mode"]
    elif compare_deterministic_modes:
        det_modes = [_normalize_mode(mode, fallback="mode") for mode in compare_deterministic_modes]
    elif deterministic_eval_mode.strip().lower() != "mode":
        det_modes = [_normalize_mode(deterministic_eval_mode, fallback="mode")]
    elif sample_actions is not None:
        det_modes = ["sample"] if sample_actions else ["mode"]
    else:
        det_modes = [_normalize_mode(deterministic_eval_mode, fallback="mode")]

    # De-duplicate while preserving user order.
    seen_modes: set = set()
    det_modes = [m for m in det_modes if not (m in seen_modes or seen_modes.add(m))]

    if sample_actions_stochastic is not None:
        sto_mode = "sample" if sample_actions_stochastic else "mean"
    elif stochastic_eval_mode.strip().lower() != "sample":
        sto_mode = _normalize_mode(stochastic_eval_mode, fallback="sample")
    elif sample_actions is not None:
        sto_mode = "sample" if sample_actions else "mean"
    else:
        sto_mode = _normalize_mode(stochastic_eval_mode, fallback="sample")

    print(f"   🎯 Deterministic eval policy modes: {det_modes}")
    print(f"   🎯 Stochastic eval policy mode:     {sto_mode}")

    def _eval_policy_action(obs_tensor, eval_mode: str):
        """Get action and alpha values from policy (single actor forward pass)."""
        # Prepare state input
        if agent_eval.is_sequential:
            sequence = agent_eval._build_sequence(obs_tensor)
            state_input, needs_squeeze = agent_eval.prepare_state_input(sequence)
        else:
            state_input, needs_squeeze = agent_eval.prepare_state_input(obs_tensor)
        
        # Single actor forward pass to get alpha values
        alpha_raw = agent_eval.actor(state_input, training=False)
        alpha = tf.cast(alpha_raw, tf.float32)
        alpha = tf.maximum(alpha, tf.constant(1e-6, dtype=alpha.dtype))  # Ensure alpha > 0
        alpha_values = alpha.numpy()[0] if needs_squeeze else alpha.numpy()  # Store for return
        
        # Create Dirichlet distribution from alpha
        import tensorflow_probability as tfp
        tfd = tfp.distributions
        dirichlet = tfd.Dirichlet(alpha)
        
        mode_name = _normalize_mode(eval_mode, fallback="mode")

        if mode_name == "sample":
            action = dirichlet.sample()
        elif mode_name == "mean":
            sum_alpha = tf.reduce_sum(alpha, axis=-1, keepdims=True)
            action = alpha / tf.maximum(sum_alpha, 1e-12)
        elif mode_name == "mean_plus_noise":
            sum_alpha = tf.reduce_sum(alpha, axis=-1, keepdims=True)
            mean_action = alpha / tf.maximum(sum_alpha, 1e-12)
            noisy = mean_action + tf.random.normal(
                tf.shape(mean_action), mean=0.0, stddev=0.01, dtype=mean_action.dtype
            )
            noisy = tf.nn.relu(noisy) + 1e-12
            action = noisy / tf.reduce_sum(noisy, axis=-1, keepdims=True)
        else:
            # Dirichlet mode: (alpha-1)/(sum(alpha)-K) when alpha_i > 1; otherwise vertex on argmax(alpha)
            min_alpha = tf.reduce_min(alpha, axis=-1, keepdims=True)
            use_formula = min_alpha > 1.0
            
            sum_alpha = tf.reduce_sum(alpha, axis=-1, keepdims=True)
            k = tf.cast(tf.shape(alpha)[-1], alpha.dtype)
            mode_formula = (alpha - 1.0) / (sum_alpha - k)
            
            max_indices = tf.argmax(alpha, axis=-1)
            mode_vertex = tf.one_hot(
                max_indices,
                depth=tf.shape(alpha)[-1],
                dtype=alpha.dtype,
            )
            
            action = tf.where(use_formula, mode_formula, mode_vertex)
        
        # Get log probability
        log_prob = dirichlet.log_prob(action)
        
        # Get value estimate
        value = tf.cast(agent_eval.critic(state_input, training=False), tf.float32)
        
        # Squeeze if needed
        if needs_squeeze:
            action = tf.squeeze(action, 0)
            log_prob = tf.squeeze(log_prob, 0)
            value = tf.squeeze(value, 0)
        
        return action, log_prob, value, alpha_values

    def _count_unique_rows(arr: np.ndarray, decimals: int = 6) -> int:
        if arr.size == 0:
            return 0
        flat = np.round(arr.reshape(arr.shape[0], -1), decimals=decimals)
        return int(np.unique(flat, axis=0).shape[0])

    def _diagnostics_from_actions_alphas(
        actions_arr: np.ndarray,
        alphas_arr: np.ndarray,
    ) -> Dict[str, Any]:
        if actions_arr.size == 0:
            action_uniques = 0
        else:
            action_uniques = _count_unique_rows(actions_arr)
        if alphas_arr.size == 0:
            alpha_le1_fraction = 0.0
            argmax_alpha_uniques = 0
        else:
            alpha_le1_fraction = float(np.mean(alphas_arr <= 1.0))
            argmax_alpha_uniques = int(np.unique(np.argmax(alphas_arr, axis=1)).shape[0])
        return {
            "action_uniques": action_uniques,
            "alpha_le1_fraction": alpha_le1_fraction,
            "argmax_alpha_uniques": argmax_alpha_uniques,
        }

    def _constraint_diagnostics_from_env(env) -> Dict[str, float]:
        hhi = np.asarray(getattr(env, "concentration_hhi_history", []), dtype=np.float64)
        top_weight = np.asarray(getattr(env, "top_weight_history", []), dtype=np.float64)
        l1 = np.asarray(getattr(env, "action_realization_l1_history", []), dtype=np.float64)
        return {
            "mean_concentration_hhi": float(np.mean(hhi)) if hhi.size else 0.0,
            "mean_top_weight": float(np.mean(top_weight)) if top_weight.size else 0.0,
            "mean_action_realization_l1": float(np.mean(l1)) if l1.size else 0.0,
            "max_action_realization_l1": float(np.max(l1)) if l1.size else 0.0,
        }

    def _weights_columns(weight_dim: int) -> List[str]:
        if weight_dim <= 0:
            return []
        asset_names = list(getattr(data_processor, "asset_tickers", []) or [])
        if len(asset_names) + 1 == weight_dim:
            return asset_names + ["Cash"]
        if len(asset_names) == weight_dim:
            return asset_names
        return [f"W{i}" for i in range(weight_dim)]

    def _save_track_artifacts(
        output_dir: Path,
        file_stem: str,
        *,
        track: str,
        weights: np.ndarray,
        actions: np.ndarray,
        alphas: np.ndarray,
        run_ids: Optional[List[int]] = None,
    ) -> None:
        if not save_eval_artifacts:
            return
        output_dir.mkdir(parents=True, exist_ok=True)

        # weights_{track}.csv
        if weights.size > 0:
            w_cols = _weights_columns(weights.shape[1])
            df_w = pd.DataFrame(weights, columns=w_cols if w_cols else None)
            if run_ids is not None and len(run_ids) == len(df_w):
                df_w.insert(0, "run", run_ids)
            df_w.insert(0, "step", np.arange(len(df_w)))
            df_w.to_csv(output_dir / f"{file_stem}_weights_{track}.csv", index=False)

        # actions_{track}.csv
        if actions.size > 0:
            a_cols = _weights_columns(actions.shape[1])
            df_a = pd.DataFrame(actions, columns=a_cols if a_cols else None)
            if run_ids is not None and len(run_ids) == len(df_a):
                df_a.insert(0, "run", run_ids)
            df_a.insert(0, "step", np.arange(len(df_a)))
            df_a.to_csv(output_dir / f"{file_stem}_actions_{track}.csv", index=False)

        # alphas_{track}.csv
        if alphas.size > 0:
            alpha_cols = [f"alpha_{i}" for i in range(alphas.shape[1])]
            df_alpha = pd.DataFrame(alphas, columns=alpha_cols)
            if run_ids is not None and len(run_ids) == len(df_alpha):
                df_alpha.insert(0, "run", run_ids)
            df_alpha.insert(0, "step", np.arange(len(df_alpha)))
            df_alpha.to_csv(output_dir / f"{file_stem}_alphas_{track}.csv", index=False)

    evaluation_rows: List[Dict[str, Any]] = []
    unique_test_dates = pd.to_datetime(test_df["Date"]).drop_duplicates().sort_values().reset_index(drop=True)
    test_start_date = unique_test_dates.iloc[0].strftime("%Y-%m-%d") if len(unique_test_dates) else ""
    test_end_date = unique_test_dates.iloc[-1].strftime("%Y-%m-%d") if len(unique_test_dates) else ""

    deterministic_track_outputs: Dict[str, Dict[str, Any]] = {}

    for det_mode in det_modes:
        track_name = f"det_{det_mode}"
        print("\n" + "=" * 80)
        print(f"DETERMINISTIC EVALUATION ({track_name})")
        print("=" * 80)

        obs, info = env_test_deterministic.reset(seed=random_seed)

        # Capture start date for deterministic run
        det_start_idx = env_test_deterministic.day
        det_start_date = test_df.iloc[det_start_idx]["Date"].strftime("%Y-%m-%d")
        det_regime = _classify_market_regime(det_start_date)

        done = False
        step_count = 0
        deterministic_actions_list = []
        deterministic_alphas_list = []

        while not done:
            action, _, _, alpha_values = _eval_policy_action(obs, det_mode)
            deterministic_actions_list.append(action.numpy().copy())
            deterministic_alphas_list.append(alpha_values.copy())
            obs, reward, done, truncated, info = env_test_deterministic.step(action)
            step_count += 1
            if done or truncated:
                break

        portfolio_history = np.array(env_test_deterministic.portfolio_history)
        deterministic_weights_array = np.array(env_test_deterministic.weights_history)
        deterministic_actions_array = np.array(deterministic_actions_list)
        deterministic_alphas_array = np.array(deterministic_alphas_list)

        returns = np.diff(portfolio_history) / portfolio_history[:-1] if len(portfolio_history) > 1 else np.array([])
        weight_changes = []
        for idx in range(1, len(env_test_deterministic.weights_history)):
            weight_changes.append(
                np.abs(
                    env_test_deterministic.weights_history[idx]
                    - env_test_deterministic.weights_history[idx - 1]
                )
            )

        metrics_det = calculate_episode_metrics(
            portfolio_values=portfolio_history,
            returns=returns,
            weight_changes=weight_changes,
            risk_free_rate=0.02,
            trading_days_per_year=252,
        )

        trading_years_det = step_count / 252
        total_return_det = metrics_det["total_return"]
        if trading_years_det > 0 and total_return_det > -1:
            annualized_return_det = (1 + total_return_det) ** (1 / trading_years_det) - 1
        else:
            annualized_return_det = 0.0
        metrics_det["annualized_return"] = annualized_return_det
        metrics_det["days_traded"] = step_count

        diag_det = _diagnostics_from_actions_alphas(
            deterministic_actions_array,
            deterministic_alphas_array,
        )
        constraint_diag_det = _constraint_diagnostics_from_env(env_test_deterministic)

        print("\n📊 DETERMINISTIC TEST RESULTS:")
        print(f"   Eval Track: {track_name}")
        print(f"   Start Date: {det_start_date}")
        print(f"   Market Regime: {det_regime}")
        print(f"   Episode Length: {step_count} days ({trading_years_det:.2f} years)")
        print(f"   Final Portfolio Value: ${portfolio_history[-1]:,.2f}")
        print(f"   Total Return: {total_return_det*100:+.2f}%")
        print(f"   Annualized Return: {annualized_return_det*100:+.2f}%")
        print(f"   Sharpe Ratio: {metrics_det['sharpe_ratio']:.4f} (annualized)")
        print(f"   Sortino Ratio: {metrics_det['sortino_ratio']:.4f} (annualized)")
        print(f"   Max Drawdown: {metrics_det['max_drawdown_abs']*100:.2f}%")
        print(f"   Volatility (Ann.): {metrics_det['volatility']*100:.2f}%")
        print(f"   Turnover: {metrics_det['turnover']*100:.2f}%")
        print(f"   Win Rate: {metrics_det['win_rate']*100:.2f}%")
        print(
            f"   Diagnostics: action_uniques={diag_det['action_uniques']}, "
            f"alpha<=1 frac={diag_det['alpha_le1_fraction']:.3f}, "
            f"argmax_alpha_uniques={diag_det['argmax_alpha_uniques']}"
        )

        deterministic_days = step_count
        deterministic_final_value = float(portfolio_history[-1]) if len(portfolio_history) else None
        deterministic_row = _build_training_metrics_row(
            metrics_det,
            episode_length=deterministic_days,
            initial_balance=getattr(env_test_deterministic, "initial_balance", None),
            final_balance=deterministic_final_value,
            profile_name=info.get("profile_name") if info else "N/A",
            turnover_scalar=getattr(env_test_deterministic, "turnover_penalty_scalar", None),
            termination_reason=(info or {}).get("termination_reason", "deterministic_eval"),
            drawdown_info=info,
        )
        deterministic_row.update(
            {
                "checkpoint": checkpoint_description,
                "architecture": experiment6.architecture,
                "eval_track": track_name,
                "evaluation_type": "deterministic",
                "run": 0,
                "seed": random_seed,
                "test_start": test_start_date,
                "test_end": test_end_date,
                "start_date": det_start_date,
                "market_regime": det_regime,
                "days_traded": deterministic_days,
                "trading_years": trading_years_det,
                "final_value": deterministic_final_value,
                "total_return": total_return_det,
                "annualized_return": annualized_return_det,
                "sharpe_ratio": metrics_det.get("sharpe_ratio", 0.0),
                "sortino_ratio": metrics_det.get("sortino_ratio", 0.0),
                "max_drawdown": metrics_det.get("max_drawdown_abs", 0.0),
                "volatility": metrics_det.get("volatility", 0.0),
                "turnover": metrics_det.get("turnover", 0.0),
                "turnover_pct": metrics_det.get("turnover", 0.0) * 100.0,
                "win_rate": metrics_det.get("win_rate", 0.0),
                "action_uniques": diag_det["action_uniques"],
                "alpha_le1_fraction": diag_det["alpha_le1_fraction"],
                "argmax_alpha_uniques": diag_det["argmax_alpha_uniques"],
                "mean_concentration_hhi": constraint_diag_det["mean_concentration_hhi"],
                "mean_top_weight": constraint_diag_det["mean_top_weight"],
                "mean_action_realization_l1": constraint_diag_det["mean_action_realization_l1"],
                "max_action_realization_l1": constraint_diag_det["max_action_realization_l1"],
            }
        )
        evaluation_rows.append(deterministic_row)

        deterministic_track_outputs[track_name] = {
            "metrics": metrics_det,
            "portfolio_history": portfolio_history,
            "weights": deterministic_weights_array,
            "actions": deterministic_actions_array,
            "alphas": deterministic_alphas_array,
        }

    if not deterministic_track_outputs:
        raise RuntimeError("No deterministic evaluation tracks were executed.")

    # Preserve backward-compatible primary deterministic outputs.
    primary_det_track = "det_mode" if "det_mode" in deterministic_track_outputs else list(deterministic_track_outputs.keys())[0]
    primary_det = deterministic_track_outputs[primary_det_track]
    metrics_det = primary_det["metrics"]
    portfolio_history = primary_det["portfolio_history"]
    deterministic_weights_array = primary_det["weights"]
    deterministic_actions_array = primary_det["actions"]
    deterministic_alphas_array = primary_det["alphas"]

    print("\n" + "=" * 80)
    print(f"STOCHASTIC EVALUATIONS (Random Start = True, {num_eval_runs} Runs)")
    print("=" * 80)

    stochastic_records: List[Dict[str, Any]] = []
    stochastic_weights_list: List[np.ndarray] = []  # Track weights for each run
    stochastic_actions_list: List[np.ndarray] = []  # Track actions for each run
    stochastic_alphas_list: List[np.ndarray] = []   # Track alphas for each run

    for run_idx in range(num_eval_runs):
        run_seed = random_seed + 100 + run_idx
        obs, info = env_test_random.reset(seed=run_seed)
        
        # Capture start date for this stochastic run
        run_start_idx = env_test_random.day
        run_start_date = test_df.iloc[run_start_idx]['Date'].strftime('%Y-%m-%d')
        run_regime = _classify_market_regime(run_start_date)
        
        done = False
        step_count = 0
        run_actions_list = []  # Track actions for this run
        run_alphas_list = []   # Track alphas for this run
        run_turnover_list: List[float] = []
        run_raw_turnover_list: List[float] = []

        while not done:
            action, _, _, alpha_values = _eval_policy_action(obs, sto_mode)
            run_actions_list.append(action.numpy().copy())  # Convert to numpy first
            run_alphas_list.append(alpha_values.copy())  # Already numpy from helper
            obs, reward, done, truncated, info = env_test_random.step(action)
            step_info = info if isinstance(info, dict) else {}
            turnover_val = step_info.get("turnover")
            raw_turnover_val = step_info.get("raw_turnover")
            if turnover_val is not None:
                run_turnover_list.append(float(turnover_val))
            if raw_turnover_val is not None:
                run_raw_turnover_list.append(float(raw_turnover_val))
            step_count += 1
            if done or truncated:
                break

        portfolio_history_run = np.array(env_test_random.portfolio_history)
        returns_run = (
            np.diff(portfolio_history_run) / portfolio_history_run[:-1]
            if len(portfolio_history_run) > 1
            else np.array([])
        )
        weight_changes_run = []
        for idx in range(1, len(env_test_random.weights_history)):
            weight_changes_run.append(
                np.abs(
                    env_test_random.weights_history[idx]
                    - env_test_random.weights_history[idx - 1]
                )
            )

        metrics_run = calculate_episode_metrics(
            portfolio_values=portfolio_history_run,
            returns=returns_run,
            weight_changes=weight_changes_run,
            risk_free_rate=0.02,
            trading_days_per_year=252,
        )
        trading_years_run = step_count / 252
        total_return_run = metrics_run["total_return"]
        if trading_years_run > 0 and total_return_run > -1:
            annualized_return_run = (1 + total_return_run) ** (1 / trading_years_run) - 1
        else:
            annualized_return_run = 0.0

        # Capture weights/actions/alphas for this run before building diagnostics.
        run_weights = np.array(env_test_random.weights_history)
        run_actions = np.array(run_actions_list)
        run_alphas = np.array(run_alphas_list)
        constraint_diag_run = _constraint_diagnostics_from_env(env_test_random)
        turnover_step_values = np.asarray(run_turnover_list, dtype=np.float64)
        turnover_step_values = turnover_step_values[np.isfinite(turnover_step_values)]
        raw_turnover_step_values = np.asarray(run_raw_turnover_list, dtype=np.float64)
        raw_turnover_step_values = raw_turnover_step_values[np.isfinite(raw_turnover_step_values)]
        turnover_step_mean = float(np.mean(turnover_step_values)) if turnover_step_values.size else 0.0
        turnover_step_p95 = float(np.percentile(turnover_step_values, 95)) if turnover_step_values.size else 0.0
        turnover_step_max = float(np.max(turnover_step_values)) if turnover_step_values.size else 0.0
        raw_turnover_step_mean = float(np.mean(raw_turnover_step_values)) if raw_turnover_step_values.size else 0.0
        turnover_target_step = float(getattr(env_test_random, "target_turnover_per_step", 0.0) or 0.0)
        if turnover_step_values.size and turnover_target_step > 0.0:
            turnover_exceed_rate = float(np.mean(turnover_step_values > turnover_target_step))
            turnover_excess_mean = float(
                np.mean(np.maximum(0.0, turnover_step_values - turnover_target_step))
            )
        else:
            turnover_exceed_rate = 0.0
            turnover_excess_mean = 0.0
        executed_to_raw_turnover_ratio = (
            float(turnover_step_mean / max(raw_turnover_step_mean, 1e-12))
            if raw_turnover_step_mean > 0.0
            else 0.0
        )

        run_final_value = float(portfolio_history_run[-1]) if len(portfolio_history_run) else None
        eval_row = _build_training_metrics_row(
            metrics_run,
            episode_length=step_count,
            initial_balance=getattr(env_test_random, "initial_balance", None),
            final_balance=run_final_value,
            profile_name=info.get("profile_name") if info else "N/A",
            turnover_scalar=getattr(env_test_random, "turnover_penalty_scalar", None),
            termination_reason=(info or {}).get("termination_reason", "stochastic_eval"),
            drawdown_info=info,
        )
        eval_row.update(
            {
                "checkpoint": checkpoint_description,
                "architecture": experiment6.architecture,
                "eval_track": "stochastic",
                "evaluation_type": "stochastic",
                "run": run_idx + 1,
                "seed": run_seed,
                "test_start": test_start_date,
                "test_end": test_end_date,
                "start_date": run_start_date,
                "market_regime": run_regime,
                "days_traded": step_count,
                "trading_years": trading_years_run,
                "final_value": run_final_value,
                "total_return": total_return_run,
                "annualized_return": annualized_return_run,
                "sharpe_ratio": metrics_run.get("sharpe_ratio", 0.0),
                "sortino_ratio": metrics_run.get("sortino_ratio", 0.0),
                "max_drawdown": metrics_run.get("max_drawdown_abs", 0.0),
                "volatility": metrics_run.get("volatility", 0.0),
                "turnover": metrics_run.get("turnover", 0.0),
                "turnover_pct": metrics_run.get("turnover", 0.0) * 100.0,
                "turnover_step_mean": turnover_step_mean,
                "turnover_step_p95": turnover_step_p95,
                "turnover_step_max": turnover_step_max,
                "turnover_target_step": turnover_target_step,
                "turnover_exceed_rate": turnover_exceed_rate,
                "turnover_excess_mean": turnover_excess_mean,
                "raw_turnover_step_mean": raw_turnover_step_mean,
                "executed_to_raw_turnover_ratio": executed_to_raw_turnover_ratio,
                "win_rate": metrics_run.get("win_rate", 0.0),
                "action_uniques": _count_unique_rows(run_actions),
                "alpha_le1_fraction": float(np.mean(run_alphas <= 1.0)) if run_alphas.size else 0.0,
                "argmax_alpha_uniques": int(np.unique(np.argmax(run_alphas, axis=1)).shape[0]) if run_alphas.size else 0,
                "mean_concentration_hhi": constraint_diag_run["mean_concentration_hhi"],
                "mean_top_weight": constraint_diag_run["mean_top_weight"],
                "mean_action_realization_l1": constraint_diag_run["mean_action_realization_l1"],
                "max_action_realization_l1": constraint_diag_run["max_action_realization_l1"],
            }
        )
        evaluation_rows.append(eval_row)

        stochastic_records.append(
            {
                "run": run_idx + 1,
                "seed": run_seed,
                "start_date": run_start_date,
                "market_regime": run_regime,
                "days_traded": step_count,
                "trading_years": trading_years_run,
                "final_value": portfolio_history_run[-1],
                "total_return": total_return_run,
                "annualized_return": annualized_return_run,
                "sharpe_ratio": metrics_run["sharpe_ratio"],
                "sortino_ratio": metrics_run["sortino_ratio"],
                "max_drawdown": metrics_run["max_drawdown_abs"],
                "volatility": metrics_run["volatility"],
                "turnover": metrics_run["turnover"],
                "turnover_pct": metrics_run["turnover"] * 100.0,
                "turnover_step_mean": turnover_step_mean,
                "turnover_step_p95": turnover_step_p95,
                "turnover_step_max": turnover_step_max,
                "turnover_target_step": turnover_target_step,
                "turnover_exceed_rate": turnover_exceed_rate,
                "turnover_excess_mean": turnover_excess_mean,
                "raw_turnover_step_mean": raw_turnover_step_mean,
                "executed_to_raw_turnover_ratio": executed_to_raw_turnover_ratio,
                "win_rate": metrics_run["win_rate"],
                "mean_concentration_hhi": constraint_diag_run["mean_concentration_hhi"],
                "mean_top_weight": constraint_diag_run["mean_top_weight"],
                "mean_action_realization_l1": constraint_diag_run["mean_action_realization_l1"],
                "max_action_realization_l1": constraint_diag_run["max_action_realization_l1"],
            }
        )

        stochastic_weights_list.append(run_weights)
        stochastic_actions_list.append(run_actions)
        stochastic_alphas_list.append(run_alphas)

        print(f"\n🎲 Run {run_idx + 1}/{num_eval_runs} (Seed={run_seed}):")
        print(f"   Start Date: {run_start_date} | Regime: {run_regime}")
        print(f"   Days Traded: {step_count} ({trading_years_run:.2f} years)")
        print(f"   Total Return: {total_return_run*100:+.2f}%")
        print(f"   Annualized Return: {annualized_return_run*100:+.2f}%")
        print(f"   Sharpe: {metrics_run['sharpe_ratio']:.4f}")
        print(f"   Max DD: {metrics_run['max_drawdown_abs']*100:.2f}%")
        print(
            "   Turnover (episode): "
            f"{metrics_run['turnover']*100:.2f}%"
        )
        print(
            "   Turnover (step): "
            f"mean={turnover_step_mean*100:.3f}% | "
            f"p95={turnover_step_p95*100:.3f}% | "
            f"max={turnover_step_max*100:.3f}%"
        )
        if turnover_target_step > 0.0:
            print(
                "   Turnover vs target: "
                f"target={turnover_target_step*100:.3f}% | "
                f"exceed_rate={turnover_exceed_rate*100:.1f}% | "
                f"mean_excess={turnover_excess_mean*100:.3f}%"
            )
        print(
            "   Raw/Executed turnover: "
            f"raw_mean={raw_turnover_step_mean*100:.3f}% | "
            f"executed/raw={executed_to_raw_turnover_ratio:.3f}"
        )

    df_stochastic = pd.DataFrame(stochastic_records)

    # Only print stochastic statistics if runs were performed
    if num_eval_runs > 0 and not df_stochastic.empty:
        print("\n" + "=" * 80)
        print("SUMMARY: STOCHASTIC EVALUATION STATISTICS")
        print("=" * 80)
        for label, column in [
            ("Total Return (%)", "total_return"),
            ("Annualized Return (%)", "annualized_return"),
            ("Sharpe Ratio (annualized)", "sharpe_ratio"),
        ]:
            print(f"\n{label}:")
            print(f"   Mean: {df_stochastic[column].mean()*100:+.2f}%" if "Return" in label else f"   Mean: {df_stochastic[column].mean():.4f}")
            print(f"   Std:  {df_stochastic[column].std()*100:.2f}%" if "Return" in label else f"   Std:  {df_stochastic[column].std():.4f}")
            print(
                f"   Min:  {df_stochastic[column].min()*100:+.2f}%"
                if "Return" in label
                else f"   Min:  {df_stochastic[column].min():.4f}"
            )
            print(
                f"   Max:  {df_stochastic[column].max()*100:+.2f}%"
                if "Return" in label
                else f"   Max:  {df_stochastic[column].max():.4f}"
            )

        print("\nMax Drawdown (%):")
        print(f"   Mean: {df_stochastic['max_drawdown'].mean()*100:.2f}%")
        print(f"   Std:  {df_stochastic['max_drawdown'].std()*100:.2f}%")
        print(f"   Min:  {df_stochastic['max_drawdown'].min()*100:.2f}%")
        print(f"   Max:  {df_stochastic['max_drawdown'].max()*100:.2f}%")

        print("\nTurnover (%):")
        print(f"   Mean: {df_stochastic['turnover'].mean()*100:.2f}%")
        print(f"   Std:  {df_stochastic['turnover'].std()*100:.2f}%")
        if "turnover_step_mean" in df_stochastic.columns:
            print("\nTurnover Step Detail (%):")
            print(f"   Mean(step mean): {df_stochastic['turnover_step_mean'].mean()*100:.3f}%")
            print(f"   Mean(step p95):  {df_stochastic['turnover_step_p95'].mean()*100:.3f}%")
            print(f"   Mean(step max):  {df_stochastic['turnover_step_max'].mean()*100:.3f}%")
            print(f"   Mean exceed rate: {df_stochastic['turnover_exceed_rate'].mean()*100:.1f}%")
            print(f"   Mean excess over target: {df_stochastic['turnover_excess_mean'].mean()*100:.3f}%")
            print(f"   Mean executed/raw ratio: {df_stochastic['executed_to_raw_turnover_ratio'].mean():.3f}")
    else:
        print("\n💡 Skipped stochastic evaluation (num_eval_runs=0)")

    eval_results_path = None
    eval_log_dir = results_root / "logs"
    if save_eval_logs:
        log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        eval_log_dir.mkdir(parents=True, exist_ok=True)
        selector_for_name = (model_family or "").strip().lower()
        if checkpoint_path_override:
            eval_name = f"exp6_custom_eval_{log_timestamp}.csv"
        elif selector_for_name in {"normal", "latest"}:
            eval_name = f"exp6_normal_eval_{log_timestamp}.csv"
        elif selector_for_name == "rare":
            eval_name = f"exp6_rare_eval_{log_timestamp}.csv"
        elif selector_for_name == "clip":
            eval_name = f"exp6_clip_ep{clip_episode}_eval_{log_timestamp}.csv"
        elif use_final_model:
            eval_name = f"exp6_final_eval_{log_timestamp}.csv"
        elif use_rare_model:
            eval_name = f"exp6_rare_ep{checkpoint_episode}_eval_{log_timestamp}.csv"
        else:
            eval_name = f"exp6_checkpoint_ep{checkpoint_episode}_eval_{log_timestamp}.csv"
        eval_results_path = eval_log_dir / eval_name
        df_eval_log = pd.DataFrame(evaluation_rows)
        if not df_eval_log.empty:
            df_eval_log = df_eval_log.reindex(columns=EVALUATION_FIELDNAMES)
        else:
            df_eval_log = pd.DataFrame(columns=EVALUATION_FIELDNAMES)
        df_eval_log.to_csv(eval_results_path, index=False)
        print(f"\n💾 Evaluation results saved: {eval_results_path}")

        # Per-track artifacts
        file_stem = Path(eval_results_path).stem
        for track, out in deterministic_track_outputs.items():
            _save_track_artifacts(
                eval_log_dir,
                file_stem,
                track=track,
                weights=out["weights"],
                actions=out["actions"],
                alphas=out["alphas"],
            )

        if stochastic_weights_list:
            sto_weights = np.concatenate(stochastic_weights_list, axis=0)
            sto_actions = np.concatenate(stochastic_actions_list, axis=0) if stochastic_actions_list else np.array([])
            sto_alphas = np.concatenate(stochastic_alphas_list, axis=0) if stochastic_alphas_list else np.array([])

            run_ids_weights = []
            for idx, arr in enumerate(stochastic_weights_list, start=1):
                run_ids_weights.extend([idx] * len(arr))
            run_ids_actions = []
            for idx, arr in enumerate(stochastic_actions_list, start=1):
                run_ids_actions.extend([idx] * len(arr))
            run_ids_alphas = []
            for idx, arr in enumerate(stochastic_alphas_list, start=1):
                run_ids_alphas.extend([idx] * len(arr))

            _save_track_artifacts(
                eval_log_dir,
                file_stem,
                track="stochastic",
                weights=sto_weights,
                actions=sto_actions,
                alphas=sto_alphas,
                run_ids=run_ids_weights if len(run_ids_weights) == len(sto_weights) else None,
            )
            # Ensure run_id alignment for action/alpha artifacts by rewriting with dedicated run id columns.
            if sto_actions.size > 0:
                a_cols = _weights_columns(sto_actions.shape[1])
                df_a = pd.DataFrame(sto_actions, columns=a_cols if a_cols else None)
                if len(run_ids_actions) == len(df_a):
                    df_a.insert(0, "run", run_ids_actions)
                df_a.insert(0, "step", np.arange(len(df_a)))
                df_a.to_csv(eval_log_dir / f"{file_stem}_actions_stochastic.csv", index=False)
            if sto_alphas.size > 0:
                alpha_cols = [f"alpha_{i}" for i in range(sto_alphas.shape[1])]
                df_alpha = pd.DataFrame(sto_alphas, columns=alpha_cols)
                if len(run_ids_alphas) == len(df_alpha):
                    df_alpha.insert(0, "run", run_ids_alphas)
                df_alpha.insert(0, "step", np.arange(len(df_alpha)))
                df_alpha.to_csv(eval_log_dir / f"{file_stem}_alphas_stochastic.csv", index=False)

        print(f"💾 Per-track artifacts saved in: {eval_log_dir}")

    return Experiment6Evaluation(
        actor_weights_path=actor_weights_path,
        critic_weights_path=critic_weights_path,
        deterministic_metrics=metrics_det,
        deterministic_portfolio=portfolio_history,
        deterministic_weights=deterministic_weights_array,
        deterministic_actions=deterministic_actions_array,
        stochastic_results=df_stochastic,
        stochastic_weights=stochastic_weights_list,
        stochastic_actions=stochastic_actions_list,
        deterministic_alphas=deterministic_alphas_array,
        stochastic_alphas=stochastic_alphas_list,
        eval_results_path=str(eval_results_path) if eval_results_path else "",
        checkpoint_description=checkpoint_description,
        agent=agent_eval,
        env_test_deterministic=env_test_deterministic,
        env_test_random=env_test_random,
    )


def build_evaluation_track_summary(
    evaluation: Experiment6Evaluation,
    *,
    include_stochastic_mean: bool = True,
) -> pd.DataFrame:
    """
    Build a compact summary table from Experiment6Evaluation attributes only.
    """
    rows: List[Dict[str, Any]] = []
    det = evaluation.deterministic_metrics or {}
    rows.append(
        {
            "eval_track": "deterministic",
            "sharpe_ratio": det.get("sharpe_ratio"),
            "sortino_ratio": det.get("sortino_ratio"),
            "max_drawdown": det.get("max_drawdown_abs"),
            "volatility": det.get("volatility"),
            "turnover": det.get("turnover"),
            "win_rate": det.get("win_rate"),
            "total_return": det.get("total_return"),
            "annualized_return": det.get("annualized_return"),
        }
    )

    if include_stochastic_mean and isinstance(evaluation.stochastic_results, pd.DataFrame) and not evaluation.stochastic_results.empty:
        s = evaluation.stochastic_results
        rows.append(
            {
                "eval_track": "stochastic_mean",
                "sharpe_ratio": s.get("sharpe_ratio", pd.Series(dtype=float)).mean(),
                "sortino_ratio": s.get("sortino_ratio", pd.Series(dtype=float)).mean(),
                "max_drawdown": s.get("max_drawdown", pd.Series(dtype=float)).mean(),
                "volatility": s.get("volatility", pd.Series(dtype=float)).mean(),
                "turnover": s.get("turnover", pd.Series(dtype=float)).mean(),
                "win_rate": s.get("win_rate", pd.Series(dtype=float)).mean(),
                "total_return": s.get("total_return", pd.Series(dtype=float)).mean(),
                "annualized_return": s.get("annualized_return", pd.Series(dtype=float)).mean(),
            }
        )
    return pd.DataFrame(rows)


def build_ablation_table(
    evaluations: Dict[str, Experiment6Evaluation],
) -> pd.DataFrame:
    """
    Build an ablation-style table keyed by experiment label.
    """
    rows: List[Dict[str, Any]] = []
    for label, evaluation in evaluations.items():
        det = evaluation.deterministic_metrics or {}
        sto = evaluation.stochastic_results if isinstance(evaluation.stochastic_results, pd.DataFrame) else pd.DataFrame()
        rows.append(
            {
                "label": label,
                "checkpoint_description": evaluation.checkpoint_description,
                "det_sharpe": det.get("sharpe_ratio"),
                "det_sortino": det.get("sortino_ratio"),
                "det_max_drawdown": det.get("max_drawdown_abs"),
                "det_volatility": det.get("volatility"),
                "det_turnover": det.get("turnover"),
                "sto_mean_sharpe": sto["sharpe_ratio"].mean() if "sharpe_ratio" in sto else np.nan,
                "sto_std_sharpe": sto["sharpe_ratio"].std() if "sharpe_ratio" in sto else np.nan,
                "sto_mean_return": sto["total_return"].mean() if "total_return" in sto else np.nan,
                "sto_mean_max_drawdown": sto["max_drawdown"].mean() if "max_drawdown" in sto else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(by="det_sharpe", ascending=False, na_position="last")


def build_weight_path_frames(
    evaluation: Experiment6Evaluation,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return deterministic and stochastic-average weight paths for plotting.
    """
    det_weights = np.array(evaluation.deterministic_weights)
    if det_weights.ndim != 2:
        det_df = pd.DataFrame()
    else:
        det_df = pd.DataFrame(det_weights)
        det_df.insert(0, "step", np.arange(len(det_df)))

    if not evaluation.stochastic_weights:
        sto_df = pd.DataFrame()
    else:
        min_len = min(arr.shape[0] for arr in evaluation.stochastic_weights if arr.ndim == 2)
        aligned = [arr[:min_len] for arr in evaluation.stochastic_weights if arr.ndim == 2]
        mean_sto = np.mean(np.stack(aligned, axis=0), axis=0) if aligned else np.array([])
        if isinstance(mean_sto, np.ndarray) and mean_sto.ndim == 2:
            sto_df = pd.DataFrame(mean_sto)
            sto_df.insert(0, "step", np.arange(len(sto_df)))
        else:
            sto_df = pd.DataFrame()
    return det_df, sto_df


def compare_agent_vs_baseline(
    evaluation: Experiment6Evaluation,
    baseline_returns: pd.Series,
) -> Dict[str, float]:
    """
    Compare deterministic agent returns with a baseline return series.

    baseline_returns should be daily simple returns aligned to evaluation period length.
    """
    portfolio = np.array(evaluation.deterministic_portfolio)
    if len(portfolio) < 2:
        raise ValueError("deterministic_portfolio is too short for return comparison.")
    agent_returns = pd.Series(np.diff(portfolio) / portfolio[:-1]).dropna()
    baseline = baseline_returns.reset_index(drop=True).astype(float)
    min_len = min(len(agent_returns), len(baseline))
    if min_len == 0:
        raise ValueError("No overlapping samples between agent and baseline returns.")
    agent_returns = agent_returns.iloc[:min_len]
    baseline = baseline.iloc[:min_len]

    def _sharpe(returns: pd.Series) -> float:
        std = float(returns.std(ddof=1))
        if std == 0.0:
            return 0.0
        return float(np.sqrt(252.0) * returns.mean() / std)

    return {
        "agent_sharpe": _sharpe(agent_returns),
        "baseline_sharpe": _sharpe(baseline),
        "agent_mean_return": float(agent_returns.mean()),
        "baseline_mean_return": float(baseline.mean()),
        "agent_volatility": float(agent_returns.std(ddof=1) * np.sqrt(252.0)),
        "baseline_volatility": float(baseline.std(ddof=1) * np.sqrt(252.0)),
    }


def preflight_checkpoint_loadability(
    *,
    checkpoint_prefixes: Iterable[Union[str, Path]],
    phase1_data: Phase1Dataset,
    config: Dict[str, Any],
    random_seed: int = 42,
    use_covariance: bool = True,
    architecture: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fast compatibility scan: build agent/env and test checkpoint load only.

    Useful before full evaluation to skip incompatible checkpoints.
    """
    rows: List[Dict[str, Any]] = []
    cfg = copy.deepcopy(config)
    inferred_arch = str(
        architecture
        or cfg.get("agent_params", {}).get("actor_critic_type", "TCN")
    ).upper()
    base_agent_params = copy.deepcopy(cfg.get("agent_params", {}))

    for raw_prefix in checkpoint_prefixes:
        prefix = str(raw_prefix)
        actor_path = f"{prefix}_actor.weights.h5"
        critic_path = f"{prefix}_critic.weights.h5"
        hints = _infer_checkpoint_architecture_hints(
            actor_path,
            fallback_architecture=inferred_arch,
            fallback_use_attention=bool(base_agent_params.get("use_attention", False)),
            fallback_use_fusion=bool(base_agent_params.get("use_fusion", False)),
        )
        row: Dict[str, Any] = {
            "checkpoint_prefix": prefix,
            "actor_path": actor_path,
            "critic_path": critic_path,
            "exists_actor": Path(actor_path).exists(),
            "exists_critic": Path(critic_path).exists(),
            "inferred_architecture": hints.get("actor_critic_type"),
            "inferred_use_attention": hints.get("use_attention"),
            "inferred_use_fusion": hints.get("use_fusion"),
            "inference_source": hints.get("source"),
            "compatible": False,
            "error_type": "",
            "error_message": "",
        }
        if not row["exists_actor"] or not row["exists_critic"]:
            row["error_type"] = "FileNotFoundError"
            row["error_message"] = "Missing actor/critic weights file"
            rows.append(row)
            continue

        stub_agent_params = copy.deepcopy(base_agent_params)
        stub_agent_params["actor_critic_type"] = hints["actor_critic_type"]
        stub_agent_params["use_attention"] = bool(hints["use_attention"])
        stub_agent_params["use_fusion"] = bool(hints["use_fusion"])

        stub = create_experiment6_result_stub(
            random_seed=random_seed,
            use_covariance=use_covariance,
            architecture=str(hints["actor_critic_type"]).upper(),
            checkpoint_path=prefix,
            agent_config=stub_agent_params,
            base_agent_params=None,
        )
        try:
            _ = evaluate_experiment6_checkpoint(
                experiment6=stub,
                phase1_data=phase1_data,
                config=cfg,
                random_seed=random_seed,
                checkpoint_path_override=prefix,
                num_eval_runs=0,
                deterministic_eval_mode="mean",
                save_eval_logs=False,
                save_eval_artifacts=False,
                load_only=True,
            )
            row["compatible"] = True
        except Exception as exc:
            row["error_type"] = type(exc).__name__
            row["error_message"] = str(exc)
        rows.append(row)

    return pd.DataFrame(rows)


def load_run_checkpoint_prefixes_from_metadata(
    metadata_path: Union[str, Path],
    *,
    results_root: Optional[Union[str, Path]] = None,
    allowed_types: Optional[Iterable[str]] = None,
    require_both_files: bool = True,
) -> List[str]:
    """
    Read run-scoped checkpoint records from metadata and return checkpoint prefixes.

    This avoids accidentally evaluating stale checkpoints from previous runs that
    happen to coexist in the same results directory.
    """
    meta_path = Path(metadata_path)
    if not meta_path.exists():
        return []

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return []

    checkpointing = payload.get("Checkpointing", {}) if isinstance(payload, dict) else {}
    records = checkpointing.get("saved_checkpoints_for_this_run", [])
    if not isinstance(records, list):
        return []

    allowed = {str(t) for t in allowed_types} if allowed_types is not None else None
    # Backward-compatible aliasing: deterministic validation checkpoints are high-watermark-style.
    allowed_aliases = {
        "high_watermark": {"high_watermark", "deterministic_validation_high_watermark"},
        "deterministic_validation_high_watermark": {"deterministic_validation_high_watermark", "high_watermark"},
    }
    root = Path(results_root) if results_root is not None else None

    expanded_allowed: Optional[set] = None
    if allowed is not None:
        expanded_allowed = set()
        for token in allowed:
            expanded_allowed.update(allowed_aliases.get(token, {token}))

    def _remap_to_root(path_obj: Path, root_dir: Path) -> Path:
        if path_obj.exists():
            return path_obj
        parts = list(path_obj.parts)
        for anchor_name in ("tcn_fusion_results", "tcn_att_results", "tcn_results"):
            if anchor_name in parts:
                idx = parts.index(anchor_name)
                rel_tail = Path(*parts[idx + 1 :]) if idx + 1 < len(parts) else Path()
                candidate = root_dir / rel_tail
                if candidate.exists():
                    return candidate
        by_name = root_dir / path_obj.name
        return by_name

    prefixes: List[str] = []
    seen: set = set()
    for rec in records:
        if not isinstance(rec, dict):
            continue
        rec_type = str(rec.get("type", ""))
        if expanded_allowed is not None and rec_type not in expanded_allowed:
                continue
        actor_path = rec.get("actor_path")
        critic_path = rec.get("critic_path")
        if not actor_path or not critic_path:
            continue
        actor = Path(actor_path)
        critic = Path(critic_path)
        if root is not None:
            actor = _remap_to_root(actor, root)
            critic = _remap_to_root(critic, root)
        if require_both_files and (not actor.exists() or not critic.exists()):
            continue
        prefix = str(actor).replace("_actor.weights.h5", "")
        if prefix in seen:
            continue
        seen.add(prefix)
        prefixes.append(prefix)

    return prefixes
