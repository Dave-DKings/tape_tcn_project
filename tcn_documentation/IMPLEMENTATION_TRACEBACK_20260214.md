# Implementation Traceback Log (2026-02-14)

## Scope
This log captures the applied fixes for:
- PPO stability and reduced aggressiveness.
- Action-execution mismatch reduction.
- Explicit anti-concentration controls.
- Stronger drawdown influence (including regime-aware scaling).
- Evaluation logging upgrades (stochastic start date + concentration/L1 diagnostics).
- Data/analysis date window and TCN capacity updates.

## Files Updated

### 1) `src/config.py`

#### Data window settings
- `DATA_FETCH_START_DATE`: `2006-01-01`
- `DATA_FETCH_END_DATE`: `2025-12-31`
- `ANALYSIS_START_DATE`: `2008-01-01`
- `ANALYSIS_END_DATE`: `2025-12-31`

#### TCN model capacity
- Updated TCN filters for current TCN variants:
- `tcn_filters: [64, 128, 128]`

#### PPO aggressiveness reduction
- `policy_clip: 0.2 -> 0.15`
- `num_ppo_epochs: 5 -> 3`
- `actor_lr: 0.0007 -> 0.0003`
- `critic_lr: 0.0007 -> 0.0005`
- Added `target_kl: 0.03`

#### Concentration + action-realization controls
Added in `environment_params`:
- `concentration_penalty_scalar: 12.0`
- `concentration_target_hhi: 0.14`
- `top_weight_penalty_scalar: 8.0`
- `target_top_weight: 0.22`
- `action_realization_penalty_scalar: 2.0`

#### Drawdown influence strengthening + regime scaling
- `drawdown_constraint.penalty_coef: 5.0 -> 7.5`
- Added `dd_regime_scaling`:
- `enabled: true`
- `vol_window: 21`
- `low_vol_threshold: 0.12`
- `high_vol_threshold: 0.25`
- `low_mult: 0.90`
- `mid_mult: 1.00`
- `high_mult: 1.35`

### 2) `src/agents/ppo_agent_tf.py`

#### KL early-stop safeguard
Added PPO controls:
- `target_kl`
- `kl_stop_multiplier`
- `minibatches_before_kl_stop`

Update loop behavior:
- Detects KL overshoot (`approx_kl > target_kl * kl_stop_multiplier`) after minimum minibatches.
- Triggers early stop within PPO update epoch loop.
- Logs warning and records stats:
- `early_stop_kl_triggered`
- `early_stop_kl`
- `early_stop_epoch`

### 3) `src/environment_tape_rl.py`

#### Action-execution mismatch fix (highest-priority logic)
Replaced previous ad-hoc post-processing with direct feasible projection:
- Added `_project_weights_to_constraints(...)`
- Enforces in one projection pass:
- non-negativity
- sum-to-one
- max risky-asset cap
- minimum cash floor
- Reduces large action-to-realized distortions and stabilizes execution behavior.

#### Concentration and mismatch accounting
Per-step diagnostics and penalties now integrated consistently:
- concentration HHI tracking
- top weight tracking
- action-realization L1 tracking
- cumulative concentration penalty
- cumulative action-realization penalty

#### Drawdown regime-scaling consistency
- Added `_get_drawdown_regime_multiplier()` using rolling realized volatility.
- Updated `_apply_drawdown_dual_controller()` to return:
- `(penalty, current_drawdown, excess, regime_multiplier)`
- Drawdown penalty now multiplies by regime factor in high-vol states.
- Fixed function return signature mismatch and runtime consistency.

### 4) `src/notebook_helpers/tcn_phase1.py`

#### Evaluation CSV schema expanded
`EVALUATION_EXTRA_FIELDNAMES` now includes:
- `start_date`
- `market_regime`
- `mean_concentration_hhi`
- `mean_top_weight`
- `mean_action_realization_l1`
- `max_action_realization_l1`

#### New eval diagnostics helper
- Added `_constraint_diagnostics_from_env(env)`.
- Reads histories from environment:
- `concentration_hhi_history`
- `top_weight_history`
- `action_realization_l1_history`

#### Deterministic/stochastic row enrichment
Evaluation rows now persist:
- per-run stochastic `start_date`
- market regime label
- concentration diagnostics
- action-realization L1 diagnostics

This ensures diagnostics are no longer dropped during `reindex(columns=EVALUATION_FIELDNAMES)`.

## Validation Performed
- Syntax checks executed successfully:
- `src/environment_tape_rl.py`
- `src/notebook_helpers/tcn_phase1.py`
- `src/agents/ppo_agent_tf.py`
- `src/config.py`

## Expected Behavioral Impact
- Lower PPO instability and reduced clipping/KL runaway risk.
- Better alignment between action intent and executed portfolio weights.
- Reduced structural concentration pressure (explicitly penalized).
- Stronger drawdown discipline, especially in high-volatility regimes.
- Richer evaluation logs for diagnostics, OOS analysis, and write-up traceability.

## Notes for Reproducibility
- Use fresh kernel/session before training/evaluation to avoid stale objects.
- Re-run data preparation + split cells before new training/eval passes.
- Ensure evaluation exports are regenerated from updated helpers to include new columns.

## Rollback Guidance
If needed, rollback by file-level Git restore in this order:
1. `src/environment_tape_rl.py`
2. `src/agents/ppo_agent_tf.py`
3. `src/notebook_helpers/tcn_phase1.py`
4. `src/config.py`

Then re-run a short (1k–2k step) sanity training and deterministic/stochastic eval smoke test.
