# Implementation Traceback Log — TAPE Reward Overhaul (2026-02-14)

## Scope
This log captures the applied fixes for:
- Replacing the Gaussian utility function with an asymmetric sigmoid to eliminate vanishing gradients and overperformance penalties.
- Replacing the symmetric turnover proximity reward (which rewarded trading) with a one-sided soft ceiling (industry-aligned, penalty-only).
- Reducing penalty scalars (concentration, top weight, drawdown) that were overwhelming the base reward signal.
- Fixing MDD direction semantics (stored as negative → `'increasing'`, not `'decreasing'`).
- Recalibrating turnover ceiling from 0.02 (wrong scale) to 0.50 (calibrated to archive behavior).
- Ensuring backward compatibility with legacy Gaussian profiles.

## Motivation / Problem Statement
The previous TAPE system had several interacting issues:

1. **Gaussian utility function** (`exp(-(x-μ)²/2σ²)`) produced:
   - Near-zero utility for early-training agents (Sharpe=0.1 → utility ≈ 0.01), giving no learning signal.
   - Penalties for exceeding targets (Sharpe=2.0 → utility = 0.14), discouraging excellence.
   - A near-binary TAPE score (≈0 or ≈1) with no useful gradient in between.

2. **Turnover proximity** rewarded the agent for hitting a specific trading frequency, conflicting with transaction costs in the base return that discourage unnecessary trading.

3. **Penalty stacking**: concentration (12.0), top weight (8.0), and drawdown (λ_max=15.0, coef=7.5) scalars created combined penalties that overwhelmed the portfolio return signal.

4. **Turnover ceiling miscalibration**: ceiling at 0.02 (2%) per step, but actual per-step turnover averaged ~0.54 (54%), causing permanent punishment on every step.

## Files Updated

### 1) `src/reward_utils.py`

#### New function: `asymmetric_sigmoid_utility()`
Added at line 408. Core sigmoid utility replacing the Gaussian:
```
U(x) = 1 / (1 + exp(-k × (x - mu)))
```
- `k = k_minus` when x is on the penalty side of mu
- `k = k_plus` when x is on the reward side of mu
- `direction` parameter: `'increasing'` (higher = better) or `'decreasing'` (lower = better)
- Bounds truncation (`a`, `b`) preserved for safety
- Numerical stability: exponent clipped to [-20, 20]

Parameters:
- `x: float` — metric value
- `mu: float` — sigmoid midpoint (50% utility)
- `k_minus: float` — steepness below mu
- `k_plus: float` — steepness above mu
- `a, b: float` — truncation bounds
- `direction: str` — `'increasing'` or `'decreasing'`

#### Legacy function: `skewed_utility_function()` (retained)
- Renamed docstring to `[LEGACY]`
- Kept for backward compatibility with any old profiles using `sigma_sq_minus`/`sigma_sq_plus`
- Body unchanged

#### Updated function: `calculate_tape_score()`
- Auto-detects profile format:
  - New: `k_minus`/`k_plus` → calls `asymmetric_sigmoid_utility()`
  - Legacy: `sigma_sq_minus`/`sigma_sq_plus` → calls `skewed_utility_function()`
- Reads `directions` from profile (defaults to `['increasing'] * N` if absent)
- Passes `direction` to sigmoid for correct score orientation
- Weight normalization and metric key mapping unchanged

#### Updated function: `test_utility_function()`
- Now tests asymmetric sigmoid with side-by-side Gaussian comparison
- Tests MDD as `'increasing'` direction (less-negative = better)
- Fixed Unicode characters (box-drawing `─`) to ASCII dashes for Windows compatibility

### 2) `src/config.py`

#### Profile rewrites (lines 219–254)

All three profiles rewritten from Gaussian (`sigma_sq_minus`/`sigma_sq_plus`) to sigmoid (`k_minus`/`k_plus`):

**PROFILE_BALANCED_GROWTH:**
| Parameter | Old | New |
|-----------|-----|-----|
| `sigma_sq_minus` | `[0.09, 0.16, 0.0025, 0.04, 0.02]` | Removed |
| `sigma_sq_plus` | `[0.25, 0.36, 0.01, 0.01, 0.04]` | Removed |
| `k_minus` | — | `[4.0, 3.0, 5.0, 1.0, 2.0]` |
| `k_plus` | — | `[1.0, 1.0, 1.0, 4.0, 1.0]` |
| `weights` | `[0.35, 0.25, 0.25, 0.10, 0.05]` | `[0.30, 0.25, 0.25, 0.15, 0.05]` |
| `directions` (MDD) | `'decreasing'` | `'increasing'` |

**PROFILE_AGGRESSIVE_ALPHA_SEEKER:**
| Parameter | Old | New |
|-----------|-----|-----|
| `sigma_sq_minus` | `[0.09, 0.16, 0.01, 2.0, 0.0025]` | Removed |
| `sigma_sq_plus` | `[1.0, 1.44, 0.0225, 4.0, 0.16]` | Removed |
| `k_minus` | — | `[5.0, 4.0, 3.0, 0.5, 3.0]` |
| `k_plus` | — | `[0.5, 0.5, 0.5, 2.0, 0.5]` |
| `directions` (MDD) | `'decreasing'` | `'increasing'` |

**PROFILE_CAPITAL_PRESERVATION:**
| Parameter | Old | New |
|-----------|-----|-----|
| `sigma_sq_minus` | `[0.01, 0.0225, 0.0004, 0.5, 0.0225]` | Removed |
| `sigma_sq_plus` | `[0.09, 0.16, 0.001, 1.0, 0.0225]` | Removed |
| `k_minus` | — | `[3.0, 2.5, 8.0, 0.5, 2.0]` |
| `k_plus` | — | `[2.0, 2.0, 2.0, 3.0, 1.5]` |
| `directions` (MDD) | `'decreasing'` | `'increasing'` |

**MDD direction fix rationale:**
MDD is stored as negative (e.g., `-0.15` = 15% drawdown). Numerically: `-0.05 > -0.15 > -0.25`. Less-negative = less drawdown = better. Therefore the utility should INCREASE as the value increases → `'increasing'`, not `'decreasing'`.

#### PHASE1 penalty scalar reductions (lines 307–332)
| Parameter | Old | New | Rationale |
|-----------|-----|-----|-----------|
| `target_turnover` | `0.02` | `0.50` | Calibrated to archive avg (~0.54) |
| `concentration_penalty_scalar` | `12.0` | `4.0` | Was overwhelming base return |
| `top_weight_penalty_scalar` | `8.0` | `3.0` | Was overwhelming base return |
| `drawdown_constraint.penalty_coef` | `7.5` | `3.0` | Stacked too aggressively |
| `drawdown_constraint.lambda_max` | `15.0` | `5.0` | λ ran away to extremely high values |

#### PHASE2 penalty scalar reductions (lines 535–573)
Same reductions as PHASE1 applied to:
- `concentration_penalty_scalar: 12.0 -> 4.0`
- `top_weight_penalty_scalar: 8.0 -> 3.0`
- `target_turnover: 0.02 -> 0.50`
- `drawdown_constraint.penalty_coef: 7.5 -> 3.0`
- `drawdown_constraint.lambda_max: 15.0 -> 5.0`
- `drawdown_constraint_overrides.sequential.penalty_coef: 7.5 -> 3.0`
- `drawdown_constraint_overrides.sequential.lambda_max: 15.0 -> 5.0`

### 3) `src/environment_tape_rl.py`

#### Profile validation update (line 197–207)
Previous: required `['mu', 'sigma_sq_minus', 'sigma_sq_plus', 'weights', 'a_bounds', 'b_bounds', 'metrics_order']`

New: requires `['mu', 'weights', 'a_bounds', 'b_bounds', 'metrics_order']` + either:
- `k_minus` and `k_plus` (sigmoid), OR
- `sigma_sq_minus` and `sigma_sq_plus` (legacy Gaussian)

Raises `ValueError` if neither pair is present.

#### Component 3 rewrite: turnover proximity → one-sided ceiling (lines 703–720)

Previous (symmetric proximity — non-standard):
```python
deviation = abs(turnover - target)
if deviation <= allowed_deviation:
    turnover_reward = proximity * scalar   # POSITIVE reward for trading
else:
    turnover_reward = -excess * scalar     # Penalty for over OR under
```

New (one-sided ceiling — industry-aligned):
```python
if turnover > ceiling:
    excess_ratio = (turnover - ceiling) / ceiling
    turnover_reward = -excess_ratio * scalar   # Penalty ONLY for excess
else:
    turnover_reward = 0.0                      # No reward, no penalty
```

Rationale: Transaction costs in the base return already discourage unnecessary trading. The ceiling is a safety rail for extreme churning, not an incentive to trade at a specific rate. Aligns with Grinold & Kahn, Gârleanu & Pedersen.

#### Docstring updates
- `target_turnover`: documented as ceiling, not target
- `turnover_penalty_scalar`: documented as excess-only penalty
- `turnover_target_band`: marked as DEPRECATED
- `_get_reward()` Component 3 description updated

#### Logging update (line 230)
- `"Component 3: Turnover Penalty (target=…, band=…)"` → `"Component 3: Turnover Ceiling (max=…, penalty_scalar=…)"`

### 4) `src/profile_manager.py`

#### Import update (lines 36, 42)
Added `asymmetric_sigmoid_utility` to both import paths:
```python
from reward_utils import calculate_rolling_performance, asymmetric_sigmoid_utility
```

#### Adaptive scoring update: `_select_adaptive_profile()` (lines 466–530)
Previous: inline Gaussian scoring only:
```python
sigma = sqrt(sigma_plus[idx]) if value >= target else sqrt(sigma_minus[idx])
component_score = exp(-0.5 * ((value - target) / sigma)^2)
```

New: auto-detects profile format and branches:
- Sigmoid profiles (`k_minus`/`k_plus`): calls `asymmetric_sigmoid_utility()` with direction, bounds
- Legacy profiles (`sigma_sq_minus`/`sigma_sq_plus`): uses original Gaussian scoring

## Validation Performed

1. **Syntax check**: `ast.parse()` passed for `environment_tape_rl.py`
2. **Import check**: Successfully imported `config`, `reward_utils`, `profile_manager` with new changes
3. **Utility function test** (`test_utility_function()`):
   - Sigmoid vs Gaussian comparison — sigmoid provides 663× more signal at Sharpe=-0.5 and 5.4× at Sharpe=2.0
   - MDD direction verified: MDD=-0.05 → 0.5250, MDD=-0.15 → 0.5000, MDD=-0.25 → 0.3775
4. **TAPE score test** (`test_tape_score()`):
   - Score at exact targets = 0.5000 (perfect sigmoid midpoint)
   - Score produces smooth gradient across agent quality levels:
     - Early random: 0.1755
     - Archive baseline: 0.3946
     - Good trained: 0.5374
     - Excellent: 0.6385
5. **Config consistency**: PHASE1 and PHASE2 penalty scalars verified aligned

## Expected Behavioral Impact

- **Early training**: TAPE score ~0.18 instead of ~0.01 — agent receives meaningful terminal bonus from first episodes, accelerating learning.
- **Excellent episodes**: TAPE score rises continuously (0.54 → 0.64+) instead of collapsing due to overperformance penalty. Agent always has incentive to improve.
- **Turnover**: Agent no longer rewarded for trading; only penalized for extreme churning (>50% per step). Normal trading behavior (30-50%) is unaffected.
- **Penalty balance**: Reduced concentration/drawdown/top-weight penalties allow the base return signal to dominate during early training, with penalties serving as guardrails rather than primary drivers.
- **MDD scoring**: Correctly rewards less drawdown (less-negative values score higher).

## Notes for Reproducibility
- Use fresh kernel/session before training to avoid stale cached objects.
- Re-run data preparation cells before new training passes.
- The `turnover_target_band` parameter is now deprecated but harmless — it is still present in configs but no longer read by Component 3 logic.
- Legacy profiles using `sigma_sq_minus`/`sigma_sq_plus` will still work everywhere (auto-detected).

## Rollback Guidance
If needed, rollback by file-level Git restore in this order:
1. `src/reward_utils.py` — revert to Gaussian-only utility
2. `src/config.py` — restore `sigma_sq_minus`/`sigma_sq_plus` profiles and old penalty scalars
3. `src/environment_tape_rl.py` — restore symmetric proximity and old validation
4. `src/profile_manager.py` — restore Gaussian-only adaptive scoring

Then re-run a short (1k–2k step) sanity training and stochastic eval smoke test.
