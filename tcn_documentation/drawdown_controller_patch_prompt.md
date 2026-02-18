# Drawdown Dual Controller — Patch Request

I need you to generate a patch for my RL portfolio environment's drawdown dual controller. There are 3 fixes needed across 2 files.

---

## FIX 1: λ should carry forward between episodes (with decay)

**File:** `src/environment_tape_rl.py`  
**Method:** `_reset_drawdown_controller_state` (lines 620-633)

The current code resets `self.drawdown_lambda` back to `lambda_init` (0.50) every episode. This wastes the learned penalty signal — the agent has to re-discover how severe drawdowns are from scratch each episode.

**Instead:** Carry forward λ from the previous episode but apply an exponential decay:
```python
self.drawdown_lambda = max(self.drawdown_lambda_floor, self.drawdown_lambda * self.drawdown_lambda_carry_decay)
```
This preserves memory of past drawdowns but allows recovery.

**Current code to modify:**
```python
def _reset_drawdown_controller_state(self) -> None:
    """Reset dual-controller stats at episode start."""
    self.running_peak = self.initial_balance
    if self.drawdown_constraint_enabled:
        self.drawdown_lambda = max(self.drawdown_lambda_init, self.drawdown_lambda_floor)  # ← CHANGE THIS LINE
    else:
        self.drawdown_lambda = 0.0
    self.drawdown_lambda_peak = self.drawdown_lambda
    self.drawdown_penalty_sum = 0.0
    self.drawdown_excess_accumulator = 0.0
    self.current_drawdown = 0.0
    self.drawdown_triggered = False
    if self.drawdown_constraint_enabled:
        self.drawdown_trigger_boundary = max(0.0, self.drawdown_target + self.drawdown_tolerance)
```

**New init param needed:** In `__init__`, read `drawdown_lambda_carry_decay` from the drawdown_constraint config dict and store as `self.drawdown_lambda_carry_decay`. Default to 0.7 if not present.

The `__init__` method reads drawdown config starting around line 350-380. Look for where `self.drawdown_lambda_max`, `self.drawdown_lambda_floor`, etc. are assigned and add:
```python
self.drawdown_lambda_carry_decay = float(dd_cfg.get("lambda_carry_decay", 0.7))
```

---

## FIX 2: Set lambda_floor to 0.0 (no penalty when doing well)

**File:** `src/config.py`  
**Location:** `drawdown_constraint` dict inside `environment_params` (around line 325-338)

The current `lambda_floor` of 0.50 means the agent ALWAYS faces a baseline drawdown penalty even when drawdown is 0%. This discourages the agent from learning because it can never fully escape the penalty.

**Change:**
```diff
 "drawdown_constraint": {
     "enabled": True,
     "target": 0.18,
     "penalty_coef": 3.0,
     "dual_learning_rate": 0.40,
     "lambda_init": 0.50,
-    "lambda_floor": 0.50,
+    "lambda_floor": 0.0,
     "lambda_max": 5.0,
     "tolerance": -0.015,
     "penalty_reference": "trigger_boundary",
     "cooling_rate": 0.35,
+    "lambda_carry_decay": 0.7,
 }
```

---

## FIX 3: Reduce penalty_coef default to 1.5

**File:** `src/config.py`  
**Location:** Same `drawdown_constraint` dict (around line 328)

The current penalty can reach:
```
penalty = coef × λ_max × excess × regime_mult
        = 3.0  × 5.0   × 0.30  × 1.35
        = 6.075 per step
```

The base return signal is only ~0.005 per step. A penalty of 6.075 is **1000x stronger** than the reward signal — this drowns out all learning about returns.

**Change:**
```diff
-    "penalty_coef": 3.0,
+    "penalty_coef": 1.5,
```

With `penalty_coef=1.5`, the max penalty becomes `1.5 × 5.0 × 0.30 × 1.35 = 3.04`, which is still strong but allows the return signal to influence the gradient.

---

## Summary of All Changes

| File | Location | What | Old | New |
|:-----|:---------|:-----|:----|:----|
| `environment_tape_rl.py` | `__init__` (~line 360) | Add new param read | N/A | `self.drawdown_lambda_carry_decay = float(dd_cfg.get("lambda_carry_decay", 0.7))` |
| `environment_tape_rl.py` | `_reset_drawdown_controller_state` (line 623) | Decayed carry-forward instead of hard reset | `self.drawdown_lambda = max(self.drawdown_lambda_init, self.drawdown_lambda_floor)` | `self.drawdown_lambda = max(self.drawdown_lambda_floor, self.drawdown_lambda * self.drawdown_lambda_carry_decay)` |
| `config.py` | `drawdown_constraint` (line 328) | Reduce penalty coefficient | `"penalty_coef": 3.0` | `"penalty_coef": 1.5` |
| `config.py` | `drawdown_constraint` (line 331) | Remove permanent floor penalty | `"lambda_floor": 0.50` | `"lambda_floor": 0.0` |
| `config.py` | `drawdown_constraint` (new) | Add carry decay config | N/A | `"lambda_carry_decay": 0.7` |

**Generate the minimal patch — only touch the lines that need to change. Do not rewrite entire functions unnecessarily.**
