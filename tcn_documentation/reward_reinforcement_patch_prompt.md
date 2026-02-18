# Reward Reinforcement & Penalty Rebalancing — Patch Request

The agent's step-level penalties (concentration, action realization, drawdown) are drowning out the positive return signal. Total penalties reach 1.0-5.0 per step while the base reward is only ±0.5. This makes it impossible for the agent to learn what "good returns" look like. Two fixes are needed: (1) reduce penalty scalars, (2) add a penalty budget cap, and (3) add intra-episode milestone bonuses to capture strong mid-episode performance.

---

## FIX 1: Reduce Penalty Scalars

**File:** `src/config.py`  
**Location:** `environment_params` dict (around lines 308-315)

The current scalars are too aggressive. Reduce them so penalties don't dominate the base return signal (~0.5 per step).

**Current values:**
```python
"concentration_penalty_scalar": 4.0,
"top_weight_penalty_scalar": 3.0,
"action_realization_penalty_scalar": 2.0,
```

**Replace with:**
```python
"concentration_penalty_scalar": 2.0,        # was 4.0 — halved to let return signal through
"top_weight_penalty_scalar": 1.5,            # was 3.0 — halved
"action_realization_penalty_scalar": 0.5,    # was 2.0 — drastically reduced (penalizes constraint projection, not agent's fault)
```

**Rationale:**
- `concentration`: Halved to 2.0.
- `top_weight`: Halved to 1.5.
- `action_realization`: Slashed to 0.5. This penalizes the agent for proposing unfeasible weights. It should be a gentle nudge, not a sledgehammer.

---

## FIX 2: Per-Step Penalty Budget Cap

**File:** `src/environment_tape_rl.py`  
**Method:** `step()` — replace the penalty application block (lines 1161-1186)

Cap the total penalty so it never exceeds a configurable multiple of the absolute base reward signal. This ensures the return signal is always visible in the gradient.

### 2a. Add config param to `__init__`

Add near where other penalty params are read (around lines 200-230):
```python
self.penalty_budget_ratio = float(env_params.get('penalty_budget_ratio', 2.0))
```

### 2b. Replace penalty application logic in `step()`

**Current code (lines 1161-1186):**
```python
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

        reward -= concentration_penalty
        reward -= action_realization_penalty
        self.concentration_penalty_sum += concentration_penalty
        self.action_realization_penalty_sum += action_realization_penalty

        drawdown_penalty = 0.0
        avg_drawdown_excess = 0.0
        current_drawdown = 0.0
        drawdown_regime_multiplier = 1.0
        if self.drawdown_constraint_enabled:
            drawdown_penalty, current_drawdown, excess_drawdown, drawdown_regime_multiplier = self._apply_drawdown_dual_controller()
            reward -= drawdown_penalty
            reward = np.clip(reward, -150.0, 150.0)
            avg_drawdown_excess = self.drawdown_excess_accumulator / max(1, self.episode_step_count)
```

**Replace with:**
```python
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
        if self.drawdown_constraint_enabled:
            drawdown_penalty, current_drawdown, excess_drawdown, drawdown_regime_multiplier = self._apply_drawdown_dual_controller()
            avg_drawdown_excess = self.drawdown_excess_accumulator / max(1, self.episode_step_count)

        # Penalty budget cap: total penalties cannot exceed penalty_budget_ratio × |base reward signal|
        total_penalty = concentration_penalty + action_realization_penalty + drawdown_penalty
        base_signal = abs(reward)  # reward here is base + DSR + turnover (before penalties)
        max_allowed_penalty = max(base_signal, 0.1) * self.penalty_budget_ratio
        if total_penalty > max_allowed_penalty:
            # Scale all penalties proportionally to fit within budget
            scale_factor = max_allowed_penalty / max(total_penalty, 1e-8)
            concentration_penalty *= scale_factor
            action_realization_penalty *= scale_factor
            drawdown_penalty *= scale_factor
            total_penalty = max_allowed_penalty

        reward -= total_penalty
        reward = np.clip(reward, -150.0, 150.0)

        self.concentration_penalty_sum += concentration_penalty
        self.action_realization_penalty_sum += action_realization_penalty
```

### 2c. Add config default to `src/config.py`

In `environment_params` (around line 315):
```python
"penalty_budget_ratio": 2.0,  # Total penalties capped at 2× the base reward signal magnitude
```

---

## FIX 3: Intra-Episode TAPE Milestone Bonuses

**File:** `src/environment_tape_rl.py`  
**Method:** `step()` — insert after the penalty block (before STEP 9: SAVE TO MEMORY at line 1189)

Every 252 steps (1 trading year), compute a rolling TAPE score. If it exceeds a threshold, add a small bonus to reinforce good behavior as it happens.

### 3a. Add config params to `__init__`

Add near where other TAPE params are read (around lines 216-220):
```python
self.tape_milestone_interval = int(env_params.get('tape_milestone_interval', 252))
self.tape_milestone_threshold = float(env_params.get('tape_milestone_threshold', 0.25))
self.tape_milestone_scalar = float(env_params.get('tape_milestone_scalar', 2.0))
```

### 3b. Insert milestone logic in `step()`

Insert after the penalty budget cap block, before `# STEP 9: SAVE TO MEMORY`:
```python
        # ═══════════════════════════════════════════════════════════════
        # INTRA-EPISODE TAPE MILESTONE BONUS
        # ═══════════════════════════════════════════════════════════════
        if (self.reward_system == 'tape'
            and self.tape_milestone_interval > 0
            and self.episode_step_count > 0
            and self.episode_step_count % self.tape_milestone_interval == 0
            and len(self.episode_return_history) > 10):

            milestone_metrics = calculate_episode_metrics(
                portfolio_values=np.array(self.episode_portfolio_values),
                returns=np.array(self.episode_return_history),
                weight_changes=self.episode_weight_changes,
                risk_free_rate=0.02,
                trading_days_per_year=252
            )
            milestone_tape = calculate_tape_score(
                metrics=milestone_metrics,
                profile=self.tape_profile
            )

            # Only reward if above threshold (don't penalize bad milestones)
            if milestone_tape > self.tape_milestone_threshold:
                milestone_bonus = milestone_tape * self.tape_milestone_scalar
                reward += milestone_bonus
                logger.info(
                    f"   🏆 TAPE Milestone at step {self.episode_step_count}: "
                    f"score={milestone_tape:.4f}, bonus={milestone_bonus:.3f}"
                )
```

### 3c. Add config defaults to `src/config.py`

In `environment_params` (near `tape_terminal_scalar`):
```python
"tape_milestone_interval": 252,      # Compute intra-episode TAPE every N steps (252 = 1yr)
"tape_milestone_threshold": 0.25,    # Only give bonus if TAPE > this value
"tape_milestone_scalar": 2.0,        # Bonus = milestone_tape × this scalar
```

---

## Summary of All Changes

| File | Location | Change |
|:-----|:---------|:-------|
| `config.py` | `environment_params` (~line 311) | `concentration_penalty_scalar`: 4.0 → 2.0 |
| `config.py` | `environment_params` (~line 313) | `top_weight_penalty_scalar`: 3.0 → 1.5 |
| `config.py` | `environment_params` (~line 315) | `action_realization_penalty_scalar`: 2.0 → 0.5 |
| `config.py` | `environment_params` (new) | Add `penalty_budget_ratio: 2.0` |
| `config.py` | `environment_params` (new) | Add 3 milestone params |
| `environment_tape_rl.py` | `__init__` (~line 220) | Add `penalty_budget_ratio` + 3 milestone params |
| `environment_tape_rl.py` | `step()` (lines 1161-1186) | Restructure: compute all penalties first, apply budget cap, then subtract |
| `environment_tape_rl.py` | `step()` (after penalties) | Insert milestone bonus block |
