# Debugging TAPE Score Issue — Patch Request

The TAPE score is consistently 0.0000 in training logs, meaning the terminal bonus is failing. We need to add extensive debug logging to the `step()` method in `environment_tape_rl.py` to identify why.

---

## FIX: Add Debug Logging to `step()`

**File:** `src/environment_tape_rl.py`  
**Location:** `step()` method, termination block (lines 840-884)

**Objective:**
1. Print `self.reward_system` (verify it's 'tape')
2. Print `episode_metrics` (verify it's not empty/zero)
3. Print `tape_score` (verify calculation result)
4. Verify `info` dict content

**Replace lines 840-886 with this debug-enhanced block:**

```python
            # 🎯 TAPE Terminal Bonus: Holistic episode-level "tie-breaker"
            logger.info(f"DEBUG: Terminating with reward_system='{self.reward_system}'")
            
            if self.reward_system == 'tape':
                # Calculate episode-level metrics for TAPE scoring
                episode_metrics = calculate_episode_metrics(
                    portfolio_values=np.array(self.episode_portfolio_values),
                    returns=np.array(self.episode_return_history),
                    weight_changes=self.episode_weight_changes,
                    risk_free_rate=0.02,
                    trading_days_per_year=252
                )
                
                logger.info(f"DEBUG: metrics keys: {list(episode_metrics.keys())}")
                logger.info(f"DEBUG: Sharpe={episode_metrics.get('sharpe_ratio')}, MDD={episode_metrics.get('max_drawdown')}")
                
                # Calculate TAPE score using the active profile (0 to 1)
                tape_score = calculate_tape_score(
                    metrics=episode_metrics,
                    profile=self.tape_profile
                )
                
                logger.info(f"DEBUG: Calculated tape_score: {tape_score}")
                
                if tape_score is None:
                    logger.error("DEBUG: tape_score is None! Forcing to 0.0")
                    tape_score = 0.0
                
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
```

**Also apply the fix to `tcn_phase1.py` to print TAPE score even if None:**

**File:** `src/notebook_helpers/tcn_phase1.py`  
**Location:** Line 1909-1910

**Replace:**
```python
                tape_score = info.get("tape_score")
                if tape_score is not None:
```

**With:**
```python
                tape_score = info.get("tape_score")
                # Force print for debugging
                if True: 
                    score_val = tape_score if tape_score is not None else -1.0
```
(And update subsequent usage to `score_val` or keep indentation logic to ensure it prints).

**Better yet, for `tcn_phase1.py`, just adding a fallback log:**

```python
                tape_score = info.get("tape_score")
                if tape_score is None:
                     print(f"   ⚠️ DEBUG: tape_score is None for Episode {training_episode_count}")
                if tape_score is not None:
```
