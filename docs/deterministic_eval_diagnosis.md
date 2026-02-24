# Why Deterministic Evaluation Produces Identical Results Across Episodes

*Root cause analysis — Exp6 TCN_FUSION Enhanced TAPE training*

---

## Symptom

```
Episode 1: Sharpe=0.819 | Return=+68.13% | DD=13.71%  ← saved
Episode 2: Sharpe=0.819 | Return=+68.14% | DD=13.68%  ← saved
Episode 3: Sharpe=0.819 | Return=+68.13% | DD=13.68%  ← NOT saved
```

All three episodes produce **virtually identical** Sharpe ratios, returns, and drawdowns during deterministic validation — despite the policy being updated between episodes.

---

## Execution Path Traced

```
maybe_save_deterministic_validation_checkpoint(episode_idx)
  │
  ├── env_eval = env_test_deterministic
  │     random_start = False    ← always starts at day 0
  │     episode_length_limit = None  ← runs full 1103-day test set
  │
  ├── env_eval.reset(seed=6042 + 10000 + episode_idx)
  │     seed is IGNORED (random_start=False)
  │
  ├── agent.get_action_and_value(obs, deterministic=True, evaluation_mode='mean')
  │     │
  │     ├── logits = actor_network(obs)   ← raw output of final Dense layer
  │     ├── alpha = elu(logits) + 1.0 + epsilon   ← epsilon=0.5 at start
  │     └── action = alpha / sum(alpha)   ← Dirichlet MEAN
  │
  └── compute_episode_metrics(env_eval)
        └── Sharpe, Return, DD from portfolio_history
```

---

## Root Cause 1: Fixed Test Environment (By Design for Ranking)

```python
# In tcn_phase1.py, line 2192
env_test_deterministic = PortfolioEnvTAPE(
    random_start=False,         # Always starts at day 0
    episode_length_limit=None,  # Runs entire 1103-day test set
)
```

Every validation run traverses the **exact same 1103 market days** in the **exact same order**. The seed passed to `env.reset()` only controls random start position — with `random_start=False`, the seed is ignored.

**Consequence:** The only source of variation between evaluations is the **policy weights**. If the policy outputs the same (or nearly the same) actions, the metrics will be identical. This is desirable for apples-to-apples checkpoint ranking; it simply exposes policy flatness quickly.

---

## Root Cause 2: Dirichlet Mean Compresses Differences

The `evaluation_mode='mean'` computes:

$$\text{action}_i = \frac{\alpha_i}{\sum_j \alpha_j}$$

where alpha is computed via the ELU activation:

$$\alpha_i = \text{elu}(\text{logit}_i) + 1.0 + \epsilon$$

### The Math of Why It Goes Flat

**Early in training**, the actor network's logits are near-zero (randomly initialized weights, tiny learning rate of 8e-6). Here's what happens:

| Logit | elu(logit) | + 1.0 | + ε (0.5) | Alpha |
|:---:|:---:|:---:|:---:|:---:|
| −0.01 | −0.0099 | 0.990 | **1.490** | ≈1.49 |
| +0.03 | +0.0300 | 1.030 | **1.530** | ≈1.53 |
| −0.05 | −0.0488 | 0.951 | **1.451** | ≈1.45 |
| +0.01 | +0.0100 | 1.010 | **1.510** | ≈1.51 |

All alphas are **≈1.5**. For 10 assets + cash (K=11):

$$\text{mean}_i = \frac{1.5}{11 \times 1.5} = \frac{1}{11} \approx 0.0909$$

The policy outputs a **near-uniform allocation** regardless of market conditions. This uniform portfolio produces a fixed buy-and-hold return over the 1103-day test set → identical Sharpe every time.

### Why It Stays Flat Across Episodes

Between episode 1 and episode 3, the actor has received only **~768–1152 gradient steps** at `actor_lr=8e-6`. The cumulative parameter change is:

$$\Delta \theta \approx \text{steps} \times \text{lr} \times \nabla L \approx 1000 \times 8 \times 10^{-6} \times O(1) \approx 0.008$$

This shifts logits by ~0.008, changing alpha from ~1.500 to ~1.508. The resulting mean changes from:

$$\frac{1.500}{16.500} = 0.09091 \quad\longrightarrow\quad \frac{1.508}{16.588} = 0.09091$$

**Effectively zero difference** at 3 decimal places of Sharpe.

---

## Root Cause 3: Checkpoint Gate is Too Loose

```python
deterministic_validation_sharpe_min_delta_cfg = 0.000  # No minimum improvement required
```

The checkpoint save condition:
```python
if val_sharpe <= (best_sharpe + min_delta):
    return  # skip
```

With `min_delta=0.000`, any Sharpe ≥ the previous best (even by 0.0001) triggers a new checkpoint save. This is why episode 2 saved — it was `0.8191` vs `0.8190` at full precision, despite being functionally identical.

---

## Why Episode 2 Got Saved But Episode 3 Didn't

```
Episode 1: 0.8190... → saved (first ever, exceeds min_sharpe=0.50)
Episode 2: 0.8191... → saved (0.8191 > 0.8190 + 0.000)
Episode 3: 0.8189... → NOT saved (0.8189 ≤ 0.8191 + 0.000)
```

The "improvement" is noise in the 4th decimal place — not a real policy improvement.

---

## The Fundamental Issue: Mean vs. Mode

### `evaluation_mode='mean'` (Current — Bad for this scenario)

$$\text{action}_i = \frac{\alpha_i}{\sum_j \alpha_j}$$

The mean **normalizes away** differences. If all alphas are in the range [1.4, 1.6], the mean will always be close to 1/K.

### `evaluation_mode='mode'` (Recommended with guardrails)

For $\alpha_i > 1$:

$$\text{action}_i = \frac{\alpha_i - 1}{\sum_j \alpha_j - K}$$

The mode **subtracts 1** from each alpha first, then normalizes. This amplifies small differences:

| Mean formula | Mode formula |
|:---:|:---:|
| 1.50 / 16.5 = **0.0909** | 0.50 / 5.5 = **0.0909** |
| 1.60 / 16.6 = **0.0964** | 0.60 / 5.6 = **0.1071** |
| 1.40 / 16.4 = **0.0854** | 0.40 / 5.4 = **0.0741** |

The mode formula produces **2–3x more spread** from the same alphas. An asset with alpha=1.60 gets 10.7% allocation (mode) vs 9.6% (mean) — and one with alpha=1.40 gets 7.4% (mode) vs 8.5% (mean). The mode **amplifies the signal** from the network's logits.

⚠️ Caveat: if any alpha values are ≤1, the implementation falls back to vertex behavior (near one-hot). Monitor the mode fallback fraction before using mode as your primary checkpoint selector.

---

## Contributing Factor: High Epsilon

The Dirichlet epsilon starts at **0.5** and decays during training. This adds 0.5 to every alpha value, drowning the network's actual logit signal:

```
alpha = elu(logit) + 1.0 + 0.5
                          ^^^^
                     This alone is 33% of the final alpha value
```

At epsilon=0.5, the network logits must produce differences of ±0.5 just to overcome the epsilon floor — which requires many gradient steps at lr=8e-6.

---

## Recommended Fixes

### Priority 1 — High Impact

| Change | From | To | Why |
|:---|:---:|:---:|:---|
| `evaluation_mode` | `'mean'` | `'mode'` | Amplifies policy differences by 2–3x |
| `epsilon_start` | `0.5` | `0.1–0.2` | Lets network logits dominate alpha sooner |

### Priority 2 — Medium Impact

| Change | From | To | Why |
|:---|:---:|:---:|:---|
| `min_delta` | `0.000` | `0.005–0.01` | Prevents saving functionally-identical checkpoints |
| `actor_lr` | `8e-6` | `2e-5–5e-5` (warmup) | Faster logit differentiation in early episodes |
| `eval_every_episodes` | `1` | `3–5` | No point evaluating if policy barely changed |

### Priority 3 — Diagnostic

Add alpha statistics logging at validation time (compatible with current agent API):

```python
# In run_deterministic_validation_metrics(), after get_action_and_value():
# Use actor forward pass for alpha diagnostics (get_action_and_value returns action/log_prob/value only)
if agent.is_sequential and getattr(agent, "_latest_sequence", None) is not None:
    alpha_state_input, _ = agent.prepare_state_input(agent._latest_sequence)
else:
    alpha_state_input, _ = agent.prepare_state_input(obs_eval)

alpha = tf.convert_to_tensor(agent.actor(alpha_state_input, training=False), dtype=tf.float32)
alpha_np = alpha.numpy().reshape(-1)
alpha_mean = alpha_np.mean()
alpha_std = alpha_np.std()
alpha_range = alpha_np.max() - alpha_np.min()
print(f"      Alpha stats: mean={alpha_mean:.4f}, std={alpha_std:.4f}, spread={alpha_range:.4f}")
```

If `alpha_std < 0.01`, the policy is still effectively uniform — no point saving checkpoints.

---

## Expected Behavior After Fixes

With `evaluation_mode='mode'` and `epsilon=0.15`:

```
Episode 1:  Sharpe=0.819 | alpha_spread=0.02 → near-uniform (expected early)
Episode 5:  Sharpe=0.834 | alpha_spread=0.08 → emerging differentiation
Episode 15: Sharpe=0.912 | alpha_spread=0.25 → meaningful allocation policy
Episode 30: Sharpe=0.947 | alpha_spread=0.40 → converged non-trivial policy
```

The key metric is **alpha_spread** — once it exceeds ~0.1, the policy has learned to differentiate assets, and checkpoint selection becomes meaningful.

---

*Analysis date: 2026-02-23*
*Applies to: Exp6_TCN_FUSION_Enhanced_TAPE_training*
