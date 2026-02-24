"""Build the Actuarial Features Explained notebook programmatically."""
import json, os

cells = []

def md(source):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": source})

# ═══════════════════════════════════════════════════════════════════
# TITLE
# ═══════════════════════════════════════════════════════════════════
md(r"""# Actuarial Features in TAPE-TCN: A Complete Guide

*From Insurance Mathematics to Portfolio Risk Management*

---""")

# ═══════════════════════════════════════════════════════════════════
# SECTION 1: WHAT ARE ACTUARIAL FEATURES?
# ═══════════════════════════════════════════════════════════════════
md(r"""## 1. What Are "Actuarial Features"?

Imagine you're an **insurance company**. When a building catches fire, you need to answer two questions:

1. **How severe is this damage?** *(Is it a small kitchen fire or a total loss?)*
2. **How long will it take to repair?** *(2 weeks? 6 months?)*

Now replace "building on fire" with **"portfolio losing money"** (a *drawdown*). The exact same actuarial math applies:

1. **How severe is this drawdown?** *(Is it a routine −3% dip or a catastrophic −30% crash?)*
2. **How long will recovery take?** *(Will we bounce back in 10 days or 200 days?)*

Your RL agent needs answers to these questions to make intelligent decisions. A **−10% drawdown** that will probably recover in 5 days is *very different* from a −10% drawdown that will probably take 6 months.""")

# ═══════════════════════════════════════════════════════════════════
# SECTION 2: THE 4 FEATURES
# ═══════════════════════════════════════════════════════════════════
md(r"""## 2. The 4 Actuarial Features

| Feature | Plain English | Range | What It Tells the Agent |
|:---|:---|:---:|:---|
| `Actuarial_Expected_Recovery` | "How many more days until we're whole again?" | 0 – ~100 days | **Patience signal** — high → stay cautious |
| `Actuarial_Prob_30d` | "What's the chance we recover within 30 days?" | 0.0 – 1.0 | **Short-term confidence gate** |
| `Actuarial_Prob_60d` | "What's the chance we recover within 60 days?" | 0.0 – 1.0 | **Medium-term confidence gate** |
| `Actuarial_Reserve_Severity` | "How bad is this compared to all past drawdowns?" | 0.0 – 1.0 | **Tail-risk alarm** (1.0 = worst ever) |""")

# ═══════════════════════════════════════════════════════════════════
# SECTION 3: DRAWDOWNS
# ═══════════════════════════════════════════════════════════════════
md(r"""## 3. Step 1 — What Is a Drawdown?

A **drawdown** is the percentage decline from a portfolio's peak value to its current value:

$$
\text{Drawdown}_t \;=\; \frac{P_t \;-\; P_{\text{peak}}}{P_{\text{peak}}}
$$

where:
- $P_t$ = portfolio value today  
- $P_{\text{peak}} = \max(P_1, P_2, \dots, P_t)$ = highest value reached so far (the **running maximum**)

### Worked Example

| Day | Price | Running Peak | Drawdown |
|:---:|------:|-----------:|:--------:|
| 1 | \$100 | \$100 | 0% *(at peak)* |
| 2 | \$105 | \$105 | 0% *(new peak!)* |
| 3 | \$98 | \$105 | −6.67% |
| 4 | \$92 | \$105 | **−12.38%** |
| 5 | \$96 | \$105 | −8.57% |
| 6 | \$107 | \$107 | 0% *(recovered!)* |

This gives us one complete **drawdown event**:

| Property | Value |
|:---|:---|
| **Start** | Day 3 |
| **End** | Day 6 |
| **Maximum Depth** | 12.38% (at Day 4) |
| **Duration** | 3 days |""")

# ═══════════════════════════════════════════════════════════════════
# SECTION 4: EVENT EXTRACTION
# ═══════════════════════════════════════════════════════════════════
md(r"""## 4. Step 2 — Extracting Drawdown Events

The code scans the entire price history and identifies every "episode" where the price dropped below a prior peak and then recovered:

$$
\text{Event}_k = \bigl\{\;\text{start\_date},\;\; \text{end\_date},\;\; \text{duration},\;\; \text{max\_depth},\;\; \text{peak}\;\bigr\}
$$

Over a typical **18-year training window** (2003–2021), you might extract **50–200 drawdown events** per asset, ranging from minor 2-day dips to the 2008 financial crisis (~400+ trading days).""")

# ═══════════════════════════════════════════════════════════════════
# SECTION 5: SEVERITY GRADING
# ═══════════════════════════════════════════════════════════════════
md(r"""## 5. Step 3 — Severity Grading: "How Bad Is This?"

### 5.1 Severity Buckets

Each drawdown is classified into a **severity bucket** based on its maximum depth:

$$
\text{Buckets} = \{5\%,\; 10\%,\; 15\%,\; 20\%,\; 25\%,\; 30\%\}
$$

**Classification rule:** find the first bucket $\geq$ the drawdown depth.

| Drawdown Depth | Assigned Bucket | Interpretation |
|:---:|:---:|:---|
| −3% | 5% | Mild dip |
| −8% | 10% | Correction |
| −14% | 15% | Significant |
| −22% | 25% | Bear market |
| −40% | 30% | Catastrophic (capped at largest bucket) |""")

md(r"""### 5.2 Severity Percentile — `Actuarial_Reserve_Severity`

This answers: *"Out of all drawdowns we've ever seen, what percentage were **less severe** than this one?"*

#### Empirical CDF

Given all $n$ historical drawdown depths sorted in ascending order:

$$
d_{(1)} \;\leq\; d_{(2)} \;\leq\; \cdots \;\leq\; d_{(n)}
$$

The **severity percentile** for a current drawdown of depth $d$ is:

$$
\boxed{\;\text{Severity Percentile} \;=\; \frac{\text{rank}(d)}{n}\;}
$$

where $\text{rank}(d)$ is the position where $d$ would be inserted into the sorted array (computed via binary search, `np.searchsorted`).

#### Example

Suppose you've seen **100 historical drawdowns**, and only 5 of them were worse than the current −18% drop:

$$
\text{Severity Percentile} = \frac{95}{100} = 0.95
$$

> **Interpretation:** The current drawdown is **worse than 95% of all historical drawdowns** — a serious alarm. 🚨""")

# ═══════════════════════════════════════════════════════════════════
# SECTION 6: SURVIVAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════
md(r"""## 6. Step 4 — Survival Analysis: "How Long Will Recovery Take?"

This is where the insurance math gets powerful. We use **Kaplan-Meier Survival Estimation**, the same method used in medical clinical trials to estimate "time to recovery."

### 6.1 What Is a Survival Function?

The **survival function** $S(t)$ answers:

> *"What is the probability that a drawdown lasts **longer** than $t$ days?"*

$$
S(t) \;=\; P\!\left(\text{Duration} > t\right)
$$

Key properties:
- $S(0) = 1.0$ → At day 0, 100% of drawdowns are still ongoing (obviously)
- $S(t)$ is monotonically **decreasing** — over time, more drawdowns recover
- $S(\infty) = 0$ → Eventually, all drawdowns end""")

md(r"""### 6.2 The Kaplan-Meier Estimator

Given $n$ observed drawdown durations in a severity bucket, the Kaplan-Meier estimator computes:

$$
\boxed{\;\hat{S}(t) \;=\; \prod_{t_i \,\leq\, t} \left(1 - \frac{d_i}{n_i}\right)\;}
$$

where:
| Symbol | Meaning |
|:---:|:---|
| $t_i$ | Time point where at least one drawdown ended (a "recovery event") |
| $d_i$ | Number of drawdowns that recovered at time $t_i$ |
| $n_i$ | Number of drawdowns **still ongoing** just before time $t_i$ (the "risk set") |

### 6.3 Worked Example

Suppose we have **10 drawdowns** in the 10% severity bucket with these durations (in trading days):

$$
\{5,\; 8,\; 12,\; 15,\; 15,\; 22,\; 30,\; 45,\; 60,\; 90\}
$$

| Time $t_i$ | Recovered $(d_i)$ | Still Ongoing $(n_i)$ | $\displaystyle 1 - \frac{d_i}{n_i}$ | $\hat{S}(t_i)$ |
|:---:|:---:|:---:|:---:|:---:|
| 5 | 1 | 10 | 0.900 | **0.900** |
| 8 | 1 | 9 | 0.889 | **0.800** |
| 12 | 1 | 8 | 0.875 | **0.700** |
| 15 | 2 | 7 | 0.714 | **0.500** |
| 22 | 1 | 5 | 0.800 | **0.400** |
| 30 | 1 | 4 | 0.750 | **0.300** |
| 45 | 1 | 3 | 0.667 | **0.200** |
| 60 | 1 | 2 | 0.500 | **0.100** |
| 90 | 1 | 1 | 0.000 | **0.000** |

**Reading the table:**  
- After **15 days**, 50% of drawdowns in this bucket have recovered  
- After **30 days**, 70% have recovered  
- After **90 days**, 100% have recovered""")

md(r"""### 6.4 Fallback (No `lifelines` Library)

If the Kaplan-Meier library isn't available, the code uses a simple **empirical fallback**:

$$
P\!\left(\text{recovery by } T \;\middle|\; \text{elapsed } t\right) \;=\; \frac{\#\bigl\{\text{events with duration} \leq T\bigr\}}{\#\bigl\{\text{events with duration} > t\bigr\}}
$$

This is mathematically equivalent to the KM estimator in the uncensored case, just computed differently.""")

# ═══════════════════════════════════════════════════════════════════
# SECTION 7: CONDITIONAL PROBABILITIES
# ═══════════════════════════════════════════════════════════════════
md(r"""## 7. Step 5 — Conditional Recovery Probabilities

This is the most nuanced piece. We don't just want $P(\text{recover by day 30})$. We want the **conditional probability given what we already know**.

If we're already 10 days into a drawdown, the question is:

> *"Given that we've already been falling for 10 days, what's the probability we recover in the **next** 30 days?"*

### The Conditional Survival Formula

$$
\boxed{\;P\!\left(\text{recover within next } h \text{ days} \;\middle|\; T > t\right) \;=\; 1 - \frac{S(t + h)}{S(t)}\;}
$$

where:
| Symbol | Meaning |
|:---:|:---|
| $t$ | Days already elapsed in the current drawdown |
| $h$ | Prediction horizon (30 days or 60 days) |
| $S(\cdot)$ | Kaplan-Meier survival function for the current severity bucket |""")

md(r"""### Worked Example — `Actuarial_Prob_30d`

Using our survival table from Section 6.3, suppose we're **12 days** into a 10%-severity drawdown:

**Known values:**
- $S(12) = 0.700$ → 70% of drawdowns last longer than 12 days  
- $S(12 + 30) = S(42) \approx 0.200$ *(interpolating between $S(30) = 0.300$ and $S(45) = 0.200$)*

$$
P\!\left(\text{recover within 30 days} \;\middle|\; \text{12 days in}\right) = 1 - \frac{S(42)}{S(12)} = 1 - \frac{0.200}{0.700} = 1 - 0.286 = \mathbf{0.714}
$$

> ✅ **Interpretation:** Given that we're already 12 days into this 10%-type drawdown, there's a **71.4% chance** we recover within the next 30 days.

This becomes the value of **`Actuarial_Prob_30d = 0.714`**.

---

### Worked Example — `Actuarial_Prob_60d`

Similarly, for the 60-day horizon:

- $S(12 + 60) = S(72) \approx 0.050$

$$
P\!\left(\text{recover within 60 days} \;\middle|\; \text{12 days in}\right) = 1 - \frac{S(72)}{S(12)} = 1 - \frac{0.050}{0.700} = \mathbf{0.929}
$$

> ✅ **92.9% chance of recovery within 60 days** → **`Actuarial_Prob_60d = 0.929`**""")

# ═══════════════════════════════════════════════════════════════════
# SECTION 8: EXPECTED RECOVERY TIME
# ═══════════════════════════════════════════════════════════════════
md(r"""## 8. Step 6 — Expected Recovery Time

The **expected remaining recovery time** uses the **median survival time** from the Kaplan-Meier curve:

$$
\text{Median Survival} = \min\bigl\{t : \hat{S}(t) \leq 0.5\bigr\}
$$

From our example: $\hat{S}(15) = 0.500$, so the **median survival time is 15 days**.

The expected *remaining* time is:

$$
\boxed{\;\text{Expected Recovery} = \max\!\left(0,\;\; \text{Median Survival} - \text{Days Elapsed}\right)\;}
$$

| Scenario | Calculation | Result |
|:---|:---|:---|
| 5 days into drawdown | $\max(0,\; 15 - 5)$ | **10 days remaining** |
| 12 days in | $\max(0,\; 15 - 12)$ | **3 days remaining** |
| 20 days in | $\max(0,\; 15 - 20)$ | **0 days** *(exceeded median — most similar drawdowns already recovered)* |""")

# ═══════════════════════════════════════════════════════════════════
# SECTION 9: EXPANDING WINDOW
# ═══════════════════════════════════════════════════════════════════
md(r"""## 9. How the Expanding Window Prevents Cheating

A critical design choice: the actuarial models are fitted using an **expanding window** to prevent **lookahead bias** (using future information that wouldn't be available in live trading).

$$
\begin{aligned}
\text{Day 252:} &\quad \text{Model trained on days } [1,\, 251] \;\;\rightarrow\;\; \text{Predict for day 252} \\
\text{Day 253:} &\quad \text{Model trained on days } [1,\, 252] \;\;\rightarrow\;\; \text{Predict for day 253} \\
\text{Day 500:} &\quad \text{Model trained on days } [1,\, 499] \;\;\rightarrow\;\; \text{Predict for day 500} \\
&\;\;\vdots\\
\text{Day 4411:} &\quad \text{Model trained on days } [1,\, 4410] \;\rightarrow\;\; \text{Predict for day 4411}
\end{aligned}
$$

The model **never sees the future**. On day 500, it only knows about drawdowns that occurred and *fully completed* before day 500.

- **Early predictions** (days 252–500): Based on limited history, less accurate but still informative
- **Late predictions** (days 3000+): Based on ~12 years of drawdown events, highly calibrated

The model is **re-fitted every time a drawdown episode completes** (price recovers to its prior peak), incorporating the new evidence.""")

# ═══════════════════════════════════════════════════════════════════
# SECTION 10: AGENT DASHBOARDS
# ═══════════════════════════════════════════════════════════════════
md(r"""## 10. How These Features Work Together for the RL Agent

Think of the 4 features as a **risk dashboard** the agent reads at every timestep:

### Scenario A — "Hold Steady"

```
╔══════════════════════════════════════════════════════════╗
║  ACTUARIAL RISK DASHBOARD — AAPL — Day 2847            ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  Current Drawdown:     -8.5%                             ║
║  Days in Drawdown:     14 days                           ║
║                                                          ║
║  ┌─ Severity ──────────────────────────────────────┐     ║
║  │ Reserve_Severity = 0.72                         │     ║
║  │ → "Worse than 72% of historical drawdowns"      │     ║
║  └─────────────────────────────────────────────────┘     ║
║                                                          ║
║  ┌─ Recovery Outlook ──────────────────────────────┐     ║
║  │ Expected_Recovery = 6 days remaining             │     ║
║  │ Prob_30d = 0.89 → "89% chance of recovery"      │     ║
║  │ Prob_60d = 0.97 → "97% chance of recovery"      │     ║
║  └─────────────────────────────────────────────────┘     ║
║                                                          ║
║  AGENT INTERPRETATION:                                   ║
║  "Somewhat severe, but historically recovers fast.       ║
║   Hold position — don't panic sell."                     ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
```

### Scenario B — "Reduce Exposure"

```
╔══════════════════════════════════════════════════════════╗
║  ACTUARIAL RISK DASHBOARD — AAPL — Day 3200            ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  Current Drawdown:     -23%                              ║
║  Days in Drawdown:     85 days                           ║
║                                                          ║
║  ┌─ Severity ──────────────────────────────────────┐     ║
║  │ Reserve_Severity = 0.96                         │     ║
║  │ → "Worse than 96% of ALL historical drawdowns"  │     ║
║  └─────────────────────────────────────────────────┘     ║
║                                                          ║
║  ┌─ Recovery Outlook ──────────────────────────────┐     ║
║  │ Expected_Recovery = 45 days remaining            │     ║
║  │ Prob_30d = 0.22 → "Only 22% chance in 30 days"  │     ║
║  │ Prob_60d = 0.58 → "Coin flip in 60 days"        │     ║
║  └─────────────────────────────────────────────────┘     ║
║                                                          ║
║  AGENT INTERPRETATION:                                   ║
║  "Historically rare and slow to recover.                 ║
║   REDUCE EXPOSURE — this could get worse."               ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
```""")

# ═══════════════════════════════════════════════════════════════════
# SECTION 11: WHY NO OTHER FEATURE PROVIDES THIS
# ═══════════════════════════════════════════════════════════════════
md(r"""## 11. Why No Other Feature Provides This

| Existing Feature | What It Measures | What It's Missing |
|:---|:---|:---|
| `RollingVolatility_21d` | How much prices wiggle | **Doesn't say how long** pain will last |
| `DownsideSemiVar_21d` | How much prices fall | **Doesn't estimate** recovery probability |
| `RSI_14` | Is the asset oversold? | **No historical context** on similar drawdowns |
| `Regime_Volatility_Ratio` | Is vol increasing? | **Doesn't quantify** severity vs. history |
| `ShortTerm_Reversal_5` | Recent bounce pattern | **5-day window** is too short for structural assessment |

> The actuarial features are the **only features that model the TIME DIMENSION of risk** — not just *"how much are we losing?"* but *"how long does this kind of loss typically last, and what's the probability of recovery given how long we've already been in it?"*""")

# ═══════════════════════════════════════════════════════════════════
# SECTION 12: COMPLETE PIPELINE
# ═══════════════════════════════════════════════════════════════════
md(r"""## 12. The Complete Mathematical Pipeline

```
                    ┌──────────────────┐
                    │   Raw Prices     │
                    │  P₁, P₂, …, Pₜ  │
                    └────────┬─────────┘
                             │
                             ▼
                ┌───────────────────────────┐
                │  1. Compute Drawdowns     │
                │     & Running Maximum     │     DDₜ = (Pₜ - max(P₁…Pₜ)) / max(P₁…Pₜ)
                └────────────┬──────────────┘
                             │
                             ▼
                ┌───────────────────────────┐
                │  2. Extract Events        │
                │  (Peak → Trough → Peak)   │     ~50–200 events over 18 years
                └────────────┬──────────────┘
                             │
            ┌────────────────┴───────────────────┐
            │                                    │
            ▼                                    ▼
  ┌──────────────────────┐           ┌──────────────────────────┐
  │  3a. Severity CDF    │           │  3b. Kaplan-Meier        │
  │  Sort max_depths     │           │  Ŝ(t) = ∏(1 - dᵢ/nᵢ)   │
  │  Percentile = rank/n │           │  Per severity bucket     │
  └──────────┬───────────┘           └──────────┬───────────────┘
             │                                  │
             ▼                                  ▼
  ┌──────────────────────┐        ┌───────────────────────────────┐
  │  Reserve_Severity    │        │  Prob_30d = 1 - S(t+30)/S(t)  │
  │  (0.0 to 1.0)       │        │  Prob_60d = 1 - S(t+60)/S(t)  │
  └──────────────────────┘        │  Expected = max(0, median - t) │
                                  └───────────────────────────────┘
```""")

# ═══════════════════════════════════════════════════════════════════
# SECTION 13: NOTATION SUMMARY
# ═══════════════════════════════════════════════════════════════════
md(r"""## 13. Summary of Mathematical Notation

| Symbol | Meaning |
|:---:|:---|
| $P_t$ | Asset price at time $t$ |
| $P_{\text{peak}}$ | Running maximum price: $\max(P_1, \dots, P_t)$ |
| $d$ | Drawdown depth (positive number, e.g., $0.15 = 15\%$ drop) |
| $S(t)$ | Survival function — $P(\text{Duration} > t)$ |
| $\hat{S}(t)$ | Kaplan-Meier estimate of $S(t)$ |
| $d_i$ | Number of recoveries at time $t_i$ |
| $n_i$ | Number of drawdowns still ongoing at $t_i$ (risk set) |
| $h$ | Prediction horizon (30 or 60 days) |
| $t$ | Days already elapsed in current drawdown |""")

# ═══════════════════════════════════════════════════════════════════
# REFERENCES
# ═══════════════════════════════════════════════════════════════════
md(r"""## References

1. **Kaplan-Meier Estimator** — Kaplan, E.L. & Meier, P. (1958). *"Nonparametric Estimation from Incomplete Observations."* Journal of the American Statistical Association, 53(282), 457–481.  
2. **Chain Ladder Method** — Mack, T. (1993). *"Distribution-free calculation of the standard error of chain ladder reserve estimates."* ASTIN Bulletin, 23(2), 213–225.  
3. **Survival Analysis in Finance** — Lando, D. (2004). *Credit Risk Modeling: Theory and Applications.* Princeton University Press.

---

*Document generated for the TAPE-TCN project — Adaptive Portfolio RL.*""")

# ═══════════════════════════════════════════════════════════════════
# BUILD NOTEBOOK
# ═══════════════════════════════════════════════════════════════════
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

out_path = os.path.join(os.path.dirname(__file__), "actuarial_features_explained.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print(f"Notebook written to: {out_path}")
print(f"Total cells: {len(cells)}")
