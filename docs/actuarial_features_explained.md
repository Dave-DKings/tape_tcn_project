# Actuarial Features in TAPE-TCN: A Complete Guide

*From Insurance Mathematics to Portfolio Risk Management*

---

## What Are "Actuarial Features"?

Imagine you're an insurance company. When a building catches fire, you need to answer two questions:

1. **How severe is this damage?** (Is it a small kitchen fire or a total loss?)
2. **How long will it take to repair?** (2 weeks? 6 months?)

Now replace "building on fire" with **"portfolio losing money"** (a drawdown). The exact same actuarial math applies:

1. **How severe is this drawdown?** (Is it a routine -3% dip or a catastrophic -30% crash?)
2. **How long will recovery take?** (Will we bounce back in 10 days or 200 days?)

Your RL agent needs answers to these questions to make intelligent decisions. A -10% drawdown that will probably recover in 5 days is very different from a -10% drawdown that will probably take 6 months.

---

## The 4 Actuarial Features

| Feature | Plain English | Range | What It Tells the Agent |
|---|---|---|---|
| `Actuarial_Expected_Recovery` | "How many more days until we're whole again?" | 0 to ~100 days | Patience signal — high = stay cautious |
| `Actuarial_Prob_30d` | "What's the chance we recover within 30 days?" | 0.0 to 1.0 | Short-term confidence gate |
| `Actuarial_Prob_60d` | "What's the chance we recover within 60 days?" | 0.0 to 1.0 | Medium-term confidence gate |
| `Actuarial_Reserve_Severity` | "How bad is this compared to all past drawdowns?" | 0.0 to 1.0 | Tail-risk alarm (1.0 = worst ever) |

---

## Step-by-Step Math Walkthrough

### Step 1: What Is a Drawdown?

A **drawdown** is the percentage decline from a portfolio's peak value to its current value.

$$
\text{Drawdown}_t = \frac{P_t - P_{\text{peak}}}{P_{\text{peak}}}
$$

Where:
- $P_t$ = portfolio value today
- $P_{\text{peak}}$ = the highest value the portfolio has reached so far (running maximum)

**Example:**

| Day | Price | Running Peak | Drawdown |
|---|---|---|---|
| 1 | $100 | $100 | 0% (at peak) |
| 2 | $105 | $105 | 0% (new peak!) |
| 3 | $98 | $105 | -6.67% |
| 4 | $92 | $105 | -12.38% |
| 5 | $96 | $105 | -8.57% |
| 6 | $107 | $107 | 0% (recovered!) |

This gives us one complete **drawdown event**:
- **Start**: Day 3
- **End**: Day 6
- **Maximum Depth**: -12.38% (at Day 4)
- **Duration**: 3 days (from Day 3 to Day 6)

---

### Step 2: Extracting Drawdown Events (_extract_drawdown_events)

The code scans the entire price history and identifies every "episode" where the price dropped below a prior peak and then recovered. Each event is recorded as:

```
Event = {
    start_date:  When the drawdown began
    end_date:    When price recovered to peak
    duration:    Number of days from start to recovery
    max_depth:   Deepest point of the drawdown (e.g., 0.1238 for -12.38%)
    peak:        The peak price before the drop
}
```

Over a typical 18-year training window (2003–2021), you might extract **50–200 drawdown events** per asset, ranging from minor 2-day dips to the 2008 financial crisis (which lasted ~400+ trading days).

---

### Step 3: Severity Grading — "How Bad Is This?"

#### The Severity Buckets

Each drawdown is classified into a **severity bucket** based on its maximum depth:

```
Buckets: [5%, 10%, 15%, 20%, 25%, 30%]
```

**Classification Rule**: Find the first bucket ≥ the drawdown depth.

| Drawdown Depth | Assigned Bucket |
|---|---|
| -3% | 5% (mild dip) |
| -8% | 10% (correction) |
| -14% | 15% (significant) |
| -22% | 25% (bear market) |
| -40% | 30% (catastrophic — capped at largest bucket) |

#### The Severity Percentile (`Actuarial_Reserve_Severity`)

This answers: *"Out of all drawdowns we've ever seen, what percentage were less severe than this one?"*

**Math — Empirical CDF:**

Given all $n$ historical drawdown depths sorted in ascending order:

$$
d_{(1)} \leq d_{(2)} \leq \cdots \leq d_{(n)}
$$

The severity percentile for a current drawdown of depth $d$ is:

$$
\text{Severity Percentile} = \frac{\text{rank}(d)}{n}
$$

Where $\text{rank}(d)$ is the position where $d$ would be inserted into the sorted array.

**Example**: If you've seen 100 drawdowns, and only 5 of them were worse than the current -18% drop, then:

$$
\text{Severity Percentile} = \frac{95}{100} = 0.95
$$

This means the current drawdown is **worse than 95% of all historical drawdowns** — a serious alarm.

---

### Step 4: Survival Analysis — "How Long Will Recovery Take?"

This is where the insurance math gets powerful. We use **Kaplan-Meier Survival Estimation**, the same method used in medical trials to estimate "time to recovery."

#### What Is a Survival Function?

The **survival function** $S(t)$ answers: *"What is the probability that a drawdown lasts longer than $t$ days?"*

$$
S(t) = P(\text{Duration} > t)
$$

- $S(0) = 1.0$ → At day 0, 100% of drawdowns are still ongoing (obviously)
- $S(30) = 0.4$ → 40% of drawdowns last longer than 30 days
- $S(100) = 0.05$ → Only 5% last longer than 100 days

#### Kaplan-Meier Estimator

Given $n$ observed drawdown durations in a severity bucket, the Kaplan-Meier estimator computes:

$$
\hat{S}(t) = \prod_{t_i \leq t} \left(1 - \frac{d_i}{n_i}\right)
$$

Where:
- $t_i$ = time point where at least one drawdown ended (a "recovery event")
- $d_i$ = number of drawdowns that recovered at time $t_i$
- $n_i$ = number of drawdowns still ongoing just before time $t_i$ (the "risk set")

**Worked Example:**

Suppose we have 10 drawdowns in the 10% severity bucket with these durations (in days):

```
[5, 8, 12, 15, 15, 22, 30, 45, 60, 90]
```

| Time $t_i$ | Recovered ($d_i$) | Still Ongoing ($n_i$) | $1 - d_i/n_i$ | $\hat{S}(t_i)$ |
|---|---|---|---|---|
| 5 | 1 | 10 | 0.900 | 0.900 |
| 8 | 1 | 9 | 0.889 | 0.800 |
| 12 | 1 | 8 | 0.875 | 0.700 |
| 15 | 2 | 7 | 0.714 | 0.500 |
| 22 | 1 | 5 | 0.800 | 0.400 |
| 30 | 1 | 4 | 0.750 | 0.300 |
| 45 | 1 | 3 | 0.667 | 0.200 |
| 60 | 1 | 2 | 0.500 | 0.100 |
| 90 | 1 | 1 | 0.000 | 0.000 |

Reading the table: After 15 days, 50% of drawdowns in this bucket have recovered. After 30 days, 70% have recovered. After 90 days, 100% have recovered.

#### Fallback (No `lifelines` Library)

If Kaplan-Meier isn't available, the code uses a simple **empirical fallback**:

$$
P(\text{recovery by } T \mid \text{already elapsed } t) = \frac{\#\{\text{events with duration} \leq T\}}{\#\{\text{events with duration} > t\}}
$$

---

### Step 5: Conditional Recovery Probabilities — "Given We're Already X Days In..."

This is the most nuanced piece. We don't just want $P(\text{recover by day 30})$ — we want the **conditional probability given what we already know**.

If we're already 10 days into a drawdown, we need:

$$
P(\text{recover within next 30 days} \mid \text{already 10 days in}) = 1 - \frac{S(10 + 30)}{S(10)}
$$

**The Conditional Survival Formula:**

$$
P(\text{recover by } t + h \mid T > t) = 1 - \frac{S(t + h)}{S(t)}
$$

Where:
- $t$ = days already elapsed in the current drawdown
- $h$ = horizon (30 days or 60 days)
- $S(\cdot)$ is the Kaplan-Meier survival function for the current severity bucket

**Worked Example:**

Using our survival table above, suppose we're 12 days into a 10%-severity drawdown:

- $S(12) = 0.700$ (70% of drawdowns last longer than 12 days)
- $S(12 + 30) = S(42) \approx 0.200$ (interpolating between $S(30)=0.300$ and $S(45)=0.200$)

$$
P(\text{recover within 30 days} \mid \text{12 days in}) = 1 - \frac{0.200}{0.700} = 1 - 0.286 = 0.714
$$

**Interpretation**: Given that we're already 12 days into this 10%-type drawdown, there's a **71.4% chance we recover within the next 30 days**.

This becomes the value of `Actuarial_Prob_30d`.

Similarly for the 60-day horizon:
- $S(12 + 60) = S(72) \approx 0.050$

$$
P(\text{recover within 60 days} \mid \text{12 days in}) = 1 - \frac{0.050}{0.700} = 0.929
$$

**92.9% chance of recovery within 60 days** → `Actuarial_Prob_60d = 0.929`

---

### Step 6: Expected Recovery Time

The **expected remaining recovery time** uses the **median survival time** from the Kaplan-Meier curve:

$$
\text{Median Survival} = \min\{t : \hat{S}(t) \leq 0.5\}
$$

From our example: $\hat{S}(15) = 0.500$, so the **median survival time is 15 days**.

The expected *remaining* time is:

$$
\text{Expected Recovery} = \max\left(0, \;\text{Median Survival} - \text{Days Elapsed}\right)
$$

If we're 5 days in: Expected remaining = $\max(0, 15 - 5) = 10$ days
If we're 20 days in: Expected remaining = $\max(0, 15 - 20) = 0$ days (we've exceeded median — most similar drawdowns already recovered)

---

## How the Expanding Window Prevents Cheating

A critical design choice: the actuarial models are fitted using an **expanding window** to prevent **lookahead bias**.

```
Day 252:   Model trained on days [1, 251]    → Predict for day 252
Day 253:   Model trained on days [1, 252]    → Predict for day 253
Day 500:   Model trained on days [1, 499]    → Predict for day 500
...
Day 4411:  Model trained on days [1, 4410]   → Predict for day 4411
```

The model **never sees the future**. On day 500, it only knows about drawdowns that occurred and *fully completed* before day 500. This means:

- **Early predictions** (days 252–500): Based on limited history, less accurate but still informative.
- **Late predictions** (days 3000+): Based on ~12 years of drawdown events, highly calibrated.

The model is **re-fitted every time a drawdown episode completes** (price recovers to its prior peak), incorporating the new evidence.

---

## How These Features Work Together For the RL Agent

Think of the 4 features as a **risk dashboard**:

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

Contrast with a different scenario:

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
```

---

## Why No Other Feature Provides This

| Existing Feature | What It Measures | What It's Missing |
|---|---|---|
| `RollingVolatility_21d` | How much prices wiggle | Doesn't say how long pain will last |
| `DownsideSemiVar_21d` | How much prices fall | Doesn't estimate recovery probability |
| `RSI_14` | Is the asset oversold? | No historical context on similar drawdowns |
| `Regime_Volatility_Ratio` | Is vol increasing? | Doesn't quantify severity vs. history |
| `ShortTerm_Reversal_5` | Recent bounce pattern | 5-day window is too short for structural assessment |

The actuarial features are the **only features that model the TIME DIMENSION of risk** — not just "how much are we losing?" but "how long does this kind of loss typically last, and what's the probability of recovery given how long we've already been in it?"

---

## The Complete Mathematical Pipeline

```
┌──────────────────┐
│  Raw Prices      │
│  P₁, P₂, ..., Pₜ │
└────────┬─────────┘
         │
         ▼
┌──────────────────────────┐
│  1. Compute Drawdowns    │     Drawdownₜ = (Pₜ - max(P₁..Pₜ)) / max(P₁..Pₜ)
│     & Running Maximum    │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│  2. Extract Events       │     Each episode: {start, end, duration, max_depth}
│     (Peak → Trough → Peak)│     e.g., 50-200 events over 18 years
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────┐
│  3. Fit Models (per severity bucket)                     │
│                                                          │
│  ┌─ Severity CDF ──────────────────────┐                │
│  │  Sort all max_depths                 │                │
│  │  Percentile = rank(d) / n            │ ──→ Reserve_Severity
│  └──────────────────────────────────────┘                │
│                                                          │
│  ┌─ Kaplan-Meier Survival ─────────────┐                │
│  │  Ŝ(t) = ∏(1 - dᵢ/nᵢ)              │                │
│  │  For each bucket:                    │                │
│  │    Prob_30d = 1 - S(t+30)/S(t)       │ ──→ Prob_30d  │
│  │    Prob_60d = 1 - S(t+60)/S(t)       │ ──→ Prob_60d  │
│  │    Expected = max(0, median - t)     │ ──→ Expected_Recovery
│  └──────────────────────────────────────┘                │
└──────────────────────────────────────────────────────────┘
```

---

## Summary of Mathematical Notation

| Symbol | Meaning |
|---|---|
| $P_t$ | Asset price at time $t$ |
| $P_{\text{peak}}$ | Running maximum price up to time $t$ |
| $d$ | Drawdown depth (positive number, e.g., 0.15 = 15% drop) |
| $S(t)$ | Survival function — probability drawdown lasts > $t$ days |
| $\hat{S}(t)$ | Kaplan-Meier estimate of $S(t)$ |
| $d_i$ | Number of recoveries at time $t_i$ |
| $n_i$ | Number of drawdowns still ongoing at time $t_i$ (risk set) |
| $h$ | Prediction horizon (30 or 60 days) |
| $t$ | Days already elapsed in current drawdown |

---

## References

- **Kaplan-Meier Estimator**: Kaplan, E.L. & Meier, P. (1958). "Nonparametric Estimation from Incomplete Observations." *JASA*, 53(282), 457–481.
- **Chain Ladder Method**: Mack, T. (1993). "Distribution-free calculation of the standard error of chain ladder reserve estimates." *ASTIN Bulletin*, 23(2), 213–225.
- **Survival Analysis in Finance**: Lando, D. (2004). *Credit Risk Modeling: Theory and Applications*. Princeton University Press.
