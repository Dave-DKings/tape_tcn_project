"""
Actuarial Documentation Builder
Builds comprehensive Actuarial Features documentation
"""
import json

def build_actuarial_docs():
    """Build comprehensive Actuarial Features documentation"""
    
    cells = []
    
    # === TITLE ===
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["# Actuarial Features Reference\n", "\n",
                  "**Version**: 2.0 (Expanded February 2026)\n", "\n", 
                  "**Scope**: Technical documentation of the actuarial feature block, which enriches the RL agent's state space with non-Markovian drawdown recovery dynamics and tail risk estimations.\n"]
    })
    
    # === Introduction ===
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Introduction: Why Actuarial Features? <a id='section1'></a>\n",
            "\n",
            "Standard financial RL agents often treat price history as a Markovian state (Price, Returns, Volumes). However, **drawdowns** are inherently path-dependent and non-Markovian: the \"pain\" and \"recovery probability\" depend heavily on how *long* the drawdown has persisted and how *severe* it is relative to history.\n",
            "\n",
            "We integrate **Actuarial Science** (specifically Survival Analysis and Chain Ladder theory) to give the TCN agent explicit \"situational awareness\" about:\n",
            "1.  **Recovery Time**: How long until we break even?\n",
            "2.  **Survival Probability**: What are the odds of recovering in the next 30/60 days?\n",
            "3.  **Reserve Severity**: How bad is this current crash relative to all historical crashes?\n"
        ]
    })

    # === Section 2: Core Drawdown Definitions ===
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Core Drawdown Definitions <a id='section2'></a>\n",
            "\n",
            "For a price series $P_t$, we define a **Drawdown Event** starting at $t_0$:\n",
            "\n",
            "**Drawdown Depth** $D_t$:\n",
            "$$\n",
            "D_t = 1 - \\frac{P_t}{\\max_{s \\le t} P_s}\n",
            "$$\n",
            "\n",
            "**Drawdown Age** $e_t$:\n",
            "$$\n",
            "e_t = t - t_{\\text{last peak}}\n",
            "$$\n",
            "\n",
            "The combination $(D_t, e_t)$ forms the **Actuarial State** of the asset."
        ]
    })
    
    # === Section 3: Recovery Time (Survival Analysis) ===
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Recovery Time Estimation (Survival Analysis) <a id='section3'></a>\n",
            "\n",
            "We model the \"time to recovery\" $T$ using the **Kaplan-Meier Estimator** for the survival function $S(t) = P(T > t)$.\n",
            "\n",
            "### 3.1 Conditional Probability\n",
            "Given a drawdown has already lasted $e_t$ days, what is the probability it recovers within next $h$ days?\n",
            "\n",
            "$$\n",
            "P(\\text{Recovers} \\in [t, t+h] \\mid T > e_t) = 1 - \\frac{S(e_t + h)}{S(e_t)}\n",
            "$$\n",
            "\n",
            "We compute this for:\n",
            "- **30-Day Probability** (`Actuarial_Prob_30d`)\n",
            "- **60-Day Probability** (`Actuarial_Prob_60d`)\n",
            "\n",
            "### 3.2 Expected Remaining Time\n",
            "The expected *remaining* time to recovery is derived from the median survival time:\n",
            "\n",
            "$$\n",
            "E[\\text{Remaining}] = \\max(0, \\text{Median}(S) - e_t)\n",
            "$$\n",
            "\n",
            "This provides the feature `Actuarial_Expected_Recovery`."
        ]
    })
    
    # === Section 4: Reserve Severity (Tail Risk) ===
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Reserve Severity (Tail Risk) <a id='section4'></a>\n",
            "\n",
            "To normalize \"how bad\" a crash is across different assets, we use an **Empirical CDF** of historical max drawdowns.\n",
            "\n",
            "For a current depth $D_t$, the **Severity Score** $\\sigma_t$ is:\n",
            "\n",
            "$$\n",
            "\\sigma_t = F_{\\text{historic inverse}}(D_t)\n",
            "$$\n",
            "\n",
            "Where $F$ is the cumulative distribution function of all *completed* historical drawdowns. \n",
            "- $\\sigma_t \\approx 0.0$: Minor noise.\n",
            "- $\\sigma_t \\approx 0.99$: Statistical \"Black Swan\" event for this asset.\n",
            "\n",
            "This provides the feature `Actuarial_Reserve_Severity`."
        ]
    })
    
    # === Section 5: Code Implementation Map ===
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Code Implementation Map <a id='section5'></a>\n",
            "\n",
            "### `src/actuarial.py`\n",
            "The logic is encapsulated in `DrawdownReserveEstimator`.\n",
            "\n",
            "- **`fit(price_history)`**: \n",
            "  - Extracts events (`_extract_drawdown_events`).\n",
            "  - Fits Kaplan-Meier curves (`_fit_survival_models`).\n",
            "  - Fits severity CDF (`_fit_severity_cdf`).\n",
            "\n",
            "- **`predict(current_drawdown, days_elapsed)`**: \n",
            "  - Calculates conditional probabilities using survival curves.\n",
            "  - Looks up severity percentile.\n",
            "\n",
            "### `src/data_utils.py`\n",
            "- **`add_actuarial_features()`**: \n",
            "  - Manages the **expanding window** (No-Lookahead).\n",
            "  - Fits the estimator on data $[0, t-1]$ to predict at $t$.\n",
            "  - Falls back to defaults during warm-up period (first 252 days)."
        ]
    })
    
    # === Section 6: Integration with TCN ===
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6. Integration with TCN State Space <a id='section6'></a>\n",
            "\n",
            "These 4 features are appended to the standard price/volume features.\n",
            "\n",
            "1. **Normalization**: They are standard scaled (z-score) along with other features.\n",
            "2. **State Vector**: \n",
            "   $$ S_t = [ \\text{Returns}_t, \\text{Vol}_t, \\dots, \\sigma_t, P(\\text{Rec}<30), E[\\text{Time}] ] $$\n",
            "3. **Agent Utility**: \n",
            "   - **Deep Drawdowns**: High severity signals \"cheap\" buying opportunities (mean reversion).\n",
            "   - **Stagnant Drawdowns**: low probability of 30d recovery signals \"dead money\" (avoid).\n",
            "\n",
            "This empowers the TCN to make \"actuarially sound\" decisions during market stress."
        ]
    })

    # Validate output
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"}
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    
    return notebook

if __name__ == "__main__":
    nb = build_actuarial_docs()
    output_path = r"c:\Users\Owner\new_project\adaptive_portfolio_rl\tcn_documentation\08_actuarial_features_reference.ipynb"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print(f"Created actuarial docs with {len(nb['cells'])} cells")
