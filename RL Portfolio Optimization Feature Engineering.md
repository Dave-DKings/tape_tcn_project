# **State-of-the-Art Feature Engineering and State Representation for Reinforcement Learning Portfolio Optimization Agents**

The evolution of quantitative finance from linear, frequentist models to autonomous, adaptive agents marks a definitive transition in how market complexity is managed. In the contemporary landscape of 2024 and 2025, reinforcement learning (RL) has emerged as the premier framework for dynamic asset allocation, primarily due to its ability to treat portfolio optimization as a sequential decision-making process under uncertainty.1 However, the success of a Deep Reinforcement Learning (DRL) agent is not merely a function of the algorithm—such as Proximal Policy Optimization (PPO), Advantage Actor-Critic (A2C), or G-learning—but is fundamentally dictated by the architecture of its state space.3 The variables selected for an agent’s observation space must encapsulate price momentum, volatility regimes, fundamental health, macroeconomic context, and systemic risk, all while being processed to maintain stationarity and minimize the noise that frequently leads to the catastrophic failure of policy gradients.5

## **The Foundation of State Representation: Price and Return Dynamics**

At the core of the Markov Decision Process (MDP) for portfolio optimization lies the price-derivative suite, predominantly represented by log-returns. The user’s variable set includes LogReturn\_1d, LogReturn\_5d, LogReturn\_10d, and LogReturn\_21d. In the state of the art, log-returns are favored over simple arithmetic returns because they provide time-additivity and are more likely to exhibit the stationarity required for stable neural network training.5 The mathematical formulation for these features, ![][image1], allows the agent to observe multi-scale temporal patterns.

Recent research into Transformer-Enhanced Deep Reinforcement Learning suggests that providing a hierarchy of returns—from 1-day to 21-day windows—enables the agent to learn long-term temporal correlations that are often missed by single-horizon models.2 However, the inclusion of LogReturn\_1d\_ZScore and CrossSectional\_ZScore\_LogReturn\_1d introduces a critical layer of cross-sectional awareness. In a portfolio context, knowing that an asset returned 2% is less informative than knowing that it returned 2% while the market average was 0.5%. This cross-sectional normalization allows the agent to differentiate between idiosyncratic alpha and beta-driven market movements, which is essential for effective relative-weighting strategies.8

### **Short-Term Reversal and Momentum Dynamics**

The inclusion of ShortTerm\_Reversal\_5 alongside Residual\_Momentum\_21 represents a sophisticated approach to capturing the dual nature of equity returns. Short-term reversal metrics exploit the tendency of prices to mean-revert over brief horizons due to liquidity imbalances, while residual momentum filters out market-wide trends to isolate stock-specific strength. For an RL agent, these variables act as "strategy gates"; the agent may learn a policy that buys during short-term reversals in a long-term uptrend, a behavior often referred to as "buying the dip".10 To handle these effectively, SOTA approaches utilize a "rolling rank" normalization, where the ShortTerm\_Reversal\_5 value is mapped to its percentile within a 252-day window. This prevents the agent from reacting over-sensitively to standard price fluctuations while ensuring it recognizes extreme mean-reversion opportunities.

## **Volatility and Higher-Order Risk Moments**

Volatility representation has progressed beyond simple standard deviation. The user’s variables RollingVolatility\_21d, DownsideSemiVar\_21d, RealizedSkew\_21d, and RealizedKurtosis\_21d provide the agent with a comprehensive view of the risk distribution’s shape.

### **Asymmetric Risk and Downside Semi-Variance**

Traditional volatility metrics treat upside and downside movements identically. However, portfolio optimization agents often operate under objective functions that penalize drawdowns more severely than they reward gains, such as the Sharpe or Sortino ratios.7 DownsideSemiVar\_21d is a critical signal in this regard, as it isolates the variance of negative returns. By providing this to the agent, the policy can learn to deleverage specifically when "bad volatility" increases, even if the overall market volatility remains stable. The state-of-the-art approach for normalizing downside variance is the use of a log-transform followed by a robust scaler:

![][image2]  
This transformation compresses the extreme outliers typical of financial stress periods, ensuring that the neural network’s gradients do not explode during a market crash.11

### **Skewness, Kurtosis, and Vol-of-Vol**

The inclusion of RealizedSkew\_21d and RealizedKurtosis\_21d allows the agent to anticipate "fat-tail" events. A spike in realized kurtosis suggests that the market is entering a regime of extreme, infrequent moves, which should prompt the agent to seek insurance or shift toward defensive assets. Similarly, VolOfVol\_63 (Volatility of Volatility) informs the agent about the stability of the current risk environment. High Vol-of-Vol indicates that the current volatility estimate is unreliable, requiring the agent to decrease its confidence in mean-variance assumptions. In 2024, agents trained with Vol-of-Vol features have shown significantly better performance in "crash-avoidance" compared to those using only simple rolling volatility.2

| Risk Variable | Market Implication | SOTA Normalization |
| :---- | :---- | :---- |
| **RollingVolatility\_21d** | Broad uncertainty level | Rolling Z-Score or Percentile |
| **DownsideSemiVar\_21d** | Specific risk of loss | Robust Scaler (Median/IQR) |
| **RealizedSkew\_21d** | Directional bias of tail risk | Min-Max Bounded \[-1, 1\] |
| **VolOfVol\_63** | Stability of risk regime | Log-differencing and Standardizing |

## **Technical Trend and Momentum Oscillators**

The provided indicator suite—comprising EMAs, Bollinger Bands, MACD, RSI, and Stochastics—represents the "heuristic layer" of the state space. While these are traditional technical analysis tools, their integration into RL requires careful consideration of redundancy and stationarity.5

### **The Redundancy of Trend Indicators**

A significant risk in the user’s current list is the high multi-collinearity between trend indicators. EMA\_12, EMA\_26, SMA\_50, and the BBM\_20\_2.0 (Bollinger Middle Band) all capture the central tendency of the price over similar horizons. Providing all of these to a Deep Q-Network (DQN) or PPO agent often introduces "noise" rather than "signal," as the agent may assign conflicting weights to identical underlying patterns, leading to policy instability.5

State-of-the-art research suggests that instead of raw moving average levels, which are non-stationary and share the scale of the price, agents should receive "Relative Distance" features. For example, Regime\_Price\_vs\_SMA\_Short represents the percentage deviation of the current price from the moving average. This transformation creates a mean-stationary series that allows the agent to generalize across different price levels—a stock trading at $100 and a stock trading at $1000 can both be "2% above their SMA," allowing the agent to learn a universal policy.8

### **MACD and Oscillator Convergence**

The MACD suite (MACD\_12\_26\_9, MACDh, MACDs) provides information on trend acceleration. The most valuable component for an RL agent is often the MACDh (Histogram), which represents the divergence between the signal line and the MACD line. This acts as a "second-order" derivative of price, informing the agent not just that the price is rising, but whether the rise is accelerating or slowing down.

Momentum oscillators such as RSI\_14, STOCHk\_14\_3\_3, STOCHd\_14\_3\_3, WILLR\_14, and MFI\_14 (Money Flow Index) are naturally bounded between 0 and 100 or \-100 and 0\. While these are "agent-ready," state-of-the-art normalization often shifts them to a ![][image3] range centered at 0\. For instance:

![][image4]  
This centering is crucial for activation functions like Tanh or Leaky ReLU, as it aligns the "overbought" and "oversold" signals with the positive and negative output ranges of the neurons.5

## **Volume, Flow, and Liquidity Representation**

Volume is the primary validator of price action. The variables OBV (On-Balance Volume), OBV\_Delta\_Norm\_21, VOL\_SMA\_20, and Volume\_Percentile\_63 provide a multi-dimensional view of market participation.

### **Handling Volume Spikes with Percentiles**

One of the greatest challenges in financial feature engineering is the "fat-tail" nature of volume. A single high-volume day (e.g., an earnings announcement) can be an order of magnitude larger than the mean. If normalized using a standard Z-score, these spikes compress all other volume data to near-zero, effectively destroying the signal. The user’s inclusion of Volume\_Percentile\_63 is a state-of-the-art solution. By mapping the current volume to its 63-day percentile rank, the agent receives a uniform distribution $$ that is robust to outliers and informs it whether the current participation is "high" or "low" relative to recent history.5

### **On-Balance Volume and Capital Flow**

OBV in its raw form is a non-stationary running total that trends indefinitely. It is largely unusable for RL in this state. However, the user’s OBV\_Delta\_Norm\_21 is a highly effective transformation. It measures the change in OBV over a 21-day window and normalizes it, likely by the total volume over that period. This provides a "Net Capital Flow" signal. When combined with price returns, it allows the agent to distinguish between "strong" moves (price up on high OBV) and "weak" moves (price up on low or declining OBV).15

## **Macroeconomic and Fixed Income: The Global State**

A defining feature of the next generation of RL agents (2024-2025) is the integration of "Global State" variables. While technical indicators describe the "how" of price movement, macroeconomic variables like EFFR, SOFR, DGS10, and YieldCurve\_Spread describe the "why." These features allow the agent to learn regime-specific policies.9

### **Interest Rate Levels and Z-Scores**

The user provides absolute levels (EFFR\_level, SOFR\_level) and their standardized versions (EFFR\_zscore, FEDFUNDS\_zscore). In a reinforcement learning context, absolute levels are often non-stationary and can lead to "distributional shift" when the market moves from a zero-interest-rate policy (ZIRP) to a high-rate environment. The zscore and diff (first difference) variables are superior for RL. For example, EFFR\_diff informs the agent of a "policy shock," which may trigger a discrete state transition in the agent's internal representation of market risk.9

### **The Yield Curve and Inflation Expectations**

The YieldCurve\_Spread and YieldCurve\_Inverted\_Flag are powerful predictors of business cycles. An inverted yield curve is a "regime-shift" signal that can be used as a "gate" in the agent's neural network. Similarly, TIPS10Y\_level and BreakevenInf10Y\_level provide the agent with a view of real rates and inflation expectations.

* **Real Rate (![][image5]):** High real rates generally increase the discount rate for future cash flows, hurting growth stocks.  
* **Breakeven Inflation:** Rising inflation expectations may lead the agent to favor commodities or energy over fixed-income assets.

To handle these variables, SOTA approaches often use **Normalize-and-Project (NaP)**. NaP ensures that even if interest rates move from 1% to 5%, the "effective learning rate" of the agent remains constant, preventing it from becoming "blind" to small rate changes in a high-rate regime.11

| Fixed Income Variable | Information Provided | Redundancy Note |
| :---- | :---- | :---- |
| **DGS10\_level** | Long-term rate baseline | Essential Global State |
| **T10Y2Y\_level** | Yield curve slope | Highly predictive of regimes |
| **EFFR\_diff** | Monetary policy velocity | Superior to absolute levels |
| **IG\_Credit\_zscore** | Systemic liquidity risk | Critical for crash avoidance |

## **Fundamental Health and the Latency Challenge**

The inclusion of fundamental variables like Fundamental\_FCFE\_Delta, Fundamental\_Revenue\_Delta, and Fundamental\_NCFO\_Delta allows the agent to incorporate "Value" and "Quality" signals. However, fundamental data introduces a unique problem in RL: **Staleness**.16

### **Fundamental Staleness and Relevance Decay**

Unlike price data, which is updated every minute, fundamental data is updated quarterly. The variables Fundamental\_Staleness\_Days and Fundamental\_Staleness\_Quarters are sophisticated additions that inform the agent about the "age" of the fundamental signal. In a state-of-the-art DRL framework, these staleness metrics can be used to apply a "decay function" to the fundamental features. As Staleness\_Days increases, the weight of the Revenue\_Delta in the agent’s decision-making process should decrease, as the information becomes "priced in" or outdated.16

### **Actuarial Risk and Recovery Metrics**

Variables such as Actuarial\_Prob\_30d, Actuarial\_Prob\_60d, and Actuarial\_Reserve\_Severity suggest a specialized agent capable of managing credit-sensitive or distressed assets. These actuarial probabilities act as "survival filters." Even if an asset has high momentum (MomentumRank\_21d), a spike in Actuarial\_Prob\_30d (probability of a credit event) should override the momentum signal, prompting an immediate exit. This hierarchical logic—where fundamental risk overrides technical momentum—is a hallmark of "Safety-First" RL agents developed for institutional deployment in 2025\.16

## **Structural Factor Analysis and Market Regimes**

The variables Regime\_Volatility\_Ratio, Regime\_Momentum\_Long, and Regime\_Breadth\_Positive are high-level synthetic features that define the environment's current "state."

### **Covariance Eigenvalues and Correlation Breakdowns**

The inclusion of Covariance\_Eigenvalue\_0, Covariance\_Eigenvalue\_1, and Covariance\_Eigenvalue\_2 is an advanced quantitative technique. These represent the variance explained by the primary factors of the asset universe's covariance matrix:

1. **Eigenvalue 0:** Usually represents the "Market Factor." A spike here suggests that all stocks are moving together, indicating high systemic risk and low diversification potential.  
2. **Eigenvalues 1 & 2:** Represent "Style" or "Sector" factors (e.g., Growth vs. Value). For an RL agent, these values provide the "internal geometry" of the market. During a "correlation breakdown" (where Eigenvalue 0 dominates), the agent should learn to reduce its gross exposure, as the benefits of portfolio diversification have vanished.8

### **Beta, BetaRank, and Flags**

Beta\_to\_Market and BetaRank inform the agent about an asset’s sensitivity to the broader market. The user’s list includes binary flags: HighBeta\_Flag and LowBeta\_Flag. In contemporary DRL, binary flags are often considered redundant if the continuous variable (Beta) is already present. Neural networks are exceptionally efficient at learning threshold-based "flags" from continuous data. Including them as separate features often adds unnecessary dimensions to the state space, which can slow down convergence.18

## **Feature Selection and Denoising: The Quest for Signal**

With over 90 variables, the agent faces the "Curse of Dimensionality." Many of these variables are noisy, redundant, or purely spurious. State-of-the-art handling involves rigorous feature selection and denoising before the data ever reaches the RL agent.18

### **Noise-Augmented Bootstrap Feature Selection (NABFS)**

The gold standard for identifying informative features in 2024 is the **NABFS** framework. This method involves:

1. **Augmentation:** Inserting several "shadow" features into the dataset—variables that are purely random noise but share the same statistical distribution as the real data.  
2. **Importance Benchmarking:** Calculating the SHAP (SHapley Additive exPlanations) importance of every feature.  
3. **Pruning:** Any feature (e.g., WILLR\_14 or BBU\_20\_2.0) that does not significantly outperform the "noise" features is permanently removed from the state space.19 This ensures that the agent is not "hallucinating" patterns in indicators that are mathematically redundant or historically irrelevant.

### **Denoising Autoencoders and Latent Representations**

Rather than providing raw technical indicators, many SOTA agents use a **Stacked Sparse Denoising Autoencoder (SDAE)**. The 90+ variables are fed into an autoencoder that is trained to reconstruct the original signals from a compressed "latent" layer (e.g., 16 or 32 dimensions). This process filters out the high-frequency "jitter" and captures the "hidden state" of the market. The RL agent then learns its policy based on this 16-dimensional latent vector, which significantly speeds up training and improves out-of-sample generalization.3

## **Normalization SOTA: Solving Plasticity Loss**

Reinforcement learning on financial time series is uniquely prone to **Plasticity Loss**—a phenomenon where the neural network's weights become stagnant and the agent stops learning as the market environment changes. This is almost always caused by improper normalization of features like volume or interest rates.11

### **Divisive vs. Range Normalization**

Recent studies in both biological and artificial RL identify two primary ways to normalize rewards and inputs: **Divisive Normalization** and **Range Normalization**.22

* **Range Normalization:** ![][image6]. This is ideal for bounded technical indicators like RSI or the YieldCurve\_Inverted\_Flag.  
* **Divisive Normalization:** ![][image7] or ![][image8]. This is superior for log-returns and volume-based features, as it maintains the relative magnitude of price "shocks" while keeping the values within a manageable range for the activation functions.11

### **Normalize-and-Project (NaP) for Macro Variables**

For non-stationary macro variables like EFFR or DGS10, the **Normalize-and-Project** protocol is the current state of the art. NaP inserts normalization layers before each non-linearity in the network and maintains a constant parameter norm. This ensures that the "effective learning rate" remains stable even if the interest rate spreads (T10Y2Y\_level) move into territory the agent has never seen before, such as extreme inversion.11

## **Identifying Redundancy and Noise: A Pruning Roadmap**

Based on the synthesis of SOTA literature and quantitative analysis, the following variables in the user's list are identified as highly redundant or likely to introduce noise:

1. **Trend Overlap Cluster:** EMA\_12, EMA\_26, BBM\_20\_2.0, and SMA\_50.  
   * *Recommendation:* Keep only Regime\_Price\_vs\_SMA\_Short and Regime\_Price\_vs\_SMA\_Long. Remove the raw levels to maintain stationarity.8  
2. **Oscillator Overlap Cluster:** RSI\_14, STOCHk\_14\_3\_3, STOCHd\_14\_3\_3, and WILLR\_14.  
   * *Recommendation:* Keep RSI\_14 and STOCHd (the smoothed stochastic). WILLR is mathematically nearly identical to STOCHk and introduces redundant dimensions.5  
3. **Macro-Interest Rate Cluster:** EFFR\_level, FEDFUNDS\_level, and SOFR\_level.  
   * *Recommendation:* Keep only SOFR\_level and SOFR\_diff. SOFR has become the standard replacement for LIBOR and EFFR in modern credit markets. Including all three is purely redundant as they move in near-perfect correlation.9  
4. **Binary Beta Flags:** HighBeta\_Flag and LowBeta\_Flag.  
   * *Recommendation:* Remove. The agent can derive these from Beta\_to\_Market or BetaRank. Binary flags can create "discontinuities" in the policy gradient that impede smooth learning.18  
5. **Volatility Overlap:** RollingVolatility\_21d and RollingVolatility\_21d\_ZScore.  
   * *Recommendation:* Keep only the ZScore or the Rank. Raw volatility has a wide range that can destabilize weights; the standardized versions are much safer for DRL agents.9

| Pruning Action | Redundant Variables | Keep Instead | Logic |
| :---- | :---- | :---- | :---- |
| **Consolidate Rates** | EFFR, FEDFUNDS, DGS2 | SOFR, DGS10, T10Y2Y | The spread and the 10Y level capture the curve’s essence with fewer dimensions. |
| **Streamline Momentum** | WILLR, STOCHk, MFI | RSI, STOCHd | RSI and smoothed Stochastics provide a cleaner signal with less high-frequency noise. |
| **Normalize Volume** | OBV, VOL\_SMA\_20 | OBV\_Delta\_Norm, Vol\_Percentile | Percentiles and Deltas are stationary; raw totals and averages are not. |
| **Simplify Flags** | HighBeta\_Flag, LowBeta\_Flag | BetaRank | Continuous ranks allow for more nuanced policy gradients than binary switches. |

## **Deep Insights into State-of-the-Art Feature Integration**

The integration of these variables into a cohesive state space requires more than just selection; it requires a deep understanding of the second and third-order relationships between them.

### **Interaction between Actuarial Risk and Credit Cycles**

A critical 2025 insight is the interaction between Actuarial\_Prob\_30d and HY\_Credit\_zscore. In isolation, a rising credit spread (HY\_Credit\_zscore) might merely suggest a shift toward value stocks. However, when paired with an increase in Actuarial\_Prob\_30d, it indicates a high-probability systemic liquidity event. An RL agent can learn this non-linear interaction—effectively creating a "Crash Mode" that overrides all other alpha signals.16

### **Covariance Eigenvalues as Leverage Constraints**

The first eigenvalue of the covariance matrix (Covariance\_Eigenvalue\_0) should be viewed as a dynamic "leverage constraint." When this value is high, the "effective dimensionality" of the market is low—everything is moving together. In this state, even a diversified portfolio is exposed to a single risk factor. State-of-the-art agents use this as a scaling factor for their "Action Space," reducing the magnitude of their position sizes when systemic correlation spikes.8

### **The Role of Staleness in Factor Decay**

Fundamental\_Staleness\_Days is not just a metadata field; it is a "confidence" feature. For an agent utilizing Fundamental\_Revenue\_Delta, the SHAP importance of that delta should be a function of the staleness. An agent that does not "see" the staleness will mistakenly apply a three-month-old revenue signal to a high-frequency trading decision, leading to a mismatch in information horizons. Successful agents in 2024 learn to "gate" fundamental signals using their staleness metrics.16

## **SOTA Normalization and Variable Handling Summary**

The final stage of feature engineering for a portfolio agent is the application of a rigorous normalization matrix.

| Variable Class | State of the Art Handling | Reasoning |
| :---- | :---- | :---- |
| **Returns** | Log-transform \+ Cross-Sectional Z-Score | Ensures stationarity and relative value awareness.8 |
| **Macro/Rates** | First-differencing \+ Normalize-and-Project (NaP) | Prevents plasticity loss during rate regime shifts.11 |
| **Volume/Flow** | Rolling Percentile Rank (e.g., 63-day) | Handles fat-tails and extreme volume spikes.5 |
| **Fundamentals** | Delta-Differencing \+ Staleness Decay | Captures growth velocity while accounting for data latency.16 |
| **Oscillators** | Mean-Centering to ![][image3] | Aligns overbought/oversold signals with neuron activations.14 |
| **Risk Metrics** | Robust Scaler (Median/IQR) | Protects gradients from volatility-driven outliers.2 |

## **Strategic Recommendations for Agent Training**

The user’s portfolio agent should not be trained on the raw variable set. Instead, the following strategic pipeline is recommended based on the identified 2024-2025 best practices:

1. **Denoising with Autoencoders:** Pass the consolidated 60-70 features (after removing the identifies redundancies) through a Denoising Autoencoder to extract a lower-dimensional latent representation.3  
2. **Regime-Dependent Reward Shaping:** Use the Regime\_Volatility\_Ratio and Covariance\_Eigenvalue\_0 to adjust the agent's reward function. In high-risk regimes, the agent should be rewarded more for capital preservation (e.g., penalizing drawdown more heavily).7  
3. **Sim-to-Real Adaptation:** Since financial data is notoriously non-stationary, use the "Random Noise Augmentation" technique. Train the agent on the real data augmented with synthetic "stress scenarios" (e.g., simulating a 5-standard deviation spike in HY\_Credit\_zscore). This prepares the agent for "Black Swan" events that are under-represented in the historical training set.14  
4. **SHAP-Based Auditing:** During the validation phase, use SHAP values to ensure the agent is not over-weighting redundant indicators like EMA\_12. If EMA\_12 and EMA\_26 both show high SHAP values, it is a definitive sign of redundancy that will likely lead to over-trading and high transaction costs in live environments.24

## **Conclusions and Practical Synthesis**

Developing a reinforcement learning agent for portfolio optimization in the current era requires a shift from "broad feature collection" to "rigorous state representation." The user’s variable set is an excellent foundation, covering every major pillar of financial analysis. However, the raw inclusion of 90+ features will likely overwhelm a standard DRL agent, leading to the "curse of dimensionality" and poor convergence.

The path to a state-of-the-art agent lies in the **hierarchical pruning** of redundant trend indicators and oscillators, the **stationary transformation** of macroeconomic levels into differences and Z-scores, and the **sophisticated handling of volume** through percentile ranking. Furthermore, the inclusion of "internal" features like Fundamental\_Staleness\_Days and "structural" features like Covariance\_Eigenvalues differentiates a professional-grade agent from a basic algorithmic strategy. By applying modern normalization techniques such as **NaP** and **Robust Scaling**, the user can ensure the agent maintains its "plasticity"—the ability to learn and adapt—even as the global financial environment transitions between inflation regimes, credit cycles, and volatility states.11 This approach not only enhances the agent’s alpha-generating potential but also provides a robust framework for managing the systemic risks that define 21st-century markets.

#### **Works cited**

1. \[2511.18076\] Reinforcement Learning for Portfolio Optimization with a Financial Goal and Defined Time Horizons \- arXiv.org, accessed February 18, 2026, [https://arxiv.org/abs/2511.18076](https://arxiv.org/abs/2511.18076)  
2. AI-Driven Portfolio Optimization System for Dynamic Asset Allocation | Advances in Consumer Research, accessed February 18, 2026, [https://acr-journal.com/article/ai-driven-portfolio-optimization-system-for-dynamic-asset-allocation-1838/](https://acr-journal.com/article/ai-driven-portfolio-optimization-system-for-dynamic-asset-allocation-1838/)  
3. (PDF) Deep Reinforcement Learning Approach to Portfolio Optimization in the Australian Stock Market \- ResearchGate, accessed February 18, 2026, [https://www.researchgate.net/publication/384149782\_Deep\_Reinforcement\_Learning\_Approach\_to\_Portfolio\_Optimization\_in\_the\_Australian\_Stock\_Market](https://www.researchgate.net/publication/384149782_Deep_Reinforcement_Learning_Approach_to_Portfolio_Optimization_in_the_Australian_Stock_Market)  
4. DeepStock: Reinforcement Learning with Policy Regularizations for Inventory Management, accessed February 18, 2026, [https://papers.ssrn.com/sol3/Delivery.cfm/5784782.pdf?abstractid=5784782\&mirid=1](https://papers.ssrn.com/sol3/Delivery.cfm/5784782.pdf?abstractid=5784782&mirid=1)  
5. Feature Engineering in Reinforcement Learning for Algorithmic Trading \- TU Delft Repository, accessed February 18, 2026, [https://repository.tudelft.nl/file/File\_f1226238-ebc5-4691-b687-3eb4c5e5663c?preview=1](https://repository.tudelft.nl/file/File_f1226238-ebc5-4691-b687-3eb4c5e5663c?preview=1)  
6. ReLMM: Reinforcement Learning Optimizes Feature Selection in Modeling Materials \- PMC, accessed February 18, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC11734688/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11734688/)  
7. Application of Deep Reinforcement Learning to At-the-Money S\&P 500 Options Hedging \- arXiv.org, accessed February 18, 2026, [https://arxiv.org/html/2510.09247v1](https://arxiv.org/html/2510.09247v1)  
8. Reinforcement learning meets technical analysis: combining moving ..., accessed February 18, 2026, [https://www.tandfonline.com/doi/full/10.1080/23322039.2025.2490818](https://www.tandfonline.com/doi/full/10.1080/23322039.2025.2490818)  
9. Using Machine Learning on Macroeconomic, Technical, and ... \- MDPI, accessed February 18, 2026, [https://www.mdpi.com/2078-2489/16/7/584](https://www.mdpi.com/2078-2489/16/7/584)  
10. Reinforcement Learning for Trading Strategies A Reproducible Comparison with Classical Baselines \- SSRN, accessed February 18, 2026, [https://papers.ssrn.com/sol3/papers.cfm?abstract\_id=6018997](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6018997)  
11. Normalization and effective learning rates in reinforcement learning \- NIPS, accessed February 18, 2026, [https://proceedings.neurips.cc/paper\_files/paper/2024/file/c04d37be05ba74419d2d5705972a9d64-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2024/file/c04d37be05ba74419d2d5705972a9d64-Paper-Conference.pdf)  
12. Machine learning techniques for technical indicator selection \- Consensus, accessed February 18, 2026, [https://consensus.app/search/machine-learning-techniques-for-technical-indicato/J8SfhnqzSPywjUKT8d\_Nag/](https://consensus.app/search/machine-learning-techniques-for-technical-indicato/J8SfhnqzSPywjUKT8d_Nag/)  
13. An Algorithmic Trading Approach Merging Machine Learning With Multi-Indicator Strategies for Optimal Performance \- ResearchGate, accessed February 18, 2026, [https://www.researchgate.net/publication/387006426\_An\_Algorithmic\_Trading\_Approach\_Merging\_Machine\_Learning\_With\_Multi-Indicator\_Strategies\_for\_Optimal\_Performance](https://www.researchgate.net/publication/387006426_An_Algorithmic_Trading_Approach_Merging_Machine_Learning_With_Multi-Indicator_Strategies_for_Optimal_Performance)  
14. Reinforcement Learning-Based Market Making as a Stochastic Control on Non-Stationary Limit Order Book Dynamics, accessed February 18, 2026, [https://arxiv.org/html/2509.12456v2](https://arxiv.org/html/2509.12456v2)  
15. A3C-Feature selection-Stock market prediction: part three | by A ..., accessed February 18, 2026, [https://medium.com/@abatrek059/a3c-feature-selection-stock-market-prediction-part-three-d0871947e65d](https://medium.com/@abatrek059/a3c-feature-selection-stock-market-prediction-part-three-d0871947e65d)  
16. \[2403.07916\] Advancing Investment Frontiers: Industry-grade Deep Reinforcement Learning for Portfolio Optimization \- arXiv, accessed February 18, 2026, [https://arxiv.org/abs/2403.07916](https://arxiv.org/abs/2403.07916)  
17. Automated Trading Framework Using LLM-Driven Features and Deep Reinforcement Learning \- MDPI, accessed February 18, 2026, [https://www.mdpi.com/2504-2289/9/12/317](https://www.mdpi.com/2504-2289/9/12/317)  
18. Reinforcement Learning for feature selection | Towards Data Science, accessed February 18, 2026, [https://towardsdatascience.com/reinforcement-learning-for-feature-selection-be1e7eeb0acc/](https://towardsdatascience.com/reinforcement-learning-for-feature-selection-be1e7eeb0acc/)  
19. A Feature Selection Technique Through Noise-Based Hypothesis Testing \- arXiv, accessed February 18, 2026, [https://arxiv.org/pdf/2511.20851](https://arxiv.org/pdf/2511.20851)  
20. Automation and Feature Selection Enhancement with Reinforcement Learning (RL) \- arXiv, accessed February 18, 2026, [https://arxiv.org/html/2503.11991v1](https://arxiv.org/html/2503.11991v1)  
21. Feature Importance Metrics: Gain, Permutation, and SHAP Values \- Machine Learning Interview Guide | bugfree.ai, accessed February 18, 2026, [https://bugfree.ai/knowledge-hub/feature-importance-metrics-gain-permutation-shap-values](https://bugfree.ai/knowledge-hub/feature-importance-metrics-gain-permutation-shap-values)  
22. The functional form of value normalization in human reinforcement learning \- PMC \- NIH, accessed February 18, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC10393293/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10393293/)  
23. A Machine Learning Model for Predicting the SOFR Term-structure \- Spectrum: Concordia University Research Repository, accessed February 18, 2026, [https://spectrum.library.concordia.ca/996135/1/Guo\_MSc\_F2025.pdf](https://spectrum.library.concordia.ca/996135/1/Guo_MSc_F2025.pdf)  
24. Exploring the Reliability of SHAP Values in Reinforcement Learning \- TH Köln, accessed February 18, 2026, [https://www.gm.th-koeln.de/ciopwebpub/Engelh24a.d/Evaluation\_of\_SHAP\_for\_RL\_XAI2024.pdf](https://www.gm.th-koeln.de/ciopwebpub/Engelh24a.d/Evaluation_of_SHAP_for_RL_XAI2024.pdf)  
25. Explainable machine learning to predict the cost of capital \- Frontiers, accessed February 18, 2026, [https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1578190/full](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1578190/full)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAIgAAAAZCAYAAADqgGa0AAAFGklEQVR4Xu2aXailUxjHH/mI8RWmmYSmg5KINDHNJA25IB9JLoTcuJgUczGKcmMiycQNMzei01z4inKBUpTtoyjlo4imkUM+ihBxQz6en/U+9trPWWu9H/u855zdvL/6N2ev9e61117rv571rLVHZGBgYGCgPYf7gh451BesEg5SHac62Fcc6FyrOtYX9shtvmCVgEHo25O+YpY4UvWM6nvVgurEidr2YI6PfKFyiGq76jHVXtVXqg9VT1Rl6A7VKfaGhhyletkXKhfJuN13VZ+rno3Kdqk2Sf+rG5PcJ8u7YJYUJm6r6m/VlzKdQTaoPlXd5CskDNQW1XWqN1Xvqa6vXqMHVb+o/pJ2EWGz6nlfKKEv1u4/qtur16aPq/LnpP/JO0F1p4QxmFkwxzQGOVv1s9RP7qmq71SX+goJn00fmLgmrFe9KqHNFJgf8+TaI5JRN3LlfcDC2S0zbJJpDXKX6kfVmb7CcbmESTnJVygXqv6U/IR6iEBPSTBCCgzE1pJqj+SRKEbda66uDx5Rfa2a8xUGYSZ2j3+90qQMQv/oJycS+zvX75HqLdXRrtxzv4RJSU0qJqOOSFQH739adZmviDDDEbE8G1W/S/i8uqi3FJCb8VlX+wq4QIKTcewZEtzEYLIXM+CrAW8QvozpGgn9J7H8Q7W/eiaGSXjcFzpIKEeSXtEbVPsk1LFf18GWxvgRJXKY4VIRgkTV6vrOQcAMyQKZgL0Wp++Q0KF5Ced2krEfVGeNH20Eq9eSvaa64r93lvEGARvgV6IyIgTmPjcqA5670ZV5WD1mOsyGCLuMxcPSbnsj9yBBzcEWtiDhs36S8ef9JsHknGBSkbALTe5h1qo+kbBAWCj/w56LSJY4KVhyRvg73R5aBZQMcndUZlGAFRHDc1e5Mg+Rk+dSx9K2vKg6whdGMM6Md5O8qCuckDAbE1+HjRuaMIhBIwuSTs5gq4RnSiGzT0oG4V+jq0HipDBuL0dpu2Llb/OFDst1RpKZkAi2fR8Rm4Lpm1yE1RqEZIkokkrOgC/EylrOK+qYvg3CBPwqYRyIniUwU8kAc5VKkFvQp0V7fgK2RraAtpjpS301bGseScYgfqA9mGOnL0ywRvWSjPfyJiLXqWMpDEKelYNJ4BmiZN1ksCWU8guOt3X5A0lzvKWXIArkFm4JTP+t1Bse7I4H43J7PQHbRt3AcEo42RcuI1y1f6M6rXpN4kXiyKTeaw9J2CI/UF0sk5NE/3MRkuc+k3FbdUkdE5YzAKeXOGlOgYn5rHekfsvGjPS9LfRvXsJ3JuG9UoIhyTdT2JE7GW1YbTSW+9KwUtuLRYQ44vjX6IZEeRxxOKmlFgG5hG8L5bCwnWNnpRR2CedVMgCRLU4ymaPSKZGJ5vcb6+cbqptVx1R1OfMThbnfweCLoMHcGwFjxCeFWYSwz3G1SUgvUVrRTArRIznIHfFJZlOD2PayV/WFlOcXCAD0vXTyysJvCTaw50jHRlaYdRJ+ANst5UhZh61oxoCxiOHW9AVZukgbJ5lcORw/WV3E+knEPF+CYdh+c6dUjtssok6Qe9yj2iPLc6vXF6wibijnfUULuEx7W3WrTBptTkI4z01AF9heWdkPSLjJbcPrqluqv5m/h2Rxnw1yFG7TpwL3phqfNTA4t5zTfJfUMZDV96hM124KtosuP3dwbI3/L0lu/jDe+77wQIdj3Hm+cEr4/xxdL7NWisMknAQv8RUDAwMDAwM98y92ozgBZL7DLAAAAABJRU5ErkJggg==>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAmwAAAA/CAYAAABdEJRVAAAL3ElEQVR4Xu3da4h1VRnA8ccuUphWFGkUvBpGVJpBlxerD1ZGRRiiRUF0I0iRLpBZ2oeabtCVJLOi1DIJqYwKE0yDjvShIIkCQ4kCk0oMTBKKbmbr39rPO2vW7HOZM2deZ5r/DxYzZ5191tnn7IH9zLNuEZIkSZIkSZIkSZIkSZIkSZIkSZIkSZIkSZIkSZIkSZIkSZIkSZIkSZIkSZIkSZIkSZIkSdJ+9tBSLijliP6JLXhkKW+L7bWx33ynlLP7yiU8u5Sb+sotemIpP+wrw+sqSdLKPK6UO0r5cymP6J5bxBeiBm0tbuBXlPKlprxleO7ypu7KoQ4/KeX85vFeclTU7/BPpTy+qef3Y5rHq0IgdFHUQOj0qO/9zw1HLO7XpRzsK4uvx8br9/JSHlbKJ5u6q2L9tSeWctbwe2svX1dJknaVZ5Xy19h6wEawt9ZXNsiu3F/KqU3djaV8sHmcHlLK1bH1c9gtCM5+N/zEY0r5ZdRM2Co9vZS7urr3Rv2et+rcUk7oKxufi9pue8yBqMH3g5u69NOo17G116+rJEm7xrIBGwHZSX1lgxv976Pe+MkGkYn6VGzOyKWXDWUv6gO2rFt1ho3v8taubpmA7dio7czqriTQ/kfU64zjSrkhpr9m2t/DXr6ukiTtGmMB22OjBlZkUsik9fI183DTJ5h4V9RutFkIBG4v5Qld/arQPt16IANGSdTncz2CmzYQa/HdPCrGAza074E8lp893j+DIY4ZOx++y77rcSxgm9fWV6IG0/O8tpT/DD/fE9ODNRxdyiQ2B/47fV0lSdoX2oCNcnIp/yrl57F+g+amTddW+mgsFrCBYILXL2IsIFkFuvKuK+VnpVw41JENuq2U5w2PGRfWBj5ks/geEp/34cPvDLK/p3nui1FfmwEbgdx5UQOYxPM5OP/4qBkuxvvxHRPYkc36UdTxYNTx/gRKie7FsaCwDdh43atjflt3xsYxhLPQdvtZZ/lbKc/pK2PnrqskSfvGWIaNGyw3+UQ25rfNY2723PTnIVigLcqs7EwisCMA2Qm0e32sB11khNoA7YzhMV23uC/q8emaqBnD/EzZVYixDBvf66R5/PfY+J3SxjnNY17fdh0S4FzWPOb6EHA+uqlDn2HLc5nVFteboHsesqx5/RbBcXyPvZ28rpIk7QvTArb2xksAQEnf6B6PYdkJBt6TySJD0wY403Ae027sb486I3JW+VZs7pJLtNu2zXFtlrDPNPId3BybZ0vmcfxMiwRsZNHoViQL9Yuo7bfn07+e77/NgjE7dBKbP9+0gG1WW2Tgpn3PiXNlkgiBHwHXIuPQ+r+bNOu6SpKkBSwTsHHznxew0aV6INZnCjKLcB6CmZ26sW8lYCPL1gdUadmAjaU3CCpzhmXffv/6PsjivCbDz9YyAdu8AOqIUn4QNUikcO24hv0s0F7/d5N28rpKkrQvLBOwcfNtx3e1GBPG+l4tbvq0OW2GaOIYslg7YSsBG/j91vWn/xfEvGr4yXm2GcMnRe0inhaw8V3ye7adbXA+GUjNC7LAe/BerWUCNp6/pnnc4tzIjLZOiPoeH+7qe/eWckpfGTt7XSVJ2hfo6iL4ossOmV163aEj6qKwf2gec/NtgwQQjD05aiaJhXNbzFLkeN5rbA2vdHcpT+0rV4Bz+3QpHxp+B7MWyfwQoKD/Hs6Pes455o1Fagk8QUDaBqUsNMuxL4z6+WiT9lg4FqyfRgB4cHj83KjHM47ss0Md3/FTht9p461Ru3jbGZ6c3wuax+Bz0VZ+r8+I6W0lJj8wHq53Uilfjs3ZMMbN8R5/jOljEcm+/SrqDOPeTl1XSdL/OW4u74+N45M+NjxHhqitP3Ko17rs5hy7OS+LG3qbtdotCODabFWLep7P5U9mZRAJdDg2l/TgNf0Egnnomryur1zCqVHHsa0SAe3YOLfdel0lSXsAAcdpUW9+ZA7oLnrx8ByZFwZZkylqs0za6CUxfoNeFjd1ut403blRs1XbRdbw2uHnqjADNjOQLa+rJGnbMlPEWKVjo2ZB3hR1fSzNd0tfsaSLYnwDcW1Gd2p2rW4Hf+uTGA+ytorMItnontdVkrQyZInIsrGw6NmxecC1piN4oGwX46kO9JWair/RVXxf2e2/HQR+LAEyFvh5XSVJK0XARnlm/4QkSZJ2B1byJ2CbNgNuu2bNjOzNOoeToy4tMa+cGaudECBJkvSAYr/FT0SdNdfut7gqLNswb8FRsJzG4Rrvw8SKNcvS5QO7vKytuPTtb7f4z4QkaUsYY8O4NRCsEbSx5MGqsMjpb/rKKVjbjAVNJUmSNDgu6pID2QXJLNH7Y+MG3YmxbZeX8t2m7jOlfDvqivEMuD4+6lIJPH79cMwrY+Om6S+KOjOPY3LJEN6PvTkXWWPrklgfbzersCzJ6cNrJEmS9hwWOCXr1We+GGf276gBT7u6PIuj3h41AGL1eAK8i2N9VhwL7T4o1rceYiX6XNWdulw1nrWoPh61vauirmNGwEj3JMiuGWTtHvw95K4H20VbWxnH2ONvbmyRXeq3064kSbtSZtGyZGBFPZmw9rncuoc1pgi0qGOMWR7bIuh63/D7OaUcPfxOUJhbCdHlSfDGpIAMCCexnuFj259+n0jNx84BBMGrDlwuLeWCvnIJ/P3Q1qzJJPPwWoL7fukM6tkWq6+XJGnfYf/G06Lus/j9qAOm2TMxrZXy/FgPzO4p5c1Rx8LRxfm0oZ7NwDMgOzHqvpaT4fGBqPtKvnR4rLo4LNuHESh/L2rA267Mf8dQMhBiseP7SnnnoSNqQE6AzNhE2slZtHzXZDRZP2wM4xq5RuA9eW9en5u3bwXBfbbVOi3quMn8B2HaThvMDE50m/fnTBaQ+u0EhJIk7XkEC+8o5cZYX6eNGzgbal8YNSDjZsl4tndHzXiwITc3+ptL+drwGo6hLW7CHynlmFJeEbVtFi8lGPz8cKwqglyCmX6LMBbrZfP1PhD6atTj++CF4Iz6Flkp6vr9P+kyv6WrA5u3bzVgo621vrJDgN+eG+d+U4zvtkGAd29fGbWe70qSJOmwy4DtjKZubaija7pHsMMkDjJObTfhWMAG6nKMYZrE+OSPZQK2SWwcDzmGnTbIqC260wbLxJzQV0ZdS3CsXpIkaUeNBWx0R49l0dJZUbuxs4saswK2fqIHgVmOR2z1ARvvn8EY3ZJjExR4zTwElgSY18diW07x+Zh93OOzjNVLkiTtqLGAjcdjwVfK11zZ1GXAxuQEAqLzSrmtlPObYxJB1lj3Yhuw0QZZOCam0C0Oxjj2mb1FAjaQGZsVhLb4DHS798fyGamXJEk6rFYdsOVm53+JOhasD3pwZ4zP1u0zbHSlkhXLiRDMDKbNUw4dUdtaBGPyOL+xbt4ek14msbl7lswj9ZIkSYfVWMB2+1A3Tc7obLsHx7pE6Vq8q6tjaRCOJYvVGwvY2vFvPNdm57KteegGZewawV8bAE7D+0yGny3qKJIkSYfVWMB29VDXByxprZRbY2O2aixgY+xaX4dVBWyYF7DRtUqwhkW3RiOTN4nNn//HQ70kSdJhlQEb2bBENyZrkbGIMePFboq67MclQz3H9wvJjgVs2XbKHSf6CQtpmYCNtqbh/Ps11TgfFlCehe7ay/rKqN2vY/WSJEk7hjFoBDBZ+swX+7KSSWNh2jdE3VWCCQDZpXhU1ICqbYMyGZ4HY9muiDr5gDXTwPIYLJ2ReM8M+Cj9eWWb7eM8T9pqkfVrj7u7eW7aThs9PhMTHHqs5zZWL0mS9IAi23ZmKW+M2m3K/qwg28XixPOwaC7ZOXY/SHS5MoB/FWiLSQKrwjIizE7tx7lRv8j4N0mSpAdcm6FiTNgyCOK+GasJfmjr2lhNW7i4lBu6OoJW6vsxbZIkSbsS21URrOX4tmUdKOU1feWS2OZqVW3RbcoSIK2DQ70kSdK+w8bxl/aVS6ItZncu68io+9P2qGcsnyRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJkiRJ0t73X+xq0vpVTpsNAAAAAElFTkSuQmCC>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC8AAAAUCAYAAAAdmmTCAAAA60lEQVR4XmNgGAUDD/5D8QEg5kGVGvxgIbrAUAK0cnwSEG8EYm50CRIAMwMBc6jpeGkgTgbiuQyUJUWYOa8YCJhDTccjA7yWEgl8GQiYM+p4LACvpUSCUceTA/BaSiQg2fGaQBxCApaHaMMAeC0lEowsx1ML4LWUSDAoHZ8GxAVAzIgugQbo6nhWIBYHYkkGiKXngVgdyocBkCMOAPEDBkhNig3AzMlkQDVHAFkRCFDT8cZA/JUBYiE6hgFQaBcD8VsGiHpsAJc5GG7FEKATmA3E+uiCpIKBcnw3A+E0TxAMhOPlgVgFXZBYAAA1x1cXVZOAAQAAAABJRU5ErkJggg==>

[image4]: <data:image/png;base64,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>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFQAAAAZCAYAAACvrQlzAAAD0ElEQVR4Xu2YS6hOURTHlxDKmzxCFxMpRWEirzBgQApRjEgMRFHkETdlYCBlKLoZKRSSUgxuSMoAAyUxIFEkUeTN+lln3W/dfc/5vvPd7oA6v/p3+/brrL32WvtxRSoqKir+SfqqRqtGqXpnZf0z/XP0UZ1sQhutWwdjQ53rjGpeaMM3toe6F6oHqtNZGdqlmuAdAoNV31SvMn1WjVFdUs0O7ZwBql5pYYDFmaMampSzOIek61yOx0bKQOlsN+rEZNVP1WWpNXiv+q26q9qkOqG6lZVdtW4dDFdtUD0Wq7+jWqkaGdowQSaxWnVTdU+1LvuNjqo+iNmxLesDM8ScT2Q641U/VI+k8zdw1GLVfbFJp2DDWtU7MYc8EbNlYlZP5M8Vs4t5/FIdFBszwneei7U5r1ofK1mVi7Egg8avxZwdIbq2JGWwRMwAjJ2a1EUYj3Fpn0Kku6HQqvouNskUnBntYDtgosvF+qcOHSIWHCzksFDudqfQjnGuiUV8hPnl2f8XKq+khVKLxLhHkbZnJX+wI2J92qXrZCLLxNqNSyvEHIcD3SHtqk+qmaGNc1vyy4sc6o67IDYPh/kTBNHJ0Co2Dpk6LZS3qB6G311gP2xNygaJDbYvKcdIQjwvAm+I9cGx9XDHx0k5e6Q2CSLuWfb7QGyUQeqmkQNFDvWxybCIZ8X0pJzFYjHpQ18gyq+rdnujsuAwVjMvEosgjfk4EViERx3tUlh59jTqMJjMIEP47Qs1Syyt61HkUBxZz6HpXFks0p0+bBU485xqr9RuGaVg426TrqnRCD6cbhEpK6TmIA4a9FLsIDomNrmUKWL2eD9XkW3ddSj9UiaJ2Uc/IhOnNg2nJhv+zrSiDhjvUVQPbgru+GYZIXb6PhUbg5M+j550KItGYNGPm0y38L0j72QtoswWwabvJ6fvSfUoinS3L+9Agp50aL0tqjRcRRgg3u8awT2M1CiKGmDT/yjF16AI207eBMEdkHdLgCKHNjqU8hbIr3jY3C1iiDOpspDKjfZPnM646WU8D+oXpIUZXGG4BTS7h/p1LbXTFzrPJt/zsblp+MgasdT9KuVOMtosFWvP3yJYHH9FHZbGJzVZgtN4WTl8a4dYuq8K5RHG3Sr2nbxs4WnLAciVC2jfpnrb0aJGi9iLi7HoVxo/hOiYp07PqkBRn/2xkXIq1EUVQeRxPdms+iK2EDwTOYzQIsnPnnR8V7vUohUH8sTlfwEccDy134iNGfG7chTz+C/pJ7UJkjULxRaVPa5M1pSBf8Aw5nxpnC0VFRUVFRUVFRU9zx+tMRPkXUA7wgAAAABJRU5ErkJggg==>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAJgAAAAaCAYAAABLupXyAAAFmElEQVR4Xu2ba6imUxTH/3K/JZpIQ3NGIndyaVyHchkJE2bQ0BRiQmJcQzlIInK/5DYkl5AvLrmF8kWjiJJSU0zTfOCDUj5QLutn7XXeffY5c+Y973mfc573zP7Xv+d99n6e593vc/577bXWXkeqmC7sYnzfuMa41nhHar8vnX9jPCO1TQZbGZ+XP+M344upjXP4nfHquLhi9uNo45/Z+ZDxw+y8V6wzvmXcIp2/bTzNuNnIFRWbDC4wXmK833hU0dcrQrh3G68v+io2Mexm/MN4atkxBWxr/MD4j/HSoq812MP4gvGZgjHgG+XrfbS/ZDwn9VV0j83l/tKBZccUgV+HwBaVHW0BjujZxsuN/8pNLqI6KPUfY/wo9d1jXGyck/oqugM+0bXydzs8umsEvNPzNkDe+e6dS0fARH/d+IvxtaKvG+xk3LJsbBIhMNb2AIP42Hhn1lYxOSAEBMZy9oN8uSzRi8CIRrc2Pi63jpMFotynbGwSDBSREZ3MV2dZnFaVzyIQ0T0k95UAwkJgCC3aesHBclclEL5YLlwMAwHFCvkYEDjH2+RR596pD5xsfNV4u9wdmpfa+46IShDZSlVxTQV3yd/jX/I/IPgstcH1qW0yuFh+H/f/ndq2kYuNc5bKJ+T+3nHGT+Viwc/mM1aQa3c0LpcLjs8XyiNRlvJbjWeqIcRM4AfgODILKgYXD6cjAn9KLron5UKCERgg0lXyv//Lxv1SeyMgXxMCG0Tw4ghMSj9mIpJN356bZxki2r/ZuEwuuHONJ8kt2yHypO9exjfkEe4rxkONe3Jjv3G+8Qv51gMiww+r6ODYASWC4ni8cWHWzjnHExL5fKLc0pXP6IYbxbdyZaNkopK6nzUa4T9Vjs8JgTmdlz4TZRDCfqnqh1X0ATiBXxdtOICtzhBvANsZ39XY2TURfzUewM0V/QPpB5KnhNFsFZXpCCILXj5pC0RG6At2ludSSNBFGoM+MsoBwuLV6uzqE/piERcaHzX+aNzf+Incub5KHrnQHlZ0qXGJ/BmE1z+n9raAHZAow/ldvmUTZThRijPTyEuFGCPIx9hLqVBXOF1jZ/F7G+mHYEheDYDVm5vaEN1X6TOCIFpheQ0QxfBDuR5RcQ4QE0KO5fms1A7YlopkJOJrm8BA5A0ZKxiSl+GQCG0LBmGM44IBIw5wuHxGAET0k9xqAQS3yvi5PBxGKJF4zEHkEvkXnpsLfpnx++y8LYi8YWz/kGsKC9wW5GPEn27jGMegFMCw/EcwK46Ul6VENji2R7BaWMawZCXYOSCrDEKIIL4LQbO90TaQN8RXpdarX3Ve/UaMsZ+1aI0iFwA+FMseKQ2K3ubIrQ3LHtbrJvlmObMHkSCWsHwBolYqMwMIMZ4fyyMWbjzLN9OICcSkaitijP2sRWsUOPVEawGEhB+WA+e/FBL3RLBQIs+kc19+L8/CqW4b+N1sIhPg4Oc0DSYvlRXlTkRwvIqLfIzDo7u6Au992qsuKhxRhsPkKKsZmkAvAsvHGL5ixYDgQXXSO/g5MAfBzQrjLcbnjNeltlNSP1s198pdB3w4REDh5yPGG7LregVpiHyM+GH5GHFbqPqIsh6ESGVFlPXg8xIU0Aco6blInbKeiobAskOtV6RRAJYhr/PimsvUyRViSTgS/BDwgKezNvKDi+TPwQd9TF6e0yv4/rwWDWDB8jHi1yLCKOvhOKROWc9y+diirIeSHnxqnh1ZgooZBP9GRtJ5B7kfwxL3jvEI4wJ5eoe+N+UBDMI6TG7lCG6mI2rGekbVBekgEFUXiI4qi6i64DNCi6qLRiouKroHtVUEK4gF0cyXi45lcFfjNcYr5YlPltFn0xErg/Voeh+YJRnnn/HFOInosbr8X8a+8jFdIZ8ID/ht/1dc8Hti6R2F/wC+Onkdw/O79wAAAABJRU5ErkJggg==>

[image7]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAF0AAAAZCAYAAABTuCK5AAADFUlEQVR4Xu2ZS6hOURTHlzzyCJFIyGOgvJUoUgwUkkfyGBgqA2RAKAauZGYgkVJcRiRlQoSBMhGliJRHISkMjAwoj/+vtfd3zj3X/XyH++Vc3/7Xr87Ze5/znbv22mutva9Z0t+qnzgtXohP4mxo2yzeiEdiexyc1L2aKC6JPuF+qFgqetVGJDVFX8QhsUusLfQlNUnXxXfz0NKwh48VZ8SpArwE7TaPX7H9nKUZzeuAudGXFTvqabhYLbaIH+bLBUPPCP0LxI3Qd1isESNCX5LZBfFBnLcstpfSW3PjHjdfKoPMPfuI6Jsbl2Q2W7SH6/nmztpW6y0hjI3RMT6ZOYaUZPCOmmnujEQJNMA8tj+tjSihOGMYfqclg/9K78zt801sFf3NJ4B72k+I3rXRDSjOGA+THKg7k5os4vgec6NDwyVQhcQ3k/zXlWCFef76J9oo7phvazE6cT0p0/g/4Ld6aD5wuvmZQjo/6KirlkWBRqkrNjxxZqg3qTvvWorreUXvLUOXWiIeFNrYYZXeaVVAA8UV6+xx9fgopvFws0UpeFB8NT8GKJaGlEJ8ECUkho9l0DCxX0y2rKSkj51ZFEcL9yxLxPvMV84icUw8E1PFLfMEtk1MCe3ROzaI9ebvGC1ei5Whr0dquXWebeJVvX5AE8Q889UxJrQxEffDNUY6ah6aovaKl+bjMTT3CAMzuTG0rQrtiCMHSljEhGD0hVl3awqjYDA0x/zwHmHYV+bejZiEdnFbTDI3HhNWFAbFuIj35p1gk3hiLX7mUzRKm/mmiu3xXPHZslAwynx7jHezgqLHF8UOeHC4jpOD4m8xyXHb3ZLCU4m/iErnpPnxb6x0MC6JGQ8n1AwJ7Rgvhpa88ODHuXvGXA7XI833Dhy6kYOqIv7Wm2KW2CHei8WWrf5uF4mTKiGKEEJcz4sEW/wAnunqXCK/I+S5/LO8Cy+vyg6Z7+C/RRfDPSsUx+jRib7qIjwSJmPip1ojHLLCk5okCofnliV+SmoS/bjaiKRuF+GFY13CyzXzM6uqhL7/XuSdhoz9E8ODqu3nLCRfAAAAAElFTkSuQmCC>

[image8]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAHkAAAAcCAYAAABFwxCgAAAEcUlEQVR4Xu2aXaiURRjH/1JCZRR9kIhBGhEUhkFgfkUiiZrURR9CmtcKdlVgkXghKCR4ISJFiUl6UUTZlSgVFQUhBVKQCn3gB1Lghd14Y2D5/Hhmzs7OeffsnnfP7tk9zQ/+nN2Z951933meeeaZmSMVJpM/TOdMl0ynTQ+bfjJdMP1qum3kysLQcqPpQ9NfpvtN00xvmLabbk6uKww5t5tOBO0yTW+unnhmmNaYXsj0eKh/pKLuvlBXqM8rpv/k/d9zHjSdNP0p/1F0WR5CYIfpWii/Ip9Tngh1hfosMl2Vh+u+wY9hyIumuVk5IYXEoK8PNIUhEn5k+l7Nfd0XvpEb+rg8CVgsH+WPJtcUuuNd04LwOc7N/O0bcZ4gjBBOSOnjAxW6h+XRkqzsX9OqrKynEDoI1xj6mOn55upCTWLe84/piBrJ1gY1ch1G9L2hvKcw5+6T/zAeVpiC4GEfyA2MoYd1cX6radY4dJf+R0nle0Gvyo3MvFyYQtxg+lie6cW5eWfTFQWWPXEvYRC1XGNAqNqiRiof5+YzppnxoiGiV+F62RCoEjLonzV6qxKD4x1xzZzyjDz7JvVn9B8y/dZ0hS8XPg11X8vbf1v+W9vkm/R75Bv21B2Wb5niXO+bToXv3P+J6UsNBm/J+4X3iNNbqrNq7BC2Sl5xrHV5YU1atoPX4r375Q9Dh/M9rb8n1F01Pa3GiL7DtFW+LOCl2Fwn1BPKIiwFflBjdLwp/40nTXvl62+O2DAcyd5m00OhPDrbWtOL8jbolPOhfLLB+b+Vb+/Oaa4agf5YYfpd1RGiysgvyduMR5APyPvogrxfSIirjiHzdkZYrdHx/GibegRz5BskT5lmhzIM/2P4zEsxQln3RV6XvwDXY1i+Awa9yfRc+PxsKAfygRhBcIBBMTLw/n+bPs8rMu5U9bYlg+DlvFAe1eIRJNCX2zU6kqZUtTNhYAQMBI/JPQ4w5Dn56AUe9KB8y5SHx1g4SM5SuTGBdlOnWy8P34PEJnk4rhqpKUTBHPqL+6ugTXYeAedv136rdromGgsIt4xOHuw1091yg8QHJKH7Qh7mcAyMF50jggcz70aIJLH9OIpxgirnmEyIPukU0yljGTluLTP9dXIM2aqdrqGzeTnAQO+YDqiRnWMktvEYwYTuOJdg4BiqU3CMX5LvXPNZ+Exu8J1pt/pwuF4DcpaYm3TKWEZeJG+TBLTdKIZW7XQNicUtyXcehnk5hZfORyz3cG8VqddyX3ovbTG/DSJEKkLseObGsYzMCOYIMj/yjdDPaR+2aqcwgdDpX2l8hzmtjEwkJPSvUvPcnIITpCuhqnYKE0yd/9HKjcx0xh5CegyJgfN19kL5v2qlFCP3GEZdnQP/1MjxCJJki2NI4AiS40fKTsiXXOziMSXko7sYuYdgXBLOdqzMCzR6JHcK6+V5WVmddgodQHgmTLf7r5nF8pVDTl0jcx//PZueJ9Rpp9AGEqy4N91O+ZwamW/amBd2SL6sqttOYdi5DvjcD7HgqmttAAAAAElFTkSuQmCC>