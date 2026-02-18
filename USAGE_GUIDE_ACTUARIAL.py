# =============================================================================
# 📋 COMPLETE USAGE GUIDE: Actuarial Drawdown Features
# =============================================================================
# This cell shows you EXACTLY how to use the actuarial triangle system.
# Copy the relevant sections into your tcn_architecture_analysis.ipynb

"""
OVERVIEW:
---------
The actuarial drawdown triangle adds 7 new features to your agent:
1. expected_recovery_days: When will drawdown likely end (chain ladder prediction)
2. remaining_recovery_days: Days left until predicted recovery
3. recovery_prob_30d: Probability of recovery within 30 days
4. recovery_prob_60d: Probability of recovery within 60 days  
5. severity_percentile: How bad is this drawdown vs. historical?
6. drawdown_severity: Current drawdown magnitude (0.15 = -15%)
7. in_drawdown: Binary flag (1=in drawdown, 0=not)

This gives your agent predictive information about drawdown recovery timing!
"""

# =============================================================================
# STEP 1: BUILD TRIANGLE (Add this AFTER Cell 19 - Training)
# =============================================================================

print("\n" + "="*80)
print("🔧 STEP 1: BUILDING ACTUARIAL DRAWDOWN TRIANGLE")
print("="*80)

import joblib
from src.actuarial_drawdown_reserve import DrawdownDevelopmentTriangle

# Collect portfolio trajectories by replaying trained agent
collected_trajectories = []

print("📊 Collecting portfolio trajectories...")
print("   (Replaying trained agent for 50 episodes)")

for replay_ep in range(50):
    obs, info = env_train_exp6.reset()
    episode_values = [env_train_exp6.portfolio_value]
    episode_dates = [env_train_exp6.dates[env_train_exp6.day]]
    done = False
    
    while not done:
        # Use stochastic policy to get variety
        action, _, _ = agent_exp6.get_action_and_value(obs, deterministic=False)
        obs, reward, terminated, truncated, info = env_train_exp6.step(action)
        
        episode_values.append(env_train_exp6.portfolio_value)
        episode_dates.append(env_train_exp6.dates[env_train_exp6.day])
        done = terminated or truncated
    
    # Store trajectory
    collected_trajectories.append((
        np.array(episode_values),
        pd.DatetimeIndex(episode_dates)
    ))
    
    if (replay_ep + 1) % 10 == 0:
        print(f"   ✓ Collected {replay_ep + 1}/50 episodes")

# Build and fit the triangle
print("\n🔨 Fitting development triangle...")
triangle_exp6 = DrawdownDevelopmentTriangle()
triangle_exp6.fit(collected_trajectories)

# Display statistics
print("\n📈 Drawdown Event Summary:")
summary_df = triangle_exp6.summary_statistics()
print(summary_df.to_string(index=False))

# Show development factors
print("\n📊 Development Factors (Chain Ladder):")
for bucket, factors in triangle_exp6.dev_factors.items():
    if factors:
        print(f"\n  {bucket.upper()} Drawdowns:")
        for factor_name, factor_value in factors.items():
            print(f"    {factor_name}: {factor_value:.3f}")

# Test prediction on a sample drawdown
print("\n🔮 Example Prediction:")
print("   Scenario: Portfolio down 15% for 10 days")
test_prediction = triangle_exp6.predict_recovery_time(
    current_drawdown_pct=0.15,
    days_elapsed=10,
    current_portfolio_value=85000,
    peak_portfolio_value=100000
)
print(f"   → Expected recovery: {test_prediction['expected_recovery_days']:.0f} days total")
print(f"   → Remaining time: {test_prediction['remaining_days']:.0f} days")
print(f"   → P(recover in 30d): {test_prediction['recovery_probability_30d']:.1%}")
print(f"   → P(recover in 60d): {test_prediction['recovery_probability_60d']:.1%}")
print(f"   → Severity bucket: {test_prediction['severity_bucket']}")

# Save for use in evaluation
save_path = 'results/drawdown_triangle_exp6.pkl'
joblib.dump(triangle_exp6, save_path)
print(f"\n✅ Triangle saved to: {save_path}")
print("   Ready for use in Cell 20 (evaluation)!")


# =============================================================================
# STEP 2: LOAD TRIANGLE (Add this AT START of Cell 20 - Evaluation)
# =============================================================================

print("\n" + "="*80)
print("🔧 STEP 2: LOADING ACTUARIAL DRAWDOWN TRIANGLE")
print("="*80)

import joblib
from pathlib import Path

# Load the pre-trained triangle
triangle_path = Path('results/drawdown_triangle_exp6.pkl')

if triangle_path.exists():
    triangle_exp6 = joblib.load(triangle_path)
    print("✅ Actuarial triangle loaded successfully!")
    print(f"   Events analyzed: {len(triangle_exp6.events)}")
    print(f"   Severity buckets: {list(triangle_exp6.severity_buckets.keys())}")
    use_actuarial = True
else:
    print("⚠️  Triangle not found!")
    print(f"   Expected at: {triangle_path}")
    print("   Run STEP 1 (Cell 19+) first to build the triangle")
    triangle_exp6 = None
    use_actuarial = False


# =============================================================================
# STEP 3: APPLY TO ENVIRONMENTS (Modify your existing env creation)
# =============================================================================

print("\n" + "="*80)
print("🔧 STEP 3: CREATING EVALUATION ENVIRONMENTS WITH ACTUARIAL FEATURES")
print("="*80)

# BEFORE (your existing code):
# env_eval_det = PortfolioEnvTAPE(
#     config=config,
#     data_processor=data_processor,
#     processed_data=test_df,
#     mode='test',
#     ... other params ...
# )

# AFTER (add triangle):
env_eval_det = PortfolioEnvTAPE(
    config=config,
    data_processor=data_processor,
    processed_data=test_df,
    mode='test',
    start_idx=0,
    end_idx=len(test_df['Date'].unique()),
    action_normalization='dirichlet',
    exclude_covariance=False,
    random_start=False,
    episode_length_limit=None,
    reward_system='tape',
    tape_profile=PROFILE_BALANCED_GROWTH,
    tape_terminal_scalar=10.0,
    tape_terminal_clip=10.0,
    dsr_window=60,
    dsr_scalar=5.0,
    target_turnover=0.5,
    turnover_penalty_scalar=3.0,
    gamma=0.99
)

# 🆕 ENABLE ACTUARIAL FEATURES
if use_actuarial and triangle_exp6 is not None:
    env_eval_det.set_drawdown_triangle(triangle_exp6)
    print("✅ Actuarial features ENABLED for deterministic evaluation")
else:
    print("⚠️  Actuarial features DISABLED (triangle not available)")

# Repeat for stochastic evaluation environment
env_eval_stoch = PortfolioEnvTAPE(
    config=config,
    data_processor=data_processor,
    processed_data=test_df,
    mode='test',
    start_idx=0,
    end_idx=len(test_df['Date'].unique()),
    action_normalization='dirichlet',
    exclude_covariance=False,
    random_start=True,  # Different from deterministic
    episode_length_limit=None,
    reward_system='tape',
    tape_profile=PROFILE_BALANCED_GROWTH,
    tape_terminal_scalar=10.0,
    tape_terminal_clip=10.0,
    dsr_window=60,
    dsr_scalar=5.0,
    target_turnover=0.5,
    turnover_penalty_scalar=3.0,
    gamma=0.99
)

# 🆕 ENABLE ACTUARIAL FEATURES
if use_actuarial and triangle_exp6 is not None:
    env_eval_stoch.set_drawdown_triangle(triangle_exp6)
    print("✅ Actuarial features ENABLED for stochastic evaluation")

# Now run evaluation as normal - the agent automatically gets the 7 extra features!
print("\n🚀 Environments ready with actuarial features")
print(f"   State dimension: {env_eval_det.observation_space.shape[0]} features")
print(f"   (125 standard + 7 actuarial = 132 total)")


# =============================================================================
# STEP 4: VERIFY FEATURES (Optional - for debugging)
# =============================================================================

print("\n" + "="*80)
print("🔍 STEP 4: VERIFY ACTUARIAL FEATURES")
print("="*80)

# Reset environment and check observation shape
obs, info = env_eval_det.reset()

print(f"✅ Observation shape: {obs.shape}")
print(f"   Expected: (132,) if actuarial enabled, (125,) if disabled")

if use_actuarial:
    # Extract actuarial features (last 7 elements)
    actuarial_obs = obs[-7:]
    feature_names = [
        'expected_recovery_days',
        'remaining_recovery_days', 
        'recovery_prob_30d',
        'recovery_prob_60d',
        'severity_percentile',
        'drawdown_severity',
        'in_drawdown'
    ]
    
    print("\n📊 Actuarial Features (current state):")
    for name, value in zip(feature_names, actuarial_obs):
        print(f"   {name:25s}: {value:8.4f}")
    
    if actuarial_obs[-1] > 0.5:  # in_drawdown flag
        print("\n   ⚠️  Currently IN DRAWDOWN")
    else:
        print("\n   ✅ Currently NOT in drawdown")

print("\n" + "="*80)
print("✨ INTEGRATION COMPLETE - Ready to evaluate!")
print("="*80)


# =============================================================================
# PERFORMANCE COMPARISON (Add after evaluation completes)
# =============================================================================
"""
After running evaluation, compare performance WITH vs WITHOUT actuarial features:

Results WITHOUT Actuarial Features (baseline):
- Sharpe Ratio: 1.018
- Max Drawdown: 17.57%
- Total Return: 30.62%

Results WITH Actuarial Features (expected improvements):
- Sharpe Ratio: 1.05-1.15 (+3-13%)
- Max Drawdown: 12-15% (-20-30%)
- Total Return: 32-38% (+5-25%)

Why? Agent can now:
1. Stay aggressive during short-term drawdowns (if recovery is predicted soon)
2. Reduce exposure during prolonged drawdowns (if recovery > 60 days)
3. Better distinguish temporary dips from structural declines
"""
