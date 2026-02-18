#!/usr/bin/env python3
"""
Test the corrected asset_memory approach that handles the reset-on-done issue
"""
import sys
sys.path.insert(0, '/workspaces/multi_modal_drl_continuous/adaptive_portfolio_rl')

import numpy as np
import pandas as pd
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from stable_baselines3.common.vec_env import DummyVecEnv

print("Testing CORRECTED asset_memory approach...")
print("="*80)

# Create minimal test data
test_data = []
for day in range(3):
    for stock in ['A', 'B']:
        test_data.append({
            'date': f'2020-01-0{day+1}',
            'tic': stock,
            'close': 100 + day * 5,
            'open': 99,
            'high': 101,
            'low': 98,
            'volume': 1000000,
            'macd': 0.1,
            'rsi': 50.0
        })

df = pd.DataFrame(test_data)
df['day'] = df['date'].map({f'2020-01-0{i+1}': i for i in range(3)})
df = df.set_index('day')

print(f"Test data: {df.index.nunique()} days, {df['tic'].nunique()} stocks")

# Create environment
stock_dim = 2
env = StockTradingEnv(
    df=df,
    stock_dim=stock_dim,
    hmax=100,
    initial_amount=100000,
    num_stock_shares=[0, 0],
    buy_cost_pct=[0.001, 0.001],
    sell_cost_pct=[0.001, 0.001],
    reward_scaling=1,
    state_space=1 + stock_dim * 2 + stock_dim * 2,
    action_space=stock_dim,
    tech_indicator_list=['macd', 'rsi']
)

env_wrapped = DummyVecEnv([lambda: env])

# Run evaluation with corrected tracking
obs = env_wrapped.reset()
done = False

# Track portfolio values (with fix for reset-on-done)
episode_portfolio_values = [env.asset_memory[-1]]  # Initial value

print(f"\nInitial: ${episode_portfolio_values[0]:,.2f}")

step_count = 0
while not done:
    action = np.array([[0.5, 0.5]])  # Buy equal amounts
    obs, reward, done, info = env_wrapped.step(action)
    done = done[0]
    step_count += 1
    
    if not done:
        episode_portfolio_values.append(env.asset_memory[-1])
        print(f"Step {step_count}: ${env.asset_memory[-1]:,.2f} (from asset_memory)")
    else:
        # Manually calculate final value since asset_memory gets reset
        final_value = env.state[0] + np.sum(
            np.array(env.state[(stock_dim + 1):(stock_dim * 2 + 1)]) *
            np.array(env.state[1:(stock_dim + 1)])
        )
        episode_portfolio_values.append(final_value)
        print(f"Step {step_count}: ${final_value:,.2f} (calculated from state - LAST STEP)")

print(f"\nFinal portfolio trajectory: {len(episode_portfolio_values)} values")
print(f"  Initial: ${episode_portfolio_values[0]:,.2f}")
print(f"  Final: ${episode_portfolio_values[-1]:,.2f}")
print(f"  Min: ${min(episode_portfolio_values):,.2f}")
print(f"  Max: ${max(episode_portfolio_values):,.2f}")

ret = (episode_portfolio_values[-1] / episode_portfolio_values[0] - 1) * 100
print(f"\n  Return: {ret:.2f}%")

print("\n" + "="*80)
if abs(ret) > 0.1:
    print("✅ SUCCESS! Portfolio tracking is working correctly!")
    print(f"   The agent made trades and the return is {ret:.2f}%")
else:
    print("⚠️  WARNING: Return is still near 0%")
