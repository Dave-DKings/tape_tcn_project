
import sys
import os
import numpy as np
import traceback

# Add src to path
sys.path.append(os.path.abspath("."))

from src.config import PHASE1_CONFIG, ASSET_TICKERS, NUM_ASSETS
from src.environment_tape_rl import PortfolioEnvTAPE

print("="*60)
print("CONFIGURATION SANITY CHECK")
print("="*60)

# 1. Check General Config
print(f"ASSET_TICKERS ({len(ASSET_TICKERS)}): {ASSET_TICKERS}")
print(f"NUM_ASSETS: {NUM_ASSETS}")

# 2. Check Phase 1 Params
p1_env = PHASE1_CONFIG["environment_params"]
p1_train = PHASE1_CONFIG["training_params"]

print(f"\nPHASE 1 SETTINGS:")
print(f"   - Initial Allocation Mode: {p1_env.get('initial_allocation_mode')}")
print(f"   - Initial Cash Position: {p1_env.get('initial_cash_position')}")
print(f"   - Max Single Position: {p1_train.get('max_single_position')}%")
print(f"   - Min Cash Buffer: {p1_train.get('min_cash_position')}")

# 3. Check wiring in Environment
print("\n" + "="*60)
print("ENVIRONMENT WIRING CHECK")
print("="*60)

# Create dummy data processor and data
class DummyProcessor:
    def get_feature_columns(self, phase): return ["feat1", "feat2"]

dummy_data = os.path.join("data", "daily_ohlcv_assets.csv") # Just a path string for check
import pandas as pd
# Create dummy processed data
dates = pd.date_range(start="2020-01-01", periods=10)
# Create a DataFrame with Date, Close, and features.
# To be safe with num_assets logic, ensure Close column exists.
# The environment uses date filtering logic, so we just need a Date column.
df = pd.DataFrame({"Date": dates, "Close": 100.0, "feat1": 0.1, "feat2": 0.2})

try:
    env = PortfolioEnvTAPE(
        config=PHASE1_CONFIG,
        data_processor=DummyProcessor(),
        processed_data=df,
        mode="train"
    )
    
    obs, info = env.reset()
    
    print(f"Environment Initialized Successfully")
    print(f"   - Initial Value: ${info['portfolio_value']:,.2f}")
    print(f"   - Initial Date: {info['date']}")
    
    # Check Initial Weights
    # Access weights from env internal state
    weights = env.current_weights
    
    print(f"\nINITIAL WEIGHTS CHECK:")
    print(f"   - Weights shape: {weights.shape}")
    print(f"   - Cash Weight: {weights[-1]:.4f} (Expected ~0.05)")
    print(f"   - Risky Asset Weights (first 5): {weights[:5]}")
    print(f"   - Sum: {np.sum(weights):.6f}")
    
    if np.isclose(weights[-1], 0.05, atol=0.01):
        print("\nSANITY CHECK PASSED! Initial cash weight is correct.")
    else:
        print(f"\nWARNING: Initial cash weight mismatch! Got {weights[-1]}, Expected 0.05")
    
    print("\nSANITY CHECK COMPLETED SUCCESSFULLY.")

except Exception as e:
    print(f"\nSANITY CHECK FAILED:")
    print(e)
    traceback.print_exc()
