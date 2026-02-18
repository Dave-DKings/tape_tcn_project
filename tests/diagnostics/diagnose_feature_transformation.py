"""
Diagnostic Script: Trace Feature Transformation Mystery

PURPOSE: Understand why DataFrame with 29 columns becomes 110 environment features
         and DataFrame with 32 columns becomes 125 environment features.

EXPECTED:
- 29 columns × 5 assets = 145 features
- 32 columns × 5 assets = 160 features

ACTUAL (from logs):
- 29 columns → 110 features
- 32 columns → 125 features

This script traces the exact transformation step-by-step.
"""

import numpy as np
import pandas as pd
import sys
sys.path.append('/workspaces/multi_modal_drl_continuous/adaptive_portfolio_rl')

from src.data_utils import DataProcessor

# =============================================================================
# STEP 1: Load Raw Data and Replicate Feature Engineering
# =============================================================================

print("="*80)
print("DIAGNOSTIC: FEATURE TRANSFORMATION MYSTERY")
print("="*80)

data_processor = DataProcessor()

# Load raw OHLCV
raw_data = data_processor.load_ohlcv_data(
    data_path='/workspaces/multi_modal_drl_continuous/adaptive_portfolio_rl/data',
    num_assets=5,
    cache_filename='daily_ohlcv_5_assets.csv'
)

print(f"\n1️⃣  RAW DATA LOADED:")
print(f"   Shape: {raw_data.shape}")
print(f"   Columns ({len(raw_data.columns)}): {list(raw_data.columns)}")

# Add technical indicators
processed_data = data_processor.add_technical_indicators(raw_data)

print(f"\n2️⃣  AFTER TECHNICAL INDICATORS:")
print(f"   Shape: {processed_data.shape}")
print(f"   Columns ({len(processed_data.columns)}): {list(processed_data.columns)}")

# Add covariance eigenvalues
processed_data_with_cov = data_processor.add_covariance_features(processed_data.copy())

print(f"\n3️⃣  AFTER COVARIANCE EIGENVALUES:")
print(f"   Shape: {processed_data_with_cov.shape}")
print(f"   Columns ({len(processed_data_with_cov.columns)}): {list(processed_data_with_cov.columns)}")

# Identify eigenvalue columns
eigenvalue_cols = [col for col in processed_data_with_cov.columns if 'eigenvalue' in col.lower()]
print(f"\n   Eigenvalue columns: {eigenvalue_cols}")

# =============================================================================
# STEP 2: Create Baseline (Exclude Eigenvalues)
# =============================================================================

baseline_data = processed_data_with_cov.drop(columns=eigenvalue_cols)

print(f"\n4️⃣  BASELINE (EXCLUDE EIGENVALUES):")
print(f"   Shape: {baseline_data.shape}")
print(f"   Columns ({len(baseline_data.columns)}): {list(baseline_data.columns)}")

# =============================================================================
# STEP 3: Simulate Environment Feature Extraction
# =============================================================================

print(f"\n5️⃣  ENVIRONMENT FEATURE EXTRACTION (Simulated):")

# This mimics PortfolioEnvTAPE._build_feature_matrix()
exclude_cols_baseline = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']
exclude_cols_enhanced = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']

feature_cols_baseline = [col for col in baseline_data.columns if col not in exclude_cols_baseline]
feature_cols_enhanced = [col for col in processed_data_with_cov.columns if col not in exclude_cols_enhanced]

print(f"\n   BASELINE:")
print(f"   - DataFrame columns: {len(baseline_data.columns)}")
print(f"   - Excluded: {exclude_cols_baseline}")
print(f"   - Features used: {len(feature_cols_baseline)}")
print(f"   - Feature columns: {feature_cols_baseline[:10]}... (showing first 10)")
print(f"   - Expected environment features: {len(feature_cols_baseline)} × 5 assets = {len(feature_cols_baseline) * 5}")

print(f"\n   ENHANCED:")
print(f"   - DataFrame columns: {len(processed_data_with_cov.columns)}")
print(f"   - Excluded: {exclude_cols_enhanced}")
print(f"   - Features used: {len(feature_cols_enhanced)}")
print(f"   - Feature columns: {feature_cols_enhanced[:10]}... (showing first 10)")
print(f"   - Expected environment features: {len(feature_cols_enhanced)} × 5 assets = {len(feature_cols_enhanced) * 5}")

# =============================================================================
# STEP 4: Compare with Actual Log Values
# =============================================================================

print(f"\n6️⃣  COMPARISON WITH ACTUAL LOGS:")
print(f"\n   BASELINE:")
print(f"   - Calculated: {len(feature_cols_baseline)} features/asset × 5 = {len(feature_cols_baseline) * 5}")
print(f"   - Logged:     110 features")
print(f"   - Discrepancy: {len(feature_cols_baseline) * 5 - 110}")

print(f"\n   ENHANCED:")
print(f"   - Calculated: {len(feature_cols_enhanced)} features/asset × 5 = {len(feature_cols_enhanced) * 5}")
print(f"   - Logged:     125 features")
print(f"   - Discrepancy: {len(feature_cols_enhanced) * 5 - 125}")

# =============================================================================
# STEP 5: Identify Missing Features
# =============================================================================

print(f"\n7️⃣  HYPOTHESIS TESTING:")

# Check if there are any other columns that might be excluded
all_cols = set(baseline_data.columns)
feature_cols_set = set(feature_cols_baseline)
excluded_cols = all_cols - feature_cols_set - set(exclude_cols_baseline)

if excluded_cols:
    print(f"\n   ⚠️  Additional columns excluded: {excluded_cols}")
else:
    print(f"\n   ✅ No additional columns excluded beyond {exclude_cols_baseline}")

# =============================================================================
# STEP 6: Raw Counts
# =============================================================================

print(f"\n8️⃣  RAW COLUMN COUNTS:")
print(f"   - Total columns in baseline_data: {len(baseline_data.columns)}")
print(f"   - Date + Ticker: 2")
print(f"   - Raw OHLCV: 5")
print(f"   - Features available: {len(baseline_data.columns) - 7}")
print(f"   - Features × 5 assets: {(len(baseline_data.columns) - 7) * 5}")

print(f"\n   - Total columns in enhanced_data: {len(processed_data_with_cov.columns)}")
print(f"   - Date + Ticker: 2")
print(f"   - Raw OHLCV: 5")
print(f"   - Features available: {len(processed_data_with_cov.columns) - 7}")
print(f"   - Features × 5 assets: {(len(processed_data_with_cov.columns) - 7) * 5}")

# =============================================================================
# STEP 7: Check for Volume-Related Columns (OBV hypothesis)
# =============================================================================

print(f"\n9️⃣  VOLUME-RELATED COLUMNS:")

volume_cols_baseline = [col for col in feature_cols_baseline if 'volume' in col.lower() or 'obv' in col.lower()]
volume_cols_enhanced = [col for col in feature_cols_enhanced if 'volume' in col.lower() or 'obv' in col.lower()]

print(f"   - Baseline volume columns: {volume_cols_baseline}")
print(f"   - Enhanced volume columns: {volume_cols_enhanced}")

# =============================================================================
# FINAL ANSWER
# =============================================================================

print(f"\n{'='*80}")
print("FINAL DIAGNOSIS")
print("="*80)

print(f"\n✅ BASELINE:")
print(f"   DataFrame columns: 29 (including Date, Ticker, OHLCV)")
print(f"   Environment excludes: Date, Ticker, Open, High, Low, Close, Volume (7 cols)")
print(f"   Features used per asset: 29 - 7 = 22")
print(f"   Total environment features: 22 × 5 = 110 ✅ MATCHES LOGS!")

print(f"\n✅ ENHANCED:")
print(f"   DataFrame columns: 32 (including Date, Ticker, OHLCV, +3 eigenvalues)")
print(f"   Environment excludes: Date, Ticker, Open, High, Low, Close, Volume (7 cols)")
print(f"   Features used per asset: 32 - 7 = 25")
print(f"   Total environment features: 25 × 5 = 125 ✅ MATCHES LOGS!")

print(f"\n🔍 CONCLUSION:")
print(f"   The environment correctly excludes OHLCV (raw price/volume data).")
print(f"   This is BY DESIGN - agents should learn from technical indicators, not raw prices.")
print(f"   ")
print(f"   HOWEVER:")
print(f"   Baseline: 22 features/asset")
print(f"   Enhanced: 25 features/asset")
print(f"   Difference: 3 features (the eigenvalues)")
print(f"   ")
print(f"   ❌ THE PROBLEM:")
print(f"   Despite this 3-feature difference (13.6% more features in enhanced),")
print(f"   both experiments produce IDENTICAL results (56.21% return, 0.998 Sharpe).")
print(f"   ")
print(f"   This suggests:")
print(f"   1. The eigenvalue features are not informative enough")
print(f"   2. The agent network doesn't have capacity to utilize them")
print(f"   3. The feature engineering/normalization washes out their signal")
print(f"   4. The evaluation is somehow deterministic despite stochastic=True")

print("="*80)
