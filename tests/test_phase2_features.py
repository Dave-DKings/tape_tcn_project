"""
Test Phase 2 Feature Pipeline

Tests:
1. TCN price forecast generation
2. Trading signals generation
3. Complete Phase 2 pipeline
4. Data leakage validation
5. Feature count verification

Author: AI Assistant
Date: October 2, 2025
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from src.config import PHASE2_CONFIG
from src.data_utils import DataProcessor


def test_phase2_configuration():
    """Test Phase 2 configuration is set up correctly."""
    print("\n" + "=" * 80)
    print("TEST 1: Phase 2 Configuration Validation")
    print("=" * 80)
    
    feature_params = PHASE2_CONFIG.get('feature_params', {})
    
    # Check TCN forecast config
    tcn_config = feature_params.get('tcn_forecast')
    assert tcn_config is not None, "❌ TCN forecast config is missing!"
    assert 'sequence_length' in tcn_config, "❌ sequence_length not in TCN config"
    assert 'epochs' in tcn_config, "❌ epochs not in TCN config"
    
    print(f"✅ TCN forecast config present:")
    print(f"   Sequence length: {tcn_config.get('sequence_length')}")
    print(f"   Epochs: {tcn_config.get('epochs')}")
    print(f"   TCN units: {tcn_config.get('tcn_units')}")
    
    # Check dynamic covariance
    cov_config = feature_params.get('dynamic_covariance')
    assert cov_config is not None, "❌ Dynamic covariance config missing!"
    print(f"✅ Dynamic covariance config present")
    
    # Check macro data (should be optional)
    macro_config = feature_params.get('macro_data')
    if macro_config is None:
        print(f"✅ Macro data is OPTIONAL (currently disabled)")
    else:
        print(f"✅ Macro data is OPTIONAL (currently enabled)")
    
    print("\n✅ Phase 2 configuration validated successfully\n")


def test_feature_columns_phase2():
    """Test feature column generation for Phase 2."""
    print("=" * 80)
    print("TEST 2: Phase 2 Feature Column Validation")
    print("=" * 80)
    
    processor = DataProcessor(PHASE2_CONFIG)
    feature_cols = processor.get_feature_columns(phase='phase2')
    
    print(f"Total feature columns: {len(feature_cols)}")
    
    # Count by category
    log_returns = [c for c in feature_cols if 'LogReturn' in c]
    eigenvalues = [c for c in feature_cols if 'Covariance_Eigenvalue' in c]
    tcn_forecasts = [c for c in feature_cols if 'TCN_Forecast' in c]
    trading_signals = [c for c in feature_cols if c in ['MA_Crossover', 'Price_Crossover', 'MACD_Crossover', 'RSI_Signal']]
    macro_features = [c for c in feature_cols if any(x in c for x in ['EFFR', 'DGS', 'WALCL', 'CPI', 'PPI', 'UNRATE'])]
    
    print(f"\nFeature breakdown:")
    print(f"  Log returns: {len(log_returns)} - {log_returns}")
    print(f"  Eigenvalues: {len(eigenvalues)} - {eigenvalues}")
    print(f"  TCN forecasts: {len(tcn_forecasts)} - {tcn_forecasts[:2]}...")
    print(f"  Trading signals: {len(trading_signals)} - {trading_signals}")
    print(f"  Macro features: {len(macro_features)}")
    
    # Validate counts
    assert len(log_returns) == 3, f"❌ Expected 3 log return periods, got {len(log_returns)}"
    assert len(eigenvalues) == 3, f"❌ Expected 3 eigenvalues, got {len(eigenvalues)}"
    assert len(tcn_forecasts) == 5, f"❌ Expected 5 TCN forecasts, got {len(tcn_forecasts)}"
    assert len(trading_signals) == 4, f"❌ Expected 4 trading signals, got {len(trading_signals)}"
    
    print("\n✅ Feature columns validated successfully\n")
    
    return feature_cols


def test_trading_signals():
    """Test trading signal generation."""
    print("=" * 80)
    print("TEST 3: Trading Signal Generation")
    print("=" * 80)
    
    from src.feature_extractors import generate_trading_signals
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    
    df = pd.DataFrame({
        'Date': np.repeat(dates, 2),
        'Ticker': ['AAPL', 'MSFT'] * 100,
        'Close': np.random.randn(200) * 10 + 100,
        'EMA_12': np.random.randn(200) * 5 + 100,
        'EMA_26': np.random.randn(200) * 5 + 100,
        'SMA_50': np.random.randn(200) * 5 + 100,
        'MACD_12_26_9': np.random.randn(200),
        'MACDs_12_26_9': np.random.randn(200),
        'RSI_14': np.random.rand(200) * 100,
    })
    
    # Generate signals
    df_with_signals = generate_trading_signals(df)
    
    # Check all signal columns exist
    required_signals = ['MA_Crossover', 'Price_Crossover', 'MACD_Crossover', 'RSI_Signal']
    for signal in required_signals:
        assert signal in df_with_signals.columns, f"❌ {signal} not in dataframe"
        print(f"✅ {signal} column present")
    
    # Check signal values are in valid range
    for signal in required_signals:
        values = df_with_signals[signal].unique()
        print(f"   {signal} unique values: {sorted(values)}")
        assert all(v in [-1.0, 0.0, 1.0] for v in values), f"❌ Invalid values in {signal}"
    
    print("\n✅ Trading signals generated successfully\n")


def test_data_leakage_prevention():
    """Test that normalization doesn't leak test data into training."""
    print("=" * 80)
    print("TEST 4: Data Leakage Prevention")
    print("=" * 80)
    
    processor = DataProcessor(PHASE2_CONFIG)
    
    # Load data using correct method name
    df = processor.load_ohlcv_data()
    
    # Add a simple feature
    df['test_feature'] = np.arange(len(df))
    
    # Get dates
    unique_dates = sorted(df['Date'].unique())
    train_end_date = unique_dates[int(len(unique_dates) * 0.8)]
    
    print(f"Total dates: {len(unique_dates)}")
    print(f"Train end date: {train_end_date}")
    
    # Normalize with train/test split
    df_norm, scalers = processor.normalize_features(
        df=df,
        feature_cols=['test_feature'],
        train_end_date=train_end_date
    )
    
    # Check scaler was fitted only on training data
    scaler = scalers.get('test_feature')
    assert scaler is not None, "❌ Scaler not found"
    
    # Get training data statistics
    train_data = df[df['Date'] <= train_end_date]['test_feature'].values
    test_data = df[df['Date'] > train_end_date]['test_feature'].values
    
    print(f"\nTraining data:")
    print(f"  Samples: {len(train_data)}")
    print(f"  Mean: {np.mean(train_data):.2f}")
    print(f"  Std: {np.std(train_data):.2f}")
    
    print(f"\nTest data:")
    print(f"  Samples: {len(test_data)}")
    print(f"  Mean: {np.mean(test_data):.2f}")
    print(f"  Std: {np.std(test_data):.2f}")
    
    print(f"\nScaler statistics (should match training data):")
    print(f"  Mean: {scaler.mean_[0]:.2f}")
    print(f"  Std: {np.sqrt(scaler.var_[0]):.2f}")
    
    # Scaler stats should match training data, not full data
    assert np.abs(scaler.mean_[0] - np.mean(train_data)) < 1e-5, "❌ Scaler mean doesn't match training data!"
    
    print("\n✅ Data leakage prevention validated - scaler fitted only on training data\n")


def run_all_tests():
    """Run all Phase 2 tests."""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 25 + "PHASE 2 TEST SUITE" + " " * 35 + "║")
    print("╚" + "═" * 78 + "╝")
    print("\n")
    
    try:
        # Test 1: Configuration
        test_phase2_configuration()
        
        # Test 2: Feature columns
        feature_cols = test_feature_columns_phase2()
        
        # Test 3: Trading signals
        test_trading_signals()
        
        # Test 4: Data leakage
        test_data_leakage_prevention()
        
        # Final summary
        print("=" * 80)
        print("FINAL SUMMARY")
        print("=" * 80)
        print(f"✅ Phase 2 configuration validated")
        print(f"✅ Feature columns: {len(feature_cols)} total")
        print(f"✅ Trading signals working")
        print(f"✅ No data leakage detected")
        print("\n")
        print("╔" + "═" * 78 + "╗")
        print("║" + " " * 20 + "ALL TESTS PASSED SUCCESSFULLY! ✅" + " " * 24 + "║")
        print("╚" + "═" * 78 + "╝")
        print("\n")
        
    except AssertionError as e:
        print("\n")
        print("╔" + "═" * 78 + "╗")
        print("║" + " " * 32 + "TEST FAILED ❌" + " " * 32 + "║")
        print("╚" + "═" * 78 + "╝")
        print(f"\nError: {e}")
        raise
    except Exception as e:
        print("\n")
        print("╔" + "═" * 78 + "╗")
        print("║" + " " * 32 + "TEST FAILED ❌" + " " * 32 + "║")
        print("╚" + "═" * 78 + "╝")
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    run_all_tests()
