#!/usr/bin/env python3
"""
Test script to verify the environment creation fix.
"""

from src.config import get_active_config
from src.data_utils import DataProcessor
from src.environment import PortfolioEnvTF

def test_environment_creation():
    """Test if the environment can be created successfully with the updated configuration."""
    
    print("🧪 Testing Environment Creation Fix")
    print("=" * 50)
    
    # Load configuration
    config = get_active_config('phase1')
    print(f"✅ Config loaded - Assets: {config['ASSET_TICKERS']}")
    
    # Initialize data processor
    data_processor = DataProcessor(config)
    print("✅ DataProcessor initialized")
    
    # Load and process data
    processed_df, scalers = data_processor.prepare_features_phase1()
    print(f"✅ Data processed - Shape: {processed_df.shape}, Dates: {len(processed_df['Date'].unique())}")
    
    # Split data
    unique_dates = processed_df['Date'].unique()
    total_days = len(unique_dates)
    train_end = int(total_days * 0.8)
    val_end = int(total_days * 0.9)
    
    print(f"✅ Data split - Training: {train_end}, Validation: {val_end-train_end}, Test: {total_days-val_end}")
    
    # Setup environment config
    env_config = config['environment_params'].copy()
    env_config.update({
        'ASSET_TICKERS': config['ASSET_TICKERS'],
        'NUM_ASSETS': config['NUM_ASSETS'],
        'BASE_DATA_PATH': config['BASE_DATA_PATH'],
        'PATH_DAILY_OHLCV': config['PATH_DAILY_OHLCV']
    })
    env_config['INITIAL_CAPITAL'] = env_config.get('initial_balance', 100000.0)
    env_config['TRANSACTION_COST_RATE'] = env_config.get('transaction_cost_pct', 0.001)
    
    # Test environment creation
    try:
        train_env = PortfolioEnvTF(
            config=env_config,
            data_processor=data_processor,
            processed_data=processed_df,
            mode='train',
            start_idx=0,
            end_idx=train_end
        )
        
        print(f"✅ Training environment created successfully!")
        print(f"   - Environment initialized")
        print(f"   - State dimension: {train_env.observation_space.shape[0]}")
        print(f"   - Action dimension: {train_env.action_space.shape[0]}")
        
        # Test environment reset
        state, info = train_env.reset()
        print(f"✅ Environment reset successful!")
        print(f"   - Initial state shape: {state.shape}")
        print(f"   - Portfolio value: ${info.get('portfolio_value', 0):,.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Environment creation failed: {e}")
        return False

if __name__ == "__main__":
    success = test_environment_creation()
    if success:
        print("\n🎉 ALL TESTS PASSED! Environment creation is working correctly.")
        print("✅ The notebook should now work without the 'Insufficient data' error.")
    else:
        print("\n❌ Tests failed. There may still be issues to resolve.")