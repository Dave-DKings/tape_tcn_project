"""
Test script for PortfolioEnvTF with mock data
Tests the environment without requiring external data downloads
"""

import logging
import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.environment import PortfolioEnvTF
from src.config import get_active_config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_mock_data_processor():
    """Create a mock data processor with sample data."""
    
    class MockDataProcessor:
        def __init__(self, config):
            self.config = config
            self.asset_tickers = config['ASSET_TICKERS']
            self.num_assets = len(self.asset_tickers)
        
        def get_feature_columns(self, phase='phase1'):
            """Mock feature columns."""
            if phase == 'phase1':
                # Simple feature set for testing
                features = [
                    'LogReturn_1d',
                    'EMA_12', 'EMA_26',
                    'RSI_14', 'MACD_12_26_9',
                    'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0'
                ]
                return features
            return []
    
    config = get_active_config('phase1')
    return MockDataProcessor(config)

def create_mock_processed_data():
    """Create mock processed data for testing."""
    
    config = get_active_config('phase1')
    asset_tickers = config['ASSET_TICKERS']
    num_assets = len(asset_tickers)
    num_days = 100  # 100 trading days for testing
    
    # Create date range
    dates = pd.date_range('2023-01-01', periods=num_days, freq='D')
    
    # Create mock data
    data = []
    
    for date in dates:
        for ticker in asset_tickers:
            # Mock feature values (normalized)
            row = {
                'Date': date,
                'Ticker': ticker,
                'LogReturn_1d': np.random.normal(0.001, 0.02),  # ~0.1% daily return, 2% volatility
                'EMA_12': np.random.normal(0, 1),
                'EMA_26': np.random.normal(0, 1),
                'RSI_14': np.random.normal(0, 1),
                'MACD_12_26_9': np.random.normal(0, 1),
                'BBL_20_2.0': np.random.normal(0, 1),
                'BBM_20_2.0': np.random.normal(0, 1),
                'BBU_20_2.0': np.random.normal(0, 1),
            }
            data.append(row)
    
    df = pd.DataFrame(data)
    return df

def test_portfolio_environment_mock():
    """Test PortfolioEnvTF with mock data."""
    
    logger.info("=" * 80)
    logger.info("STARTING PORTFOLIO ENVIRONMENT TEST (MOCK DATA)")
    logger.info("=" * 80)
    
    try:
        # Create mock data and processor
        logger.info("Creating mock data...")
        config = get_active_config('phase1')
        processor = create_mock_data_processor()
        processed_df = create_mock_processed_data()
        
        logger.info(f"Mock data created: {processed_df.shape}")
        logger.info(f"Date range: {processed_df['Date'].min()} to {processed_df['Date'].max()}")
        logger.info(f"Assets: {config['ASSET_TICKERS']}")
        
        # Split data
        split_ratio = 0.8
        total_dates = len(processed_df['Date'].unique())
        split_idx = int(len(processed_df) * split_ratio)
        
        logger.info(f"Data split: {split_idx} training samples, {len(processed_df) - split_idx} test samples")
        
        # Create environment
        logger.info("\n" + "=" * 60)
        logger.info("CREATING PORTFOLIO ENVIRONMENT")
        logger.info("=" * 60)
        
        env = PortfolioEnvTF(
            config=config,
            data_processor=processor,
            processed_data=processed_df,
            mode='train',
            start_idx=0,
            end_idx=split_idx
        )
        
        logger.info("Environment created successfully!")
        
        # Test environment reset
        logger.info("\nTesting environment reset...")
        obs, info = env.reset(seed=42)
        
        logger.info(f"✅ Reset successful")
        logger.info(f"  Observation shape: {obs.shape}")
        logger.info(f"  Initial portfolio value: ${info['portfolio_value']:,.2f}")
        logger.info(f"  Initial weights: {[f'{w:.3f}' for w in info['weights']]}")
        logger.info(f"  Action space: {env.action_space}")
        logger.info(f"  Observation space: {env.observation_space}")
        
        # Test action space
        logger.info("\nTesting action space...")
        
        # Test different types of actions
        test_actions = [
            np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # All in first asset
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),  # All in cash
            np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.0]),  # Equal weights, no cash
            np.array([0.1, 0.2, 0.3, 0.2, 0.1, 0.1]),  # Diversified portfolio
        ]
        
        for i, action in enumerate(test_actions):
            logger.info(f"\n  Test Action {i+1}: {[f'{a:.2f}' for a in action]}")
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            logger.info(f"    ✅ Step successful")
            logger.info(f"    Normalized weights: {[f'{w:.3f}' for w in info['weights']]}")
            logger.info(f"    Weight sum: {np.sum(info['weights']):.6f}")
            logger.info(f"    Reward: {reward:.4f}")
            logger.info(f"    Portfolio value: ${info['portfolio_value']:,.2f}")
            logger.info(f"    Turnover: {info['turnover']:.4f}")
            logger.info(f"    Transaction costs: ${info['transaction_costs']:.2f}")
            
            if terminated:
                logger.info(f"    Episode terminated")
                break
        
        # Test episode completion
        logger.info("\n" + "=" * 60)
        logger.info("TESTING FULL EPISODE")
        logger.info("=" * 60)
        
        env.reset(seed=42)
        total_reward = 0.0
        step_count = 0
        max_test_steps = min(20, env.max_steps)
        
        for step in range(max_test_steps):
            # Random action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            step_count += 1
            
            if step % 5 == 0:  # Log every 5 steps
                logger.info(f"Step {step + 1}: Reward={reward:.4f}, Portfolio=${info['portfolio_value']:,.2f}")
            
            if terminated:
                logger.info(f"Episode terminated at step {step + 1}")
                break
        
        # Get final metrics
        logger.info("\nCalculating portfolio metrics...")
        metrics = env.get_portfolio_metrics()
        
        logger.info("\n📊 Portfolio Performance Metrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                if key in ['total_return', 'annualized_return']:
                    logger.info(f"  {key}: {value:.2%}")
                elif key in ['sharpe_ratio', 'max_drawdown']:
                    logger.info(f"  {key}: {value:.4f}")
                elif 'value' in key.lower() or 'cost' in key.lower():
                    logger.info(f"  {key}: ${value:,.2f}")
                else:
                    logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")
        
        # Test rendering
        logger.info("\nTesting environment rendering...")
        env.render()
        
        # Test edge cases
        logger.info("\n" + "=" * 60)
        logger.info("TESTING EDGE CASES")
        logger.info("=" * 60)
        
        env.reset()
        
        # Test zero action
        logger.info("\nTesting zero action...")
        zero_action = np.zeros(env.action_space.shape[0])
        obs, reward, terminated, truncated, info = env.step(zero_action)
        logger.info(f"✅ Zero action handled: weights={[f'{w:.3f}' for w in info['weights']]}")
        
        # Test large action
        logger.info("\nTesting large action values...")
        large_action = np.ones(env.action_space.shape[0]) * 100
        obs, reward, terminated, truncated, info = env.step(large_action)
        logger.info(f"✅ Large action normalized: weights={[f'{w:.3f}' for w in info['weights']]}")
        
        logger.info("\n" + "=" * 80)
        logger.info("🎉 PORTFOLIO ENVIRONMENT TEST COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Portfolio environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_portfolio_environment_mock()
    exit(0 if success else 1)