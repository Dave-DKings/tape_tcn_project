"""
Test script to compare Equal Weight vs Market Cap Weight initialization strategies
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
    num_days = 50  # 50 trading days for testing
    
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

def test_initialization_strategies():
    """Test and compare different initialization strategies."""
    
    logger.info("=" * 80)
    logger.info("COMPARING PORTFOLIO INITIALIZATION STRATEGIES")
    logger.info("=" * 80)
    
    # Create mock data
    processor = create_mock_data_processor()
    processed_df = create_mock_processed_data()
    
    strategies = ['equal', 'market_cap']
    results = {}
    
    for strategy in strategies:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"TESTING {strategy.upper()} WEIGHT INITIALIZATION")
        logger.info(f"{'=' * 60}")
        
        # Get config and modify initialization method
        config = get_active_config('phase1')
        config['INITIAL_WEIGHTS_METHOD'] = strategy
        
        # Create environment
        env = PortfolioEnvTF(
            config=config,
            data_processor=processor,
            processed_data=processed_df,
            mode='train',
            start_idx=0,
            end_idx=None
        )
        
        # Reset and get initial weights
        obs, info = env.reset(seed=42)
        initial_weights = info['weights'].copy()
        
        logger.info(f"Initial Weights ({strategy}):")
        for i, ticker in enumerate(config['ASSET_TICKERS']):
            logger.info(f"  {ticker}: {initial_weights[i]:.3f} ({initial_weights[i]*100:.1f}%)")
        logger.info(f"  CASH: {initial_weights[-1]:.3f} ({initial_weights[-1]*100:.1f}%)")
        
        # Run a short simulation
        total_reward = 0.0
        portfolio_values = [env.portfolio_value]
        
        for step in range(min(20, env.max_steps)):
            # Use a simple random strategy for comparison
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            portfolio_values.append(info['portfolio_value'])
            
            if terminated:
                break
        
        # Calculate performance metrics
        final_value = portfolio_values[-1]
        total_return = (final_value / env.initial_balance) - 1.0
        avg_reward = total_reward / len(portfolio_values) if portfolio_values else 0.0
        
        results[strategy] = {
            'initial_weights': initial_weights.copy(),
            'final_portfolio_value': final_value,
            'total_return': total_return,
            'total_reward': total_reward,
            'avg_reward': avg_reward,
            'num_steps': len(portfolio_values) - 1
        }
        
        logger.info(f"\nPerformance Summary ({strategy}):")
        logger.info(f"  Initial Value: ${env.initial_balance:,.2f}")
        logger.info(f"  Final Value: ${final_value:,.2f}")
        logger.info(f"  Total Return: {total_return:.2%}")
        logger.info(f"  Avg Reward: {avg_reward:.4f}")
        logger.info(f"  Steps: {len(portfolio_values) - 1}")
    
    # Compare strategies
    logger.info(f"\n{'=' * 80}")
    logger.info("STRATEGY COMPARISON")
    logger.info(f"{'=' * 80}")
    
    logger.info("\nInitial Weight Allocations:")
    logger.info(f"{'Asset':<8} {'Equal':<12} {'Market Cap':<12} {'Difference':<12}")
    logger.info("-" * 50)
    
    equal_weights = results['equal']['initial_weights']
    market_cap_weights = results['market_cap']['initial_weights']
    
    for i, ticker in enumerate(config['ASSET_TICKERS']):
        diff = market_cap_weights[i] - equal_weights[i]
        logger.info(f"{ticker:<8} {equal_weights[i]:.3f} ({equal_weights[i]*100:5.1f}%) "
                   f"{market_cap_weights[i]:.3f} ({market_cap_weights[i]*100:5.1f}%) "
                   f"{diff:+.3f} ({diff*100:+5.1f}%)")
    
    # Cash allocation
    cash_diff = market_cap_weights[-1] - equal_weights[-1]
    logger.info(f"{'CASH':<8} {equal_weights[-1]:.3f} ({equal_weights[-1]*100:5.1f}%) "
               f"{market_cap_weights[-1]:.3f} ({market_cap_weights[-1]*100:5.1f}%) "
               f"{cash_diff:+.3f} ({cash_diff*100:+5.1f}%)")
    
    logger.info("\nPerformance Comparison:")
    logger.info(f"{'Metric':<20} {'Equal':<15} {'Market Cap':<15} {'Difference':<15}")
    logger.info("-" * 70)
    
    equal_return = results['equal']['total_return']
    market_return = results['market_cap']['total_return']
    return_diff = market_return - equal_return
    
    equal_reward = results['equal']['avg_reward']
    market_reward = results['market_cap']['avg_reward']
    reward_diff = market_reward - equal_reward
    
    logger.info(f"{'Total Return':<20} {equal_return:<15.2%} {market_return:<15.2%} {return_diff:+15.2%}")
    logger.info(f"{'Avg Reward':<20} {equal_reward:<15.4f} {market_reward:<15.4f} {reward_diff:+15.4f}")
    
    # Analysis and recommendations
    logger.info(f"\n{'=' * 80}")
    logger.info("ANALYSIS & RECOMMENDATIONS")
    logger.info(f"{'=' * 80}")
    
    logger.info("\n🔍 Weight Distribution Analysis:")
    
    # Calculate concentration metrics
    equal_concentration = np.sum(equal_weights[:-1]**2)  # Exclude cash from concentration
    market_concentration = np.sum(market_cap_weights[:-1]**2)
    
    logger.info(f"  Equal Weight Concentration (HHI): {equal_concentration:.3f}")
    logger.info(f"  Market Cap Concentration (HHI): {market_concentration:.3f}")
    
    if market_concentration > equal_concentration:
        logger.info("  ⚠️  Market cap weighting is more concentrated (higher risk)")
    else:
        logger.info("  ✅ Market cap weighting provides better diversification")
    
    # Get actual market caps for context
    try:
        import yfinance as yf
        logger.info("\n📊 Current Market Caps (for context):")
        
        for ticker in config['ASSET_TICKERS']:
            try:
                stock = yf.Ticker(ticker)
                market_cap = stock.info.get('marketCap', 0)
                if market_cap > 0:
                    logger.info(f"  {ticker}: ${market_cap/1e9:.1f}B")
            except:
                logger.info(f"  {ticker}: Market cap data unavailable")
                
    except ImportError:
        logger.info("\n📊 yfinance not available for market cap display")
    
    logger.info("\n💡 Key Insights:")
    logger.info("  1. Equal Weight Strategy:")
    logger.info("     ✅ Better diversification across all assets")
    logger.info("     ✅ Equal exposure to each company's performance") 
    logger.info("     ❌ May overweight smaller, riskier companies")
    logger.info("     ❌ Ignores market consensus on company values")
    
    logger.info("  2. Market Cap Weight Strategy:")
    logger.info("     ✅ Reflects market consensus on company values")
    logger.info("     ✅ Mimics major market indices (S&P 500 style)")
    logger.info("     ✅ More realistic for institutional investors")
    logger.info("     ❌ May be concentrated in large-cap stocks")
    logger.info("     ❌ Less diversification benefit")
    
    logger.info("\n🏆 Recommendation:")
    if market_concentration - equal_concentration > 0.1:
        logger.info("  Use EQUAL WEIGHTS for better diversification in this portfolio")
        logger.info("  The market cap weights are too concentrated for this asset set")
    else:
        logger.info("  Use MARKET CAP WEIGHTS for more realistic market dynamics")
        logger.info("  The concentration risk is acceptable for this asset set")
    
    logger.info(f"\n{'=' * 80}")
    logger.info("COMPARISON TEST COMPLETED!")
    logger.info(f"{'=' * 80}")
    
    return results

if __name__ == "__main__":
    results = test_initialization_strategies()