"""
Test script to compare Equal Weight vs Volume-Based Weight initialization strategies
Uses real OHLCV data to calculate volume-weighted initial positions
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
from src.data_utils import DataProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_volume_vs_equal_weights():
    """Test and compare equal weights vs volume-based weights using real data."""
    
    logger.info("=" * 80)
    logger.info("COMPARING EQUAL vs VOLUME-BASED WEIGHT INITIALIZATION")
    logger.info("=" * 80)
    
    try:
        # Prepare real data
        logger.info("Loading and processing real OHLCV data...")
        config = get_active_config('phase1')
        
        # Initialize DataProcessor and prepare data
        processor = DataProcessor(config)
        processed_df, scalers = processor.prepare_features_phase1()
        
        logger.info(f"Data prepared: {processed_df.shape}")
        logger.info(f"Date range: {processed_df['Date'].min()} to {processed_df['Date'].max()}")
        
        # Use a smaller subset for testing (last 100 days)
        recent_dates = sorted(processed_df['Date'].unique())[-100:]
        test_data = processed_df[processed_df['Date'].isin(recent_dates)].copy()
        
        logger.info(f"Using recent data: {len(recent_dates)} days, {test_data.shape[0]} rows")
        
        strategies = ['equal', 'volume_weighted']
        results = {}
        
        for strategy in strategies:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"TESTING {strategy.upper().replace('_', ' ')} INITIALIZATION")
            logger.info(f"{'=' * 60}")
            
            # Get config and modify initialization method
            test_config = get_active_config('phase1')
            test_config['INITIAL_WEIGHTS_METHOD'] = strategy
            
            # Create environment
            env = PortfolioEnvTF(
                config=test_config,
                data_processor=processor,
                processed_data=test_data,
                mode='train',
                start_idx=0,
                end_idx=None
            )
            
            # Reset and get initial weights
            obs, info = env.reset(seed=42)
            initial_weights = info['weights'].copy()
            
            logger.info(f"\nInitial Weights ({strategy}):")
            total_asset_weight = 0.0
            for i, ticker in enumerate(test_config['ASSET_TICKERS']):
                logger.info(f"  {ticker}: {initial_weights[i]:.3f} ({initial_weights[i]*100:.1f}%)")
                total_asset_weight += initial_weights[i]
            logger.info(f"  CASH: {initial_weights[-1]:.3f} ({initial_weights[-1]*100:.1f}%)")
            logger.info(f"  Total Asset Weight: {total_asset_weight:.3f} ({total_asset_weight*100:.1f}%)")
            
            # Run a simulation to compare performance
            total_reward = 0.0
            portfolio_values = [env.portfolio_value]
            transaction_costs = []
            turnovers = []
            
            # Use a simple buy-and-hold strategy for fair comparison
            num_steps = min(30, env.max_steps)
            
            for step in range(num_steps):
                # Buy and hold - just maintain initial weights
                action = initial_weights.copy()  # Maintain initial allocation
                
                obs, reward, terminated, truncated, info = env.step(action)
                
                total_reward += reward
                portfolio_values.append(info['portfolio_value'])
                transaction_costs.append(info['transaction_costs'])
                turnovers.append(info['turnover'])
                
                if terminated:
                    break
            
            # Calculate performance metrics
            final_value = portfolio_values[-1]
            total_return = (final_value / env.initial_balance) - 1.0
            avg_reward = total_reward / len(portfolio_values) if portfolio_values else 0.0
            
            # Risk metrics
            if len(portfolio_values) > 1:
                returns = np.diff(portfolio_values) / portfolio_values[:-1]
                volatility = np.std(returns) * np.sqrt(252)  # Annualized
                sharpe = (total_return * 252 / len(returns)) / volatility if volatility > 0 else 0.0
                
                # Max drawdown
                cumulative = np.array(portfolio_values) / env.initial_balance
                running_max = np.maximum.accumulate(cumulative)
                drawdowns = (cumulative - running_max) / running_max
                max_drawdown = np.min(drawdowns)
            else:
                volatility = 0.0
                sharpe = 0.0
                max_drawdown = 0.0
            
            results[strategy] = {
                'initial_weights': initial_weights.copy(),
                'final_portfolio_value': final_value,
                'total_return': total_return,
                'total_reward': total_reward,
                'avg_reward': avg_reward,
                'volatility': volatility,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_drawdown,
                'avg_transaction_costs': np.mean(transaction_costs) if transaction_costs else 0.0,
                'avg_turnover': np.mean(turnovers) if turnovers else 0.0,
                'num_steps': len(portfolio_values) - 1
            }
            
            logger.info(f"\nPerformance Summary ({strategy}):")
            logger.info(f"  Initial Value: ${env.initial_balance:,.2f}")
            logger.info(f"  Final Value: ${final_value:,.2f}")
            logger.info(f"  Total Return: {total_return:.2%}")
            logger.info(f"  Volatility: {volatility:.2%}")
            logger.info(f"  Sharpe Ratio: {sharpe:.3f}")
            logger.info(f"  Max Drawdown: {max_drawdown:.2%}")
            logger.info(f"  Avg Transaction Costs: ${np.mean(transaction_costs):.2f}")
            logger.info(f"  Avg Turnover: {np.mean(turnovers):.4f}")
            logger.info(f"  Steps: {len(portfolio_values) - 1}")
        
        # Detailed comparison
        logger.info(f"\n{'=' * 80}")
        logger.info("DETAILED STRATEGY COMPARISON")
        logger.info(f"{'=' * 80}")
        
        equal_weights = results['equal']['initial_weights']
        volume_weights = results['volume_weighted']['initial_weights']
        
        logger.info("\n📊 Initial Weight Allocations:")
        logger.info(f"{'Asset':<8} {'Equal':<15} {'Volume-Based':<15} {'Difference':<15}")
        logger.info("-" * 65)
        
        for i, ticker in enumerate(test_config['ASSET_TICKERS']):
            diff = volume_weights[i] - equal_weights[i]
            logger.info(f"{ticker:<8} {equal_weights[i]:>6.3f} ({equal_weights[i]*100:>5.1f}%) "
                       f"{volume_weights[i]:>6.3f} ({volume_weights[i]*100:>5.1f}%) "
                       f"{diff:>+6.3f} ({diff*100:>+5.1f}%)")
        
        # Cash allocation
        cash_diff = volume_weights[-1] - equal_weights[-1]
        logger.info(f"{'CASH':<8} {equal_weights[-1]:>6.3f} ({equal_weights[-1]*100:>5.1f}%) "
                   f"{volume_weights[-1]:>6.3f} ({volume_weights[-1]*100:>5.1f}%) "
                   f"{cash_diff:>+6.3f} ({cash_diff*100:>+5.1f}%)")
        
        logger.info("\n📈 Performance Comparison:")
        logger.info(f"{'Metric':<20} {'Equal':<15} {'Volume-Based':<15} {'Difference':<15}")
        logger.info("-" * 70)
        
        equal_perf = results['equal']
        volume_perf = results['volume_weighted']
        
        metrics = [
            ('Total Return', 'total_return', '{:.2%}'),
            ('Volatility', 'volatility', '{:.2%}'),
            ('Sharpe Ratio', 'sharpe_ratio', '{:.3f}'),
            ('Max Drawdown', 'max_drawdown', '{:.2%}'),
            ('Avg TX Costs', 'avg_transaction_costs', '${:.2f}'),
            ('Avg Turnover', 'avg_turnover', '{:.4f}')
        ]
        
        for metric_name, metric_key, fmt in metrics:
            equal_val = equal_perf[metric_key]
            volume_val = volume_perf[metric_key]
            diff_val = volume_val - equal_val
            
            logger.info(f"{metric_name:<20} {fmt.format(equal_val):<15} "
                       f"{fmt.format(volume_val):<15} {fmt.format(diff_val):<15}")
        
        # Analysis and insights
        logger.info(f"\n{'=' * 80}")
        logger.info("ANALYSIS & INSIGHTS")
        logger.info(f"{'=' * 80}")
        
        # Weight concentration analysis
        equal_concentration = np.sum(equal_weights[:-1]**2)  # Exclude cash
        volume_concentration = np.sum(volume_weights[:-1]**2)
        
        logger.info(f"\n🔍 Portfolio Concentration (HHI):")
        logger.info(f"  Equal Weights: {equal_concentration:.3f}")
        logger.info(f"  Volume-Based: {volume_concentration:.3f}")
        
        if volume_concentration > equal_concentration:
            logger.info(f"  📊 Volume-based is more concentrated (+{volume_concentration-equal_concentration:.3f})")
        else:
            logger.info(f"  📊 Volume-based is more diversified ({volume_concentration-equal_concentration:.3f})")
        
        # Performance analysis
        return_diff = volume_perf['total_return'] - equal_perf['total_return']
        sharpe_diff = volume_perf['sharpe_ratio'] - equal_perf['sharpe_ratio']
        
        logger.info(f"\n💹 Performance Analysis:")
        if return_diff > 0:
            logger.info(f"  ✅ Volume-based outperformed by {return_diff:.2%}")
        else:
            logger.info(f"  ❌ Volume-based underperformed by {return_diff:.2%}")
        
        if sharpe_diff > 0:
            logger.info(f"  ✅ Volume-based has better risk-adjusted returns (+{sharpe_diff:.3f} Sharpe)")
        else:
            logger.info(f"  ❌ Volume-based has worse risk-adjusted returns ({sharpe_diff:.3f} Sharpe)")
        
        # Recommendations
        logger.info(f"\n🏆 RECOMMENDATION:")
        
        if sharpe_diff > 0.1:  # Significantly better Sharpe ratio
            logger.info("  📈 USE VOLUME-BASED WEIGHTS")
            logger.info("     ✅ Better risk-adjusted returns")
            logger.info("     ✅ Reflects market liquidity and activity")
            logger.info("     ✅ More realistic for institutional trading")
        elif abs(sharpe_diff) < 0.05:  # Similar performance
            logger.info("  ⚖️  EITHER STRATEGY IS REASONABLE")
            logger.info("     📊 Similar risk-adjusted performance")
            logger.info("     🎯 Choice depends on investment philosophy:")
            logger.info("        - Equal weights: Better diversification")
            logger.info("        - Volume weights: Market-driven allocation")
        else:
            logger.info("  📊 USE EQUAL WEIGHTS")
            logger.info("     ✅ Better diversification benefits")
            logger.info("     ✅ Simpler and more transparent")
            logger.info("     ✅ Better risk-adjusted returns in this case")
        
        # Volume insights
        logger.info(f"\n💡 Volume-Based Weight Insights:")
        max_weight_idx = np.argmax(volume_weights[:-1])
        max_weight_ticker = test_config['ASSET_TICKERS'][max_weight_idx]
        max_weight_value = volume_weights[max_weight_idx]
        
        logger.info(f"  📊 Highest allocation: {max_weight_ticker} ({max_weight_value:.1%})")
        logger.info(f"  💰 Volume-based weights reflect:")
        logger.info(f"     - Trading liquidity (higher volume = easier to trade)")
        logger.info(f"     - Market activity (active stocks get higher weights)")
        logger.info(f"     - Implicit market cap correlation (large stocks trade more)")
        logger.info(f"     - Real-world trading considerations")
        
        logger.info(f"\n{'=' * 80}")
        logger.info("COMPARISON TEST COMPLETED!")
        logger.info(f"{'=' * 80}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = test_volume_vs_equal_weights()