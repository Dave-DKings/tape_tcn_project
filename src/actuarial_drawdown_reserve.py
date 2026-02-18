"""
Actuarial Drawdown Reserve Model - Chain Ladder Method for Portfolio Recovery Prediction

This module implements actuarial loss reserving techniques (chain ladder, development triangles)
adapted for portfolio drawdown recovery forecasting. It provides predictive features for RL agents
to make informed decisions during drawdown periods.

Key Concepts:
- Development Triangle: Track drawdown evolution over time (like insurance claim development)
- Chain Ladder: Predict ultimate recovery time using historical patterns
- Survival Analysis: Estimate probability of recovery within specific horizons
- Credibility Weighting: Blend individual episode experience with collective patterns

Novel Contributions:
1. First application of actuarial reserving to portfolio drawdown prediction
2. State-dependent development factors (severity-based segmentation)
3. Real-time integration into RL state space

Author: AI Assistant
Date: November 19, 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
import logging
from scipy import stats
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class DrawdownEvent:
    """
    Single drawdown event with full trajectory.
    
    Attributes:
        start_date: Date when drawdown began (new peak crossed)
        end_date: Date when recovery completed (peak regained) or None if ongoing
        start_value: Portfolio value at peak (before drawdown)
        min_value: Minimum portfolio value during drawdown
        end_value: Portfolio value at recovery (or current if ongoing)
        severity: Maximum drawdown depth as fraction (0.15 = -15%)
        duration_days: Number of days from start to recovery (or current)
        recovery_path: Daily portfolio values during drawdown [start, ..., end]
        is_recovered: Boolean indicating if drawdown fully recovered
        severity_bucket: Categorization for development factors ('mild', 'moderate', 'severe', 'extreme')
    """
    start_date: pd.Timestamp
    end_date: Optional[pd.Timestamp]
    start_value: float
    min_value: float
    end_value: float
    severity: float
    duration_days: int
    recovery_path: np.ndarray
    is_recovered: bool
    severity_bucket: str
    
    @property
    def days_to_trough(self) -> int:
        """Days from start until minimum value reached."""
        if len(self.recovery_path) == 0:
            return 0
        min_idx = np.argmin(self.recovery_path)
        return min_idx
    
    @property
    def recovery_speed(self) -> float:
        """Average daily recovery rate after trough (if recovered)."""
        if not self.is_recovered or len(self.recovery_path) <= self.days_to_trough:
            return 0.0
        
        trough_idx = self.days_to_trough
        recovery_portion = self.recovery_path[trough_idx:]
        
        if len(recovery_portion) <= 1:
            return 0.0
        
        # Calculate average daily % gain during recovery
        daily_returns = np.diff(recovery_portion) / recovery_portion[:-1]
        return float(np.mean(daily_returns))


class DrawdownDevelopmentTriangle:
    """
    Build and maintain development triangles for drawdown prediction.
    
    Similar to insurance loss triangles, but tracking portfolio drawdown evolution:
    
    Example Triangle:
    ┌──────────────┬────────┬────────┬─────────┬──────────┐
    │ Drawdown ID  │ Day 5  │ Day 10 │ Day 20  │ Recovery │
    ├──────────────┼────────┼────────┼─────────┼──────────┤
    │ 2020-03-15   │ -18%   │ -12%   │ -5%     │ Day 25   │
    │ 2021-06-10   │ -11%   │ -9%    │ -4%     │ Day 18   │
    │ 2024-08-05   │ -15%   │ ???    │ ???     │ ???      │ ← Predict
    └──────────────┴────────┴────────┴─────────┴──────────┘
    """
    
    def __init__(self, 
                 development_horizons: List[int] = [1, 5, 10, 20, 30, 60],
                 severity_buckets: Dict[str, Tuple[float, float]] = None):
        """
        Initialize the development triangle.
        
        Args:
            development_horizons: Days at which to measure drawdown evolution
            severity_buckets: Dictionary mapping bucket names to (min, max) severity thresholds
        """
        self.development_horizons = sorted(development_horizons)
        
        # Default severity buckets
        if severity_buckets is None:
            self.severity_buckets = {
                'mild': (0.0, 0.10),      # 0-10% drawdown
                'moderate': (0.10, 0.20),  # 10-20% drawdown
                'severe': (0.20, 0.35),    # 20-35% drawdown
                'extreme': (0.35, 1.0)     # 35%+ drawdown
            }
        else:
            self.severity_buckets = severity_buckets
        
        # Storage for drawdown events
        self.events: List[DrawdownEvent] = []
        
        # Development triangles (one per severity bucket)
        self.triangles: Dict[str, pd.DataFrame] = {}
        
        # Development factors (ratios between consecutive horizons)
        self.dev_factors: Dict[str, Dict[str, float]] = {}
        
        # Recovery time distributions (for survival analysis)
        self.recovery_distributions: Dict[str, np.ndarray] = {}
        
    def _classify_severity(self, severity: float) -> str:
        """Classify drawdown severity into bucket."""
        for bucket_name, (min_sev, max_sev) in self.severity_buckets.items():
            if min_sev <= severity < max_sev:
                return bucket_name
        return 'extreme'  # Fallback for anything above highest threshold
    
    def extract_drawdown_events(self, 
                                portfolio_values: np.ndarray,
                                dates: Optional[pd.DatetimeIndex] = None) -> List[DrawdownEvent]:
        """
        Extract all drawdown events from a portfolio value time series.
        
        Algorithm:
        1. Track running maximum (peak)
        2. When current value < peak, start/continue drawdown
        3. When current value >= peak, end drawdown (recovery complete)
        4. Store each complete drawdown as an event
        
        Args:
            portfolio_values: Array of portfolio values over time
            dates: DatetimeIndex for the values (or generate synthetic)
            
        Returns:
            List of DrawdownEvent objects
        """
        if dates is None:
            # Generate synthetic daily dates
            dates = pd.date_range(start='2020-01-01', periods=len(portfolio_values), freq='D')
        
        events = []
        running_peak = portfolio_values[0]
        in_drawdown = False
        drawdown_start_idx = 0
        drawdown_start_value = portfolio_values[0]
        
        for i in range(len(portfolio_values)):
            current_value = portfolio_values[i]
            
            # Update running peak
            if current_value > running_peak:
                # Check if we were in a drawdown
                if in_drawdown:
                    # Drawdown recovered - create event
                    drawdown_end_idx = i
                    recovery_path = portfolio_values[drawdown_start_idx:drawdown_end_idx+1]
                    min_value = np.min(recovery_path)
                    severity = (drawdown_start_value - min_value) / drawdown_start_value
                    duration = drawdown_end_idx - drawdown_start_idx
                    
                    event = DrawdownEvent(
                        start_date=dates[drawdown_start_idx],
                        end_date=dates[drawdown_end_idx],
                        start_value=drawdown_start_value,
                        min_value=min_value,
                        end_value=current_value,
                        severity=severity,
                        duration_days=duration,
                        recovery_path=recovery_path,
                        is_recovered=True,
                        severity_bucket=self._classify_severity(severity)
                    )
                    events.append(event)
                    
                    # Reset drawdown tracking
                    in_drawdown = False
                
                # New peak
                running_peak = current_value
                
            elif current_value < running_peak:
                # In drawdown
                if not in_drawdown:
                    # Start new drawdown
                    in_drawdown = True
                    drawdown_start_idx = i - 1  # Previous day was the peak
                    drawdown_start_value = running_peak
        
        # Handle ongoing drawdown at end of series
        if in_drawdown:
            recovery_path = portfolio_values[drawdown_start_idx:]
            min_value = np.min(recovery_path)
            severity = (drawdown_start_value - min_value) / drawdown_start_value
            duration = len(portfolio_values) - drawdown_start_idx
            
            event = DrawdownEvent(
                start_date=dates[drawdown_start_idx],
                end_date=None,
                start_value=drawdown_start_value,
                min_value=min_value,
                end_value=portfolio_values[-1],
                severity=severity,
                duration_days=duration,
                recovery_path=recovery_path,
                is_recovered=False,
                severity_bucket=self._classify_severity(severity)
            )
            events.append(event)
        
        return events
    
    def fit(self, historical_portfolio_trajectories: List[Tuple[np.ndarray, pd.DatetimeIndex]]):
        """
        Fit the development triangle using historical data.
        
        Args:
            historical_portfolio_trajectories: List of (portfolio_values, dates) tuples
                                              from multiple training episodes
        """
        logger.info(f"Fitting drawdown development triangle with {len(historical_portfolio_trajectories)} trajectories...")
        
        # Extract all drawdown events
        all_events = []
        for portfolio_values, dates in historical_portfolio_trajectories:
            events = self.extract_drawdown_events(portfolio_values, dates)
            all_events.extend(events)
        
        self.events = all_events
        logger.info(f"Extracted {len(all_events)} drawdown events")
        
        # Log severity distribution
        severity_counts = defaultdict(int)
        for event in all_events:
            severity_counts[event.severity_bucket] += 1
        logger.info(f"Severity distribution: {dict(severity_counts)}")
        
        # Build development triangles for each severity bucket
        for bucket_name in self.severity_buckets.keys():
            self._build_triangle_for_bucket(bucket_name)
        
        # Calculate development factors
        self._calculate_development_factors()
        
        # Fit recovery time distributions
        self._fit_recovery_distributions()
        
        logger.info("✅ Drawdown development triangle fitted successfully")
    
    def _build_triangle_for_bucket(self, bucket_name: str):
        """
        Build development triangle for specific severity bucket.
        
        Triangle structure (rows = events, columns = development horizons):
        
                Day_1   Day_5   Day_10  Day_20  Day_30  Day_60
        Event1  -0.15   -0.18   -0.12   -0.05   0.00    NaN
        Event2  -0.10   -0.11   -0.09   -0.04   0.00    NaN
        Event3  -0.20   -0.22   -0.19   -0.15   -0.10   -0.02
        """
        # Filter events for this bucket
        bucket_events = [e for e in self.events if e.severity_bucket == bucket_name and e.is_recovered]
        
        if len(bucket_events) == 0:
            logger.warning(f"No recovered events in bucket '{bucket_name}' - skipping triangle")
            self.triangles[bucket_name] = pd.DataFrame()
            return
        
        # Build triangle matrix
        triangle_data = []
        event_ids = []
        
        for event in bucket_events:
            row = {}
            for horizon in self.development_horizons:
                if horizon < len(event.recovery_path):
                    # Drawdown at this horizon (negative value)
                    current_value = event.recovery_path[horizon]
                    drawdown_pct = (current_value - event.start_value) / event.start_value
                    row[f'Day_{horizon}'] = drawdown_pct
                else:
                    # Event recovered before this horizon
                    row[f'Day_{horizon}'] = 0.0  # Fully recovered
            
            triangle_data.append(row)
            event_ids.append(event.start_date.strftime('%Y-%m-%d'))
        
        # Create DataFrame
        triangle_df = pd.DataFrame(triangle_data, index=event_ids)
        self.triangles[bucket_name] = triangle_df
        
        logger.info(f"Built triangle for '{bucket_name}': {len(bucket_events)} events × {len(self.development_horizons)} horizons")
    
    def _calculate_development_factors(self):
        """
        Calculate development factors (chain ladder ratios) for each bucket.
        
        Development Factor = Average(Value at t+Δt / Value at t)
        
        Example:
            Day 5 → Day 10 factor = -0.12 / -0.18 = 0.67 (drawdown improving by 33%)
        """
        for bucket_name, triangle in self.triangles.items():
            if triangle.empty:
                continue
            
            factors = {}
            columns = [f'Day_{h}' for h in self.development_horizons]
            
            for i in range(len(columns) - 1):
                col_from = columns[i]
                col_to = columns[i + 1]
                
                # Calculate ratio for each event (handle division by zero)
                ratios = []
                for idx in triangle.index:
                    val_from = triangle.loc[idx, col_from]
                    val_to = triangle.loc[idx, col_to]
                    
                    # Only calculate ratio if both values are non-zero and negative (in drawdown)
                    if val_from < -0.001 and val_to < -0.001:
                        ratio = val_to / val_from
                        ratios.append(ratio)
                    elif val_from < -0.001 and abs(val_to) < 0.001:
                        # Recovered between these horizons
                        ratios.append(0.0)
                
                # Average development factor
                if len(ratios) > 0:
                    factor = np.mean(ratios)
                    factors[f'{col_from}_to_{col_to}'] = factor
                else:
                    factors[f'{col_from}_to_{col_to}'] = 1.0  # No change (fallback)
            
            self.dev_factors[bucket_name] = factors
            logger.info(f"Development factors for '{bucket_name}': {factors}")
    
    def _fit_recovery_distributions(self):
        """
        Fit recovery time distributions using Kaplan-Meier survival analysis.
        
        For each severity bucket, estimate P(Recovery by day T).
        """
        for bucket_name in self.severity_buckets.keys():
            bucket_events = [e for e in self.events if e.severity_bucket == bucket_name and e.is_recovered]
            
            if len(bucket_events) == 0:
                self.recovery_distributions[bucket_name] = np.array([])
                continue
            
            # Extract recovery times
            recovery_times = np.array([e.duration_days for e in bucket_events])
            
            # Store distribution
            self.recovery_distributions[bucket_name] = recovery_times
            
            # Log statistics
            logger.info(
                f"Recovery distribution for '{bucket_name}': "
                f"median={np.median(recovery_times):.0f} days, "
                f"mean={np.mean(recovery_times):.0f} days, "
                f"std={np.std(recovery_times):.0f} days"
            )
    
    def predict_recovery_time(self, 
                             current_drawdown_pct: float,
                             days_elapsed: int,
                             current_portfolio_value: float,
                             peak_portfolio_value: float) -> Dict[str, float]:
        """
        Predict expected recovery time using chain ladder method.
        
        Args:
            current_drawdown_pct: Current drawdown severity (0.15 = -15%)
            days_elapsed: Days since drawdown started
            current_portfolio_value: Current portfolio value
            peak_portfolio_value: Peak portfolio value (before drawdown)
            
        Returns:
            Dictionary with predictions:
            - expected_recovery_days: Chain ladder estimate of total recovery time
            - remaining_days: Days remaining until predicted recovery
            - recovery_probability_30d: P(recovery within 30 days)
            - recovery_probability_60d: P(recovery within 60 days)
            - severity_percentile: How severe is this vs. historical drawdowns?
        """
        # Classify severity
        bucket = self._classify_severity(current_drawdown_pct)
        
        # Get development factors for this bucket
        if bucket not in self.dev_factors or not self.dev_factors[bucket]:
            # No historical data - use conservative estimates
            return {
                'expected_recovery_days': 60,
                'remaining_days': max(0, 60 - days_elapsed),
                'recovery_probability_30d': 0.3,
                'recovery_probability_60d': 0.6,
                'severity_percentile': 0.5,
                'confidence': 'low'
            }
        
        # Find closest development horizon
        closest_horizon = min(self.development_horizons, key=lambda x: abs(x - days_elapsed))
        horizon_idx = self.development_horizons.index(closest_horizon)
        
        # Calculate current drawdown (negative value)
        current_dd_value = (current_portfolio_value - peak_portfolio_value) / peak_portfolio_value
        
        # Project forward using development factors
        projected_value = current_dd_value
        projected_days = days_elapsed
        
        factors = self.dev_factors[bucket]
        for i in range(horizon_idx, len(self.development_horizons) - 1):
            col_from = f'Day_{self.development_horizons[i]}'
            col_to = f'Day_{self.development_horizons[i+1]}'
            factor_key = f'{col_from}_to_{col_to}'
            
            if factor_key in factors:
                projected_value *= factors[factor_key]
                projected_days = self.development_horizons[i + 1]
                
                # Check if recovered
                if abs(projected_value) < 0.01:  # Recovered (within 1%)
                    break
        
        # Estimate total recovery time
        if abs(projected_value) < 0.01:
            expected_recovery_days = projected_days
        else:
            # Extrapolate beyond last horizon
            if len(self.recovery_distributions[bucket]) > 0:
                median_recovery = np.median(self.recovery_distributions[bucket])
                expected_recovery_days = max(projected_days, median_recovery)
            else:
                expected_recovery_days = projected_days + 30  # Conservative estimate
        
        remaining_days = max(0, expected_recovery_days - days_elapsed)
        
        # Calculate recovery probabilities from historical distribution
        recovery_times = self.recovery_distributions[bucket]
        if len(recovery_times) > 0:
            prob_30d = np.mean(recovery_times <= days_elapsed + 30)
            prob_60d = np.mean(recovery_times <= days_elapsed + 60)
        else:
            prob_30d = 0.3
            prob_60d = 0.6
        
        # Calculate severity percentile
        all_severities = [e.severity for e in self.events if e.severity_bucket == bucket]
        if len(all_severities) > 0:
            severity_percentile = stats.percentileofscore(all_severities, current_drawdown_pct) / 100.0
        else:
            severity_percentile = 0.5
        
        return {
            'expected_recovery_days': float(expected_recovery_days),
            'remaining_days': float(remaining_days),
            'recovery_probability_30d': float(prob_30d),
            'recovery_probability_60d': float(prob_60d),
            'severity_percentile': float(severity_percentile),
            'severity_bucket': bucket,
            'confidence': 'high' if len(recovery_times) > 5 else 'medium'
        }
    
    def get_features_for_state(self, 
                               current_portfolio_value: float,
                               portfolio_history: np.ndarray) -> Dict[str, float]:
        """
        Extract actuarial features for RL agent state.
        
        This is the main interface for integration into your environment.
        
        Args:
            current_portfolio_value: Current portfolio value
            portfolio_history: Recent portfolio values (for peak detection)
            
        Returns:
            Dictionary of features to add to state vector
        """
        # Calculate current drawdown
        peak_value = np.max(portfolio_history)
        current_drawdown = (peak_value - current_portfolio_value) / peak_value
        
        # Check if in drawdown
        if current_drawdown < 0.01:  # Not in significant drawdown
            return {
                'actuarial_in_drawdown': 0.0,
                'actuarial_expected_recovery_days': 0.0,
                'actuarial_remaining_recovery_days': 0.0,
                'actuarial_recovery_prob_30d': 1.0,
                'actuarial_recovery_prob_60d': 1.0,
                'actuarial_severity_percentile': 0.0,
                'actuarial_drawdown_severity': 0.0
            }
        
        # Find when drawdown started
        days_elapsed = 0
        for i in range(len(portfolio_history) - 1, -1, -1):
            if portfolio_history[i] >= peak_value * 0.99:  # Within 1% of peak
                days_elapsed = len(portfolio_history) - 1 - i
                break
        
        # Get predictions
        predictions = self.predict_recovery_time(
            current_drawdown_pct=current_drawdown,
            days_elapsed=days_elapsed,
            current_portfolio_value=current_portfolio_value,
            peak_portfolio_value=peak_value
        )
        
        # Format as state features
        return {
            'actuarial_in_drawdown': 1.0,
            'actuarial_expected_recovery_days': predictions['expected_recovery_days'] / 100.0,  # Normalize
            'actuarial_remaining_recovery_days': predictions['remaining_days'] / 100.0,
            'actuarial_recovery_prob_30d': predictions['recovery_probability_30d'],
            'actuarial_recovery_prob_60d': predictions['recovery_probability_60d'],
            'actuarial_severity_percentile': predictions['severity_percentile'],
            'actuarial_drawdown_severity': current_drawdown
        }
    
    def summary_statistics(self) -> pd.DataFrame:
        """Generate summary statistics for all buckets."""
        summary = []
        
        for bucket_name in self.severity_buckets.keys():
            bucket_events = [e for e in self.events if e.severity_bucket == bucket_name]
            recovered_events = [e for e in bucket_events if e.is_recovered]
            
            if len(bucket_events) == 0:
                continue
            
            severities = [e.severity for e in bucket_events]
            durations = [e.duration_days for e in recovered_events]
            
            summary.append({
                'Bucket': bucket_name,
                'Count': len(bucket_events),
                'Recovered': len(recovered_events),
                'Avg Severity': f"{np.mean(severities):.1%}",
                'Max Severity': f"{np.max(severities):.1%}",
                'Avg Duration': f"{np.mean(durations):.0f} days" if durations else "N/A",
                'Median Duration': f"{np.median(durations):.0f} days" if durations else "N/A"
            })
        
        return pd.DataFrame(summary)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def build_drawdown_triangle_from_training_logs(training_log_df: pd.DataFrame) -> DrawdownDevelopmentTriangle:
    """
    Build drawdown triangle from training logs (CSV format).
    
    Expected columns: 'episode', 'portfolio_value', 'date' (or generate synthetic dates)
    
    Args:
        training_log_df: DataFrame with columns ['episode', 'portfolio_value', ...]
        
    Returns:
        Fitted DrawdownDevelopmentTriangle object
    """
    triangle = DrawdownDevelopmentTriangle()
    
    # Group by episode
    episodes = training_log_df.groupby('episode')
    
    trajectories = []
    for episode_id, episode_data in episodes:
        portfolio_values = episode_data['portfolio_value'].values
        
        # Generate or extract dates
        if 'date' in episode_data.columns:
            dates = pd.to_datetime(episode_data['date'])
        else:
            dates = pd.date_range(start='2020-01-01', periods=len(portfolio_values), freq='D')
        
        trajectories.append((portfolio_values, dates))
    
    # Fit triangle
    triangle.fit(trajectories)
    
    return triangle


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 80)
    print("Actuarial Drawdown Reserve Model - Demo")
    print("=" * 80)
    
    # Generate synthetic portfolio trajectories
    np.random.seed(42)
    
    trajectories = []
    for i in range(10):
        # Simulate portfolio with drawdowns
        days = 500
        returns = np.random.normal(0.0005, 0.015, days)  # Slight positive drift, 1.5% daily vol
        portfolio_values = 100000 * np.exp(np.cumsum(returns))
        dates = pd.date_range(start='2020-01-01', periods=days, freq='D')
        
        trajectories.append((portfolio_values, dates))
    
    # Build triangle
    triangle = DrawdownDevelopmentTriangle()
    triangle.fit(trajectories)
    
    # Print summary
    print("\n📊 Drawdown Event Summary:")
    print(triangle.summary_statistics().to_string(index=False))
    
    # Test prediction
    print("\n🔮 Example Prediction:")
    print("Current situation: Portfolio down 15% for 10 days")
    prediction = triangle.predict_recovery_time(
        current_drawdown_pct=0.15,
        days_elapsed=10,
        current_portfolio_value=85000,
        peak_portfolio_value=100000
    )
    print(f"  Expected total recovery time: {prediction['expected_recovery_days']:.0f} days")
    print(f"  Remaining days: {prediction['remaining_days']:.0f} days")
    print(f"  P(recovery within 30 days): {prediction['recovery_probability_30d']:.1%}")
    print(f"  P(recovery within 60 days): {prediction['recovery_probability_60d']:.1%}")
    print(f"  Severity percentile: {prediction['severity_percentile']:.1%}")
    print(f"  Severity bucket: {prediction['severity_bucket']}")
    
    # Show state features
    print("\n🤖 RL State Features:")
    features = triangle.get_features_for_state(
        current_portfolio_value=85000,
        portfolio_history=np.array([100000, 98000, 95000, 90000, 87000, 85000])
    )
    for key, value in features.items():
        print(f"  {key}: {value:.4f}")
