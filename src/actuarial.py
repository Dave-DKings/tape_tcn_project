"""
Actuarial Drawdown Reserve Estimator
====================================

This module implements the core actuarial logic for the TAPE framework.
It adapts Chain Ladder and Survival Analysis methods to predict portfolio
drawdown recovery times and severities.

Key Classes:
    - DrawdownReserveEstimator: Main class for fitting models and generating predictions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from collections import defaultdict

# Configure logging
logger = logging.getLogger(__name__)

try:
    from lifelines import KaplanMeierFitter
    LIFELINES_AVAILABLE = True
    LIFELINES_IMPORT_ERROR = None
except ImportError as exc:
    KaplanMeierFitter = None
    LIFELINES_AVAILABLE = False
    LIFELINES_IMPORT_ERROR = str(exc)

class DrawdownReserveEstimator:
    """
    Estimates drawdown recovery metrics using Actuarial Chain Ladder and Survival Analysis.
    
    Attributes:
        lookback_window (int): Number of days to look back for fitting models.
        buckets (List[float]): Severity buckets for classification (e.g., [0.05, 0.10, 0.20]).
        horizons (List[int]): Development horizons for Chain Ladder (e.g., [30, 60, 90]).
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the estimator with configuration parameters.
        
        Args:
            config: Configuration dictionary containing 'actuarial_params'.
        """
        self.params = config.get('actuarial_params', {})
        self.buckets = self.params.get('severity_buckets', [0.05, 0.10, 0.15, 0.20])
        self.horizons = self.params.get('development_horizons', [10, 20, 30, 60, 90, 120])
        self.min_events = self.params.get('min_events_for_credibility', 5)
        self.log_optional_dependency_status = bool(self.params.get('log_optional_dependency_status', False))

        # Optional dependency handling: keep fallback behavior, avoid warning noise by default.
        if (not LIFELINES_AVAILABLE) and self.log_optional_dependency_status:
            logger.info(
                "Optional dependency 'lifelines' unavailable (%s). "
                "Using simple-average survival fallback.",
                LIFELINES_IMPORT_ERROR or "unknown import error",
            )
        
        # Model storage
        self.ldfs = {}  # Link Development Factors per bucket
        self.survival_curves = {}  # KM curves per bucket
        self.severity_cdf = None  # Empirical CDF for severity grading
        self.fitted = False
        
    def fit(self, price_history: pd.Series) -> None:
        """
        Fit the actuarial models using historical price data.
        
        Args:
            price_history: Series of asset prices (indexed by Date).
        """
        # 1. Extract Drawdown Events
        events = self._extract_drawdown_events(price_history)
        
        if not events:
            logger.warning("No drawdown events found in history. Models cannot be fitted.")
            self.fitted = False
            return
            
        # 2. Build Severity CDF (Percentile Grading)
        max_drawdowns = [e['max_depth'] for e in events]
        self.severity_cdf = self._fit_severity_cdf(max_drawdowns)
        
        # 3. Build Chain Ladder LDFs
        self.ldfs = self._build_chain_ladder_ldfs(events)
        
        # 4. Fit Survival Curves
        self.survival_curves = self._fit_survival_models(events)
        
        self.fitted = True
        logger.info(f"Actuarial models fitted on {len(events)} events.")

    def predict(self, current_drawdown: float, days_elapsed: int) -> Dict[str, float]:
        """
        Generate actuarial predictions for a current drawdown state.
        
        Args:
            current_drawdown: Current drawdown depth (positive float, e.g., 0.15 for -15%).
            days_elapsed: Number of days since the drawdown started.
            
        Returns:
            Dict containing:
                - expected_recovery_days
                - prob_recovery_30d
                - prob_recovery_60d
                - severity_percentile
        """
        if not self.fitted:
            return self._get_default_predictions()
            
        # 1. Determine Severity Bucket
        bucket = self._get_bucket(current_drawdown)
        
        # 2. Get Severity Percentile
        severity_percentile = self._get_severity_percentile(current_drawdown)
        
        # 3. Predict Recovery Time & Probabilities (Survival Analysis)
        recovery_metrics = self._predict_recovery_survival(bucket, days_elapsed)
        
        return {
            "Actuarial_Expected_Recovery": recovery_metrics['expected_days'],
            "Actuarial_Prob_30d": recovery_metrics['prob_30d'],
            "Actuarial_Prob_60d": recovery_metrics['prob_60d'],
            "Actuarial_Reserve_Severity": severity_percentile
        }

    def _extract_drawdown_events(self, prices: pd.Series) -> List[Dict]:
        """Identify discrete drawdown episodes from price history."""
        events = []
        peak = prices.iloc[0]
        peak_date = prices.index[0]
        in_drawdown = False
        start_date = None
        
        for date, price in prices.items():
            if price >= peak:
                if in_drawdown:
                    # Drawdown ended
                    duration = (date - start_date).days
                    max_depth = (peak - min_price) / peak
                    events.append({
                        'start_date': start_date,
                        'end_date': date,
                        'duration': duration,
                        'max_depth': max_depth,
                        'peak': peak
                    })
                    in_drawdown = False
                
                peak = price
                peak_date = date
            else:
                if not in_drawdown:
                    in_drawdown = True
                    start_date = date
                    min_price = price
                else:
                    min_price = min(min_price, price)
                    
        return events

    def _fit_severity_cdf(self, max_drawdowns: List[float]) -> Any:
        """Fit empirical CDF for severity ranking."""
        sorted_dds = np.sort(max_drawdowns)
        return sorted_dds

    def _get_severity_percentile(self, depth: float) -> float:
        """Calculate percentile rank of current depth."""
        if self.severity_cdf is None or len(self.severity_cdf) == 0:
            return 0.5
        # Find insertion point
        idx = np.searchsorted(self.severity_cdf, depth)
        return idx / len(self.severity_cdf)

    def _get_bucket(self, depth: float) -> float:
        """Classify depth into a bucket."""
        for b in self.buckets:
            if depth <= b:
                return b
        return self.buckets[-1] # Catch-all for severe

    def _build_chain_ladder_ldfs(self, events: List[Dict]) -> Dict:
        """
        Construct Chain Ladder Link Development Factors.
        (Simplified implementation for prototype: using average worsening factors)
        """
        # Placeholder for full triangle logic - for now, we focus on Survival Analysis
        # as it drives the primary features requested.
        return {}

    def _fit_survival_models(self, events: List[Dict]) -> Dict:
        """Fit Kaplan-Meier curves for each severity bucket."""
        models = {}
        
        # Group events by bucket
        events_by_bucket = defaultdict(list)
        for e in events:
            b = self._get_bucket(e['max_depth'])
            events_by_bucket[b].append(e['duration'])
            
        for bucket, durations in events_by_bucket.items():
            if len(durations) < self.min_events:
                continue
                
            if LIFELINES_AVAILABLE:
                kmf = KaplanMeierFitter()
                # All historical events are "observed" (uncensored) for this simple fit
                kmf.fit(durations, event_observed=[1]*len(durations))
                models[bucket] = kmf
            else:
                # Fallback: Store raw durations to compute simple stats
                models[bucket] = np.array(durations)
                
        return models

    def _predict_recovery_survival(self, bucket: float, days_elapsed: int) -> Dict:
        """Predict recovery using fitted survival models."""
        model = self.survival_curves.get(bucket)
        
        if model is None:
            return {"expected_days": 20.0, "prob_30d": 0.5, "prob_60d": 0.5} # Default fallback
            
        if LIFELINES_AVAILABLE:
            # Conditional survival: P(T > t + horizon | T > t) = S(t + horizon) / S(t)
            curr_survival = model.predict(days_elapsed)
            if curr_survival < 1e-6:
                curr_survival = 1e-6
                
            prob_30d = 1.0 - (model.predict(days_elapsed + 30) / curr_survival)
            prob_60d = 1.0 - (model.predict(days_elapsed + 60) / curr_survival)
            
            # Expected remaining time (median)
            median_survival = model.median_survival_time_
            if np.isinf(median_survival):
                median_survival = 100.0 # Cap
            expected_days = max(0, median_survival - days_elapsed)
            
        else:
            # Simple fallback logic
            durations = model
            remaining = durations[durations > days_elapsed]
            if len(remaining) == 0:
                expected_days = 30.0
                prob_30d = 0.1
                prob_60d = 0.2
            else:
                expected_days = np.median(remaining) - days_elapsed
                prob_30d = np.mean(remaining <= days_elapsed + 30)
                prob_60d = np.mean(remaining <= days_elapsed + 60)
                
        return {
            "expected_days": float(expected_days),
            "prob_30d": float(prob_30d),
            "prob_60d": float(prob_60d)
        }

    def _get_default_predictions(self) -> Dict[str, float]:
        return {
            "Actuarial_Expected_Recovery": 0.0,
            "Actuarial_Prob_30d": 0.5,
            "Actuarial_Prob_60d": 0.5,
            "Actuarial_Reserve_Severity": 0.0
        }
