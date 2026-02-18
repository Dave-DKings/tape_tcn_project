"""
Hybrid Profile Orchestrator - Dynamic Utility Profile Management

This module implements the sophisticated Hybrid Meta-Strategy system that
dynamically switches the agent's utility profile based on a three-tier
decision hierarchy:

1. Performance Override Rules (Highest Priority)
   - Monitor agent's recent performance
   - Switch to protective/recovery profiles when needed
   - Examples: Capital Preservation on high drawdown

2. Curriculum Learning (Medium Priority)
   - Structured learning phases for agent development
   - Foundation → Exploration → Specialization → Mastery
   - Ensures robust learning progression

3. Market Heuristic Rules (Lowest Priority, Default)
   - Market regime detection and adaptive responses
   - Uses VIX, yield curve, volatility, trend indicators
   - Fallback strategy when no overrides/curriculum active

The system processes in order: Performance → Curriculum → Market → Default

Author: AI Assistant
Date: October 2, 2025
"""

import numpy as np
import pandas as pd
import sys
import os

# Ensure imports work correctly
try:
    from reward_utils import calculate_rolling_performance, asymmetric_sigmoid_utility
except ImportError:
    # Add parent directory to path if needed
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    from reward_utils import calculate_rolling_performance, asymmetric_sigmoid_utility
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridProfileOrchestrator:
    """
    Orchestrates dynamic switching of utility profiles using a hybrid strategy.
    
    This class implements the three-tier decision hierarchy for profile switching:
    1. Performance-based overrides (safety and opportunity)
    2. Curriculum-based learning progression
    3. Market-adaptive heuristics (fallback)
    
    The orchestrator maintains state about:
    - Current active profile
    - Episode history and rolling performance
    - Market indicators and regime
    - Curriculum phase progress
    - Switching constraints and cooldowns
    """
    
    def __init__(self, 
                 config: Dict[str, Any],
                 available_profiles: List[Dict[str, Any]],
                 log_file: Optional[str] = None):
        """
        Initialize the Hybrid Profile Orchestrator.
        
        Args:
            config: Configuration dictionary for profile switching
            available_profiles: List of available utility profile dictionaries
            log_file: Optional file path for detailed decision logging
        """
        self.config = config
        self.available_profiles = {p['name']: p for p in available_profiles}
        self.profile_names = list(self.available_profiles.keys())
        
        # Extract configuration components
        self.switching_strategy = config.get('switching_strategy', 'fixed')
        self.hybrid_config = config.get('hybrid_switching_config', {})
        
        # Performance override configuration
        self.perf_override_config = self.hybrid_config.get('performance_override_rules', {})
        self.perf_override_enabled = self.perf_override_config.get('enable', False)
        
        # Curriculum configuration
        self.curriculum_config = self.hybrid_config.get('curriculum_config', {})
        self.curriculum_enabled = self.curriculum_config.get('enable', False)
        
        # Market heuristic configuration
        self.market_config = self.hybrid_config.get('market_heuristic_rules', {})
        self.market_enabled = self.market_config.get('enable', False)
        
        # Adaptive scoring configuration (fallback optimisation)
        self.adaptive_config = self.hybrid_config.get('adaptive_scoring', {})
        self.adaptive_enabled = self.adaptive_config.get('enable', True)
        self.adaptive_min_delta = self.adaptive_config.get('min_score_delta', 0.05)
        self.adaptive_rolling_window = self.adaptive_config.get('rolling_window', 5)
        
        # Switching constraints
        self.constraints = self.hybrid_config.get('switching_constraints', {})
        self.min_episodes_before_switch = self.constraints.get('min_episodes_before_switch', 5)
        self.switch_cooldown = self.constraints.get('switch_cooldown_episodes', 2)
        
        # State tracking
        self.current_profile_name = self.profile_names[0]  # Default to first profile
        self.current_episode = 0
        self.episodes_in_current_profile = 0
        self.episodes_since_last_switch = 0
        self.total_switches = 0
        
        # Episode history
        self.episode_history = []
        
        # Decision logging
        self.log_file = log_file
        self.decision_log = []
        
        logger.info(f"HybridProfileOrchestrator initialized with strategy: {self.switching_strategy}")
        logger.info(f"  Performance Override: {self.perf_override_enabled}")
        logger.info(f"  Curriculum Learning: {self.curriculum_enabled}")
        logger.info(f"  Market Heuristics: {self.market_enabled}")
        logger.info(f"  Available Profiles: {self.profile_names}")
    
    def get_current_profile(self) -> Dict[str, Any]:
        """
        Get the current active utility profile.
        
        Returns:
            Current utility profile dictionary
        """
        return self.available_profiles[self.current_profile_name]
    
    def select_profile(self,
                      episode_metrics: Optional[Dict[str, float]] = None,
                      market_indicators: Optional[Dict[str, float]] = None) -> Tuple[str, str]:
        """
        Select the appropriate utility profile based on the hybrid strategy.
        
        This is the main entry point that implements the three-tier hierarchy:
        1. Check performance override rules
        2. Check curriculum phase
        3. Check market heuristic rules
        4. Use default profile
        
        Args:
            episode_metrics: Metrics from the most recent episode
            market_indicators: Current market indicator values
            
        Returns:
            Tuple of (profile_name, selection_reason)
        """
        self.current_episode += 1
        self.episodes_in_current_profile += 1
        self.episodes_since_last_switch += 1
        
        # Store episode info in history
        if episode_metrics is not None:
            self.episode_history.append({
                'episode': self.current_episode,
                'metrics': episode_metrics,
                'profile': self.current_profile_name,
                'timestamp': datetime.now().isoformat()
            })
        
        # Check if we can switch (cooldown and minimum episodes constraints)
        can_switch = self._check_switching_constraints()
        
        selected_profile = self.current_profile_name
        reason = "No change - constraints active"
        
        if can_switch or self.current_episode == 1:  # Always allow first episode
            # Tier 1: Performance Override Rules (Highest Priority)
            if self.perf_override_enabled:
                override_profile, override_reason = self._check_performance_overrides()
                if override_profile is not None:
                    selected_profile = override_profile
                    reason = f"Performance Override: {override_reason}"
                    self._execute_profile_switch(selected_profile, reason)
                    return selected_profile, reason
            
            # Tier 2: Curriculum Learning (Medium Priority)
            if self.curriculum_enabled:
                curriculum_profile, curriculum_reason = self._check_curriculum_phase()
                if curriculum_profile is not None:
                    selected_profile = curriculum_profile
                    reason = f"Curriculum: {curriculum_reason}"
                    self._execute_profile_switch(selected_profile, reason)
                    return selected_profile, reason
        
            # Tier 3: Market Heuristic Rules (Lowest Priority)
            if self.market_enabled and market_indicators is not None:
                market_profile, market_reason = self._check_market_heuristics(market_indicators)
                if market_profile is not None:
                    selected_profile = market_profile
                    reason = f"Market Heuristic: {market_reason}"
                    self._execute_profile_switch(selected_profile, reason)
                    return selected_profile, reason
            
            # Tier 4: Adaptive scoring across all profiles (data-driven fallback)
            if self.adaptive_enabled:
                adaptive_profile, adaptive_reason = self._select_adaptive_profile(episode_metrics)
                if adaptive_profile is not None and adaptive_profile != self.current_profile_name:
                    selected_profile = adaptive_profile
                    reason = adaptive_reason
                    self._execute_profile_switch(selected_profile, reason)
                    return selected_profile, reason
        
        # If no changes, continue with current profile
        return self.current_profile_name, "Continuing current profile"
    
    def _check_switching_constraints(self) -> bool:
        """
        Check if switching is allowed based on constraints.
        
        Returns:
            True if switching is allowed, False otherwise
        """
        # Check minimum episodes in current profile
        if self.episodes_in_current_profile < self.min_episodes_before_switch:
            return False
        
        # Check cooldown since last switch
        if self.episodes_since_last_switch < self.switch_cooldown:
            return False
        
        return True
    
    def _check_performance_overrides(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Check if any performance override rules are triggered.
        
        Returns:
            Tuple of (profile_name, reason) or (None, None) if no override
        """
        if not self.perf_override_enabled:
            return None, None
        
        rules = self.perf_override_config.get('rules', [])
        lookback = self.perf_override_config.get('lookback_episodes', 20)
        
        if len(self.episode_history) < 2:
            return None, None
        
        # Calculate rolling performance metrics
        rolling_metrics = calculate_rolling_performance(self.episode_history, window=lookback)
        
        # Sort rules by priority
        sorted_rules = sorted(rules, key=lambda r: r.get('priority', 999))
        
        # Check each rule
        for rule in sorted_rules:
            condition = rule.get('condition', {})
            metric_name = condition.get('metric')
            operator = condition.get('operator')
            threshold = condition.get('threshold')
            
            if metric_name not in rolling_metrics:
                continue
            
            metric_value = rolling_metrics[metric_name]
            
            # Evaluate condition
            triggered = False
            if operator == 'greater_than' and metric_value > threshold:
                triggered = True
            elif operator == 'less_than' and metric_value < threshold:
                triggered = True
            elif operator == 'equals' and abs(metric_value - threshold) < 1e-6:
                triggered = True
            
            if triggered:
                action = rule.get('action', {})
                target_profile = action.get('switch_to')
                rule_name = rule.get('name', 'unknown')
                
                if target_profile and target_profile in self.available_profiles:
                    reason = f"{rule_name} ({metric_name}={metric_value:.3f})"
                    return target_profile, reason
        
        return None, None
    
    def _check_curriculum_phase(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Check which curriculum phase we're in and return appropriate profile.
        
        Returns:
            Tuple of (profile_name, reason) or (None, None) if not applicable
        """
        if not self.curriculum_enabled:
            return None, None
        
        phases = self.curriculum_config.get('phases', [])
        
        # Find current phase
        for phase in phases:
            episode_range = phase.get('episode_range', [0, float('inf')])
            if episode_range[0] <= self.current_episode < episode_range[1]:
                phase_name = phase.get('name', 'unknown')
                profile_spec = phase.get('profile')
                
                # Handle different profile specifications
                if profile_spec == 'cyclical':
                    # Cycle through all profiles
                    cycle_length = phase.get('cycle_length', 30)
                    phase_offset = self.current_episode - episode_range[0]
                    profile_idx = (phase_offset // cycle_length) % len(self.profile_names)
                    profile_name = self.profile_names[profile_idx]
                    reason = f"{phase_name} - Cyclical rotation"
                    return profile_name, reason
                
                elif profile_spec == 'market_adaptive':
                    # Use market heuristics (delegate to tier 3)
                    return None, None
                
                elif profile_spec == 'hybrid_full':
                    # Use full hybrid logic (continue to tier 3)
                    return None, None
                
                elif profile_spec in self.available_profiles:
                    # Fixed profile for this phase
                    reason = f"{phase_name}"
                    return profile_spec, reason
        
        return None, None
    
    def _check_market_heuristics(self, 
                                 market_indicators: Dict[str, float]) -> Tuple[Optional[str], Optional[str]]:
        """
        Check market heuristic rules and return appropriate profile.
        
        Args:
            market_indicators: Dictionary of current market indicators
            
        Returns:
            Tuple of (profile_name, reason) or (None, None) if no match
        """
        if not self.market_enabled:
            return None, None
        
        rules = self.market_config.get('rules', [])
        
        # Evaluate rules by confidence (highest first)
        sorted_rules = sorted(rules, key=lambda r: r.get('confidence', 0.5), reverse=True)
        
        for rule in sorted_rules:
            rule_name = rule.get('name', 'unknown')
            conditions = rule.get('conditions', [])
            logic = rule.get('logic', 'AND')
            action = rule.get('action', {})
            
            # Evaluate all conditions
            condition_results = []
            for cond in conditions:
                indicator = cond.get('indicator')
                operator = cond.get('operator')
                threshold = cond.get('threshold')
                
                if indicator not in market_indicators:
                    condition_results.append(False)
                    continue
                
                value = market_indicators[indicator]
                
                # Evaluate condition
                if operator == 'greater_than':
                    condition_results.append(value > threshold)
                elif operator == 'less_than':
                    condition_results.append(value < threshold)
                elif operator == 'equals':
                    condition_results.append(value == threshold)
                else:
                    condition_results.append(False)
            
            # Apply logic
            if len(condition_results) == 0:
                rule_triggered = True  # No conditions means always true (default rule)
            elif logic == 'AND':
                rule_triggered = all(condition_results)
            elif logic == 'OR':
                rule_triggered = any(condition_results)
            else:
                rule_triggered = False
            
            # If rule triggered, return the action
            if rule_triggered:
                target_profile = action.get('switch_to')
                if target_profile and target_profile in self.available_profiles:
                    return target_profile, rule_name
        
        return None, None
    
    def _execute_profile_switch(self, new_profile: str, reason: str):
        """
        Execute a profile switch and update internal state.
        
        Args:
            new_profile: Name of the new profile to switch to
            reason: Reason for the switch
        """
        if new_profile != self.current_profile_name:
            old_profile = self.current_profile_name
            self.current_profile_name = new_profile
            self.episodes_in_current_profile = 0
            self.episodes_since_last_switch = 0
            self.total_switches += 1
            
            # Log the decision
            decision = {
                'episode': self.current_episode,
                'timestamp': datetime.now().isoformat(),
                'old_profile': old_profile,
                'new_profile': new_profile,
                'reason': reason
            }
            self.decision_log.append(decision)
            
            logger.info(f"Episode {self.current_episode}: Profile switched from "
                       f"{old_profile} to {new_profile}. Reason: {reason}")
            
            # Write to log file if specified
            if self.log_file:
                self._write_decision_log(decision)
    
    def _select_adaptive_profile(self, episode_metrics: Optional[Dict[str, float]]) -> Tuple[Optional[str], Optional[str]]:
        """
        Score all available profiles against the latest metrics and suggest a switch
        when a materially better match is found.
        """
        # Determine which metrics to evaluate
        metrics = episode_metrics
        if metrics is None and self.episode_history:
            metrics = self.episode_history[-1]['metrics']
        
        if metrics is None:
            return None, None
        
        if self.adaptive_config.get('use_history_average', True) and self.episode_history:
            window = max(1, self.adaptive_rolling_window)
            recent = self.episode_history[-window:]
            averaged = {}
            for entry in recent:
                for key, value in entry['metrics'].items():
                    averaged.setdefault(key, []).append(value)
            metrics = {k: float(np.mean(v)) for k, v in averaged.items() if len(v) > 0}
        
        metric_alias = {
            'sharpe': 'sharpe_ratio',
            'sortino': 'sortino_ratio',
            'mdd': 'max_drawdown',  # stored as positive magnitude
            'turnover': 'turnover',
            'skew': 'return_skew'
        }
        
        profile_scores = {}
        score_components = {}
        
        for profile_name, profile in self.available_profiles.items():
            mu = profile.get('mu')
            weights = profile.get('weights')
            order = profile.get('metrics_order', [])
            directions = profile.get('directions', [])
            a_bounds = profile.get('a_bounds')
            b_bounds = profile.get('b_bounds')
            
            # Detect profile format: sigmoid (k_minus/k_plus) vs legacy (sigma_sq)
            use_sigmoid = 'k_minus' in profile and 'k_plus' in profile
            
            if use_sigmoid:
                k_minus = profile.get('k_minus')
                k_plus = profile.get('k_plus')
            else:
                sigma_minus = profile.get('sigma_sq_minus')
                sigma_plus = profile.get('sigma_sq_plus')
            
            if mu is None or weights is None:
                continue
            if not use_sigmoid and (sigma_minus is None or sigma_plus is None):
                continue
            
            total_weight = 0.0
            score = 0.0
            component_details = []
            
            for idx, metric_key in enumerate(order):
                mapped_metric = metric_alias.get(metric_key)
                if mapped_metric is None:
                    continue
                
                raw_value = metrics.get(mapped_metric)
                if raw_value is None:
                    continue
                
                value = raw_value
                if metric_key == 'mdd':
                    # Convert drawdown magnitude into signed value (target is negative)
                    value = -abs(raw_value)
                
                target = float(mu[idx])
                
                if use_sigmoid:
                    # Use asymmetric sigmoid utility
                    direction = directions[idx] if idx < len(directions) else 'increasing'
                    a = float(a_bounds[idx]) if a_bounds is not None else -10.0
                    b = float(b_bounds[idx]) if b_bounds is not None else 10.0
                    component_score = asymmetric_sigmoid_utility(
                        x=value, mu=target,
                        k_minus=float(k_minus[idx]),
                        k_plus=float(k_plus[idx]),
                        a=a, b=b,
                        direction=direction
                    )
                else:
                    # Legacy Gaussian scoring
                    sigma = float(np.sqrt(sigma_plus[idx]) if value >= target else np.sqrt(sigma_minus[idx]))
                    sigma = max(sigma, 1e-6)
                    diff = (value - target) / sigma
                    component_score = np.exp(-0.5 * diff * diff)
                
                weight = float(weights[idx])
                score += weight * component_score
                total_weight += weight
                component_details.append((metric_key, value, target, component_score))
            
            if total_weight > 0:
                score /= total_weight
                profile_scores[profile_name] = score
                score_components[profile_name] = component_details
        
        if not profile_scores:
            return None, None
        
        best_profile = max(profile_scores, key=profile_scores.get)
        best_score = profile_scores[best_profile]
        current_score = profile_scores.get(self.current_profile_name, -np.inf)
        delta = best_score - current_score
        
        if best_profile != self.current_profile_name and delta >= self.adaptive_min_delta:
            components = []
            for metric, value, target, component_score in score_components.get(best_profile, []):
                components.append(f"{metric}:{value:.3f}->{target:.3f}")
            component_text = ", ".join(components)
            reason = (f"Adaptive scoring: {best_profile} score {best_score:.3f} "
                      f"(Δ {delta:.3f} vs {self.current_profile_name})")
            if component_text:
                reason += f" | metrics {component_text}"
            return best_profile, reason
        
        return None, None
    
    def _write_decision_log(self, decision: Dict[str, Any]):
        """
        Write a decision to the log file.
        
        Args:
            decision: Decision dictionary to log
        """
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(decision) + '\n')
        except Exception as e:
            logger.warning(f"Failed to write to decision log: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about profile switching behavior.
        
        Returns:
            Dictionary of statistics
        """
        # Count episodes per profile
        profile_counts = {name: 0 for name in self.profile_names}
        for ep in self.episode_history:
            profile = ep.get('profile')
            if profile in profile_counts:
                profile_counts[profile] += 1
        
        stats = {
            'total_episodes': self.current_episode,
            'total_switches': self.total_switches,
            'current_profile': self.current_profile_name,
            'episodes_in_current_profile': self.episodes_in_current_profile,
            'profile_distribution': profile_counts,
            'switch_rate': self.total_switches / max(1, self.current_episode)
        }
        
        return stats
    
    def reset(self):
        """Reset the orchestrator to initial state."""
        self.current_episode = 0
        self.episodes_in_current_profile = 0
        self.episodes_since_last_switch = 0
        self.total_switches = 0
        self.episode_history = []
        self.decision_log = []
        self.current_profile_name = self.profile_names[0]
        
        logger.info("HybridProfileOrchestrator reset to initial state")


class SimpleProfileManager:
    """
    Simplified profile manager for fixed or cyclical strategies.
    
    This is used when the full hybrid meta-strategy is not needed.
    """
    
    def __init__(self,
                 available_profiles: List[Dict[str, Any]],
                 strategy: str = 'fixed',
                 episodes_per_profile: int = 50):
        """
        Initialize simple profile manager.
        
        Args:
            available_profiles: List of available utility profiles
            strategy: 'fixed' or 'cyclical'
            episodes_per_profile: Episodes per profile for cyclical strategy
        """
        self.available_profiles = {p['name']: p for p in available_profiles}
        self.profile_names = list(self.available_profiles.keys())
        self.strategy = strategy
        self.episodes_per_profile = episodes_per_profile
        
        self.current_profile_name = self.profile_names[0]
        self.current_episode = 0
        
        logger.info(f"SimpleProfileManager initialized with strategy: {strategy}")
    
    def get_current_profile(self) -> Dict[str, Any]:
        """Get the current active profile."""
        return self.available_profiles[self.current_profile_name]
    
    def select_profile(self,
                      episode_metrics: Optional[Dict[str, float]] = None,
                      market_indicators: Optional[Dict[str, float]] = None) -> Tuple[str, str]:
        """
        Select profile based on simple strategy.
        
        Args:
            episode_metrics: Episode metrics (unused for simple strategies)
            market_indicators: Market indicators (unused for simple strategies)
            
        Returns:
            Tuple of (profile_name, reason)
        """
        self.current_episode += 1
        
        if self.strategy == 'fixed':
            return self.current_profile_name, "Fixed profile"
        
        elif self.strategy == 'cyclical':
            # Rotate through profiles
            profile_idx = (self.current_episode // self.episodes_per_profile) % len(self.profile_names)
            self.current_profile_name = self.profile_names[profile_idx]
            return self.current_profile_name, f"Cyclical rotation (episode {self.current_episode})"
        
        return self.current_profile_name, "Unknown strategy"
    
    def reset(self):
        """Reset to initial state."""
        self.current_episode = 0
        self.current_profile_name = self.profile_names[0]


def create_profile_manager(config: Dict[str, Any],
                          available_profiles: List[Dict[str, Any]]) -> Any:
    """
    Factory function to create the appropriate profile manager.
    
    Args:
        config: Configuration dictionary
        available_profiles: List of available utility profiles
        
    Returns:
        ProfileManager instance (HybridProfileOrchestrator or SimpleProfileManager)
    """
    strategy = config.get('switching_strategy', 'fixed')
    
    if strategy == 'hybrid_meta_strategy':
        log_file = None
        hybrid_config = config.get('hybrid_switching_config', {})
        logging_config = hybrid_config.get('logging', {})
        if logging_config.get('log_decisions', False):
            log_file = logging_config.get('decision_log_file', 'profile_decisions.log')
        
        return HybridProfileOrchestrator(
            config=config,
            available_profiles=available_profiles,
            log_file=log_file
        )
    
    elif strategy == 'cyclical':
        cyclical_config = config.get('cyclical_config', {})
        episodes_per_profile = cyclical_config.get('episodes_per_profile', 50)
        
        return SimpleProfileManager(
            available_profiles=available_profiles,
            strategy='cyclical',
            episodes_per_profile=episodes_per_profile
        )
    
    else:  # 'fixed' or unknown
        return SimpleProfileManager(
            available_profiles=available_profiles,
            strategy='fixed'
        )


# ============================================================================
# TESTING AND VALIDATION FUNCTIONS
# ============================================================================

def test_profile_manager():
    """Test profile manager with mock data."""
    print("\n=== Testing Profile Manager ===")
    
    # Import config
    try:
        from .config import PHASE2_CONFIG, ALL_PROFILES_LIST
    except ImportError:
        from config import PHASE2_CONFIG, ALL_PROFILES_LIST
    
    # Create orchestrator
    pm_config = PHASE2_CONFIG.get('profile_manager_params')
    if not pm_config:
        raise ValueError("profile_manager_params not configured in PHASE2_CONFIG")
    orchestrator = create_profile_manager(pm_config, ALL_PROFILES_LIST)
    
    print(f"Created orchestrator: {type(orchestrator).__name__}")
    print(f"Current profile: {orchestrator.get_current_profile()['name']}")
    
    # Simulate episodes
    for episode in range(10):
        mock_metrics = {
            'sharpe_ratio': np.random.randn() * 0.5 + 1.0,
            'max_drawdown': -np.random.rand() * 0.2
        }
        
        profile_name, reason = orchestrator.select_profile(episode_metrics=mock_metrics)
        print(f"  Episode {episode + 1}: Profile={profile_name}, Reason={reason}")
    
    # Print statistics
    stats = orchestrator.get_statistics()
    print(f"\nStatistics: {stats}")


if __name__ == "__main__":
    test_profile_manager()
    print("\n=== Profile Manager Module Tests Complete ===")
