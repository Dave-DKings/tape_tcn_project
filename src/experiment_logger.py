"""
Experiment Logger Module
========================
Saves training and testing metrics to structured files (CSV by default, Excel optional).

Usage:
    from src.experiment_logger import ExperimentLogger
    
    logger = ExperimentLogger(experiment_name="TCN_Baseline", phase="training")
    logger.log_episode(episode=1, days=828, initial=100000, final=134000, ...)
    logger.save()  # Saves to CSV/Excel depending on configuration
"""

import pandas as pd
import os
from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np


class ExperimentLogger:
    """
    Logs experiment metrics and saves them to Excel files.
    
    Separate files are created for training and testing phases.
    """
    
    def __init__(self, experiment_name: str, phase: str = "training", 
                 output_dir: str = None,
                 output_format: str = "csv"):
        """
        Initialize experiment logger.
        
        Args:
            experiment_name: Name of the experiment (e.g., "TCN_Baseline")
            phase: "training" or "testing"
            output_dir: Directory to save Excel files (default: results/)
            output_format: "csv" or "xlsx"
        """
        self.experiment_name = experiment_name
        self.phase = phase
        self.output_format = output_format.lower()
        if self.output_format not in {"csv", "xlsx"}:
            raise ValueError(f"Unsupported output_format: {output_format}")
        
        # Set output directory
        if output_dir is None:
            base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            output_dir = os.path.join(base_path, 'results', 'logs')
        
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize data storage
        self.episodes_data = []
        self.summary_data = {}
        
        # Timestamp for unique filenames
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def log_episode(self, episode: int = None, update: int = None,
                   days: int = 0, initial_balance: float = 0.0,
                   final_balance: float = 0.0, total_return_pct: float = 0.0,
                   volatility: float = 0.0, sharpe_ratio: float = 0.0,
                   sortino_ratio: float = 0.0, max_drawdown: float = 0.0,
                   win_rate: float = 0.0, **kwargs):
        """
        Log metrics from a single episode.
        
        Args:
            episode: Episode number (for testing)
            update: Update number (for training)
            days: Number of days traded
            initial_balance: Starting portfolio value
            final_balance: Ending portfolio value
            total_return_pct: Total return percentage
            volatility: Annualized volatility
            sharpe_ratio: Sharpe ratio
            sortino_ratio: Sortino ratio
            max_drawdown: Maximum drawdown
            win_rate: Win rate (% of positive return days)
            **kwargs: Additional metrics (e.g., actor_loss, critic_loss, entropy, etc.)
        """
        episode_data = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'episode': episode if episode is not None else len(self.episodes_data) + 1,
            'update': update,
            'days_traded': days,
            'initial_balance': initial_balance,
            'final_balance': final_balance,
            'total_return_pct': total_return_pct,
            'volatility_pct': volatility * 100 if volatility < 10 else volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown_pct': abs(max_drawdown) * 100 if abs(max_drawdown) < 1 else abs(max_drawdown),
            'win_rate_pct': win_rate * 100 if win_rate <= 1 else win_rate,
        }
        
        # Add any additional metrics (RL training metrics)
        episode_data.update(kwargs)
        
        self.episodes_data.append(episode_data)
        
    def log_summary(self, **kwargs):
        """
        Log summary statistics for the entire experiment.
        
        Args:
            **kwargs: Summary metrics (mean_return, std_return, etc.)
        """
        self.summary_data.update(kwargs)
        
    def save(self, filename: str = None) -> str:
        """
        Save logged data to Excel file.
        
        Args:
            filename: Optional custom filename (without extension)
            
        Returns:
            Path(s) to saved file(s)
        """
        if filename is None:
            filename = f"{self.experiment_name}_{self.phase}_{self.timestamp}"

        base_filename, ext = os.path.splitext(filename)
        if not base_filename:
            base_filename = f"{self.experiment_name}_{self.phase}_{self.timestamp}"

        if self.output_format == "xlsx":
            if not ext:
                filename = f"{base_filename}.xlsx"
            elif ext.lower() != ".xlsx":
                filename = f"{base_filename}.xlsx"

            filepath = os.path.join(self.output_dir, filename)
            
            # Create Excel writer
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                # Sheet 1: Episode-by-episode data
                if self.episodes_data:
                    df_episodes = pd.DataFrame(self.episodes_data)
                    df_episodes.to_excel(writer, sheet_name='Episodes', index=False)
                
                # Sheet 2: Summary statistics
                if self.episodes_data:
                    df_summary = self._calculate_summary()
                    df_summary.to_excel(writer, sheet_name='Summary', index=True)
                
                # Sheet 3: Custom summary data
                if self.summary_data:
                    df_custom = pd.DataFrame([self.summary_data])
                    df_custom.to_excel(writer, sheet_name='Custom_Summary', index=False)
            
            print(f"✅ Experiment logs saved: {filepath}")
            return filepath

        # CSV fallback/output
        filepaths = []
        base_path = os.path.join(self.output_dir, base_filename)
        
        if self.episodes_data:
            df_episodes = pd.DataFrame(self.episodes_data)
            episodes_path = f"{base_path}_episodes.csv"
            df_episodes.to_csv(episodes_path, index=False)
            filepaths.append(episodes_path)
        
        if self.episodes_data:
            df_summary = self._calculate_summary()
            summary_path = f"{base_path}_summary.csv"
            df_summary.to_csv(summary_path, index=True)
            filepaths.append(summary_path)
        
        if self.summary_data:
            df_custom = pd.DataFrame([self.summary_data])
            custom_path = f"{base_path}_custom_summary.csv"
            df_custom.to_csv(custom_path, index=False)
            filepaths.append(custom_path)
        
        if not filepaths:
            # Create an empty placeholder file to indicate save attempt
            placeholder_path = f"{base_path}.csv"
            pd.DataFrame().to_csv(placeholder_path, index=False)
            filepaths.append(placeholder_path)
        
        print(f"✅ Experiment logs saved (CSV): {', '.join(filepaths)}")
        return filepaths if len(filepaths) > 1 else filepaths[0]
    
    def _calculate_summary(self) -> pd.DataFrame:
        """Calculate summary statistics from episode data."""
        df = pd.DataFrame(self.episodes_data)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        summary_stats = {}
        for col in numeric_cols:
            if col not in ['episode', 'update', 'timestamp']:
                summary_stats[col] = {
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'median': df[col].median(),
                }
        
        summary_df = pd.DataFrame(summary_stats).T
        summary_df.index.name = 'Metric'
        
        return summary_df
    
    def clear(self):
        """Clear logged data (useful for starting new experiment)."""
        self.episodes_data = []
        self.summary_data = {}


def create_training_logger(experiment_name: str, output_dir: str = None) -> ExperimentLogger:
    """
    Convenience function to create a training logger.
    
    Args:
        experiment_name: Name of the experiment
        output_dir: Directory to save logs
        
    Returns:
        ExperimentLogger instance
    """
    return ExperimentLogger(experiment_name, phase="training", output_dir=output_dir)


def create_testing_logger(experiment_name: str, output_dir: str = None) -> ExperimentLogger:
    """
    Convenience function to create a testing logger.
    
    Args:
        experiment_name: Name of the experiment
        output_dir: Directory to save logs
        
    Returns:
        ExperimentLogger instance
    """
    return ExperimentLogger(experiment_name, phase="testing", output_dir=output_dir)


# Example usage:
if __name__ == "__main__":
    # Training example
    train_logger = create_training_logger("TCN_Baseline_Test")
    
    for i in range(5):
        train_logger.log_episode(
            update=i+1,
            days=828,
            initial_balance=100000.0,
            final_balance=100000.0 + np.random.randn() * 10000,
            total_return_pct=np.random.randn() * 20,
            volatility=0.15 + np.random.randn() * 0.05,
            sharpe_ratio=0.8 + np.random.randn() * 0.3,
            sortino_ratio=1.0 + np.random.randn() * 0.3,
            max_drawdown=-(0.10 + np.random.rand() * 0.10),
            win_rate=0.55 + np.random.rand() * 0.10
        )
    
    train_logger.log_summary(
        total_episodes=5,
        best_sharpe=max([ep['sharpe_ratio'] for ep in train_logger.episodes_data])
    )
    
    train_logger.save()
    
    print("Example logger test completed successfully!")
