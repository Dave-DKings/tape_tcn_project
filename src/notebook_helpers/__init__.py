"""Helper utilities exposed for keeping notebooks lightweight."""

from .tcn_phase1 import (
    configure_episode_length_curriculum,
    prepare_phase1_dataset,
    split_dataset_by_date,
    identify_covariance_columns,
    build_stage1_experiments,
    build_stage2_experiments,
    Phase1Dataset,
    Experiment6Result,
    run_experiment6_tape,
    Experiment6Evaluation,
    evaluate_experiment6_checkpoint,
    create_experiment6_result_stub,
    load_training_metadata_into_config,
)
from .backtests import (
    calculate_metrics_from_trajectory,
    run_backtest_deterministic,
    run_backtest_stochastic,
    run_backtest_walk_forward,
    describe_backtest_helpers,
)

__all__ = [
    "configure_episode_length_curriculum",
    "prepare_phase1_dataset",
    "split_dataset_by_date",
    "identify_covariance_columns",
    "build_stage1_experiments",
    "build_stage2_experiments",
    "Phase1Dataset",
    "Experiment6Result",
    "run_experiment6_tape",
    "Experiment6Evaluation",
    "evaluate_experiment6_checkpoint",
    "create_experiment6_result_stub",
    "load_training_metadata_into_config",
    "calculate_metrics_from_trajectory",
    "run_backtest_deterministic",
    "run_backtest_stochastic",
    "run_backtest_walk_forward",
    "describe_backtest_helpers",
]
