"""Dirichlet policy execution utilities."""

from .dirichlet_exec import (
    dirichlet_mean,
    dirichlet_mode_boundary_aware,
    controlled_stochastic_sample,
    apply_cash_tilt,
    smooth_weights,
    gated_deterministic_step,
)

__all__ = [
    'dirichlet_mean',
    'dirichlet_mode_boundary_aware',
    'controlled_stochastic_sample',
    'apply_cash_tilt',
    'smooth_weights',
    'gated_deterministic_step',
]
