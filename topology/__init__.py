"""Topological data analysis engine for TTAS."""

from .filtrations import FiltrationResult, build_tri_parameter_filtration
from .invariants import (
    compute_time_slice_invariants,
    euler_characteristic_surface,
    persistence_diagrams,
    persistence_entropy,
)
from .multiparameter import MultiparameterResult, compute_multiparameter_persistence
from .silhouettes import betti_curve, compute_silhouette_suite, persistence_silhouette

__all__ = [
    "FiltrationResult",
    "build_tri_parameter_filtration",
    "compute_time_slice_invariants",
    "euler_characteristic_surface",
    "persistence_diagrams",
    "persistence_entropy",
    "MultiparameterResult",
    "compute_multiparameter_persistence",
    "betti_curve",
    "compute_silhouette_suite",
    "persistence_silhouette",
]
