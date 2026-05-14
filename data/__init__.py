"""Data construction utilities for TTAS.

The data package synthesizes and enriches the Tulsa housing manifold used by
the topology, decision, and dashboard layers.
"""

from .fetch_data import TulsaDataConfig, generate_tulsa_manifold, load_or_create_dataset
from .preprocess import FEATURE_COLUMNS, add_topological_parameters, prepare_feature_matrix

__all__ = [
    "TulsaDataConfig",
    "generate_tulsa_manifold",
    "load_or_create_dataset",
    "FEATURE_COLUMNS",
    "add_topological_parameters",
    "prepare_feature_matrix",
]
