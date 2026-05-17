"""Data construction utilities for TTAS.

The data package synthesizes and enriches the Tulsa housing manifold used by
the topology, decision, and dashboard layers.
"""

from .fetch_data import TulsaDataConfig, generate_tulsa_manifold, get_data_mode, load_or_create_dataset
from .lineage import build_lineage_frame, build_source_summary_frame, classify_columns_by_nature, get_data_nature, get_natures_for_columns, nature_badge_html
from .preprocess import FEATURE_COLUMNS, add_topological_parameters, prepare_feature_matrix
from .real_data import enrich_with_public_sources

__all__ = [
    "TulsaDataConfig",
    "generate_tulsa_manifold",
    "get_data_mode",
    "load_or_create_dataset",
    "FEATURE_COLUMNS",
    "add_topological_parameters",
    "prepare_feature_matrix",
    "enrich_with_public_sources",
    "build_lineage_frame",
    "build_source_summary_frame",
    "classify_columns_by_nature",
    "get_data_nature",
    "get_natures_for_columns",
    "nature_badge_html",
]
