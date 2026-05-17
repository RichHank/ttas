"""Decision functions for the Tulsa Topological Affordability Spacetime."""

from .path_integral import UserBiography, buy_signal
from .phase_transition import euler_curvature_alert, infer_market_regime
from .opportunity_mapper import build_opportunity_graph
from .topological_boundary import compute_topological_boundary
from .analyst_notes import generate_all_regime_notes, RegimeNote
from .validation import build_validation_dashboard_data

__all__ = [
    "UserBiography",
    "buy_signal",
    "euler_curvature_alert",
    "infer_market_regime",
    "build_opportunity_graph",
    "compute_topological_boundary",
    "generate_all_regime_notes",
    "RegimeNote",
    "build_validation_dashboard_data",
]
