"""Persistence silhouettes and Betti curves."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.preprocess import add_topological_parameters, monthly_sample
from .filtrations import build_tri_parameter_filtration
from .invariants import finite_intervals, persistence_diagrams


def betti_curve(diagram: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """Evaluate beta(t) as the count of intervals alive at t."""

    finite = finite_intervals(diagram)
    if finite.size == 0:
        return np.zeros(len(xs), dtype=float)
    return np.asarray([np.sum((finite[:, 0] <= x) & (x < finite[:, 1])) for x in xs], dtype=float)


def persistence_silhouette(diagram: np.ndarray, xs: np.ndarray, power: float = 1.0) -> np.ndarray:
    """Compute the weighted persistence silhouette.

    The silhouette is a weighted average of tent functions, with weights
    persistence^power.
    """

    finite = finite_intervals(diagram)
    if finite.size == 0:
        return np.zeros(len(xs), dtype=float)
    lengths = finite[:, 1] - finite[:, 0]
    weights = np.power(lengths, power)
    total = weights.sum()
    if total <= 0:
        return np.zeros(len(xs), dtype=float)
    tents = []
    for birth, death in finite:
        midpoint = 0.5 * (birth + death)
        height = 0.5 * (death - birth)
        tents.append(np.maximum(0.0, height - np.abs(xs - midpoint)))
    return np.average(np.vstack(tents), axis=0, weights=weights)


def compute_silhouette_suite(
    df: pd.DataFrame,
    current_date: str | pd.Timestamp | None = None,
    baseline_date: str | pd.Timestamp | None = None,
    max_points: int = 260,
    grid_points: int = 220,
) -> dict[str, object]:
    """Compare current and baseline silhouettes and Betti curves."""

    enriched = add_topological_parameters(df) if "affordability_index" not in df.columns else df.copy()
    current_date = pd.Timestamp(current_date or enriched["date"].max())
    if baseline_date is None:
        baseline_date = sorted(pd.to_datetime(enriched["date"].unique()))[min(23, enriched["date"].nunique() - 1)]
    baseline_date = pd.Timestamp(baseline_date)

    current_slice = monthly_sample(enriched, date=current_date, max_points=max_points)
    baseline_slice = monthly_sample(enriched, date=baseline_date, max_points=max_points)
    current_filtration = build_tri_parameter_filtration(current_slice, max_points=max_points, grid_size=8)
    baseline_filtration = build_tri_parameter_filtration(baseline_slice, max_points=max_points, grid_size=8)
    current_diagrams = persistence_diagrams(current_filtration.feature_points, maxdim=1)
    baseline_diagrams = persistence_diagrams(baseline_filtration.feature_points, maxdim=1)

    max_death = 1.0
    for diagram in [*current_diagrams.values(), *baseline_diagrams.values()]:
        finite = finite_intervals(diagram)
        if finite.size:
            max_death = max(max_death, float(np.max(finite[:, 1])))
    xs = np.linspace(0.0, max_death, grid_points)
    records = []
    for homology in ["H0", "H1"]:
        for x, cur_sil, base_sil, cur_betti, base_betti in zip(
            xs,
            persistence_silhouette(current_diagrams[homology], xs),
            persistence_silhouette(baseline_diagrams[homology], xs),
            betti_curve(current_diagrams[homology], xs),
            betti_curve(baseline_diagrams[homology], xs),
        ):
            records.append(
                {
                    "scale": float(x),
                    "homology": homology,
                    "current_silhouette": float(cur_sil),
                    "baseline_silhouette": float(base_sil),
                    "silhouette_delta": float(cur_sil - base_sil),
                    "current_betti": float(cur_betti),
                    "baseline_betti": float(base_betti),
                    "betti_delta": float(cur_betti - base_betti),
                }
            )
    return {
        "current_date": current_date,
        "baseline_date": baseline_date,
        "frame": pd.DataFrame(records),
        "current_diagrams": current_diagrams,
        "baseline_diagrams": baseline_diagrams,
    }
