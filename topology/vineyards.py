"""Time-varying persistence vineyards and change-point detection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.preprocess import FEATURE_COLUMNS, add_topological_parameters, minmax_scale
from .invariants import finite_intervals, persistence_diagrams, persistence_entropy


def _feature_points(df: pd.DataFrame, max_points: int = 260) -> np.ndarray:
    sample = df
    if len(sample) > max_points:
        sample = sample.sample(max_points, random_state=918)
    columns = [column for column in FEATURE_COLUMNS if column in sample.columns]
    return minmax_scale(sample[columns].to_numpy(dtype=float))


def bottleneck_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Compute bottleneck distance with persim or a Hausdorff fallback."""

    left_finite = finite_intervals(left)
    right_finite = finite_intervals(right)
    if left_finite.size == 0 and right_finite.size == 0:
        return 0.0
    try:
        from persim import bottleneck

        return float(bottleneck(left_finite, right_finite))
    except Exception:
        if left_finite.size == 0:
            return float(np.max(right_finite[:, 1] - right_finite[:, 0]))
        if right_finite.size == 0:
            return float(np.max(left_finite[:, 1] - left_finite[:, 0]))
        distances = np.sqrt(((left_finite[:, None, :] - right_finite[None, :, :]) ** 2).sum(axis=2))
        return float(max(distances.min(axis=1).max(), distances.min(axis=0).max()))


def compute_sliding_window_vineyards(
    df: pd.DataFrame,
    window_months: int = 12,
    stride_months: int = 1,
    max_points: int = 260,
) -> tuple[pd.DataFrame, dict[pd.Timestamp, dict[str, np.ndarray]]]:
    """Compute persistence diagrams and bottleneck drift over sliding windows."""

    enriched = add_topological_parameters(df) if "affordability_index" not in df.columns else df.copy()
    months = sorted(pd.to_datetime(enriched["date"].unique()))
    if len(months) < window_months:
        return pd.DataFrame(), {}

    records = []
    diagrams_by_window: dict[pd.Timestamp, dict[str, np.ndarray]] = {}
    baseline_diagram = None
    for start in range(0, len(months) - window_months + 1, stride_months):
        window_dates = months[start : start + window_months]
        window_df = enriched[enriched["date"].isin(window_dates)]
        points = _feature_points(window_df, max_points=max_points)
        diagrams = persistence_diagrams(points, maxdim=1)
        label = pd.Timestamp(window_dates[-1])
        diagrams_by_window[label] = diagrams
        if baseline_diagram is None:
            baseline_diagram = diagrams["H1"]
        distance = bottleneck_distance(diagrams["H1"], baseline_diagram)
        records.append(
            {
                "window_start": pd.Timestamp(window_dates[0]),
                "window_end": label,
                "bottleneck_to_baseline": distance,
                "entropy_h0": persistence_entropy(diagrams["H0"]),
                "entropy_h1": persistence_entropy(diagrams["H1"]),
                "h1_features": int(len(finite_intervals(diagrams["H1"]))),
            }
        )
    return pd.DataFrame(records), diagrams_by_window


def bayesian_blocks_change_points(sequence: pd.DataFrame, value_column: str = "bottleneck_to_baseline") -> pd.DataFrame:
    """Detect topological change points.

    Astropy's Bayesian blocks are used when available. The fallback flags large
    positive first differences in the bottleneck series.
    """

    if sequence.empty:
        return pd.DataFrame(columns=["date", "score", "method"])
    values = sequence[value_column].to_numpy(dtype=float)
    dates = pd.to_datetime(sequence["window_end"]).reset_index(drop=True)
    try:
        from astropy.stats import bayesian_blocks

        edges = bayesian_blocks(np.arange(len(values)), values)
        indices = sorted(set(int(round(edge)) for edge in edges[1:-1] if 0 <= int(round(edge)) < len(values)))
        return pd.DataFrame({"date": dates.iloc[indices].to_numpy(), "score": values[indices], "method": "bayesian_blocks"})
    except Exception:
        diffs = np.diff(values, prepend=values[0])
        threshold = diffs.mean() + 1.35 * (diffs.std() + 1e-9)
        indices = np.where(diffs >= threshold)[0]
        return pd.DataFrame({"date": dates.iloc[indices].to_numpy(), "score": values[indices], "method": "zscore_fallback"})


def vineyard_tracks(diagrams_by_window: dict[pd.Timestamp, dict[str, np.ndarray]], homology: str = "H1") -> pd.DataFrame:
    """Flatten persistence diagrams into animated vineyard tracks."""

    records = []
    for date, diagrams in diagrams_by_window.items():
        finite = finite_intervals(diagrams.get(homology, np.empty((0, 2))))
        if finite.size == 0:
            continue
        persistence = finite[:, 1] - finite[:, 0]
        order = np.argsort(-persistence)
        for track_id, idx in enumerate(order[:40]):
            records.append(
                {
                    "date": pd.Timestamp(date),
                    "track_id": int(track_id),
                    "birth": float(finite[idx, 0]),
                    "death": float(finite[idx, 1]),
                    "persistence": float(persistence[idx]),
                    "homology": homology,
                }
            )
    return pd.DataFrame(records)
