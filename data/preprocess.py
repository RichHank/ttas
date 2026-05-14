"""Preprocess the twelve-dimensional Tulsa manifold for TDA.

This module turns raw property-month observations into filtration-ready
coordinates. The three primary multiparameter axes are

    lambda_1 = affordability index,
    lambda_2 = spatial density,
    lambda_3 = opportunity score.

The implementation favors transparent numerical operations over hidden
pipelines so the topology modules can expose their assumptions directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "median_listing_price",
    "rent_to_price_ratio",
    "inventory_velocity",
    "property_tax_rate",
    "school_rating",
    "street_centrality",
    "amenity_density",
    "crime_index",
    "flood_risk_score",
    "walk_transit_score",
    "economic_mobility_index",
    "dti_max",
]

TOPOLOGY_PARAMETER_COLUMNS = [
    "affordability_index",
    "spatial_density",
    "opportunity_score",
]


def minmax_scale(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Scale columns to [0, 1] without requiring scikit-learn."""

    arr = np.asarray(values, dtype=float)
    lo = np.nanmin(arr, axis=0)
    hi = np.nanmax(arr, axis=0)
    return (arr - lo) / (hi - lo + eps)


def robust_zscore(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Median/IQR standardization for heterogeneous housing features."""

    arr = np.asarray(values, dtype=float)
    med = np.nanmedian(arr, axis=0)
    q25 = np.nanpercentile(arr, 25, axis=0)
    q75 = np.nanpercentile(arr, 75, axis=0)
    return (arr - med) / (q75 - q25 + eps)


def prepare_feature_matrix(
    df: pd.DataFrame,
    dti_max_override: float | None = None,
    feature_columns: list[str] | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Return a robust-scaled feature matrix and the dataframe used to build it."""

    feature_columns = feature_columns or FEATURE_COLUMNS
    working = df.copy()
    if dti_max_override is not None:
        working["dti_max"] = float(dti_max_override)

    missing = [column for column in feature_columns if column not in working.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing}")

    numeric = working[feature_columns].replace([np.inf, -np.inf], np.nan)
    numeric = numeric.fillna(numeric.median(numeric_only=True))
    matrix = robust_zscore(numeric.to_numpy(dtype=float))
    return matrix, working


def _nearest_neighbor_density(matrix: np.ndarray, k: int = 8) -> np.ndarray:
    """Compute inverse mean k-neighbor distance with an sklearn fallback."""

    n = len(matrix)
    if n <= 1:
        return np.ones(n)
    k_eff = min(max(k, 2), n)
    try:
        from sklearn.neighbors import NearestNeighbors

        nbrs = NearestNeighbors(n_neighbors=k_eff, metric="euclidean")
        distances, _ = nbrs.fit(matrix).kneighbors(matrix)
        mean_distance = distances[:, 1:].mean(axis=1)
    except Exception:
        diff = matrix[:, None, :] - matrix[None, :, :]
        distances = np.sqrt(np.sum(diff * diff, axis=2))
        distances.sort(axis=1)
        mean_distance = distances[:, 1:k_eff].mean(axis=1)

    raw_density = 1.0 / (mean_distance + 1e-9)
    return minmax_scale(raw_density.reshape(-1, 1)).ravel()


def add_topological_parameters(df: pd.DataFrame, k_neighbors: int = 8) -> pd.DataFrame:
    """Add lambda_1, lambda_2, lambda_3 plus an entropy proxy.

    The affordability axis is oriented so larger values mean "more viable to
    buy" after accounting for local income and DTI:

        lambda_1 = 1 - ownership_cost / max_affordable_payment.

    The density axis lambda_2 is the inverse local distance in the scaled
    twelve-dimensional manifold. The opportunity axis lambda_3 is a convex
    blend of schools, mobility, amenities, safety, and flood resilience.
    """

    matrix, working = prepare_feature_matrix(df)
    max_budget = working["annual_income_estimate"].to_numpy(dtype=float) * working["dti_max"].to_numpy(dtype=float) / 12.0
    ownership = working["ownership_cost_monthly"].to_numpy(dtype=float)
    raw_affordability = 1.0 - ownership / np.maximum(max_budget, 1.0)
    working["affordability_index"] = minmax_scale(raw_affordability.reshape(-1, 1)).ravel()
    working["spatial_density"] = _nearest_neighbor_density(matrix, k=k_neighbors)
    working["opportunity_score"] = (
        0.32 * working["school_rating"].to_numpy(dtype=float)
        + 0.24 * working["economic_mobility_index"].to_numpy(dtype=float)
        + 0.18 * working["amenity_density"].to_numpy(dtype=float)
        + 0.14 * (1.0 - working["crime_index"].to_numpy(dtype=float))
        + 0.12 * (1.0 - working["flood_risk_score"].to_numpy(dtype=float))
    )
    working["opportunity_score"] = minmax_scale(working[["opportunity_score"]].to_numpy()).ravel()

    scaled = minmax_scale(matrix)
    p = np.abs(scaled) + 1e-9
    p = p / p.sum(axis=1, keepdims=True)
    working["persistent_entropy_proxy"] = -np.sum(p * np.log(p), axis=1) / np.log(p.shape[1])
    working["topology_weight"] = (
        0.40 * working["affordability_index"]
        + 0.25 * working["spatial_density"]
        + 0.35 * working["opportunity_score"]
    )
    return working


def monthly_sample(df: pd.DataFrame, date: str | pd.Timestamp | None = None, max_points: int = 600) -> pd.DataFrame:
    """Select a reproducible monthly slice for topology computations."""

    working = df.copy()
    if date is None:
        date = working["date"].max()
    date = pd.Timestamp(date)
    slice_df = working[working["date"] == date]
    if len(slice_df) <= max_points:
        return slice_df.reset_index(drop=True)
    return slice_df.sample(max_points, random_state=918).reset_index(drop=True)
