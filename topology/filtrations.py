"""Tri-parameter filtrations for Tulsa housing topology.

The filtration is indexed by

    (lambda_1, lambda_2, lambda_3)
      = (affordability, spatial density, opportunity).

A simplex enters the complex at the componentwise maximum of its vertices'
lambda-coordinates. This skeletonized Vietoris-Rips construction is the local
fallback for environments where `multipers` is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from data.preprocess import FEATURE_COLUMNS, TOPOLOGY_PARAMETER_COLUMNS, add_topological_parameters, minmax_scale


@dataclass
class FiltrationResult:
    """Container for a finite tri-parameter simplicial filtration."""

    lambda_points: np.ndarray
    feature_points: np.ndarray
    edges: np.ndarray
    triangles: np.ndarray
    edge_births: np.ndarray
    triangle_births: np.ndarray
    grid_axes: tuple[np.ndarray, np.ndarray, np.ndarray]
    backend: str = "pure-python"
    metadata: dict[str, object] = field(default_factory=dict)


def _select_points(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if len(df) <= max_points:
        return df.reset_index(drop=True)
    weights = df.get("topology_weight", pd.Series(np.ones(len(df)), index=df.index)).to_numpy(dtype=float)
    weights = np.maximum(weights, 1e-9)
    probabilities = weights / weights.sum()
    rng = np.random.default_rng(918)
    idx = rng.choice(df.index.to_numpy(), size=max_points, replace=False, p=probabilities)
    return df.loc[idx].sort_values(["date", "zip_code", "property_id"]).reset_index(drop=True)


def _feature_matrix(df: pd.DataFrame) -> np.ndarray:
    if {"umap_x", "umap_y", "umap_z"}.issubset(df.columns):
        return df[["umap_x", "umap_y", "umap_z"]].to_numpy(dtype=float)
    available = [column for column in FEATURE_COLUMNS if column in df.columns]
    return minmax_scale(df[available].to_numpy(dtype=float))


def _pairwise_distances(points: np.ndarray) -> np.ndarray:
    diff = points[:, None, :] - points[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def skeletonized_edges(feature_points: np.ndarray, quantile: float = 0.055, max_edges: int = 4500) -> np.ndarray:
    """Build a sparse Rips-like one-skeleton from a distance quantile."""

    n = len(feature_points)
    if n < 2:
        return np.empty((0, 2), dtype=int)
    distances = _pairwise_distances(feature_points)
    upper = distances[np.triu_indices(n, k=1)]
    positive = upper[upper > 0]
    if positive.size == 0:
        return np.empty((0, 2), dtype=int)
    threshold = float(np.quantile(positive, quantile))
    candidates = np.argwhere(np.triu(distances <= threshold, k=1))
    if len(candidates) <= max_edges:
        return candidates.astype(int)
    edge_distances = distances[candidates[:, 0], candidates[:, 1]]
    keep = np.argsort(edge_distances)[:max_edges]
    return candidates[keep].astype(int)


def skeletonized_triangles(edges: np.ndarray, n_vertices: int, max_triangles: int = 6000) -> np.ndarray:
    """Fill short 3-cycles to approximate a sparse two-skeleton."""

    if len(edges) == 0 or n_vertices < 3:
        return np.empty((0, 3), dtype=int)
    adjacency = [set() for _ in range(n_vertices)]
    for i, j in edges:
        adjacency[int(i)].add(int(j))
        adjacency[int(j)].add(int(i))

    triangles: list[tuple[int, int, int]] = []
    for i in range(n_vertices):
        neighbors = sorted(j for j in adjacency[i] if j > i)
        for offset, j in enumerate(neighbors):
            common = adjacency[j].intersection(neighbors[offset + 1 :])
            for k in sorted(common):
                triangles.append((i, j, k))
                if len(triangles) >= max_triangles:
                    return np.asarray(triangles, dtype=int)
    return np.asarray(triangles, dtype=int)


def simplex_birth(lambda_points: np.ndarray, simplices: np.ndarray) -> np.ndarray:
    """Return componentwise birth coordinates for edges or triangles."""

    if simplices.size == 0:
        return np.empty((0, lambda_points.shape[1]), dtype=float)
    return lambda_points[simplices].max(axis=1)


def build_tri_parameter_filtration(
    df: pd.DataFrame,
    max_points: int = 520,
    grid_size: int = 14,
    edge_quantile: float = 0.055,
) -> FiltrationResult:
    """Construct the TTAS tri-parameter filtration.

    If `multipers` is importable we record that the research backend is
    available in the metadata. The local skeleton is still created because it
    supports the dashboard, fallback invariants, and smoke tests.
    """

    enriched = add_topological_parameters(df) if "affordability_index" not in df.columns else df.copy()
    selected = _select_points(enriched, max_points=max_points)
    lambda_points = minmax_scale(selected[TOPOLOGY_PARAMETER_COLUMNS].to_numpy(dtype=float))
    feature_points = _feature_matrix(selected)
    feature_points = minmax_scale(feature_points)

    edges = skeletonized_edges(feature_points, quantile=edge_quantile)
    triangles = skeletonized_triangles(edges, n_vertices=len(selected))
    edge_births = simplex_birth(lambda_points, edges)
    triangle_births = simplex_birth(lambda_points, triangles)
    axes = tuple(np.linspace(0.0, 1.0, grid_size) for _ in range(3))

    backend = "pure-python"
    multipers_available = False
    try:
        import multipers  # noqa: F401

        multipers_available = True
        backend = "multipers-ready"
    except Exception:
        multipers_available = False

    return FiltrationResult(
        lambda_points=lambda_points,
        feature_points=feature_points,
        edges=edges,
        triangles=triangles,
        edge_births=edge_births,
        triangle_births=triangle_births,
        grid_axes=axes,
        backend=backend,
        metadata={
            "n_vertices": int(len(selected)),
            "n_edges": int(len(edges)),
            "n_triangles": int(len(triangles)),
            "selected_property_ids": selected["property_id"].tolist() if "property_id" in selected else [],
            "multipers_available": multipers_available,
            "lambda_columns": TOPOLOGY_PARAMETER_COLUMNS,
        },
    )
