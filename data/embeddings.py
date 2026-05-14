"""Manifold embeddings for Tulsa affordability spacetime.

The prompt asks for diffusion maps, supervised UMAP, and Takens time-delay
embeddings. This module attempts the specialized libraries first, then falls
back to deterministic spectral/PCA routines so the repository remains runnable
on a fresh workstation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .preprocess import FEATURE_COLUMNS, add_topological_parameters, prepare_feature_matrix


def pca_embedding(matrix: np.ndarray, n_components: int = 3) -> np.ndarray:
    """Deterministic PCA via SVD."""

    centered = matrix - matrix.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(centered, full_matrices=False)
    coords = u[:, :n_components] * s[:n_components]
    if coords.shape[1] < n_components:
        coords = np.pad(coords, ((0, 0), (0, n_components - coords.shape[1])))
    return coords


def diffusion_map_embedding(matrix: np.ndarray, n_components: int = 3, epsilon: float | None = None) -> np.ndarray:
    """Compute a diffusion map or a spectral fallback.

    A diffusion map approximates the eigenfunctions of a heat kernel on the
    empirical manifold. The returned coordinates are

        psi_j(x) lambda_j

    for the leading non-trivial eigenvectors.
    """

    if len(matrix) > 3500:
        return pca_embedding(matrix, n_components=n_components)
    try:
        from pydiffmap import diffusion_map as dm

        model = dm.DiffusionMap.from_sklearn(n_evecs=n_components + 1, epsilon=epsilon or "bgh")
        coords = model.fit_transform(matrix)
        return np.asarray(coords[:, 1 : n_components + 1], dtype=float)
    except Exception:
        diff = matrix[:, None, :] - matrix[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        if epsilon is None:
            positive = dist2[dist2 > 0]
            epsilon = float(np.median(positive)) if positive.size else 1.0
        kernel = np.exp(-dist2 / max(epsilon, 1e-9))
        degree = kernel.sum(axis=1, keepdims=True)
        markov = kernel / np.maximum(degree, 1e-12)
        values, vectors = np.linalg.eig(markov)
        order = np.argsort(-np.real(values))
        values = np.real(values[order])
        vectors = np.real(vectors[:, order])
        coords = vectors[:, 1 : n_components + 1] * values[1 : n_components + 1]
        if coords.shape[1] < n_components:
            coords = np.pad(coords, ((0, 0), (0, n_components - coords.shape[1])))
        return coords


def umap_embedding(
    matrix: np.ndarray,
    labels: np.ndarray | None = None,
    n_components: int = 3,
    random_state: int = 918,
) -> np.ndarray:
    """Compute supervised UMAP when available, otherwise PCA."""

    if len(matrix) > 6000:
        return pca_embedding(matrix, n_components=n_components)
    try:
        import umap

        model = umap.UMAP(
            n_components=n_components,
            n_neighbors=24,
            min_dist=0.08,
            metric="euclidean",
            random_state=random_state,
        )
        return np.asarray(model.fit_transform(matrix, y=labels), dtype=float)
    except Exception:
        return pca_embedding(matrix, n_components=n_components)


def takens_embedding(series: np.ndarray, delay: int = 2, dimension: int = 3) -> np.ndarray:
    """Create a Takens time-delay embedding for a scalar trajectory.

    For a price path p_t, the embedding is

        T_t = (p_t, p_{t + tau}, ..., p_{t + (m - 1) tau}).
    """

    arr = np.asarray(series, dtype=float)
    window = (dimension - 1) * delay + 1
    if len(arr) < window:
        return np.empty((0, dimension))
    rows = []
    for start in range(len(arr) - window + 1):
        rows.append([arr[start + j * delay] for j in range(dimension)])
    return np.asarray(rows, dtype=float)


def compute_embeddings(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Attach diffusion and UMAP coordinates to the property manifold."""

    enriched = add_topological_parameters(df) if "affordability_index" not in df.columns else df.copy()
    matrix, enriched = prepare_feature_matrix(enriched, feature_columns=FEATURE_COLUMNS)
    labels = (enriched["rent_vs_buy"].to_numpy() == "buy").astype(int) if "rent_vs_buy" in enriched else None

    diff = diffusion_map_embedding(matrix, n_components=3)
    umap3 = umap_embedding(matrix, labels=labels, n_components=3)
    for idx, axis in enumerate(["x", "y", "z"]):
        enriched[f"diffusion_{axis}"] = diff[:, idx]
        enriched[f"umap_{axis}"] = umap3[:, idx]

    attractors: dict[str, np.ndarray] = {}
    for zip_code, group in enriched.groupby("zip_code"):
        trajectory = group.groupby("date")["median_listing_price"].median().sort_index().to_numpy()
        embedded = takens_embedding(trajectory, delay=2, dimension=3)
        attractors[str(zip_code)] = embedded
    return enriched, attractors
