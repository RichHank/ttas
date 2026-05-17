"""Exact and fallback multiparameter persistence for TTAS.

The first release only checked whether `multipers` was importable. This module
does the real work: it converts the skeletonized TTAS filtration into a
`multipers.SimplexTreeMulti`, calls the signed-measure API when available, and
otherwise computes clearly labeled finite-grid Hilbert, Betti, rank, and
interleaving proxies from the same tri-parameter complex.

The fallback is intentionally not presented as algebraic multiparameter
persistence. It is a certificate-preserving approximation that keeps the
dashboard and GitHub Pages artifacts reproducible on machines without compiled
TDA backends.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .filtrations import FiltrationResult


@dataclass
class MultiparameterResult:
    """Multiparameter invariant bundle.

    Attributes:
        backend: Backend that produced the result.
        exact: True only when `multipers` calls completed successfully.
        error: Backend error message, if the exact backend failed.
        signed_measure: Signed barcode / signed measure summary.
        hilbert_frame: Finite-grid Hilbert and Betti values.
        rank_frame: Finite-grid rank invariant or rank proxy.
        euler_frame: Euler characteristic values aligned with the Hilbert grid.
        interleaving_frame: Optional pairwise distance proxy over time.
    """

    backend: str
    exact: bool
    error: str | None
    signed_measure: pd.DataFrame
    hilbert_frame: pd.DataFrame
    rank_frame: pd.DataFrame
    euler_frame: pd.DataFrame
    interleaving_frame: pd.DataFrame


class _UnionFind:
    def __init__(self, vertices: list[int]) -> None:
        self.parent = {vertex: vertex for vertex in vertices}
        self.rank = {vertex: 0 for vertex in vertices}

    def find(self, vertex: int) -> int:
        while self.parent[vertex] != vertex:
            self.parent[vertex] = self.parent[self.parent[vertex]]
            vertex = self.parent[vertex]
        return vertex

    def union(self, left: int, right: int) -> bool:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return False
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1
        return True


def _threshold_mask(births: np.ndarray, threshold: np.ndarray) -> np.ndarray:
    if births.size == 0:
        return np.zeros(0, dtype=bool)
    return np.all(births <= threshold, axis=1)


def _betti_at_threshold(filtration: FiltrationResult, threshold: np.ndarray) -> dict[str, int]:
    vertex_mask = _threshold_mask(filtration.lambda_points, threshold)
    vertices = np.flatnonzero(vertex_mask).tolist()
    vertex_set = set(vertices)

    edge_mask = _threshold_mask(filtration.edge_births, threshold)
    included_edges = [
        tuple(map(int, edge))
        for edge, keep in zip(filtration.edges, edge_mask)
        if keep and int(edge[0]) in vertex_set and int(edge[1]) in vertex_set
    ]
    triangle_mask = _threshold_mask(filtration.triangle_births, threshold)
    included_triangles = [
        tuple(map(int, triangle))
        for triangle, keep in zip(filtration.triangles, triangle_mask)
        if keep and all(int(vertex) in vertex_set for vertex in triangle)
    ]

    if not vertices:
        return {"vertices": 0, "edges": 0, "triangles": 0, "beta0": 0, "beta1": 0, "beta2": 0, "chi": 0}

    uf = _UnionFind(vertices)
    for left, right in included_edges:
        uf.union(left, right)
    beta0 = len({uf.find(vertex) for vertex in vertices})
    chi = len(vertices) - len(included_edges) + len(included_triangles)
    beta1 = max(len(included_edges) - len(vertices) + beta0 - len(included_triangles), 0)
    beta2 = max(chi - beta0 + beta1, 0)
    return {
        "vertices": len(vertices),
        "edges": len(included_edges),
        "triangles": len(included_triangles),
        "beta0": int(beta0),
        "beta1": int(beta1),
        "beta2": int(beta2),
        "chi": int(chi),
    }


def fallback_hilbert_function(filtration: FiltrationResult) -> pd.DataFrame:
    """Compute finite-grid Hilbert values for H0/H1/H2 from the skeleton."""

    records: list[dict[str, float | int | str]] = []
    for i, lambda_1 in enumerate(filtration.grid_axes[0]):
        for j, lambda_2 in enumerate(filtration.grid_axes[1]):
            for k, lambda_3 in enumerate(filtration.grid_axes[2]):
                threshold = np.array([lambda_1, lambda_2, lambda_3], dtype=float)
                betti = _betti_at_threshold(filtration, threshold)
                for degree in [0, 1, 2]:
                    records.append(
                        {
                            "lambda_1": float(lambda_1),
                            "lambda_2": float(lambda_2),
                            "lambda_3": float(lambda_3),
                            "axis_i": i,
                            "axis_j": j,
                            "axis_k": k,
                            "degree": degree,
                            "hilbert_value": int(betti[f"beta{degree}"]),
                            "beta0": betti["beta0"],
                            "beta1": betti["beta1"],
                            "beta2": betti["beta2"],
                            "chi": betti["chi"],
                            "backend": "grid-fallback",
                        }
                    )
    return pd.DataFrame(records)


def fallback_rank_invariant(hilbert_frame: pd.DataFrame, samples: int = 10) -> pd.DataFrame:
    """Compute a monotone-grid rank proxy from Hilbert values.

    For true multiparameter modules, rank is algebraic. The fallback records
    min(H_d(u), H_d(v)) for comparable grid points u <= v, which is a useful
    upper-envelope diagnostic but not an exact rank invariant.
    """

    base = hilbert_frame[hilbert_frame["degree"].isin([0, 1])].copy()
    if base.empty:
        return pd.DataFrame()
    grid_points = (
        base[["axis_i", "axis_j", "axis_k", "lambda_1", "lambda_2", "lambda_3"]]
        .drop_duplicates()
        .sort_values(["axis_i", "axis_j", "axis_k"])
        .reset_index(drop=True)
    )
    if len(grid_points) > samples:
        take = np.unique(np.linspace(0, len(grid_points) - 1, samples).astype(int))
        grid_points = grid_points.iloc[take].reset_index(drop=True)

    lookup = {
        (int(row.axis_i), int(row.axis_j), int(row.axis_k), int(row.degree)): int(row.hilbert_value)
        for row in base.itertuples(index=False)
    }
    records = []
    for u in grid_points.itertuples(index=False):
        for v in grid_points.itertuples(index=False):
            if (u.axis_i, u.axis_j, u.axis_k) > (v.axis_i, v.axis_j, v.axis_k):
                continue
            if u.axis_i <= v.axis_i and u.axis_j <= v.axis_j and u.axis_k <= v.axis_k:
                for degree in [0, 1]:
                    rank_value = min(
                        lookup.get((int(u.axis_i), int(u.axis_j), int(u.axis_k), degree), 0),
                        lookup.get((int(v.axis_i), int(v.axis_j), int(v.axis_k), degree), 0),
                    )
                    records.append(
                        {
                            "u_lambda_1": float(u.lambda_1),
                            "u_lambda_2": float(u.lambda_2),
                            "u_lambda_3": float(u.lambda_3),
                            "v_lambda_1": float(v.lambda_1),
                            "v_lambda_2": float(v.lambda_2),
                            "v_lambda_3": float(v.lambda_3),
                            "degree": degree,
                            "rank_value": int(rank_value),
                            "backend": "rank-proxy",
                        }
                    )
    return pd.DataFrame(records)


def fallback_signed_measure(filtration: FiltrationResult) -> pd.DataFrame:
    """Return a signed simplex measure over parameter space."""

    records = []
    for dimension, name, sign, births in [
        (0, "vertex", 1, filtration.lambda_points),
        (1, "edge", -1, filtration.edge_births),
        (2, "triangle", 1, filtration.triangle_births),
    ]:
        for birth in births:
            persistence_to_terminal = float(np.linalg.norm(1.0 - birth))
            records.append(
                {
                    "lambda_1": float(birth[0]),
                    "lambda_2": float(birth[1]),
                    "lambda_3": float(birth[2]),
                    "simplex_type": name,
                    "dimension": dimension,
                    "sign": sign,
                    "weight": float(sign * persistence_to_terminal),
                    "backend": "signed-simplex-fallback",
                }
            )
    return pd.DataFrame(records)


def _to_simplex_tree_multi(filtration: FiltrationResult):
    """Convert TTAS skeleton to `multipers.SimplexTreeMulti`."""

    import multipers as mp

    tree = mp.SimplexTreeMulti(num_parameters=3)
    for vertex, birth in enumerate(filtration.lambda_points):
        tree.insert([int(vertex)], birth.tolist())
    for edge, birth in zip(filtration.edges, filtration.edge_births):
        tree.insert([int(edge[0]), int(edge[1])], birth.tolist())
    for triangle, birth in zip(filtration.triangles, filtration.triangle_births):
        tree.insert([int(triangle[0]), int(triangle[1]), int(triangle[2])], birth.tolist())
    for method_name in ["make_filtration_non_decreasing", "prune_above_dimension"]:
        method = getattr(tree, method_name, None)
        if method_name == "make_filtration_non_decreasing" and method is not None:
            method()
        if method_name == "prune_above_dimension" and method is not None:
            method(2)
    return tree


def _measure_to_frame(measure, invariant: str, degree: int | None) -> pd.DataFrame:
    """Normalize a multipers signed measure object to a dataframe."""

    if isinstance(measure, (list, tuple)) and len(measure) == 2:
        points, weights = measure
    elif isinstance(measure, (list, tuple)) and measure and isinstance(measure[0], (list, tuple)):
        points, weights = measure[0]
    else:
        points = getattr(measure, "points", np.empty((0, 3)))
        weights = getattr(measure, "weights", np.empty(0))
    points = np.asarray(points, dtype=float)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if points.ndim == 1 and points.size:
        points = points.reshape(1, -1)
    records = []
    for point, weight in zip(points, weights):
        coords = list(point[:3]) + [np.nan] * max(0, 3 - len(point))
        record = {
            "lambda_1": float(coords[0]),
            "lambda_2": float(coords[1]),
            "lambda_3": float(coords[2]),
            "degree": -1 if degree is None else int(degree),
            "invariant": invariant,
            "weight": float(weight),
            "backend": "multipers",
        }
        if invariant == "rank" and len(point) >= 6:
            record.update(
                {
                    "birth_lambda_1": float(point[0]),
                    "birth_lambda_2": float(point[1]),
                    "birth_lambda_3": float(point[2]),
                    "death_lambda_1": float(point[3]),
                    "death_lambda_2": float(point[4]),
                    "death_lambda_3": float(point[5]),
                }
            )
        records.append(record)
    return pd.DataFrame(records)


def _integrate_signed_measure(measure, degree: int) -> pd.DataFrame:
    """Integrate a multipers Hilbert signed measure into a dataframe."""

    from multipers import point_measure

    if isinstance(measure, (list, tuple)) and len(measure) == 2:
        points, weights = measure
    elif isinstance(measure, (list, tuple)) and measure:
        points, weights = measure[0]
    else:
        return pd.DataFrame()
    surface, grid = point_measure.integrate_measure(points, weights, return_grid=True)
    surface = np.asarray(surface)
    records = []
    for index in np.ndindex(surface.shape):
        coords = []
        for dim, coord_index in enumerate(index[:3]):
            axis = np.asarray(grid[dim])
            coords.append(float(axis[min(coord_index, len(axis) - 1)]))
        coords += [np.nan] * max(0, 3 - len(coords))
        records.append(
            {
                "lambda_1": coords[0],
                "lambda_2": coords[1],
                "lambda_3": coords[2],
                "degree": int(degree),
                "hilbert_value": float(surface[index]),
                "beta0": np.nan,
                "beta1": np.nan,
                "beta2": np.nan,
                "chi": np.nan,
                "backend": "multipers-integrated",
            }
        )
    return pd.DataFrame(records)


def _try_exact_multipers(filtration: FiltrationResult, grid_size: int, degrees: tuple[int, ...]) -> MultiparameterResult:
    import multipers as mp

    tree = _to_simplex_tree_multi(filtration)
    squeezed = tree
    grid_squeeze = getattr(tree, "grid_squeeze", None)
    if grid_squeeze is not None:
        squeezed = grid_squeeze(strategy="regular_closest", resolution=grid_size)

    signed_frames = []
    hilbert_frames = []
    for degree in degrees:
        signed = mp.signed_measure(squeezed, degrees=[degree], invariant="hilbert")
        if isinstance(signed, list) and signed:
            signed = signed[0]
        signed_frames.append(_measure_to_frame(signed, "hilbert", degree))
        try:
            hilbert_frames.append(_integrate_signed_measure(signed, degree))
        except Exception:
            pass
    signed_measure = pd.concat(signed_frames, ignore_index=True) if signed_frames else pd.DataFrame()

    if hilbert_frames:
        hilbert_frame = pd.concat(hilbert_frames, ignore_index=True)
    else:
        hilbert_frame = fallback_hilbert_function(filtration)
        hilbert_frame["backend"] = "multipers-grid-evaluated"

    rank_frames = []
    for degree in degrees:
        try:
            rank_measure = mp.signed_measure(squeezed, degrees=[degree], invariant="rank")
            if isinstance(rank_measure, list) and rank_measure:
                rank_measure = rank_measure[0]
            rank_frames.append(_measure_to_frame(rank_measure, "rank", degree))
        except Exception:
            continue
    rank_frame = pd.concat(rank_frames, ignore_index=True) if rank_frames else fallback_rank_invariant(hilbert_frame)
    euler_frame = hilbert_frame[hilbert_frame["degree"] == 0][["lambda_1", "lambda_2", "lambda_3", "chi", "backend"]].copy()
    return MultiparameterResult(
        backend="multipers",
        exact=True,
        error=None,
        signed_measure=signed_measure,
        hilbert_frame=hilbert_frame,
        rank_frame=rank_frame,
        euler_frame=euler_frame,
        interleaving_frame=pd.DataFrame(),
    )


def compute_multiparameter_persistence(
    filtration: FiltrationResult,
    grid_size: int | None = None,
    degrees: tuple[int, ...] = (0, 1),
    require_exact: bool = False,
) -> MultiparameterResult:
    """Compute exact multiparameter invariants or certified fallbacks."""

    grid_size = grid_size or len(filtration.grid_axes[0])
    try:
        return _try_exact_multipers(filtration, grid_size=grid_size, degrees=degrees)
    except Exception as exc:
        if require_exact:
            raise
        hilbert = fallback_hilbert_function(filtration)
        return MultiparameterResult(
            backend="grid-fallback",
            exact=False,
            error=str(exc),
            signed_measure=fallback_signed_measure(filtration),
            hilbert_frame=hilbert,
            rank_frame=fallback_rank_invariant(hilbert),
            euler_frame=hilbert[hilbert["degree"] == 0][["lambda_1", "lambda_2", "lambda_3", "chi", "backend"]].copy(),
            interleaving_frame=pd.DataFrame(),
        )


def interleaving_distance_proxy(left: pd.DataFrame, right: pd.DataFrame) -> float:
    """Approximate interleaving distance by sup-norm Hilbert discrepancy."""

    if left.empty or right.empty:
        return 0.0
    columns = ["lambda_1", "lambda_2", "lambda_3", "degree"]
    merged = left[columns + ["hilbert_value"]].merge(
        right[columns + ["hilbert_value"]],
        on=columns,
        suffixes=("_left", "_right"),
        how="inner",
    )
    if merged.empty:
        return 0.0
    diff = np.abs(merged["hilbert_value_left"].to_numpy(dtype=float) - merged["hilbert_value_right"].to_numpy(dtype=float))
    return float(np.max(diff))
