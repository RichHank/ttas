"""Persistent and multiparameter invariants for TTAS.

The module exposes a practical invariant suite:

* one-parameter persistence diagrams, using `ripser` when present;
* a pure-Python H0/H1 approximation for portability;
* signed barcode summaries and Hilbert/rank invariant grids;
* Euler characteristic surfaces chi(lambda_1, lambda_2, lambda_3).

The Euler characteristic is computed from the finite skeleton as

    chi = beta_0 - beta_1 + beta_2 approx |V| - |E| + |T|.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.preprocess import add_topological_parameters, monthly_sample
from .filtrations import FiltrationResult, build_tri_parameter_filtration


class UnionFind:
    """Tiny disjoint-set data structure for H0 persistence."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> bool:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return False
        if self.rank[root_left] < self.rank[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        if self.rank[root_left] == self.rank[root_right]:
            self.rank[root_left] += 1
        return True


def _pairwise_distances(points: np.ndarray) -> np.ndarray:
    diff = points[:, None, :] - points[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def h0_persistence(points: np.ndarray) -> np.ndarray:
    """Compute H0 persistence via Kruskal's algorithm."""

    n = len(points)
    if n <= 1:
        return np.empty((0, 2), dtype=float)
    distances = _pairwise_distances(points)
    pairs = np.argwhere(np.triu(np.ones((n, n), dtype=bool), k=1))
    weights = distances[pairs[:, 0], pairs[:, 1]]
    order = np.argsort(weights)
    uf = UnionFind(n)
    intervals: list[tuple[float, float]] = []
    for index in order:
        i, j = int(pairs[index, 0]), int(pairs[index, 1])
        if uf.union(i, j):
            intervals.append((0.0, float(weights[index])))
            if len(intervals) == n - 1:
                break
    return np.asarray(intervals, dtype=float)


def h1_cycle_approximation(points: np.ndarray, max_cycles: int = 120) -> np.ndarray:
    """Approximate H1 intervals from non-tree edges.

    This is not a replacement for exact persistent homology. It is a transparent
    fallback: non-MST edges create candidate cycles, and the nearest triangle
    scale provides a finite death time. The approximation preserves the
    "bubble fingerprint" intuition used by the dashboard.
    """

    n = len(points)
    if n < 4:
        return np.empty((0, 2), dtype=float)
    distances = _pairwise_distances(points)
    pairs = np.argwhere(np.triu(np.ones((n, n), dtype=bool), k=1))
    weights = distances[pairs[:, 0], pairs[:, 1]]
    order = np.argsort(weights)
    uf = UnionFind(n)
    mst_edges: set[tuple[int, int]] = set()
    for index in order:
        i, j = int(pairs[index, 0]), int(pairs[index, 1])
        if uf.union(i, j):
            mst_edges.add((min(i, j), max(i, j)))

    intervals: list[tuple[float, float]] = []
    for index in order:
        i, j = int(pairs[index, 0]), int(pairs[index, 1])
        edge = (min(i, j), max(i, j))
        if edge in mst_edges:
            continue
        birth = float(weights[index])
        common_scale = np.minimum(distances[i], distances[j])
        death = float(max(birth, np.partition(common_scale, min(3, n - 1))[min(3, n - 1)]))
        if death > birth * 1.02:
            intervals.append((birth, death))
        if len(intervals) >= max_cycles:
            break
    return np.asarray(intervals, dtype=float)


def persistence_diagrams(points: np.ndarray, maxdim: int = 1) -> dict[str, np.ndarray]:
    """Compute persistence diagrams with ripser or local fallback."""

    try:
        from ripser import ripser

        diagrams = ripser(points, maxdim=maxdim)["dgms"]
        result = {"H0": np.asarray(diagrams[0], dtype=float)}
        if maxdim >= 1 and len(diagrams) > 1:
            result["H1"] = np.asarray(diagrams[1], dtype=float)
        else:
            result["H1"] = np.empty((0, 2), dtype=float)
        return result
    except Exception:
        return {
            "H0": h0_persistence(points),
            "H1": h1_cycle_approximation(points) if maxdim >= 1 else np.empty((0, 2), dtype=float),
        }


def finite_intervals(diagram: np.ndarray) -> np.ndarray:
    if diagram.size == 0:
        return np.empty((0, 2), dtype=float)
    arr = np.asarray(diagram, dtype=float)
    return arr[np.isfinite(arr).all(axis=1) & (arr[:, 1] > arr[:, 0])]


def persistence_entropy(diagram: np.ndarray) -> float:
    """Persistent entropy H = -sum p_i log(p_i)."""

    finite = finite_intervals(diagram)
    if finite.size == 0:
        return 0.0
    lengths = finite[:, 1] - finite[:, 0]
    total = lengths.sum()
    if total <= 0:
        return 0.0
    p = lengths / total
    return float(-np.sum(p * np.log(p + 1e-12)) / np.log(len(p) + 1e-12))


def persistence_landscape(diagram: np.ndarray, xs: np.ndarray, layers: int = 5) -> np.ndarray:
    """Evaluate a persistence landscape on a grid."""

    finite = finite_intervals(diagram)
    if finite.size == 0:
        return np.zeros((layers, len(xs)), dtype=float)
    tents = []
    for birth, death in finite:
        mid = 0.5 * (birth + death)
        height = 0.5 * (death - birth)
        tents.append(np.maximum(0.0, height - np.abs(xs - mid)))
    values = np.vstack(tents)
    values.sort(axis=0)
    top = values[::-1][:layers]
    if top.shape[0] < layers:
        top = np.pad(top, ((0, layers - top.shape[0]), (0, 0)))
    return top


def euler_characteristic_surface(filtration: FiltrationResult) -> tuple[np.ndarray, pd.DataFrame]:
    """Compute chi(lambda_1, lambda_2, lambda_3) on the filtration grid."""

    axes = filtration.grid_axes
    chi = np.zeros((len(axes[0]), len(axes[1]), len(axes[2])), dtype=float)
    records: list[dict[str, float]] = []
    vertex_births = filtration.lambda_points
    edge_births = filtration.edge_births
    triangle_births = filtration.triangle_births

    for i, a in enumerate(axes[0]):
        for j, b in enumerate(axes[1]):
            for k, c in enumerate(axes[2]):
                threshold = np.array([a, b, c], dtype=float)
                vertices = int(np.sum(np.all(vertex_births <= threshold, axis=1)))
                edges = int(np.sum(np.all(edge_births <= threshold, axis=1))) if len(edge_births) else 0
                triangles = int(np.sum(np.all(triangle_births <= threshold, axis=1))) if len(triangle_births) else 0
                value = float(vertices - edges + triangles)
                chi[i, j, k] = value
                records.append(
                    {
                        "lambda_1": float(a),
                        "lambda_2": float(b),
                        "lambda_3": float(c),
                        "vertices": vertices,
                        "edges": edges,
                        "triangles": triangles,
                        "chi": value,
                    }
                )
    return chi, pd.DataFrame(records)


def signed_barcode_summary(filtration: FiltrationResult) -> pd.DataFrame:
    """Summarize signed barcode mass from simplex births.

    The signed barcode is approximated as positive vertex mass, negative edge
    mass, and positive triangle mass over lambda-space. Exact signed barcodes
    can be supplied by `multipers.signed_barcodes` in a research environment.
    """

    records = []
    for sign, name, births in [
        (1, "vertex", filtration.lambda_points),
        (-1, "edge", filtration.edge_births),
        (1, "triangle", filtration.triangle_births),
    ]:
        if births.size == 0:
            continue
        lengths = np.linalg.norm(1.0 - births, axis=1)
        for value in lengths:
            records.append({"simplex_type": name, "sign": sign, "barcode_length": float(value), "signed_mass": float(sign * value)})
    return pd.DataFrame(records)


def rank_invariant_grid(diagram: np.ndarray, grid: np.ndarray | None = None) -> pd.DataFrame:
    """Evaluate a one-parameter rank invariant on a birth/death grid."""

    finite = finite_intervals(diagram)
    grid = grid if grid is not None else np.linspace(0.0, 1.0, 20)
    records = []
    for birth_threshold in grid:
        for death_threshold in grid:
            rank = int(np.sum((finite[:, 0] <= birth_threshold) & (finite[:, 1] >= death_threshold))) if finite.size else 0
            records.append({"birth_threshold": float(birth_threshold), "death_threshold": float(death_threshold), "rank": rank})
    return pd.DataFrame(records)


def compute_time_slice_invariants(
    df: pd.DataFrame,
    date: str | pd.Timestamp | None = None,
    max_points: int = 360,
    grid_size: int = 12,
) -> dict[str, object]:
    """Compute the complete local invariant suite for a monthly slice."""

    enriched = add_topological_parameters(df) if "affordability_index" not in df.columns else df
    sample = monthly_sample(enriched, date=date, max_points=max_points)
    filtration = build_tri_parameter_filtration(sample, max_points=max_points, grid_size=grid_size)
    diagrams = persistence_diagrams(filtration.feature_points, maxdim=1)
    chi, chi_frame = euler_characteristic_surface(filtration)
    signed = signed_barcode_summary(filtration)
    rank_h0 = rank_invariant_grid(diagrams["H0"])
    rank_h1 = rank_invariant_grid(diagrams["H1"])
    return {
        "date": pd.Timestamp(sample["date"].iloc[0]) if len(sample) else pd.NaT,
        "filtration": filtration,
        "diagrams": diagrams,
        "euler_surface": chi,
        "euler_frame": chi_frame,
        "signed_barcodes": signed,
        "rank_h0": rank_h0,
        "rank_h1": rank_h1,
        "persistent_entropy_h0": persistence_entropy(diagrams["H0"]),
        "persistent_entropy_h1": persistence_entropy(diagrams["H1"]),
    }
