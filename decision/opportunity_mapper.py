"""KeplerMapper-style opportunity graph for the 12D housing manifold."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from data.preprocess import FEATURE_COLUMNS, add_topological_parameters, minmax_scale


def _lens(df: pd.DataFrame) -> np.ndarray:
    return df[["affordability_index", "opportunity_score"]].to_numpy(dtype=float)


def build_opportunity_graph(
    df: pd.DataFrame,
    n_bins: int = 6,
    overlap: float = 0.20,
    min_cluster_size: int = 5,
) -> dict[str, object]:
    """Build a Mapper graph from affordability and opportunity lenses."""

    enriched = add_topological_parameters(df) if "affordability_index" not in df.columns else df.copy()
    try:
        import kmapper as km
        from sklearn.cluster import DBSCAN

        mapper = km.KeplerMapper(verbose=0)
        data = minmax_scale(enriched[FEATURE_COLUMNS].to_numpy(dtype=float))
        lens = _lens(enriched)
        graph = mapper.map(
            lens,
            data,
            cover=km.Cover(n_cubes=n_bins, perc_overlap=overlap),
            clusterer=DBSCAN(eps=0.22, min_samples=min_cluster_size),
        )
        return {"backend": "kmapper", "graph": graph, "frame": mapper_graph_frame(graph, enriched)}
    except Exception:
        return _fallback_mapper(enriched, n_bins=n_bins, min_cluster_size=min_cluster_size)


def _fallback_mapper(df: pd.DataFrame, n_bins: int, min_cluster_size: int) -> dict[str, object]:
    lens = _lens(df)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    assignments: dict[str, list[int]] = {}
    for idx, (affordability, opportunity) in enumerate(lens):
        a_bin = min(np.searchsorted(bins, affordability, side="right") - 1, n_bins - 1)
        o_bin = min(np.searchsorted(bins, opportunity, side="right") - 1, n_bins - 1)
        key = f"A{a_bin}-O{o_bin}"
        assignments.setdefault(key, []).append(idx)

    nodes = {key: values for key, values in assignments.items() if len(values) >= min_cluster_size}
    links: dict[str, list[str]] = {key: [] for key in nodes}
    for left, right in itertools.combinations(nodes.keys(), 2):
        la, lo = [int(part[1:]) for part in left.split("-")]
        ra, ro = [int(part[1:]) for part in right.split("-")]
        if abs(la - ra) + abs(lo - ro) == 1:
            links[left].append(right)
            links[right].append(left)
    graph = {"nodes": nodes, "links": links}
    return {"backend": "fallback-grid", "graph": graph, "frame": mapper_graph_frame(graph, df)}


def mapper_graph_frame(graph: dict[str, object], df: pd.DataFrame) -> pd.DataFrame:
    """Summarize Mapper nodes for plotting and interaction."""

    nodes = graph.get("nodes", {})
    records = []
    for node_id, indices in nodes.items():
        subset = df.iloc[list(indices)]
        records.append(
            {
                "node_id": str(node_id),
                "size": int(len(subset)),
                "affordability_index": float(subset["affordability_index"].mean()),
                "opportunity_score": float(subset["opportunity_score"].mean()),
                "hilbert_value": float(len(subset)),
                "signed_barcode_length": float(subset["persistent_entropy_proxy"].mean()),
                "median_listing_price": float(subset["median_listing_price"].median()),
                "neighborhoods": ", ".join(sorted(subset["neighborhood"].astype(str).unique())[:4]),
                "property_ids": subset["property_id"].head(20).tolist() if "property_id" in subset else [],
            }
        )
    return pd.DataFrame(records)
