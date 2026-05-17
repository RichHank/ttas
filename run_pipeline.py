"""Run the TTAS data, topology, counterfactual, and decision pipeline locally."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data.embeddings import compute_embeddings
from data.fetch_data import TulsaDataConfig, load_or_create_dataset
from data.preprocess import add_topological_parameters
from decision.opportunity_mapper import build_opportunity_graph
from decision.path_integral import UserBiography, buy_signal
from decision.phase_transition import euler_curvature_alert, predict_regime_with_gp, train_gp_regime_classifier
from decision.topological_boundary import compute_topological_boundary
from topology.causal_tda import topological_ate, transfer_entropy
from topology.invariants import compute_time_slice_invariants
from topology.multiparameter import compute_multiparameter_persistence
from topology.silhouettes import compute_silhouette_suite
from topology.vineyards import bayesian_blocks_change_points, compute_sliding_window_vineyards, vineyard_tracks


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if hasattr(value, "__dict__"):
        return value.__dict__
    return str(value)


def run(args: argparse.Namespace) -> dict[str, object]:
    root = Path(__file__).resolve().parent
    cache = root / "outputs" / "cache"
    figures = root / "outputs" / "figures"
    cache.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    config = TulsaDataConfig(
        properties_per_zip_month=args.properties_per_zip_month,
        cache_dir=cache,
        use_public_data=args.use_public_data,
        use_osmnx=args.use_osmnx,
    )
    raw = load_or_create_dataset(config=config, refresh=args.refresh)
    enriched = add_topological_parameters(raw)
    embedded, attractors = compute_embeddings(enriched)
    embedded_path = cache / "tulsa_embedded.csv"
    embedded.to_csv(embedded_path, index=False)
    np.savez(cache / "takens_attractors.npz", **attractors)

    latest = pd.Timestamp(embedded["date"].max())
    invariants = compute_time_slice_invariants(embedded, date=latest, max_points=args.max_points, grid_size=args.grid_size)
    invariants["euler_frame"].to_csv(cache / "euler_surface_latest.csv", index=False)
    invariants["signed_barcodes"].to_csv(cache / "signed_barcodes_latest.csv", index=False)
    invariants["rank_h0"].to_csv(cache / "rank_invariant_h0_latest.csv", index=False)
    invariants["rank_h1"].to_csv(cache / "rank_invariant_h1_latest.csv", index=False)
    multiparameter = compute_multiparameter_persistence(invariants["filtration"], grid_size=args.grid_size)
    multiparameter.signed_measure.to_csv(cache / "multiparameter_signed_measure.csv", index=False)
    multiparameter.hilbert_frame.to_csv(cache / "multiparameter_hilbert_function.csv", index=False)
    multiparameter.rank_frame.to_csv(cache / "multiparameter_rank_invariant.csv", index=False)

    sequence, diagrams = compute_sliding_window_vineyards(
        embedded,
        window_months=args.window_months,
        stride_months=1,
        max_points=args.max_points,
    )
    tracks = vineyard_tracks(diagrams)
    changes = bayesian_blocks_change_points(sequence)
    sequence.to_csv(cache / "vineyard_sequence.csv", index=False)
    tracks.to_csv(cache / "vineyard_tracks.csv", index=False)
    changes.to_csv(cache / "topological_change_points.csv", index=False)

    mapper = build_opportunity_graph(embedded[embedded["date"] == latest], n_bins=6)
    mapper["frame"].to_csv(cache / "opportunity_mapper_nodes.csv", index=False)

    silhouettes = compute_silhouette_suite(embedded, current_date=latest, max_points=args.max_points)
    silhouettes["frame"].to_csv(cache / "silhouettes_betti_curves.csv", index=False)
    boundary = compute_topological_boundary(embedded, date=latest, family_size=args.family_size, max_points=max(80, args.max_points // 2))
    boundary.to_csv(cache / "topological_decision_boundary.csv", index=False)
    gp_model, gp_training = train_gp_regime_classifier(
        embedded,
        max_slices=args.gp_slices,
        max_points=max(80, args.max_points // 2),
        grid_size=max(5, min(args.grid_size, 8)),
    )
    gp_training.to_csv(cache / "gp_regime_training.csv", index=False)

    ate = topological_ate(embedded, shock_bps=args.shock_bps, date=latest, max_points=args.max_points)
    signal = buy_signal(
        embedded,
        UserBiography(annual_income=args.income, dti_max=args.dti, family_size=args.family_size),
        date=latest,
        max_points=args.max_points,
    )
    alert = euler_curvature_alert(invariants["euler_surface"])
    regime_prediction = predict_regime_with_gp(gp_model, invariants["euler_surface"])
    regime = regime_prediction["regime"]

    rent_series = embedded.groupby("date")["rent_to_price_ratio"].median().to_numpy(dtype=float)
    buy_series = embedded.groupby("date")["affordability_index"].median().to_numpy(dtype=float)
    te_rent_to_buy = transfer_entropy(rent_series, buy_series)
    te_buy_to_rent = transfer_entropy(buy_series, rent_series)

    summary = {
        "rows": int(len(embedded)),
        "latest_month": latest,
        "embedded_path": str(embedded_path),
        "filtration_backend": invariants["filtration"].backend,
        "vertices": invariants["filtration"].metadata["n_vertices"],
        "edges": invariants["filtration"].metadata["n_edges"],
        "triangles": invariants["filtration"].metadata["n_triangles"],
        "persistent_entropy_h0": invariants["persistent_entropy_h0"],
        "persistent_entropy_h1": invariants["persistent_entropy_h1"],
        "multiparameter_backend": multiparameter.backend,
        "multiparameter_exact": multiparameter.exact,
        "multiparameter_error": multiparameter.error,
        "regime": regime,
        "regime_backend": regime_prediction["backend"],
        "phase_alert": {key: value for key, value in alert.items() if key != "curvature"},
        "change_points": len(changes),
        "topological_ate_h0": ate["topological_ate_h0"],
        "topological_ate_h1": ate["topological_ate_h1"],
        "transfer_entropy_rent_to_buy": te_rent_to_buy,
        "transfer_entropy_buy_to_rent": te_buy_to_rent,
        "buy_signal": {
            "S_B": signal["S_B"],
            "normalized_signal": signal["normalized_signal"],
            "decision": signal["decision"],
            "restricted_count": signal["restricted_count"],
        },
        "boundary_points": int(boundary["boundary"].sum()) if not boundary.empty else 0,
    }
    with (cache / "pipeline_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=_json_default)

    if args.write_html:
        from visualizations.plots import (
            make_causal_fig,
            make_boundary_fig,
            make_decision_fig,
            make_euler_surface_fig,
            make_gp_regime_fig,
            make_mapper_fig,
            make_multiparameter_fig,
            make_silhouette_fig,
            make_spacetime_fig,
            make_vineyard_fig,
        )

        make_spacetime_fig(embedded, latest).write_html(figures / "spacetime_manifold.html", include_plotlyjs="cdn")
        make_euler_surface_fig(invariants["euler_frame"]).write_html(figures / "topological_heart_of_tulsa.html", include_plotlyjs="cdn")
        make_vineyard_fig(tracks, sequence).write_html(figures / "bottleneck_fingerprint_of_a_bubble.html", include_plotlyjs="cdn")
        make_mapper_fig(mapper).write_html(figures / "affordability_black_hole.html", include_plotlyjs="cdn")
        make_causal_fig(ate).write_html(figures / "causal_shock_lab.html", include_plotlyjs="cdn")
        make_decision_fig(signal).write_html(figures / "decision_boundary_navigator.html", include_plotlyjs="cdn")
        make_multiparameter_fig(multiparameter).write_html(figures / "multiparameter_persistence_lab.html", include_plotlyjs="cdn")
        make_silhouette_fig(silhouettes).write_html(figures / "silhouettes_and_betti_curves.html", include_plotlyjs="cdn")
        make_boundary_fig(boundary).write_html(figures / "topological_decision_boundary.html", include_plotlyjs="cdn")
        make_gp_regime_fig(gp_training, regime_prediction).write_html(figures / "gp_regime_classifier.html", include_plotlyjs="cdn")

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the TTAS local research pipeline.")
    parser.add_argument("--refresh", action="store_true", help="Regenerate the cached synthetic manifold.")
    parser.add_argument("--properties-per-zip-month", type=int, default=10)
    parser.add_argument("--max-points", type=int, default=260)
    parser.add_argument("--grid-size", type=int, default=12)
    parser.add_argument("--window-months", type=int, default=12)
    parser.add_argument("--shock-bps", type=float, default=-100.0)
    parser.add_argument("--income", type=float, default=92_000.0)
    parser.add_argument("--dti", type=float, default=0.38)
    parser.add_argument("--family-size", type=int, default=3)
    parser.add_argument("--gp-slices", type=int, default=12)
    parser.add_argument("--use-public-data", action="store_true", help="Apply available public-data enrichments.")
    parser.add_argument("--use-osmnx", action="store_true", help="Use OSMnx spatial enrichment when --use-public-data is set.")
    parser.add_argument("--write-html", action="store_true", help="Write Plotly HTML artifacts.")
    parser.add_argument("--validate", action="store_true", help="Run model validation and cache results.")
    parser.add_argument("--lineage", action="store_true", help="Write data lineage metadata to cache.")
    parser.add_argument("--report", action="store_true", help="Generate a self-contained HTML report.")
    parser.add_argument("--full", action="store_true", help="Run the full pipeline: data, topology, validate, lineage, report.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run(args)
    print(json.dumps(result, indent=2, default=_json_default))

    # Post-pipeline steps
    root = Path(__file__).resolve().parent
    cache = root / "outputs" / "cache"

    if args.validate or args.full:
        try:
            from decision.validation import build_validation_dashboard_data
            from data.fetch_data import load_or_create_dataset
            from data.preprocess import add_topological_parameters
            df = add_topological_parameters(load_or_create_dataset())
            val = build_validation_dashboard_data(df)
            with (cache / "validation.json").open("w", encoding="utf-8") as fh:
                json.dump(val, fh, indent=2, default=_json_default)
            print(f"Validation cached to {cache / 'validation.json'}")
        except Exception as exc:
            print(f"Validation failed: {exc}")

    if args.lineage or args.full:
        try:
            from data.lineage import build_lineage_json
            build_lineage_json(cache / "lineage.json")
            print(f"Lineage cached to {cache / 'lineage.json'}")
        except Exception as exc:
            print(f"Lineage failed: {exc}")

    if args.report or args.full:
        try:
            from scripts.generate_report import build_report
            path = build_report()
            print(f"Report written to {path}")
        except Exception as exc:
            print(f"Report failed: {exc}")
