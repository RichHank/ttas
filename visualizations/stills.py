"""Generate cinematic still frames for the TTAS portfolio."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.embeddings import compute_embeddings
from data.fetch_data import load_or_create_dataset
from data.preprocess import add_topological_parameters
from decision.opportunity_mapper import build_opportunity_graph
from decision.phase_transition import predict_regime_with_gp, train_gp_regime_classifier
from decision.topological_boundary import compute_topological_boundary
from topology.invariants import compute_time_slice_invariants
from topology.multiparameter import compute_multiparameter_persistence
from topology.silhouettes import compute_silhouette_suite
from topology.vineyards import compute_sliding_window_vineyards, vineyard_tracks
from .plots import (
    make_boundary_fig,
    make_euler_surface_fig,
    make_gp_regime_fig,
    make_mapper_fig,
    make_multiparameter_fig,
    make_silhouette_fig,
    make_spacetime_fig,
    make_vineyard_fig,
)


def _write(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.write_image(str(path), scale=2)
    except Exception:
        html_path = path.with_suffix(".html")
        fig.write_html(str(html_path), include_plotlyjs="cdn")


def generate_portfolio_stills(output_dir: str | Path = "outputs/figures") -> list[Path]:
    """Write still images or HTML fallbacks for README-ready artifacts."""

    output = Path(output_dir)
    df = add_topological_parameters(load_or_create_dataset())
    embedded, _ = compute_embeddings(df)
    latest = embedded["date"].max()
    invariants = compute_time_slice_invariants(embedded, date=latest, max_points=260, grid_size=12)
    sequence, diagrams = compute_sliding_window_vineyards(embedded, window_months=12, max_points=220)
    tracks = vineyard_tracks(diagrams)
    mapper = build_opportunity_graph(embedded[embedded["date"] == latest], n_bins=6)
    multiparameter = compute_multiparameter_persistence(invariants["filtration"], grid_size=12)
    silhouettes = compute_silhouette_suite(embedded, current_date=latest, max_points=220)
    boundary = compute_topological_boundary(embedded, date=latest, max_points=140)
    gp_model, gp_training = train_gp_regime_classifier(embedded, max_slices=12, max_points=140, grid_size=7)
    gp_prediction = predict_regime_with_gp(gp_model, invariants["euler_surface"])

    specs = [
        ("topological_heart_of_tulsa.png", make_euler_surface_fig(invariants["euler_frame"])),
        ("bottleneck_fingerprint_of_a_bubble.png", make_vineyard_fig(tracks, sequence)),
        ("affordability_black_hole.png", make_mapper_fig(mapper)),
        ("spacetime_manifold.png", make_spacetime_fig(embedded, date=latest)),
        ("multiparameter_persistence_lab.png", make_multiparameter_fig(multiparameter)),
        ("silhouettes_and_betti_curves.png", make_silhouette_fig(silhouettes)),
        ("topological_decision_boundary.png", make_boundary_fig(boundary)),
        ("gp_regime_classifier.png", make_gp_regime_fig(gp_training, gp_prediction)),
    ]
    written = []
    for filename, fig in specs:
        path = output / filename
        _write(fig, path)
        written.append(path if path.exists() else path.with_suffix(".html"))
    return written


if __name__ == "__main__":
    for artifact in generate_portfolio_stills():
        print(artifact)
