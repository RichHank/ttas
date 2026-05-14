"""Plotly figures for the TTAS dashboard and portfolio stills."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


TTAS_TEMPLATE = "plotly_dark"


def _layout(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        template=TTAS_TEMPLATE,
        title={"text": title, "x": 0.02, "xanchor": "left"},
        margin={"l": 28, "r": 24, "t": 56, "b": 28},
        paper_bgcolor="#09100f",
        plot_bgcolor="#09100f",
        font={"family": "Inter, Segoe UI, Arial, sans-serif", "color": "#f1f6f3"},
        coloraxis_colorbar={"title": ""},
    )
    return fig


def make_spacetime_fig(df: pd.DataFrame, date: str | pd.Timestamp | None = None) -> go.Figure:
    """3D UMAP scatter colored by persistent entropy proxy."""

    working = df.copy()
    if date is not None:
        working = working[working["date"] == pd.Timestamp(date)]
    if working.empty:
        working = df.copy()
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=working["umap_x"],
                y=working["umap_y"],
                z=working["umap_z"],
                mode="markers",
                marker={
                    "size": 4,
                    "color": working.get("persistent_entropy_proxy", pd.Series(np.zeros(len(working)))),
                    "colorscale": [[0, "#78dcca"], [0.5, "#f7c948"], [1, "#ff6b6b"]],
                    "opacity": 0.78,
                    "colorbar": {"title": "entropy"},
                },
                customdata=np.stack(
                    [
                        working["neighborhood"].astype(str),
                        working["median_listing_price"].round(0).astype(int),
                        working["rent_vs_buy"].astype(str),
                    ],
                    axis=1,
                ),
                hovertemplate="%{customdata[0]}<br>$%{customdata[1]:,}<br>%{customdata[2]}<extra></extra>",
            )
        ]
    )
    fig.update_scenes(
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2",
        zaxis_title="UMAP 3",
        bgcolor="#09100f",
    )
    return _layout(fig, "Spacetime Manifold")


def make_euler_surface_fig(euler_frame: pd.DataFrame) -> go.Figure:
    """Render chi(lambda_1, lambda_2, lambda_3) as an isosurface."""

    frame = euler_frame.copy()
    if frame.empty:
        return _layout(go.Figure(), "Euler Characteristic Surface")
    value = frame["chi"].to_numpy(dtype=float)
    surface_count = 4 if np.nanmax(value) > np.nanmin(value) else 1
    fig = go.Figure(
        data=[
            go.Isosurface(
                x=frame["lambda_1"],
                y=frame["lambda_2"],
                z=frame["lambda_3"],
                value=value,
                isomin=float(np.quantile(value, 0.18)),
                isomax=float(np.quantile(value, 0.92)),
                surface_count=surface_count,
                colorscale=[[0, "#355c7d"], [0.45, "#78dcca"], [0.75, "#f7c948"], [1, "#ff6b6b"]],
                opacity=0.52,
                caps={"x": {"show": False}, "y": {"show": False}, "z": {"show": False}},
                colorbar={"title": "chi"},
            )
        ]
    )
    fig.update_scenes(
        xaxis_title="lambda 1",
        yaxis_title="lambda 2",
        zaxis_title="lambda 3",
        bgcolor="#09100f",
    )
    return _layout(fig, "Euler Characteristic Surface")


def make_vineyard_fig(tracks: pd.DataFrame, sequence: pd.DataFrame | None = None) -> go.Figure:
    """Animated vineyard tracks and bottleneck drift."""

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scatter"}, {"type": "scatter"}]],
        subplot_titles=("H1 vineyard", "bottleneck drift"),
    )
    if not tracks.empty:
        for track_id, group in tracks.groupby("track_id"):
            fig.add_trace(
                go.Scatter(
                    x=group["date"],
                    y=group["persistence"],
                    mode="lines+markers",
                    name=f"H1 {track_id}",
                    line={"width": 1.2},
                    marker={"size": 5},
                    hovertemplate="birth %{customdata[0]:.3f}<br>death %{customdata[1]:.3f}<extra></extra>",
                    customdata=group[["birth", "death"]],
                    showlegend=False,
                ),
                row=1,
                col=1,
            )
    if sequence is not None and not sequence.empty:
        fig.add_trace(
            go.Scatter(
                x=sequence["window_end"],
                y=sequence["bottleneck_to_baseline"],
                mode="lines+markers",
                name="bottleneck",
                line={"color": "#ff6b6b", "width": 2.5},
            ),
            row=1,
            col=2,
        )
    fig.update_xaxes(title_text="time")
    fig.update_yaxes(title_text="persistence", row=1, col=1)
    fig.update_yaxes(title_text="distance", row=1, col=2)
    return _layout(fig, "Persistence Vineyard")


def make_mapper_fig(mapper_result: dict[str, object]) -> go.Figure:
    """Plot the opportunity Mapper graph."""

    graph = mapper_result.get("graph", {})
    frame = mapper_result.get("frame", pd.DataFrame())
    if frame is None or frame.empty:
        return _layout(go.Figure(), "Topological Opportunity Mapper")
    node_ids = frame["node_id"].tolist()
    n = len(node_ids)
    angles = np.linspace(0.0, 2.0 * math.pi, max(n, 1), endpoint=False)
    positions = {node_id: (math.cos(angle), math.sin(angle)) for node_id, angle in zip(node_ids, angles)}

    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    links = graph.get("links", {}) if isinstance(graph, dict) else {}
    for left, neighbors in links.items():
        for right in neighbors:
            if left not in positions or right not in positions or str(left) > str(right):
                continue
            x0, y0 = positions[left]
            x1, y1 = positions[right]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line={"color": "#49645f", "width": 1.0}, hoverinfo="skip"))
    fig.add_trace(
        go.Scatter(
            x=[positions[node][0] for node in node_ids],
            y=[positions[node][1] for node in node_ids],
            mode="markers",
            marker={
                "size": np.clip(frame["size"].to_numpy(dtype=float) * 1.9, 10, 48),
                "color": frame["signed_barcode_length"],
                "colorscale": [[0, "#78dcca"], [0.55, "#f7c948"], [1, "#ff6b6b"]],
                "line": {"color": "#f1f6f3", "width": 0.8},
                "colorbar": {"title": "signed"},
            },
            customdata=frame[["node_id", "size", "median_listing_price", "neighborhoods"]],
            hovertemplate="%{customdata[0]}<br>%{customdata[1]} homes<br>$%{customdata[2]:,.0f}<br>%{customdata[3]}<extra></extra>",
        )
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
    return _layout(fig, "Topological Opportunity Mapper")


def make_causal_fig(ate: dict[str, object]) -> go.Figure:
    """Plot factual versus counterfactual topological effect."""

    fig = go.Figure(
        data=[
            go.Bar(
                x=["H0", "H1"],
                y=[ate.get("topological_ate_h0", 0.0), ate.get("topological_ate_h1", 0.0)],
                marker={"color": ["#78dcca", "#ff6b6b"]},
                hovertemplate="%{x}: %{y:.4f}<extra></extra>",
            )
        ]
    )
    fig.update_yaxes(title_text="Wasserstein / bottleneck proxy")
    return _layout(fig, "Causal Shock Lab")


def make_decision_fig(signal: dict[str, object]) -> go.Figure:
    """Plot the landscape difference used by S(B)."""

    xs = signal["grid"]
    full = signal["full_landscape"].mean(axis=0)
    sub = signal["sub_landscape"].mean(axis=0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=full, mode="lines", line={"color": "#78dcca", "width": 2}, name="full"))
    fig.add_trace(go.Scatter(x=xs, y=sub, mode="lines", line={"color": "#ff6b6b", "width": 2}, name="sublevel"))
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=sub - full,
            mode="lines",
            line={"color": "#f7c948", "width": 1.5},
            fill="tozeroy",
            name="integrand",
        )
    )
    fig.update_yaxes(title_text="landscape amplitude")
    fig.update_xaxes(title_text="filtration scale")
    return _layout(fig, f"Decision Boundary Navigator: {signal['decision']} ({signal['normalized_signal']:.2f})")
