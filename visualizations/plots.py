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
    return _layout(fig, "Counterfactual Shock Lab")


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


def make_multiparameter_fig(result) -> go.Figure:
    """Plot Hilbert slices and signed multiparameter mass."""

    hilbert = result.hilbert_frame.copy()
    signed = result.signed_measure.copy()
    fig = make_subplots(
        rows=1,
        cols=3,
        specs=[[{"type": "heatmap"}, {"type": "heatmap"}, {"type": "scatter3d"}]],
        subplot_titles=("Hilbert H0 slice", "Hilbert H1 slice", f"Signed measure ({result.backend})"),
    )
    if not hilbert.empty:
        mid_k = int(hilbert["axis_k"].median()) if "axis_k" in hilbert else None
        for col, degree in [(1, 0), (2, 1)]:
            subset = hilbert[hilbert["degree"] == degree]
            if mid_k is not None:
                subset = subset[subset["axis_k"] == mid_k]
            pivot = subset.pivot_table(index="lambda_2", columns="lambda_1", values="hilbert_value", aggfunc="mean")
            fig.add_trace(
                go.Heatmap(
                    x=pivot.columns,
                    y=pivot.index,
                    z=pivot.to_numpy(),
                    colorscale=[[0, "#102522"], [0.5, "#78dcca"], [1, "#f7c948"]],
                    colorbar={"title": f"H{degree}"} if col == 2 else None,
                    showscale=col == 2,
                ),
                row=1,
                col=col,
            )
    if not signed.empty:
        fig.add_trace(
            go.Scatter3d(
                x=signed["lambda_1"],
                y=signed["lambda_2"],
                z=signed["lambda_3"],
                mode="markers",
                marker={
                    "size": np.clip(np.abs(signed.get("weight", pd.Series(np.ones(len(signed))))).to_numpy(dtype=float) * 9 + 3, 3, 18),
                    "color": signed.get("weight", pd.Series(np.zeros(len(signed)))),
                    "colorscale": [[0, "#ff6b6b"], [0.5, "#f1f6f3"], [1, "#78dcca"]],
                    "opacity": 0.72,
                },
                hovertemplate="lambda=(%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>",
            ),
            row=1,
            col=3,
        )
    fig.update_xaxes(title_text="lambda 1", row=1, col=1)
    fig.update_yaxes(title_text="lambda 2", row=1, col=1)
    fig.update_xaxes(title_text="lambda 1", row=1, col=2)
    fig.update_yaxes(title_text="lambda 2", row=1, col=2)
    fig.update_scenes(xaxis_title="lambda 1", yaxis_title="lambda 2", zaxis_title="lambda 3", bgcolor="#09100f")
    title = "Multiparameter Persistence Lab"
    if not result.exact:
        title += " (fallback certified)"
    return _layout(fig, title)


def make_silhouette_fig(suite: dict[str, object]) -> go.Figure:
    """Plot persistence silhouettes and Betti curves against baseline."""

    frame = suite["frame"].copy()
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("H0 silhouette", "H1 silhouette", "H0 Betti curve", "H1 Betti curve"),
    )
    for col, homology in [(1, "H0"), (2, "H1")]:
        subset = frame[frame["homology"] == homology]
        fig.add_trace(go.Scatter(x=subset["scale"], y=subset["baseline_silhouette"], mode="lines", name=f"{homology} baseline", line={"color": "#78dcca"}), row=1, col=col)
        fig.add_trace(go.Scatter(x=subset["scale"], y=subset["current_silhouette"], mode="lines", name=f"{homology} current", line={"color": "#ff6b6b"}), row=1, col=col)
        fig.add_trace(go.Scatter(x=subset["scale"], y=subset["baseline_betti"], mode="lines", showlegend=False, line={"color": "#78dcca"}), row=2, col=col)
        fig.add_trace(go.Scatter(x=subset["scale"], y=subset["current_betti"], mode="lines", showlegend=False, line={"color": "#ff6b6b"}), row=2, col=col)
    fig.update_xaxes(title_text="filtration scale")
    fig.update_yaxes(title_text="silhouette", row=1)
    fig.update_yaxes(title_text="beta", row=2)
    return _layout(fig, "Persistence Silhouettes and Betti Curves")


def make_boundary_fig(boundary: pd.DataFrame) -> go.Figure:
    """Plot sampled biography-space decision boundary."""

    if boundary.empty:
        return _layout(go.Figure(), "Topological Decision Boundary")
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x=sorted(boundary["annual_income"].unique()),
            y=sorted(boundary["dti_max"].unique()),
            z=boundary.pivot_table(index="dti_max", columns="annual_income", values="normalized_signal", aggfunc="mean").to_numpy(),
            colorscale=[[0, "#ff6b6b"], [0.5, "#102522"], [1, "#78dcca"]],
            colorbar={"title": "S(B)"},
        )
    )
    edge = boundary[boundary["boundary"]]
    if not edge.empty:
        fig.add_trace(
            go.Scatter(
                x=edge["annual_income"],
                y=edge["dti_max"],
                mode="markers",
                marker={"symbol": "x", "size": 10, "color": "#f7c948", "line": {"width": 2}},
                name="boundary",
                customdata=edge[["decision", "h1_longest_persistence", "restricted_count"]],
                hovertemplate="$%{x:,.0f}<br>DTI %{y:.2f}<br>%{customdata[0]}<br>H1 %{customdata[1]:.3f}<br>%{customdata[2]} homes<extra></extra>",
            )
        )
    fig.update_xaxes(title_text="annual income")
    fig.update_yaxes(title_text="DTI max")
    return _layout(fig, "Topological Decision Boundary")


def make_gp_regime_fig(training_frame: pd.DataFrame, prediction: dict[str, object]) -> go.Figure:
    """Plot GP training regimes and current prediction."""

    fig = go.Figure()
    if not training_frame.empty:
        color_map = {"Stable": "#78dcca", "Overheated": "#f7c948", "Crash": "#ff6b6b", "Opportunity": "#9bd67d"}
        fig.add_trace(
            go.Scatter(
                x=training_frame["date"],
                y=training_frame["f7"],
                mode="markers+lines",
                marker={"size": 10, "color": [color_map.get(label, "#f1f6f3") for label in training_frame["regime"]]},
                customdata=training_frame[["regime"]],
                hovertemplate="%{x|%b %Y}<br>peak curvature %{y:.2f}<br>%{customdata[0]}<extra></extra>",
            )
        )
    fig.add_annotation(
        text=f"Current regime: {prediction.get('regime')} ({prediction.get('backend')})",
        xref="paper",
        yref="paper",
        x=0.02,
        y=0.96,
        showarrow=False,
        font={"size": 15, "color": "#f1f6f3"},
    )
    fig.update_yaxes(title_text="Euler curvature peak")
    fig.update_xaxes(title_text="training month")
    return _layout(fig, "Gaussian Process Regime Classifier")


def make_validation_price_fig(price_result: dict) -> go.Figure:
    """Predicted vs Actual scatter with identity line."""
    monthly = price_result.get("monthly_df")
    fig = go.Figure()
    if monthly is not None and not monthly.empty:
        fig.add_trace(
            go.Scatter(
                x=monthly["median_listing_price"] if "median_listing_price" in monthly else monthly["monthly_rent_estimate"],
                y=monthly["syn_median"] if "syn_median" in monthly else monthly["monthly_rent_estimate"],
                mode="markers",
                marker={"size": 8, "color": "#78dcca", "opacity": 0.7},
                name="Monthly median",
                hovertemplate="Actual: $%{x:,.0f}<br>Predicted: $%{y:,.0f}<extra></extra>",
            )
        )
        # Identity line
        vals = monthly.iloc[:, 1:3].to_numpy(dtype=float).flatten()
        vals = vals[~np.isnan(vals)]
        if len(vals) > 0:
            lo, hi = float(np.min(vals)), float(np.max(vals))
            fig.add_trace(
                go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line={"color": "#f7c948", "dash": "dash", "width": 1},
                           name="Identity", hoverinfo="skip")
            )
    fig.update_xaxes(title_text="Actual")
    fig.update_yaxes(title_text="Predicted")
    mode = price_result.get("mode", "unknown")
    rmse = price_result.get("rmse")
    r2 = price_result.get("r2")
    title = f"Predicted vs Actual ({mode})"
    if rmse is not None:
        title += f" — RMSE=${rmse:,.0f}"
    if r2 is not None and not np.isnan(r2):
        title += f", R²={r2:.3f}"
    return _layout(fig, title)


def make_validation_residual_fig(price_result: dict) -> go.Figure:
    """Residual error over time."""
    monthly = price_result.get("monthly_df")
    fig = go.Figure()
    if monthly is not None and not monthly.empty and "residual" in monthly.columns:
        fig.add_trace(
            go.Scatter(
                x=monthly["date"],
                y=monthly["residual"],
                mode="lines+markers",
                line={"color": "#ff6b6b", "width": 2},
                marker={"size": 6},
                name="Residual",
                hovertemplate="%{x|%b %Y}<br>Residual: $%{y:,.0f}<extra></extra>",
            )
        )
        fig.add_hline(y=0, line={"color": "#8ba19a", "width": 0.8, "dash": "dash"})
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Residual ($)")
    return _layout(fig, "Residual Error Over Time")


def make_validation_zip_fig(zip_errors: pd.DataFrame) -> go.Figure:
    """ZIP-level deviation bar chart."""
    fig = go.Figure()
    if not zip_errors.empty:
        colors = ["#ff6b6b" if abs(v) > 15 else ("#f7c948" if abs(v) > 8 else "#78dcca") for v in zip_errors["metro_deviation_pct"]]
        fig.add_trace(
            go.Bar(
                x=zip_errors["zip_code"],
                y=zip_errors["metro_deviation_pct"],
                marker={"color": colors},
                text=zip_errors["metro_deviation_pct"].apply(lambda v: f"{v:+.1f}%"),
                textposition="outside",
                hovertemplate="%{x}<br>Median: $%{customdata:,.0f}<br>Deviation: %{y:+.1f}%<extra></extra>",
                customdata=zip_errors["median_price"],
            )
        )
    fig.update_xaxes(title_text="ZIP Code")
    fig.update_yaxes(title_text="Deviation from Metro Median (%)")
    return _layout(fig, "ZIP-Level Price Deviation")


def make_validation_regime_fig(regime_result: dict) -> go.Figure:
    """Confusion matrix heatmap for regime classification."""
    labels = regime_result.get("confusion_labels", [])
    matrix = regime_result.get("confusion_matrix", [])
    if not matrix or not labels:
        return _layout(go.Figure(), "Regime Classification Report")
    accuracy = regime_result.get("accuracy")
    title = "Regime Classification Report"
    if accuracy is not None:
        title += f" — Accuracy: {accuracy:.1%}"

    # Row-normalize for heatmap
    mat = np.array(matrix, dtype=float)
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    mat_norm = mat / row_sums

    fig = go.Figure(
        data=go.Heatmap(
            z=mat_norm,
            x=labels,
            y=labels,
            colorscale=[[0, "#102522"], [0.5, "#78dcca"], [1, "#f7c948"]],
            text=[[f"{int(v)}" for v in row] for row in mat],
            texttemplate="%{text}",
            textfont={"color": "#f1f6f3"},
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{text}<extra></extra>",
            colorbar={"title": "Fraction"},
        )
    )
    fig.update_xaxes(title_text="Predicted")
    fig.update_yaxes(title_text="Actual")
    return _layout(fig, title)
