"""Dash application for the Tulsa Topological Affordability Spacetime."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from dash import Dash, Input, Output, State, dcc, html

from data.embeddings import compute_embeddings
from data.fetch_data import load_or_create_dataset
from data.preprocess import add_topological_parameters
from decision.opportunity_mapper import build_opportunity_graph
from decision.path_integral import UserBiography, buy_signal
from topology.causal_tda import topological_ate
from topology.invariants import compute_time_slice_invariants
from topology.vineyards import bayesian_blocks_change_points, compute_sliding_window_vineyards, vineyard_tracks
from visualizations.plots import (
    make_causal_fig,
    make_decision_fig,
    make_euler_surface_fig,
    make_mapper_fig,
    make_spacetime_fig,
    make_vineyard_fig,
)


CACHE_DIR = PROJECT_ROOT / "outputs" / "cache"
EMBEDDED_PATH = CACHE_DIR / "tulsa_embedded.csv"


def load_market() -> pd.DataFrame:
    """Load cached dashboard data or build it locally."""

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if EMBEDDED_PATH.exists():
        return pd.read_csv(EMBEDDED_PATH, parse_dates=["date"])
    df = add_topological_parameters(load_or_create_dataset())
    embedded, _ = compute_embeddings(df)
    embedded.to_csv(EMBEDDED_PATH, index=False)
    return embedded


MARKET = load_market()
MONTHS = sorted(pd.to_datetime(MARKET["date"].unique()))
LATEST = MONTHS[-1]
LATEST_SLICE = MARKET[MARKET["date"] == LATEST]
INVARIANTS = compute_time_slice_invariants(MARKET, date=LATEST, max_points=260, grid_size=12)
SEQUENCE, DIAGRAMS = compute_sliding_window_vineyards(MARKET, window_months=12, max_points=220)
TRACKS = vineyard_tracks(DIAGRAMS)
CHANGE_POINTS = bayesian_blocks_change_points(SEQUENCE)
MAPPER = build_opportunity_graph(LATEST_SLICE, n_bins=6)
DEFAULT_ATE = topological_ate(MARKET, shock_bps=-100.0, date=LATEST, max_points=260)
DEFAULT_SIGNAL = buy_signal(MARKET, UserBiography(), date=LATEST, max_points=260)


app = Dash(__name__, title="TTAS", suppress_callback_exceptions=True)
server = app.server


def metric(label: str, value: str) -> html.Div:
    return html.Div([html.Span(label), html.Strong(value)], className="metric")


app.layout = html.Div(
    className="shell",
    children=[
        html.Div(
            className="topbar",
            children=[
                html.Div(
                    className="brand",
                    children=[
                        html.H1("Tulsa Topological Affordability Spacetime"),
                        html.P("Multiparameter persistence, causal shocks, and rent-vs-buy topology"),
                    ],
                ),
                html.Div(
                    className="metric-strip",
                    children=[
                        metric("properties", f"{len(MARKET):,}"),
                        metric("latest", LATEST.strftime("%b %Y")),
                        metric("H1 entropy", f"{INVARIANTS['persistent_entropy_h1']:.2f}"),
                        metric("change points", str(len(CHANGE_POINTS))),
                    ],
                ),
            ],
        ),
        dcc.Tabs(
            id="tabs",
            value="tab-manifold",
            className="tabs",
            children=[
                dcc.Tab(label="Spacetime Manifold", value="tab-manifold", className="tab", selected_className="tab--selected"),
                dcc.Tab(label="Euler Surface", value="tab-euler", className="tab", selected_className="tab--selected"),
                dcc.Tab(label="Persistence Vineyard", value="tab-vineyard", className="tab", selected_className="tab--selected"),
                dcc.Tab(label="Opportunity Mapper", value="tab-mapper", className="tab", selected_className="tab--selected"),
                dcc.Tab(label="Causal Shock Lab", value="tab-causal", className="tab", selected_className="tab--selected"),
                dcc.Tab(label="Decision Navigator", value="tab-decision", className="tab", selected_className="tab--selected"),
            ],
        ),
        html.Div(id="tab-body", className="workspace"),
    ],
)


@app.callback(Output("tab-body", "children"), Input("tabs", "value"))
def render_tab(tab: str):
    if tab == "tab-manifold":
        return html.Div(
            children=[
                html.Div(
                    className="control-row",
                    children=[
                        html.Div(
                            className="control",
                            children=[
                                html.Label("Month"),
                                dcc.Slider(
                                    id="month-slider",
                                    min=0,
                                    max=len(MONTHS) - 1,
                                    value=len(MONTHS) - 1,
                                    step=1,
                                    marks={0: MONTHS[0].strftime("%Y"), len(MONTHS) - 1: MONTHS[-1].strftime("%Y")},
                                ),
                            ],
                        )
                    ],
                ),
                dcc.Graph(id="manifold-graph", figure=make_spacetime_fig(MARKET, LATEST), className="graph-frame"),
            ]
        )
    if tab == "tab-euler":
        return dcc.Graph(figure=make_euler_surface_fig(INVARIANTS["euler_frame"]), className="graph-frame")
    if tab == "tab-vineyard":
        return dcc.Graph(figure=make_vineyard_fig(TRACKS, SEQUENCE), className="graph-frame")
    if tab == "tab-mapper":
        return dcc.Graph(figure=make_mapper_fig(MAPPER), className="graph-frame")
    if tab == "tab-causal":
        return html.Div(
            children=[
                html.Div(
                    className="control-row",
                    children=[
                        html.Div(className="control", children=[html.Label("Rate shock bps"), dcc.Input(id="shock-bps", type="number", value=-100)]),
                        html.Div(className="control", children=[html.Label("Run"), html.Button("Recalculate", id="shock-run", n_clicks=0)]),
                    ],
                ),
                dcc.Graph(id="causal-graph", figure=make_causal_fig(DEFAULT_ATE), className="graph-frame"),
            ]
        )
    return html.Div(
        children=[
            html.Div(
                className="control-row",
                children=[
                    html.Div(className="control", children=[html.Label("Annual income"), dcc.Input(id="income", type="number", value=92_000)]),
                    html.Div(className="control", children=[html.Label("DTI max"), dcc.Input(id="dti", type="number", value=0.38, step=0.01)]),
                    html.Div(className="control", children=[html.Label("Family size"), dcc.Input(id="family", type="number", value=3)]),
                    html.Div(className="control", children=[html.Label("Run"), html.Button("Recalculate", id="decision-run", n_clicks=0)]),
                ],
            ),
            dcc.Graph(id="decision-graph", figure=make_decision_fig(DEFAULT_SIGNAL), className="graph-frame"),
        ]
    )


@app.callback(Output("manifold-graph", "figure"), Input("month-slider", "value"), prevent_initial_call=True)
def update_manifold(month_index: int):
    return make_spacetime_fig(MARKET, MONTHS[int(month_index)])


@app.callback(
    Output("causal-graph", "figure"),
    Input("shock-run", "n_clicks"),
    State("shock-bps", "value"),
    prevent_initial_call=True,
)
def update_causal(_: int, shock_bps: float):
    ate = topological_ate(MARKET, shock_bps=float(shock_bps or -100), date=LATEST, max_points=260)
    return make_causal_fig(ate)


@app.callback(
    Output("decision-graph", "figure"),
    Input("decision-run", "n_clicks"),
    State("income", "value"),
    State("dti", "value"),
    State("family", "value"),
    prevent_initial_call=True,
)
def update_decision(_: int, income: float, dti: float, family: int):
    biography = UserBiography(annual_income=float(income or 92_000), dti_max=float(dti or 0.38), family_size=int(family or 3))
    signal = buy_signal(MARKET, biography, date=LATEST, max_points=260)
    return make_decision_fig(signal)


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8050)
