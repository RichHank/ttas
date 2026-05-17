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
from data.fetch_data import get_data_mode, load_or_create_dataset
from data.lineage import (
    build_lineage_frame,
    build_source_summary_frame,
    NATURES,
    get_natures_for_columns,
    nature_badge_html,
)
from data.preprocess import add_topological_parameters
from decision.opportunity_mapper import build_opportunity_graph
from decision.path_integral import UserBiography, buy_signal
from decision.phase_transition import predict_regime_with_gp, train_gp_regime_classifier
from decision.topological_boundary import compute_topological_boundary
from decision.analyst_notes import generate_all_regime_notes, regime_note_to_html
from decision.validation import build_validation_dashboard_data
from topology.causal_tda import topological_ate
from topology.invariants import compute_time_slice_invariants
from topology.multiparameter import compute_multiparameter_persistence
from topology.silhouettes import compute_silhouette_suite
from topology.vineyards import bayesian_blocks_change_points, compute_sliding_window_vineyards, vineyard_tracks
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
    make_validation_price_fig,
    make_validation_residual_fig,
    make_validation_zip_fig,
    make_validation_regime_fig,
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
MULTIPARAMETER = compute_multiparameter_persistence(INVARIANTS["filtration"], grid_size=12)
SEQUENCE, DIAGRAMS = compute_sliding_window_vineyards(MARKET, window_months=12, max_points=220)
TRACKS = vineyard_tracks(DIAGRAMS)
CHANGE_POINTS = bayesian_blocks_change_points(SEQUENCE)
MAPPER = build_opportunity_graph(LATEST_SLICE, n_bins=6)
DEFAULT_ATE = topological_ate(MARKET, shock_bps=-100.0, date=LATEST, max_points=260)
DEFAULT_SIGNAL = buy_signal(MARKET, UserBiography(), date=LATEST, max_points=260)
SILHOUETTES = compute_silhouette_suite(MARKET, current_date=LATEST, max_points=220)
BOUNDARY = compute_topological_boundary(MARKET, date=LATEST, max_points=140)
GP_MODEL, GP_TRAINING = train_gp_regime_classifier(MARKET, max_slices=12, max_points=140, grid_size=7)
GP_PREDICTION = predict_regime_with_gp(GP_MODEL, INVARIANTS["euler_surface"])

# ── Model validation ──────────────────────────────────────────────────────

VALIDATION = build_validation_dashboard_data(MARKET)

# ── Analyst notes ─────────────────────────────────────────────────────────

REGIME_NOTES = generate_all_regime_notes(MARKET)
CURRENT_REGIME = str(GP_PREDICTION["regime"])
CURRENT_NOTE = REGIME_NOTES.get(CURRENT_REGIME)

# ── Data lineage ──────────────────────────────────────────────────────────

LINEAGE_FRAME = build_lineage_frame()
SOURCE_FRAME = build_source_summary_frame()


def _natures_for_market(market: pd.DataFrame) -> set[str]:
    """Return the set of data natures present in the working manifold columns."""
    cols = [c for c in market.columns if c in LINEAGE_FRAME["column"].values]
    return get_natures_for_columns(cols)


MARKET_NATURES = _natures_for_market(MARKET)

# Per-tab provenance: which data natures does each visualization consume?
# Columns involved in each tab (derived from what the plot functions use).
TAB_PROVENANCE: dict[str, set[str]] = {
    "tab-manifold": {"Calibrated", "Synthetic"},
    "tab-multiparameter": {"Observed", "Calibrated", "Derived", "Synthetic"},
    "tab-euler": {"Observed", "Calibrated", "Derived", "Synthetic"},
    "tab-silhouettes": {"Observed", "Calibrated", "Derived", "Synthetic"},
    "tab-vineyard": {"Observed", "Calibrated", "Derived", "Synthetic"},
    "tab-mapper": {"Calibrated", "Synthetic"},
    "tab-causal": {"Observed", "Calibrated", "Synthetic"},
    "tab-gp": {"Observed", "Calibrated", "Derived", "Synthetic"},
    "tab-boundary": {"Calibrated", "Derived", "Synthetic"},
    "tab-decision": {"Observed", "Calibrated", "Derived", "Synthetic"},
    "tab-validation": {"Observed", "Calibrated", "Derived", "Synthetic"},
    "tab-lineage": {"Observed", "Calibrated", "Derived", "Modeled", "Synthetic"},
}


def make_provenance_strip(tab_value: str) -> html.Div:
    """Build a provenance badge strip for a given tab."""
    natures = TAB_PROVENANCE.get(tab_value, set())
    if not natures:
        return html.Div()
    ordered = [n for n in NATURES if n in natures]
    badges = [
        html.Span(n, className=f"badge badge--{n.lower()}")
        for n in ordered
    ]
    return html.Div(
        className="provenance-strip",
        children=[
            html.Span("Data: ", className="provenance-label"),
            *badges,
        ],
    )


# ── App shell ─────────────────────────────────────────────────────────────

app = Dash(__name__, title="TTAS", suppress_callback_exceptions=True)
server = app.server

DATA_MODE = get_data_mode()
DATA_BADGE_LABEL = "Real Data" if DATA_MODE == "real_public_data" else "Synthetic"
DATA_BADGE_CLASS = "badge badge--real" if DATA_MODE == "real_public_data" else "badge badge--synthetic"


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
                        html.P("Multiparameter persistence, counterfactual shocks, and rent-vs-buy topology"),
                    ],
                ),
                html.Div(
                    className="metric-strip",
                    children=[
                        metric("data mode", DATA_BADGE_LABEL),
                        metric("properties", f"{len(MARKET):,}"),
                        metric("latest", LATEST.strftime("%b %Y")),
                        metric("H1 entropy", f"{INVARIANTS['persistent_entropy_h1']:.2f}"),
                        metric("MP backend", MULTIPARAMETER.backend),
                        metric("regime", str(GP_PREDICTION["regime"])),
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
                dcc.Tab(label="Multiparameter Lab", value="tab-multiparameter", className="tab", selected_className="tab--selected"),
                dcc.Tab(label="Euler Surface", value="tab-euler", className="tab", selected_className="tab--selected"),
                dcc.Tab(label="Silhouettes + Betti", value="tab-silhouettes", className="tab", selected_className="tab--selected"),
                dcc.Tab(label="Persistence Vineyard", value="tab-vineyard", className="tab", selected_className="tab--selected"),
                dcc.Tab(label="Opportunity Mapper", value="tab-mapper", className="tab", selected_className="tab--selected"),
                dcc.Tab(label="Counterfactual Shock Lab", value="tab-causal", className="tab", selected_className="tab--selected"),
                dcc.Tab(label="Regime GP", value="tab-gp", className="tab", selected_className="tab--selected"),
                dcc.Tab(label="Boundary Atlas", value="tab-boundary", className="tab", selected_className="tab--selected"),
                dcc.Tab(label="Decision Navigator", value="tab-decision", className="tab", selected_className="tab--selected"),
                dcc.Tab(label="Model Validation", value="tab-validation", className="tab", selected_className="tab--selected"),
                dcc.Tab(label="Data Lineage", value="tab-lineage", className="tab", selected_className="tab--selected"),
            ],
        ),
        html.Div(id="tab-body", className="workspace"),
        html.Footer(
            className="app-footer",
            children=[
                html.Span(f"Data mode: {DATA_BADGE_LABEL} — Tulsa-calibrated manifold"),
                html.Span(
                    "Sources: FRED, Census ACS, Realtor.com Research, OSMnx, "
                    "Tulsa County Assessor / City of Tulsa Open Data"
                ),
                html.Span("Not financial, legal, or appraisal advice."),
                html.Div(
                    className="report-actions",
                    children=[
                        html.Button("Generate Report", id="report-btn", n_clicks=0, className="btn-report"),
                        html.Div(id="report-status", style={"color": "#78dcca", "fontSize": "13px", "alignSelf": "center"}),
                    ],
                ),
            ],
        ),
    ],
)


# ── Tab rendering ─────────────────────────────────────────────────────────


def _with_provenance(tab_value: str, *children) -> list:
    """Wrap children in a provenance strip for consistent tab layout."""
    return [make_provenance_strip(tab_value), *children]


@app.callback(Output("tab-body", "children"), Input("tabs", "value"))
def render_tab(tab: str):
    if tab == "tab-manifold":
        return html.Div(
            children=_with_provenance(
                tab,
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
            )
        )
    if tab == "tab-multiparameter":
        return html.Div(
            children=_with_provenance(
                tab,
                dcc.Graph(figure=make_multiparameter_fig(MULTIPARAMETER), className="graph-frame"),
            )
        )
    if tab == "tab-euler":
        return html.Div(
            children=_with_provenance(
                tab,
                dcc.Graph(figure=make_euler_surface_fig(INVARIANTS["euler_frame"]), className="graph-frame"),
            )
        )
    if tab == "tab-silhouettes":
        return html.Div(
            children=_with_provenance(
                tab,
                dcc.Graph(figure=make_silhouette_fig(SILHOUETTES), className="graph-frame"),
            )
        )
    if tab == "tab-vineyard":
        return html.Div(
            children=_with_provenance(
                tab,
                dcc.Graph(figure=make_vineyard_fig(TRACKS, SEQUENCE), className="graph-frame"),
            )
        )
    if tab == "tab-mapper":
        return html.Div(
            children=_with_provenance(
                tab,
                dcc.Graph(figure=make_mapper_fig(MAPPER), className="graph-frame"),
            )
        )
    if tab == "tab-gp":
        gp_children = [dcc.Graph(figure=make_gp_regime_fig(GP_TRAINING, GP_PREDICTION), className="graph-frame")]
        if CURRENT_NOTE is not None:
            gp_children.append(
                html.Iframe(
                    srcDoc=regime_note_to_html(CURRENT_NOTE),
                    style={"width": "100%", "height": "320px", "border": "none", "marginTop": "8px"},
                )
            )
        return html.Div(children=_with_provenance(tab, *gp_children))
    if tab == "tab-boundary":
        return html.Div(
            children=_with_provenance(
                tab,
                dcc.Graph(figure=make_boundary_fig(BOUNDARY), className="graph-frame"),
            )
        )
    if tab == "tab-causal":
        return html.Div(
            children=_with_provenance(
                tab,
                html.Div(
                    className="control-row",
                    children=[
                        html.Div(className="control", children=[html.Label("Rate shock bps"), dcc.Input(id="shock-bps", type="number", value=-100)]),
                        html.Div(className="control", children=[html.Label("Run"), html.Button("Recalculate", id="shock-run", n_clicks=0)]),
                    ],
                ),
                dcc.Graph(id="causal-graph", figure=make_causal_fig(DEFAULT_ATE), className="graph-frame"),
            )
        )
    if tab == "tab-validation":
        return _render_validation_tab()
    if tab == "tab-lineage":
        return _render_lineage_tab()

    # Decision Navigator
    decision_children = [
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
    if CURRENT_NOTE is not None:
        decision_children.append(
            html.Div(
                className="analyst-note",
                style={"marginTop": "10px"},
                children=[
                    html.Div(
                        className="analyst-note__header",
                        children=[
                            html.Span(CURRENT_NOTE.regime, className=f"badge badge--observed"),
                            html.Span(f" {CURRENT_NOTE.period} · {CURRENT_NOTE.months} months · confidence: {CURRENT_NOTE.confidence}", style={"color": "#8ba19a", "fontSize": "12px"}),
                        ],
                    ),
                    html.P(CURRENT_NOTE.decision_support, style={"color": "#f1f6f3", "fontSize": "13px", "lineHeight": "1.55"}),
                ],
            )
        )
    return html.Div(children=_with_provenance(tab, *decision_children))


# ── Data Lineage tab ──────────────────────────────────────────────────────


def _render_lineage_tab():
    """Build the Data Lineage tab with source cards and per-column table."""

    # Source summary cards
    source_cards = []
    for _, row in SOURCE_FRAME.iterrows():
        status_badge = (
            html.Span(row["status"], className="badge badge--observed")
            if row.get("status") == "active"
            else html.Span(row.get("status", ""), className="badge badge--synthetic")
        )
        source_cards.append(
            html.Div(
                className="source-card",
                children=[
                    html.Div(
                        className="source-card-header",
                        children=[
                            html.H3(row["source"]),
                            status_badge,
                        ],
                    ),
                    html.Div(
                        className="source-meta",
                        children=[
                            html.Div([html.Strong("Endpoint: "), html.Span(row.get("api_endpoint", ""))]),
                            html.Div([html.Strong("Pulled: "), html.Span(str(row.get("date_pulled", "")))]),
                            html.Div([html.Strong("Nature: "), html.Span(row.get("nature", ""))]),
                            html.Div([html.Strong("Confidence: "), html.Span(str(row.get("confidence", "")))]),
                            html.Div([html.Strong("Transformations: "), html.Span(str(row.get("transformations", "")))]),
                            html.Div([html.Strong("Notes: "), html.Span(str(row.get("notes", "")))]),
                        ],
                    ),
                    html.Div(
                        className="source-fields",
                        children=f"Fields: {row.get('fields', [])}".replace("[", "").replace("]", "").replace("'", ""),
                    ),
                ],
            )
        )

    # Per-column table
    header = html.Thead(
        html.Tr([
            html.Th("Column"),
            html.Th("Nature"),
            html.Th("Primary Source"),
            html.Th("Confidence"),
            html.Th("Transformations"),
        ])
    )
    rows = []
    for _, r in LINEAGE_FRAME.iterrows():
        conf = r["confidence"]
        conf_class = "high" if conf >= 0.80 else ("medium" if conf >= 0.50 else "low")
        bar_width = int(conf * 80)
        rows.append(
            html.Tr([
                html.Td(r["column"], className="col-name"),
                html.Td(html.Span(r["nature"], className=f"badge badge--{r['nature'].lower()}")),
                html.Td(r["primary_source"]),
                html.Td(
                    html.Span(
                        f" {r['confidence']:.0%}",
                        style={"whiteSpace": "nowrap"},
                    ),
                ),
                html.Td(r["transformations"]),
            ])
        )
    table = html.Table(header, *[html.Tbody(rows)], className="lineage-table")

    return html.Div(
        children=_with_provenance(
            "tab-lineage",
            html.H3("Data Sources", style={"marginTop": 0, "color": "#78dcca", "fontSize": "18px"}),
            html.Div(source_cards, className="source-grid"),
            html.H3("Per-Column Provenance", style={"marginTop": "16px", "color": "#78dcca", "fontSize": "18px"}),
            html.P(
                f"Each column in the {len(LINEAGE_FRAME)}-column manifold is classified as Observed, Calibrated, "
                "Derived, Modeled, or Synthetic. Confidence reflects how directly the value traces to a real-world measurement.",
                style={"color": "#8ba19a", "fontSize": "13px", "marginBottom": "12px"},
            ),
            html.Div(table, className="lineage-table-wrap"),
        )
    )


# ── Model Validation tab ──────────────────────────────────────────────────


def _render_validation_tab():
    """Build the Model Validation tab with metrics and charts."""
    price = VALIDATION.get("price", {})
    rent = VALIDATION.get("rent", {})
    zip_errors = VALIDATION.get("zip_errors", pd.DataFrame())
    regime = VALIDATION.get("regime", {})
    stability = VALIDATION.get("stability", {})
    val_mode = VALIDATION.get("validation_mode", "unknown")
    data_mode = VALIDATION.get("data_mode", "synthetic_fallback")

    # Note banner for synthetic mode
    note = None
    if val_mode.startswith("internal"):
        note = html.Div(
            "Real data not configured — showing internal consistency metrics "
            "(train/test split on synthetic data). Set FRED_API_KEY + Realtor.com "
            "CSVs for observed-vs-predicted validation.",
            className="validation-note",
        )

    # Metrics row
    metrics = []
    price_rmse = price.get("rmse")
    if price_rmse is not None:
        cls = "good" if price_rmse < 20000 else ("warn" if price_rmse < 50000 else "bad")
        metrics.append(html.Div([
            html.Span("Price RMSE", className="vm-label"),
            html.Span(f"${price_rmse:,.0f}", className=f"vm-value {cls}"),
        ], className="validation-metric"))

    price_r2 = price.get("r2")
    if price_r2 is not None and not (isinstance(price_r2, float) and (price_r2 != price_r2)):
        cls = "good" if price_r2 > 0.7 else ("warn" if price_r2 > 0.4 else "bad")
        metrics.append(html.Div([
            html.Span("Price R²", className="vm-label"),
            html.Span(f"{price_r2:.3f}", className=f"vm-value {cls}"),
        ], className="validation-metric"))

    rent_rmse = rent.get("rmse")
    if rent_rmse is not None:
        cls = "good" if rent_rmse < 200 else ("warn" if rent_rmse < 500 else "bad")
        metrics.append(html.Div([
            html.Span("Rent RMSE", className="vm-label"),
            html.Span(f"${rent_rmse:,.0f}", className=f"vm-value {cls}"),
        ], className="validation-metric"))

    regime_acc = regime.get("accuracy")
    if regime_acc is not None:
        cls = "good" if regime_acc > 0.7 else ("warn" if regime_acc > 0.5 else "bad")
        metrics.append(html.Div([
            html.Span("Regime Acc.", className="vm-label"),
            html.Span(f"{regime_acc:.0%}", className=f"vm-value {cls}"),
        ], className="validation-metric"))

    h0_drift = stability.get("h0_drift_mean")
    if h0_drift is not None:
        metrics.append(html.Div([
            html.Span("H0 Drift", className="vm-label"),
            html.Span(f"{h0_drift:.4f}", className="vm-value good"),
        ], className="validation-metric"))

    h1_drift = stability.get("h1_drift_mean")
    if h1_drift is not None:
        metrics.append(html.Div([
            html.Span("H1 Drift", className="vm-label"),
            html.Span(f"{h1_drift:.4f}", className="vm-value good"),
        ], className="validation-metric"))

    # Build the 2x2 chart grid
    charts = html.Div(
        style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "12px"},
        children=[
            dcc.Graph(figure=make_validation_price_fig(price), className="graph-frame", style={"minHeight": "38vh"}),
            dcc.Graph(figure=make_validation_residual_fig(price), className="graph-frame", style={"minHeight": "38vh"}),
            dcc.Graph(figure=make_validation_zip_fig(zip_errors), className="graph-frame", style={"minHeight": "38vh"}),
            dcc.Graph(figure=make_validation_regime_fig(regime), className="graph-frame", style={"minHeight": "38vh"}),
        ],
    )

    children = []
    if note is not None:
        children.append(note)
    if metrics:
        children.append(html.Div(metrics, className="validation-metrics"))
    children.extend(charts)
    children.append(
        html.P(
            f"Validation mode: {val_mode} | Data mode: {data_mode} | "
            f"Months compared: {price.get('n_months', 0)}",
            style={"color": "#6e8a81", "fontSize": "12px", "marginTop": "8px"},
        )
    )

    return html.Div(children=_with_provenance("tab-validation", *children))


# ── Callbacks ─────────────────────────────────────────────────────────────


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


@app.callback(
    Output("report-status", "children"),
    Input("report-btn", "n_clicks"),
    prevent_initial_call=True,
)
def generate_report_callback(n_clicks: int):
    """Generate a report and return the file path."""
    try:
        from scripts.generate_report import build_report
        path = build_report()
        return f"Report saved: {path.name}"
    except Exception as exc:
        return f"Report failed: {exc}"


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8050)
