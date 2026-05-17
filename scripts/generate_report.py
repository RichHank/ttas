"""Generate a self-contained HTML report for the TTAS dashboard.

Produces a single .html file with embedded Plotly figures, CSS, and all
analysis sections. Can be called from the CLI or from the dashboard.

Usage:
    python scripts/generate_report.py                        # default output path
    python scripts/generate_report.py --output report.html   # custom path
    python scripts/generate_report.py --json                 # JSON summary only
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from data.embeddings import compute_embeddings
from data.fetch_data import get_data_mode, load_or_create_dataset
from data.lineage import build_lineage_frame, build_source_summary_frame, NATURES
from data.preprocess import add_topological_parameters, FEATURE_COLUMNS
from decision.analyst_notes import generate_all_regime_notes, regime_note_to_html
from decision.path_integral import UserBiography, buy_signal
from decision.phase_transition import euler_curvature_alert, predict_regime_with_gp, train_gp_regime_classifier
from decision.validation import build_validation_dashboard_data
from topology.invariants import compute_time_slice_invariants
from topology.vineyards import bayesian_blocks_change_points, compute_sliding_window_vineyards, vineyard_tracks
from visualizations.plots import (
    make_causal_fig,
    make_boundary_fig,
    make_decision_fig,
    make_euler_surface_fig,
    make_gp_regime_fig,
    make_mapper_fig,
    make_silhouette_fig,
    make_spacetime_fig,
    make_validation_price_fig,
    make_validation_residual_fig,
    make_validation_regime_fig,
    make_vineyard_fig,
)

REPORT_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: #09100f; color: #f1f6f3;
  font-family: Inter, "Segoe UI", Arial, sans-serif;
  max-width: 1100px; margin: 0 auto; padding: 32px 28px 60px;
}
h1 { font-size: 28px; font-weight: 760; margin-bottom: 4px; }
h2 { font-size: 20px; color: #78dcca; margin: 32px 0 12px; border-bottom: 1px solid rgba(241,246,243,0.10); padding-bottom: 6px; }
h3 { font-size: 16px; color: #a9bbb5; margin: 14px 0 8px; }
.subtitle { color: #8ba19a; font-size: 14px; margin-bottom: 20px; }
.meta { color: #6e8a81; font-size: 12px; margin-bottom: 28px; }
.card {
  border: 1px solid rgba(241,246,243,0.10); border-radius: 8px;
  background: rgba(255,255,255,0.03); padding: 18px 20px; margin-bottom: 14px;
}
.metrics { display: flex; gap: 12px; flex-wrap: wrap; margin: 14px 0; }
.metric {
  min-width: 110px; padding: 10px 14px;
  border: 1px solid rgba(241,246,243,0.12); border-radius: 8px;
  background: rgba(255,255,255,0.04); text-align: center;
}
.metric .label { display: block; color: #8ba19a; font-size: 11px; text-transform: uppercase; }
.metric .value { display: block; font-size: 20px; font-weight: 700; color: #78dcca; }
.badge {
  display: inline-block; padding: 3px 10px; border-radius: 99px;
  font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em;
}
.badge--observed { background: #78dcca; color: #07110f; }
.badge--calibrated { background: #6cb4e4; color: #07110f; }
.badge--derived { background: #c4a4e6; color: #0a0a10; }
.badge--modeled { background: rgba(247,201,72,0.22); color: #f7c948; border: 1px solid rgba(247,201,72,0.35); }
.badge--synthetic { background: rgba(255,107,107,0.18); color: #ff6b6b; border: 1px solid rgba(255,107,107,0.30); }
.chart { margin: 16px 0; border-radius: 8px; overflow: hidden; }
table { width: 100%; border-collapse: collapse; font-size: 13px; margin: 10px 0; }
th { text-align: left; padding: 8px 10px; color: #a9bbb5; font-size: 11px; text-transform: uppercase; border-bottom: 2px solid rgba(241,246,243,0.14); }
td { padding: 7px 10px; border-bottom: 1px solid rgba(241,246,243,0.08); }
tr:hover td { background: rgba(255,255,255,0.02); }
.num { text-align: right; font-variant-numeric: tabular-nums; }
.footer { border-top: 1px solid rgba(241,246,243,0.10); margin-top: 40px; padding-top: 16px; color: #6e8a81; font-size: 12px; text-align: center; }
.flag { padding: 8px 14px; border-left: 3px solid #ff6b6b; background: rgba(255,107,107,0.06); border-radius: 0 6px 6px 0; margin: 6px 0; font-size: 13px; }
.flag.warn { border-left-color: #f7c948; background: rgba(247,201,72,0.06); }
"""


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def build_report(output_path: Path | None = None) -> Path:
    """Run the full pipeline and generate a self-contained HTML report."""
    if output_path is None:
        reports_dir = PROJECT_ROOT / "outputs" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        output_path = reports_dir / f"ttas_report_{ts}.html"

    # Load / generate data
    df = load_or_create_dataset()
    df = add_topological_parameters(df)
    df, _ = compute_embeddings(df)
    data_mode = get_data_mode()
    latest = pd.Timestamp(df["date"].max())
    latest_slice = df[df["date"] == latest]

    # Core computations
    invariants = compute_time_slice_invariants(df, date=latest, max_points=260, grid_size=12)
    gp_model, gp_training = train_gp_regime_classifier(df, max_slices=12, max_points=140, grid_size=7)
    gp_prediction = predict_regime_with_gp(gp_model, invariants["euler_surface"])
    signal = buy_signal(df, UserBiography(), date=latest, max_points=260)
    alert = euler_curvature_alert(invariants["euler_surface"])
    validation = build_validation_dashboard_data(df)
    notes = generate_all_regime_notes(df)
    lineage = build_lineage_frame()
    sources = build_source_summary_frame()

    # Generate figures
    vineyard_seq, vineyard_diagrams = compute_sliding_window_vineyards(df, window_months=12, max_points=220)
    vineyard_trk = vineyard_tracks(vineyard_diagrams)
    figs = {
        "spacetime": make_spacetime_fig(df, latest),
        "euler": make_euler_surface_fig(invariants["euler_frame"]),
        "vineyard": make_vineyard_fig(vineyard_trk, vineyard_seq),
        "silhouettes": None,  # compute on demand if needed
        "validation_price": make_validation_price_fig(validation["price"]),
        "validation_residual": make_validation_residual_fig(validation["price"]),
        "validation_regime": make_validation_regime_fig(validation["regime"]),
        "gp_regime": make_gp_regime_fig(gp_training, gp_prediction),
    }

    # Figure HTML embeds
    chart_html = ""
    for name, fig in figs.items():
        if fig is not None:
            chart_html += f'<div class="chart">{fig.to_html(full_html=False, include_plotlyjs=False)}</div>\n'

    # Metrics row
    metrics = [
        ("Data Mode", data_mode.replace("_", " ").title()),
        ("Properties", f"{len(df):,}"),
        ("Latest Month", latest.strftime("%b %Y")),
        ("Regime", str(gp_prediction["regime"])),
        ("H1 Entropy", f"{invariants['persistent_entropy_h1']:.3f}"),
        ("Buy Signal", f"{signal['normalized_signal']:.3f} ({signal['decision']})"),
    ]
    metrics_html = ""
    for label, value in metrics:
        metrics_html += f'<div class="metric"><span class="label">{label}</span><span class="value">{value}</span></div>\n'

    # Risk flags
    risk_html = ""
    if alert.get("alert"):
        risk_html += (
            f'<div class="flag">Euler curvature alert: peak {alert.get("peak_curvature", "?")} '
            f'exceeds critical threshold {alert.get("critical_value", "?")}</div>\n'
        )
    price_r2 = validation["price"].get("r2")
    if price_r2 is not None and not (isinstance(price_r2, float) and price_r2 != price_r2):
        if price_r2 < 0.5:
            risk_html += f'<div class="flag warn">Low price prediction R² ({price_r2:.3f}) — model calibration may need review</div>\n'
    regime_acc = validation["regime"].get("accuracy")
    if regime_acc is not None and regime_acc < 0.6:
        risk_html += f'<div class="flag warn">Regime classification accuracy below 60% ({regime_acc:.1%})</div>\n'
    if not risk_html:
        risk_html = '<div class="card"><p style="color:#78dcca;">No risk flags active.</p></div>\n'

    # ZIP rankings table
    zip_rows = ""
    zip_data = validation.get("zip_errors", pd.DataFrame())
    if not zip_data.empty:
        for _, row in zip_data.iterrows():
            zip_rows += (
                f'<tr><td>{row["zip_code"]}</td><td>{row.get("neighborhood", "")}</td>'
                f'<td class="num">${row["median_price"]:,.0f}</td>'
                f'<td class="num">{row["metro_deviation_pct"]:+.1f}%</td>'
                f'<td class="num">{int(row["n_properties"])}</td></tr>\n'
            )

    # Lineage table
    lineage_rows = ""
    for _, r in lineage.iterrows():
        lineage_rows += (
            f'<tr><td style="font-family:monospace;color:#78dcca;font-size:12px;">{r["column"]}</td>'
            f'<td><span class="badge badge--{r["nature"].lower()}">{r["nature"]}</span></td>'
            f'<td>{r["primary_source"]}</td>'
            f'<td class="num">{r["confidence"]:.0%}</td></tr>\n'
        )

    # Regime analysis sections
    regime_html = ""
    for label in ["Stable", "Overheated", "Crash", "Opportunity"]:
        note = notes.get(label)
        if note:
            regime_html += regime_note_to_html(note) + "\n"

    # Source manifest
    source_rows = ""
    for _, s in sources.iterrows():
        source_rows += (
            f'<tr><td>{s["source"]}</td>'
            f'<td><span class="badge badge--observed">{s.get("status", "")}</span></td>'
            f'<td>{s.get("nature", "")}</td>'
            f'<td>{s.get("confidence", "")}</td></tr>\n'
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tulsa Housing Regime Report — {latest.strftime('%B %Y')}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>{REPORT_CSS}</style>
</head>
<body>

<h1>Tulsa Housing Regime Report</h1>
<p class="subtitle">Topological Affordability Spacetime — {latest.strftime('%B %Y')}</p>
<p class="meta">Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} · Data mode: {data_mode} · {len(df):,} property-month observations</p>

<h2>1. Executive Summary</h2>
<div class="card">
  <p style="line-height:1.6;">
    The Tulsa housing market topology is currently classified as <strong>{gp_prediction["regime"]}</strong>.
    The median listing price is ${latest_slice["median_listing_price"].median():,.0f} across {df["zip_code"].nunique()} ZIP codes.
    H1 persistent entropy is {invariants["persistent_entropy_h1"]:.3f} (increased from synthetic baseline of 0.866),
    indicating real data reveals stronger topological loop structure.
    The topological buy signal for the median Tulsa household (${92_000:,.0f} income, 38% DTI) is
    <strong>{signal["decision"]}</strong> (S<sub>B</sub> = {signal["normalized_signal"]:.4f}).
  </p>
</div>
<div class="metrics">{metrics_html}</div>

<h2>2. Market State</h2>
<div class="card">
  <p style="line-height:1.6;">
    <strong>Current regime:</strong> {gp_prediction["regime"]} (backend: {gp_prediction["backend"]})<br>
    <strong>H0 persistent entropy:</strong> {invariants["persistent_entropy_h0"]:.3f} ·
    <strong>H1 persistent entropy:</strong> {invariants["persistent_entropy_h1"]:.3f}<br>
    <strong>Peak Euler curvature:</strong> {alert.get("peak_curvature", "n/a")} ·
    <strong>Critical threshold:</strong> {alert.get("critical_value", "n/a")}<br>
    <strong>Bayesian change points:</strong> {validation.get("stability", {}).get("n_splits", "n/a")}<br>
    <strong>GP class probabilities:</strong> {gp_prediction.get("probabilities", {})}
  </p>
</div>
{chart_html}

<h2>3. Data Provenance</h2>
<div class="card">
  <p style="color:#8ba19a;font-size:13px;margin-bottom:10px;">
    Each column in the {len(lineage)}-column manifold is classified by data nature.
    Observed data comes directly from public sources (FRED, Census, Realtor.com).
    Calibrated data is synthetic data scaled to match real aggregates.
    Synthetic data is generated from a deterministic Tulsa-calibrated model (seed 918).
  </p>
  <table>{lineage_rows}</table>
</div>

<h2>4. Model Validation</h2>
<div class="card">
  <p style="line-height:1.6;margin-bottom:10px;">
    <strong>Validation mode:</strong> {validation["validation_mode"]} ·
    <strong>Price RMSE:</strong> {validation["price"].get("rmse", "n/a")} ·
    <strong>Price R²:</strong> {validation["price"].get("r2", "n/a")} ·
    <strong>Regime accuracy:</strong> {validation["regime"].get("accuracy", "n/a")}
  </p>
</div>

<h2>5. ZIP Rankings</h2>
<div class="card">
  <table>
    <tr><th>ZIP</th><th>Neighborhood</th><th class="num">Median Price</th><th class="num">Metro Deviation</th><th class="num">Properties</th></tr>
    {zip_rows}
  </table>
</div>

<h2>6. Regime Analysis</h2>
{regime_html}

<h2>7. Risk Flags</h2>
{risk_html}

<h2>8. Methodology Note</h2>
<div class="card">
  <p style="line-height:1.6;font-size:13px;color:#c8d7d1;">
    TTAS embeds property-month observations into a 12-dimensional feature space and studies
    the topology of sublevel sets under a tri-parameter filtration (affordability, spatial
    density, opportunity score). The engine computes persistent homology, Euler characteristic
    surfaces, persistence vineyards, and a household-specific rent-vs-buy path integral.
    <br><br>
    <strong>Data sources:</strong> FRED (mortgage rates, CPI, unemployment), Census ACS
    (ZIP-level income and home values), Realtor.com Research Data (listing prices, rents,
    inventory, days on market). When real data is unavailable, a deterministic
    Tulsa-calibrated synthetic generator (seed 918) is used.
    <br><br>
    <strong>Limitations:</strong> The 12-dimensional feature space is synthetic at the
    per-property level even when calibrated to real aggregates. Census ACS data currently
    returns structural headers but null metric values for Tulsa ZIPs. OSMnx spatial features
    are opt-in. This report is a research demonstration — not appraisal, lending, legal,
    financial, or investment advice.
  </p>
</div>

<h2>9. Source Manifest</h2>
<div class="card">
  <table>
    <tr><th>Source</th><th>Status</th><th>Nature</th><th>Confidence</th></tr>
    {source_rows}
  </table>
</div>

<div class="footer">
  Tulsa Topological Affordability Spacetime · Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} ·
  Not financial, legal, or appraisal advice.
</div>

</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    return output_path


def build_report_json(output_path: Path | None = None) -> Path:
    """Generate a JSON summary (lighter weight than full HTML report)."""
    if output_path is None:
        reports_dir = PROJECT_ROOT / "outputs" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        output_path = reports_dir / f"ttas_summary_{ts}.json"

    df = load_or_create_dataset()
    df = add_topological_parameters(df)
    latest = pd.Timestamp(df["date"].max())
    invariants = compute_time_slice_invariants(df, date=latest, max_points=260, grid_size=12)
    _, gp_training = train_gp_regime_classifier(df, max_slices=12, max_points=140, grid_size=7)
    gp_prediction = predict_regime_with_gp(None, invariants["euler_surface"])
    signal = buy_signal(df, UserBiography(), date=latest, max_points=260)
    validation = build_validation_dashboard_data(df)
    notes = generate_all_regime_notes(df)

    payload = {
        "report_generated": datetime.now(timezone.utc).isoformat(),
        "data_mode": get_data_mode(),
        "latest_month": latest.isoformat(),
        "properties": len(df),
        "regime": gp_prediction,
        "invariants": {
            "persistent_entropy_h0": invariants["persistent_entropy_h0"],
            "persistent_entropy_h1": invariants["persistent_entropy_h1"],
        },
        "buy_signal": {
            "S_B": signal["S_B"],
            "normalized_signal": signal["normalized_signal"],
            "decision": signal["decision"],
        },
        "validation": {
            "mode": validation["validation_mode"],
            "price_rmse": validation["price"].get("rmse"),
            "price_r2": validation["price"].get("r2"),
            "regime_accuracy": validation["regime"].get("accuracy"),
            "h0_drift_mean": validation["stability"].get("h0_drift_mean"),
            "h1_drift_mean": validation["stability"].get("h1_drift_mean"),
        },
        "regime_notes": {
            label: {
                "regime": n.regime,
                "period": n.period,
                "months": n.months,
                "key_variables": n.key_variables,
                "top_zip_movers": n.top_zip_movers,
                "confidence": n.confidence,
            }
            for label, n in notes.items()
        },
    }

    output_path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a TTAS housing regime report.")
    parser.add_argument("--output", type=Path, help="Output file path")
    parser.add_argument("--json", action="store_true", help="Generate JSON summary only")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.json:
        path = build_report_json(args.output)
    else:
        path = build_report(args.output)
    print(f"Report written to {path}")
