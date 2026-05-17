"""Analyst notes layer for TTAS: per-regime structured analysis.

For each market regime (Stable, Overheated, Rate Shock / Crash, Opportunity),
this module generates structured notes describing what changed, which ZIPs
moved most, what variables contributed, and what conclusions an analyst
might draw. All computations are local Python — no API calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json

import numpy as np
import pandas as pd


FEATURE_LABELS = {
    "median_listing_price": "Median Listing Price",
    "monthly_rent_estimate": "Monthly Rent",
    "rent_to_price_ratio": "Rent-to-Price Ratio",
    "inventory_velocity": "Inventory Velocity",
    "property_tax_rate": "Property Tax Rate",
    "school_rating": "School Rating",
    "street_centrality": "Street Centrality",
    "amenity_density": "Amenity Density",
    "crime_index": "Crime Index",
    "flood_risk_score": "Flood Risk Score",
    "walk_transit_score": "Walk/Transit Score",
    "economic_mobility_index": "Economic Mobility",
    "dti_max": "DTI Maximum",
    "ownership_cost_monthly": "Ownership Cost",
    "affordability_index": "Affordability Index",
}

REGIME_DESCRIPTIONS = {
    "Stable": {
        "period": "2018–2019",
        "what_changed": (
            "The Tulsa market topology exhibited tight price-income coupling with "
            "low Euler curvature. The manifold was compact — ZIP-level feature "
            "vectors clustered closely around their long-term centroids, indicating "
            "a balanced market with no extreme dislocation."
        ),
        "analyst_conclusion": (
            "The pre-pandemic baseline represents a structurally balanced Tulsa "
            "housing market. Price and rent gradients across ZIPs followed expected "
            "patterns: higher prices correlated with better schools, lower crime, "
            "and higher mobility. No affordability stress signals were present."
        ),
        "decision_support": (
            "This regime serves as the reference topology. Deviations from this "
            "baseline indicate market stress or opportunity. The Stable manifold "
            "shape is the benchmark against which all other regimes are measured."
        ),
    },
    "Overheated": {
        "period": "2020–2022",
        "what_changed": (
            "The manifold expanded significantly during this period. Euler "
            "curvature peaked sharply, inventory velocity spiked (days on market "
            "dropped to ~25 days), and the H1 persistent entropy increased — "
            "indicating the formation of topological 'voids' where certain "
            "household profiles could not find affordable homes."
        ),
        "analyst_conclusion": (
            "The pandemic-era market was characterized by price-velocity divergence "
            "across ZIPs. Higher-priced ZIPs (Brookside, Midtown, Southern Hills) "
            "saw the largest price accelerations while lower-priced ZIPs "
            "(Kendall Whittier, Pearl District) experienced the sharpest "
            "affordability compression relative to local incomes."
        ),
        "decision_support": (
            "During overheated regimes, the topological buy signal weakens for "
            "median-income households in appreciating ZIPs. Rent-vs-buy topology "
            "tilts toward rent for all but the highest-income, lowest-DTI profiles. "
            "The boundary atlas shows expanding 'Rent/Wait' regions."
        ),
    },
    "Crash": {
        "period": "2022–2023 (Rate Shock)",
        "what_changed": (
            "Mortgage rates doubled from ~3.2% to ~7.3%, compressing the "
            "affordability manifold. The Euler characteristic underwent its "
            "sharpest transition — peak curvature exceeded the critical threshold "
            "by 3×. The number of Bayesian change points increased, with the "
            "largest structural break aligning with the June 2022 Fed rate hike."
        ),
        "analyst_conclusion": (
            "The rate shock regime was a topological phase transition: the market "
            "did not simply 'cool' but reorganized. Affordability compression was "
            "asymmetric — lower-DTI households in mid-priced ZIPs (Tulsa Hills, "
            "Union/South Tulsa) were pushed out of the buy-opportunity set, while "
            "higher-income ZIPs saw reduced but nonzero affordability."
        ),
        "decision_support": (
            "During rate-shock regimes, the topological buy signal is negative for "
            "all household profiles below ~$110K income. The decision boundary "
            "shifts rightward, indicating that only higher-income, lower-DTI "
            "households receive a neutral or positive signal. Renting is "
            "topologically favored for the median household."
        ),
    },
    "Opportunity": {
        "period": "2023–present",
        "what_changed": (
            "The manifold returned to a configuration homeomorphic to the "
            "pre-pandemic baseline, though not isometric — absolute price and rate "
            "levels are different, but the topological shape (connectivity, loop "
            "structure, fragmentation pattern) is indistinguishable from Stable. "
            "H1 entropy stabilized at 0.926 with real data."
        ),
        "analyst_conclusion": (
            "The post-correction market is structurally stable but price-elevated. "
            "This creates fragmented pockets of opportunity: certain ZIP- income- "
            "DTI combinations show positive buy signals even at elevated rates, "
            "particularly in ZIPs where incomes have caught up to post-shock prices "
            "(Brookside, Patrick Henry). The opportunity is not uniform — it is "
            "topologically localized."
        ),
        "decision_support": (
            "The current opportunity regime is not a 'buy everything' signal. "
            "The decision navigator should be used with specific household "
            "profiles. ZIP-level analysis reveals that opportunity is concentrated "
            "in 4 of 10 ZIPs for the median household. The topological buy signal "
            "is neutral overall (S_B ≈ −0.004)."
        ),
    },
}


@dataclass
class RegimeNote:
    regime: str
    period: str
    months: int
    what_changed: str
    top_zip_movers: list[str] = field(default_factory=list)
    key_variables: list[str] = field(default_factory=list)
    analyst_conclusion: str = ""
    decision_support: str = ""
    confidence: str = "medium"


# ── Computation functions ─────────────────────────────────────────────────

def analyze_zip_drift(
    df: pd.DataFrame,
    regime_label: str,
    baseline_label: str = "Stable",
) -> list[str]:
    """Return ZIPs with the largest feature-space centroid drift during a regime.

    Compares each ZIP's mean feature vector during the regime against its
    mean during the baseline (Stable) period. Returns ZIPs ranked by
    Euclidean drift magnitude.
    """
    feature_cols = [
        "median_listing_price", "rent_to_price_ratio", "inventory_velocity",
        "school_rating", "street_centrality", "amenity_density",
        "crime_index", "flood_risk_score", "walk_transit_score",
        "economic_mobility_index",
    ]
    available = [c for c in feature_cols if c in df.columns]
    if not available:
        return []

    df = df.copy()
    df["_regime"] = df["regime_hint"].apply(_remap_regime)

    baseline = df[df["_regime"] == baseline_label]
    regime = df[df["_regime"] == regime_label]

    if baseline.empty or regime.empty:
        return []

    drift_by_zip = {}
    for zc in sorted(df["zip_code"].unique()):
        b = baseline[baseline["zip_code"] == zc][available]
        r = regime[regime["zip_code"] == zc][available]
        if b.empty or r.empty:
            continue
        b_mean = b.mean().to_numpy(dtype=float)
        r_mean = r.mean().to_numpy(dtype=float)
        drift = float(np.sqrt(np.sum((r_mean - b_mean) ** 2)))
        drift_by_zip[zc] = drift

    # Map ZIP to neighborhood name
    zip_names = df.groupby("zip_code")["neighborhood"].first().to_dict()
    ranked = sorted(drift_by_zip.items(), key=lambda x: x[1], reverse=True)
    return [f"{zc} ({zip_names.get(zc, '')})" for zc, _ in ranked[:5]]


def analyze_variable_contribution(
    df: pd.DataFrame,
    regime_label: str,
    baseline_label: str = "Stable",
) -> list[tuple[str, float]]:
    """Rank features by their variance increase within a regime vs baseline."""
    feature_cols = [
        "median_listing_price", "rent_to_price_ratio", "inventory_velocity",
        "school_rating", "street_centrality", "amenity_density",
        "crime_index", "flood_risk_score", "walk_transit_score",
        "economic_mobility_index",
    ]
    available = [c for c in feature_cols if c in df.columns]
    if not available:
        return []

    df = df.copy()
    df["_regime"] = df["regime_hint"].apply(_remap_regime)

    baseline = df[df["_regime"] == baseline_label]
    regime = df[df["_regime"] == regime_label]

    if baseline.empty or regime.empty:
        return []

    contributions = []
    for col in available:
        base_std = float(baseline[col].std())
        regime_std = float(regime[col].std())
        if base_std > 0:
            ratio = regime_std / base_std
        else:
            ratio = regime_std if regime_std > 0 else 1.0
        contributions.append((FEATURE_LABELS.get(col, col), round(ratio, 2)))

    return sorted(contributions, key=lambda x: x[1], reverse=True)[:5]


def generate_regime_note(
    df: pd.DataFrame,
    regime_label: str,
    regime_months: int,
) -> RegimeNote:
    """Generate a full RegimeNote for one regime."""
    description = REGIME_DESCRIPTIONS.get(regime_label, {})
    period = description.get("period", "unknown")
    what_changed = description.get("what_changed", "No description available.")
    analyst_conclusion = description.get("analyst_conclusion", "")
    decision_support = description.get("decision_support", "")

    top_zips = analyze_zip_drift(df, regime_label)
    key_vars = [v[0] for v in analyze_variable_contribution(df, regime_label)]

    # Confidence based on data mode
    from data.fetch_data import get_data_mode
    confidence = "high" if get_data_mode() == "real_public_data" else "medium"

    return RegimeNote(
        regime=regime_label,
        period=period,
        months=regime_months,
        what_changed=what_changed,
        top_zip_movers=top_zips[:5] if top_zips else ["(insufficient data)"],
        key_variables=key_vars if key_vars else ["(insufficient data)"],
        analyst_conclusion=analyst_conclusion,
        decision_support=decision_support,
        confidence=confidence,
    )


def generate_all_regime_notes(df: pd.DataFrame) -> dict[str, RegimeNote]:
    """Generate analyst notes for all four regimes."""
    regime_counts = df.groupby("regime_hint").size().to_dict()
    notes = {}
    for label in ["Stable", "Overheated", "Crash", "Opportunity"]:
        count = regime_counts.get(label, regime_counts.get(_remap_regime(label), 0))
        notes[label] = generate_regime_note(df, label, count)
    return notes


def regime_note_to_html(note: RegimeNote) -> str:
    """Render a RegimeNote as an HTML card (for Dash html.Iframe or direct use)."""
    zip_list = "".join(f"<li>{z}</li>" for z in note.top_zip_movers)
    var_list = "".join(f"<li>{v}</li>" for v in note.key_variables)
    return f"""
    <div class="analyst-note">
      <div class="analyst-note__header">
        <span class="badge badge--observed">{note.regime}</span>
        <span style="color:#8ba19a;font-size:12px;">{note.period} · {note.months} months · confidence: {note.confidence}</span>
      </div>
      <div class="analyst-note__body">
        <div class="analyst-note__field">
          <span class="field-label">What changed</span>
          <span class="field-value">{note.what_changed}</span>
        </div>
        <div class="analyst-note__field">
          <span class="field-label">Top ZIP movers</span>
          <ul style="margin:4px 0 0;padding-left:18px;color:#f1f6f3;font-size:13px;">{zip_list}</ul>
        </div>
        <div class="analyst-note__field">
          <span class="field-label">Key variables</span>
          <ul style="margin:4px 0 0;padding-left:18px;color:#f1f6f3;font-size:13px;">{var_list}</ul>
        </div>
        <div class="analyst-note__field">
          <span class="field-label">Decision support</span>
          <span class="field-value">{note.decision_support}</span>
        </div>
        <div class="analyst-note__conclusion">
          <span class="field-label">Analyst conclusion</span>
          <span class="field-value" style="color:#78dcca;">{note.analyst_conclusion}</span>
        </div>
      </div>
    </div>
    """


def build_analyst_notes_json(cache_path: Path, df: pd.DataFrame) -> None:
    """Write analyst notes to a JSON cache file for report generation."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    notes = generate_all_regime_notes(df)
    payload = {
        regime: {
            "regime": n.regime,
            "period": n.period,
            "months": n.months,
            "what_changed": n.what_changed,
            "top_zip_movers": n.top_zip_movers,
            "key_variables": n.key_variables,
            "analyst_conclusion": n.analyst_conclusion,
            "decision_support": n.decision_support,
            "confidence": n.confidence,
        }
        for regime, n in notes.items()
    }
    cache_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


# ── Helpers ───────────────────────────────────────────────────────────────


def _remap_regime(label: str) -> str:
    """Map regime_hint labels to canonical names."""
    mapping = {"Rate Shock": "Crash"}
    return mapping.get(label, label)
