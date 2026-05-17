"""Per-column data lineage, provenance tracking, and nature classification.

Each column in the TTAS manifold is classified into one of five natures:

    Observed    — directly from an external API or file (e.g., FRED mortgage rate)
    Calibrated  — synthetic data scaled to match real aggregate observations
    Derived     — computed deterministically from observed or calibrated columns
    Modeled     — estimated via a statistical or ML model
    Synthetic   — generated from a parametric random process (no real input)

This module provides per-column classification, a build function for the
provenance DataFrame, and helpers consumed by the dashboard, badges, and reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json

import pandas as pd


# ── Per-column provenance ────────────────────────────────────────────────

NATURES = ("Observed", "Calibrated", "Derived", "Modeled", "Synthetic")

COLUMN_LINEAGE: dict[str, dict] = {
    # ── Observed (directly from external source) ─────────────────────
    "mortgage_rate_30y": {
        "nature": "Observed",
        "primary_source": "FRED MORTGAGE30US",
        "endpoint": "fred.stlouisfed.org — series MORTGAGE30US",
        "date_pulled": "2026-05-14",
        "fields_used": ["30-year fixed mortgage rate (weekly, resampled to monthly)"],
        "transformations": ["api_fetch", "monthly_resample", "interpolate_gaps"],
        "confidence": 0.95,
    },
    "cpi_all_urban": {
        "nature": "Observed",
        "primary_source": "FRED CPIAUCSL",
        "endpoint": "fred.stlouisfed.org — series CPIAUCSL",
        "date_pulled": "2026-05-14",
        "fields_used": ["Consumer Price Index for All Urban Consumers"],
        "transformations": ["api_fetch", "monthly_resample"],
        "confidence": 0.95,
    },
    "unemployment_rate": {
        "nature": "Observed",
        "primary_source": "FRED UNRATE",
        "endpoint": "fred.stlouisfed.org — series UNRATE",
        "date_pulled": "2026-05-14",
        "fields_used": ["Civilian Unemployment Rate"],
        "transformations": ["api_fetch", "monthly_resample"],
        "confidence": 0.95,
    },
    "median_sales_price_us": {
        "nature": "Observed",
        "primary_source": "FRED MSPUS",
        "endpoint": "fred.stlouisfed.org — series MSPUS",
        "date_pulled": "2026-05-14",
        "fields_used": ["Median Sales Price of Houses Sold for the United States"],
        "transformations": ["api_fetch", "quarterly_to_monthly_interpolation"],
        "confidence": 0.90,
    },
    "median_household_income": {
        "nature": "Observed",
        "primary_source": "Census ACS 5-Year Estimates (Table S1901)",
        "endpoint": "api.census.gov — ACS 5-Year, Tulsa County ZIPs",
        "date_pulled": "2026-05-14",
        "fields_used": ["Median household income in the past 12 months (ZIP-level)"],
        "transformations": ["api_fetch", "zip_code_join"],
        "confidence": 0.85,
    },
    "owner_occupied_share": {
        "nature": "Observed",
        "primary_source": "Census ACS 5-Year Estimates (Table S2502)",
        "endpoint": "api.census.gov — ACS 5-Year, Tulsa County ZIPs",
        "date_pulled": "2026-05-14",
        "fields_used": ["Owner-occupied housing units as share of total"],
        "transformations": ["api_fetch", "zip_code_join"],
        "confidence": 0.85,
    },
    "median_home_value": {
        "nature": "Observed",
        "primary_source": "Census ACS 5-Year Estimates (Table S2502)",
        "endpoint": "api.census.gov — ACS 5-Year, Tulsa County ZIPs",
        "date_pulled": "2026-05-14",
        "fields_used": ["Median home value (ZIP-level)"],
        "transformations": ["api_fetch", "zip_code_join"],
        "confidence": 0.85,
    },

    # ── Calibrated (synthetic scaled to real aggregates) ─────────────
    "median_listing_price": {
        "nature": "Calibrated",
        "primary_source": "Realtor.com Research Data",
        "endpoint": "realtor.com/research/data — Tulsa metro CSV download",
        "date_pulled": "2026-05-14",
        "fields_used": ["median_listing_price_mm (monthly, Tulsa metro)"],
        "transformations": [
            "synthetic_generation",
            "real_aggregate_calibration",
            "monthly_median_scaling_to_realtor_com",
        ],
        "confidence": 0.70,
    },
    "monthly_rent_estimate": {
        "nature": "Calibrated",
        "primary_source": "Realtor.com Research Data (rent) + synthetic rent-to-price model",
        "endpoint": "realtor.com/research/data — median_rent column",
        "date_pulled": "2026-05-14",
        "fields_used": ["median_rent (monthly, Tulsa metro)"],
        "transformations": [
            "synthetic_rent_to_price_ratio",
            "real_aggregate_calibration",
            "monthly_median_scaling",
        ],
        "confidence": 0.65,
    },
    "inventory_velocity": {
        "nature": "Calibrated",
        "primary_source": "Realtor.com Research Data (days on market)",
        "endpoint": "realtor.com/research/data — median_days_on_market column",
        "date_pulled": "2026-05-14",
        "fields_used": ["median_days_on_market (monthly, Tulsa metro)"],
        "transformations": [
            "synthetic_generation",
            "dom_to_velocity_conversion",
            "real_aggregate_calibration",
        ],
        "confidence": 0.70,
    },

    # ── Derived (computed from other columns) ────────────────────────
    "ownership_cost_monthly": {
        "nature": "Derived",
        "primary_source": "Computed from median_listing_price, mortgage_rate_30y, property_tax_rate",
        "endpoint": "local — standard annuity formula (80% LTV, 30-yr fixed)",
        "date_pulled": "n/a (recomputed per run)",
        "fields_used": ["median_listing_price", "mortgage_rate_30y", "property_tax_rate"],
        "transformations": ["mortgage_annuity_formula", "tax_and_insurance_estimate"],
        "confidence": 0.80,
    },
    "rent_margin_monthly": {
        "nature": "Derived",
        "primary_source": "Computed from annual_income_estimate, dti_max, monthly_rent_estimate",
        "endpoint": "local",
        "date_pulled": "n/a (recomputed per run)",
        "fields_used": ["annual_income_estimate", "dti_max", "monthly_rent_estimate"],
        "transformations": ["budget_less_rent"],
        "confidence": 0.75,
    },
    "buy_margin_monthly": {
        "nature": "Derived",
        "primary_source": "Computed from annual_income_estimate, dti_max, ownership_cost_monthly",
        "endpoint": "local",
        "date_pulled": "n/a (recomputed per run)",
        "fields_used": ["annual_income_estimate", "dti_max", "ownership_cost_monthly"],
        "transformations": ["budget_less_ownership_cost"],
        "confidence": 0.75,
    },
    "rent_to_price_ratio": {
        "nature": "Derived",
        "primary_source": "Computed from monthly_rent_estimate / median_listing_price",
        "endpoint": "local",
        "date_pulled": "n/a (recomputed per run)",
        "fields_used": ["monthly_rent_estimate", "median_listing_price"],
        "transformations": ["annualized_rent_to_price_ratio"],
        "confidence": 0.70,
    },
    "affordability_index": {
        "nature": "Derived",
        "primary_source": "Computed in preprocess.py — ownership_cost / max_affordable_payment",
        "endpoint": "local",
        "date_pulled": "n/a (recomputed per run)",
        "fields_used": ["ownership_cost_monthly", "annual_income_estimate", "dti_max"],
        "transformations": ["affordability_gap_normalization"],
        "confidence": 0.75,
    },

    # ── Modeled (statistically estimated) ────────────────────────────
    "rent_vs_buy": {
        "nature": "Modeled",
        "primary_source": "Decision rule from buy_margin, rent_margin, opportunity score",
        "endpoint": "local — threshold-based classifier",
        "date_pulled": "n/a (recomputed per run)",
        "fields_used": ["buy_margin_monthly", "rent_margin_monthly", "opportunity_score"],
        "transformations": ["threshold_rule", "buy_margin_gt_-0.10*rent", "opportunity_gt_0.42"],
        "confidence": 0.60,
    },
    "regime_hint": {
        "nature": "Modeled",
        "primary_source": "Deterministic regime classifier based on month number",
        "endpoint": "local — heuristic segmentation",
        "date_pulled": "n/a (recomputed per run)",
        "fields_used": ["date (month index)"],
        "transformations": ["month_index_threshold: <24 Stable, <57 Overheated, <72 Rate Shock, >=72 Opportunity"],
        "confidence": 0.55,
    },

    # ── Synthetic (purely generated, no real data) ───────────────────
    "school_rating": {
        "nature": "Synthetic",
        "primary_source": "Tulsa ZIP profiles (hardcoded anchors in fetch_data.py)",
        "endpoint": "local — neighborhood profile + micro-location noise",
        "date_pulled": "n/a (constant profile)",
        "fields_used": ["ZIP_code → school anchor value + N(0, 0.035) noise"],
        "transformations": ["profile_lookup", "micro_location_noise", "clip_0_1"],
        "confidence": 0.40,
    },
    "street_centrality": {
        "nature": "Synthetic",
        "primary_source": "Tulsa ZIP profiles (hardcoded anchors)",
        "endpoint": "local — profile + N(0, 0.045) noise (or OSMnx when available)",
        "date_pulled": "n/a",
        "fields_used": ["ZIP_code → centrality anchor + noise"],
        "transformations": ["profile_lookup", "noise", "clip_0_1"],
        "confidence": 0.40,
    },
    "amenity_density": {
        "nature": "Synthetic",
        "primary_source": "Tulsa ZIP profiles (hardcoded anchors)",
        "endpoint": "local — profile + seasonal + N(0, 0.04) noise (or OSMnx POI when available)",
        "date_pulled": "n/a",
        "fields_used": ["ZIP_code → amenity anchor + seasonal + noise"],
        "transformations": ["profile_lookup", "seasonal_adjustment", "noise", "clip_0_1"],
        "confidence": 0.40,
    },
    "crime_index": {
        "nature": "Synthetic",
        "primary_source": "Tulsa ZIP profiles (hardcoded anchors)",
        "endpoint": "local — profile − trend + N(0, 0.05) noise",
        "date_pulled": "n/a",
        "fields_used": ["ZIP_code → crime anchor − long_trend + noise"],
        "transformations": ["profile_lookup", "trend_adjustment", "noise", "clip_0_1"],
        "confidence": 0.35,
    },
    "flood_risk_score": {
        "nature": "Synthetic",
        "primary_source": "Tulsa ZIP profiles (hardcoded anchors)",
        "endpoint": "local — profile + seasonal cos + N(0, 0.025) noise",
        "date_pulled": "n/a",
        "fields_used": ["ZIP_code → flood anchor + seasonal + noise"],
        "transformations": ["profile_lookup", "seasonal_cos", "noise", "clip_0_1"],
        "confidence": 0.35,
    },
    "walk_transit_score": {
        "nature": "Synthetic",
        "primary_source": "Computed from synthetic walk and transit components",
        "endpoint": "local — 0.68*walk + 0.32*transit",
        "date_pulled": "n/a",
        "fields_used": ["walk (synthetic)", "transit (derived from walk + centrality)"],
        "transformations": ["weighted_combination", "clip_0_1"],
        "confidence": 0.35,
    },
    "economic_mobility_index": {
        "nature": "Synthetic",
        "primary_source": "Tulsa ZIP profiles (hardcoded anchors)",
        "endpoint": "local — profile + trend + N(0, 0.035) noise",
        "date_pulled": "n/a",
        "fields_used": ["ZIP_code → mobility anchor + long_trend + noise"],
        "transformations": ["profile_lookup", "trend_adjustment", "noise", "clip_0_1"],
        "confidence": 0.35,
    },
    "dti_max": {
        "nature": "Synthetic",
        "primary_source": "Beta distribution draw (a=2.2, b=3.0), scaled to [0.28, 0.47]",
        "endpoint": "local — random generator with fixed seed (918)",
        "date_pulled": "n/a",
        "fields_used": ["rng.beta(2.2, 3.0) scaled to DTI range"],
        "transformations": ["beta_sample", "scale_to_range", "clip"],
        "confidence": 0.30,
    },
    "annual_income_estimate": {
        "nature": "Calibrated",
        "primary_source": "Census ACS (ZIP median income) + synthetic per-property noise",
        "endpoint": "api.census.gov — calibrated per property",
        "date_pulled": "2026-05-14",
        "fields_used": ["ZIP median income (Census) + property-level N(1.0, 0.08) variation"],
        "transformations": ["synthetic_generation", "zip_income_calibration", "property_noise"],
        "confidence": 0.65,
    },
}


# ── Source-level summary ──────────────────────────────────────────────────

SOURCE_SUMMARIES = [
    {
        "source": "FRED (Federal Reserve Economic Data)",
        "api_endpoint": "api.stlouisfed.org/fred/series/observations",
        "date_pulled": "2026-05-14",
        "fields": ["mortgage_rate_30y", "cpi_all_urban", "unemployment_rate", "median_sales_price_us"],
        "transformations": "API fetch → monthly resample → interpolate gaps",
        "nature": "Observed",
        "confidence": "High (0.90–0.95)",
        "status": "active",
        "raw_path": "data/raw/fred/fred_monthly.csv",
        "notes": "953 rows, 1947–2026. Mortgage rate series is direct; CPI and unemployment are national-level context variables.",
    },
    {
        "source": "Census ACS 5-Year Estimates",
        "api_endpoint": "api.census.gov/data/2023/acs/acs5",
        "date_pulled": "2026-05-14",
        "fields": ["median_household_income", "owner_occupied_share", "median_home_value", "rent_burden_count"],
        "transformations": "API fetch → ZIP-level join to Tulsa County ZIPs",
        "nature": "Observed",
        "confidence": "Medium-High (0.85)",
        "status": "active (null values — structural fetch only)",
        "raw_path": "data/raw/census/census_zip_metrics.csv",
        "notes": "24 Tulsa ZIPs returned with null metric values. ACS 5-year estimates have 1-year lag; 2023 estimates reflect 2019–2023 data.",
    },
    {
        "source": "Realtor.com Research Data",
        "api_endpoint": "realtor.com/research/data (CSV download, not API)",
        "date_pulled": "2026-05-14",
        "fields": ["median_listing_price", "median_rent", "active_listing_count", "median_days_on_market"],
        "transformations": "CSV import → monthly aggregation → outer join with synthetic manifold",
        "nature": "Observed (aggregates) / Calibrated (per-property)",
        "confidence": "Medium-High (0.70–0.85)",
        "status": "active — 1 CSV imported, 119 monthly rows",
        "raw_path": "data/raw/realtor/realtor_monthly.csv",
        "notes": "Tulsa metro data, monthly. Listing prices and rents are observed; per-property values are calibrated synthetic. Data extends 2016–2026; pre-2016 values in timeseries are interpolated.",
    },
    {
        "source": "OSMnx Street Network + POI",
        "api_endpoint": "OpenStreetMap via osmnx Python library (on-demand)",
        "date_pulled": "n/a (opt-in)",
        "fields": ["street_centrality", "amenity_density"],
        "transformations": "Graph centrality (closeness) + POI kernel density around property coordinates",
        "nature": "Observed (when enabled) / Synthetic (default)",
        "confidence": "Medium (0.55) when enabled; Low (0.40) when synthetic",
        "status": "opt-in — requires --use-osmnx flag",
        "raw_path": "n/a (computed on-the-fly)",
        "notes": "When enabled, replaces synthetic centrality and amenity with real OSM data. Not enabled by default due to network dependency.",
    },
    {
        "source": "Tulsa County Assessor / City of Tulsa Open Data",
        "api_endpoint": "planned — not yet integrated",
        "date_pulled": "n/a",
        "fields": ["parcel_values", "property_class", "sale_dates"],
        "transformations": "planned: parcel-level join by address/coordinates",
        "nature": "Planned (Observed)",
        "confidence": "n/a",
        "status": "planned",
        "raw_path": "n/a",
        "notes": "Would provide parcel-level ground truth for property values and transaction dates. Currently not integrated.",
    },
]


# ── Public API ────────────────────────────────────────────────────────────


def build_lineage_frame() -> pd.DataFrame:
    """Convert COLUMN_LINEAGE into a sortable DataFrame for the dashboard."""
    rows = []
    for col, meta in COLUMN_LINEAGE.items():
        rows.append({
            "column": col,
            "nature": meta["nature"],
            "primary_source": meta["primary_source"],
            "endpoint": meta.get("endpoint", ""),
            "date_pulled": meta.get("date_pulled", ""),
            "fields_used": ", ".join(meta.get("fields_used", [])),
            "transformations": " → ".join(meta.get("transformations", [])),
            "confidence": meta.get("confidence", 0.0),
        })
    return pd.DataFrame(rows).sort_values(["nature", "confidence"], ascending=[True, False])


def get_data_nature(column: str) -> str:
    """Return the provenance nature for a single column, or 'Unknown'."""
    if column in COLUMN_LINEAGE:
        return COLUMN_LINEAGE[column]["nature"]
    return "Unknown"


def classify_columns_by_nature() -> dict[str, list[str]]:
    """Group all columns by their provenance nature."""
    groups: dict[str, list[str]] = {n: [] for n in NATURES}
    groups["Unknown"] = []
    for col, meta in COLUMN_LINEAGE.items():
        nature = meta["nature"]
        groups.setdefault(nature, []).append(col)
    return groups


def build_source_summary_frame() -> pd.DataFrame:
    """Convert SOURCE_SUMMARIES into a DataFrame for the dashboard."""
    return pd.DataFrame(SOURCE_SUMMARIES)


def get_natures_for_columns(columns: list[str]) -> set[str]:
    """Return the set of unique data natures present in a list of columns."""
    return {get_data_nature(c) for c in columns}


def nature_badge_html(nature: str) -> str:
    """Return an HTML span with the appropriate badge class for a nature."""
    css_class = {
        "Observed": "badge badge--observed",
        "Calibrated": "badge badge--calibrated",
        "Derived": "badge badge--derived",
        "Modeled": "badge badge--modeled",
        "Synthetic": "badge badge--synthetic",
    }.get(nature, "badge")
    return f'<span class="{css_class}">{nature}</span>'


def build_lineage_json(cache_path: Path) -> None:
    """Write lineage data to a JSON cache file for report generation."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "columns": COLUMN_LINEAGE,
        "sources": SOURCE_SUMMARIES,
    }
    cache_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
