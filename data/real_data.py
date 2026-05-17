"""Optional public-data enrichment for TTAS.

The default repository remains deterministic and synthetic. This module adds
real-data hooks for environments with API keys and geospatial dependencies:

* FRED `MORTGAGE30US` for weekly 30-year fixed mortgage rates;
* Census ACS for ZIP-level income, home values, and owner-occupied share;
* Realtor.com Research CSVs for listing price, rent, inventory, DOM;
* OSMnx street graph centrality and amenity density around Tulsa.

If a dependency, API key, or network call is unavailable, the function returns
the original dataframe plus explicit source notes instead of failing the
pipeline.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def get_data_mode() -> str:
    """Return the current data mode by reading source_manifest.json.

    Returns:
        "real_public_data" if at least one real source is available,
        "synthetic_fallback" otherwise.
    """
    manifest_path = PROCESSED_DIR / "source_manifest.json"
    if not manifest_path.exists():
        return "synthetic_fallback"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return manifest.get("mode", "synthetic_fallback")
    except Exception:
        return "synthetic_fallback"


def load_real_timeseries() -> pd.DataFrame | None:
    """Load the real-market monthly timeseries if available.

    Returns None when no real data has been built.
    """
    path = PROCESSED_DIR / "tulsa_market_timeseries.csv"
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, parse_dates=["date"])
    except Exception:
        return None


def load_real_zip_metrics() -> pd.DataFrame | None:
    """Load real ZIP-level census metrics if available."""
    path = PROCESSED_DIR / "tulsa_zip_metrics.json"
    if not path.exists():
        return None
    try:
        return pd.read_json(path)
    except Exception:
        return None


def load_source_manifest() -> dict | None:
    """Return the source manifest or None."""
    path = PROCESSED_DIR / "source_manifest.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def calibrate_synthetic_from_real(synthetic: pd.DataFrame) -> pd.DataFrame:
    """Calibrate synthetic data using real aggregate timeseries when available.

    Replaces synthetic mortgage rates with FRED data and adjusts median listing
    prices and rents to track real Realtor.com aggregate values.
    """
    real = load_real_timeseries()
    if real is None or real.empty:
        return synthetic

    df = synthetic.copy()
    real_indexed = real.set_index("date")

    # 1. Replace mortgage rate with FRED series if available
    if "mortgage_rate_30y" in real_indexed.columns:
        for date_val in df["date"].unique():
            if date_val in real_indexed.index:
                fred_rate = real_indexed.loc[date_val, "mortgage_rate_30y"]
                if pd.notna(fred_rate):
                    mask = df["date"] == date_val
                    df.loc[mask, "mortgage_rate_30y"] = float(fred_rate)

    # 2. Scale synthetic prices to track real median listing price
    if "median_listing_price" in real_indexed.columns:
        for date_val in df["date"].unique():
            if date_val in real_indexed.index:
                real_price = real_indexed.loc[date_val, "median_listing_price"]
                if pd.notna(real_price):
                    mask = df["date"] == date_val
                    syn_price = df.loc[mask, "median_listing_price"].median()
                    if syn_price > 0:
                        scale = float(real_price) / syn_price
                        df.loc[mask, "median_listing_price"] *= scale
                        df.loc[mask, "monthly_rent_estimate"] *= scale
                        df.loc[mask, "ownership_cost_monthly"] *= scale
                        df.loc[mask, "rent_margin_monthly"] *= scale
                        df.loc[mask, "buy_margin_monthly"] *= scale

    # 3. Calibrate rent with real median rent
    if "median_rent" in real_indexed.columns:
        for date_val in df["date"].unique():
            if date_val in real_indexed.index:
                real_rent = real_indexed.loc[date_val, "median_rent"]
                if pd.notna(real_rent):
                    mask = df["date"] == date_val
                    syn_rent = df.loc[mask, "monthly_rent_estimate"].median()
                    if syn_rent > 0:
                        scale = float(real_rent) / syn_rent
                        df.loc[mask, "monthly_rent_estimate"] *= scale

    # 4. Calibrate days on market
    if "median_days_on_market" in real_indexed.columns:
        for date_val in df["date"].unique():
            if date_val in real_indexed.index:
                real_dom = real_indexed.loc[date_val, "median_days_on_market"]
                if pd.notna(real_dom):
                    mask = df["date"] == date_val
                    median_syn_dom = df.loc[mask, "inventory_velocity"].apply(
                        lambda v: 30.0 / max(v, 0.001)
                    ).median()
                    if median_syn_dom > 0:
                        ratio = float(real_dom) / median_syn_dom
                        df.loc[mask, "inventory_velocity"] = (
                            df.loc[mask, "inventory_velocity"] / max(ratio, 0.001)
                        )

    # 5. Calibrate income from Census ACS if available (annual — apply to all rows)
    zip_metrics = load_real_zip_metrics()
    if zip_metrics is not None and not zip_metrics.empty:
        for _, row in zip_metrics.iterrows():
            zc = str(row.get("zip_code", ""))
            income = row.get("median_household_income")
            if pd.notna(income) and zc in df["zip_code"].values:
                mask = df["zip_code"] == zc
                syn_income = df.loc[mask, "annual_income_estimate"].median()
                if syn_income > 0:
                    scale = float(income) / syn_income
                    df.loc[mask, "annual_income_estimate"] *= scale

    return df


def apply_fred_mortgage_rates(df: pd.DataFrame, api_key: str | None = None) -> tuple[pd.DataFrame, str]:
    """Replace synthetic mortgage rates with FRED MORTGAGE30US if available."""

    api_key = api_key or os.getenv("FRED_API_KEY")
    if not api_key:
        return df, "FRED skipped: no FRED_API_KEY."
    try:
        from fredapi import Fred

        fred = Fred(api_key=api_key)
        series = fred.get_series("MORTGAGE30US")
        monthly = series.resample("MS").mean().interpolate().rename("fred_mortgage_rate_30y")
        working = df.copy()
        rates = pd.DataFrame({"date": pd.to_datetime(monthly.index), "fred_mortgage_rate_30y": monthly.to_numpy(dtype=float)})
        working = working.merge(rates, on="date", how="left")
        working["mortgage_rate_30y"] = working["fred_mortgage_rate_30y"].fillna(working["mortgage_rate_30y"])
        working = working.drop(columns=["fred_mortgage_rate_30y"])
        return working, "FRED MORTGAGE30US applied."
    except Exception as exc:
        return df, f"FRED skipped: {exc}"


def apply_osmnx_spatial_features(df: pd.DataFrame, place: str = "Tulsa, Oklahoma, USA") -> tuple[pd.DataFrame, str]:
    """Replace profile-level centrality and amenities with OSMnx features."""

    try:
        import networkx as nx
        import osmnx as ox

        graph = ox.graph_from_place(place, network_type="drive", simplify=True)
        undirected = ox.convert.to_undirected(graph)
        centrality = nx.closeness_centrality(undirected)
        nodes = ox.distance.nearest_nodes(graph, X=df["lon"].to_numpy(dtype=float), Y=df["lat"].to_numpy(dtype=float))
        centrality_values = np.asarray([centrality.get(node, 0.0) for node in nodes], dtype=float)
        if centrality_values.max() > centrality_values.min():
            centrality_values = (centrality_values - centrality_values.min()) / (centrality_values.max() - centrality_values.min())

        tags = {"amenity": ["hospital", "clinic", "school", "restaurant", "cafe"], "leisure": ["park"]}
        pois = ox.features_from_place(place, tags=tags)
        poi_points = pois.geometry.representative_point()
        poi_coords = np.column_stack([poi_points.x.to_numpy(dtype=float), poi_points.y.to_numpy(dtype=float)])
        property_coords = df[["lon", "lat"]].to_numpy(dtype=float)
        amenity_density = []
        for lon, lat in property_coords:
            degree_dist = np.sqrt(np.sum((poi_coords - np.array([lon, lat])) ** 2, axis=1))
            amenity_density.append(float(np.sum(degree_dist <= 0.025)))
        amenity_density = np.asarray(amenity_density, dtype=float)
        if amenity_density.max() > amenity_density.min():
            amenity_density = (amenity_density - amenity_density.min()) / (amenity_density.max() - amenity_density.min())

        working = df.copy()
        working["street_centrality"] = centrality_values
        working["amenity_density"] = amenity_density
        return working, "OSMnx street centrality and amenity density applied."
    except Exception as exc:
        return df, f"OSMnx skipped: {exc}"


def enrich_with_public_sources(df: pd.DataFrame, fred_api_key: str | None = None, use_osmnx: bool = False) -> pd.DataFrame:
    """Apply public-data enrichments and record source notes."""

    working, fred_note = apply_fred_mortgage_rates(df, api_key=fred_api_key)
    osmnx_note = "OSMnx skipped: disabled."
    if use_osmnx:
        working, osmnx_note = apply_osmnx_spatial_features(working)
    working = working.copy()
    working["public_data_notes"] = f"{fred_note} {osmnx_note}"
    return working
