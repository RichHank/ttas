"""Build the unified Tulsa market dataset from all available real sources.

Orchestrates data from FRED, Census ACS, Realtor.com, and Tulsa open data
into a single monthly timeseries that the Dash app can consume.

When real data sources are unavailable, marks them as missing in the
source_manifest.json so the app can fall back to synthetic data.

Outputs:
- data/processed/tulsa_market_timeseries.csv  (monthly aggregate series)
- data/processed/tulsa_zip_metrics.json        (ZIP-level snapshot)
- data/processed/source_manifest.json          (full provenance record)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
RAW_DIR = ROOT / "data" / "raw"


# ── source manifest builder ────────────────────────────────────────────────

def build_manifest(statuses: dict) -> dict:
    """Assemble the source manifest from all import statuses."""

    sources = []
    available = False

    # FRED
    fred = statuses.get("fred", {})
    if fred.get("fetched"):
        sources.append({
            "name": "FRED (Federal Reserve Economic Data)",
            "fields": ["30_year_mortgage_rate", "cpi_all_urban",
                       "unemployment_rate", "median_sales_price_us"],
            "geography": "National (US), some Tulsa-specific series",
            "frequency": "monthly",
            "status": "available",
            "path": fred.get("path", ""),
        })
        available = True
    else:
        sources.append({
            "name": "FRED (Federal Reserve Economic Data)",
            "fields": ["30_year_mortgage_rate", "cpi_all_urban",
                       "unemployment_rate", "median_sales_price_us"],
            "geography": "National (US)",
            "frequency": "monthly",
            "status": "unavailable — set FRED_API_KEY",
        })

    # Census ACS
    census = statuses.get("census", {})
    if census.get("fetched"):
        sources.append({
            "name": "Census ACS 5-Year Estimates",
            "fields": ["median_household_income", "owner_occupied_share",
                       "median_home_value", "rent_burden_count"],
            "geography": "Tulsa County + ZIP Code Tabulation Areas",
            "frequency": "annual (most recent: 2023)",
            "status": "available",
            "path": census.get("zip_metrics_path", ""),
        })
        available = True
    else:
        sources.append({
            "name": "Census ACS 5-Year Estimates",
            "fields": ["median_household_income", "owner_occupied_share",
                       "median_home_value", "rent_burden_count"],
            "geography": "Tulsa County + ZIP Code Tabulation Areas",
            "frequency": "annual",
            "status": "unavailable — set CENSUS_API_KEY",
        })

    # Realtor.com
    realtor = statuses.get("realtor", {})
    files_found = realtor.get("files_found", 0)
    files_imported = realtor.get("files_imported", 0)
    if files_imported > 0:
        sources.append({
            "name": "Realtor.com Research Data",
            "fields": ["median_listing_price", "median_rent",
                       "active_listing_count", "median_days_on_market"],
            "geography": f"Tulsa metro / county / ZIP ({realtor.get('geography_types', [])})",
            "frequency": "monthly",
            "status": f"available — {files_imported}/{files_found} files imported",
            "path": realtor.get("path", ""),
        })
        available = True
    else:
        sources.append({
            "name": "Realtor.com Research Data",
            "fields": ["median_listing_price", "median_rent",
                       "active_listing_count", "median_days_on_market"],
            "geography": "Tulsa metro / county / ZIP",
            "frequency": "monthly",
            "status": "unavailable — download CSVs from realtor.com/research/data/",
        })

    # Tulsa open data (future)
    sources.append({
        "name": "Tulsa County Assessor / City of Tulsa Open Data",
        "fields": ["parcel_assessed_value", "property_class", "last_sale_date"],
        "geography": "Tulsa County parcels",
        "frequency": "as-available",
        "status": "planned — not yet integrated",
    })

    manifest = {
        "dataset_name": "Tulsa Housing Market Dataset",
        "mode": "real_public_data" if available else "synthetic_fallback",
        "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sources": sources,
        "synthetic_data_active": not available,
        "synthetic_note": ("Real data sources are unavailable; using Tulsa-calibrated "
                           "synthetic manifold as fallback. To enable real data, set "
                           "FRED_API_KEY and CENSUS_API_KEY env vars and place "
                           "Realtor.com Research CSVs in data/raw/realtor/.") if not available else None,
    }

    return manifest


# ── timeseries builder ─────────────────────────────────────────────────────

REQUIRED_COLUMNS = [
    "median_listing_price",
    "median_rent",
    "active_listing_count",
    "median_days_on_market",
    "mortgage_rate_30y",
    "median_household_income",
    "owner_occupied_share",
    "rent_burden_pct",
]


def build_timeseries(statuses: dict) -> pd.DataFrame | None:
    """Merge all available monthly series into a single timeseries.

    Returns None if no real data is available at all.
    """

    frames: list[pd.DataFrame] = []
    date_range_info: dict = {}

    # 1. FRED monthly
    fred_path = PROCESSED_DIR / "fred_monthly.csv"
    if fred_path.exists():
        fred_df = pd.read_csv(fred_path, parse_dates=["date"])
        frames.append(fred_df.set_index("date"))
        date_range_info["fred"] = (str(fred_df["date"].min()), str(fred_df["date"].max()))
        print(f"  FRED: {len(fred_df)} rows, {list(fred_df.columns)}")

    # 2. Realtor.com monthly
    realtor_path = PROCESSED_DIR / "realtor_monthly.csv"
    if realtor_path.exists():
        realtor_df = pd.read_csv(realtor_path, parse_dates=["date"])
        frames.append(realtor_df.set_index("date"))
        date_range_info["realtor"] = (str(realtor_df["date"].min()), str(realtor_df["date"].max()))
        print(f"  Realtor: {len(realtor_df)} rows, {list(realtor_df.columns)}")

    if not frames:
        print("No real data available for timeseries build.")
        return None

    # Merge on date index
    merged = frames[0]
    for other in frames[1:]:
        merged = merged.join(other, how="outer")

    merged = merged.sort_index().interpolate(limit=3).ffill().bfill()
    merged.index.name = "date"
    result = merged.reset_index()

    # Compute derived columns
    if "median_listing_price" in result.columns and "median_rent" in result.columns:
        result["price_to_rent_ratio"] = (
            result["median_listing_price"] /
            (result["median_rent"] * 12.0)
        ).round(2)

    if "median_listing_price" in result.columns and "mortgage_rate_30y" in result.columns:
        rate_monthly = result["mortgage_rate_30y"] / 100.0 / 12.0
        loan = result["median_listing_price"] * 0.80
        valid = rate_monthly > 0
        factor = (1.0 + rate_monthly[valid]) ** 360
        pi = pd.Series(0.0, index=result.index)
        pi[valid] = (loan[valid] * rate_monthly[valid] * factor /
                     (factor - 1.0))
        result["estimated_monthly_payment"] = (pi + result["median_listing_price"] * 0.0116 / 12.0
                                                + result["median_listing_price"] * 0.0042 / 12.0).round(2)

    if "estimated_monthly_payment" in result.columns and "median_rent" in result.columns:
        result["buy_vs_rent_spread"] = (
            result["estimated_monthly_payment"] - result["median_rent"]
        ).round(2)

    # Save
    output_path = PROCESSED_DIR / "tulsa_market_timeseries.csv"
    result.to_csv(output_path, index=False)

    statuses["timeseries"] = {
        "path": str(output_path),
        "row_count": len(result),
        "columns": list(result.columns),
        "date_range": [str(result["date"].min()), str(result["date"].max())],
        "sources": date_range_info,
    }

    print(f"\nTimeseries: {len(result)} rows -> {output_path}")
    print(f"  Columns: {list(result.columns)}")
    return result


# ── main ────────────────────────────────────────────────────────────────────

def run() -> dict:
    """Run the full dataset build. Returns the manifest dict."""

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    statuses: dict = {}
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # 1. Check FRED
    fred_path = PROCESSED_DIR / "fred_monthly.csv"
    if fred_path.exists():
        df = pd.read_csv(fred_path)
        statuses["fred"] = {
            "fetched": True, "path": str(fred_path),
            "row_count": len(df), "columns": list(df.columns),
        }
        print(f"Found FRED data: {len(df)} rows")
    else:
        statuses["fred"] = {"fetched": False}
        print("FRED data not found — run scripts/fetch_fred.py first")

    # 2. Check Census
    census_path = PROCESSED_DIR / "census_zip_metrics.csv"
    if census_path.exists():
        df = pd.read_csv(census_path)
        statuses["census"] = {
            "fetched": True, "path": str(census_path),
            "zip_count": len(df),
        }
        print(f"Found Census data: {len(df)} ZIPs")
    else:
        statuses["census"] = {"fetched": False}
        print("Census data not found — run scripts/fetch_census.py first")

    # 3. Check Realtor.com
    realtor_path = PROCESSED_DIR / "realtor_monthly.csv"
    if realtor_path.exists():
        df = pd.read_csv(realtor_path)
        statuses["realtor"] = {
            "files_found": 1, "files_imported": 1,
            "path": str(realtor_path), "row_count": len(df),
        }
        print(f"Found Realtor data: {len(df)} rows")
    else:
        raw_csvs = list((RAW_DIR / "realtor").glob("*.csv")) if (RAW_DIR / "realtor").exists() else []
        statuses["realtor"] = {
            "files_found": len(raw_csvs), "files_imported": 0,
        }
        if raw_csvs:
            print(f"Realtor CSVs need import: {len(raw_csvs)} files in data/raw/realtor/"
                  f" — run scripts/import_realtor_csv.py")
        else:
            print("Realtor data not found — download CSVs from realtor.com/research/data/")

    # 4. Build timeseries
    build_timeseries(statuses)

    # 5. Build manifest
    manifest = build_manifest(statuses)

    manifest_path = PROCESSED_DIR / "source_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)

    print(f"\nManifest: {manifest_path}")
    print(f"Data mode: {manifest['mode']}")

    # Write pipeline summary
    summary_path = RAW_DIR / f"pipeline_status_{now_str}.json"
    with open(summary_path, "w") as f:
        json.dump({"statuses": statuses, "manifest_mode": manifest["mode"],
                   "last_updated": manifest["last_updated"]},
                  f, indent=2, default=str)

    return manifest


def main():
    parser = argparse.ArgumentParser(description="Build the Tulsa market dataset.")
    parser.add_argument("--json", action="store_true", help="Output manifest as JSON to stdout")
    args = parser.parse_args()

    manifest = run()

    if args.json:
        print(json.dumps(manifest, indent=2, default=str))

    return 1 if manifest["mode"] == "synthetic_fallback" else 0


if __name__ == "__main__":
    raise SystemExit(main())
