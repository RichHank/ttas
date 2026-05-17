"""Import Realtor.com Research Data CSVs for Tulsa metro/ZIP areas.

Realtor.com provides downloadable CSV exports from:
https://www.realtor.com/research/data/

Expected CSV layout (Realtor.com Research format):
- Month ending in YYYY-MM-DD
- Geography (ZIP, county, or metro name)
- Median listing price, median rent, active listing count, days on market, etc.

USAGE:
1. Download CSVs from https://www.realtor.com/research/data/
   for Tulsa metro / Tulsa County / Tulsa ZIP codes
2. Place them in data/raw/realtor/
3. Run: python scripts/import_realtor_csv.py

The script auto-detects ZIP-level vs county-level vs metro-level CSVs.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "realtor"
PROCESSED_DIR = ROOT / "data" / "processed"

# Known column name variants across Realtor.com Research CSV formats
COLUMN_MAP = {
    # Date columns
    "Month": "date",
    "month": "date",
    "month_date_yyyymm": "date",
    "Period Begin": "date",
    # Geography columns
    "Geography Name": "geography_name",
    "cbsa_title": "geography_name",
    "Zip Code": "zip_code",
    "ZipCode": "zip_code",
    "County Name": "county_name",
    "county_name": "county_name",
    "Metro Name": "metro_name",
    # Price columns
    "Median Listing Price": "median_listing_price",
    "median_listing_price": "median_listing_price",
    "Median Listing Price YY": "median_listing_price_yy",
    "median_listing_price_yy": "median_listing_price_yy",
    "Median Listing Price MM": "median_listing_price_mm",
    "median_listing_price_mm": "median_listing_price_mm",
    "Average Listing Price": "average_listing_price",
    "average_listing_price": "average_listing_price",
    # Rent columns
    "Median Rent": "median_rent",
    "Median Rent YY": "median_rent_yy",
    # Inventory columns
    "Active Listing Count": "active_listing_count",
    "active_listing_count": "active_listing_count",
    "Median Days on Market": "median_days_on_market",
    "median_days_on_market": "median_days_on_market",
    "Days on Market": "median_days_on_market",
    "New Listing Count": "new_listing_count",
    "new_listing_count": "new_listing_count",
    "Price Reduced Count": "price_reduced_count",
    "price_reduced_count": "price_reduced_count",
    "Pending Listing Count": "pending_listing_count",
    "Total Listing Count": "total_listing_count",
    "total_listing_count": "total_listing_count",
    "Median PPSF": "median_price_per_sqft",
    # Metadata columns (pass through)
    "cbsa_code": "cbsa_code",
    "HouseholdRank": "household_rank",
}

REQUIRED_FIELDS = [
    "date", "median_listing_price", "median_rent",
    "active_listing_count", "median_days_on_market",
]


def detect_format(df: pd.DataFrame) -> str:
    """Detect whether the CSV is ZIP, county, or metro level."""
    cols = set(df.columns)
    has_zip = any(c in cols for c in ["Zip Code", "ZipCode", "zip_code"])
    has_county = "County Name" in cols or "county_name" in cols
    has_metro = any(c in cols for c in ["Metro Name", "metro_name", "cbsa_title"])
    if has_zip:
        return "zip"
    if has_county:
        return "county"
    if has_metro:
        return "metro"
    return "unknown"


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to canonical names and parse dates."""
    df = df.copy()
    df = df.rename(columns=COLUMN_MAP)

    # Parse date column
    if "date" in df.columns:
        try:
            df["date"] = pd.to_datetime(df["date"])
        except Exception:
            df["date"] = pd.to_datetime(df["date"].astype(str).str[:7].str.strip())

    # Ensure numeric columns
    for col in ["median_listing_price", "median_rent", "active_listing_count",
                "median_days_on_market", "average_listing_price",
                "median_price_per_sqft", "new_listing_count",
                "price_reduced_count", "pending_listing_count",
                "total_listing_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def import_all() -> tuple[pd.DataFrame, dict]:
    """Import all CSVs from data/raw/realtor/, normalize, and merge.

    Returns (combined_timeseries, status_dict).
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(RAW_DIR.glob("*.csv"))
    if not csv_files:
        msg = "No CSV files found in data/raw/realtor/"
        print(f"Realtor: skipped — {msg}", file=sys.stderr)
        return pd.DataFrame(), {"error": msg, "files_found": 0}

    status: dict = {"source": "Realtor.com Research Data", "files_found": len(csv_files),
                    "files_imported": 0, "files_failed": [], "geography_types": set()}

    all_frames: list[pd.DataFrame] = []

    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
            fmt = detect_format(df)
            df = normalize_columns(df)
            df["source_file"] = csv_path.name
            df["geography_type"] = fmt
            all_frames.append(df)
            status["geography_types"].add(fmt)
            status["files_imported"] += 1
            print(f"  Imported {csv_path.name} ({fmt}, {len(df)} rows)")
        except Exception as exc:
            status["files_failed"].append(str(csv_path.name))
            print(f"  Failed {csv_path.name}: {exc}", file=sys.stderr)

    if not all_frames:
        status["error"] = "All CSV imports failed."
        return pd.DataFrame(), status

    combined = pd.concat(all_frames, ignore_index=True)

    # Convert set to list for JSON serialization
    status["geography_types"] = sorted(status["geography_types"])

    # Build a monthly metro/ZIP timeseries
    if "date" in combined.columns:
        combined["month"] = combined["date"].dt.to_period("M")
    else:
        combined["month"] = None

    # Aggregate: median across all geographies for each month
    agg_cols = [c for c in ["median_listing_price", "median_rent",
                             "active_listing_count", "median_days_on_market",
                             "median_price_per_sqft"]
                if c in combined.columns]

    if agg_cols and "month" in combined.columns:
        monthly = combined.groupby("month")[agg_cols].median().reset_index()
        monthly["date"] = monthly["month"].astype(str)
        monthly = monthly.drop(columns=["month"])

        processed_path = PROCESSED_DIR / "realtor_monthly.csv"
        monthly.to_csv(processed_path, index=False)
        status["path"] = str(processed_path)
        status["row_count"] = len(monthly)
        print(f"Realtor: wrote {len(monthly)} monthly rows to {processed_path}")
    else:
        # Save raw combined even if no aggregation possible
        raw_combined_path = PROCESSED_DIR / "realtor_combined.csv"
        combined.to_csv(raw_combined_path, index=False)
        status["path"] = str(raw_combined_path)

    return combined, status


def run() -> dict:
    """Main entry point for the import script. Returns status dict."""

    _, status = import_all()

    # Write summary
    summary_path = RAW_DIR / "realtor_summary.json"
    serializable = {k: (sorted(v) if isinstance(v, set) else v)
                    for k, v in status.items()}
    with open(summary_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)

    return status


def main():
    parser = argparse.ArgumentParser(description="Import Realtor.com Research CSVs.")
    parser.add_argument("--csv", help="Path to a single CSV file to import")
    args = parser.parse_args()

    if args.csv:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        import shutil
        dest = RAW_DIR / Path(args.csv).name
        shutil.copy2(args.csv, dest)
        print(f"Copied {args.csv} -> {dest}")

    result = run()
    return 1 if result.get("error") and not result.get("files_imported") else 0


if __name__ == "__main__":
    raise SystemExit(main())
