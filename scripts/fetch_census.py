"""Fetch Census ACS 5-year estimates for Tulsa County / Tulsa MSA.

Variables fetched (2023 ACS5):
- B19013_001E:  Median household income
- B25003_001E:  Total housing units
- B25003_002E:  Owner-occupied units
- B25077_001E:  Median home value (owner-occupied)
- B25070_001E:  Rent burden — gross rent as % of income (first bucket only;
                full distribution requires B25070_002E–B25070_011E)

Geography: Tulsa County (FIPS 40143) and ZIP Code Tabulation Areas for
common Tulsa ZIPs.

Requires CENSUS_API_KEY env var or --api-key flag.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "census"
PROCESSED_DIR = ROOT / "data" / "processed"

CENSUS_VARIABLES = {
    "B19013_001E": "median_household_income",
    "B25003_001E": "total_housing_units",
    "B25003_002E": "owner_occupied_units",
    "B25077_001E": "median_home_value",
    "B25070_001E": "rent_burden_30pct_plus_count",
}

# Tulsa MSA = Tulsa County (40143) + surrounding counties
# For v1 we target Tulsa County + a few key ZIPs
TULSA_COUNTY_FIPS = "40143"
OKLAHOMA_FIPS = "40"

TULSA_ZIPS = [
    "74103", "74104", "74105", "74114", "74119", "74120",
    "74132", "74133", "74135", "74137",
    # Additional major Tulsa ZIPs
    "74106", "74107", "74110", "74112", "74115", "74116",
    "74126", "74127", "74128", "74129", "74134", "74136",
    "74145", "74146",
]


def fetch_acs(api_key: str, year: int = 2023) -> dict[str, pd.DataFrame]:
    """Fetch ACS5 estimates for Tulsa County and ZIP-level geographies."""

    import requests

    base_url = f"https://api.census.gov/data/{year}/acs/acs5"
    variables = ",".join(CENSUS_VARIABLES.keys())

    results: dict[str, pd.DataFrame] = {}

    # 1. County-level
    county_url = f"{base_url}?get=NAME,{variables}&for=county:{TULSA_COUNTY_FIPS}&in=state:{OKLAHOMA_FIPS}&key={api_key}"
    try:
        resp = requests.get(county_url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        df = pd.DataFrame(data[1:], columns=data[0])
        for col in CENSUS_VARIABLES:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        results["tulsa_county"] = df
        print(f"  Census county: {len(df)} rows")
    except Exception as exc:
        print(f"  Census county: skipped ({exc})", file=sys.stderr)

    # 2. ZIP-level (ZCTA5)
    zip_list = ",".join(TULSA_ZIPS)
    zip_url = f"{base_url}?get=NAME,{variables}&for=zip%20code%20tabulation%20area:{zip_list}&key={api_key}"
    try:
        resp = requests.get(zip_url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        df = pd.DataFrame(data[1:], columns=data[0])
        for col in CENSUS_VARIABLES:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["zip_code"] = df["zip code tabulation area"]
        df = df.drop(columns=["zip code tabulation area"], errors="ignore")
        results["tulsa_zips"] = df
        print(f"  Census ZIPs: {len(df)} rows")
    except Exception as exc:
        print(f"  Census ZIPs: skipped ({exc})", file=sys.stderr)

    return results


def build_zip_metrics(census_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build a clean ZIP-level metrics DataFrame from raw Census data."""

    df_zip = census_data.get("tulsa_zips")
    if df_zip is None or df_zip.empty:
        return pd.DataFrame()

    metrics = pd.DataFrame()
    metrics["zip_code"] = df_zip["zip_code"].astype(str)

    income = df_zip.get("median_household_income")
    metrics["median_household_income"] = income if income is not None else pd.NA

    total = df_zip.get("total_housing_units")
    owner = df_zip.get("owner_occupied_units")
    if total is not None and owner is not None:
        metrics["owner_occupied_share"] = (owner / total.replace(0, pd.NA)).round(4)
    else:
        metrics["owner_occupied_share"] = pd.NA

    home_value = df_zip.get("median_home_value")
    metrics["median_home_value"] = home_value if home_value is not None else pd.NA

    rent_burden = df_zip.get("rent_burden_30pct_plus_count")
    metrics["rent_burden_count"] = rent_burden if rent_burden is not None else pd.NA

    metrics["data_year"] = 2023
    return metrics


def run(api_key: str | None = None) -> dict:
    """Fetch Census ACS data, save raw + processed. Returns status dict."""

    api_key = api_key or os.getenv("CENSUS_API_KEY")
    status: dict = {"source": "Census ACS", "fetched": False, "error": None}

    if not api_key:
        status["error"] = "No CENSUS_API_KEY env var set."
        print("Census: skipped — no API key.", file=sys.stderr)
        return status

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    census_data = fetch_acs(api_key)
    if not census_data:
        status["error"] = "Failed to fetch any Census data."
        return status

    now_str = datetime.utcnow().strftime("%Y-%m-%d")

    # Save raw
    for key, df in census_data.items():
        raw_path = RAW_DIR / f"{key}_{now_str}.csv"
        df.to_csv(raw_path, index=False)

    # Build ZIP metrics
    zip_metrics = build_zip_metrics(census_data)
    if not zip_metrics.empty:
        zip_path = PROCESSED_DIR / "census_zip_metrics.csv"
        zip_metrics.to_csv(zip_path, index=False)

        json_path = PROCESSED_DIR / "tulsa_zip_metrics.json"
        zip_metrics.to_json(json_path, orient="records", indent=2)
        status["zip_metrics_path"] = str(json_path)

    status["fetched"] = True
    status["geo_levels"] = list(census_data.keys())
    summary_path = RAW_DIR / "census_summary.json"
    with open(summary_path, "w") as f:
        json.dump(status, f, indent=2, default=str)

    print(f"Census: wrote ZIP metrics to {PROCESSED_DIR}")
    return status


def main():
    parser = argparse.ArgumentParser(description="Fetch Census ACS data.")
    parser.add_argument("--api-key", help="Census API key (or set CENSUS_API_KEY)")
    args = parser.parse_args()
    result = run(api_key=args.api_key)
    return 1 if result.get("error") else 0


if __name__ == "__main__":
    raise SystemExit(main())
