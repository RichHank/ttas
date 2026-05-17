"""Fetch FRED economic time series for the Tulsa housing dashboard.

Series fetched:
- MORTGAGE30US: 30-year fixed mortgage rate (weekly → monthly average)
- CPIAUCSL:   Consumer Price Index (monthly)
- UNRATE:     National unemployment rate (monthly)
- MSPUS:      Median sales price of houses sold, US (quarterly → interpolated)

Requires FRED_API_KEY env var or --api-key flag. Falls back gracefully.
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
RAW_DIR = ROOT / "data" / "raw" / "fred"
PROCESSED_DIR = ROOT / "data" / "processed"

FRED_SERIES = {
    "MORTGAGE30US": {"name": "30_year_mortgage_rate", "freq": "monthly"},
    "CPIAUCSL":     {"name": "cpi_all_urban",          "freq": "monthly"},
    "UNRATE":       {"name": "unemployment_rate",      "freq": "monthly"},
    "MSPUS":        {"name": "median_sales_price_us",  "freq": "quarterly"},
}


def fetch_series(api_key: str, series_id: str) -> pd.Series | None:
    """Fetch a single FRED series. Returns None on failure."""
    try:
        from fredapi import Fred

        fred = Fred(api_key=api_key)
        raw = fred.get_series(series_id)
        return raw
    except Exception as exc:
        print(f"  FRED {series_id}: skipped ({exc})", file=sys.stderr)
        return None


def resample_to_monthly(series: pd.Series, name: str, freq: str) -> pd.Series:
    """Convert a FRED series to calendar-month frequency."""
    series = series.copy()
    series.index = pd.to_datetime(series.index)

    if freq == "monthly":
        monthly = series.resample("MS").first()
    elif freq == "quarterly":
        monthly = series.resample("MS").interpolate("linear")
    elif freq == "weekly":
        monthly = series.resample("MS").mean()
    else:
        monthly = series.resample("MS").first()

    monthly.name = name
    monthly = monthly.interpolate().ffill().bfill()
    return monthly.round(6)


def run(api_key: str | None = None) -> dict:
    """Fetch all FRED series, save raw + processed CSVs. Returns status dict."""

    api_key = api_key or os.getenv("FRED_API_KEY")
    status: dict = {"source": "FRED", "fetched": [], "skipped": [], "error": None}

    if not api_key:
        status["error"] = "No FRED_API_KEY env var set."
        print("FRED: skipped — no API key. Set FRED_API_KEY env var.", file=sys.stderr)
        return status

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    monthly_series: dict[str, pd.Series] = {}
    now_str = datetime.utcnow().strftime("%Y-%m-%d")

    for series_id, cfg in FRED_SERIES.items():
        raw = fetch_series(api_key, series_id)
        if raw is None:
            status["skipped"].append(series_id)
            continue

        raw_path = RAW_DIR / f"{series_id}_{now_str}.csv"
        raw.to_csv(raw_path)
        monthly = resample_to_monthly(raw, cfg["name"], cfg["freq"])
        monthly_series[cfg["name"]] = monthly
        status["fetched"].append(series_id)
        print(f"  FRED {series_id}: {len(raw)} observations -> {len(monthly)} monthly")

    if not monthly_series:
        status["error"] = "No FRED series were successfully fetched."
        return status

    df = pd.DataFrame(monthly_series)
    df.index.name = "date"
    df = df.reset_index()

    processed_path = PROCESSED_DIR / "fred_monthly.csv"
    df.to_csv(processed_path, index=False)

    status["path"] = str(processed_path)
    status["date_range"] = [str(df["date"].min()), str(df["date"].max())]
    status["row_count"] = len(df)

    summary_path = RAW_DIR / "fred_summary.json"
    with open(summary_path, "w") as f:
        json.dump(status, f, indent=2, default=str)

    print(f"FRED: wrote {len(df)} rows to {processed_path}")
    return status


def main():
    parser = argparse.ArgumentParser(description="Fetch FRED economic data.")
    parser.add_argument("--api-key", help="FRED API key (or set FRED_API_KEY)")
    args = parser.parse_args()
    result = run(api_key=args.api_key)
    if result.get("error"):
        print(result["error"])
    return 0 if result.get("fetched") else 1


if __name__ == "__main__":
    raise SystemExit(main())
