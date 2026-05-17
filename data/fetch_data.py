"""Acquire and synthesize the Tulsa affordability manifold.

The research prompt asks for a twelve-dimensional point cloud over Tulsa from
2018 Q1 through 2025 Q4. Public housing, census, OpenStreetMap, and school
quality feeds are not always available without API keys or per-site terms, so
this module implements a reproducible public-data-ready generator:

* if optional API keys are present, callers can inject real series upstream;
* otherwise, the code creates a calibrated synthetic manifold with known shocks;
* every row remains auditable, with the latent drivers written as columns.

Mathematically each property is a point

    x_i(t) in R^12

with coordinates matching the prompt: price, rent pressure, inventory velocity,
tax, schools, centrality, amenities, crime, flood risk, walk/transit access,
mobility, and the household DTI constraint.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

import numpy as np
import pandas as pd


TULSA_ZIP_PROFILES = {
    "74103": {
        "name": "Downtown Tulsa",
        "lat": 36.1540,
        "lon": -95.9928,
        "price_anchor": 247_000,
        "income_anchor": 68_000,
        "school": 0.58,
        "centrality": 0.94,
        "amenity": 0.95,
        "crime": 0.62,
        "flood": 0.18,
        "walk": 0.87,
        "mobility": 0.61,
    },
    "74104": {
        "name": "Kendall Whittier",
        "lat": 36.1517,
        "lon": -95.9581,
        "price_anchor": 216_000,
        "income_anchor": 61_000,
        "school": 0.63,
        "centrality": 0.78,
        "amenity": 0.82,
        "crime": 0.55,
        "flood": 0.14,
        "walk": 0.76,
        "mobility": 0.58,
    },
    "74105": {
        "name": "Brookside",
        "lat": 36.0994,
        "lon": -95.9737,
        "price_anchor": 335_000,
        "income_anchor": 92_000,
        "school": 0.77,
        "centrality": 0.72,
        "amenity": 0.86,
        "crime": 0.34,
        "flood": 0.24,
        "walk": 0.79,
        "mobility": 0.72,
    },
    "74114": {
        "name": "Midtown",
        "lat": 36.1239,
        "lon": -95.9368,
        "price_anchor": 392_000,
        "income_anchor": 108_000,
        "school": 0.84,
        "centrality": 0.68,
        "amenity": 0.78,
        "crime": 0.28,
        "flood": 0.12,
        "walk": 0.68,
        "mobility": 0.79,
    },
    "74119": {
        "name": "Riverview",
        "lat": 36.1410,
        "lon": -96.0009,
        "price_anchor": 268_000,
        "income_anchor": 72_000,
        "school": 0.66,
        "centrality": 0.88,
        "amenity": 0.83,
        "crime": 0.45,
        "flood": 0.36,
        "walk": 0.81,
        "mobility": 0.64,
    },
    "74120": {
        "name": "Pearl District",
        "lat": 36.1573,
        "lon": -95.9756,
        "price_anchor": 232_000,
        "income_anchor": 64_000,
        "school": 0.59,
        "centrality": 0.91,
        "amenity": 0.88,
        "crime": 0.57,
        "flood": 0.20,
        "walk": 0.84,
        "mobility": 0.59,
    },
    "74132": {
        "name": "Tulsa Hills",
        "lat": 36.0617,
        "lon": -96.0281,
        "price_anchor": 302_000,
        "income_anchor": 86_000,
        "school": 0.72,
        "centrality": 0.38,
        "amenity": 0.58,
        "crime": 0.31,
        "flood": 0.16,
        "walk": 0.41,
        "mobility": 0.68,
    },
    "74133": {
        "name": "Union / South Tulsa",
        "lat": 36.0613,
        "lon": -95.8855,
        "price_anchor": 281_000,
        "income_anchor": 82_000,
        "school": 0.75,
        "centrality": 0.42,
        "amenity": 0.64,
        "crime": 0.29,
        "flood": 0.10,
        "walk": 0.46,
        "mobility": 0.70,
    },
    "74135": {
        "name": "Patrick Henry",
        "lat": 36.1121,
        "lon": -95.9264,
        "price_anchor": 255_000,
        "income_anchor": 74_000,
        "school": 0.71,
        "centrality": 0.58,
        "amenity": 0.70,
        "crime": 0.36,
        "flood": 0.13,
        "walk": 0.56,
        "mobility": 0.66,
    },
    "74137": {
        "name": "Southern Hills",
        "lat": 36.0339,
        "lon": -95.9334,
        "price_anchor": 415_000,
        "income_anchor": 119_000,
        "school": 0.88,
        "centrality": 0.34,
        "amenity": 0.56,
        "crime": 0.22,
        "flood": 0.09,
        "walk": 0.36,
        "mobility": 0.82,
    },
}


@dataclass(frozen=True)
class TulsaDataConfig:
    """Configuration for the TTAS manifold generator."""

    start: str = "2018-01-01"
    end: str = "2025-12-01"
    properties_per_zip_month: int = 10
    random_seed: int = 918
    cache_dir: Path = Path("outputs/cache")
    filename: str = "tulsa_manifold.csv"
    fred_api_key: str | None = None
    use_public_data: bool = False
    use_osmnx: bool = False

    @property
    def cache_path(self) -> Path:
        return self.cache_dir / self.filename


def monthly_index(start: str = "2018-01-01", end: str = "2025-12-01") -> pd.DatetimeIndex:
    """Return a monthly index from Q1 2018 through Q4 2025 by default."""

    return pd.date_range(start=start, end=end, freq="MS")


def synthetic_mortgage_rate(month_number: int, rng: np.random.Generator) -> float:
    """Generate a Tulsa-facing 30-year mortgage rate path with known shocks."""

    base = 4.35 - 1.15 * np.exp(-0.5 * ((month_number - 30) / 9.0) ** 2)
    fed_shock = 3.05 / (1.0 + np.exp(-(month_number - 54) / 4.6))
    cooling = -0.65 / (1.0 + np.exp(-(month_number - 78) / 4.0))
    noise = rng.normal(0.0, 0.08)
    return float(np.clip(base + fed_shock + cooling + noise, 2.55, 8.15))


def _clipped(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return float(np.clip(value, low, high))


def _housing_cost(price: float, mortgage_rate: float, tax_rate: float) -> float:
    """Approximate monthly ownership cost with taxes and insurance.

    The mortgage component uses the fixed-payment annuity formula

        P = L r (1 + r)^n / ((1 + r)^n - 1)

    where L is the 80 percent loan principal, r is the monthly rate, and
    n = 360. It is intentionally transparent rather than lender-specific.
    """

    loan = 0.80 * price
    monthly_rate = mortgage_rate / 100.0 / 12.0
    if monthly_rate <= 0:
        principal_interest = loan / 360.0
    else:
        factor = (1.0 + monthly_rate) ** 360
        principal_interest = loan * monthly_rate * factor / (factor - 1.0)
    monthly_tax = price * tax_rate / 12.0
    monthly_insurance = price * 0.0042 / 12.0
    maintenance = price * 0.008 / 12.0
    return float(principal_interest + monthly_tax + monthly_insurance + maintenance)


def generate_tulsa_manifold(config: TulsaDataConfig | None = None) -> pd.DataFrame:
    """Create the high-dimensional Tulsa property manifold.

    The generator is deterministic for a given seed and creates a compact but
    expressive testbed for the complete topology pipeline. It encodes three
    named market regimes:

    * a pre-pandemic baseline from 2018-2019;
    * an overheating price/velocity phase from mid-2020 through 2022;
    * a higher-rate compression phase from 2023 onward.

    Returns:
        A tidy dataframe where each row is a property-month observation.
    """

    config = config or TulsaDataConfig()
    rng = np.random.default_rng(config.random_seed)
    months = monthly_index(config.start, config.end)
    rows: list[dict[str, object]] = []

    for month_number, date in enumerate(months):
        mortgage_rate = synthetic_mortgage_rate(month_number, rng)
        season = np.sin(2.0 * np.pi * (date.month - 1) / 12.0)
        long_trend = month_number / max(len(months) - 1, 1)
        pandemic_heat = np.exp(-0.5 * ((month_number - 44) / 11.0) ** 2)
        rate_drag = 1.0 / (1.0 + np.exp(-(month_number - 60) / 5.0))
        rent_pressure = 0.62 * pandemic_heat + 0.25 * rate_drag

        for zip_code, profile in TULSA_ZIP_PROFILES.items():
            zip_growth = rng.normal(0.0, 0.012)
            neighborhood_alpha = 0.70 + 0.40 * profile["mobility"] + 0.15 * profile["amenity"]
            price_index = (
                1.0
                + 0.20 * long_trend
                + 0.18 * pandemic_heat * neighborhood_alpha
                - 0.045 * rate_drag
                + 0.018 * season
                + zip_growth
            )
            income_index = 1.0 + 0.032 * (date.year - 2018) + 0.006 * season
            tax_rate = 0.0116 + rng.normal(0.0, 0.00045)

            for local_id in range(config.properties_per_zip_month):
                property_mix = rng.normal(1.0, 0.14)
                micro_location = rng.normal(0.0, 0.035)
                price = profile["price_anchor"] * price_index * property_mix
                price = float(np.clip(price, 85_000, 850_000))
                annual_income = profile["income_anchor"] * income_index * rng.normal(1.0, 0.08)

                annual_rent_ratio = (
                    0.061
                    + 0.012 * rent_pressure
                    - 0.006 * profile["price_anchor"] / 450_000
                    + rng.normal(0.0, 0.003)
                )
                annual_rent_ratio = float(np.clip(annual_rent_ratio, 0.035, 0.102))
                monthly_rent = price * annual_rent_ratio / 12.0

                days_on_market = (
                    54.0
                    - 20.0 * pandemic_heat
                    + 15.0 * rate_drag
                    - 5.0 * season
                    + rng.normal(0.0, 6.5)
                )
                inventory_velocity = float(30.0 / np.clip(days_on_market, 8.0, 120.0))

                school = _clipped(profile["school"] + 0.04 * micro_location + rng.normal(0.0, 0.035))
                centrality = _clipped(profile["centrality"] + rng.normal(0.0, 0.045))
                amenity = _clipped(profile["amenity"] + 0.03 * season + rng.normal(0.0, 0.04))
                crime = _clipped(profile["crime"] - 0.10 * long_trend + rng.normal(0.0, 0.05))
                flood = _clipped(profile["flood"] + 0.04 * np.cos(2.0 * np.pi * date.month / 12.0) + rng.normal(0.0, 0.025))
                walk = _clipped(profile["walk"] + 0.06 * centrality + rng.normal(0.0, 0.035))
                transit = _clipped(0.55 * walk + 0.34 * centrality + rng.normal(0.0, 0.04))
                mobility = _clipped(profile["mobility"] + 0.04 * long_trend + rng.normal(0.0, 0.035))
                dti_max = float(np.clip(0.33 + 0.06 * rng.beta(2.2, 3.0), 0.28, 0.47))

                ownership_cost = _housing_cost(price, mortgage_rate, tax_rate)
                max_housing_budget = annual_income * dti_max / 12.0
                buy_margin = max_housing_budget - ownership_cost
                rent_margin = max_housing_budget - monthly_rent
                opportunity = 0.46 * school + 0.33 * mobility + 0.21 * amenity - 0.18 * crime - 0.08 * flood
                rent_vs_buy = "buy" if buy_margin > -0.10 * monthly_rent and opportunity > 0.42 else "rent"

                rows.append(
                    {
                        "property_id": f"{zip_code}-{date.strftime('%Y%m')}-{local_id:03d}",
                        "date": date,
                        "year": date.year,
                        "month": date.month,
                        "zip_code": zip_code,
                        "neighborhood": profile["name"],
                        "lat": profile["lat"] + rng.normal(0.0, 0.006),
                        "lon": profile["lon"] + rng.normal(0.0, 0.006),
                        "median_listing_price": round(price, 2),
                        "monthly_rent_estimate": round(monthly_rent, 2),
                        "rent_to_price_ratio": annual_rent_ratio,
                        "inventory_velocity": inventory_velocity,
                        "property_tax_rate": tax_rate,
                        "school_rating": school,
                        "street_centrality": centrality,
                        "amenity_density": amenity,
                        "crime_index": crime,
                        "flood_risk_score": flood,
                        "walk_transit_score": 0.68 * walk + 0.32 * transit,
                        "economic_mobility_index": mobility,
                        "dti_max": dti_max,
                        "annual_income_estimate": round(float(annual_income), 2),
                        "mortgage_rate_30y": mortgage_rate,
                        "ownership_cost_monthly": round(ownership_cost, 2),
                        "rent_margin_monthly": round(float(rent_margin), 2),
                        "buy_margin_monthly": round(float(buy_margin), 2),
                        "regime_hint": _regime_hint(month_number),
                        "rent_vs_buy": rent_vs_buy,
                    }
                )

    df = pd.DataFrame(rows)
    return df.sort_values(["date", "zip_code", "property_id"]).reset_index(drop=True)


def _regime_hint(month_number: int) -> str:
    if month_number < 24:
        return "Stable"
    if month_number < 57:
        return "Overheated"
    if month_number < 72:
        return "Rate Shock"
    return "Opportunity"


def get_data_mode() -> str:
    """Report whether real public data is active.

    Returns "real_public_data" when at least one real source has been loaded,
    or "synthetic_fallback" when using purely synthetic data.
    """
    try:
        from .real_data import get_data_mode as _get_mode
        return _get_mode()
    except Exception:
        return "synthetic_fallback"


def load_or_create_dataset(config: TulsaDataConfig | None = None, refresh: bool = False) -> pd.DataFrame:
    """Load a cached manifold or generate it locally.

    When real public data is available (via scripts/build_dataset.py), the
    synthetic manifold is calibrated to track real aggregate metrics. When no
    real data exists, a fully synthetic Tulsa-calibrated manifold is used.
    """

    config = config or TulsaDataConfig(fred_api_key=os.getenv("FRED_API_KEY"))
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    if config.cache_path.exists() and not refresh:
        df = pd.read_csv(config.cache_path, parse_dates=["date"])
        return df

    df = generate_tulsa_manifold(config)

    # Try real-data calibration (replaces synthetic aggregates with real ones)
    data_mode = get_data_mode()
    if data_mode == "real_public_data":
        try:
            from .real_data import calibrate_synthetic_from_real

            df = calibrate_synthetic_from_real(df)
            df["public_data_notes"] = "Real public data calibrating synthetic manifold."
        except Exception as exc:
            df["public_data_notes"] = f"Real-data calibration skipped: {exc}"
    elif config.use_public_data:
        try:
            from .real_data import enrich_with_public_sources

            df = enrich_with_public_sources(df, fred_api_key=config.fred_api_key, use_osmnx=config.use_osmnx)
        except Exception as exc:
            df["public_data_notes"] = f"Public-data enrichment skipped: {exc}"
    else:
        df["public_data_notes"] = "Synthetic Tulsa-calibrated manifold (no real data sources active)."

    df.to_csv(config.cache_path, index=False)
    return df


def main() -> None:
    """CLI entrypoint used by setup scripts and local smoke tests."""

    df = load_or_create_dataset(refresh=True)
    path = TulsaDataConfig().cache_path
    print(f"Wrote {len(df):,} property-month rows to {path}")


if __name__ == "__main__":
    main()
