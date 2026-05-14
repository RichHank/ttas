"""Rent-vs-buy path integral over persistence landscapes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from data.preprocess import FEATURE_COLUMNS, add_topological_parameters, minmax_scale
from topology.invariants import persistence_diagrams, persistence_landscape


@dataclass(frozen=True)
class UserBiography:
    """Household vector B embedded into the Tulsa manifold."""

    annual_income: float = 92_000.0
    dti_max: float = 0.38
    family_size: int = 3
    down_payment_fraction: float = 0.20
    target_zip: str | None = None

    @property
    def max_purchase_price(self) -> float:
        """Prompt convention: affordable homes satisfy price <= 3 * income."""

        return 3.0 * self.annual_income

    @property
    def monthly_budget(self) -> float:
        return self.annual_income * self.dti_max / 12.0


def restricted_affordable_market(df: pd.DataFrame, biography: UserBiography) -> pd.DataFrame:
    """Return the sublevel set of homes affordable to B."""

    working = df.copy()
    if biography.target_zip:
        local = working[working["zip_code"].astype(str) == str(biography.target_zip)]
        if len(local) >= 8:
            working = local
    restricted = working[
        (working["median_listing_price"] <= biography.max_purchase_price)
        & (working["ownership_cost_monthly"] <= biography.monthly_budget * 1.10)
    ]
    if len(restricted) < 8:
        restricted = working.nsmallest(min(max(8, len(working)), 80), "median_listing_price")
    return restricted


def _points(df: pd.DataFrame, max_points: int = 360) -> np.ndarray:
    sample = df.sample(min(max_points, len(df)), random_state=918) if len(df) > max_points else df
    matrix = sample[[column for column in FEATURE_COLUMNS if column in sample.columns]].to_numpy(dtype=float)
    return minmax_scale(matrix)


def buy_signal(
    df: pd.DataFrame,
    biography: UserBiography | None = None,
    date: str | pd.Timestamp | None = None,
    layers: int = 5,
    max_points: int = 360,
) -> dict[str, object]:
    """Compute the topological action integral S(B).

    The signal is

        S(B) = integral_0^infty Lambda_sub(t) - Lambda_full(t) dt

    evaluated on the H1 persistence landscape. Positive values indicate that
    the affordable sublevel set retains topological structure not explained by
    the full market, a proxy for an opportunity pocket.
    """

    biography = biography or UserBiography()
    working = add_topological_parameters(df) if "affordability_index" not in df.columns else df.copy()
    if date is not None:
        working = working[working["date"] == pd.Timestamp(date)]
    if working.empty:
        raise ValueError("No market observations available for the requested date.")

    restricted = restricted_affordable_market(working, biography)
    full_diagrams = persistence_diagrams(_points(working, max_points=max_points), maxdim=1)
    sub_diagrams = persistence_diagrams(_points(restricted, max_points=max_points), maxdim=1)

    full_h1 = full_diagrams["H1"]
    sub_h1 = sub_diagrams["H1"]
    max_death = 1.0
    for diagram in [full_h1, sub_h1, full_diagrams["H0"], sub_diagrams["H0"]]:
        finite = diagram[np.isfinite(diagram).all(axis=1)] if diagram.size else np.empty((0, 2))
        if finite.size:
            max_death = max(max_death, float(finite[:, 1].max()))
    xs = np.linspace(0.0, max_death, 240)
    full_landscape = persistence_landscape(full_h1, xs, layers=layers)
    sub_landscape = persistence_landscape(sub_h1, xs, layers=layers)
    if not np.any(sub_landscape):
        sub_landscape = persistence_landscape(sub_diagrams["H0"], xs, layers=layers)
        full_landscape = persistence_landscape(full_diagrams["H0"], xs, layers=layers)

    integrand = sub_landscape.mean(axis=0) - full_landscape.mean(axis=0)
    raw_signal = float(np.trapz(integrand, xs))
    normalized = float(np.tanh(12.0 * raw_signal))
    decision = "Buy Opportunity" if normalized > 0.08 else "Rent / Wait" if normalized < -0.08 else "Neutral"
    return {
        "biography": biography,
        "date": pd.Timestamp(date) if date is not None else pd.Timestamp(working["date"].max()),
        "restricted_count": int(len(restricted)),
        "full_count": int(len(working)),
        "S_B": raw_signal,
        "normalized_signal": normalized,
        "decision": decision,
        "grid": xs,
        "full_landscape": full_landscape,
        "sub_landscape": sub_landscape,
        "full_diagrams": full_diagrams,
        "sub_diagrams": sub_diagrams,
    }
