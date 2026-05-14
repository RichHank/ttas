"""Causal topological inference for interest-rate shocks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.preprocess import FEATURE_COLUMNS, add_topological_parameters, minmax_scale
from .invariants import persistence_diagrams
from .vineyards import bottleneck_distance


def _ownership_cost(price: np.ndarray, mortgage_rate: np.ndarray, tax_rate: np.ndarray) -> np.ndarray:
    loan = 0.80 * price
    monthly_rate = mortgage_rate / 100.0 / 12.0
    factor = (1.0 + monthly_rate) ** 360
    principal_interest = loan * monthly_rate * factor / np.maximum(factor - 1.0, 1e-9)
    monthly_tax = price * tax_rate / 12.0
    insurance = price * 0.0042 / 12.0
    maintenance = price * 0.008 / 12.0
    return principal_interest + monthly_tax + insurance + maintenance


def counterfactual_interest_rate(df: pd.DataFrame, shock_bps: float = -100.0, price_elasticity: float = -0.018) -> pd.DataFrame:
    """Create a counterfactual market under an interest-rate shock.

    A negative shock_bps removes part of the factual rate increase. Prices are
    adjusted using a modest semi-elasticity, then affordability variables are
    recalculated before topology is recomputed.
    """

    cf = df.copy()
    shock_pct = shock_bps / 100.0
    cf["mortgage_rate_30y"] = np.clip(cf["mortgage_rate_30y"].to_numpy(dtype=float) + shock_pct, 1.5, 10.0)
    price_multiplier = np.exp(price_elasticity * shock_pct)
    cf["median_listing_price"] = cf["median_listing_price"].to_numpy(dtype=float) * price_multiplier
    annual_ratio = cf["rent_to_price_ratio"].to_numpy(dtype=float)
    cf["monthly_rent_estimate"] = cf["median_listing_price"].to_numpy(dtype=float) * annual_ratio / 12.0
    cf["ownership_cost_monthly"] = _ownership_cost(
        cf["median_listing_price"].to_numpy(dtype=float),
        cf["mortgage_rate_30y"].to_numpy(dtype=float),
        cf["property_tax_rate"].to_numpy(dtype=float),
    )
    max_budget = cf["annual_income_estimate"].to_numpy(dtype=float) * cf["dti_max"].to_numpy(dtype=float) / 12.0
    cf["buy_margin_monthly"] = max_budget - cf["ownership_cost_monthly"].to_numpy(dtype=float)
    cf["rent_margin_monthly"] = max_budget - cf["monthly_rent_estimate"].to_numpy(dtype=float)
    cf = cf.drop(columns=[column for column in ["affordability_index", "spatial_density", "opportunity_score"] if column in cf.columns])
    return add_topological_parameters(cf)


def _points(df: pd.DataFrame, max_points: int = 360) -> np.ndarray:
    sample = df.sample(min(max_points, len(df)), random_state=918) if len(df) > max_points else df
    columns = [column for column in FEATURE_COLUMNS if column in sample.columns]
    return minmax_scale(sample[columns].to_numpy(dtype=float))


def topological_ate(
    df: pd.DataFrame,
    shock_bps: float = -100.0,
    date: str | pd.Timestamp | None = None,
    max_points: int = 360,
) -> dict[str, object]:
    """Estimate the topological average treatment effect of a rate shock."""

    working = add_topological_parameters(df) if "affordability_index" not in df.columns else df.copy()
    if date is not None:
        working = working[working["date"] == pd.Timestamp(date)]
    factual = _points(working, max_points=max_points)
    counterfactual_df = counterfactual_interest_rate(working, shock_bps=shock_bps)
    counterfactual = _points(counterfactual_df, max_points=max_points)
    factual_diagrams = persistence_diagrams(factual, maxdim=1)
    counterfactual_diagrams = persistence_diagrams(counterfactual, maxdim=1)
    h0 = bottleneck_distance(factual_diagrams["H0"], counterfactual_diagrams["H0"])
    h1 = bottleneck_distance(factual_diagrams["H1"], counterfactual_diagrams["H1"])
    return {
        "shock_bps": shock_bps,
        "date": pd.Timestamp(date) if date is not None else None,
        "topological_ate_h0": h0,
        "topological_ate_h1": h1,
        "factual_diagrams": factual_diagrams,
        "counterfactual_diagrams": counterfactual_diagrams,
    }


def transfer_entropy(source: np.ndarray, target: np.ndarray, bins: int = 5) -> float:
    """Estimate discrete transfer entropy T_{source -> target}.

    The estimator computes

        sum p(y_{t+1}, y_t, x_t) log p(y_{t+1} | y_t, x_t) / p(y_{t+1} | y_t).
    """

    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    n = min(len(source), len(target))
    if n < 4:
        return 0.0
    source = np.digitize(source[:n], np.quantile(source[:n], np.linspace(0, 1, bins + 1)[1:-1]))
    target = np.digitize(target[:n], np.quantile(target[:n], np.linspace(0, 1, bins + 1)[1:-1]))

    counts_xyz: dict[tuple[int, int, int], int] = {}
    counts_yx: dict[tuple[int, int], int] = {}
    counts_yy: dict[tuple[int, int], int] = {}
    counts_y: dict[int, int] = {}
    for t in range(n - 1):
        key_xyz = (int(target[t + 1]), int(target[t]), int(source[t]))
        key_yx = (int(target[t]), int(source[t]))
        key_yy = (int(target[t + 1]), int(target[t]))
        key_y = int(target[t])
        counts_xyz[key_xyz] = counts_xyz.get(key_xyz, 0) + 1
        counts_yx[key_yx] = counts_yx.get(key_yx, 0) + 1
        counts_yy[key_yy] = counts_yy.get(key_yy, 0) + 1
        counts_y[key_y] = counts_y.get(key_y, 0) + 1

    total = float(n - 1)
    te = 0.0
    for (y_next, y_now, x_now), count in counts_xyz.items():
        p_xyz = count / total
        p_y_next_given_yx = count / counts_yx[(y_now, x_now)]
        p_y_next_given_y = counts_yy.get((y_next, y_now), 0) / counts_y[y_now]
        if p_y_next_given_y > 0:
            te += p_xyz * np.log(p_y_next_given_yx / p_y_next_given_y)
    return float(max(te, 0.0))
