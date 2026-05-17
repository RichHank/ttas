"""Model validation for TTAS: compare calibrated predictions against real data.

When real public data (Realtor.com, FRED, Census) is available, this module
compares the calibrated synthetic manifold against observed aggregates.
When no real data exists, it runs internal consistency checks (subsample
stability, train/test split metrics).

All computations are local Python — no API calls.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def _safe_metric(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception:
        return float("nan")


def validate_price_predictions(df: pd.DataFrame) -> dict:
    """Compare calibrated synthetic prices vs Realtor.com observed aggregates.

    The synthetic manifold carries 'median_listing_price' (calibrated
    per-property). The real timeseries carries monthly aggregate medians.
    We compare monthly medians of the synthetic data against the real
    medians stored in the calibration source.

    Returns dict with keys:
        rmse, r2, mae, n_months, monthly_df, mode
    """
    real_ts = _load_real_timeseries()
    if real_ts is None or "median_listing_price" not in real_ts.columns:
        return _internal_price_consistency(df)

    real = real_ts[["date", "median_listing_price"]].dropna().copy()
    real["date"] = pd.to_datetime(real["date"])

    syn_med = df.groupby("date")["median_listing_price"].median().reset_index()
    syn_med.columns = ["date", "syn_median"]

    merged = real.merge(syn_med, on="date", how="inner")
    if len(merged) < 6:
        return _internal_price_consistency(df)

    y_true = merged["median_listing_price"].to_numpy(dtype=float)
    y_pred = merged["syn_median"].to_numpy(dtype=float)

    monthly = merged.copy()
    monthly["residual"] = y_true - y_pred
    monthly["abs_pct_error"] = np.abs(monthly["residual"]) / np.maximum(y_true, 1.0)

    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(_safe_metric(r2_score, y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "n_months": len(merged),
        "monthly_df": monthly,
        "mode": "real_vs_calibrated",
    }


def validate_rent_predictions(df: pd.DataFrame) -> dict:
    """Compare calibrated synthetic rents vs Realtor.com observed rents."""
    real_ts = _load_real_timeseries()
    if real_ts is None or "median_rent" not in real_ts.columns:
        return _internal_rent_consistency(df)

    real = real_ts[["date", "median_rent"]].dropna().copy()
    real["date"] = pd.to_datetime(real["date"])

    syn_med = df.groupby("date")["monthly_rent_estimate"].median().reset_index()
    syn_med.columns = ["date", "syn_median"]

    merged = real.merge(syn_med, on="date", how="inner")
    if len(merged) < 6:
        return _internal_rent_consistency(df)

    y_true = merged["median_rent"].to_numpy(dtype=float)
    y_pred = merged["syn_median"].to_numpy(dtype=float)

    monthly = merged.copy()
    monthly["residual"] = y_true - y_pred
    monthly["abs_pct_error"] = np.abs(monthly["residual"]) / np.maximum(y_true, 1.0)

    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(_safe_metric(r2_score, y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "n_months": len(merged),
        "monthly_df": monthly,
        "mode": "real_vs_calibrated",
    }


def validate_zip_level_errors(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-ZIP median price differences from overall median.

    Since real per-ZIP price data isn't always available, this compares
    each ZIP's median against the metro-wide median as a consistency check.
    Large outliers may indicate calibration issues.
    """
    metro_median = df["median_listing_price"].median()
    rows = []
    for zc in sorted(df["zip_code"].unique()):
        sub = df[df["zip_code"] == zc]
        zip_median = sub["median_listing_price"].median()
        rows.append({
            "zip_code": str(zc),
            "neighborhood": sub["neighborhood"].iloc[0] if not sub.empty else "",
            "median_price": round(zip_median, 0),
            "metro_deviation_pct": round((zip_median - metro_median) / metro_median * 100, 1),
            "n_properties": len(sub),
        })
    return pd.DataFrame(rows)


def validate_regime_consistency(df: pd.DataFrame) -> dict:
    """Compare GP regime predictions vs ground-truth regime_hint labels.

    The regime_hint column is a deterministic label based on month index.
    The GP classifier is trained on Euler surface features.
    Comparing them measures internal consistency of the topological pipeline.
    """
    if "regime_hint" not in df.columns:
        return {"accuracy": None, "confusion": None, "monthly_df": None, "mode": "no_ground_truth"}

    # Build per-month regime predictions from GP (re-use the dashboard computation)
    from decision.phase_transition import build_euler_regime_training_frame, train_gp_regime_classifier, predict_regime_with_gp
    from topology.invariants import compute_time_slice_invariants

    try:
        gp_model, gp_training = train_gp_regime_classifier(df, max_slices=12, max_points=140, grid_size=7)
    except Exception:
        return {"accuracy": None, "confusion": None, "monthly_df": None, "mode": "gp_training_failed"}

    monthly_rows = []
    for _, row in gp_training.iterrows():
        date = row["date"]
        euler = compute_time_slice_invariants(df, date=date, max_points=140, grid_size=7)["euler_surface"]
        pred = predict_regime_with_gp(gp_model, euler)
        actual = _remap_regime(row["regime"])
        monthly_rows.append({
            "date": date,
            "actual": actual,
            "predicted": pred["regime"],
            "match": actual == pred["regime"],
        })

    comparison = pd.DataFrame(monthly_rows)
    accuracy = comparison["match"].mean() if not comparison.empty else 0.0

    # Build confusion matrix
    labels = ["Stable", "Overheated", "Crash", "Opportunity"]
    matrix = []
    for actual_label in labels:
        row_vals = []
        for pred_label in labels:
            cnt = int(((comparison["actual"] == actual_label) & (comparison["predicted"] == pred_label)).sum())
            row_vals.append(cnt)
        matrix.append(row_vals)

    return {
        "accuracy": float(accuracy),
        "confusion_labels": labels,
        "confusion_matrix": matrix,
        "monthly_df": comparison,
        "mode": "gp_vs_regime_hint",
    }


def validate_topological_stability(df: pd.DataFrame, n_splits: int = 5, max_points: int = 120) -> dict:
    """Out-of-sample topological stability via subsampling.

    Subsamples 80% of properties at the latest date, recomputes persistence
    diagrams, and measures bottleneck drift from the full-sample diagram.
    Lower drift = more stable topology.
    """
    from topology.invariants import persistence_diagrams
    from topology.vineyards import bottleneck_distance
    from data.preprocess import prepare_feature_matrix

    latest = pd.Timestamp(df["date"].max())
    latest_df = df[df["date"] == latest]
    if len(latest_df) < 20:
        return {"mode": "insufficient_data"}

    X_full, _ = prepare_feature_matrix(latest_df)
    diagrams_full = persistence_diagrams(X_full)

    drifts_h0 = []
    drifts_h1 = []
    n_samples = min(n_splits, len(latest_df) // 20)

    for _ in range(n_samples):
        sub = latest_df.sample(frac=0.8, random_state=None)
        X_sub, _ = prepare_feature_matrix(sub)
        diagrams_sub = persistence_diagrams(X_sub)
        try:
            drifts_h0.append(bottleneck_distance(diagrams_full["H0"], diagrams_sub["H0"]))
            drifts_h1.append(bottleneck_distance(diagrams_full["H1"], diagrams_sub["H1"]))
        except Exception:
            continue

    return {
        "h0_drift_mean": float(np.mean(drifts_h0)) if drifts_h0 else None,
        "h0_drift_std": float(np.std(drifts_h0)) if drifts_h0 else None,
        "h1_drift_mean": float(np.mean(drifts_h1)) if drifts_h1 else None,
        "h1_drift_std": float(np.std(drifts_h1)) if drifts_h1 else None,
        "n_splits": n_samples,
        "mode": "subsample_stability",
    }


def build_validation_dashboard_data(df: pd.DataFrame) -> dict:
    """Master function returning all validation metrics for the dashboard.

    Returns a dict with keys:
        price, rent, zip_errors, regime, stability, data_mode, validation_mode
    """
    data_mode = "real_public_data" if _load_real_timeseries() is not None else "synthetic_fallback"

    price = validate_price_predictions(df)
    rent = validate_rent_predictions(df)
    zip_errors = validate_zip_level_errors(df)
    regime = validate_regime_consistency(df)
    stability = validate_topological_stability(df)

    validation_mode = price.get("mode", "internal")

    return {
        "price": price,
        "rent": rent,
        "zip_errors": zip_errors,
        "regime": regime,
        "stability": stability,
        "data_mode": data_mode,
        "validation_mode": validation_mode,
    }


# ── Internal consistency fallbacks (no real data) ─────────────────────────


def _load_real_timeseries() -> pd.DataFrame | None:
    try:
        from data.real_data import load_real_timeseries
        return load_real_timeseries()
    except Exception:
        return None


def _remap_regime(label: str) -> str:
    """Map regime_hint labels to canonical names used by the GP."""
    mapping = {"Rate Shock": "Crash"}
    return mapping.get(label, label)


def _internal_price_consistency(df: pd.DataFrame) -> dict:
    """Self-consistency check: train/test split on synthetic data."""
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor

    monthly = df.groupby("date")["median_listing_price"].median().reset_index()
    if len(monthly) < 12:
        return {"rmse": None, "r2": None, "mae": None, "n_months": len(monthly), "monthly_df": None, "mode": "internal_insufficient"}

    # Use last 20% of time as test
    split_idx = int(len(monthly) * 0.8)
    train = monthly.iloc[:split_idx]
    test = monthly.iloc[split_idx:]

    t = np.arange(len(train)).reshape(-1, 1)
    y = train["median_listing_price"].to_numpy(dtype=float)

    try:
        model = RandomForestRegressor(n_estimators=30, max_depth=4, random_state=918)
        model.fit(t, y)
        t_test = np.arange(split_idx, len(monthly)).reshape(-1, 1)
        y_pred = model.predict(t_test)
        y_test = test["median_listing_price"].to_numpy(dtype=float)

        monthly_out = test.copy()
        monthly_out["residual"] = y_test - y_pred
        monthly_out["abs_pct_error"] = np.abs(monthly_out["residual"]) / np.maximum(y_test, 1.0)

        return {
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "r2": float(_safe_metric(r2_score, y_test, y_pred)),
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "n_months": len(test),
            "monthly_df": monthly_out,
            "mode": "internal_train_test_split",
        }
    except Exception:
        return {"rmse": None, "r2": None, "mae": None, "n_months": len(monthly), "monthly_df": None, "mode": "internal_failed"}


def _internal_rent_consistency(df: pd.DataFrame) -> dict:
    """Self-consistency check for rent data (train/test split)."""
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor

    monthly = df.groupby("date")["monthly_rent_estimate"].median().reset_index()
    if len(monthly) < 12:
        return {"rmse": None, "r2": None, "mae": None, "n_months": len(monthly), "monthly_df": None, "mode": "internal_insufficient"}

    split_idx = int(len(monthly) * 0.8)
    train = monthly.iloc[:split_idx]
    test = monthly.iloc[split_idx:]

    t = np.arange(len(train)).reshape(-1, 1)
    y = train["monthly_rent_estimate"].to_numpy(dtype=float)

    try:
        model = RandomForestRegressor(n_estimators=30, max_depth=4, random_state=918)
        model.fit(t, y)
        t_test = np.arange(split_idx, len(monthly)).reshape(-1, 1)
        y_pred = model.predict(t_test)
        y_test = test["monthly_rent_estimate"].to_numpy(dtype=float)

        monthly_out = test.copy()
        monthly_out["residual"] = y_test - y_pred
        monthly_out["abs_pct_error"] = np.abs(monthly_out["residual"]) / np.maximum(y_test, 1.0)

        return {
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "r2": float(_safe_metric(r2_score, y_test, y_pred)),
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "n_months": len(test),
            "monthly_df": monthly_out,
            "mode": "internal_train_test_split",
        }
    except Exception:
        return {"rmse": None, "r2": None, "mae": None, "n_months": len(monthly), "monthly_df": None, "mode": "internal_failed"}
