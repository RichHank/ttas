"""Euler characteristic phase transition detection."""

from __future__ import annotations

import numpy as np
import pandas as pd


REGIME_ORDER = ["Stable", "Overheated", "Crash", "Opportunity"]


def euler_curvature(euler_surface: np.ndarray) -> np.ndarray:
    """Approximate local curvature of chi by second finite differences."""

    chi = np.asarray(euler_surface, dtype=float)
    if min(chi.shape) < 3:
        return np.zeros_like(chi)
    gradients = np.gradient(chi)
    curvature = np.zeros_like(chi)
    for gradient in gradients:
        second = np.gradient(gradient)
        for component in second:
            curvature += component * component
    return np.sqrt(curvature)


def euler_curvature_alert(
    euler_surface: np.ndarray,
    quantile: float = 0.92,
    critical_value: float | None = None,
) -> dict[str, object]:
    """Flag phase transitions where Euler curvature is unusually high."""

    curvature = euler_curvature(euler_surface)
    threshold = float(critical_value if critical_value is not None else np.quantile(curvature, quantile))
    peak = float(np.max(curvature)) if curvature.size else 0.0
    index = tuple(int(i) for i in np.unravel_index(np.argmax(curvature), curvature.shape)) if curvature.size else (0, 0, 0)
    return {
        "alert": bool(peak >= threshold and peak > 0),
        "peak_curvature": peak,
        "critical_value": threshold,
        "peak_index": index,
        "curvature": curvature,
    }


def euler_feature_vector(euler_surface: np.ndarray) -> np.ndarray:
    """Compress chi(lambda) into classifier-ready summary features."""

    chi = np.asarray(euler_surface, dtype=float)
    curvature = euler_curvature(chi)
    return np.array(
        [
            float(np.mean(chi)),
            float(np.std(chi)),
            float(np.min(chi)),
            float(np.max(chi)),
            float(np.quantile(chi, 0.15)),
            float(np.quantile(chi, 0.85)),
            float(np.mean(curvature)),
            float(np.max(curvature)),
        ],
        dtype=float,
    )


def infer_market_regime(euler_surface: np.ndarray, rate_level: float | None = None) -> str:
    """Infer a regime label from Euler features and optional rate context."""

    features = euler_feature_vector(euler_surface)
    chi_spread = features[3] - features[2]
    curvature_peak = features[7]
    if rate_level is not None and rate_level >= 6.6 and curvature_peak >= 2.0:
        return "Crash"
    if curvature_peak >= 2.6 or chi_spread >= 40:
        return "Overheated"
    if features[0] > 8.0 and features[6] < 0.85:
        return "Opportunity"
    return "Stable"


def fit_gaussian_process_regime_classifier(training_frame: pd.DataFrame):
    """Fit a GP classifier when scikit-learn is available.

    The expected frame has columns f0...f7 plus a `regime` label. The fallback
    returns None and callers can use `infer_market_regime`.
    """

    try:
        from sklearn.gaussian_process import GaussianProcessClassifier
        from sklearn.gaussian_process.kernels import RBF

        feature_columns = [column for column in training_frame.columns if column.startswith("f")]
        x = training_frame[feature_columns].to_numpy(dtype=float)
        y = training_frame["regime"].to_numpy()
        model = GaussianProcessClassifier(kernel=1.0 * RBF(length_scale=1.0), random_state=918)
        return model.fit(x, y)
    except Exception:
        return None


def build_euler_regime_training_frame(
    df: pd.DataFrame,
    max_slices: int = 14,
    max_points: int = 160,
    grid_size: int = 7,
) -> pd.DataFrame:
    """Build supervised Euler-surface features for GP regime training."""

    from topology.invariants import compute_time_slice_invariants

    months = sorted(pd.to_datetime(df["date"].unique()))
    if len(months) > max_slices:
        indices = np.unique(np.linspace(0, len(months) - 1, max_slices).astype(int))
        months = [months[int(index)] for index in indices]
    records = []
    for month in months:
        invariants = compute_time_slice_invariants(df, date=month, max_points=max_points, grid_size=grid_size)
        features = euler_feature_vector(invariants["euler_surface"])
        slice_df = df[df["date"] == month]
        label = str(slice_df["regime_hint"].mode().iloc[0]) if "regime_hint" in slice_df and not slice_df.empty else infer_market_regime(invariants["euler_surface"])
        label = "Crash" if label == "Rate Shock" else label
        record = {"date": pd.Timestamp(month), "regime": label}
        record.update({f"f{i}": float(value) for i, value in enumerate(features)})
        records.append(record)
    return pd.DataFrame(records)


def train_gp_regime_classifier(
    df: pd.DataFrame,
    max_slices: int = 14,
    max_points: int = 160,
    grid_size: int = 7,
) -> tuple[object | None, pd.DataFrame]:
    """Train a Gaussian Process classifier from monthly Euler features."""

    frame = build_euler_regime_training_frame(df, max_slices=max_slices, max_points=max_points, grid_size=grid_size)
    model = fit_gaussian_process_regime_classifier(frame) if not frame.empty and frame["regime"].nunique() > 1 else None
    return model, frame


def predict_regime_with_gp(model: object | None, euler_surface: np.ndarray) -> dict[str, object]:
    """Predict regime with a trained GP, falling back to deterministic rules."""

    if model is None:
        return {"regime": infer_market_regime(euler_surface), "backend": "threshold-fallback", "probabilities": {}}
    features = euler_feature_vector(euler_surface).reshape(1, -1)
    try:
        regime = str(model.predict(features)[0])
        probabilities = {}
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(features)[0]
            probabilities = {str(label): float(prob) for label, prob in zip(model.classes_, probs)}
        return {"regime": regime, "backend": "gaussian-process", "probabilities": probabilities}
    except Exception:
        return {"regime": infer_market_regime(euler_surface), "backend": "threshold-fallback", "probabilities": {}}
