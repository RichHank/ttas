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
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel

        feature_columns = [column for column in training_frame.columns if column.startswith("f")]
        x = training_frame[feature_columns].to_numpy(dtype=float)
        y = training_frame["regime"].to_numpy()
        model = GaussianProcessClassifier(kernel=1.0 * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1), random_state=918)
        return model.fit(x, y)
    except Exception:
        return None
