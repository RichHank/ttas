"""Biography-space topological decision boundary."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .path_integral import UserBiography, buy_signal
from topology.invariants import finite_intervals


def _longest_h1(signal: dict[str, object]) -> float:
    diagram = signal["sub_diagrams"].get("H1", np.empty((0, 2)))
    finite = finite_intervals(diagram)
    if finite.size == 0:
        return 0.0
    return float(np.max(finite[:, 1] - finite[:, 0]))


def compute_topological_boundary(
    df: pd.DataFrame,
    date: str | pd.Timestamp | None = None,
    income_values: np.ndarray | None = None,
    dti_values: np.ndarray | None = None,
    family_size: int = 3,
    max_points: int = 180,
) -> pd.DataFrame:
    """Sample biography-space and identify topology-changing cells.

    A point is marked as a boundary point when one of its grid neighbors changes
    decision label or crosses the observed H1-persistence median. This gives the
    dashboard a genuine decision surface instead of a single scalar S(B).
    """

    income_values = income_values if income_values is not None else np.linspace(55_000, 165_000, 13)
    dti_values = dti_values if dti_values is not None else np.linspace(0.28, 0.46, 10)
    records = []
    for income in income_values:
        for dti in dti_values:
            biography = UserBiography(annual_income=float(income), dti_max=float(dti), family_size=family_size)
            signal = buy_signal(df, biography=biography, date=date, max_points=max_points)
            longest_h1 = _longest_h1(signal)
            records.append(
                {
                    "annual_income": float(income),
                    "dti_max": float(dti),
                    "family_size": int(family_size),
                    "S_B": float(signal["S_B"]),
                    "normalized_signal": float(signal["normalized_signal"]),
                    "decision": signal["decision"],
                    "restricted_count": int(signal["restricted_count"]),
                    "h1_longest_persistence": longest_h1,
                    "h1_active": bool(longest_h1 > 0.0),
                }
            )
    frame = pd.DataFrame(records)
    if frame.empty:
        frame["boundary"] = False
        return frame

    h1_threshold = float(frame["h1_longest_persistence"].median())
    lookup = {
        (float(row.annual_income), float(row.dti_max)): row
        for row in frame.itertuples(index=False)
    }
    income_sorted = sorted(float(x) for x in income_values)
    dti_sorted = sorted(float(x) for x in dti_values)
    boundary_flags = []
    for row in frame.itertuples(index=False):
        i = income_sorted.index(float(row.annual_income))
        j = dti_sorted.index(float(row.dti_max))
        neighbors = []
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if 0 <= i + di < len(income_sorted) and 0 <= j + dj < len(dti_sorted):
                neighbors.append(lookup[(income_sorted[i + di], dti_sorted[j + dj])])
        is_boundary = False
        for neighbor in neighbors:
            decision_changed = neighbor.decision != row.decision
            h1_changed = (neighbor.h1_longest_persistence >= h1_threshold) != (row.h1_longest_persistence >= h1_threshold)
            signal_changed = np.sign(neighbor.normalized_signal) != np.sign(row.normalized_signal)
            is_boundary = is_boundary or decision_changed or h1_changed or signal_changed
        boundary_flags.append(bool(is_boundary))
    frame["boundary"] = boundary_flags
    frame["h1_threshold"] = h1_threshold
    return frame
