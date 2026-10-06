from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from shelfcash_forecast.evaluation.metrics import quantile_crossing_rate


def calibration_metrics(
    frame: pd.DataFrame,
    desired_coverage: float,
) -> dict[str, Any]:
    if frame.empty:
        return {}

    actual = frame["target"].to_numpy(dtype=float)
    raw_lower = frame["p25"].to_numpy(dtype=float)
    raw_upper = frame["p75"].to_numpy(dtype=float)
    calibrated_lower = frame["interval_lower"].to_numpy(dtype=float)
    calibrated_upper = frame["interval_upper"].to_numpy(dtype=float)

    raw_coverage = float(np.mean((actual >= raw_lower) & (actual <= raw_upper)))
    calibrated_coverage = float(
        np.mean((actual >= calibrated_lower) & (actual <= calibrated_upper))
    )

    return {
        "row_count": int(len(frame)),
        "nominal_coverage": desired_coverage,
        "raw_coverage": raw_coverage,
        "calibrated_coverage": calibrated_coverage,
        "raw_coverage_gap": raw_coverage - desired_coverage,
        "calibrated_coverage_gap": calibrated_coverage - desired_coverage,
        "raw_average_width": float(np.mean(raw_upper - raw_lower)),
        "calibrated_average_width": float(
            np.mean(calibrated_upper - calibrated_lower)
        ),
        "raw_crossing_rate": quantile_crossing_rate(
            frame, ("p25_raw", "p50_raw", "p75_raw")
        ),
        "corrected_crossing_rate": quantile_crossing_rate(
            frame, ("p25", "p50", "p75")
        ),
        "calibrated_interval_invalid_rate": float(
            np.mean(calibrated_lower > calibrated_upper)
        ),
        "calibrated_interval_nonfinite_rate": float(
            np.mean(~np.isfinite(calibrated_lower) | ~np.isfinite(calibrated_upper))
        ),
        "calibrated_interval_excludes_median_rate": float(
            np.mean(
                (frame["p50"].to_numpy(dtype=float) < calibrated_lower)
                | (frame["p50"].to_numpy(dtype=float) > calibrated_upper)
            )
        ),
    }


def calibration_breakdown_by_horizon(
    frame: pd.DataFrame,
    desired_coverage: float,
) -> dict[str, dict[str, Any]]:
    return {
        str(int(horizon)): calibration_metrics(group, desired_coverage)
        for horizon, group in frame.groupby("horizon", observed=True)
    }
