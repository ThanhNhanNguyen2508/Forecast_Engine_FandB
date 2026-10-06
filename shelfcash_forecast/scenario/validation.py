from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _filled_pivot(
    frame: pd.DataFrame,
    index: str,
    columns: list[str],
) -> pd.DataFrame:
    pivot = frame.pivot_table(
        index=index,
        columns=columns,
        values="scaled_residual",
        aggfunc="first",
    )
    return pivot.apply(lambda column: column.fillna(column.median()), axis=0)


def scenario_reproduction_diagnostics(
    history: pd.DataFrame,
    sampled: pd.DataFrame,
    *,
    clipped_count: int,
    total_count: int,
) -> dict[str, Any]:
    """Compare sampled scaled residuals with the eligible residual history."""

    dimensions = ["store_id", "product_id", "horizon"]
    historical_groups = history.groupby(dimensions, observed=True)["scaled_residual"]
    sampled_groups = sampled.groupby(dimensions, observed=True)["scaled_residual"]
    common = sorted(set(historical_groups.groups) & set(sampled_groups.groups))
    mean_errors: list[float] = []
    quantile_errors: list[float] = []
    for key in common:
        historical_values = historical_groups.get_group(key).to_numpy(dtype=float)
        sampled_values = sampled_groups.get_group(key).to_numpy(dtype=float)
        mean_errors.append(abs(float(sampled_values.mean() - historical_values.mean())))
        quantile_errors.extend(
            np.abs(
                np.quantile(sampled_values, [0.25, 0.5, 0.75])
                - np.quantile(historical_values, [0.25, 0.5, 0.75])
            ).tolist()
        )

    correlation_error: float | None = None
    try:
        historical_pivot = _filled_pivot(history, "forecast_origin", dimensions)
        sampled_pivot = _filled_pivot(sampled, "scenario_id", dimensions)
        shared_columns = historical_pivot.columns.intersection(sampled_pivot.columns)
        if len(shared_columns) >= 2:
            historical_corr = historical_pivot[shared_columns].corr().to_numpy()
            sampled_corr = sampled_pivot[shared_columns].corr().to_numpy()
            finite = np.isfinite(historical_corr) & np.isfinite(sampled_corr)
            if finite.any():
                correlation_error = float(
                    np.mean(np.abs(historical_corr[finite] - sampled_corr[finite]))
                )
    except (KeyError, ValueError):
        correlation_error = None

    return {
        "marginal_mean_error": float(np.mean(mean_errors)) if mean_errors else None,
        "marginal_quantile_error": (
            float(np.mean(quantile_errors)) if quantile_errors else None
        ),
        "correlation_reproduction_error": correlation_error,
        "negative_clipping_count": clipped_count,
        "negative_clipping_rate": clipped_count / total_count if total_count else 0.0,
    }
