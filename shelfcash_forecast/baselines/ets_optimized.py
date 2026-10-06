from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from shelfcash_forecast.baselines.ets import (
    MINIMUM_ETS_HISTORY,
    _history_for_row,
    _prepare_daily_history,
)


ETS_OPTIMIZED_FIT_METHOD = (
    "statsmodels_additive_trend_seasonal7_optimized_trailing_contiguous_v1"
)


@dataclass(frozen=True)
class OptimizedETSAttempt:
    prediction: np.ndarray | None
    warning: str | None
    history_count: int
    converged: bool | None


def _fit_optimized(history: pd.Series, maximum_horizon: int) -> OptimizedETSAttempt:
    daily, preparation_warning = _prepare_daily_history(history)
    if daily is None:
        return OptimizedETSAttempt(
            prediction=None,
            warning=preparation_warning,
            history_count=int(history.notna().sum()),
            converged=None,
        )
    if len(daily) < MINIMUM_ETS_HISTORY:
        return OptimizedETSAttempt(None, "ETS_HISTORY_TOO_SHORT", len(daily), None)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fitted = ExponentialSmoothing(
                daily.to_numpy(dtype=float),
                trend="add",
                seasonal="add",
                seasonal_periods=7,
                initialization_method="estimated",
            ).fit(optimized=True, remove_bias=False, use_brute=True)
            prediction = np.asarray(fitted.forecast(maximum_horizon), dtype=float)
        if not np.isfinite(prediction).all():
            return OptimizedETSAttempt(
                None, "ETS_OPTIMIZED_NONFINITE", len(daily), False
            )
        mle_retvals = getattr(fitted, "mle_retvals", {}) or {}
        converged = bool(mle_retvals.get("success", True))
        warning = None
        if caught:
            warning = "ETS_OPTIMIZED_WARNING:" + type(caught[0].message).__name__
        if not converged:
            warning = "ETS_OPTIMIZED_NOT_CONVERGED"
        return OptimizedETSAttempt(
            np.maximum(0.0, prediction), warning, len(daily), converged
        )
    except (ValueError, RuntimeError, FloatingPointError, OverflowError) as exc:
        return OptimizedETSAttempt(
            None,
            f"ETS_OPTIMIZED_FIT_FAILED:{type(exc).__name__}",
            len(daily),
            False,
        )


def ets_optimized_predict_rows_detailed(
    rows: pd.DataFrame,
    panel: pd.DataFrame,
    fallback: pd.Series,
) -> pd.DataFrame:
    """Optimized statsmodels ETS with the same causal history/fallback contract."""

    demand_column = "feature_demand" if "feature_demand" in panel else "demand_proxy"
    maximum_horizon = int(rows["horizon"].max())
    cache: dict[tuple[str, str, pd.Timestamp], OptimizedETSAttempt] = {}
    output: list[float] = []
    attempts: list[OptimizedETSAttempt] = []
    for position, row in enumerate(rows.itertuples(index=False)):
        key = (
            str(row.store_key),
            str(row.product_key),
            pd.Timestamp(row.cutoff_date),
        )
        if key not in cache:
            history = _history_for_row(
                panel,
                demand_column,
                row.store_key,
                row.product_key,
                row.cutoff_date,
            )
            cache[key] = _fit_optimized(history, maximum_horizon)
        attempt = cache[key]
        value = (
            None
            if attempt.prediction is None
            else float(attempt.prediction[int(row.horizon) - 1])
        )
        output.append(float(fallback.iloc[position]) if value is None else value)
        attempts.append(attempt)
    return pd.DataFrame(
        {
            "baseline_name": "ETS_OPTIMIZED",
            "prediction": output,
            "fallback_used": [attempt.prediction is None for attempt in attempts],
            "fallback_model": [
                "SEASONAL_NAIVE" if attempt.prediction is None else None
                for attempt in attempts
            ],
            "fallback_reason": [attempt.warning for attempt in attempts],
            "history_count": [attempt.history_count for attempt in attempts],
            "fit_method": ETS_OPTIMIZED_FIT_METHOD,
            "converged": [attempt.converged for attempt in attempts],
            "warnings": [attempt.warning or "" for attempt in attempts],
        },
        index=rows.index,
    )
