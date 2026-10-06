from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

MINIMUM_ETS_HISTORY = 28
ETS_FIT_METHOD = "bounded_fixed_parameter_grid_trailing_contiguous_v2"
ETS_PARAMETER_GRID = (
    (0.20, 0.05, 0.10),
    (0.40, 0.10, 0.20),
    (0.60, 0.10, 0.20),
    (0.80, 0.20, 0.10),
)


@dataclass(frozen=True)
class ETSForecastAttempt:
    prediction: np.ndarray | None
    warning: str | None
    fit_method: str
    history_count: int


def _prepare_daily_history(history: pd.Series) -> tuple[pd.Series | None, str | None]:
    if not isinstance(history.index, pd.DatetimeIndex):
        return None, "ETS_HISTORY_DATES_UNAVAILABLE"
    clean = history.copy()
    clean.index = pd.to_datetime(clean.index).normalize()
    clean = clean.sort_index()
    if clean.index.has_duplicates:
        return None, "ETS_HISTORY_NOT_CONTIGUOUS"
    if clean.empty:
        return None, "ETS_HISTORY_TOO_SHORT"
    expected = pd.date_range(clean.index.min(), clean.index.max(), freq="D")
    daily = clean.reindex(expected)
    invalid = daily.isna()
    if invalid.any():
        # An ancient missing/closed day must not disable ETS forever.  Use the
        # latest fully observed daily suffix, without imputing a value or using
        # anything after the forecast origin.
        last_invalid_position = int(np.flatnonzero(invalid.to_numpy())[-1])
        daily = daily.iloc[last_invalid_position + 1 :]
    if len(daily) < MINIMUM_ETS_HISTORY:
        return None, "ETS_TRAILING_CONTIGUOUS_HISTORY_TOO_SHORT"
    values = daily.astype(float)
    if not np.isfinite(values.to_numpy()).all():
        return None, "ETS_HISTORY_NON_FINITE"
    return values, None


def _forecast_ets_attempt(
    history: pd.Series,
    maximum_horizon: int,
) -> ETSForecastAttempt:
    daily, warning = _prepare_daily_history(history)
    history_count = int(history.notna().sum())
    if daily is None:
        return ETSForecastAttempt(None, warning, ETS_FIT_METHOD, history_count)
    history_count = len(daily)

    best_sse = float("inf")
    best_prediction: np.ndarray | None = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for alpha, beta, gamma in ETS_PARAMETER_GRID:
            try:
                model = ExponentialSmoothing(
                    daily.to_numpy(),
                    trend="add",
                    seasonal="add",
                    seasonal_periods=7,
                    initialization_method="estimated",
                )
                fitted = model.fit(
                    smoothing_level=alpha,
                    smoothing_trend=beta,
                    smoothing_seasonal=gamma,
                    optimized=False,
                    remove_bias=False,
                )
                sse = float(fitted.sse)
                prediction = np.asarray(
                    fitted.forecast(maximum_horizon), dtype=float
                )
            except (ValueError, RuntimeError, FloatingPointError, OverflowError):
                continue
            if np.isfinite(sse) and np.isfinite(prediction).all() and sse < best_sse:
                best_sse = sse
                best_prediction = np.maximum(0.0, prediction)
    return ETSForecastAttempt(
        best_prediction,
        None if best_prediction is not None else "ETS_FIT_FAILED",
        ETS_FIT_METHOD,
        history_count,
    )


def forecast_ets_path(
    history: pd.Series,
    maximum_horizon: int,
) -> np.ndarray | None:
    """Deterministic bounded additive Holt-Winters path without SciPy minimization."""

    return _forecast_ets_attempt(history, maximum_horizon).prediction


def forecast_ets_series(history: pd.Series, horizon: int) -> float | None:
    path = forecast_ets_path(history, horizon)
    return None if path is None else float(path[horizon - 1])


def ets_predict_rows(
    rows: pd.DataFrame,
    panel: pd.DataFrame,
    fallback: pd.Series,
) -> pd.Series:
    return ets_predict_rows_detailed(rows, panel, fallback)["prediction"]


def _history_for_row(
    panel: pd.DataFrame,
    demand_column: str,
    store_key: object,
    product_key: object,
    cutoff_date: object,
) -> pd.Series:
    selected = panel.loc[
        panel["store_key"].eq(store_key)
        & panel["product_key"].eq(product_key)
        & panel["date"].le(cutoff_date),
        ["date", demand_column],
    ].sort_values("date")
    return pd.Series(
        selected[demand_column].to_numpy(),
        index=pd.DatetimeIndex(selected["date"]),
        dtype=float,
    )


def ets_predict_rows_detailed(
    rows: pd.DataFrame,
    panel: pd.DataFrame,
    fallback: pd.Series,
) -> pd.DataFrame:
    """ETS predictions with invocation-local cache and explicit fallback provenance."""

    indexed_panel = panel.sort_values("date")
    demand_column = "feature_demand" if "feature_demand" in panel else "demand_proxy"
    maximum_horizon = int(rows["horizon"].max())
    cache: dict[tuple[str, str, pd.Timestamp], ETSForecastAttempt] = {}
    output: list[float] = []
    attempts: list[ETSForecastAttempt] = []

    for position, row in enumerate(rows.itertuples(index=False)):
        key = (
            str(row.store_key),
            str(row.product_key),
            pd.Timestamp(row.cutoff_date),
        )
        if key not in cache:
            history = _history_for_row(
                indexed_panel,
                demand_column,
                row.store_key,
                row.product_key,
                row.cutoff_date,
            )
            cache[key] = _forecast_ets_attempt(history, maximum_horizon)
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
            "baseline_name": "ETS_HOLT_WINTERS",
            "prediction": output,
            "fallback_used": [attempt.prediction is None for attempt in attempts],
            "fallback_model": [
                "SEASONAL_NAIVE" if attempt.prediction is None else None
                for attempt in attempts
            ],
            "fallback_reason": [attempt.warning for attempt in attempts],
            "history_count": [attempt.history_count for attempt in attempts],
            "fit_method": [attempt.fit_method for attempt in attempts],
            "warnings": [attempt.warning or "" for attempt in attempts],
        },
        index=rows.index,
    )
