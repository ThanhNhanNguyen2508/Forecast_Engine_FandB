from __future__ import annotations

import numpy as np
import pandas as pd

from shelfcash_forecast.config import ForecastConfig

SERIES_COLUMNS = ["store_key", "product_key"]


def _confidence(
    references: pd.DataFrame,
    target_date: pd.Timestamp,
    *,
    same_weekday: bool,
    minimum_reference_count: int,
    high_support_threshold: int,
    high_recency_days: int,
    medium_recency_days: int,
    high_dispersion_threshold: float,
    medium_dispersion_threshold: float,
) -> tuple[str, tuple[str, ...], float, int]:
    values = references["quantity_sold"].astype(float)
    median = float(values.median())
    q25, q75 = np.quantile(values, [0.25, 0.75])
    relative_iqr = float((q75 - q25) / max(abs(median), 1.0))
    recency_days = int((target_date - references["date"].max()).days)
    if (
        same_weekday
        and len(references) >= high_support_threshold
        and recency_days <= high_recency_days
        and relative_iqr <= high_dispersion_threshold
    ):
        return "high", (), relative_iqr, recency_days
    if (
        len(references) >= minimum_reference_count
        and recency_days <= medium_recency_days
        and relative_iqr <= medium_dispersion_threshold
    ):
        return "medium", (), relative_iqr, recency_days

    reasons: list[str] = []
    if len(references) < minimum_reference_count:
        reasons.append("RECONSTRUCTION_INSUFFICIENT_SUPPORT")
    if recency_days > medium_recency_days:
        reasons.append("RECONSTRUCTION_STALE_HISTORY")
    if relative_iqr > medium_dispersion_threshold:
        reasons.append("RECONSTRUCTION_HIGH_DISPERSION")
    return "low", tuple(reasons), relative_iqr, recency_days


def _settings(
    config: ForecastConfig | None,
    minimum_reference_count: int | None,
    lookback_days: int | None,
) -> tuple[int, int]:
    minimum = minimum_reference_count
    lookback = lookback_days
    if config is not None:
        minimum = minimum or config.reconstruction_minimum_reference_count
        lookback = lookback or config.reconstruction_lookback_days
    return minimum or 3, lookback or 84


def reconstruct_demand(
    panel: pd.DataFrame,
    config: ForecastConfig | None = None,
    *,
    minimum_reference_count: int | None = None,
    lookback_days: int | None = None,
) -> pd.DataFrame:
    """Reconstruct censored demand from strictly earlier, non-censored sales.

    Raw ``quantity_sold`` remains unchanged. Accepted reconstructions become both
    train targets and historical feature state; unsupported censored rows remain
    excluded and do not inject their known lower bound into future lag features.
    """

    settings = config or ForecastConfig()
    minimum, lookback = _settings(
        settings, minimum_reference_count, lookback_days
    )
    frame = panel.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame = frame.sort_values(SERIES_COLUMNS + ["date"], kind="mergesort")

    stockout_known = frame["is_stockout"].notna()
    stockout_true = frame["is_stockout"].fillna(False).astype(bool)
    observed = frame["quantity_sold"].astype("Float64")
    frame["observed_sales"] = observed
    frame["target_is_censored"] = frame["store_open"] & stockout_known & stockout_true
    frame["is_censored"] = frame["target_is_censored"]
    frame["reconstructed_demand"] = observed
    frame["reconstruction_applied"] = False
    frame["reconstruction_method"] = "not_censored"
    frame["reconstruction_confidence"] = "not_applicable"
    frame["reconstruction_reference_count"] = 0
    frame["reconstruction_reference_latest_date"] = pd.NaT
    frame["reconstruction_relative_iqr"] = np.nan
    frame["reconstruction_reference_recency_days"] = pd.NA
    frame["reconstruction_candidate_estimate"] = np.nan
    frame["reconstruction_rejection_reasons"] = [
        () for _ in range(len(frame))
    ]

    valid_reference = (
        frame["store_open"]
        & frame["row_observed"]
        & observed.notna()
        & stockout_known
        & ~stockout_true
    )
    for row_index in frame.index[frame["target_is_censored"]]:
        row = frame.loc[row_index]
        target_date = pd.Timestamp(row["date"])
        start_date = target_date - pd.Timedelta(days=lookback)
        pool = frame.loc[
            valid_reference
            & (frame["store_key"] == row["store_key"])
            & (frame["product_key"] == row["product_key"])
            & (frame["date"] < target_date)
            & (frame["date"] >= start_date)
        ]
        weekday_pool = pool.loc[pool["date"].dt.weekday == target_date.weekday()]
        if len(weekday_pool) >= minimum:
            references = weekday_pool
            method = "same_store_product_weekday_median"
            same_weekday = True
        elif len(pool) >= minimum:
            references = pool
            method = "same_store_product_recent_median"
            same_weekday = False
        else:
            frame.at[row_index, "reconstruction_method"] = "insufficient_history"
            frame.at[row_index, "reconstruction_confidence"] = "low"
            frame.at[row_index, "reconstruction_reference_count"] = len(pool)
            frame.at[row_index, "reconstruction_rejection_reasons"] = (
                "RECONSTRUCTION_INSUFFICIENT_SUPPORT",
            )
            if not pool.empty:
                frame.at[row_index, "reconstruction_reference_latest_date"] = pool[
                    "date"
                ].max()
            continue

        confidence, rejection_reasons, relative_iqr, recency_days = _confidence(
            references,
            target_date,
            same_weekday=same_weekday,
            minimum_reference_count=minimum,
            high_support_threshold=settings.reconstruction_high_support_threshold,
            high_recency_days=settings.reconstruction_high_recency_days,
            medium_recency_days=settings.reconstruction_medium_recency_days,
            high_dispersion_threshold=(
                settings.reconstruction_high_dispersion_threshold
            ),
            medium_dispersion_threshold=(
                settings.reconstruction_medium_dispersion_threshold
            ),
        )
        estimate = max(
            float(row["quantity_sold"]),
            float(references["quantity_sold"].median()),
        )
        frame.at[row_index, "reconstruction_method"] = method
        frame.at[row_index, "reconstruction_confidence"] = confidence
        frame.at[row_index, "reconstruction_reference_count"] = len(references)
        frame.at[row_index, "reconstruction_reference_latest_date"] = references["date"].max()
        frame.at[row_index, "reconstruction_relative_iqr"] = relative_iqr
        frame.at[row_index, "reconstruction_reference_recency_days"] = recency_days
        frame.at[row_index, "reconstruction_candidate_estimate"] = estimate
        frame.at[row_index, "reconstruction_rejection_reasons"] = rejection_reasons
        if confidence in {"high", "medium"}:
            frame.at[row_index, "reconstructed_demand"] = estimate
            frame.at[row_index, "reconstruction_applied"] = True

    frame["target_quality"] = "observed"
    frame.loc[~frame["store_open"], "target_quality"] = "store_closed"
    frame.loc[frame["target_is_censored"], "target_quality"] = "stockout_censored"
    frame.loc[
        frame["target_is_censored"] & frame["reconstruction_applied"],
        "target_quality",
    ] = "stockout_reconstructed"
    frame.loc[frame["store_open"] & ~frame["row_observed"], "target_quality"] = (
        "missing_open_day"
    )
    frame.loc[
        frame["store_open"] & frame["row_observed"] & frame["is_stockout"].isna(),
        "target_quality",
    ] = "stockout_unknown"

    accepted = frame["reconstruction_applied"]
    frame["demand_proxy"] = frame["reconstructed_demand"].astype("Float64")
    frame["feature_demand"] = frame["demand_proxy"]
    frame.loc[frame["target_is_censored"] & ~accepted, "feature_demand"] = pd.NA
    frame["train_eligible"] = (
        frame["store_open"]
        & frame["demand_proxy"].notna()
        & (~frame["target_is_censored"] | accepted)
    )

    if (
        frame.loc[frame["target_is_censored"], "reconstructed_demand"]
        < frame.loc[frame["target_is_censored"], "observed_sales"]
    ).any():
        raise RuntimeError("Reconstructed demand cannot be below observed sales.")
    if not np.isfinite(frame.loc[frame["demand_proxy"].notna(), "demand_proxy"]).all():
        raise RuntimeError("Reconstructed demand must be finite.")
    return frame.sort_index()
