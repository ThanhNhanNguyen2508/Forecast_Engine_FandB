from __future__ import annotations

import pandas as pd

from shelfcash_forecast.config import ForecastConfig

CUTOFF_FEATURE_COLUMNS = [
    "date",
    "store_key",
    "product_key",
    "product_name",
    "unit",
    "history_observation_count",
    "last_observed_demand",
    "cutoff_lag_1",
    "cutoff_lag_2",
    "cutoff_lag_7",
    "cutoff_lag_14",
    "cutoff_lag_28",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_28",
    "rolling_count_7",
    "rolling_count_14",
    "rolling_count_28",
    "rolling_median_7",
    "rolling_median_14",
    "rolling_median_28",
    "rolling_std_7",
    "rolling_std_14",
    "rolling_std_28",
    "mean_last_7_minus_previous_7",
    "stockout_count_7",
    "stockout_count_28",
    "stockout_rate_7",
    "stockout_rate_28",
]


def build_training_table(
    feature_panel: pd.DataFrame,
    config: ForecastConfig,
) -> pd.DataFrame:
    cutoff = feature_panel[CUTOFF_FEATURE_COLUMNS].rename(columns={"date": "cutoff_date"})
    targets = feature_panel[
        [
            "date",
            "store_key",
            "product_key",
            "demand_proxy",
            "observed_sales",
            "reconstructed_demand",
            "target_is_censored",
            "reconstruction_applied",
            "train_eligible",
            "target_quality",
        ]
    ].rename(
        columns={
            "date": "target_date",
            "demand_proxy": "target",
            "train_eligible": "target_train_eligible",
        }
    )

    frames: list[pd.DataFrame] = []
    for horizon in config.horizons:
        current = cutoff.copy()
        current["horizon"] = horizon
        current["target_date"] = current["cutoff_date"] + pd.to_timedelta(
            horizon, unit="D"
        )
        current = current.merge(
            targets,
            on=["target_date", "store_key", "product_key"],
            how="left",
            validate="many_to_one",
        )
        frames.append(current)

    return pd.concat(frames, ignore_index=True).sort_values(
        ["target_date", "store_key", "product_key", "horizon"]
    ).reset_index(drop=True)


def add_target_seasonal_lags(
    rows: pd.DataFrame,
    panel: pd.DataFrame,
    config: ForecastConfig,
) -> pd.DataFrame:
    result = rows.copy()
    demand_column = "feature_demand" if "feature_demand" in panel else "demand_proxy"
    lookup = panel[["date", "store_key", "product_key", demand_column]]

    for lag in config.target_seasonal_lags:
        reference_date_column = f"seasonal_reference_date_{lag}"
        value_column = f"seasonal_lag_{lag}_target"
        safe_reference_column = f"_safe_seasonal_reference_date_{lag}"
        result[reference_date_column] = result["target_date"] - pd.to_timedelta(
            lag, unit="D"
        )
        result[safe_reference_column] = result[reference_date_column].where(
            result[reference_date_column].le(result["cutoff_date"])
        )
        reference = lookup.rename(
            columns={"date": safe_reference_column, demand_column: value_column}
        )
        result = result.merge(
            reference,
            on=[safe_reference_column, "store_key", "product_key"],
            how="left",
            validate="many_to_one",
        ).drop(columns=safe_reference_column)
    return result


def build_runtime_rows(
    feature_panel: pd.DataFrame,
    cutoff_date: str | pd.Timestamp,
    forecast_horizon: int,
) -> pd.DataFrame:
    """Build rows after an inclusive end-of-day historical cutoff."""

    cutoff = pd.Timestamp(cutoff_date).normalize()
    current = feature_panel.loc[feature_panel["date"].eq(cutoff), CUTOFF_FEATURE_COLUMNS]
    if current.empty:
        available_min = feature_panel["date"].min().date()
        available_max = feature_panel["date"].max().date()
        raise ValueError(
            f"No cutoff_date={cutoff.date()}; history spans "
            f"{available_min} through {available_max}."
        )

    frames: list[pd.DataFrame] = []
    for horizon in range(1, forecast_horizon + 1):
        rows = current.rename(columns={"date": "cutoff_date"}).copy()
        rows["horizon"] = horizon
        rows["target_date"] = cutoff + pd.Timedelta(days=horizon)
        frames.append(rows)
    return pd.concat(frames, ignore_index=True)
