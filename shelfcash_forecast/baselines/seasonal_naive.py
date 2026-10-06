from __future__ import annotations

import pandas as pd


def seasonal_naive_predict_rows(frame: pd.DataFrame) -> pd.DataFrame:
    prediction = frame["seasonal_lag_7_target"].astype(float)
    warning = pd.Series("", index=frame.index, dtype="string")
    fallback_used = prediction.isna()
    fallbacks = [
        ("rolling_median_7", "FALLBACK_ROLLING_MEDIAN_7"),
        ("rolling_median_28", "FALLBACK_ROLLING_MEDIAN_28"),
        ("last_observed_demand", "FALLBACK_LAST_OBSERVED_DEMAND"),
    ]
    for column, label in fallbacks:
        missing = prediction.isna()
        prediction = prediction.fillna(frame[column].astype(float))
        warning.loc[missing & prediction.notna()] = label
    warning.loc[prediction.isna()] = "FALLBACK_ZERO_NO_HISTORY"
    prediction = prediction.fillna(0.0).clip(lower=0.0)
    history_count = frame["history_observation_count"].fillna(0).astype(int)
    return pd.DataFrame(
        {
            "baseline_name": "SEASONAL_NAIVE_7",
            "prediction": prediction,
            "fallback_used": fallback_used,
            "history_count": history_count,
            "warnings": warning,
        },
        index=frame.index,
    )


def seasonal_naive_predict(frame: pd.DataFrame) -> pd.Series:
    return seasonal_naive_predict_rows(frame)["prediction"].astype(float)
