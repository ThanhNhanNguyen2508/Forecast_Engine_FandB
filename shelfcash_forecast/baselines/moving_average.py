from __future__ import annotations

import pandas as pd


def moving_average_predict_rows(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    """Return an auditable moving-average baseline at the cutoff date."""

    if window not in {7, 28}:
        raise ValueError("Only MA7 and MA28 are supported baselines.")
    mean_column = f"rolling_mean_{window}"
    count_column = f"rolling_count_{window}"
    prediction = (
        frame[mean_column].astype(float)
        if mean_column in frame
        else pd.Series(float("nan"), index=frame.index)
    )
    fallback_used = prediction.isna()
    prediction = prediction.fillna(frame["last_observed_demand"].astype(float))
    zero_fallback = prediction.isna()
    prediction = prediction.fillna(0.0).clip(lower=0.0)
    history_count = (
        frame[count_column].fillna(0).astype(int)
        if count_column in frame
        else frame["history_observation_count"].fillna(0).clip(upper=window).astype(int)
    )
    warnings = pd.Series("", index=frame.index, dtype="string")
    warnings.loc[fallback_used & ~zero_fallback] = "FALLBACK_LAST_OBSERVED_DEMAND"
    warnings.loc[zero_fallback] = "FALLBACK_ZERO_NO_HISTORY"
    return pd.DataFrame(
        {
            "baseline_name": f"MA{window}",
            "prediction": prediction.astype(float),
            "fallback_used": fallback_used,
            "history_count": history_count,
            "warnings": warnings,
        },
        index=frame.index,
    )


def moving_average_predict(frame: pd.DataFrame, window: int) -> pd.Series:
    return moving_average_predict_rows(frame, window)["prediction"]
