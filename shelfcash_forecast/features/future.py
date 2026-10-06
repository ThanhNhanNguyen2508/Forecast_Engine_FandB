from __future__ import annotations

import pandas as pd


def add_deterministic_future_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    target = pd.to_datetime(result["target_date"])

    result["target_day_of_week"] = target.dt.dayofweek.astype("int8")
    result["target_is_weekend"] = result["target_day_of_week"].ge(5).astype("int8")
    result["target_month"] = target.dt.month.astype("int8")
    result["target_day_of_month"] = target.dt.day.astype("int8")
    result["target_week_of_month"] = (
        ((result["target_day_of_month"] - 1) // 7) + 1
    ).astype("int8")
    result["target_week_of_year"] = target.dt.isocalendar().week.astype("int16")
    return result


def add_calendar_future_features(
    frame: pd.DataFrame,
    calendar: pd.DataFrame | None,
) -> pd.DataFrame:
    result = frame.copy()

    if calendar is None or calendar.empty:
        result["target_is_holiday"] = 0
        result["target_store_closed"] = 0
        # Keep optional numeric columns numeric.  ``pd.NA`` alone creates an
        # object column which LightGBM correctly refuses to consume.
        result["target_temperature"] = float("nan")
        result["target_rainfall"] = float("nan")
        result["calendar_available"] = 0
        return result

    calendar_future = calendar.rename(
        columns={
            "date": "target_date",
            "is_holiday": "target_is_holiday",
            "is_store_closed": "target_store_closed",
            "temperature": "target_temperature",
            "rainfall": "target_rainfall",
            "weather_kind": "target_weather_kind",
            "weather_issued_at": "target_weather_issued_at",
            "weather_available_at": "target_weather_available_at",
        }
    )
    selected = [
        "target_date",
        "target_is_holiday",
        "target_store_closed",
        "target_temperature",
        "target_rainfall",
        "target_weather_kind",
        "target_weather_issued_at",
        "target_weather_available_at",
    ]
    result = result.merge(
        calendar_future[selected],
        on="target_date",
        how="left",
        validate="many_to_one",
    )

    # Weather is only forecast-safe when it is a forecast vintage issued and
    # available at the row's forecast origin.  Realized target weather and
    # rows without vintage provenance are intentionally excluded.
    if "cutoff_date" not in result.columns:
        raise ValueError("Calendar feature join requires cutoff_date as forecast origin.")
    origin = pd.to_datetime(result["cutoff_date"], errors="coerce")
    issued = pd.to_datetime(result["target_weather_issued_at"], errors="coerce")
    available = pd.to_datetime(result["target_weather_available_at"], errors="coerce")
    forecast_weather_safe = (
        result["target_weather_kind"].astype("string").str.lower().eq("forecast")
        & issued.notna()
        & available.notna()
        & issued.le(origin)
        & available.le(origin)
    )
    result.loc[~forecast_weather_safe, ["target_temperature", "target_rainfall"]] = float("nan")

    result["calendar_available"] = (
        result["target_is_holiday"].notna()
        | result["target_store_closed"].notna()
        | result["target_temperature"].notna()
        | result["target_rainfall"].notna()
    ).astype("int8")
    result["target_is_holiday"] = (
        result["target_is_holiday"].fillna(False).astype("int8")
    )
    result["target_store_closed"] = (
        result["target_store_closed"].fillna(False).astype("int8")
    )
    result["target_temperature"] = pd.to_numeric(
        result["target_temperature"], errors="coerce"
    )
    result["target_rainfall"] = pd.to_numeric(
        result["target_rainfall"], errors="coerce"
    )
    result = result.drop(
        columns=[
            "target_weather_kind",
            "target_weather_issued_at",
            "target_weather_available_at",
        ]
    )
    return result
