from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


POINT_CORRECTOR_SEMANTICS_VERSION = "additive-median-residual-v1"


@dataclass(frozen=True)
class PointCorrectionValue:
    offset: float
    task_count: int
    distinct_target_dates: int
    source: str
    shrinkage_weight: float = 1.0


@dataclass
class PointCorrector:
    """A frozen additive correction fitted on predictions made out of sample.

    The same offset is applied to all three repaired quantiles.  This preserves
    ordering and makes the point correction independent from the later CQR
    interval expansion step.
    """

    method: str
    global_value: PointCorrectionValue
    by_horizon: dict[int, PointCorrectionValue]
    fit_target_start: str | None = None
    fit_target_end: str | None = None
    fit_actual_known_at_cutoff: str | None = None
    calibration_target_semantics: str = "observed_sales_non_stockout"
    model_id: str | None = None
    minimum_group_tasks: int = 0
    minimum_group_distinct_target_dates: int = 0
    shrinkage_strength: float = 0.0
    availability_validation: str = "NOT_VALIDATED_LEGACY_COMPATIBILITY"
    availability_timezone: str = "Asia/Bangkok"
    semantics_version: str = POINT_CORRECTOR_SEMANTICS_VERSION

    @classmethod
    def identity(cls, *, model_id: str | None = None) -> "PointCorrector":
        return cls(
            method="identity",
            global_value=PointCorrectionValue(
                offset=0.0,
                task_count=0,
                distinct_target_dates=0,
                source="identity",
            ),
            by_horizon={},
            model_id=model_id,
            availability_validation="NOT_APPLICABLE_IDENTITY",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "global_value": asdict(self.global_value),
            "by_horizon": {
                str(horizon): asdict(value)
                for horizon, value in sorted(self.by_horizon.items())
            },
            "fit_target_start": self.fit_target_start,
            "fit_target_end": self.fit_target_end,
            "fit_actual_known_at_cutoff": self.fit_actual_known_at_cutoff,
            "calibration_target_semantics": self.calibration_target_semantics,
            "model_id": self.model_id,
            "minimum_group_tasks": self.minimum_group_tasks,
            "minimum_group_distinct_target_dates": (
                self.minimum_group_distinct_target_dates
            ),
            "shrinkage_strength": self.shrinkage_strength,
            "availability_validation": self.availability_validation,
            "availability_timezone": self.availability_timezone,
            "semantics_version": self.semantics_version,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PointCorrector":
        global_payload = payload.get("global_value")
        if not isinstance(global_payload, dict):
            raise ValueError("point corrector global_value is invalid")
        by_horizon_payload = payload.get("by_horizon", {})
        if not isinstance(by_horizon_payload, dict):
            raise ValueError("point corrector by_horizon is invalid")
        corrector = cls(
            method=str(payload.get("method", "identity")),
            global_value=PointCorrectionValue(**global_payload),
            by_horizon={
                int(horizon): PointCorrectionValue(**value)
                for horizon, value in by_horizon_payload.items()
            },
            fit_target_start=payload.get("fit_target_start"),
            fit_target_end=payload.get("fit_target_end"),
            fit_actual_known_at_cutoff=payload.get("fit_actual_known_at_cutoff"),
            calibration_target_semantics=str(
                payload.get(
                    "calibration_target_semantics",
                    "observed_sales_non_stockout",
                )
            ),
            model_id=payload.get("model_id"),
            minimum_group_tasks=int(payload.get("minimum_group_tasks", 0)),
            minimum_group_distinct_target_dates=int(
                payload.get("minimum_group_distinct_target_dates", 0)
            ),
            shrinkage_strength=float(payload.get("shrinkage_strength", 0.0)),
            availability_validation=str(
                payload.get(
                    "availability_validation",
                    "NOT_VALIDATED_LEGACY_COMPATIBILITY",
                )
            ),
            availability_timezone=str(
                payload.get("availability_timezone", "Asia/Bangkok")
            ),
            semantics_version=str(
                payload.get("semantics_version", "point-corrector-incompatible")
            ),
        )
        if corrector.semantics_version != POINT_CORRECTOR_SEMANTICS_VERSION:
            raise ValueError(
                "Unsupported point corrector semantics: "
                f"{corrector.semantics_version}"
            )
        return corrector


def _validate_fit_frame(frame: pd.DataFrame) -> None:
    required = {"target", "p50", "horizon", "target_date"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Point-correction fit columns are missing: {missing}")
    if frame.empty:
        raise ValueError("Point-correction fit cohort is empty")
    numeric = frame[["target", "p50", "horizon"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Point-correction fit values must all be finite")
    if pd.to_datetime(frame["target_date"], errors="coerce").isna().any():
        raise ValueError("Point-correction target dates must all be valid")


def _local_timestamp(value: object, timezone_name: str) -> pd.Timestamp:
    try:
        timezone = ZoneInfo(timezone_name)
    except Exception as exc:
        raise ValueError(
            f"Point-correction availability timezone is invalid: {timezone_name}"
        ) from exc
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("Point-correction availability timestamp is invalid")
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(timezone)
    return timestamp.tz_convert(timezone)


def _validate_actual_availability(
    frame: pd.DataFrame,
    *,
    actual_known_at_cutoff: str,
    availability_timezone: str,
) -> tuple[pd.Timestamp, str]:
    """Enforce the conservative Demo EOD/next-run label convention.

    A label for target date ``d`` matures at local midnight starting ``d+1``.
    A date-only cutoff is that local midnight, so equality is allowed. Explicit
    timestamps are converted to the declared timezone before comparison.
    """

    cutoff = _local_timestamp(actual_known_at_cutoff, availability_timezone)
    target_dates = pd.to_datetime(frame["target_date"], errors="coerce").dt.normalize()
    if target_dates.isna().any():
        raise ValueError("Point-correction target dates must all be valid")
    timezone = ZoneInfo(availability_timezone)
    earliest_known = target_dates.dt.tz_localize(timezone) + pd.Timedelta(days=1)
    if "actual_known_at" in frame.columns:
        known_values: list[pd.Timestamp] = []
        for value in frame["actual_known_at"]:
            known_values.append(_local_timestamp(value, availability_timezone))
        actual_known = pd.Series(known_values, index=frame.index)
        inconsistent = actual_known.lt(earliest_known)
        if inconsistent.any():
            first = frame.index[inconsistent][0]
            raise ValueError(
                "Point-correction actual_known_at contradicts the declared "
                f"inclusive_end_of_day_next_run convention at row {first}"
            )
        validation = "VALIDATED_EXPLICIT_ACTUAL_KNOWN_AT"
    else:
        actual_known = pd.Series(earliest_known.to_numpy(), index=frame.index)
        validation = "VALIDATED_DERIVED_TARGET_DATE_NEXT_LOCAL_MIDNIGHT"
    future = actual_known.gt(cutoff)
    if future.any():
        first = frame.index[future][0]
        raise ValueError(
            "Point-correction fit contains a label unavailable at cutoff: "
            f"row={first}, actual_known_at={actual_known.loc[first].isoformat()}, "
            f"cutoff={cutoff.isoformat()}"
        )
    return cutoff, validation


def fit_point_corrector(
    predictions: pd.DataFrame,
    *,
    method: str,
    minimum_group_tasks: int = 30,
    minimum_group_distinct_target_dates: int = 7,
    shrinkage_strength: float = 20.0,
    actual_known_at_cutoff: str | None = None,
    availability_timezone: str = "Asia/Bangkok",
    calibration_target_semantics: str = "observed_sales_non_stockout",
    model_id: str | None = None,
) -> PointCorrector:
    if method == "identity":
        return PointCorrector.identity(model_id=model_id)
    if method not in {"global_median_residual", "horizon_shrunk_median_residual"}:
        raise ValueError(f"Unsupported point-correction method: {method}")
    if minimum_group_tasks < 1 or minimum_group_distinct_target_dates < 1:
        raise ValueError("Point-correction group support thresholds must be positive")
    if shrinkage_strength < 0:
        raise ValueError("shrinkage_strength must be nonnegative")

    frame = predictions.copy()
    _validate_fit_frame(frame)
    validated_cutoff: pd.Timestamp | None = None
    availability_validation = "NOT_VALIDATED_LEGACY_COMPATIBILITY"
    if actual_known_at_cutoff is not None:
        validated_cutoff, availability_validation = _validate_actual_availability(
            frame,
            actual_known_at_cutoff=actual_known_at_cutoff,
            availability_timezone=availability_timezone,
        )
    dates = pd.to_datetime(frame["target_date"]).dt.normalize()
    residual = frame["target"].to_numpy(dtype=float) - frame["p50"].to_numpy(
        dtype=float
    )
    global_offset = float(np.median(residual))
    global_value = PointCorrectionValue(
        offset=global_offset,
        task_count=len(frame),
        distinct_target_dates=int(dates.nunique()),
        source="global_median_residual",
    )
    by_horizon: dict[int, PointCorrectionValue] = {}
    if method == "horizon_shrunk_median_residual":
        for horizon, group in frame.assign(_target_date=dates).groupby(
            "horizon", observed=True
        ):
            task_count = len(group)
            distinct_dates = int(group["_target_date"].nunique())
            if (
                task_count < minimum_group_tasks
                or distinct_dates < minimum_group_distinct_target_dates
            ):
                continue
            group_offset = float(np.median(group["target"] - group["p50"]))
            weight = task_count / (task_count + shrinkage_strength)
            offset = weight * group_offset + (1.0 - weight) * global_offset
            by_horizon[int(horizon)] = PointCorrectionValue(
                offset=float(offset),
                task_count=task_count,
                distinct_target_dates=distinct_dates,
                source=f"horizon_{int(horizon)}_shrunk_to_global",
                shrinkage_weight=float(weight),
            )

    return PointCorrector(
        method=method,
        global_value=global_value,
        by_horizon=by_horizon,
        fit_target_start=dates.min().date().isoformat(),
        fit_target_end=dates.max().date().isoformat(),
        fit_actual_known_at_cutoff=(
            None if validated_cutoff is None else validated_cutoff.isoformat()
        ),
        calibration_target_semantics=calibration_target_semantics,
        model_id=model_id,
        minimum_group_tasks=minimum_group_tasks,
        minimum_group_distinct_target_dates=minimum_group_distinct_target_dates,
        shrinkage_strength=shrinkage_strength,
        availability_validation=availability_validation,
        availability_timezone=availability_timezone,
    )


def apply_point_corrector(
    frame: pd.DataFrame,
    corrector: PointCorrector,
) -> pd.DataFrame:
    required = {"p25", "p50", "p75", "horizon"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Point-correction input columns are missing: {missing}")
    result = frame.copy()
    values = [
        corrector.by_horizon.get(int(horizon), corrector.global_value)
        for horizon in result["horizon"]
    ]
    result["p25_repaired"] = result["p25"].astype(float)
    result["p50_repaired"] = result["p50"].astype(float)
    result["p75_repaired"] = result["p75"].astype(float)
    result["point_correction"] = [value.offset for value in values]
    result["point_correction_source"] = [value.source for value in values]
    for quantile in ("p25", "p50", "p75"):
        corrected = np.maximum(
            0.0,
            result[f"{quantile}_repaired"].to_numpy(dtype=float)
            + result["point_correction"].to_numpy(dtype=float),
        )
        result[f"{quantile}_corrected"] = corrected
        result[quantile] = corrected
    return result
