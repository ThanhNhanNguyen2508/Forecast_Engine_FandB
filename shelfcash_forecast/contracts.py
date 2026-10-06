from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ForecastPrediction(StrictContract):
    store_id: str
    product_id: str
    product_name: str
    unit: str | None = None
    target_date: date
    horizon: int = Field(ge=1) # ge : greater equal
    p25: float = Field(ge=0)
    p50: float = Field(ge=0)
    p75: float = Field(ge=0)
    interval_lower: float = Field(ge=0)
    interval_upper: float = Field(ge=0)
    baseline_p50: float = Field(ge=0)
    calibration_source: str
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_ordering(self) -> ForecastPrediction:
        if not self.p25 <= self.p50 <= self.p75:
            raise ValueError("Quantiles phải thỏa P25 <= P50 <= P75.")
        if self.interval_lower > self.interval_upper:
            raise ValueError("interval_lower không được lớn hơn interval_upper.")
        return self


class ForecastPackage(StrictContract):
    forecast_date: date
    forecast_horizon: int = Field(ge=1)
    model_version: str
    predictions: list[ForecastPrediction]
    warnings: list[str] = Field(default_factory=list)


class TrainingResult(StrictContract):
    status: str
    model_version: str
    artifact_directory: str
    data_quality: dict[str, Any]
    baseline_metrics: dict[str, Any]
    walk_forward_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    calibration_metrics: dict[str, Any]
    warnings: list[str] = Field(default_factory=list)
