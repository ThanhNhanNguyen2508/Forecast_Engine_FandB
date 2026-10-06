from __future__ import annotations

from datetime import date
from math import isfinite

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from shelfcash_forecast.exceptions import DataValidationError


class _CanonicalRow(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=False)


class CanonicalSalesRow(_CanonicalRow):
    date: date
    store_id: str = Field(min_length=1)
    product_id: str = Field(min_length=1)
    product_name: str = Field(min_length=1)
    quantity_sold: float = Field(ge=0)
    is_stockout: bool | None = None
    product_unit: str | None = None
    selling_price: float | None = Field(default=None, ge=0)
    revenue: float | None = Field(default=None, ge=0)
    promotion_name: str | None = None

    @field_validator("quantity_sold", "selling_price", "revenue")
    @classmethod
    def finite_numbers(cls, value: float | None) -> float | None:
        if value is not None and not isfinite(value):
            raise ValueError("canonical numeric values must be finite")
        return value


class CanonicalCalendarRow(_CanonicalRow):
    date: date
    store_id: str | None = None
    weekday: int = Field(ge=0, le=6)
    is_weekend: bool | None = None
    is_holiday: bool | None = None
    planned_closure: bool | None = None
    is_promotion: bool | None = None
    future_temperature: float | None = None
    future_rainfall: float | None = Field(default=None, ge=0)

    @field_validator("future_temperature", "future_rainfall")
    @classmethod
    def finite_optional_numbers(cls, value: float | None) -> float | None:
        if value is not None and not isfinite(value):
            raise ValueError("canonical numeric values must be finite")
        return value


def _optional(value: object) -> object | None:
    return None if pd.isna(value) else value


def validate_canonical_sales_frame(frame: pd.DataFrame) -> None:
    for index, row in frame.iterrows():
        try:
            CanonicalSalesRow(
                date=pd.Timestamp(row["date"]).date(),
                store_id=str(row["store_key"]),
                product_id=str(row["product_key"]),
                product_name=str(row["product_name"]),
                quantity_sold=float(row["quantity_sold"]),
                is_stockout=_optional(row["is_stockout"]),
                product_unit=_optional(row["unit"]),
                selling_price=_optional(row["selling_price"]),
                revenue=_optional(row["revenue"]),
                promotion_name=_optional(row["promotion_name"]),
            )
        except (TypeError, ValueError) as exc:
            raise DataValidationError(
                f"Invalid canonical sales row at index {index}: {exc}"
            ) from exc


def validate_canonical_calendar_frame(frame: pd.DataFrame) -> None:
    for index, row in frame.iterrows():
        try:
            timestamp = pd.Timestamp(row["date"])
            CanonicalCalendarRow(
                date=timestamp.date(),
                store_id=None,
                weekday=timestamp.weekday(),
                is_weekend=_optional(row["is_weekend"]),
                is_holiday=_optional(row["is_holiday"]),
                planned_closure=_optional(row["is_store_closed"]),
                is_promotion=_optional(row["is_promotion"]),
                future_temperature=_optional(row["temperature"]),
                future_rainfall=_optional(row["rainfall"]),
            )
        except (TypeError, ValueError) as exc:
            raise DataValidationError(
                f"Invalid global canonical calendar row at index {index}: {exc}"
            ) from exc
