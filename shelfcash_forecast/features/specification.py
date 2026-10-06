from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from pandas.api.types import is_numeric_dtype

from shelfcash_forecast.exceptions import FeatureSchemaError

CATEGORICAL_SOURCE_COLUMNS = ["store_key", "product_key"]
CATEGORICAL_MODEL_COLUMNS = ["store_code", "product_code"]

NUMERIC_FEATURES = [
    "horizon",
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
    "seasonal_lag_7_target",
    "seasonal_lag_14_target",
    "seasonal_lag_28_target",
    "target_day_of_week",
    "target_is_weekend",
    "target_month",
    "target_day_of_month",
    "target_week_of_month",
    "target_week_of_year",
    "target_is_holiday",
    "target_store_closed",
    "target_temperature",
    "target_rainfall",
    "calendar_available",
]

MODEL_FEATURES = CATEGORICAL_MODEL_COLUMNS + NUMERIC_FEATURES


def validate_runtime_feature_schema(
    frame: pd.DataFrame,
    *,
    expected_features: list[str],
    expected_categorical_features: list[str],
) -> None:
    duplicate_columns = frame.columns[frame.columns.duplicated()].tolist()
    missing = [column for column in expected_features if column not in frame]
    non_numeric = [
        column
        for column in expected_features
        if column in frame and not is_numeric_dtype(frame[column])
    ]
    categorical_mismatch = expected_categorical_features != list(
        CATEGORICAL_MODEL_COLUMNS
    )
    if missing or duplicate_columns or non_numeric or categorical_mismatch:
        raise FeatureSchemaError(
            "Runtime features are incompatible with the trained model schema.",
            details={
                "missing_features": missing,
                "duplicate_columns": duplicate_columns,
                "non_numeric_features": non_numeric,
                "expected_categorical_features": expected_categorical_features,
                "runtime_categorical_features": list(CATEGORICAL_MODEL_COLUMNS),
            },
        )


@dataclass
class CategoryEncoder:
    mappings: dict[str, dict[str, int]]
# {
#     "store_key": {
#         "STORE_01": 0,
#         "STORE_02": 1,
#     },
#     "product_key": {
#         "P001": 0,
#         "P002": 1,
#         "P003": 2,
#     },
# }


    @classmethod
    def fit(cls, frame: pd.DataFrame) -> CategoryEncoder:
        mappings: dict[str, dict[str, int]] = {}
        for source in CATEGORICAL_SOURCE_COLUMNS:
            values = (
                frame[source]
                .astype("string")
                .fillna("__MISSING__")
                .drop_duplicates()
                .tolist()
            )
            mappings[source] = {
                value: index for index, value in enumerate(sorted(values))
            }
        return cls(mappings=mappings)


    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        for source, output in zip(
            CATEGORICAL_SOURCE_COLUMNS,
            CATEGORICAL_MODEL_COLUMNS,
            strict=True,
        ):
            mapping = self.mappings[source]
            result[output] = (
                result[source]
                .astype("string")
                .fillna("__MISSING__")
                .map(mapping)
                .fillna(-1)
                .astype("int32")
            )
        return result
# hàm thêm store_code và product_code ví dụ :
# STORE_01 → 0
# P001     → 0
# P002     → 1
# Category không xuất hiện khi fit:
# unknown category → -1


    def to_dict(self) -> dict[str, dict[str, int]]:
        return self.mappings

    @classmethod
    def from_dict(cls, payload: dict[str, dict[str, int]]) -> CategoryEncoder:
        return cls(
            mappings={
                column: {str(key): int(value) for key, value in mapping.items()}
                for column, mapping in payload.items()
            }
        )
