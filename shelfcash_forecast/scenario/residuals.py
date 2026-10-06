#“Trong quá khứ, model đã forecast lệch khỏi thực tế như thế nào khi dự báo thật sự out-of-sample?”
# from __future__ import annotations
# Historical data
#       ↓
# Walk-forward forecasting
#       ↓
# Out-of-sample predictions
#       │
#       │ actual
#       │ p25
#       │ p50
#       │ p75
#       ▼
# scenario/residuals.py
#       │
#       ├── build residual history
#       ├── validate residual history
#       └── load residual artifact
#       │
#       ▼
# Walk-forward Residual History
#       │
#       ▼
# Scenario Generator
#       │
#       ▼
# ProductDemandScenarioBundle
#       │
#       ▼
# BOM Scenario Propagation
#       │
#       ▼
# IngredientDemandScenarioBundle

from pathlib import Path

import numpy as np
import pandas as pd

from shelfcash_forecast.exceptions import (
    ScenarioDataInsufficiencyError,
    ScenarioValidationError,
)

RESIDUAL_COLUMNS = [
    "forecast_origin",
    "target_date",
    "actual_known_at",
    "horizon",
    "store_id",
    "product_id",
    "actual",
    "p25",
    "p50",
    "p75",
    "raw_residual",
    "scaled_residual",
    "target_train_eligible",
    "residual_source",
]


def build_walk_forward_residual_history(
    predictions: pd.DataFrame,
    *,
    epsilon: float = 1e-6,
) -> pd.DataFrame:
    """Build auditable residuals only from out-of-sample walk-forward predictions."""

    if predictions.empty:
        return pd.DataFrame(columns=RESIDUAL_COLUMNS)
    required = {
        "cutoff_date",
        "target_date",
        "horizon",
        "store_key",
        "product_key",
        "target",
        "p25",
        "p50",
        "p75",
        "target_train_eligible",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ScenarioValidationError(
            "Walk-forward predictions thiếu cột để tạo residual artifact.",
            details={"missing_columns": sorted(missing)},
        )

    eligible = (
        predictions["target_train_eligible"].astype("boolean").fillna(False)
    )
    frame = predictions.loc[eligible].copy()
    spread = (frame["p75"] - frame["p25"]).astype(float).clip(lower=epsilon)
    result = pd.DataFrame(
        {
            "forecast_origin": pd.to_datetime(frame["cutoff_date"]).dt.normalize(),
            "target_date": pd.to_datetime(frame["target_date"]).dt.normalize(),
            "actual_known_at": pd.to_datetime(
                frame.get("actual_known_at", frame["target_date"])
            ).dt.normalize(),
            "horizon": frame["horizon"].astype(int),
            "store_id": frame["store_key"].astype(str),
            "product_id": frame["product_key"].astype(str),
            "actual": frame["target"].astype(float),
            "p25": frame["p25"].astype(float),
            "p50": frame["p50"].astype(float),
            "p75": frame["p75"].astype(float),
            "target_train_eligible": True,
            "residual_source": "walk_forward_oos",
        }
    )
    result["raw_residual"] = result["actual"] - result["p50"] # thực tế trừ đoán trung vị
    result["scaled_residual"] = result["raw_residual"] / spread.to_numpy() # scale cho residual
    result = result[RESIDUAL_COLUMNS]
    if not np.isfinite(result[["raw_residual", "scaled_residual"]]).all().all():
        raise ScenarioValidationError("Residual artifact chứa giá trị không finite.")
    duplicate_key = [
        "forecast_origin",
        "target_date",
        "horizon",
        "store_id",
        "product_id",
    ]
    if result.duplicated(duplicate_key).any():
        raise ScenarioValidationError(
            "Residual artifact chứa duplicate key.",
            code="SCENARIO_DUPLICATE_KEY",
        )
        #HERE
    return result.reset_index(drop=True)


def validate_residual_history(
    frame: pd.DataFrame,
    *,
    as_of: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    missing = set(RESIDUAL_COLUMNS) - set(frame.columns)
    if missing:
        raise ScenarioValidationError(
            "Residual history thiếu cột bắt buộc.",
            details={"missing_columns": sorted(missing)},
        )
    clean = frame.copy()
    clean["forecast_origin"] = pd.to_datetime(
        clean["forecast_origin"], errors="coerce"
    ).dt.normalize()
    clean["target_date"] = pd.to_datetime(
        clean["target_date"], errors="coerce"
    ).dt.normalize()
    clean["actual_known_at"] = pd.to_datetime(
        clean["actual_known_at"], errors="coerce"
    ).dt.normalize()
    for column in (
        "horizon",
        "actual",
        "p25",
        "p50",
        "p75",
        "raw_residual",
        "scaled_residual",
    ):
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    eligible = clean["target_train_eligible"].astype("boolean").fillna(False)
    finite = np.isfinite(
        clean[["actual", "p25", "p50", "p75", "raw_residual", "scaled_residual"]]
    ).all(axis=1)
    chronological = (
        clean["target_date"] - clean["forecast_origin"]
    ).dt.days.eq(clean["horizon"])
    quantiles_ordered = clean["p25"].le(clean["p50"]) & clean["p50"].le(
        clean["p75"]
    )
    valid_rows = (
        eligible
        & clean["forecast_origin"].notna()
        & clean["target_date"].notna()
        & clean["actual_known_at"].notna()
        & clean["actual_known_at"].ge(clean["target_date"])
        & clean["horizon"].ge(1)
        & finite
        & chronological
        & quantiles_ordered
        & clean["actual"].ge(0)
        & clean["residual_source"].eq("walk_forward_oos")
    )
    if not valid_rows.all():
        raise ScenarioValidationError(
            "Residual history contains invalid or non-OOS rows.",
            details={"invalid_row_count": int((~valid_rows).sum())},
        )
    clean = clean.loc[valid_rows].copy()
    if as_of is not None:
        origin = pd.Timestamp(as_of).normalize()
        clean = clean.loc[clean["actual_known_at"].le(origin)].copy()
    if clean.empty:
        raise ScenarioDataInsufficiencyError(
            "No eligible walk-forward residuals were known at the requested as-of time.",
            details={
                "as_of": None
                if as_of is None
                else pd.Timestamp(as_of).date().isoformat()
            },
        )
    clean["store_id"] = clean["store_id"].astype(str)
    clean["product_id"] = clean["product_id"].astype(str)
    clean["horizon"] = clean["horizon"].astype(int)
    duplicate_key = [
        "forecast_origin",
        "target_date",
        "horizon",
        "store_id",
        "product_id",
    ]
    if clean.duplicated(duplicate_key).any():
        raise ScenarioValidationError(
            "Residual history contains duplicate out-of-sample keys.",
            code="SCENARIO_DUPLICATE_KEY",
        )
    return clean.reset_index(drop=True)


def load_residual_history(artifact_directory: str | Path) -> pd.DataFrame:
    artifact_dir = Path(artifact_directory)
    parquet_path = artifact_dir / "walk_forward_residuals.parquet"
    csv_path = artifact_dir / "walk_forward_residuals.csv"
    if parquet_path.exists():
        return validate_residual_history(pd.read_parquet(parquet_path))
    if csv_path.exists():
        return validate_residual_history(pd.read_csv(csv_path))
    raise ScenarioDataInsufficiencyError(
        "Artifact không có walk-forward residual history.",
        details={"artifact_directory": str(artifact_dir)},
    )
