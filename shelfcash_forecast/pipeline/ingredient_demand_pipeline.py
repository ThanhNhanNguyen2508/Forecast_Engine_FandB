# nối forecast pipeline với BOM engine, để tính ra demand nguyên liệu cho từng store, từng target date.
# Canonical Data
#      │
#      ├── sales_history
#      ├── products / stores / ...
#      ├── recipes
#      └── unit_conversions
#      │
#      ▼
# ingredient_demand_pipeline.py
#      │
#      ├── load_artifacts()
#      │
#      ├── adapt_forecast_input()
#      │
#      ├── validate sales product unit
#      │
#      ├── predict_demand()
#      │
#      ▼
# ForecastPackage
#      │
#      ├── P25
#      ├── P50
#      └── P75
#      │
#      ▼
# propagate_ingredient_demand()
#      │
#      ▼
# IngredientDemandPackage
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from shelfcash_forecast.bom.adapter import validate_sales_product_unit_consistency
from shelfcash_forecast.bom.contracts import IngredientDemandPackage
from shelfcash_forecast.bom.engine import propagate_ingredient_demand
from shelfcash_forecast.data.adapter import adapt_forecast_input
from shelfcash_forecast.exceptions import RecipeValidationError
from shelfcash_forecast.pipeline.inference_pipeline import predict_demand
from shelfcash_forecast.registry.loader import load_artifacts


def predict_ingredient_demand(
    canonical_data: Mapping[str, pd.DataFrame],
    artifact_directory: str | Path,
    cutoff_date: str | pd.Timestamp,
    forecast_horizon: int = 7,
    *,
    execution_mode: str = "production",
) -> IngredientDemandPackage:
    """Run existing forecast inference, then deterministic Recipe/BOM propagation."""

    if "recipes" not in canonical_data:
        raise RecipeValidationError(
            "canonical_data bắt buộc có recipes cho ingredient demand.",
            details={"missing_key": "recipes"},
        )

    artifacts = load_artifacts(artifact_directory)
    adapted = adapt_forecast_input(canonical_data, artifacts.config)
    validate_sales_product_unit_consistency(
        adapted.sales_history,
        cutoff_date=cutoff_date,
    )

    forecast = predict_demand(
        canonical_data=canonical_data,
        artifact_directory=artifact_directory,
        cutoff_date=cutoff_date,
        forecast_horizon=forecast_horizon,
        execution_mode=execution_mode,
    )
    return propagate_ingredient_demand(
        forecast=forecast,
        recipes=canonical_data["recipes"],
        unit_conversions=canonical_data.get("unit_conversions"),
    )
