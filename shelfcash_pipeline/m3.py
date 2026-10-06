"""M3: reuse completed M2 forecast -> validated recipes/BOM -> ingredient demand."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from shelfcash_forecast.bom.adapter import validate_sales_product_unit_consistency
from shelfcash_forecast.bom.contracts import IngredientDemandPackage
from shelfcash_forecast.bom.engine import propagate_ingredient_demand
from shelfcash_forecast.config import ForecastConfig
from shelfcash_forecast.contracts import ForecastPackage
from shelfcash_forecast.data.adapter import adapt_forecast_input
from shelfcash_forecast.exceptions import RecipeValidationError
from shelfcash_preprocess.engine import create_forecast_input_frames, load_canonical_frames
from shelfcash_pipeline.context import write_json


def run(output: Path, bundle: Path, forecast: ForecastPackage, config: ForecastConfig) -> IngredientDemandPackage:
    frames = load_canonical_frames(bundle, required_capability="ingredient_demand")
    adapted = adapt_forecast_input(create_forecast_input_frames(bundle), config)
    validate_sales_product_unit_consistency(adapted.sales_history, cutoff_date=forecast.forecast_date)
    ingredient = propagate_ingredient_demand(
        forecast, frames["recipes"], frames.get("unit_conversions"),
    )
    destination = output / "m3"
    destination.mkdir(exist_ok=False)
    write_json(destination / "ingredient_demand.json", ingredient.model_dump(mode="json"))
    pd.DataFrame([row.model_dump(mode="json", exclude={"sources", "warnings"})
                  for row in ingredient.predictions]).to_csv(
        destination / "ingredient_rows.csv", index=False, encoding="utf-8-sig",
    )
    write_json(destination / "summary.json", {
        "rows": len(ingredient.predictions),
        "ingredients": len({row.ingredient_id for row in ingredient.predictions}),
        "is_complete": ingredient.is_complete,
        "issues": [issue.model_dump(mode="json") for issue in ingredient.issues],
        "warnings": ingredient.warnings,
        "forecast_recomputed": False,
    })
    if not ingredient.is_complete:
        raise RecipeValidationError("M3_BOM_INCOMPLETE: inspect m3/ingredient_demand.json")
    return ingredient
