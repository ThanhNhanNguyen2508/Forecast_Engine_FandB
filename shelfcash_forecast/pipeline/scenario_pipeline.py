from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from shelfcash_forecast.exceptions import (
    RecipeValidationError,
    ScenarioDataInsufficiencyError,
    ScenarioValidationError,
)
from shelfcash_forecast.pipeline.inference_pipeline import predict_demand
from shelfcash_forecast.scenario.bom import propagate_ingredient_demand_scenarios
from shelfcash_forecast.scenario.composer import generate_product_demand_scenarios
from shelfcash_forecast.scenario.contracts import IngredientDemandScenarioBundle
from shelfcash_forecast.scenario.residuals import load_residual_history
from shelfcash_forecast.scenario.yield_loss import (
    FixedRecipeYieldLossModel,
    YieldLossModel,
    fit_usage_residual_yield_loss_model,
)


def _resolve_yield_loss_model(
    canonical_data: Mapping[str, pd.DataFrame],
    *,
    cutoff_date: str | pd.Timestamp,
    minimum_samples: int,
    strict: bool,
) -> tuple[YieldLossModel, str | None]:
    if "ingredient_usage_history" not in canonical_data:
        return FixedRecipeYieldLossModel(), None
    try:
        return (
            fit_usage_residual_yield_loss_model(
                canonical_data["ingredient_usage_history"],
                canonical_data["sales_history"],
                canonical_data["recipes"],
                canonical_data.get("unit_conversions"),
                minimum_samples=minimum_samples,
                cutoff_date=cutoff_date,
            ),
            None,
        )
    except (ScenarioDataInsufficiencyError, ScenarioValidationError) as exc:
        if strict:
            raise
        warning = (
            "YIELD_LOSS_RECIPE_COVERAGE_INCOMPLETE"
            if isinstance(exc, ScenarioValidationError)
            else "YIELD_LOSS_HISTORY_INSUFFICIENT"
        )
        return (
            FixedRecipeYieldLossModel(source="recipe_fixed_fallback"),
            warning,
        )


def predict_ingredient_demand_scenarios(
    canonical_data: Mapping[str, pd.DataFrame],
    artifact_directory: str | Path,
    cutoff_date: str | pd.Timestamp,
    forecast_horizon: int = 7,
    *,
    n_scenarios: int = 100,
    seed: int = 42,
    scenario_method: str = "residual_bootstrap",
    yield_loss_minimum_samples: int = 3,
    yield_loss_strict: bool = False,
    execution_mode: str = "production",
    override_policy=None,
) -> IngredientDemandScenarioBundle:
    """Forecast, generate joint product scenarios, then apply BOM per scenario."""

    if "recipes" not in canonical_data:
        raise RecipeValidationError(
            "canonical_data bắt buộc có recipes cho ingredient scenarios.",
            details={"missing_key": "recipes"},
        )
    forecast = predict_demand(
        canonical_data=canonical_data,
        artifact_directory=artifact_directory,
        cutoff_date=cutoff_date,
        forecast_horizon=forecast_horizon,
        execution_mode=execution_mode,
        override_policy=override_policy,
    )
    residual_history = load_residual_history(artifact_directory)
    product_scenarios = generate_product_demand_scenarios(
        forecast,
        residual_history,
        n_scenarios=n_scenarios,
        seed=seed,
        method=scenario_method,
    )
    yield_loss_model, yield_loss_warning = _resolve_yield_loss_model(
        canonical_data,
        cutoff_date=cutoff_date,
        minimum_samples=yield_loss_minimum_samples,
        strict=yield_loss_strict,
    )
    bundle = propagate_ingredient_demand_scenarios(
        product_scenarios,
        canonical_data["recipes"],
        canonical_data.get("unit_conversions"),
        yield_loss_model=yield_loss_model,
    )
    if yield_loss_warning is None:
        return bundle
    return bundle.model_copy(
        update={"warnings": sorted({*bundle.warnings, yield_loss_warning})}
    )
