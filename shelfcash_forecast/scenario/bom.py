from __future__ import annotations

from shelfcash_forecast.bom.engine import propagate_ingredient_demand
from shelfcash_forecast.contracts import ForecastPackage, ForecastPrediction
from shelfcash_forecast.scenario.contracts import (
    IngredientDemandScenario,
    IngredientDemandScenarioBundle,
    IngredientDemandScenarioLine,
    IngredientScenarioContribution,
    ProductDemandScenarioBundle,
)
from shelfcash_forecast.scenario.yield_loss import (
    FixedRecipeYieldLossModel,
    YieldLossModel,
)


def propagate_ingredient_demand_scenarios(
    product_scenarios: ProductDemandScenarioBundle,
    recipes,
    unit_conversions=None,
    yield_loss_model: YieldLossModel | None = None,
) -> IngredientDemandScenarioBundle:
    """Apply the existing deterministic Recipe/BOM engine independently per scenario."""

    model = yield_loss_model or FixedRecipeYieldLossModel()
    ingredient_scenarios: list[IngredientDemandScenario] = []
    global_warnings = set(product_scenarios.warnings)

    for scenario in product_scenarios.scenarios:
        deterministic_forecast = ForecastPackage(
            forecast_date=product_scenarios.forecast_date,
            forecast_horizon=product_scenarios.horizon,
            model_version=product_scenarios.model_version,
            predictions=[
                ForecastPrediction(
                    store_id=line.store_id,
                    product_id=line.product_id,
                    product_name=line.product_name,
                    unit=line.product_unit,
                    target_date=line.target_date,
                    horizon=line.horizon,
                    p25=line.demand_quantity,
                    p50=line.demand_quantity,
                    p75=line.demand_quantity,
                    interval_lower=line.demand_quantity,
                    interval_upper=line.demand_quantity,
                    baseline_p50=line.demand_quantity,
                    calibration_source=f"scenario:{scenario.scenario_id}",
                )
                for line in scenario.lines
            ],
            warnings=product_scenarios.warnings,
        )
        deterministic_ingredients = propagate_ingredient_demand(
            deterministic_forecast,
            recipes,
            unit_conversions,
        )
        lines: list[IngredientDemandScenarioLine] = []
        for prediction in deterministic_ingredients.predictions:
            multiplier = model.multiplier(
                store_id=prediction.store_id,
                ingredient_id=prediction.ingredient_id,
                target_date=prediction.target_date,
                theoretical_quantity=prediction.p50,
            )
            lines.append(
                IngredientDemandScenarioLine(
                    scenario_id=scenario.scenario_id,
                    store_id=prediction.store_id,
                    ingredient_id=prediction.ingredient_id,
                    ingredient_name=prediction.ingredient_name,
                    target_date=prediction.target_date,
                    quantity=prediction.p50 * multiplier,
                    unit=prediction.unit,
                    yield_loss_source=model.source,
                    yield_loss_multiplier=multiplier,
                    contributions=[
                        IngredientScenarioContribution(
                            product_id=source.product_id,
                            product_name=source.product_name,
                            recipe_id=source.recipe_id,
                            recipe_version=source.recipe_version,
                            fixed_recipe_quantity=source.contribution_p50,
                            final_quantity=source.contribution_p50 * multiplier,
                            unit=source.contribution_unit,
                        )
                        for source in prediction.sources
                    ],
                    warnings=prediction.warnings,
                )
            )
        scenario_warnings = sorted(
            set(deterministic_ingredients.warnings) | set(product_scenarios.warnings)
        )
        global_warnings.update(scenario_warnings)
        ingredient_scenarios.append(
            IngredientDemandScenario(
                scenario_id=scenario.scenario_id,
                probability_weight=scenario.probability_weight,
                lines=lines,
                issues=deterministic_ingredients.issues,
                warnings=scenario_warnings,
                is_complete=deterministic_ingredients.is_complete,
            )
        )

    return IngredientDemandScenarioBundle(
        forecast_date=product_scenarios.forecast_date,
        horizon=product_scenarios.horizon,
        forecast_model_version=product_scenarios.model_version,
        scenario_method=product_scenarios.scenario_method,
        scenarios=ingredient_scenarios,
        diagnostics={
            **product_scenarios.diagnostics,
            "yield_loss_source": model.source,
        },
        warnings=sorted(global_warnings),
        is_complete=all(scenario.is_complete for scenario in ingredient_scenarios),
    )
