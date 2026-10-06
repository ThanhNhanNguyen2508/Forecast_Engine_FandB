"""M6: read existing results -> grounded local explanations; no LLM or execution."""
from __future__ import annotations

from pathlib import Path

from shelfcash_forecast.bom.contracts import IngredientDemandPackage
from shelfcash_forecast.contracts import ForecastPackage
from shelfcash_forecast.decision_intelligence.service import build_final_decision_package
from shelfcash_pipeline.context import write_json
from shelfcash_pipeline.m5 import OptimizationCheckpoint


def run(output: Path, forecast: ForecastPackage, ingredient: IngredientDemandPackage,
        optimization: OptimizationCheckpoint) -> None:
    decision = build_final_decision_package(
        optimization.request, optimization.result,
        forecast_package=forecast, ingredient_demand_package=ingredient,
        ingredient_scenario_bundle=optimization.ingredient_scenarios,
        product_scenario_bundle=optimization.product_scenarios,
    )
    destination = output / "m6"
    destination.mkdir(exist_ok=False)
    write_json(destination / "decision_package.json", decision.model_dump(mode="json"))
    write_json(destination / "summary.json", {
        "selected_optimization_mode": optimization.selected_mode,
        "decision_status": decision.decision_status,
        "recommended_strategy": decision.recommended_strategy,
        "order_count": len(decision.immediate_orders),
        "explanation_mode": "deterministic",
        "procurement_executed": False,
        "business_ready": False,
        "provenance": decision.provenance,
        "warnings": decision.warnings,
        "limitations": decision.limitations,
    })
