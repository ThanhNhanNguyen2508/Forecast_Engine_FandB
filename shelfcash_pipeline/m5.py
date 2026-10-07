"""M5: shared validated planning boundary and accepted-only independent exports."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from shelfcash_forecast.optimization.contracts import OptimizationRequest, OptimizationResult
from shelfcash_forecast.optimization.planning_service import run_bundle_planning
from shelfcash_forecast.optimization.export import export_planning_run
from shelfcash_forecast.scenario.contracts import IngredientDemandScenarioBundle, ProductDemandScenarioBundle
from shelfcash_pipeline.context import RunOptions
from shelfcash_pipeline.m4 import InventoryCheckpoint


@dataclass(frozen=True)
class OptimizationCheckpoint:
    selected_mode: str
    request: OptimizationRequest
    result: OptimizationResult
    product_scenarios: ProductDemandScenarioBundle
    ingredient_scenarios: IngredientDemandScenarioBundle


def run(options: RunOptions, output: Path, bundle: Path, inventory: InventoryCheckpoint) -> OptimizationCheckpoint:
    if options.planning_config is None:
        raise ValueError("PLANNING_CONFIG_REQUIRED: supply --planning-config for M5/M6")
    planning=run_bundle_planning(bundle,planning=options.planning_config,lots=inventory.lots,snapshot=inventory.snapshot,
        policy=inventory.policy,scenarios=inventory.inventory_scenarios,decision_date=options.cutoff_date,
        planning_end_date=options.cutoff_date+timedelta(days=options.horizon),seed=options.seed,
        optimization_mode=options.optimization_mode,execution_mode=options.execution_mode)
    destination=output/"m5";destination.mkdir(exist_ok=False)
    export_planning_run(planning,destination)
    return OptimizationCheckpoint(planning.selected_mode,planning.request,planning.result,
                                  inventory.product_scenarios,inventory.ingredient_scenarios)
