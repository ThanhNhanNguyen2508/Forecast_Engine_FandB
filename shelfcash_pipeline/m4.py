"""M4: loaded OOS residuals -> bootstrap worlds -> exact FEFO inventory simulation."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from shelfcash_forecast.bom.contracts import IngredientDemandPackage
from shelfcash_forecast.contracts import ForecastPackage
from shelfcash_forecast.inventory.adapters import advanced_inventory_scenarios
from shelfcash_forecast.inventory.contracts import (
    InventoryDemandScenario, InventoryLot, InventorySimulationPolicy,
)
from shelfcash_forecast.inventory.monte_carlo import MonteCarloInventoryRunner
from shelfcash_forecast.scenario.bom import propagate_ingredient_demand_scenarios
from shelfcash_forecast.scenario.composer import generate_product_demand_scenarios
from shelfcash_forecast.scenario.contracts import IngredientDemandScenarioBundle, ProductDemandScenarioBundle
from shelfcash_forecast.scenario.residuals import load_residual_history
from shelfcash_forecast.scenario.yield_loss import FixedRecipeYieldLossModel
from shelfcash_preprocess.engine import create_inventory_lots, load_canonical_frames
from shelfcash_preprocess.pipeline import load_bundle
from shelfcash_pipeline.context import RunOptions, write_json


@dataclass(frozen=True)
class InventoryCheckpoint:
    lots: list[InventoryLot]
    snapshot: date
    policy: InventorySimulationPolicy
    product_scenarios: ProductDemandScenarioBundle
    ingredient_scenarios: IngredientDemandScenarioBundle
    inventory_scenarios: list[InventoryDemandScenario]


def run(options: RunOptions, output: Path, bundle: Path, forecast: ForecastPackage,
        ingredient: IngredientDemandPackage) -> InventoryCheckpoint:
    if not ingredient.is_complete:
        raise ValueError("M4_REQUIRES_COMPLETE_M3")
    info = load_bundle(bundle)
    if info.manifest.context.cutoff_date != options.cutoff_date:
        raise ValueError("BUNDLE_CUTOFF_MISMATCH")
    # This adapter retains the reviewed null-expiry readiness gate.
    lots, snapshot = create_inventory_lots(bundle)
    if snapshot != options.cutoff_date:
        raise ValueError(f"INVENTORY_SNAPSHOT_BOUNDARY_MISMATCH:{snapshot}:{options.cutoff_date}")
    policy = InventorySimulationPolicy(
        unknown_expiry=info.manifest.context.metadata.get("unknown_expiry_policy", "reject"),
    )
    product = generate_product_demand_scenarios(
        forecast, load_residual_history(options.artifacts_path),
        n_scenarios=options.scenario_count, seed=options.seed, method="residual_bootstrap",
    )
    frames = load_canonical_frames(bundle, required_capability="ingredient_demand")
    scenarios = propagate_ingredient_demand_scenarios(
        product, frames["recipes"], frames.get("unit_conversions"),
        yield_loss_model=FixedRecipeYieldLossModel(),
    )
    inventory_scenarios = advanced_inventory_scenarios(scenarios)
    inventory = MonteCarloInventoryRunner().run(
        lots, inventory_scenarios, policy=policy, seed=options.seed,
        simulation_start_date=options.cutoff_date + timedelta(days=1),
        simulation_end_date=options.cutoff_date + timedelta(days=options.horizon),
    )
    destination = output / "m4"
    destination.mkdir(exist_ok=False)
    for name, package in (("product_scenarios", product), ("ingredient_scenarios", scenarios),
                          ("inventory_report", inventory)):
        write_json(destination / f"{name}.json", package.model_dump(mode="json"))
    write_json(destination / "summary.json", {
        "scenario_count": len(inventory.results),
        "scenario_method": product.scenario_method,
        "seed": options.seed,
        "initial_lots": len(lots),
        "snapshot_date": str(snapshot),
        "snapshot_boundary": "EOD",
        "simulation_start": str(inventory.simulation_start_date),
        "simulation_end": str(inventory.simulation_end_date),
        "unknown_expiry_policy": policy.unknown_expiry,
        "yield_loss_source": "recipe_fixed",
        "yield_loss_model_fitted": False,
        "risk_metrics": inventory.risk_metrics.model_dump(mode="json"),
        "warnings": inventory.warnings,
        "business_status": info.manifest.readiness["inventory_simulation"].business_status,
    })
    return InventoryCheckpoint(lots, snapshot, policy, product, scenarios, inventory_scenarios)
