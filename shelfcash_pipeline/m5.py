"""M5: explicit cost configuration -> real optimization + exact M4/critic checks."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

from shelfcash_forecast.application import _cost_assumptions
from shelfcash_forecast.optimization.contracts import OptimizationRequest, OptimizationResult
from shelfcash_forecast.optimization.optimizer import optimize_procurement
from shelfcash_forecast.optimization.strategies import default_strategy_profiles
from shelfcash_forecast.scenario.contracts import IngredientDemandScenarioBundle, ProductDemandScenarioBundle
from shelfcash_preprocess.engine import create_supplier_offers
from shelfcash_preprocess.pipeline import load_bundle
from shelfcash_pipeline.context import RunOptions, write_json
from shelfcash_pipeline.m4 import InventoryCheckpoint


@dataclass(frozen=True)
class OptimizationCheckpoint:
    selected_mode: str
    request: OptimizationRequest
    result: OptimizationResult
    product_scenarios: ProductDemandScenarioBundle
    ingredient_scenarios: IngredientDemandScenarioBundle


def run(options: RunOptions, output: Path, bundle: Path,
        inventory: InventoryCheckpoint) -> OptimizationCheckpoint:
    if options.planning_config is None:
        raise ValueError("PLANNING_CONFIG_REQUIRED: supply --planning-config for M5/M6")
    planning = json.loads(options.planning_config.read_text(encoding="utf-8-sig"))
    if options.execution_mode == "demo" and planning.get("label") != "DEMO_ONLY_NOT_FOR_OPERATION":
        raise ValueError("DEMO_PLANNING_CONFIG_LABEL_REQUIRED")
    offers = create_supplier_offers(bundle, options.cutoff_date)
    costs = _cost_assumptions(offers, planning)
    count = min(len(inventory.inventory_scenarios),
                int(planning.get("optimization_scenario_count", len(inventory.inventory_scenarios))))
    if count < 1:
        raise ValueError("OPTIMIZATION_REQUIRES_AT_LEAST_ONE_SCENARIO")
    if count < 2 and options.optimization_mode in {"stochastic", "compare"}:
        raise ValueError("STOCHASTIC_OPTIMIZATION_REQUIRES_AT_LEAST_TWO_SCENARIOS")
    scenarios = [row.model_copy(update={"probability_weight": 1.0 / count})
                 for row in inventory.inventory_scenarios[:count]]
    product = inventory.product_scenarios.model_copy(update={"scenarios": [
        row.model_copy(update={"probability_weight": 1.0 / count})
        for row in inventory.product_scenarios.scenarios[:count]
    ]})
    ingredient = inventory.ingredient_scenarios.model_copy(update={"scenarios": [
        row.model_copy(update={"probability_weight": 1.0 / count})
        for row in inventory.ingredient_scenarios.scenarios[:count]
    ]})
    modes = (["deterministic", "stochastic"] if options.optimization_mode == "compare"
             else [options.optimization_mode])
    info = load_bundle(bundle)
    runs = {}
    for mode in modes:
        request = OptimizationRequest(
            request_id=f"{info.manifest.run_id}-{mode}",
            decision_date=options.cutoff_date,
            planning_end_date=options.cutoff_date + timedelta(days=options.horizon),
            initial_inventory=inventory.lots, demand_scenarios=scenarios,
            supplier_offers=offers, cost_assumptions=costs,
            strategy_profiles=default_strategy_profiles(), budget=planning.get("budget"),
            inventory_policy=inventory.policy, seed=options.seed,
            inventory_snapshot_date=inventory.snapshot, inventory_snapshot_boundary="EOD",
            unknown_constraints=(["DEMO_CONSEQUENCE_COSTS_NOT_APPROVED"]
                                 if planning.get("label") == "DEMO_ONLY_NOT_FOR_OPERATION" else []),
            stochastic=mode == "stochastic", allow_mode_fallback=False,
        )
        runs[mode] = (request, optimize_procurement(request))
    selected = "stochastic" if "stochastic" in runs else modes[0]
    destination = output / "m5"
    destination.mkdir(exist_ok=False)
    write_json(destination / "optimization_result.json", {
        mode: {"request": request.model_dump(mode="json"), "result": result.model_dump(mode="json")}
        for mode, (request, result) in runs.items()
    })
    write_json(destination / "planning_config.json", planning)
    write_json(destination / "summary.json", {
        "requested_mode": options.optimization_mode,
        "selected_mode": selected,
        "optimization_scenario_count": count,
        "m4_diagnostic_scenario_count": len(inventory.inventory_scenarios),
        "same_evaluation_scenarios": True,
        "same_sample_optimism": True,
        "business_ready": False,
        "assumption_label": planning.get("label"),
        "results": {mode: {"status": result.status,
                           "recommended_strategy": result.recommended_strategy,
                           "actual_mode": result.provenance.get("actual_mode"),
                           "warnings": result.warnings}
                    for mode, (_, result) in runs.items()},
    })
    request, result = runs[selected]
    return OptimizationCheckpoint(selected, request, result, product, ingredient)
