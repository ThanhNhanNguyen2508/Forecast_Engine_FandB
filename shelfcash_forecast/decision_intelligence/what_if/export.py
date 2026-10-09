"""Typed export adapter for an actual public What-if result, without rebuilding demand."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from shelfcash_forecast.decision_intelligence.contracts import StrictDecisionContract
from shelfcash_forecast.decision_intelligence.what_if.contracts import WhatIfDecisionPackage
from shelfcash_forecast.optimization.contracts import OptimizationRequest, OptimizationResult
from shelfcash_forecast.optimization.planning_config import PlanningConfig
from shelfcash_forecast.optimization.planning_service import content_hash


class HypotheticalExportConfiguration(StrictDecisionContract):
    configuration_role: Literal["RESOLVED_HYPOTHETICAL_REQUEST_WITH_PARENT_CONFIG"] = "RESOLVED_HYPOTHETICAL_REQUEST_WITH_PARENT_CONFIG"
    parent_config: PlanningConfig
    parent_config_hash: str
    parent_request_hash: str
    what_if_id: str
    transformed_request_hash: str
    transformed_full_pool_hash: str
    transformed_planning_binding: dict
    execution_parameters: dict
    business_ready: Literal[False] = False
    execution_authorized: Literal[False] = False


@dataclass
class WhatIfPlanningRun:
    selected_mode: str
    request: OptimizationRequest
    result: OptimizationResult
    runs: dict
    config: HypotheticalExportConfiguration
    provenance: dict


def export_what_if_planning(package: WhatIfDecisionPackage, parent_config: PlanningConfig,
                           destination: Path, *, source_bundle: str | None, baseline_path: str):
    from shelfcash_forecast.optimization.export import export_planning_run
    request = package.modified_request
    parameters = {k: request.model_dump(mode="json")[k] for k in
                  ("seed", "limits", "candidate_generation", "stochastic", "budget", "budget_scope",
                   "currency", "strategy_profiles", "inventory_policy", "stress_scenarios", "stress_base_scenario_id")}
    config = HypotheticalExportConfiguration(parent_config=parent_config, parent_config_hash=content_hash(parent_config),
        parent_request_hash=package.baseline_request_hash, what_if_id=package.what_if_id,
        transformed_request_hash=package.modified_request_hash,
        transformed_full_pool_hash=request.scenario_provenance["full_pool_hash"],
        transformed_planning_binding=request.planning_binding, execution_parameters=parameters)
    mode = "stochastic" if request.stochastic else "deterministic"
    run = WhatIfPlanningRun(mode, request, package.optimization_result,
        {mode: (request, package.optimization_result)}, config,
        {"source_bundle": source_bundle, "baseline_path": baseline_path, "what_if_id": package.what_if_id,
         "compare_selection_policy": "PUBLIC_WHAT_IF_REOPTIMIZE_BASELINE_SELECTED_MODE",
         "demand_origin": "TRANSFORMED_EXISTING_M4_INGREDIENT_POOL", "config_role": config.configuration_role})
    return export_planning_run(run, destination, include_full_result=False, compact_attempts=True)
