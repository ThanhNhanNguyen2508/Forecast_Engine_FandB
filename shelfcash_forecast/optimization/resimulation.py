"""Independent exact physics check and full-pool final evaluation."""
from __future__ import annotations

import pandas as pd
from datetime import timedelta
from shelfcash_forecast.exceptions import InventoryError
from shelfcash_forecast.inventory.monte_carlo import MonteCarloInventoryRunner
from shelfcash_forecast.inventory.simulator import simulate_inventory_scenarios
from shelfcash_forecast.inventory.stress import run_inventory_stress_tests
from shelfcash_forecast.optimization.adapters import decisions_to_planned_inbound
from shelfcash_forecast.optimization.contracts import CandidateEvaluation, ProcurementPlan
from shelfcash_forecast.optimization.critic import critique_procurement_plan
from shelfcash_forecast.optimization.lot_milp import mean_world


def evaluate_candidate_plan(plan, request, profile, *, lead_time_model=None, shelf_life_model=None):
    if lead_time_model is not None or shelf_life_model is not None:
        raise ValueError("EXTERNAL_SUPPLY_UNCERTAINTY_NOT_SUPPORTED_BY_LOT_MODEL")
    if any(plan.scenario_recourse_orders.values()):
        raise ValueError("RECOURSE_POLICY_NOT_SUPPORTED")
    inbound = decisions_to_planned_inbound(plan.orders, plan_id=plan.plan_id)
    conversions = pd.DataFrame([r.model_dump() for r in request.unit_conversions])
    conversions = None if conversions.empty else conversions
    full = request.evaluation_scenarios or request.demand_scenarios
    physics_worlds = request.demand_scenarios if plan.provenance.get("mode") != "deterministic" else [mean_world(request)]
    def exact(worlds):
        kwargs = dict(policy=request.inventory_policy, unit_conversions=conversions,
                      cost_assumptions=request.cost_assumptions,
                      simulation_start_date=request.decision_date + timedelta(days=1),
                      simulation_end_date=request.planning_end_date)
        if all(s.probability_weight is not None for s in worlds):
            return MonteCarloInventoryRunner().run(request.initial_inventory, worlds, request.existing_inbound,
                                                  inbound, seed=request.seed, **kwargs)
        return simulate_inventory_scenarios(request.initial_inventory, worlds, request.existing_inbound, inbound, **kwargs)
    simulation = physics = stress = None; error = None
    try:
        physics = exact(physics_worlds)
        simulation = exact(full)
        if request.stress_scenarios:
            baseline = next((s for s in full if request.stress_base_scenario_id is None or s.scenario_id == request.stress_base_scenario_id), None)
            if baseline is None:
                raise ValueError("STRESS_BASE_SCENARIO_NOT_FOUND")
            stress = run_inventory_stress_tests(request.initial_inventory, baseline, request.stress_scenarios,
                request.existing_inbound, inbound, policy=request.inventory_policy, unit_conversions=conversions,
                cost_assumptions=request.cost_assumptions, simulation_start_date=simulation.simulation_start_date,
                simulation_end_date=request.planning_end_date)
    except InventoryError as exc:
        error = f"{exc.code}: {exc}"
    critic = critique_procurement_plan(plan, request, profile, simulation, stress_simulation=stress,
                                      simulation_error=error, physics_simulation=physics)
    final = ProcurementPlan.model_validate({**plan.model_dump(), "completed": critic.passed})
    return CandidateEvaluation(plan=final, simulation=simulation, physics_simulation=physics,
                               stress_simulation=stress, critic=critic)
