# Đem candidate plan vừa tối ưu xong vào M4 chạy thật.
from __future__ import annotations
# output của resimulation
# CandidateEvaluation
# │
# ├─ plan
# │    └─ completed=True/False
# │
# ├─ simulation
# │    └─ exact M4
# │
# ├─ stress_simulation
# │
# └─ critic
#      ├─ passed
#      ├─ violations
#      ├─ warnings
#      └─ details

import pandas as pd

from shelfcash_forecast.exceptions import InventoryError
from shelfcash_forecast.inventory.monte_carlo import MonteCarloInventoryRunner
from shelfcash_forecast.inventory.simulator import simulate_inventory_scenarios
from shelfcash_forecast.inventory.stress import run_inventory_stress_tests
from shelfcash_forecast.optimization.adapters import (
    decisions_to_planned_inbound,
    decisions_to_scenario_planned_inbound,
)
from shelfcash_forecast.optimization.contracts import (
    CandidateEvaluation,
    OptimizationRequest,
    ProcurementPlan,
    StrategyProfile,
)
from shelfcash_forecast.optimization.critic import critique_procurement_plan
from shelfcash_forecast.scenario.lead_time import LeadTimeModel
from shelfcash_forecast.scenario.shelf_life import ShelfLifeModel


def evaluate_candidate_plan(
    plan: ProcurementPlan,
    request: OptimizationRequest,
    profile: StrategyProfile,
    *,
    lead_time_model: LeadTimeModel | None = None,
    shelf_life_model: ShelfLifeModel | None = None,
) -> CandidateEvaluation:
    if (lead_time_model is None) != (shelf_life_model is None):
        raise ValueError(
            "lead_time_model and shelf_life_model must be supplied together."
        )
    base_inbound = (
        decisions_to_planned_inbound(plan.orders, plan_id=plan.plan_id)
        if lead_time_model is None
        else []
    )
    recourse_inbound = {
        scenario_id: decisions_to_planned_inbound(
            lines, plan_id=plan.plan_id, scenario_id=scenario_id
        )
        for scenario_id, lines in plan.scenario_recourse_orders.items()
    }
    if lead_time_model is not None and shelf_life_model is not None:
        realized_base = decisions_to_scenario_planned_inbound(
            plan.orders,
            [scenario.scenario_id for scenario in request.demand_scenarios],
            plan_id=plan.plan_id,
            lead_time_model=lead_time_model,
            shelf_life_model=shelf_life_model,
            seed=request.seed,
        )
        for scenario_id, deliveries in realized_base.items():
            recourse_inbound.setdefault(scenario_id, []).extend(deliveries)
    conversion_frame = pd.DataFrame(
        [rule.model_dump() for rule in request.unit_conversions]
    )
    if conversion_frame.empty:
        conversion_frame = None
    simulation = None
    stress_simulation = None
    simulation_error = None
    scenario_starts = [
        scenario.simulation_start_date
        for scenario in request.demand_scenarios
        if scenario.simulation_start_date is not None
    ]
    line_dates = [
        line.target_date
        for scenario in request.demand_scenarios
        for line in scenario.lines
    ]
    transition_start = min(
        scenario_starts or line_dates or [request.decision_date]
    )
    try:
        if all(
            scenario.probability_weight is not None
            for scenario in request.demand_scenarios
        ):
            simulation = MonteCarloInventoryRunner().run(
                request.initial_inventory,
                request.demand_scenarios,
                request.existing_inbound,
                base_inbound,
                scenario_planned_inbound=recourse_inbound,
                policy=request.inventory_policy,
                unit_conversions=conversion_frame,
                cost_assumptions=request.cost_assumptions,
                simulation_start_date=transition_start,
                simulation_end_date=request.planning_end_date,
                seed=request.seed,
            )
        else:
            simulation = simulate_inventory_scenarios(
                request.initial_inventory,
                request.demand_scenarios,
                request.existing_inbound,
                base_inbound,
                policy=request.inventory_policy,
                unit_conversions=conversion_frame,
                cost_assumptions=request.cost_assumptions,
                simulation_start_date=transition_start,
                simulation_end_date=request.planning_end_date,
            )
    except InventoryError as exc:
        simulation_error = f"{exc.code}: {exc}"
    if simulation is not None and request.stress_scenarios:
        source_id = request.stress_base_scenario_id
        baseline = next(
            (
                scenario
                for scenario in request.demand_scenarios
                if source_id is None or scenario.scenario_id == source_id
            ),
            None,
        )
        if baseline is None:
            simulation_error = "STRESS_BASE_SCENARIO_NOT_FOUND"
        else:
            try:
                stress_planned_inbound = (
                    base_inbound
                    if lead_time_model is None
                    else recourse_inbound.get(baseline.scenario_id, [])
                )
                stress_simulation = run_inventory_stress_tests(
                    request.initial_inventory,
                    baseline,
                    request.stress_scenarios,
                    request.existing_inbound,
                    stress_planned_inbound,
                    policy=request.inventory_policy,
                    unit_conversions=conversion_frame,
                    cost_assumptions=request.cost_assumptions,
                    simulation_start_date=transition_start,
                    simulation_end_date=request.planning_end_date,
                )
            except InventoryError as exc:
                simulation_error = f"{exc.code}: {exc}"
    critic = critique_procurement_plan(
        plan,
        request,
        profile,
        simulation,
        stress_simulation=stress_simulation,
        simulation_error=simulation_error,
    )
    completed_plan = plan.model_copy(update={"completed": critic.passed})
    return CandidateEvaluation(
        plan=completed_plan,
        simulation=simulation,
        stress_simulation=stress_simulation,
        critic=critic,
    )
