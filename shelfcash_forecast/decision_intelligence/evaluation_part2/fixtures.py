from __future__ import annotations

from datetime import date, timedelta

from shelfcash_forecast.decision_intelligence.computation_gateway import (
    ComputationGateway,
    M5ComputationGateway,
)
from shelfcash_forecast.decision_intelligence.contracts import FinalDecisionPackage
from shelfcash_forecast.decision_intelligence.service import build_final_decision_package
from shelfcash_forecast.inventory.contracts import (
    ConsequenceCostAssumption,
    InventoryDemandLine,
    InventoryDemandScenario,
    InventoryLot,
)
from shelfcash_forecast.inventory.stress import StressScenarioDefinition
from shelfcash_forecast.optimization.contracts import (
    CandidateEvaluation,
    OptimizationRequest,
    OptimizationResult,
    ProcurementPlan,
    SupplierOffer,
)


class AuditedGateway:
    def __init__(self, inner: ComputationGateway | None = None) -> None:
        self.inner = inner or M5ComputationGateway()
        self.optimize_count = 0
        self.evaluate_count = 0

    def optimize(self, request: OptimizationRequest) -> OptimizationResult:
        self.optimize_count += 1
        return self.inner.optimize(request)

    def evaluate_plan(
        self,
        plan: ProcurementPlan,
        request: OptimizationRequest,
    ) -> CandidateEvaluation:
        self.evaluate_count += 1
        return self.inner.evaluate_plan(plan, request)


class Part2Fixture:
    def __init__(
        self,
        request: OptimizationRequest,
        result: OptimizationResult,
        decision: FinalDecisionPackage,
        gateway: AuditedGateway,
    ) -> None:
        self.request = request
        self.result = result
        self.decision = decision
        self.gateway = gateway


def build_part2_fixture() -> Part2Fixture:
    decision_date = date(2026, 8, 14)
    planning_end = date(2026, 8, 16)
    scenarios = [
        InventoryDemandScenario(
            scenario_id=scenario_id,
            probability_weight=0.5,
            simulation_start_date=decision_date,
            simulation_end_date=planning_end,
            lines=[
                InventoryDemandLine(
                    scenario_id=scenario_id,
                    store_id="S1",
                    ingredient_id="I1",
                    target_date=date(2026, 8, 15),
                    quantity=quantity,
                    unit="kg",
                )
            ],
            provenance={"fixture": "part2-production-authority"},
        )
        for scenario_id, quantity in (("LOW", 4.0), ("HIGH", 8.0))
    ]
    request = OptimizationRequest(
        request_id="M6-PART2-BASELINE",
        decision_date=decision_date,
        planning_end_date=planning_end,
        initial_inventory=[
            InventoryLot(
                lot_id="L1",
                store_id="S1",
                ingredient_id="I1",
                quantity_remaining=2,
                unit="kg",
                received_date=decision_date - timedelta(days=1),
                expiry_date=planning_end,
            )
        ],
        demand_scenarios=scenarios,
        supplier_offers=[
            SupplierOffer(
                offer_id="O1",
                supplier_id="SUP1",
                store_id="S1",
                ingredient_id="I1",
                unit="kg",
                order_date=decision_date,
                pack_size=1,
                unit_price=2,
                delivery_cost=1,
                minimum_order_quantity=1,
                maximum_order_quantity=20,
                lead_time_days=0,
                shelf_life_days=2,
            )
        ],
        cost_assumptions=[
            ConsequenceCostAssumption(
                store_id="S1",
                ingredient_id="I1",
                unit="kg",
                holding_cost_per_unit_day=0.1,
                shortage_cost_per_unit=10,
                expired_cost_per_unit=1,
                waste_cost_per_unit=1,
                capacity_quantity=30,
            )
        ],
        stress_scenarios=[
            StressScenarioDefinition(
                stress_id="DELAY_STRESS",
                demand_multiplier=1.5,
                supplier_delay_days=1,
                supplier_ids={"SUP1"},
                description="Adversarial delay without probability assignment.",
            )
        ],
        stress_base_scenario_id="LOW",
        stochastic=True,
        seed=20260814,
    )
    gateway = AuditedGateway()
    result = gateway.optimize(request)
    decision = build_final_decision_package(request, result)
    return Part2Fixture(request, result, decision, gateway)


def single_scenario_request(request: OptimizationRequest) -> OptimizationRequest:
    data = request.model_dump(mode="python")
    scenario = data["demand_scenarios"][0]
    scenario["probability_weight"] = None
    data["demand_scenarios"] = [scenario]
    data["stochastic"] = False
    data["request_id"] = "M6-PART2-REALIZED"
    return OptimizationRequest.model_validate(data)


__all__ = ["AuditedGateway", "Part2Fixture", "build_part2_fixture", "single_scenario_request"]
