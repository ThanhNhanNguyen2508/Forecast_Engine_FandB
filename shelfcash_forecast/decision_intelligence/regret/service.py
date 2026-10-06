from __future__ import annotations

from shelfcash_forecast.decision_intelligence.computation_gateway import (
    ComputationGateway,
    M5ComputationGateway,
)
from shelfcash_forecast.decision_intelligence.contracts import FinalDecisionPackage
from shelfcash_forecast.decision_intelligence.integrity import sha256_content_hash
from shelfcash_forecast.decision_intelligence.regret.contracts import (
    CandidateLoss,
    DecisionRegretRequest,
    DecisionRegretResult,
)
from shelfcash_forecast.decision_intelligence.what_if.comparison import decision_snapshot_hash


def _unavailable(
    request: DecisionRegretRequest,
    status: str,
    reason: str,
) -> DecisionRegretResult:
    return DecisionRegretResult(
        regret_request_id=request.regret_request_id,
        status=status,
        selected_plan_id=request.selected_plan_id,
        reason_code=reason,
        comparator_set_complete=False,
        limitations=[
            "No zero regret is fabricated when exact comparable monetary loss is unavailable.",
            "Comparator candidates are not described as a global oracle.",
        ],
    )


def evaluate_decision_regret(
    baseline_decision: FinalDecisionPackage,
    request: DecisionRegretRequest,
    *,
    gateway: ComputationGateway | None = None,
) -> DecisionRegretResult:
    """Compute candidate-set regret using exact replay on one common scenario."""

    if not request.confirmed:
        return _unavailable(request, "FAILED_VALIDATION", "M6_REGRET_CONFIRMATION_REQUIRED")
    if baseline_decision.request_id != request.baseline_request_id or (
        decision_snapshot_hash(baseline_decision) != request.baseline_decision_hash
    ):
        return _unavailable(request, "FAILED_VALIDATION", "M6_REGRET_BASELINE_BINDING_FAILED")
    if baseline_decision.recommended_strategy != request.selected_strategy:
        return _unavailable(request, "FAILED_VALIDATION", "M6_REGRET_SELECTED_STRATEGY_MISMATCH")
    plan_ids = [plan.plan_id for plan in request.comparator_plans]
    if len(plan_ids) != len(set(plan_ids)):
        return _unavailable(request, "FAILED_VALIDATION", "M6_REGRET_DUPLICATE_COMPARATOR")
    if request.selected_plan_id not in plan_ids:
        return _unavailable(request, "FAILED_VALIDATION", "M6_REGRET_SELECTED_PLAN_NOT_INCLUDED")
    if not request.monetary_unit:
        return _unavailable(request, "INCOMPARABLE", "M6_REGRET_MONETARY_UNIT_UNAVAILABLE")
    scenarios = request.evaluation_request.demand_scenarios
    if len(scenarios) != 1:
        return _unavailable(request, "INCOMPARABLE", "M6_REGRET_EXACT_SCENARIO_REQUIRED")
    scenario_id = scenarios[0].scenario_id
    selected_gateway = gateway or M5ComputationGateway()
    losses: list[CandidateLoss] = []
    for plan in request.comparator_plans:
        evaluation = selected_gateway.evaluate_plan(plan, request.evaluation_request)
        if evaluation.simulation is None or len(evaluation.simulation.results) != 1:
            return _unavailable(request, "UNAVAILABLE", "M6_REGRET_EXACT_M4_UNAVAILABLE")
        consequence = evaluation.simulation.results[0].summary.consequence_cost
        if consequence is None:
            return _unavailable(request, "UNAVAILABLE", "M6_REGRET_CONSEQUENCE_COST_UNAVAILABLE")
        recourse = sum(
            line.purchase_cost + line.delivery_cost
            for line in plan.scenario_recourse_orders.get(scenario_id, [])
        )
        total = plan.purchase_cost + recourse + consequence
        losses.append(
            CandidateLoss(
                plan_id=plan.plan_id,
                strategy=plan.strategy,
                first_stage_purchase_and_delivery_cost=plan.purchase_cost,
                applicable_recourse_cost=recourse,
                exact_m4_consequence_cost=consequence,
                total_exact_loss=total,
                monetary_unit=request.monetary_unit,
                exact_simulation_hash=sha256_content_hash(evaluation.simulation),
                critic_passed=evaluation.critic.passed,
                hard_violations=sorted(evaluation.critic.hard_violations),
            )
        )
    losses.sort(key=lambda loss: (loss.total_exact_loss, loss.plan_id))
    selected = next(loss for loss in losses if loss.plan_id == request.selected_plan_id)
    best = losses[0]
    regret = max(0.0, selected.total_exact_loss - best.total_exact_loss)
    return DecisionRegretResult(
        regret_request_id=request.regret_request_id,
        status=(
            "COMPUTED_REALIZED"
            if request.evaluation_kind == "REALIZED"
            else "COMPUTED_HYPOTHETICAL"
        ),
        selected_plan_id=request.selected_plan_id,
        selected_exact_loss=selected.total_exact_loss,
        best_comparator_plan_id=best.plan_id,
        minimum_comparator_set_loss=best.total_exact_loss,
        candidate_set_regret=regret,
        monetary_unit=request.monetary_unit,
        candidate_losses=losses,
        reason_code="M6_REGRET_EXACT_CANDIDATE_SET_COMPARISON",
        comparator_set_complete=True,
        limitations=[
            "Regret is exact only for the supplied candidate set and common evaluation scenario.",
            "The best comparator is not a global oracle.",
        ],
    )


__all__ = ["evaluate_decision_regret"]
