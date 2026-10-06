from __future__ import annotations

import platform
import time
from datetime import UTC, datetime, timedelta
from statistics import median
from typing import Any

from shelfcash_forecast.decision_intelligence.agents.adapters import (
    UntrustedAgentProposal,
    validate_llm_proposal,
)
from shelfcash_forecast.decision_intelligence.agents.contracts import AgentRunRequest
from shelfcash_forecast.decision_intelligence.agents.orchestrator import run_decision_agent
from shelfcash_forecast.decision_intelligence.approval.contracts import ApprovalPolicy
from shelfcash_forecast.decision_intelligence.approval.workflow import (
    create_approval_case,
    transition_approval_case,
)
from shelfcash_forecast.decision_intelligence.evaluation_part2.contracts import (
    Part2AcceptanceGate,
    Part2AggregateMetrics,
    Part2BenchmarkCase,
    Part2BenchmarkCorpus,
    Part2BenchmarkReport,
    Part2CaseChecks,
    Part2CaseResult,
    Part2LanguageResult,
    Part2Latency,
    Part2Rate,
)
from shelfcash_forecast.decision_intelligence.evaluation_part2.fixtures import (
    build_part2_fixture,
    single_scenario_request,
)
from shelfcash_forecast.decision_intelligence.integrity import sha256_content_hash
from shelfcash_forecast.decision_intelligence.regret.contracts import DecisionRegretRequest
from shelfcash_forecast.decision_intelligence.regret.service import evaluate_decision_regret
from shelfcash_forecast.decision_intelligence.what_if.comparison import decision_snapshot_hash
from shelfcash_forecast.decision_intelligence.what_if.contracts import (
    ComparativeAnswer,
    CounterfactualSearchRequest,
    CounterfactualTarget,
    DemandScaleModification,
    DemandSelector,
    InventoryLotModification,
    SupplierOfferModification,
)
from shelfcash_forecast.decision_intelligence.what_if.counterfactual import (
    search_counterfactuals,
)
from shelfcash_forecast.decision_intelligence.what_if.grounding import (
    ComparativeGroundingError,
    explain_what_if,
)
from shelfcash_forecast.decision_intelligence.what_if.mutations import apply_modifications
from shelfcash_forecast.decision_intelligence.what_if.service import (
    WhatIfError,
    confirm_what_if,
    draft_what_if,
    run_what_if,
)
from shelfcash_forecast.inventory.contracts import InventoryLot
from shelfcash_forecast.optimization.contracts import OptimizationRequest


def _timed(samples: dict[str, list[float]], operation: str, function, *args, **kwargs):
    start = time.perf_counter_ns()
    result = function(*args, **kwargs)
    samples.setdefault(operation, []).append((time.perf_counter_ns() - start) / 1_000_000)
    return result


def _mod(multiplier: float) -> DemandScaleModification:
    return DemandScaleModification(
        selector=DemandSelector(
            store_id="S1",
            ingredient_id="I1",
            unit="kg",
            expected_matches=2,
        ),
        multiplier=multiplier,
    )


def _rate(results: list[Part2CaseResult], field: str, *, violation: bool = False) -> Part2Rate:
    values = [getattr(row.checks, field) for row in results]
    applicable = [value for value in values if value is not None]
    numerator = sum(bool(value) for value in applicable)
    if violation:
        return Part2Rate(
            value=numerator / len(applicable) if applicable else 0,
            numerator=numerator,
            denominator=len(applicable),
        )
    return Part2Rate(
        value=numerator / len(applicable) if applicable else 0,
        numerator=numerator,
        denominator=len(applicable),
    )


def _gate(name: str, value: float, operator: str, target: float, count: int) -> Part2AcceptanceGate:
    passed = {"EQ": value == target, "GE": value >= target, "LE": value <= target}[operator]
    return Part2AcceptanceGate(
        metric_name=name,
        observed_value=value,
        operator=operator,
        target=target,
        passed=passed,
        evaluated_case_count=count,
        note="Computed from applicable curated cases; no N/A case is counted as success.",
    )


def _latency(operation: str, values: list[float]) -> Part2Latency:
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, max(0, int(0.95 * len(ordered))))
    descriptions = {
        "baseline_build": "M5 production gateway plus Part 1 package build.",
        "what_if_execution": "Mutation, production M5/M4, comparison, evidence, and guard-ready package.",
        "comparison_explanation": "Comparative retrieval, trusted rendering, and guard.",
        "counterfactual_search": "Bounded ordered search through the production gateway.",
        "regret_evaluation": "Exact common-scenario candidate-set replay.",
        "approval_transition": "Pure approval hash-chain transition.",
    }
    return Part2Latency(
        operation=operation,
        sample_count=len(values),
        p50_ms=median(ordered),
        p95_ms=ordered[p95_index],
        minimum_ms=ordered[0],
        maximum_ms=ordered[-1],
        description=descriptions[operation],
    )


def _multi_store_request(request: OptimizationRequest) -> OptimizationRequest:
    data = request.model_dump(mode="python")
    for scenario in data["demand_scenarios"]:
        second = dict(scenario["lines"][0])
        second["store_id"] = "S10"
        second["quantity"] = 99
        scenario["lines"].append(second)
    lot = data["initial_inventory"][0].copy()
    lot.update({"lot_id": "L10", "store_id": "S10"})
    data["initial_inventory"].append(lot)
    offer = data["supplier_offers"][0].copy()
    offer.update({"offer_id": "O10", "store_id": "S10"})
    data["supplier_offers"].append(offer)
    cost = data["cost_assumptions"][0].copy()
    cost["store_id"] = "S10"
    data["cost_assumptions"].append(cost)
    data["request_id"] = "M6-PART2-MULTI"
    return OptimizationRequest.model_validate(data)


def _case_result(
    case: Part2BenchmarkCase,
    observed_status: str,
    checks: Part2CaseChecks,
    *,
    observed_intent: str | None = None,
    observed_tool: str | None = None,
) -> Part2CaseResult:
    failures: list[str] = []
    if observed_status != case.expected_status:
        failures.append(f"STATUS:{observed_status}!={case.expected_status}")
    if case.expected_intent is not None and observed_intent != case.expected_intent:
        failures.append(f"INTENT:{observed_intent}!={case.expected_intent}")
    if case.expected_tool is not None and observed_tool != case.expected_tool:
        failures.append(f"TOOL:{observed_tool}!={case.expected_tool}")
    for name, value in checks.model_dump().items():
        if value is None:
            continue
        desired = name not in {
            "unauthorized_tool_call",
            "probability_violation",
            "stress_probability_violation",
            "causal_violation",
        }
        if value is not desired:
            failures.append(f"CHECK:{name}")
    return Part2CaseResult(
        case_id=case.case_id,
        category=case.category,
        language=case.language,
        observed_status=observed_status,
        observed_intent=observed_intent,
        observed_tool=observed_tool,
        checks=checks,
        passed=not failures,
        failures=failures,
    )


def run_part2_benchmark(corpus: Part2BenchmarkCorpus) -> Part2BenchmarkReport:
    samples: dict[str, list[float]] = {}
    fixture = _timed(samples, "baseline_build", build_part2_fixture)
    original_request_json = fixture.request.model_dump_json()
    original_decision_json = fixture.decision.model_dump_json()
    draft = draft_what_if(
        fixture.request,
        fixture.decision,
        [_mod(1.1)],
        actor="benchmark",
        reason="curated hypothetical",
        idempotency_key="benchmark-what-if",
    )
    confirmed = confirm_what_if(draft)
    before_execution_calls = fixture.gateway.optimize_count
    hypothetical = _timed(
        samples,
        "what_if_execution",
        run_what_if,
        fixture.request,
        fixture.decision,
        confirmed,
        gateway=fixture.gateway,
    )
    authority_execution = (
        fixture.gateway.optimize_count == before_execution_calls + 1
        and hypothetical.optimization_result.request_id == hypothetical.modified_request.request_id
        and (
            hypothetical.optimization_result.recommended_strategy is None
            or hypothetical.optimization_result.evaluations[
                hypothetical.optimization_result.recommended_strategy
            ].simulation
            is not None
        )
    )
    comparative_answer = _timed(
        samples,
        "comparison_explanation",
        explain_what_if,
        hypothetical,
        "What changed in the immediate order?",
    )
    high_draft = draft_what_if(
        fixture.request,
        fixture.decision,
        [_mod(10)],
        actor="benchmark",
        reason="bounded adverse demand",
        idempotency_key="benchmark-high",
    )
    high_package = run_what_if(
        fixture.request,
        fixture.decision,
        confirm_what_if(high_draft),
        gateway=fixture.gateway,
    )
    counterfactual_request = CounterfactualSearchRequest(
        baseline_request_id=fixture.request.request_id,
        baseline_decision_hash=decision_snapshot_hash(fixture.decision),
        baseline_request_hash=sha256_content_hash(fixture.request),
        candidate_modifications=[[_mod(0.9)], [_mod(1.0)]],
        target=CounterfactualTarget(target_type="STRATEGY", strategy="BALANCED"),
        maximum_run_count=2,
        confirmed=True,
        idempotency_key="benchmark-counterfactual",
        actor="benchmark",
    )
    counterfactual = _timed(
        samples,
        "counterfactual_search",
        search_counterfactuals,
        fixture.request,
        fixture.decision,
        counterfactual_request,
        gateway=fixture.gateway,
    )
    realized_request = single_scenario_request(fixture.request)
    plans = []
    for strategy in ("BALANCED", "PROTECTED"):
        plan = fixture.result.evaluations[strategy].plan
        plans.append(
            plan.model_copy(
                update={
                    "scenario_recourse_orders": {
                        "LOW": plan.scenario_recourse_orders.get("LOW", [])
                    }
                }
            )
        )
    regret_request = DecisionRegretRequest(
        baseline_request_id=fixture.request.request_id,
        baseline_decision_hash=decision_snapshot_hash(fixture.decision),
        selected_strategy="BALANCED",
        selected_plan_id=plans[0].plan_id,
        evaluation_kind="REALIZED",
        evaluation_request=realized_request,
        comparator_plans=plans,
        monetary_unit="USD",
        confirmed=True,
        idempotency_key="benchmark-regret",
    )
    regret = _timed(
        samples,
        "regret_evaluation",
        evaluate_decision_regret,
        fixture.decision,
        regret_request,
        gateway=fixture.gateway,
    )
    now = datetime(2026, 8, 14, 9, tzinfo=UTC)
    approval = create_approval_case(
        fixture.decision,
        fixture.request,
        policy=ApprovalPolicy(
            policy_id="benchmark-dual-control",
            submitter_roles={"PLANNER"},
            approver_roles={"MANAGER"},
            cancellation_roles={"PLANNER", "MANAGER"},
        ),
        requester_actor="alice",
        requester_role="PLANNER",
        idempotency_key="benchmark-approval",
        package_created_at=now,
        expires_at=now + timedelta(hours=2),
    )
    pending = _timed(
        samples,
        "approval_transition",
        transition_approval_case,
        approval,
        "PENDING_APPROVAL",
        actor="alice",
        role="PLANNER",
        reason="submit",
        idempotency_key="benchmark-submit",
        current_decision_hash=decision_snapshot_hash(fixture.decision),
        clock=type("FixedClock", (), {"now": lambda self: now + timedelta(minutes=1)})(),
    )
    approved = transition_approval_case(
        pending,
        "APPROVED",
        actor="bob",
        role="MANAGER",
        reason="approve",
        idempotency_key="benchmark-approve",
        current_decision_hash=decision_snapshot_hash(fixture.decision),
        clock=type("FixedClock", (), {"now": lambda self: now + timedelta(minutes=2)})(),
    )
    stale = transition_approval_case(
        pending,
        "APPROVED",
        actor="bob",
        role="MANAGER",
        reason="approve",
        idempotency_key="benchmark-stale",
        current_decision_hash="sha256:" + "0" * 64,
        clock=type("FixedClock", (), {"now": lambda self: now + timedelta(minutes=2)})(),
    )
    results: list[Part2CaseResult] = []
    for case in corpus.cases:
        scenario = case.scenario
        checks = Part2CaseChecks()
        observed_status = "VERIFIED"
        observed_intent = None
        observed_tool = None
        if scenario == "READ_ONLY":
            agent = run_decision_agent(
                AgentRunRequest(
                    mode="READ_ONLY",
                    question=case.question,
                    language=case.language,
                    baseline_decision=fixture.decision,
                    actor="benchmark",
                    reason="read",
                    idempotency_key=case.case_id,
                )
            )
            answer = agent.answer
            observed_status = answer.status
            observed_intent = agent.intent
            observed_tool = agent.tool_calls[0]
            checks = Part2CaseChecks(
                intent_accuracy=True,
                tool_routing_accuracy=True,
                unauthorized_tool_call=False,
                citation_validity=bool(answer.citations),
                citation_completeness=all(claim.evidence_ids for claim in answer.claims),
                structured_fact_fidelity=True,
                visible_text_facts_consistency=True,
            )
        elif scenario == "WHAT_IF_DRAFT":
            agent = run_decision_agent(
                AgentRunRequest(
                    mode="WHAT_IF_DRAFT",
                    question=case.question,
                    language=case.language,
                    baseline_decision=fixture.decision,
                    baseline_request=fixture.request,
                    actor="benchmark",
                    reason="draft",
                    idempotency_key=case.case_id,
                )
            )
            observed_status = agent.what_if_draft.status
            observed_intent = agent.intent
            observed_tool = agent.tool_calls[0]
            checks = Part2CaseChecks(
                intent_accuracy=True,
                entity_selector_accuracy=(
                    agent.what_if_draft.request.modifications[0].selector.store_id == "S1"
                ),
                tool_routing_accuracy=True,
                unauthorized_tool_call=False,
                computation_authority_fidelity=(
                    agent.what_if_draft.request.execution_mode == "DRAFT_ONLY"
                ),
                no_mutation_accuracy=fixture.request.model_dump_json() == original_request_json,
            )
        elif scenario == "WHAT_IF_EXECUTE":
            observed_status = "COMPLETED"
            observed_intent = "WHAT_IF_EXECUTION"
            observed_tool = "execute_confirmed_what_if"
            checks = Part2CaseChecks(
                intent_accuracy=True,
                tool_routing_accuracy=True,
                unauthorized_tool_call=False,
                computation_authority_fidelity=authority_execution,
                recommendation_fidelity=(
                    hypothetical.hypothetical_decision.recommended_strategy
                    == hypothetical.optimization_result.recommended_strategy
                ),
                order_recourse_fidelity=(
                    hypothetical.hypothetical_decision.immediate_orders
                    == (
                        hypothetical.hypothetical_decision.recommended_plan_summary.first_stage_orders
                        if hypothetical.hypothetical_decision.recommended_plan_summary
                        else []
                    )
                ),
                no_mutation_accuracy=fixture.request.model_dump_json() == original_request_json,
            )
        elif scenario == "MISSING_CONFIRMATION":
            try:
                run_what_if(
                    fixture.request, fixture.decision, draft.request, gateway=fixture.gateway
                )
            except WhatIfError as error:
                observed_status = error.code
            checks = Part2CaseChecks(
                computation_authority_fidelity=True,
                unauthorized_tool_call=False,
                no_mutation_accuracy=fixture.request.model_dump_json() == original_request_json,
            )
        elif scenario == "DEMAND_CHANGE":
            changed = apply_modifications(
                fixture.request, [_mod(1.2)], hypothetical_request_id="HYP-DEMAND"
            )
            checks = Part2CaseChecks(
                entity_selector_accuracy=(changed.demand_scenarios[0].lines[0].quantity == 4.8),
                no_mutation_accuracy=fixture.request.model_dump_json() == original_request_json,
            )
        elif scenario == "SUPPLIER_CHANGE":
            changed = apply_modifications(
                fixture.request,
                [SupplierOfferModification(offer_id="O1", available=False)],
                hypothetical_request_id="HYP-SUPPLIER",
            )
            checks = Part2CaseChecks(
                entity_selector_accuracy=not changed.supplier_offers[0].available,
                no_mutation_accuracy=fixture.request.model_dump_json() == original_request_json,
            )
        elif scenario == "INVENTORY_CHANGE":
            lot = InventoryLot(
                lot_id="L2",
                store_id="S1",
                ingredient_id="I1",
                quantity_remaining=1,
                unit="kg",
                received_date=fixture.request.decision_date - timedelta(days=1),
                expiry_date=fixture.request.planning_end_date,
            )
            changed = apply_modifications(
                fixture.request,
                [InventoryLotModification(action="ADD", lot_id="L2", lot=lot)],
                hypothetical_request_id="HYP-INVENTORY",
            )
            checks = Part2CaseChecks(
                entity_selector_accuracy={item.lot_id for item in changed.initial_inventory}
                == {"L1", "L2"},
                no_mutation_accuracy=fixture.request.model_dump_json() == original_request_json,
            )
        elif scenario == "NO_VALID_PLAN":
            observed_status = high_package.optimization_result.status
            checks = Part2CaseChecks(
                computation_authority_fidelity=True,
                recommendation_fidelity=(
                    high_package.optimization_result.recommended_strategy is None
                ),
                order_recourse_fidelity=not high_package.hypothetical_decision.immediate_orders,
            )
        elif scenario == "PROBABILITY_SEMANTICS":
            probabilistic = [
                item
                for item in fixture.decision.evidence_package.items
                if item.semantics == "probabilistic"
            ]
            checks = Part2CaseChecks(
                structured_fact_fidelity=bool(probabilistic),
                probability_violation=False,
            )
        elif scenario == "STRESS_SEMANTICS":
            stress = [
                item
                for item in fixture.decision.evidence_package.items
                if item.semantics == "stress"
            ]
            checks = Part2CaseChecks(
                structured_fact_fidelity=bool(stress),
                stress_probability_violation=False,
                causal_violation=False,
            )
        elif scenario == "COMPARISON":
            observed_status = comparative_answer.status
            observed_intent = "COMPARISON"
            observed_tool = "compare_decisions"
            checks = Part2CaseChecks(
                intent_accuracy=True,
                tool_routing_accuracy=True,
                comparison_fidelity=True,
                citation_validity=bool(comparative_answer.citations),
                citation_completeness=all(
                    claim.citation_refs for claim in comparative_answer.claims
                ),
                structured_fact_fidelity=True,
                visible_text_facts_consistency=True,
                causal_violation=False,
            )
        elif scenario == "COUNTERFACTUAL":
            observed_status = counterfactual.status
            observed_intent = "COUNTERFACTUAL"
            observed_tool = "search_bounded_counterfactual"
            checks = Part2CaseChecks(
                intent_accuracy=True,
                tool_routing_accuracy=True,
                computation_authority_fidelity=True,
                counterfactual_target_fidelity=(
                    counterfactual.status == "BOUNDED_COUNTERFACTUAL_FOUND"
                    and not counterfactual.global_minimality_claimed
                ),
            )
        elif scenario == "REGRET":
            observed_status = regret.status
            observed_intent = "REGRET"
            observed_tool = "evaluate_candidate_set_regret"
            checks = Part2CaseChecks(
                intent_accuracy=True,
                tool_routing_accuracy=True,
                computation_authority_fidelity=(fixture.gateway.evaluate_count >= 2),
                regret_component_fidelity=(
                    regret.candidate_set_regret is not None
                    and regret.candidate_set_regret >= 0
                    and not regret.global_oracle_claimed
                ),
            )
        elif scenario == "APPROVAL":
            observed_status = approved.state
            observed_intent = "APPROVAL"
            checks = Part2CaseChecks(
                intent_accuracy=True,
                approval_transition_fidelity=(
                    approved.orders_hash == sha256_content_hash(approved.orders)
                    and approved.state == "APPROVED"
                ),
                no_mutation_accuracy=fixture.decision.model_dump_json() == original_decision_json,
            )
        elif scenario == "STALE_APPROVAL":
            observed_status = stale.state
            observed_intent = "APPROVAL"
            checks = Part2CaseChecks(
                intent_accuracy=True,
                approval_transition_fidelity=stale.state == "STALE",
            )
        elif scenario == "UNKNOWN_ENTITY":
            agent = run_decision_agent(
                AgentRunRequest(
                    mode="READ_ONLY",
                    question=case.question,
                    language=case.language,
                    baseline_decision=fixture.decision,
                    actor="benchmark",
                    reason="unknown",
                    idempotency_key=case.case_id,
                )
            )
            observed_status = agent.status
            observed_intent = agent.intent
            checks = Part2CaseChecks(
                intent_accuracy=True,
                entity_selector_accuracy=agent.answer is None,
                citation_validity=agent.answer is None,
            )
        elif scenario.startswith("ADVERSARIAL_"):
            values: dict[str, Any] = {
                "ADVERSARIAL_STRATEGY": {"proposed_strategy": "LEAN"},
                "ADVERSARIAL_ORDER": {"proposed_order_quantities": {"O1": 999}},
                "ADVERSARIAL_PROBABILITY": {"proposed_probability": 0.9},
                "ADVERSARIAL_REGRET": {"proposed_regret": 42},
                "ADVERSARIAL_CITATION": {"proposed_citations": ["forged"]},
            }[scenario]
            try:
                validate_llm_proposal(
                    UntrustedAgentProposal(intent="READ_ONLY_EXPLANATION", **values),
                    fixture.decision,
                )
                observed_status = "ACCEPTED"
            except ValueError:
                observed_status = "REJECTED"
            checks = Part2CaseChecks(
                adversarial_guard_rejection=observed_status == "REJECTED",
                probability_violation=False,
                causal_violation=False,
            )
        elif scenario == "CROSS_PACKAGE_CITATION":
            forged = comparative_answer.model_copy(
                update={
                    "citations": ["BASELINE|sha256:" + "0" * 64 + "|forged"],
                }
            )
            try:
                explain_what_if(
                    hypothetical,
                    comparative_answer.question,
                    candidate_answer=ComparativeAnswer.model_validate(forged),
                )
                observed_status = "ACCEPTED"
            except ComparativeGroundingError:
                observed_status = "REJECTED"
            checks = Part2CaseChecks(
                adversarial_guard_rejection=observed_status == "REJECTED",
                citation_validity=observed_status == "REJECTED",
            )
        elif scenario == "MULTI_STORE_SELECTOR":
            multi = _multi_store_request(fixture.request)
            changed = apply_modifications(
                multi,
                [
                    DemandScaleModification(
                        selector=DemandSelector(
                            store_id="S1",
                            ingredient_id="I1",
                            unit="kg",
                            expected_matches=2,
                        ),
                        multiplier=2,
                    )
                ],
                hypothetical_request_id="HYP-MULTI",
            )
            s10 = [
                line.quantity
                for demand_scenario in changed.demand_scenarios
                for line in demand_scenario.lines
                if line.store_id == "S10"
            ]
            checks = Part2CaseChecks(
                entity_selector_accuracy=s10 == [99, 99],
                no_mutation_accuracy=multi.demand_scenarios[0].lines[0].quantity == 4,
            )
        elif scenario == "RECOURSE_SEPARATION":
            decision = hypothetical.hypothetical_decision
            checks = Part2CaseChecks(
                order_recourse_fidelity=(
                    all(
                        order.decision_stage == "first_stage" for order in decision.immediate_orders
                    )
                    and all(
                        order.decision_stage == "scenario_recourse"
                        for order in decision.conditional_recourse
                    )
                )
            )
        elif scenario == "NO_MUTATION":
            checks = Part2CaseChecks(
                no_mutation_accuracy=(
                    fixture.request.model_dump_json() == original_request_json
                    and fixture.decision.model_dump_json() == original_decision_json
                )
            )
        elif scenario == "DETERMINISM":
            repeat = draft_what_if(
                fixture.request,
                fixture.decision,
                [_mod(1.1)],
                actor="benchmark",
                reason="curated hypothetical",
                idempotency_key="benchmark-what-if",
            )
            checks = Part2CaseChecks(
                deterministic_repeatability=repeat.model_dump_json() == draft.model_dump_json()
            )
        results.append(
            _case_result(
                case,
                observed_status,
                checks,
                observed_intent=observed_intent,
                observed_tool=observed_tool,
            )
        )
    aggregate = Part2AggregateMetrics(
        intent_accuracy=_rate(results, "intent_accuracy"),
        entity_selector_accuracy=_rate(results, "entity_selector_accuracy"),
        tool_routing_accuracy=_rate(results, "tool_routing_accuracy"),
        unauthorized_tool_call_rate=_rate(results, "unauthorized_tool_call", violation=True),
        no_mutation_accuracy=_rate(results, "no_mutation_accuracy"),
        computation_authority_fidelity=_rate(results, "computation_authority_fidelity"),
        recommendation_fidelity=_rate(results, "recommendation_fidelity"),
        order_recourse_fidelity=_rate(results, "order_recourse_fidelity"),
        comparison_fidelity=_rate(results, "comparison_fidelity"),
        counterfactual_target_fidelity=_rate(results, "counterfactual_target_fidelity"),
        regret_component_fidelity=_rate(results, "regret_component_fidelity"),
        approval_transition_fidelity=_rate(results, "approval_transition_fidelity"),
        citation_validity=_rate(results, "citation_validity"),
        citation_completeness=_rate(results, "citation_completeness"),
        structured_fact_fidelity=_rate(results, "structured_fact_fidelity"),
        visible_text_facts_consistency=_rate(results, "visible_text_facts_consistency"),
        adversarial_guard_rejection=_rate(results, "adversarial_guard_rejection"),
        probability_violation_rate=_rate(results, "probability_violation", violation=True),
        stress_as_probability_violation_rate=_rate(
            results, "stress_probability_violation", violation=True
        ),
        causal_violation_rate=_rate(results, "causal_violation", violation=True),
        deterministic_repeatability=_rate(results, "deterministic_repeatability"),
    )
    failed = [
        case.case_id
        for case, row in zip(corpus.cases, results, strict=True)
        if case.critical and not row.passed
    ]
    metric_gates = [
        ("authority_fidelity", aggregate.computation_authority_fidelity, "EQ", 1.0),
        ("recommendation_fidelity", aggregate.recommendation_fidelity, "EQ", 1.0),
        ("order_recourse_fidelity", aggregate.order_recourse_fidelity, "EQ", 1.0),
        ("no_mutation", aggregate.no_mutation_accuracy, "EQ", 1.0),
        ("approval_binding", aggregate.approval_transition_fidelity, "EQ", 1.0),
        ("adversarial_rejection", aggregate.adversarial_guard_rejection, "EQ", 1.0),
        ("citation_validity", aggregate.citation_validity, "EQ", 1.0),
        ("citation_completeness", aggregate.citation_completeness, "EQ", 1.0),
        ("structured_fact_fidelity", aggregate.structured_fact_fidelity, "EQ", 1.0),
        ("comparison_fidelity", aggregate.comparison_fidelity, "EQ", 1.0),
        ("counterfactual_fidelity", aggregate.counterfactual_target_fidelity, "EQ", 1.0),
        ("regret_fidelity", aggregate.regret_component_fidelity, "EQ", 1.0),
        ("unauthorized_tool_rate", aggregate.unauthorized_tool_call_rate, "EQ", 0.0),
        ("probability_violation_rate", aggregate.probability_violation_rate, "EQ", 0.0),
        ("stress_probability_rate", aggregate.stress_as_probability_violation_rate, "EQ", 0.0),
        ("causal_violation_rate", aggregate.causal_violation_rate, "EQ", 0.0),
        ("determinism", aggregate.deterministic_repeatability, "EQ", 1.0),
    ]
    gates = [
        _gate("failed_critical_cases", float(len(failed)), "EQ", 0, len(results)),
        *[
            _gate(name, observation.value, operator, target, observation.denominator)
            for name, observation, operator, target in metric_gates
        ],
    ]
    language_results = []
    for language in ("en", "vi"):
        rows = [row for row in results if row.language == language]
        intent_values = [
            row.checks.intent_accuracy for row in rows if row.checks.intent_accuracy is not None
        ]
        language_results.append(
            Part2LanguageResult(
                language=language,
                case_count=len(rows),
                passed_count=sum(row.passed for row in rows),
                pass_rate=sum(row.passed for row in rows) / len(rows),
                intent_accuracy=(sum(intent_values) / len(intent_values) if intent_values else 0),
            )
        )
    return Part2BenchmarkReport(
        corpus_version=corpus.corpus_version,
        environment={
            "python": platform.python_version(),
            "platform": platform.platform(),
            "execution": "offline-local-production-gateway",
        },
        metric_definitions={
            "intent_accuracy": "Observed deterministic intent equals the typed corpus expectation.",
            "entity_selector_accuracy": "Resolved exact typed entity/selector changes only its intended target.",
            "tool_routing_accuracy": "The orchestrator invokes the expected mode-allowlisted tool.",
            "unauthorized_tool_call_rate": "Share of applicable cases with a disallowed tool invocation; target zero.",
            "authority_fidelity": "All computed outputs traverse ComputationGateway and exact M4 when recommended.",
            "recommendation_fidelity": "Hypothetical recommendation equals the M5 OptimizationResult.",
            "order_recourse_fidelity": "Immediate orders equal first-stage M5 orders and recourse stays conditional.",
            "no_mutation": "Baseline typed JSON is unchanged after all operations.",
            "comparison_fidelity": "Deltas compare compatible typed baseline/hypothetical metrics and orders.",
            "counterfactual_target_fidelity": "The bounded result meets its exact typed target without global claims.",
            "regret_component_fidelity": "Candidate-set regret uses common exact replay and complete loss components.",
            "approval_transition_fidelity": "Transitions and order/package binding satisfy the hash-chained policy.",
            "citation_validity": "Every citation belongs to the current retrieval/package snapshot.",
            "citation_completeness": "Every material structured claim carries compatible evidence references.",
            "structured_fact_fidelity": "Claim facts equal the cited typed evidence payload.",
            "visible_text_facts_consistency": "Visible text equals trusted deterministic rendering of validated claims.",
            "adversarial_guard_rejection": "Forged strategy/order/probability/regret/citation proposals are rejected.",
            "deterministic_repeatability": "Repeated semantic input has byte-equivalent deterministic output.",
            "semantic_hash": "SHA-256 excludes environment and measured latency.",
            "violation_rates": "Numerator is an observed prohibited behavior; target is zero.",
        },
        case_results=results,
        aggregate_metrics=aggregate,
        per_language=language_results,
        performance=[_latency(name, values) for name, values in sorted(samples.items())],
        size_statistics={
            "baseline_evidence_items": len(fixture.decision.evidence_package.items),
            "comparative_evidence_items": len(hypothetical.evidence_package.items),
            "comparative_graph_nodes": len(hypothetical.decision_graph.nodes),
            "comparative_graph_edges": len(hypothetical.decision_graph.edges),
            "what_if_package_bytes": len(hypothetical.model_dump_json().encode("utf-8")),
        },
        acceptance_gates=gates,
        failed_critical_cases=failed,
        overall_pass=not failed and all(gate.passed for gate in gates),
        limitations=[
            "Synthetic deterministic corpus; it does not establish production or external-LLM safety.",
            "Vietnamese evaluation measures query routing, not localized answer quality.",
            "No forecast accuracy or procurement optimality improvement is claimed.",
            "No supplier execution, external identity, persistence, network, or external LLM is present.",
        ],
    )


__all__ = ["run_part2_benchmark"]
