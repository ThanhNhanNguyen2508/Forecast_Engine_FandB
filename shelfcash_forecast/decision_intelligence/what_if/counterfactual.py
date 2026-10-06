from __future__ import annotations

import math

from shelfcash_forecast.decision_intelligence.computation_gateway import ComputationGateway
from shelfcash_forecast.decision_intelligence.contracts import FinalDecisionPackage
from shelfcash_forecast.decision_intelligence.integrity import sha256_content_hash
from shelfcash_forecast.decision_intelligence.what_if.contracts import (
    CounterfactualRun,
    CounterfactualSearchRequest,
    CounterfactualSearchResult,
    CounterfactualTarget,
    WhatIfRequest,
)
from shelfcash_forecast.decision_intelligence.what_if.mutations import normalize_modifications
from shelfcash_forecast.decision_intelligence.what_if.service import run_what_if
from shelfcash_forecast.optimization.contracts import OptimizationRequest


def _compare(value: float, operator: str, threshold: float) -> bool:
    return {
        "GT": value > threshold,
        "GE": value >= threshold,
        "LT": value < threshold,
        "LE": value <= threshold,
        "EQ": math.isclose(value, threshold, rel_tol=1e-9, abs_tol=1e-9),
    }[operator]


def _target_met(package, target: CounterfactualTarget) -> bool:
    decision = package.hypothetical_decision
    if target.target_type == "STRATEGY":
        return decision.recommended_strategy == target.strategy
    if target.target_type == "DECISION_STATUS":
        return decision.decision_status == target.decision_status
    if target.target_type == "CRITIC_PASS":
        candidate = decision.recommended_plan_summary
        observed = bool(candidate and candidate.critic_passed)
        return observed is target.critic_passed
    if target.target_type == "IMMEDIATE_ORDER_THRESHOLD":
        quantity = sum(
            order.order_quantity
            for order in decision.immediate_orders
            if order.offer_id == target.offer_id
        )
        assert target.threshold is not None
        return _compare(quantity, target.operator, target.threshold)
    metric = next(
        (
            row
            for row in package.comparison.metric_comparisons
            if row.name == target.metric_name and row.hypothetical.value is not None
        ),
        None,
    )
    if metric is None or not isinstance(metric.hypothetical.value, (int, float)):
        return False
    assert target.threshold is not None
    return _compare(float(metric.hypothetical.value), target.operator, target.threshold)


def search_counterfactuals(
    baseline_request: OptimizationRequest,
    baseline_decision: FinalDecisionPackage,
    request: CounterfactualSearchRequest,
    *,
    gateway: ComputationGateway | None = None,
) -> CounterfactualSearchResult:
    if not request.confirmed:
        raise ValueError("M6_COUNTERFACTUAL_CONFIRMATION_REQUIRED")
    runs: list[CounterfactualRun] = []
    cache: dict[str, tuple[bool, str]] = {}
    found: int | None = None
    for index, modifications in enumerate(request.candidate_modifications):
        modification_hash = sha256_content_hash(normalize_modifications(modifications))
        if index >= request.maximum_run_count:
            runs.append(
                CounterfactualRun(
                    candidate_index=index,
                    modification_hash=modification_hash,
                    status="NOT_RUN_BUDGET",
                )
            )
            continue
        if modification_hash in cache:
            met, package_hash = cache[modification_hash]
        else:
            what_if_request = WhatIfRequest(
                baseline_request_id=request.baseline_request_id,
                baseline_decision_hash=request.baseline_decision_hash,
                baseline_request_hash=request.baseline_request_hash,
                idempotency_key=f"{request.idempotency_key}:{index}",
                modifications=normalize_modifications(modifications),
                actor=request.actor,
                reason=f"Bounded counterfactual candidate {index}",
                execution_mode="EXECUTE_HYPOTHETICAL",
                confirmed=True,
            )
            package = run_what_if(
                baseline_request,
                baseline_decision,
                what_if_request,
                gateway=gateway,
            )
            met = _target_met(package, request.target)
            package_hash = package.package_hash
            cache[modification_hash] = (met, package_hash)
        runs.append(
            CounterfactualRun(
                candidate_index=index,
                modification_hash=modification_hash,
                status="TARGET_MET" if met else "TARGET_NOT_MET",
                what_if_package_hash=package_hash,
            )
        )
        if met:
            found = index
            break
    budget_exhausted = (
        found is None and len(request.candidate_modifications) > request.maximum_run_count
    )
    return CounterfactualSearchResult(
        search_id=request.search_id,
        status=(
            "BOUNDED_COUNTERFACTUAL_FOUND"
            if found is not None
            else "RUN_BUDGET_EXHAUSTED"
            if budget_exhausted
            else "NOT_FOUND_IN_BOUNDED_SPACE"
        ),
        target=request.target,
        found_candidate_index=found,
        runs=runs,
        candidate_count=len(request.candidate_modifications),
        maximum_run_count=request.maximum_run_count,
        limitations=[
            "Search is finite and ordered; no global minimality or causal intervention is claimed.",
            "Every executed candidate is re-optimized and exactly validated through the gateway.",
        ],
    )


__all__ = ["search_counterfactuals"]
