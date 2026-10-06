from __future__ import annotations

from shelfcash_forecast.decision_intelligence.computation_gateway import (
    ComputationGateway,
    M5ComputationGateway,
)
from shelfcash_forecast.decision_intelligence.contracts import FinalDecisionPackage
from shelfcash_forecast.decision_intelligence.integrity import sha256_content_hash
from shelfcash_forecast.decision_intelligence.service import build_final_decision_package
from shelfcash_forecast.decision_intelligence.what_if.comparison import (
    compare_decisions,
    decision_snapshot_hash,
)
from shelfcash_forecast.decision_intelligence.what_if.contracts import (
    WhatIfAuditEvent,
    WhatIfDecisionPackage,
    WhatIfDraft,
    WhatIfModification,
    WhatIfRequest,
)
from shelfcash_forecast.decision_intelligence.what_if.evidence import (
    build_comparative_evidence,
    build_comparative_graph,
)
from shelfcash_forecast.decision_intelligence.what_if.mutations import (
    apply_modifications,
    normalize_modifications,
)
from shelfcash_forecast.optimization.contracts import OptimizationRequest


class WhatIfError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


def optimization_request_hash(request: OptimizationRequest) -> str:
    return sha256_content_hash(request)


def draft_what_if(
    baseline_request: OptimizationRequest,
    baseline_decision: FinalDecisionPackage,
    modifications: list[WhatIfModification],
    *,
    actor: str,
    reason: str,
    idempotency_key: str,
) -> WhatIfDraft:
    if baseline_request.request_id != baseline_decision.request_id:
        raise WhatIfError("M6_WHAT_IF_FOREIGN_BASELINE", "request and decision IDs differ")
    normalized = normalize_modifications(modifications)
    request = WhatIfRequest(
        baseline_request_id=baseline_request.request_id,
        baseline_decision_hash=decision_snapshot_hash(baseline_decision),
        baseline_request_hash=optimization_request_hash(baseline_request),
        idempotency_key=idempotency_key,
        modifications=normalized,
        actor=actor,
        reason=reason,
        execution_mode="DRAFT_ONLY",
        confirmed=False,
    )
    return WhatIfDraft(
        status="DRAFT_READY",
        request=request,
        normalized_modifications=normalized,
        confirmation_required=True,
        warnings=[
            "This draft has not invoked M4 or M5.",
            "Execution requires a separately validated explicit confirmation.",
        ],
    )


def confirm_what_if(draft: WhatIfDraft) -> WhatIfRequest:
    if draft.status != "DRAFT_READY" or draft.request is None:
        raise WhatIfError("M6_WHAT_IF_DRAFT_NOT_EXECUTABLE", "draft is incomplete")
    values = draft.request.model_dump(mode="python")
    values.update(
        {
            "what_if_id": "",
            "content_hash": "",
            "execution_mode": "EXECUTE_HYPOTHETICAL",
            "confirmed": True,
        }
    )
    return WhatIfRequest.model_validate(values)


def _validate_binding(
    baseline_request: OptimizationRequest,
    baseline_decision: FinalDecisionPackage,
    request: WhatIfRequest,
) -> None:
    if request.baseline_request_id != baseline_request.request_id:
        raise WhatIfError("M6_WHAT_IF_BASELINE_REQUEST_ID_MISMATCH", "foreign request")
    if baseline_decision.request_id != baseline_request.request_id:
        raise WhatIfError("M6_WHAT_IF_FOREIGN_BASELINE", "foreign decision package")
    if request.baseline_request_hash != optimization_request_hash(baseline_request):
        raise WhatIfError("M6_WHAT_IF_BASELINE_REQUEST_HASH_MISMATCH", "request changed")
    if request.baseline_decision_hash != decision_snapshot_hash(baseline_decision):
        raise WhatIfError("M6_WHAT_IF_BASELINE_DECISION_HASH_MISMATCH", "package changed")


def _validate_m5_authority(package: WhatIfDecisionPackage) -> None:
    result = package.optimization_result
    recommended = result.recommended_strategy
    if recommended is None:
        if package.hypothetical_decision.immediate_orders:
            raise WhatIfError("M6_WHAT_IF_NO_VALID_PLAN_HAS_ORDERS", "fallback orders forbidden")
        return
    evaluation = result.evaluations.get(recommended)
    if evaluation is None:
        raise WhatIfError("M6_WHAT_IF_RECOMMENDATION_MISSING", "candidate missing")
    if not evaluation.critic.passed or not evaluation.plan.completed:
        raise WhatIfError("M6_WHAT_IF_RECOMMENDATION_NOT_VALIDATED", "critic did not pass")
    if evaluation.simulation is None:
        raise WhatIfError("M6_WHAT_IF_EXACT_M4_MISSING", "exact simulation is required")


def run_what_if(
    baseline_request: OptimizationRequest,
    baseline_decision: FinalDecisionPackage,
    request: WhatIfRequest,
    *,
    gateway: ComputationGateway | None = None,
) -> WhatIfDecisionPackage:
    """Execute one confirmed hypothetical through the existing M5 authority path."""

    _validate_binding(baseline_request, baseline_decision, request)
    if request.execution_mode != "EXECUTE_HYPOTHETICAL" or not request.confirmed:
        raise WhatIfError("M6_WHAT_IF_CONFIRMATION_REQUIRED", "execution is not confirmed")
    normalized = normalize_modifications(request.modifications)
    hypothetical_request_id = f"HYP-{request.what_if_id.removeprefix('whatif-')}"
    modified_request = apply_modifications(
        baseline_request,
        normalized,
        hypothetical_request_id=hypothetical_request_id,
    )
    modified_request_hash = optimization_request_hash(modified_request)
    selected_gateway = gateway or M5ComputationGateway()
    result = selected_gateway.optimize(modified_request)
    hypothetical_decision = build_final_decision_package(modified_request, result)
    comparison = compare_decisions(baseline_decision, hypothetical_decision)
    evidence = build_comparative_evidence(
        baseline_decision,
        hypothetical_decision,
        comparison,
        normalized,
    )
    graph = build_comparative_graph(evidence, comparison)
    package = WhatIfDecisionPackage(
        what_if_id=request.what_if_id,
        baseline_request_id=baseline_request.request_id,
        baseline_decision_hash=request.baseline_decision_hash,
        baseline_request_hash=request.baseline_request_hash,
        normalized_modifications=normalized,
        modified_request=modified_request,
        modified_request_hash=modified_request_hash,
        optimization_result=result,
        hypothetical_decision=hypothetical_decision,
        comparison=comparison,
        evidence_package=evidence,
        decision_graph=graph,
        warnings=[
            "Optional M1-M3 artifacts were not carried forward as hypothetical truth.",
        ],
        limitations=[
            "This is a hypothetical model-derived run, not an observed or causal effect.",
            "No forecast or recipe/BOM recomputation was performed.",
            "No supplier order was executed.",
            "NOT_CARRIED_FORWARD_DUE_TO_HYPOTHETICAL_MUTATION: M1-M3 explanation artifacts.",
        ],
        authority_statuses={
            "baseline_binding": "VERIFIED",
            "mutation_validation": "VERIFIED",
            "m5_optimization": "COMPUTED_BY_GATEWAY",
            "exact_m4": (
                "VERIFIED"
                if result.recommended_strategy is not None
                and result.evaluations[result.recommended_strategy].simulation is not None
                else "NOT_APPLICABLE_NO_VALID_PLAN"
            ),
            "supplier_execution": "NOT_PERFORMED",
        },
        audit_trace=[
            WhatIfAuditEvent(
                sequence=0,
                action="VALIDATE_BASELINE_BINDING",
                reason_code="HASH_AND_REQUEST_ID_MATCH",
                input_hash=request.content_hash,
                output_hash=request.baseline_decision_hash,
                status="VERIFIED",
            ),
            WhatIfAuditEvent(
                sequence=1,
                action="APPLY_TYPED_MUTATIONS",
                reason_code="ALLOWLISTED_MUTATIONS_ONLY",
                input_hash=request.content_hash,
                output_hash=modified_request_hash,
                status="VERIFIED",
            ),
            WhatIfAuditEvent(
                sequence=2,
                action="CALL_M5_COMPUTATION_GATEWAY",
                reason_code="EXPLICITLY_CONFIRMED_HYPOTHETICAL",
                input_hash=modified_request_hash,
                output_hash=sha256_content_hash(result),
                status=result.status,
            ),
        ],
        provenance={
            "service": "shelfcash_m6_part2_controlled_what_if_v1",
            "hypothetical": True,
            "observed": False,
            "causal_claim": False,
            "m5_decision_authority_preserved": True,
            "computation_gateway_calls": 1,
        },
    )
    _validate_m5_authority(package)
    return package


__all__ = [
    "WhatIfError",
    "confirm_what_if",
    "draft_what_if",
    "optimization_request_hash",
    "run_what_if",
]
