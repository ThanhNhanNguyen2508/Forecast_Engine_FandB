from __future__ import annotations

import re

from shelfcash_forecast.decision_intelligence.agents.contracts import (
    AgentRunRequest,
    AgentRunResult,
    AuthorityAssessment,
)
from shelfcash_forecast.decision_intelligence.agents.intents import (
    IntentNormalizeAgent,
    ScenarioWhatIfAgent,
)
from shelfcash_forecast.decision_intelligence.agents.telemetry import trace_event
from shelfcash_forecast.decision_intelligence.agents.tools import (
    AgentToolError,
    DecisionToolRegistry,
)
from shelfcash_forecast.decision_intelligence.approval.workflow import (
    Clock,
    inspect_approval_case,
    transition_approval_case,
)
from shelfcash_forecast.decision_intelligence.computation_gateway import ComputationGateway
from shelfcash_forecast.decision_intelligence.contracts import (
    DecisionAnswer,
    FinalDecisionPackage,
)
from shelfcash_forecast.decision_intelligence.integrity import sha256_content_hash
from shelfcash_forecast.decision_intelligence.regret.service import evaluate_decision_regret
from shelfcash_forecast.decision_intelligence.retrieval import unknown_explicit_entities
from shelfcash_forecast.decision_intelligence.service import explain_decision
from shelfcash_forecast.decision_intelligence.what_if.counterfactual import (
    search_counterfactuals,
)
from shelfcash_forecast.decision_intelligence.what_if.grounding import explain_what_if
from shelfcash_forecast.decision_intelligence.what_if.service import (
    draft_what_if,
    run_what_if,
)


class DecisionContextResolverAgent:
    def resolve(self, question: str, decision: FinalDecisionPackage) -> list[str]:
        unknown = unknown_explicit_entities(question, decision.evidence_package)
        aliases = {
            "store": "store_id",
            "ingredient": "ingredient_id",
            "product": "product_id",
            "supplier": "supplier_id",
            "scenario": "scenario_id",
        }
        for alias, key in aliases.items():
            match = re.search(
                rf"\b{alias}\s*:\s*([\w.-]+)",
                question,
                flags=re.IGNORECASE | re.UNICODE,
            )
            if match is None:
                continue
            candidate = match.group(1).rstrip(".,?!:;")
            known = {
                item.entities[key].casefold()
                for item in decision.evidence_package.items
                if key in item.entities
            }
            if candidate.casefold() not in known:
                unknown[key] = candidate
        return sorted(f"UNKNOWN_{kind.upper()}:{value}" for kind, value in unknown.items())


class RetrieverAgent:
    def explain(self, decision: FinalDecisionPackage, question: str) -> DecisionAnswer:
        return explain_decision(decision, question)


class AuthoritySemanticsAgent:
    def assess(self, mode: str) -> AuthorityAssessment:
        if mode in {"WHAT_IF_EXECUTE", "COUNTERFACTUAL", "REGRET"}:
            return AuthorityAssessment(
                authority_layer="M5",
                semantics="hypothetical",
                may_compute=True,
                reason_code="COMPUTATION_GATEWAY_REQUIRED",
            )
        return AuthorityAssessment(
            authority_layer="M6",
            semantics="deterministic",
            may_compute=False,
            reason_code="READ_ONLY_OR_DRAFT",
        )


class EvidenceCriticAgent:
    def validate_answer(self, answer: DecisionAnswer, decision: FinalDecisionPackage) -> bool:
        known = {item.evidence_id for item in decision.evidence_package.items}
        return not (set(answer.citations) - known) and all(
            set(claim.evidence_ids) <= known for claim in answer.claims
        )


class GroundedGeneratorAgent:
    def generate(self, decision: FinalDecisionPackage, question: str) -> DecisionAnswer:
        return explain_decision(decision, question)


class CitationGuardrailAgent(EvidenceCriticAgent):
    pass


class OperationalFreshnessAgent:
    def check(self, decision: FinalDecisionPackage, expected_hash: str) -> bool:
        return sha256_content_hash(decision) == expected_hash


class ApprovalGateAgent:
    def inspect(self, *args, **kwargs):
        return inspect_approval_case(*args, **kwargs)

    def transition(self, *args, **kwargs):
        return transition_approval_case(*args, **kwargs)


class DecisionOrchestrator:
    """Deterministic bounded routing with per-mode tool allowlists."""

    def __init__(
        self,
        *,
        gateway: ComputationGateway | None = None,
        clock: Clock | None = None,
        registry: DecisionToolRegistry | None = None,
    ) -> None:
        self.gateway = gateway
        self.clock = clock
        self.registry = registry or DecisionToolRegistry()
        self._register_defaults()

    def _register_defaults(self) -> None:
        defaults = {
            "explain_current_decision": explain_decision,
            "draft_what_if": draft_what_if,
            "execute_confirmed_what_if": run_what_if,
            "compare_decisions": explain_what_if,
            "search_bounded_counterfactual": search_counterfactuals,
            "evaluate_candidate_set_regret": evaluate_decision_regret,
            "inspect_approval_case": inspect_approval_case,
            "transition_approval_case": transition_approval_case,
        }
        for name, function in defaults.items():
            try:
                self.registry.register(name, function)
            except AgentToolError as error:
                if "DUPLICATE_TOOL" not in str(error):
                    raise

    def run(self, request: AgentRunRequest) -> AgentRunResult:
        intent = IntentNormalizeAgent().normalize(request.question)
        trace = [
            trace_event(
                0,
                "IntentNormalizeAgent",
                "NORMALIZE_INTENT",
                f"INTENT_{intent}",
                request.question,
                intent,
                "COMPLETED",
            )
        ]
        expected_by_mode = {
            "READ_ONLY": {"READ_ONLY_EXPLANATION"},
            "WHAT_IF_DRAFT": {"WHAT_IF_DRAFT", "READ_ONLY_EXPLANATION"},
            "WHAT_IF_EXECUTE": {"WHAT_IF_DRAFT", "WHAT_IF_EXECUTION"},
            "COMPARISON": {"COMPARISON", "WHAT_IF_DRAFT"},
            "COUNTERFACTUAL": {"COUNTERFACTUAL"},
            "REGRET": {"REGRET"},
            "APPROVAL": {"APPROVAL"},
        }
        if intent not in expected_by_mode[request.mode]:
            return AgentRunResult(
                run_id=request.run_id,
                status="FAILED_VALIDATION",
                intent=intent,
                trace=trace,
                tool_calls=[],
                error_codes=["M6_AGENT_MODE_INTENT_MISMATCH"],
                limitations=["Mode cannot be escalated by natural-language intent."],
            )
        unknown = DecisionContextResolverAgent().resolve(
            request.question,
            request.baseline_decision,
        )
        if unknown:
            return AgentRunResult(
                run_id=request.run_id,
                status="INSUFFICIENT_EVIDENCE",
                intent=intent,
                trace=trace,
                tool_calls=[],
                error_codes=unknown,
                limitations=["Unknown entities are not substituted with known entities."],
            )
        tool_calls: list[str] = []

        def call(name: str, **kwargs):
            if len(tool_calls) >= request.maximum_tool_calls:
                raise AgentToolError("M6_AGENT_TOOL_BUDGET_EXHAUSTED")
            if name in tool_calls:
                raise AgentToolError("M6_AGENT_RECURSIVE_OR_DUPLICATE_TOOL_CALL")
            result = self.registry.call(request.mode, name, **kwargs)
            tool_calls.append(name)
            trace.append(
                trace_event(
                    len(trace),
                    "DecisionToolRegistry",
                    name,
                    "MODE_ALLOWLIST_VERIFIED",
                    kwargs,
                    result,
                    "COMPLETED",
                )
            )
            return result

        try:
            if request.mode == "READ_ONLY":
                answer = call(
                    "explain_current_decision",
                    decision=request.baseline_decision,
                    question=request.question,
                )
                return AgentRunResult(
                    run_id=request.run_id,
                    status="COMPLETED",
                    intent=intent,
                    answer=answer,
                    trace=trace,
                    tool_calls=tool_calls,
                )
            if request.mode == "WHAT_IF_DRAFT":
                if request.baseline_request is None:
                    raise ValueError("M6_AGENT_BASELINE_REQUEST_REQUIRED")
                if request.typed_modifications:
                    draft = call(
                        "draft_what_if",
                        baseline_request=request.baseline_request,
                        baseline_decision=request.baseline_decision,
                        modifications=request.typed_modifications,
                        actor=request.actor,
                        reason=request.reason,
                        idempotency_key=request.idempotency_key,
                    )
                else:
                    draft = ScenarioWhatIfAgent().draft(
                        request.question,
                        request.baseline_request,
                        request.baseline_decision,
                        actor=request.actor,
                        reason=request.reason,
                        idempotency_key=request.idempotency_key,
                    )
                    tool_calls.append("draft_what_if")
                    trace.append(
                        trace_event(
                            len(trace),
                            "ScenarioWhatIfAgent",
                            "draft_what_if",
                            "NATURAL_LANGUAGE_DRAFT_ONLY",
                            request.question,
                            draft,
                            draft.status,
                        )
                    )
                return AgentRunResult(
                    run_id=request.run_id,
                    status=(
                        "COMPLETED"
                        if draft.status == "DRAFT_READY"
                        else "NOT_SUPPORTED_AT_CURRENT_AUTHORITY_BOUNDARY"
                        if draft.status == "NOT_SUPPORTED"
                        else "NEEDS_CLARIFICATION"
                    ),
                    intent="WHAT_IF_DRAFT",
                    what_if_draft=draft,
                    trace=trace,
                    tool_calls=tool_calls,
                )
            if request.mode == "WHAT_IF_EXECUTE":
                if request.baseline_request is None or request.what_if_request is None:
                    raise ValueError("M6_AGENT_CONFIRMED_WHAT_IF_REQUIRED")
                package = call(
                    "execute_confirmed_what_if",
                    baseline_request=request.baseline_request,
                    baseline_decision=request.baseline_decision,
                    request=request.what_if_request,
                    gateway=self.gateway,
                )
                return AgentRunResult(
                    run_id=request.run_id,
                    status="COMPLETED",
                    intent="WHAT_IF_EXECUTION",
                    what_if_package=package,
                    trace=trace,
                    tool_calls=tool_calls,
                )
            if request.mode == "COMPARISON":
                if request.what_if_package is None:
                    raise ValueError("M6_AGENT_WHAT_IF_PACKAGE_REQUIRED")
                answer = call(
                    "compare_decisions",
                    package=request.what_if_package,
                    question=request.question,
                )
                return AgentRunResult(
                    run_id=request.run_id,
                    status="COMPLETED",
                    intent="COMPARISON",
                    answer=answer,
                    trace=trace,
                    tool_calls=tool_calls,
                )
            if request.mode == "COUNTERFACTUAL":
                if request.baseline_request is None or request.counterfactual_request is None:
                    raise ValueError("M6_AGENT_COUNTERFACTUAL_REQUEST_REQUIRED")
                result = call(
                    "search_bounded_counterfactual",
                    baseline_request=request.baseline_request,
                    baseline_decision=request.baseline_decision,
                    request=request.counterfactual_request,
                    gateway=self.gateway,
                )
                return AgentRunResult(
                    run_id=request.run_id,
                    status="COMPLETED",
                    intent="COUNTERFACTUAL",
                    result_payload=result,
                    trace=trace,
                    tool_calls=tool_calls,
                )
            if request.mode == "REGRET":
                if request.regret_request is None:
                    raise ValueError("M6_AGENT_REGRET_REQUEST_REQUIRED")
                result = call(
                    "evaluate_candidate_set_regret",
                    baseline_decision=request.baseline_decision,
                    request=request.regret_request,
                    gateway=self.gateway,
                )
                return AgentRunResult(
                    run_id=request.run_id,
                    status="COMPLETED",
                    intent="REGRET",
                    result_payload=result,
                    trace=trace,
                    tool_calls=tool_calls,
                )
            if request.approval_case is None:
                raise ValueError("M6_AGENT_APPROVAL_CASE_REQUIRED")
            if request.approval_target_state is None:
                result = call(
                    "inspect_approval_case",
                    case=request.approval_case,
                    decision=request.baseline_decision,
                    clock=self.clock,
                )
            else:
                result = call(
                    "transition_approval_case",
                    case=request.approval_case,
                    target_state=request.approval_target_state,
                    actor=request.actor,
                    role=request.approval_role or "",
                    reason=request.reason,
                    idempotency_key=request.idempotency_key,
                    current_decision_hash=sha256_content_hash(request.baseline_decision),
                    clock=self.clock,
                )
            return AgentRunResult(
                run_id=request.run_id,
                status="COMPLETED",
                intent="APPROVAL",
                result_payload=result,
                trace=trace,
                tool_calls=tool_calls,
                limitations=["Approval never invokes supplier execution."],
            )
        except AgentToolError as error:
            code = str(error).split(":", 1)[0]
            return AgentRunResult(
                run_id=request.run_id,
                status=("TOOL_BUDGET_EXHAUSTED" if "BUDGET" in code else "UNAUTHORIZED_TOOL_CALL"),
                intent=intent,
                trace=trace,
                tool_calls=tool_calls,
                error_codes=[code],
            )
        except ValueError as error:
            return AgentRunResult(
                run_id=request.run_id,
                status="FAILED_VALIDATION",
                intent=intent,
                trace=trace,
                tool_calls=tool_calls,
                error_codes=[str(error).split(":", 1)[0]],
            )


def run_decision_agent(
    request: AgentRunRequest,
    *,
    gateway: ComputationGateway | None = None,
    clock: Clock | None = None,
) -> AgentRunResult:
    return DecisionOrchestrator(gateway=gateway, clock=clock).run(request)


__all__ = [
    "ApprovalGateAgent",
    "AuthoritySemanticsAgent",
    "CitationGuardrailAgent",
    "DecisionContextResolverAgent",
    "DecisionOrchestrator",
    "EvidenceCriticAgent",
    "GroundedGeneratorAgent",
    "OperationalFreshnessAgent",
    "RetrieverAgent",
    "run_decision_agent",
]
