from __future__ import annotations

import re

from shelfcash_forecast.decision_intelligence.what_if.contracts import (
    ComparativeAnswer,
    ComparativeClaim,
    ComparativeEvidenceItem,
    WhatIfDecisionPackage,
)
from shelfcash_forecast.decision_intelligence.what_if.evidence import evidence_ref


class ComparativeGroundingError(ValueError):
    pass


def _intent(question: str) -> str:
    normalized = question.casefold()
    if any(term in normalized for term in ("order", "mua", "đặt hàng", "recourse")):
        return "changed_orders"
    if any(term in normalized for term in ("strategy", "recommend", "chiến lược", "được chọn")):
        return "changed_recommendation"
    if any(term in normalized for term in ("assumption", "thay đổi", "giả định")):
        return "changed_assumptions"
    if any(term in normalized for term in ("risk", "stockout", "waste", "expiry", "rủi ro")):
        return "changed_risk"
    return "what_if_summary"


def retrieve_comparative_evidence(
    package: WhatIfDecisionPackage,
    question: str,
    *,
    limit: int = 12,
) -> list[ComparativeEvidenceItem]:
    intent = _intent(question)
    priorities = {
        "changed_orders": ("first_stage_order_delta", "scenario_recourse_order_delta"),
        "changed_recommendation": ("recommendation_delta", "readiness_delta"),
        "changed_assumptions": ("recorded_modification",),
        "changed_risk": ("metric_delta", "readiness_delta"),
        "what_if_summary": (
            "recorded_modification",
            "recommendation_delta",
            "first_stage_order_delta",
            "readiness_delta",
        ),
    }[intent]
    tokens = set(re.findall(r"[\w.-]+", question.casefold()))

    def score(item: ComparativeEvidenceItem) -> tuple[int, int, str]:
        type_score = (
            len(priorities) - priorities.index(item.evidence_type)
            if (item.evidence_type in priorities)
            else 0
        )
        lexical = len(tokens & set(re.findall(r"[\w.-]+", item.text.casefold())))
        return (-type_score, -lexical, evidence_ref(item))

    return sorted(package.evidence_package.items, key=score)[:limit]


def _trusted_text(claims: list[ComparativeClaim]) -> str:
    if not claims:
        return "Insufficient comparative evidence for this hypothetical question."
    lines = ["In the recorded hypothetical run with the stated assumptions:"]
    for claim in claims:
        facts = claim.facts
        if claim.claim_type == "recorded_modification":
            lines.append(f"- Recorded modification: {facts['modification_type']}.")
        elif claim.claim_type == "recommendation_delta":
            lines.append(
                "- M5 recommendation: "
                f"{facts.get('baseline_strategy') or 'NONE'} to "
                f"{facts.get('hypothetical_strategy') or 'NONE'} "
                f"(changed={str(facts['changed']).lower()})."
            )
        elif claim.claim_type in {"first_stage_order_delta", "scenario_recourse_order_delta"}:
            label = (
                "Immediate order"
                if claim.claim_type.startswith("first")
                else "Conditional recourse"
            )
            lines.append(
                f"- {label} {facts['offer_id']}: {facts['baseline_quantity']} to "
                f"{facts['hypothetical_quantity']} {facts['unit']} "
                f"(delta {facts['quantity_delta']})."
            )
        elif claim.claim_type == "metric_delta":
            lines.append(
                f"- Typed metric {facts['name']}: comparison status {facts['status']}"
                + (f", delta {facts['delta']}." if facts.get("delta") is not None else ".")
            )
        elif claim.claim_type == "readiness_delta":
            lines.append(
                f"- Evidence readiness: {facts['baseline_readiness']} to "
                f"{facts['hypothetical_readiness']}."
            )
        else:
            raise ComparativeGroundingError("M6_COMPARATIVE_UNKNOWN_CLAIM_TYPE")
    lines.append("These are model-derived hypothetical results, not observed or causal effects.")
    return "\n".join(lines)


class ComparativeGroundingGuard:
    def validate(
        self,
        answer: ComparativeAnswer,
        package: WhatIfDecisionPackage,
        retrieved: list[ComparativeEvidenceItem],
    ) -> ComparativeAnswer:
        known = {evidence_ref(item): item for item in package.evidence_package.items}
        snapshot = {evidence_ref(item) for item in retrieved}
        if set(answer.citations) - snapshot:
            raise ComparativeGroundingError("M6_COMPARATIVE_CITATION_OUTSIDE_RETRIEVAL_SNAPSHOT")
        claim_citations: set[str] = set()
        for claim in answer.claims:
            if claim.causal:
                raise ComparativeGroundingError("M6_COMPARATIVE_CAUSAL_CLAIM_FORBIDDEN")
            cited = [known.get(ref) for ref in claim.citation_refs]
            if any(item is None for item in cited) or set(claim.citation_refs) - snapshot:
                raise ComparativeGroundingError("M6_COMPARATIVE_UNKNOWN_OR_CROSS_PACKAGE_CITATION")
            compatible = [item for item in cited if item is not None]
            if not any(item.evidence_type == claim.claim_type for item in compatible):
                raise ComparativeGroundingError("M6_COMPARATIVE_INCOMPATIBLE_EVIDENCE_TYPE")
            if claim.uses_probability_language and not all(
                "probabilistic" in item.semantics for item in compatible
            ):
                raise ComparativeGroundingError("M6_COMPARATIVE_PROBABILITY_SEMANTICS_VIOLATION")
            matching = [item for item in compatible if item.evidence_type == claim.claim_type]
            if not any(
                all(item.payload.get(key) == value for key, value in claim.facts.items())
                for item in matching
            ):
                raise ComparativeGroundingError("M6_COMPARATIVE_FACT_EVIDENCE_MISMATCH")
            claim_citations.update(claim.citation_refs)
        if set(answer.citations) != claim_citations:
            raise ComparativeGroundingError("M6_COMPARATIVE_CITATION_COMPLETENESS_FAILURE")
        expected_text = _trusted_text(answer.claims)
        if answer.answer_text != expected_text:
            raise ComparativeGroundingError("M6_COMPARATIVE_VISIBLE_TEXT_MISMATCH")
        lower = answer.answer_text.casefold()
        if any(term in lower for term in ("caused by", "due to", "dẫn đến", "bởi vì")):
            raise ComparativeGroundingError("M6_COMPARATIVE_CAUSAL_LANGUAGE_FORBIDDEN")
        return answer


def explain_what_if(
    package: WhatIfDecisionPackage,
    question: str,
    *,
    candidate_answer: ComparativeAnswer | None = None,
    limit: int = 12,
) -> ComparativeAnswer:
    retrieved = retrieve_comparative_evidence(package, question, limit=limit)
    if candidate_answer is not None:
        return ComparativeGroundingGuard().validate(candidate_answer, package, retrieved)
    intent = _intent(question)
    allowed_types = {
        "changed_orders": {"first_stage_order_delta", "scenario_recourse_order_delta"},
        "changed_recommendation": {"recommendation_delta", "readiness_delta"},
        "changed_assumptions": {"recorded_modification"},
        "changed_risk": {"metric_delta", "readiness_delta"},
        "what_if_summary": {
            "recorded_modification",
            "recommendation_delta",
            "first_stage_order_delta",
            "readiness_delta",
        },
    }[intent]
    claims = [
        ComparativeClaim(
            claim_type=item.evidence_type,
            facts=item.payload,
            citation_refs=[evidence_ref(item)],
            uses_probability_language=False,
            causal=False,
        )
        for item in retrieved
        if item.package_role == "DELTA" and item.evidence_type in allowed_types
    ]
    citations = sorted({ref for claim in claims for ref in claim.citation_refs})
    answer = ComparativeAnswer(
        question=question,
        status="GROUNDED" if claims else "INSUFFICIENT_EVIDENCE",
        intent=intent,
        claims=claims,
        citations=citations,
        answer_text=_trusted_text(claims),
        retrieved_refs=[evidence_ref(item) for item in retrieved],
        limitations=["Hypothetical comparison is not causal or observed evidence."],
    )
    return ComparativeGroundingGuard().validate(answer, package, retrieved)


__all__ = [
    "ComparativeGroundingError",
    "ComparativeGroundingGuard",
    "explain_what_if",
    "retrieve_comparative_evidence",
]
