from __future__ import annotations

from typing import Any

from shelfcash_forecast.decision_intelligence.contracts import FinalDecisionPackage
from shelfcash_forecast.decision_intelligence.integrity import sha256_content_hash
from shelfcash_forecast.decision_intelligence.what_if.contracts import (
    ComparativeDecisionGraph,
    ComparativeEvidenceItem,
    ComparativeEvidencePackage,
    ComparativeGraphEdge,
    DecisionComparison,
    WhatIfModification,
)


def evidence_ref(item: ComparativeEvidenceItem) -> str:
    return f"{item.package_role}|{item.package_hash}|{item.evidence_id}"


def _copy_decision_evidence(
    role: str,
    decision_hash: str,
    decision: FinalDecisionPackage,
) -> list[ComparativeEvidenceItem]:
    return [
        ComparativeEvidenceItem(
            package_role=role,
            package_hash=decision_hash,
            evidence_id=item.evidence_id,
            content_hash=item.content_hash,
            evidence_type=item.evidence_type,
            semantics=item.semantics,
            payload=item.payload,
            text=item.text,
        )
        for item in decision.evidence_package.items
    ]


def _delta_item(
    comparison_hash: str,
    evidence_type: str,
    locator: str,
    payload: dict[str, Any],
    text: str,
    *,
    semantics: str = "hypothetical_comparison",
) -> ComparativeEvidenceItem:
    material = {
        "evidence_type": evidence_type,
        "locator": locator,
        "payload": payload,
        "text": text,
        "semantics": semantics,
    }
    content_hash = sha256_content_hash(material)
    return ComparativeEvidenceItem(
        package_role="DELTA",
        package_hash=comparison_hash,
        evidence_id=f"delta-{evidence_type}-{sha256_content_hash(locator)[7:23]}",
        content_hash=content_hash,
        evidence_type=evidence_type,
        semantics=semantics,
        payload=payload,
        text=text,
    )


def build_comparative_evidence(
    baseline: FinalDecisionPackage,
    hypothetical: FinalDecisionPackage,
    comparison: DecisionComparison,
    modifications: list[WhatIfModification],
) -> ComparativeEvidencePackage:
    baseline_hash = comparison.baseline_decision_hash
    hypothetical_hash = comparison.hypothetical_decision_hash
    comparison_hash = sha256_content_hash(comparison)
    items = _copy_decision_evidence("BASELINE", baseline_hash, baseline)
    items.extend(_copy_decision_evidence("HYPOTHETICAL", hypothetical_hash, hypothetical))
    for index, modification in enumerate(modifications):
        payload = modification.model_dump(mode="json")
        items.append(
            _delta_item(
                comparison_hash,
                "recorded_modification",
                f"modifications[{index}]",
                payload,
                f"Hypothetical assumption {index + 1} is recorded as "
                f"{modification.modification_type}.",
            )
        )
    items.append(
        _delta_item(
            comparison_hash,
            "recommendation_delta",
            "comparison.recommendation",
            {
                "baseline_strategy": comparison.baseline_strategy,
                "hypothetical_strategy": comparison.hypothetical_strategy,
                "changed": comparison.strategy_changed,
            },
            "The baseline and hypothetical M5 recommendations were compared.",
        )
    )
    items.append(
        _delta_item(
            comparison_hash,
            "readiness_delta",
            "comparison.readiness",
            {
                "baseline_readiness": comparison.baseline_readiness,
                "hypothetical_readiness": comparison.hypothetical_readiness,
            },
            "Baseline and hypothetical evidence readiness were compared.",
        )
    )
    for stage, rows in (
        ("first_stage", comparison.first_stage_order_deltas),
        ("scenario_recourse", comparison.recourse_order_deltas),
    ):
        for index, row in enumerate(rows):
            items.append(
                _delta_item(
                    comparison_hash,
                    f"{stage}_order_delta",
                    f"comparison.{stage}[{index}]",
                    row.model_dump(mode="json"),
                    f"The {stage} quantity for offer {row.offer_id} changes from "
                    f"{row.baseline_quantity} to {row.hypothetical_quantity} {row.unit} "
                    "in the hypothetical run.",
                )
            )
    for index, metric in enumerate(comparison.metric_comparisons):
        items.append(
            _delta_item(
                comparison_hash,
                "metric_delta",
                f"comparison.metrics[{index}]",
                metric.model_dump(mode="json"),
                f"The typed {metric.name} comparison status is {metric.status}.",
                semantics=(
                    "probabilistic_hypothetical_comparison"
                    if "probability" in metric.name and metric.status == "COMPARABLE"
                    else "hypothetical_comparison"
                ),
            )
        )
    items.sort(key=lambda item: (item.package_role, item.package_hash, item.evidence_id))
    return ComparativeEvidencePackage(items=items)


def build_comparative_graph(
    evidence: ComparativeEvidencePackage,
    comparison: DecisionComparison,
) -> ComparativeDecisionGraph:
    baseline_node = f"BASELINE|{comparison.baseline_decision_hash}"
    hypothetical_node = f"HYPOTHETICAL|{comparison.hypothetical_decision_hash}"
    delta_node = f"DELTA|{sha256_content_hash(comparison)}"
    nodes = {baseline_node, hypothetical_node, delta_node}
    edges = [
        ComparativeGraphEdge(
            source=hypothetical_node,
            target=baseline_node,
            relation="COUNTERFACTUAL_OF",
        ),
        ComparativeGraphEdge(
            source=delta_node,
            target=baseline_node,
            relation="COMPARED_WITH",
        ),
        ComparativeGraphEdge(
            source=delta_node,
            target=hypothetical_node,
            relation="COMPARED_WITH",
        ),
    ]
    for item in evidence.items:
        ref = evidence_ref(item)
        nodes.add(ref)
        parent = {
            "BASELINE": baseline_node,
            "HYPOTHETICAL": hypothetical_node,
            "DELTA": delta_node,
        }.get(item.package_role)
        if parent:
            edges.append(
                ComparativeGraphEdge(
                    source=ref,
                    target=parent,
                    relation=(
                        "MODIFIED_BY"
                        if item.evidence_type == "recorded_modification"
                        else "VALIDATED_BY"
                    ),
                )
            )
    edges.sort(key=lambda edge: (edge.source, edge.target, edge.relation))
    return ComparativeDecisionGraph(nodes=sorted(nodes), edges=edges)


__all__ = ["build_comparative_evidence", "build_comparative_graph", "evidence_ref"]
