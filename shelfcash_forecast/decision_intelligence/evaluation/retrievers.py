from __future__ import annotations

from shelfcash_forecast.decision_intelligence.contracts import (
    DecisionGraph,
    EvidencePackage,
    RetrievedEvidence,
)
from shelfcash_forecast.decision_intelligence.retrieval import classify_intent


def _tokens(value: str) -> set[str]:
    from shelfcash_forecast.decision_intelligence.retrieval import _tokens as core_tokens

    return core_tokens(value)


class LexicalOnlyRetriever:
    """Deterministic token-overlap ablation without type, entity or graph bonuses."""

    def retrieve(
        self,
        query: str,
        evidence: EvidencePackage,
        graph: DecisionGraph,
        *,
        context: dict[str, str] | None = None,
        limit: int = 20,
    ) -> RetrievedEvidence:
        del graph, context
        if limit < 1:
            raise ValueError("Retrieval limit must be positive.")
        intent = classify_intent(query)
        if intent == "what_if":
            return RetrievedEvidence(query=query, items=[], scores={}, intent=intent)
        query_tokens = _tokens(query)
        scores: dict[str, float] = {}
        for item in evidence.items:
            text = " ".join(
                [
                    item.text,
                    item.evidence_type,
                    item.source_object,
                    *item.entities.keys(),
                    *item.entities.values(),
                ]
            )
            score = float(len(query_tokens & _tokens(text)))
            if score > 0:
                scores[item.evidence_id] = score
        item_by_id = {item.evidence_id: item for item in evidence.items}
        selected = [
            evidence_id
            for evidence_id, _ in sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
        ][:limit]
        return RetrievedEvidence(
            query=query,
            items=[item_by_id[evidence_id] for evidence_id in selected],
            scores={evidence_id: scores[evidence_id] for evidence_id in selected},
            intent=intent,
        )
