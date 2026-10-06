from __future__ import annotations

import re
from typing import Protocol

from shelfcash_forecast.decision_intelligence.contracts import (
    DecisionGraph,
    EvidenceItem,
    EvidencePackage,
    RetrievedEvidence,
)
from shelfcash_forecast.decision_intelligence.graph import neighborhood_evidence_ids


def classify_intent(query: str) -> str:
    normalized = query.casefold()
    if re.search(
        r"\bwhat\s+if\b|\bif demand\b|\bincrease\s+\d+%|\bdecrease\s+\d+%|"
        r"\bnếu\b|\bgiả sử\b|\btăng\s+\d+%|\bgiảm\s+\d+%",
        normalized,
    ):
        return "what_if"
    if "stress" in normalized:
        return "stress"
    if any(
        phrase in normalized
        for phrase in (
            "forecast evidence",
            "forecast explanation evidence",
            "bằng chứng dự báo",
            "dữ liệu dự báo",
        )
    ) and any(
        phrase in normalized
        for phrase in ("unavailable", "not supplied", "missing", "không có", "chưa cung cấp")
    ):
        return "forecast_availability"
    if any(
        phrase in normalized
        for phrase in (
            "reliable",
            "confidence",
            "complete",
            "đáng tin",
            "tin cậy",
            "đến đâu",
        )
    ):
        return "readiness"
    if any(
        phrase in normalized
        for phrase in (
            "order now",
            "ordered now",
            "buy now",
            "mua gì",
            "đặt gì",
            "cần mua bao nhiêu",
            "mua bao nhiêu",
            "đặt hàng gì ngay bây giờ",
        )
    ):
        return "immediate_order"
    if any(word in normalized for word in ("rejected", "reject", "failed", "bị loại", "thất bại")):
        return "rejection"
    if "critic" in normalized and any(
        word in normalized for word in ("validate", "validated", "validation", "pass", "passed")
    ):
        return "recommendation"
    if any(
        word in normalized
        for word in (
            "ingredient",
            "recipe",
            "contribute",
            "nguyên liệu",
            "công thức",
            "đến từ sản phẩm nào",
        )
    ):
        return "ingredient"
    if any(
        phrase in normalized
        for phrase in (
            "fefo",
            "consumed lot",
            "consumed lots",
            "lot consumption",
            "lô được dùng",
            "lô đã dùng",
            "tiêu thụ lô",
        )
    ):
        return "lot_consumption"
    mentions_lot = any(token in normalized for token in ("lot", "lots", "lô"))
    if mentions_lot and any(phrase in normalized for phrase in ("expired", "expiry", "hết hạn")):
        return "lot_expiry"
    if any(
        phrase in normalized
        for phrase in (
            "expired lot",
            "expired lots",
            "lot expiry",
            "inventory expiry",
            "lô hết hạn",
            "lô đã hết hạn",
            "hạn sử dụng",
        )
    ):
        return "lot_expiry"
    if mentions_lot and any(
        phrase in normalized
        for phrase in ("waste", "wasted", "write-off", "write off", "bị hủy", "lãng phí")
    ):
        return "lot_waste"
    if any(
        phrase in normalized
        for phrase in (
            "wasted lot",
            "wasted lots",
            "lot waste",
            "write-off",
            "write off",
            "lô bị lãng phí",
            "lô bị hủy",
            "hao hụt",
        )
    ):
        return "lot_waste"
    if any(
        phrase in normalized
        for phrase in (
            "inventory ledger",
            "daily ledger",
            "sổ kho",
            "nhật ký tồn kho",
        )
    ):
        return "inventory_ledger"
    if "risk" in normalized:
        return "inventory_risk"
    if "rủi ro hết hàng" in normalized:
        return "inventory_risk"
    if any(word in normalized for word in ("stockout", "shortage", "hết hàng", "thiếu hàng")):
        return "stockout"
    if any(word in normalized for word in ("risk", "rủi ro", "p95", "cvar")):
        return "inventory_risk"
    if any(
        word in normalized
        for word in (
            "recommend",
            "recommended",
            "khuyên",
            "đề xuất",
            "why should",
            "được chọn",
            "vì sao",
        )
    ):
        return "recommendation"
    return "generic"


def boundary_entity_match(query: str, entity: str) -> bool:
    """Match an entity as a complete identifier, never as an arbitrary substring."""

    if not entity:
        return False
    return bool(
        re.search(
            rf"(?<![\w]){re.escape(entity.casefold())}(?![\w])",
            query.casefold(),
            flags=re.UNICODE,
        )
    )


def longest_matching_entity(query: str, entities: set[str]) -> str | None:
    """Prefer longer complete IDs and break equal-length ties lexically."""

    candidates = sorted(entities, key=lambda value: (-len(value), value))
    return next(
        (value for value in candidates if boundary_entity_match(query, value)),
        None,
    )


class EvidenceRetriever(Protocol):
    def retrieve(
        self,
        query: str,
        evidence: EvidencePackage,
        graph: DecisionGraph,
        *,
        context: dict[str, str] | None = None,
        limit: int = 20,
    ) -> RetrievedEvidence: ...


TYPE_BONUSES: dict[str, dict[str, float]] = {
    "recommendation": {
        "recommendation": 30,
        "critic_verdict": 22,
        "procurement_plan": 12,
        "exact_simulation_package": 10,
        "inventory_risk": 9,
        "strategy_profile": 7,
    },
    "rejection": {
        "critic_verdict": 30,
        "procurement_plan": 15,
        "inventory_key_summary": 12,
        "inventory_risk": 9,
        "recommendation": 8,
    },
    "immediate_order": {
        "first_stage_order": 35,
        "recommendation": 30,
        "procurement_plan": 22,
        "critic_verdict": 15,
        "recourse_order": 2,
    },
    "ingredient": {
        "bom_evidence_availability": 60,
        "recipe_contribution": 45,
        "scenario_recipe_contribution": 42,
        "ingredient_demand": 30,
        "ingredient_demand_scenario": 30,
        "forecast_prediction": 8,
    },
    "stockout": {
        "inventory_key_summary": 50,
        "inventory_daily_ledger": 42,
        "stress_inventory_key": 25,
        "inventory_key_risk": 22,
        "lot_consumption": 8,
        "lot_expiry": 8,
    },
    "lot_consumption": {
        "lot_consumption": 60,
        "inventory_daily_ledger": 24,
        "inventory_key_summary": 16,
        "exact_simulation_package": 10,
    },
    "lot_expiry": {
        "lot_expiry": 60,
        "inventory_key_summary": 24,
        "inventory_daily_ledger": 16,
        "inventory_scenario_result": 12,
    },
    "lot_waste": {
        "lot_waste": 60,
        "inventory_key_summary": 24,
        "inventory_daily_ledger": 16,
        "inventory_scenario_result": 12,
    },
    "inventory_ledger": {
        "inventory_daily_ledger": 60,
        "inventory_key_summary": 28,
        "lot_consumption": 16,
        "lot_expiry": 16,
        "lot_waste": 16,
    },
    "inventory_risk": {
        "inventory_key_risk": 35,
        "inventory_risk": 32,
        "inventory_key_summary": 18,
        "critic_verdict": 8,
    },
    "stress": {
        "stress_definition": 35,
        "stress_inventory_key": 32,
        "stress_result": 25,
        "stress_simulation_package": 20,
    },
    "readiness": {
        "artifact_coherence": 70,
        "exact_simulation_availability": 65,
        "recommendation": 60,
        "critic_verdict": 50,
        "exact_simulation_package": 45,
        "forecast_evidence_availability": 35,
        "bom_evidence_availability": 35,
        "bom_issue": 14,
    },
    "forecast_availability": {
        "forecast_evidence_availability": 70,
        "forecast_prediction": 20,
        "artifact_coherence": 10,
    },
}


_TYPED_ENTITY_PATTERNS: dict[str, tuple[str, ...]] = {
    "ingredient_id": (
        r"\bingredient(?:\s+(?:id|code))?\s*[:=]?\s+([\w.-]+)",
        r"\bnguyên\s+liệu(?:\s+(?:id|mã))?\s*[:=]?\s+([\w.-]+)",
    ),
    "product_id": (
        r"\bproduct(?:\s+(?:id|code))?\s*[:=]?\s+([\w.-]+)",
        r"\bsản\s+phẩm(?:\s+(?:id|mã))?\s*[:=]?\s+([\w.-]+)",
    ),
    "store_id": (
        r"\bstore(?:\s+(?:id|code))?\s*[:=]?\s+([\w.-]+)",
        r"\bcửa\s+hàng(?:\s+(?:id|mã))?\s*[:=]?\s+([\w.-]+)",
    ),
    "supplier_id": (
        r"\bsupplier(?:\s+(?:id|code))?\s*[:=]?\s+([\w.-]+)",
        r"\bnhà\s+cung\s+cấp(?:\s+(?:id|mã))?\s*[:=]?\s+([\w.-]+)",
    ),
    "scenario_id": (
        r"\bscenario(?:\s+(?:id|code))?\s*[:=]?\s+([\w.-]+)",
        r"\bkịch\s+bản(?:\s+(?:id|mã))?\s*[:=]?\s+([\w.-]+)",
    ),
}

_ENTITY_REFERENCE_WORDS = {
    "any",
    "current",
    "demand",
    "nào",
    "này",
    "the",
    "this",
    "what",
    "which",
}


def explicit_typed_entity_mentions(query: str) -> dict[str, str]:
    """Extract explicit typed identifiers without guessing entities from normal prose."""

    mentions: dict[str, str] = {}
    for key, patterns in _TYPED_ENTITY_PATTERNS.items():
        for pattern in patterns:
            match = re.search(pattern, query, flags=re.IGNORECASE | re.UNICODE)
            if match is None:
                continue
            candidate = match.group(1).rstrip(".,?!:;")
            if candidate.casefold() not in _ENTITY_REFERENCE_WORDS:
                mentions[key] = candidate
            break
    return dict(sorted(mentions.items()))


def unknown_explicit_entities(query: str, evidence: EvidencePackage) -> dict[str, str]:
    """Return explicit typed identifiers absent from the current evidence snapshot."""

    unknown: dict[str, str] = {}
    for key, candidate in explicit_typed_entity_mentions(query).items():
        known = {item.entities[key].casefold() for item in evidence.items if key in item.entities}
        if candidate.casefold() not in known:
            unknown[key] = candidate
    return unknown


def build_retrieval_context(
    query: str,
    evidence: EvidencePackage,
    *,
    recommended_strategy: str | None,
) -> dict[str, str]:
    """Build the production query context used by service and evaluation alike."""

    strategy = longest_matching_entity(query, {"LEAN", "BALANCED", "PROTECTED"})
    if strategy is None and classify_intent(query) in {
        "immediate_order",
        "inventory_ledger",
        "inventory_risk",
        "lot_consumption",
        "lot_expiry",
        "lot_waste",
        "readiness",
        "recommendation",
        "stockout",
        "stress",
    }:
        strategy = recommended_strategy
    context = {"strategy": strategy} if strategy is not None else {}
    for key in ("store_id", "product_id", "ingredient_id", "scenario_id", "supplier_id"):
        match = longest_matching_entity(
            query,
            {item.entities[key] for item in evidence.items if key in item.entities},
        )
        if match is not None:
            context[key] = match
    return context


def _tokens(value: str) -> set[str]:
    return set(re.findall(r"[\w.-]+", value.casefold(), flags=re.UNICODE))


def _search_text(item: EvidenceItem) -> str:
    return " ".join(
        [
            item.text,
            item.evidence_type,
            item.source_object,
            item.source_path,
            *item.entities.keys(),
            *item.entities.values(),
            *item.warnings,
        ]
    )


class StructuredLocalRetriever:
    """Offline structured-first retrieval with deterministic text tie-breaking."""

    def retrieve(
        self,
        query: str,
        evidence: EvidencePackage,
        graph: DecisionGraph,
        *,
        context: dict[str, str] | None = None,
        limit: int = 20,
    ) -> RetrievedEvidence:
        if limit < 1:
            raise ValueError("Retrieval limit must be positive.")
        intent = classify_intent(query)
        if intent == "what_if":
            return RetrievedEvidence(query=query, items=[], scores={}, intent=intent)
        unknown_entities = unknown_explicit_entities(query, evidence)
        if unknown_entities:
            return RetrievedEvidence(query=query, items=[], scores={}, intent=intent)
        query_tokens = _tokens(query)
        normalized_context = {
            str(key): str(value) for key, value in sorted((context or {}).items())
        }
        bonuses = TYPE_BONUSES.get(intent, {})
        scores: dict[str, float] = {}
        for item in evidence.items:
            score = bonuses.get(item.evidence_type, 0.0)
            item_tokens = _tokens(_search_text(item))
            score += 2.0 * len(query_tokens & item_tokens)
            for key, expected in normalized_context.items():
                if item.entities.get(key, "").casefold() == expected.casefold():
                    score += 20.0
                elif key in item.entities:
                    score -= 5.0
            for strategy in ("LEAN", "BALANCED", "PROTECTED"):
                if boundary_entity_match(query, strategy):
                    score += 18.0 if item.entities.get("strategy") == strategy else -4.0
            if score > 0:
                scores[item.evidence_id] = score

        ranked_seed_ids = [
            evidence_id
            for evidence_id, _ in sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))[
                : max(3, min(limit, 8))
            ]
        ]
        neighbor_ids = neighborhood_evidence_ids(graph, set(ranked_seed_ids), depth=1)
        for evidence_id in neighbor_ids:
            if evidence_id in scores:
                scores[evidence_id] += 1.0
            elif evidence_id in {item.evidence_id for item in evidence.items}:
                scores[evidence_id] = 0.5

        item_by_id = {item.evidence_id: item for item in evidence.items}
        selected_ids = [
            evidence_id
            for evidence_id, _ in sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
            if evidence_id in item_by_id
        ][:limit]
        return RetrievedEvidence(
            query=query,
            items=[item_by_id[evidence_id] for evidence_id in selected_ids],
            scores={evidence_id: scores[evidence_id] for evidence_id in selected_ids},
            intent=intent,
        )
