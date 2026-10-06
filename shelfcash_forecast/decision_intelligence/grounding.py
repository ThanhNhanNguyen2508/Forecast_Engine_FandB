# Implementation mặc định là:

# DeterministicGroundedGenerator

# Nó chưa gọi LLM. Nó dùng template có kiểm soát để tạo:

# GroundedClaim(
#     claim_type="recommendation",
#     text="M5 recommends BALANCED...",
#     evidence_ids=["ev-m5-recommendation-..."],
#     facts={
#         "strategy": "BALANCED",
#         "recommendation_rule": "...",
#     },
# )

# Sau đó render:

# M5 recommends BALANCED. This is inherited from the recorded rule...
# [evidence:ev-m5-recommendation-...]


# BALANCED passed the M5 critic...
# [evidence:ev-m5-critic-verdict-...]


# The decision was evaluated by exact M4...
# [evidence:ev-m4-exact-simulation-package-...]

# GroundedGenerator là Protocol. Sau này có thể thay bằng LLM adapter, nhưng LLM vẫn chỉ được tạo giải thích, không được sửa M5 recommendation.
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Protocol

from shelfcash_forecast.decision_intelligence.contracts import (
    DecisionAnswer,
    FinalDecisionPackage,
    GroundedClaim,
    RetrievedEvidence,
)
from shelfcash_forecast.decision_intelligence.integrity import canonical_json
from shelfcash_forecast.decision_intelligence.retrieval import longest_matching_entity


class GroundingError(ValueError):
    """Raised when a generated answer violates the current evidence boundary."""


WHAT_IF_LIMITATIONS = [
    "M6 Part 1 is read-only and does not modify assumptions or rerun optimization."
]
INSUFFICIENT_EVIDENCE_LIMITATIONS = [
    "The supplied M1-M5 artifacts do not contain evidence for this claim."
]


def _positive_probability_language(text: str) -> bool:
    normalized = text.casefold()
    normalized = re.sub(
        r"\b(no|not|without|cannot|do not|does not)\b[^.;]{0,40}\b"
        r"(probability|probabilistic|likelihood|chance)\b",
        "",
        normalized,
    )
    normalized = re.sub(
        r"\b(không|chẳng)\b[^.;]{0,40}\b(xác suất|khả năng xảy ra)\b",
        "",
        normalized,
    )
    return bool(
        re.search(
            r"\b(probability|probabilistic|likelihood|chance)\b|"
            r"\b\d+(?:[.,]\d+)?% chance\b|"
            r"\b(xác suất|khả năng xảy ra|phần trăm khả năng)\b|"
            r"\bcó\s+\d+(?:[.,]\d+)?%\s+nguy cơ\b",
            normalized,
        )
    )


def _unsupported_forecast_causality(text: str) -> bool:
    normalized = text.casefold()
    if "unavailable" in normalized or "not provided" in normalized:
        return False
    normalized = re.sub(
        r"\b(not caused by|not due to|không (?:phải )?do|không bởi vì)\b",
        "",
        normalized,
    )
    return bool(
        re.search(
            r"\b(caused by|because of|due to|driven by|causal)\b|"
            r"\b(bởi vì|nguyên nhân là|được thúc đẩy bởi|dẫn đến)\b|"
            r"(?<!\w)do\s+(?!not\b|không\b)\S+",
            normalized,
        )
    )


class GroundedGenerator(Protocol):
    def generate(
        self,
        question: str,
        retrieved: RetrievedEvidence,
        decision: FinalDecisionPackage,
    ) -> DecisionAnswer: ...


def _strategy_in_question(question: str) -> str | None:
    return longest_matching_entity(question, {"LEAN", "BALANCED", "PROTECTED"})


def _render(claims: list[GroundedClaim]) -> str:
    return "\n".join(
        f"{claim.text} "
        + " ".join(f"[evidence:{evidence_id}]" for evidence_id in claim.evidence_ids)
        for claim in claims
    )


def _answer(
    question: str,
    retrieved: RetrievedEvidence,
    claims: list[GroundedClaim],
    *,
    status: str = "GROUNDED",
    text: str | None = None,
    limitations: list[str] | None = None,
) -> DecisionAnswer:
    citations = sorted({evidence_id for claim in claims for evidence_id in claim.evidence_ids})
    return DecisionAnswer(
        question=question,
        intent=retrieved.intent,
        status=status,
        answer_text=text or _render(claims),
        claims=claims,
        citations=citations,
        retrieved_evidence_ids=[item.evidence_id for item in retrieved.items],
        limitations=limitations or [],
        provenance={
            "generator": "deterministic_grounded_generator_v1",
            "source_of_truth": "retrieved_evidence_only",
        },
    )


class DeterministicGroundedGenerator:
    """Translate retrieved typed evidence into offline, citation-bearing claims."""

    def generate(
        self,
        question: str,
        retrieved: RetrievedEvidence,
        decision: FinalDecisionPackage,
    ) -> DecisionAnswer:
        if retrieved.intent == "what_if":
            return _answer(
                question,
                retrieved,
                [],
                status="UNSUPPORTED_INTENT",
                text="WHAT_IF_NOT_AVAILABLE_IN_M6_PART1",
                limitations=WHAT_IF_LIMITATIONS,
            )
        handlers = {
            "recommendation": self._recommendation,
            "rejection": self._rejection,
            "immediate_order": self._immediate_orders,
            "ingredient": self._ingredient,
            "stockout": self._stockout,
            "inventory_risk": self._inventory_risk,
            "stress": self._stress,
            "readiness": self._readiness,
            "forecast_availability": self._forecast_availability,
            "lot_consumption": self._generic,
            "lot_expiry": self._generic,
            "lot_waste": self._generic,
            "inventory_ledger": self._generic,
            "generic": self._generic,
        }
        claims = handlers[retrieved.intent](question, retrieved, decision)
        if not claims:
            return _answer(
                question,
                retrieved,
                [],
                status="INSUFFICIENT_EVIDENCE",
                text="INSUFFICIENT_EVIDENCE",
                limitations=INSUFFICIENT_EVIDENCE_LIMITATIONS,
            )
        return _answer(question, retrieved, claims)

    @staticmethod
    def _items(retrieved: RetrievedEvidence, evidence_type: str) -> list:
        return [item for item in retrieved.items if item.evidence_type == evidence_type]

    def _recommendation(
        self,
        question: str,
        retrieved: RetrievedEvidence,
        decision: FinalDecisionPackage,
    ) -> list[GroundedClaim]:
        recommendation = self._items(retrieved, "recommendation")
        if not recommendation:
            return []
        item = recommendation[0]
        strategy = item.payload.get("recommended_strategy")
        if strategy is None:
            claims = [
                GroundedClaim(
                    claim_type="no_valid_plan",
                    text=(
                        "M5 returned NO_VALID_PROCUREMENT_PLAN, so M6 presents no "
                        "fallback strategy or immediate order."
                    ),
                    evidence_ids=[item.evidence_id],
                    facts={"strategy": None, "status": item.payload.get("status")},
                )
            ]
            for critic in self._items(retrieved, "critic_verdict"):
                violations = list(critic.payload.get("hard_violations", []))
                if violations:
                    claims.append(
                        GroundedClaim(
                            claim_type="critic_rejection",
                            text=(
                                f"{critic.entities['strategy']} was rejected by the M5 critic for: "
                                f"{', '.join(violations)}."
                            ),
                            evidence_ids=[critic.evidence_id],
                            facts={
                                "strategy": critic.entities["strategy"],
                                "passed": bool(critic.payload.get("passed")),
                                "hard_violations": violations,
                            },
                        )
                    )
            return claims
        rule = item.payload.get("recommendation_rule")
        recommendation_facts = {
            "strategy": strategy,
            "recommendation_rule_status": item.payload["recommendation_rule_status"],
        }
        if rule is not None:
            recommendation_facts["recommendation_rule"] = rule
        claims = [
            GroundedClaim(
                claim_type="recommendation",
                text=(
                    f"M5 recommends {strategy}. This is inherited from the recorded "
                    f"rule {rule}, not from an M6 ranking."
                    if rule is not None
                    else f"M5 recommends {strategy}. The recommendation rule was not "
                    "supplied in OptimizationResult.provenance; M6 does not reconstruct it."
                ),
                evidence_ids=[item.evidence_id],
                facts=recommendation_facts,
            )
        ]
        critics = [
            candidate
            for candidate in self._items(retrieved, "critic_verdict")
            if candidate.entities.get("strategy") == strategy
        ]
        if critics:
            critic = critics[0]
            claims.append(
                GroundedClaim(
                    claim_type="critic_validation",
                    text=(
                        f"{strategy} passed the M5 critic; "
                        f"hard violations: {critic.payload.get('hard_violations') or 'none'}."
                    ),
                    evidence_ids=[critic.evidence_id],
                    facts={"strategy": strategy, "passed": bool(critic.payload["passed"])},
                )
            )
        exact = [
            candidate
            for candidate in retrieved.items
            if candidate.entities.get("strategy") == strategy
            and candidate.evidence_type
            in {"exact_simulation_package", "inventory_risk", "inventory_key_summary"}
        ]
        if exact:
            evidence = exact[0]
            claims.append(
                GroundedClaim(
                    claim_type="exact_inventory_validation",
                    text=(
                        f"The decision was evaluated by the exact M4 inventory layer: "
                        f"{evidence.text}"
                    ),
                    evidence_ids=[evidence.evidence_id],
                    facts={"strategy": strategy, "authority": "exact_m4"},
                    uses_probability_language=evidence.semantics == "probabilistic",
                )
            )
        return claims

    def _rejection(
        self,
        question: str,
        retrieved: RetrievedEvidence,
        decision: FinalDecisionPackage,
    ) -> list[GroundedClaim]:
        strategy = _strategy_in_question(question)
        critics = [
            item
            for item in self._items(retrieved, "critic_verdict")
            if strategy is None or item.entities.get("strategy") == strategy
        ]
        claims = []
        for critic in critics:
            violations = list(critic.payload.get("hard_violations", []))
            if critic.payload.get("passed"):
                text = (
                    f"{critic.entities['strategy']} was not rejected by the M5 critic; "
                    "it passed, although the recommendation rule may prefer another valid strategy."
                )
            else:
                text = (
                    f"{critic.entities['strategy']} was rejected by the M5 critic for: "
                    f"{', '.join(violations) if violations else 'unspecified hard violation'}."
                )
            claims.append(
                GroundedClaim(
                    claim_type="critic_rejection",
                    text=text,
                    evidence_ids=[critic.evidence_id],
                    facts={
                        "strategy": critic.entities["strategy"],
                        "passed": bool(critic.payload.get("passed")),
                        "hard_violations": violations,
                    },
                )
            )
            mismatch = dict(critic.payload.get("details", {}).get("candidate_model_mismatch", {}))
            if "CANDIDATE_MODEL_MISMATCH" in violations and mismatch:
                predicted_fill = mismatch.get("predicted_fill_rate")
                exact_fill = mismatch.get("simulated_expected_fill_rate")
                predicted_stockout = mismatch.get("predicted_stockout_probability")
                exact_stockout = mismatch.get("simulated_stockout_probability")
                values = []
                if predicted_fill is not None and exact_fill is not None:
                    values.append(
                        f"expected fill {float(predicted_fill):g} versus exact M4 "
                        f"{float(exact_fill):g}"
                    )
                risk_items = [
                    item
                    for item in self._items(retrieved, "inventory_risk")
                    if item.entities.get("strategy") == critic.entities.get("strategy")
                ]
                if predicted_stockout is not None and exact_stockout is not None and risk_items:
                    values.append(
                        f"stockout probability {float(predicted_stockout):g} versus "
                        f"exact M4 {float(exact_stockout):g}"
                    )
                if values:
                    mismatch_facts = {
                        "strategy": critic.entities["strategy"],
                        "predicted_fill_rate": predicted_fill,
                        "exact_fill_rate": exact_fill,
                    }
                    if risk_items and predicted_stockout is not None and exact_stockout is not None:
                        mismatch_facts.update(
                            {
                                "predicted_stockout_probability": predicted_stockout,
                                "exact_stockout_probability": exact_stockout,
                            }
                        )
                    uses_risk = (
                        bool(risk_items)
                        and predicted_stockout is not None
                        and exact_stockout is not None
                    )
                    claims.append(
                        GroundedClaim(
                            claim_type="candidate_model_mismatch",
                            text=(
                                "The solver surrogate predicted "
                                f"{'; '.join(values)}. M5 recorded that the accepted "
                                "model-gap tolerance was exceeded; exact M4 remains the "
                                "validation authority."
                            ),
                            evidence_ids=[
                                critic.evidence_id,
                                *([risk_items[0].evidence_id] if uses_risk else []),
                            ],
                            facts=mismatch_facts,
                            uses_probability_language=(uses_risk),
                        )
                    )
        return claims

    def _immediate_orders(
        self,
        question: str,
        retrieved: RetrievedEvidence,
        decision: FinalDecisionPackage,
    ) -> list[GroundedClaim]:
        recommendation = self._items(retrieved, "recommendation")
        if not recommendation:
            return []
        rec_item = recommendation[0]
        strategy = decision.recommended_strategy
        if strategy is None:
            return [
                GroundedClaim(
                    claim_type="no_valid_plan",
                    text=(
                        "M5 returned NO_VALID_PROCUREMENT_PLAN, so M6 presents no "
                        "fallback strategy or immediate order."
                    ),
                    evidence_ids=[rec_item.evidence_id],
                    facts={"strategy": None, "status": rec_item.payload.get("status")},
                )
            ]
        claims = [
            GroundedClaim(
                claim_type="immediate_order_authority",
                text=f"Immediate orders must come only from {strategy}'s first-stage plan.",
                evidence_ids=[rec_item.evidence_id],
                facts={"strategy": strategy},
            )
        ]
        order_items = {
            item.evidence_id: item for item in self._items(retrieved, "first_stage_order")
        }
        if not decision.immediate_orders:
            plans = [
                item
                for item in self._items(retrieved, "procurement_plan")
                if item.entities.get("strategy") == strategy
            ]
            if plans and plans[0].payload.get("first_stage_order_count") == 0:
                claims.append(
                    GroundedClaim(
                        claim_type="no_immediate_order",
                        text=f"{strategy}'s M5 plan contains no first-stage orders.",
                        evidence_ids=[plans[0].evidence_id],
                        facts={"strategy": strategy, "first_stage_order_count": 0},
                    )
                )
        for order in decision.immediate_orders:
            item = order_items.get(order.evidence_id)
            if item is None:
                continue
            claims.append(
                GroundedClaim(
                    claim_type="immediate_order",
                    text=(
                        f"Order {order.order_quantity:g} {order.unit} of "
                        f"{order.ingredient_id} from {order.supplier_id} on "
                        f"{order.order_date.isoformat()}."
                    ),
                    evidence_ids=[order.evidence_id],
                    facts={
                        "strategy": strategy,
                        "order_evidence_id": order.evidence_id,
                        "offer_id": order.offer_id,
                        "supplier_id": order.supplier_id,
                        "store_id": order.store_id,
                        "ingredient_id": order.ingredient_id,
                        "order_date": order.order_date.isoformat(),
                        "arrival_date": order.arrival_date.isoformat(),
                        "order_quantity": order.order_quantity,
                        "unit": order.unit,
                    },
                )
            )
        return claims

    def _ingredient(
        self,
        question: str,
        retrieved: RetrievedEvidence,
        decision: FinalDecisionPackage,
    ) -> list[GroundedClaim]:
        contributions = [
            item
            for item in retrieved.items
            if item.evidence_type in {"recipe_contribution", "scenario_recipe_contribution"}
        ]
        requested_ingredient = longest_matching_entity(
            question,
            {
                item.entities["ingredient_id"]
                for item in contributions
                if "ingredient_id" in item.entities
            },
        )
        if requested_ingredient is not None:
            contributions = [
                item
                for item in contributions
                if item.entities.get("ingredient_id") == requested_ingredient
            ]
        claims = []
        for item in contributions[:6]:
            payload = item.payload
            final = payload.get("contribution_p50", payload.get("final_quantity"))
            unit = payload.get("contribution_unit", payload.get("unit"))
            claims.append(
                GroundedClaim(
                    claim_type="recipe_contribution",
                    text=(
                        f"Product {payload['product_id']} contributes {float(final):g} "
                        f"{unit} using recipe {payload['recipe_id']}/"
                        f"{payload['recipe_version']}."
                    ),
                    evidence_ids=[item.evidence_id],
                    facts={
                        "ingredient_id": item.entities.get("ingredient_id"),
                        "product_id": payload["product_id"],
                        "recipe_id": payload["recipe_id"],
                        "recipe_version": payload["recipe_version"],
                        "final_quantity": final,
                        "unit": unit,
                    },
                    uses_probability_language=item.semantics == "probabilistic",
                )
            )
        if not claims:
            unavailable = self._items(retrieved, "bom_evidence_availability")
            if unavailable:
                return []
        return claims

    def _stockout(
        self,
        question: str,
        retrieved: RetrievedEvidence,
        decision: FinalDecisionPackage,
    ) -> list[GroundedClaim]:
        candidates = []
        for item in retrieved.items:
            shortage = item.payload.get("shortage_quantity")
            if shortage is None and item.evidence_type == "inventory_daily_ledger":
                shortage = item.payload.get("shortage_quantity")
            if shortage is not None and float(shortage) > 0:
                candidates.append((item, float(shortage)))
        candidates.sort(key=lambda pair: (-pair[1], pair[0].evidence_id))
        claims = []
        for item, shortage in candidates[:5]:
            scenario_id = item.entities.get("scenario_id")
            if item.semantics == "stress":
                context = (
                    f"Under adversarial stress scenario {scenario_id}, which is not a "
                    "probability estimate,"
                )
            elif item.semantics == "quantile":
                context = (
                    f"In quantile scenario {scenario_id}, which is not a probability estimate,"
                )
            elif item.semantics == "probabilistic":
                context = f"In explicitly weighted probabilistic scenario {scenario_id},"
            else:
                context = f"In exact scenario {scenario_id},"
            claims.append(
                GroundedClaim(
                    claim_type="stockout_location",
                    text=(
                        f"{context} stockout evidence at {item.entities.get('store_id')}/"
                        f"{item.entities.get('ingredient_id')}"
                        f"{f' on {item.event_date.isoformat()}' if item.event_date else ''}: "
                        f"shortage={shortage:g} {item.entities.get('unit', item.payload.get('unit', ''))}."
                    ),
                    evidence_ids=[item.evidence_id],
                    facts={
                        "strategy": item.entities.get("strategy"),
                        "scenario_id": item.entities.get("scenario_id"),
                        "store_id": item.entities.get("store_id"),
                        "ingredient_id": item.entities.get("ingredient_id"),
                        "shortage": shortage,
                        "unit": item.entities.get("unit", item.payload.get("unit", "")),
                        "event_date": (
                            item.event_date.isoformat() if item.event_date is not None else None
                        ),
                        "semantics": item.semantics,
                    },
                    uses_probability_language=item.semantics == "probabilistic",
                )
            )
        return claims

    def _inventory_risk(
        self,
        question: str,
        retrieved: RetrievedEvidence,
        decision: FinalDecisionPackage,
    ) -> list[GroundedClaim]:
        items = self._items(retrieved, "inventory_key_risk")
        items.sort(
            key=lambda item: (
                -float(item.payload.get("stockout_probability", 0)),
                -float(item.payload.get("expected_shortage", 0)),
                item.evidence_id,
            )
        )
        return [
            GroundedClaim(
                claim_type="probabilistic_inventory_risk",
                text=(
                    f"{item.entities['store_id']}/{item.entities['ingredient_id']} has "
                    f"stockout probability {float(item.payload['stockout_probability']):g}, "
                    f"expected shortage {float(item.payload['expected_shortage']):g} "
                    f"{item.entities['unit']}, and expected fill rate "
                    f"{float(item.payload['expected_fill_rate']):g}; expected expiry is "
                    f"{float(item.payload['expected_expired_quantity']):g} and expected "
                    f"explicit waste is {float(item.payload['expected_explicit_waste']):g} "
                    f"{item.entities['unit']}."
                ),
                evidence_ids=[item.evidence_id],
                facts={
                    "strategy": item.entities.get("strategy"),
                    "store_id": item.entities["store_id"],
                    "ingredient_id": item.entities["ingredient_id"],
                    "unit": item.entities["unit"],
                    "stockout_probability": item.payload["stockout_probability"],
                    "expected_shortage": item.payload["expected_shortage"],
                    "expected_fill_rate": item.payload["expected_fill_rate"],
                    "expected_expired_quantity": item.payload["expected_expired_quantity"],
                    "expected_explicit_waste": item.payload["expected_explicit_waste"],
                },
                uses_probability_language=True,
            )
            for item in items[:5]
        ]

    def _stress(
        self,
        question: str,
        retrieved: RetrievedEvidence,
        decision: FinalDecisionPackage,
    ) -> list[GroundedClaim]:
        claims = []
        for item in self._items(retrieved, "stress_definition"):
            claims.append(
                GroundedClaim(
                    claim_type="stress_definition",
                    text=(
                        f"Adversarial stress scenario {item.entities['stress_id']} assumes "
                        "demand multiplier "
                        f"{float(item.payload['demand_multiplier']):g} and supplier delay "
                        f"{int(item.payload['supplier_delay_days'])} day(s). It is not a "
                        "probability estimate."
                    ),
                    evidence_ids=[item.evidence_id],
                    facts={
                        "stress_id": item.entities["stress_id"],
                        "demand_multiplier": item.payload["demand_multiplier"],
                        "supplier_delay_days": item.payload["supplier_delay_days"],
                        "supplier_ids": sorted(item.payload.get("supplier_ids", [])),
                        "probabilistic": False,
                    },
                )
            )
        stress_items = self._items(retrieved, "stress_inventory_key")
        stress_items.sort(
            key=lambda item: (-float(item.payload.get("shortage_quantity", 0)), item.evidence_id)
        )
        for item in stress_items[:5]:
            expired = float(item.payload.get("expired_quantity", 0))
            capacity = float(item.payload.get("capacity_violation_quantity") or 0)
            warning_text = ", ".join(item.warnings) if item.warnings else "none"
            claims.append(
                GroundedClaim(
                    claim_type="stress_consequence",
                    text=(
                        f"Under adversarial stress scenario {item.entities['scenario_id']}, "
                        f"{item.entities['store_id']}/{item.entities['ingredient_id']} has "
                        f"shortage {float(item.payload['shortage_quantity']):g} "
                        f"{item.entities['unit']} and fill rate "
                        f"{float(item.payload['fill_rate']):g}; expiry={expired:g}, "
                        f"capacity consequence={capacity:g}, warnings={warning_text}."
                    ),
                    evidence_ids=[item.evidence_id],
                    facts={
                        "stress_id": item.entities["scenario_id"],
                        "store_id": item.entities["store_id"],
                        "ingredient_id": item.entities["ingredient_id"],
                        "unit": item.entities["unit"],
                        "shortage": item.payload["shortage_quantity"],
                        "fill_rate": item.payload["fill_rate"],
                        "expired_quantity": expired,
                        "capacity_violation_quantity": capacity,
                        "warnings": item.warnings,
                        "probabilistic": False,
                    },
                )
            )
        return claims

    def _readiness(
        self,
        question: str,
        retrieved: RetrievedEvidence,
        decision: FinalDecisionPackage,
    ) -> list[GroundedClaim]:
        decomposition = decision.confidence_decomposition
        dimensions = {
            "artifact_coherence": decomposition.artifact_coherence,
            "forecast_evidence": decomposition.forecast_evidence,
            "scenario_evidence": decomposition.scenario_evidence,
            "bom_traceability": decomposition.bom_traceability,
            "inventory_validation": decomposition.inventory_validation,
            "optimization_validity": decomposition.optimization_validity,
            "stress_evidence": decomposition.stress_evidence,
            "overall_decision_readiness": decomposition.overall_decision_readiness,
        }
        claims = []
        retrieved_ids = {item.evidence_id for item in retrieved.items}
        for name, dimension in dimensions.items():
            if not dimension.evidence_ids or not set(dimension.evidence_ids) <= retrieved_ids:
                continue
            claims.append(
                GroundedClaim(
                    claim_type="evidence_readiness",
                    text=f"{name}={dimension.status}: {dimension.reason}",
                    evidence_ids=dimension.evidence_ids,
                    facts={"dimension": name, "status": dimension.status},
                )
            )
        return claims

    def _generic(
        self,
        question: str,
        retrieved: RetrievedEvidence,
        decision: FinalDecisionPackage,
    ) -> list[GroundedClaim]:
        return [
            GroundedClaim(
                claim_type="retrieved_fact",
                text=item.text,
                evidence_ids=[item.evidence_id],
                facts={"evidence_type": item.evidence_type},
                uses_probability_language=item.semantics == "probabilistic",
            )
            for item in retrieved.items[:5]
        ]

    def _forecast_availability(
        self,
        question: str,
        retrieved: RetrievedEvidence,
        decision: FinalDecisionPackage,
    ) -> list[GroundedClaim]:
        return [
            GroundedClaim(
                claim_type="retrieved_fact",
                text=item.text,
                evidence_ids=[item.evidence_id],
                facts={"evidence_type": item.evidence_type},
            )
            for item in retrieved.items
            if item.evidence_type == "forecast_evidence_availability"
        ][:1]


@dataclass(frozen=True)
class _TrustedClaim:
    text: str
    facts: dict
    evidence_ids: list[str]
    uses_probability_language: bool = False
    causal: bool = False


def _require_one(cited: list, *evidence_types: str):
    selected = [item for item in cited if item.evidence_type in evidence_types]
    if len(selected) != 1:
        raise GroundingError(
            f"Claim requires exactly one of evidence types {sorted(evidence_types)}."
        )
    return selected[0]


def _stockout_text(item, shortage: float) -> str:
    scenario_id = item.entities.get("scenario_id")
    if item.semantics == "stress":
        context = (
            f"Under adversarial stress scenario {scenario_id}, which is not a probability estimate,"
        )
    elif item.semantics == "quantile":
        context = f"In quantile scenario {scenario_id}, which is not a probability estimate,"
    elif item.semantics == "probabilistic":
        context = f"In explicitly weighted probabilistic scenario {scenario_id},"
    else:
        context = f"In exact scenario {scenario_id},"
    return (
        f"{context} stockout evidence at {item.entities.get('store_id')}/"
        f"{item.entities.get('ingredient_id')}"
        f"{f' on {item.event_date.isoformat()}' if item.event_date else ''}: "
        f"shortage={shortage:g} {item.entities.get('unit', item.payload.get('unit', ''))}."
    )


class TrustedClaimRenderer:
    """Validate structured facts and render all user-visible factual claim text."""

    def render(
        self,
        claim: GroundedClaim,
        cited: list,
        decision: FinalDecisionPackage,
    ) -> _TrustedClaim:
        handler = getattr(self, f"_render_{claim.claim_type}", None)
        if handler is None:
            raise GroundingError(f"Unsupported grounded claim type: {claim.claim_type}")
        trusted = handler(cited, decision)
        if canonical_json(claim.facts) != canonical_json(trusted.facts):
            raise GroundingError(f"Claim facts differ from evidence: {claim.claim_type}")
        if set(claim.evidence_ids) != set(trusted.evidence_ids):
            raise GroundingError(f"Claim citations are not exact: {claim.claim_type}")
        if claim.uses_probability_language != trusted.uses_probability_language:
            raise GroundingError(f"Claim probability semantics differ: {claim.claim_type}")
        if claim.causal != trusted.causal:
            raise GroundingError(f"Claim causal semantics differ: {claim.claim_type}")
        if claim.text != trusted.text:
            raise GroundingError(
                f"Visible claim text differs from trusted rendering: {claim.claim_type}"
            )
        return trusted

    def _render_no_valid_plan(self, cited: list, decision: FinalDecisionPackage) -> _TrustedClaim:
        item = _require_one(cited, "recommendation")
        if item.payload.get("recommended_strategy") is not None:
            raise GroundingError("No-valid-plan claim conflicts with M5 recommendation.")
        return _TrustedClaim(
            text=(
                "M5 returned NO_VALID_PROCUREMENT_PLAN, so M6 presents no "
                "fallback strategy or immediate order."
            ),
            facts={"strategy": None, "status": item.payload.get("status")},
            evidence_ids=[item.evidence_id],
        )

    def _render_recommendation(self, cited: list, decision: FinalDecisionPackage) -> _TrustedClaim:
        item = _require_one(cited, "recommendation")
        strategy = item.payload.get("recommended_strategy")
        if strategy is None or strategy != decision.recommended_strategy:
            raise GroundingError("Recommendation claim does not match M5 authority.")
        rule = item.payload.get("recommendation_rule")
        status = item.payload.get("recommendation_rule_status")
        facts = {"strategy": strategy, "recommendation_rule_status": status}
        if rule is not None:
            facts["recommendation_rule"] = rule
        text = (
            f"M5 recommends {strategy}. This is inherited from the recorded rule {rule}, "
            "not from an M6 ranking."
            if rule is not None and status == "RECORDED"
            else f"M5 recommends {strategy}. The recommendation rule was not supplied in "
            "OptimizationResult.provenance; M6 does not reconstruct it."
        )
        return _TrustedClaim(text=text, facts=facts, evidence_ids=[item.evidence_id])

    def _render_critic_validation(
        self, cited: list, decision: FinalDecisionPackage
    ) -> _TrustedClaim:
        item = _require_one(cited, "critic_verdict")
        strategy = item.entities["strategy"]
        if not item.payload.get("passed") or strategy != decision.recommended_strategy:
            raise GroundingError("Critic validation claim is not for the recommended passing plan.")
        violations = list(item.payload.get("hard_violations", []))
        return _TrustedClaim(
            text=f"{strategy} passed the M5 critic; hard violations: {violations or 'none'}.",
            facts={"strategy": strategy, "passed": True},
            evidence_ids=[item.evidence_id],
        )

    def _render_critic_rejection(
        self, cited: list, decision: FinalDecisionPackage
    ) -> _TrustedClaim:
        item = _require_one(cited, "critic_verdict")
        strategy = item.entities["strategy"]
        passed = bool(item.payload.get("passed"))
        violations = list(item.payload.get("hard_violations", []))
        text = (
            f"{strategy} was not rejected by the M5 critic; it passed, although the "
            "recommendation rule may prefer another valid strategy."
            if passed
            else f"{strategy} was rejected by the M5 critic for: "
            f"{', '.join(violations) if violations else 'unspecified hard violation'}."
        )
        return _TrustedClaim(
            text=text,
            facts={"strategy": strategy, "passed": passed, "hard_violations": violations},
            evidence_ids=[item.evidence_id],
        )

    def _render_exact_inventory_validation(
        self, cited: list, decision: FinalDecisionPackage
    ) -> _TrustedClaim:
        item = _require_one(
            cited, "exact_simulation_package", "inventory_risk", "inventory_key_summary"
        )
        strategy = item.entities.get("strategy")
        if strategy != decision.recommended_strategy:
            raise GroundingError("Exact inventory claim is not for the recommended strategy.")
        return _TrustedClaim(
            text=f"The decision was evaluated by the exact M4 inventory layer: {item.text}",
            facts={"strategy": strategy, "authority": "exact_m4"},
            evidence_ids=[item.evidence_id],
            uses_probability_language=item.semantics == "probabilistic",
        )

    def _render_candidate_model_mismatch(
        self, cited: list, decision: FinalDecisionPackage
    ) -> _TrustedClaim:
        critic = _require_one(cited, "critic_verdict")
        if "CANDIDATE_MODEL_MISMATCH" not in critic.payload.get("hard_violations", []):
            raise GroundingError("Model-mismatch claim lacks the M5 violation.")
        mismatch = dict(critic.payload.get("details", {}).get("candidate_model_mismatch", {}))
        predicted_fill = mismatch.get("predicted_fill_rate")
        exact_fill = mismatch.get("simulated_expected_fill_rate")
        predicted_stockout = mismatch.get("predicted_stockout_probability")
        exact_stockout = mismatch.get("simulated_stockout_probability")
        values = []
        facts = {
            "strategy": critic.entities["strategy"],
            "predicted_fill_rate": predicted_fill,
            "exact_fill_rate": exact_fill,
        }
        expected_ids = [critic.evidence_id]
        if predicted_fill is not None and exact_fill is not None:
            values.append(
                f"expected fill {float(predicted_fill):g} versus exact M4 {float(exact_fill):g}"
            )
        risk_items = [item for item in cited if item.evidence_type == "inventory_risk"]
        uses_probability = bool(
            risk_items and predicted_stockout is not None and exact_stockout is not None
        )
        if uses_probability:
            if len(risk_items) != 1 or risk_items[0].entities.get(
                "strategy"
            ) != critic.entities.get("strategy"):
                raise GroundingError("Model-mismatch risk evidence is incompatible.")
            risk = risk_items[0]
            if not math.isclose(
                float(risk.payload["any_stockout_probability"]),
                float(exact_stockout),
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                raise GroundingError("Exact stockout mismatch value is not supported by M4.")
            values.append(
                f"stockout probability {float(predicted_stockout):g} versus exact M4 "
                f"{float(exact_stockout):g}"
            )
            facts.update(
                {
                    "predicted_stockout_probability": predicted_stockout,
                    "exact_stockout_probability": exact_stockout,
                }
            )
            expected_ids.append(risk.evidence_id)
        if not values:
            raise GroundingError("Model-mismatch evidence contains no renderable metrics.")
        return _TrustedClaim(
            text=(
                f"The solver surrogate predicted {'; '.join(values)}. M5 recorded that the "
                "accepted model-gap tolerance was exceeded; exact M4 remains the validation "
                "authority."
            ),
            facts=facts,
            evidence_ids=sorted(expected_ids),
            uses_probability_language=uses_probability,
        )

    def _render_immediate_order_authority(
        self, cited: list, decision: FinalDecisionPackage
    ) -> _TrustedClaim:
        item = _require_one(cited, "recommendation")
        strategy = item.payload.get("recommended_strategy")
        if strategy is None or strategy != decision.recommended_strategy:
            raise GroundingError("Immediate-order authority lacks an M5 recommendation.")
        return _TrustedClaim(
            text=f"Immediate orders must come only from {strategy}'s first-stage plan.",
            facts={"strategy": strategy},
            evidence_ids=[item.evidence_id],
        )

    def _render_no_immediate_order(
        self, cited: list, decision: FinalDecisionPackage
    ) -> _TrustedClaim:
        item = _require_one(cited, "procurement_plan")
        strategy = item.entities.get("strategy")
        if (
            strategy != decision.recommended_strategy
            or item.payload.get("first_stage_order_count") != 0
        ):
            raise GroundingError("No-immediate-order claim conflicts with the M5 plan.")
        return _TrustedClaim(
            text=f"{strategy}'s M5 plan contains no first-stage orders.",
            facts={"strategy": strategy, "first_stage_order_count": 0},
            evidence_ids=[item.evidence_id],
        )

    def _render_immediate_order(self, cited: list, decision: FinalDecisionPackage) -> _TrustedClaim:
        item = _require_one(cited, "first_stage_order")
        if item.evidence_id not in {order.evidence_id for order in decision.immediate_orders}:
            raise GroundingError("Generated order is outside the recommended M5 first-stage plan.")
        payload = item.payload
        strategy = item.entities.get("strategy")
        facts = {
            "strategy": strategy,
            "order_evidence_id": item.evidence_id,
            "offer_id": payload["offer_id"],
            "supplier_id": payload["supplier_id"],
            "store_id": payload["store_id"],
            "ingredient_id": payload["ingredient_id"],
            "order_date": str(payload["order_date"]),
            "arrival_date": str(payload["arrival_date"]),
            "order_quantity": payload["order_quantity"],
            "unit": payload["unit"],
        }
        return _TrustedClaim(
            text=(
                f"Order {float(payload['order_quantity']):g} {payload['unit']} of "
                f"{payload['ingredient_id']} from {payload['supplier_id']} on "
                f"{payload['order_date']}."
            ),
            facts=facts,
            evidence_ids=[item.evidence_id],
        )

    def _render_recipe_contribution(
        self, cited: list, decision: FinalDecisionPackage
    ) -> _TrustedClaim:
        item = _require_one(cited, "recipe_contribution", "scenario_recipe_contribution")
        payload = item.payload
        final = payload.get("contribution_p50", payload.get("final_quantity"))
        unit = payload.get("contribution_unit", payload.get("unit"))
        return _TrustedClaim(
            text=(
                f"Product {payload['product_id']} contributes {float(final):g} {unit} using "
                f"recipe {payload['recipe_id']}/{payload['recipe_version']}."
            ),
            facts={
                "ingredient_id": item.entities.get("ingredient_id"),
                "product_id": payload["product_id"],
                "recipe_id": payload["recipe_id"],
                "recipe_version": payload["recipe_version"],
                "final_quantity": final,
                "unit": unit,
            },
            evidence_ids=[item.evidence_id],
            uses_probability_language=item.semantics == "probabilistic",
        )

    def _render_evidence_unavailable(
        self, cited: list, decision: FinalDecisionPackage
    ) -> _TrustedClaim:
        item = _require_one(cited, "bom_evidence_availability")
        return _TrustedClaim(
            text=(
                "Ingredient totals may exist in M5, but product/recipe reasons are unavailable "
                "because M3 contribution evidence was not supplied."
            ),
            facts={"status": "UNAVAILABLE"},
            evidence_ids=[item.evidence_id],
        )

    def _render_stockout_location(
        self, cited: list, decision: FinalDecisionPackage
    ) -> _TrustedClaim:
        item = _require_one(
            cited,
            "inventory_daily_ledger",
            "inventory_key_summary",
            "stress_inventory_key",
        )
        shortage = float(item.payload["shortage_quantity"])
        if shortage <= 0:
            raise GroundingError("Stockout claim cites zero shortage.")
        return _TrustedClaim(
            text=_stockout_text(item, shortage),
            facts={
                "strategy": item.entities.get("strategy"),
                "scenario_id": item.entities.get("scenario_id"),
                "store_id": item.entities.get("store_id"),
                "ingredient_id": item.entities.get("ingredient_id"),
                "shortage": shortage,
                "unit": item.entities.get("unit", item.payload.get("unit", "")),
                "event_date": item.event_date.isoformat() if item.event_date is not None else None,
                "semantics": item.semantics,
            },
            evidence_ids=[item.evidence_id],
            uses_probability_language=item.semantics == "probabilistic",
        )

    def _render_probabilistic_inventory_risk(
        self, cited: list, decision: FinalDecisionPackage
    ) -> _TrustedClaim:
        item = _require_one(cited, "inventory_key_risk")
        if item.semantics != "probabilistic":
            raise GroundingError("Inventory probability claim lacks probabilistic evidence.")
        payload = item.payload
        return _TrustedClaim(
            text=(
                f"{item.entities['store_id']}/{item.entities['ingredient_id']} has stockout "
                f"probability {float(payload['stockout_probability']):g}, expected shortage "
                f"{float(payload['expected_shortage']):g} {item.entities['unit']}, and expected "
                f"fill rate {float(payload['expected_fill_rate']):g}; expected expiry is "
                f"{float(payload['expected_expired_quantity']):g} and expected explicit waste "
                f"is {float(payload['expected_explicit_waste']):g} {item.entities['unit']}."
            ),
            facts={
                "strategy": item.entities.get("strategy"),
                "store_id": item.entities["store_id"],
                "ingredient_id": item.entities["ingredient_id"],
                "unit": item.entities["unit"],
                "stockout_probability": payload["stockout_probability"],
                "expected_shortage": payload["expected_shortage"],
                "expected_fill_rate": payload["expected_fill_rate"],
                "expected_expired_quantity": payload["expected_expired_quantity"],
                "expected_explicit_waste": payload["expected_explicit_waste"],
            },
            evidence_ids=[item.evidence_id],
            uses_probability_language=True,
        )

    def _render_stress_definition(
        self, cited: list, decision: FinalDecisionPackage
    ) -> _TrustedClaim:
        item = _require_one(cited, "stress_definition")
        payload = item.payload
        return _TrustedClaim(
            text=(
                f"Adversarial stress scenario {item.entities['stress_id']} assumes demand "
                f"multiplier {float(payload['demand_multiplier']):g} and supplier delay "
                f"{int(payload['supplier_delay_days'])} day(s). It is not a probability estimate."
            ),
            facts={
                "stress_id": item.entities["stress_id"],
                "demand_multiplier": payload["demand_multiplier"],
                "supplier_delay_days": payload["supplier_delay_days"],
                "supplier_ids": sorted(payload.get("supplier_ids", [])),
                "probabilistic": False,
            },
            evidence_ids=[item.evidence_id],
        )

    def _render_stress_consequence(
        self, cited: list, decision: FinalDecisionPackage
    ) -> _TrustedClaim:
        item = _require_one(cited, "stress_inventory_key")
        payload = item.payload
        expired = float(payload.get("expired_quantity", 0))
        capacity = float(payload.get("capacity_violation_quantity") or 0)
        warning_text = ", ".join(item.warnings) if item.warnings else "none"
        return _TrustedClaim(
            text=(
                f"Under adversarial stress scenario {item.entities['scenario_id']}, "
                f"{item.entities['store_id']}/{item.entities['ingredient_id']} has shortage "
                f"{float(payload['shortage_quantity']):g} {item.entities['unit']} and fill rate "
                f"{float(payload['fill_rate']):g}; expiry={expired:g}, capacity consequence="
                f"{capacity:g}, warnings={warning_text}."
            ),
            facts={
                "stress_id": item.entities["scenario_id"],
                "store_id": item.entities["store_id"],
                "ingredient_id": item.entities["ingredient_id"],
                "unit": item.entities["unit"],
                "shortage": payload["shortage_quantity"],
                "fill_rate": payload["fill_rate"],
                "expired_quantity": expired,
                "capacity_violation_quantity": capacity,
                "warnings": item.warnings,
                "probabilistic": False,
            },
            evidence_ids=[item.evidence_id],
        )

    def _render_evidence_readiness(
        self, cited: list, decision: FinalDecisionPackage
    ) -> _TrustedClaim:
        dimensions = decision.confidence_decomposition.model_dump()
        matching = [
            (name, value)
            for name, value in dimensions.items()
            if isinstance(value, dict)
            and value.get("evidence_ids")
            and set(value["evidence_ids"]) == {item.evidence_id for item in cited}
        ]
        if len(matching) != 1:
            raise GroundingError("Readiness claim citations do not identify one dimension.")
        name, dimension = matching[0]
        return _TrustedClaim(
            text=f"{name}={dimension['status']}: {dimension['reason']}",
            facts={"dimension": name, "status": dimension["status"]},
            evidence_ids=sorted(dimension["evidence_ids"]),
        )

    def _render_retrieved_fact(self, cited: list, decision: FinalDecisionPackage) -> _TrustedClaim:
        if len(cited) != 1:
            raise GroundingError("A retrieved fact must cite exactly one evidence item.")
        item = cited[0]
        return _TrustedClaim(
            text=item.text,
            facts={"evidence_type": item.evidence_type},
            evidence_ids=[item.evidence_id],
            uses_probability_language=item.semantics == "probabilistic",
        )


class GroundingGuard:
    """Validate structured claims, then accept only byte-equivalent trusted rendering."""

    def __init__(self, renderer: TrustedClaimRenderer | None = None) -> None:
        self.renderer = renderer or TrustedClaimRenderer()

    def validate(
        self,
        answer: DecisionAnswer,
        decision: FinalDecisionPackage,
        retrieved: RetrievedEvidence | None = None,
    ) -> DecisionAnswer:
        evidence_by_id = {item.evidence_id: item for item in decision.evidence_package.items}
        referenced = {evidence_id for claim in answer.claims for evidence_id in claim.evidence_ids}
        unknown = (set(answer.citations) | referenced) - set(evidence_by_id)
        unknown.update(set(answer.retrieved_evidence_ids) - set(evidence_by_id))
        if unknown:
            raise GroundingError(f"Unknown evidence citation(s): {sorted(unknown)}")
        if set(answer.citations) != referenced:
            raise GroundingError("Answer citations must exactly cover claim citations.")
        if retrieved is not None:
            retrieved_ids = [item.evidence_id for item in retrieved.items]
            if answer.question != retrieved.query or answer.intent != retrieved.intent:
                raise GroundingError("Generator answer does not match the retrieved query/intent.")
            if answer.retrieved_evidence_ids != retrieved_ids:
                raise GroundingError("Generator changed the authoritative retrieval snapshot.")
            if not referenced <= set(retrieved_ids):
                raise GroundingError("A claim cites evidence outside the retrieval snapshot.")
        if answer.warnings:
            raise GroundingError("Generator-supplied free-text warnings are not trusted output.")

        if answer.status == "UNSUPPORTED_INTENT":
            if (
                answer.intent != "what_if"
                or answer.claims
                or answer.citations
                or answer.answer_text != "WHAT_IF_NOT_AVAILABLE_IN_M6_PART1"
                or answer.limitations != WHAT_IF_LIMITATIONS
            ):
                raise GroundingError("Unsupported intent response is not the trusted sentinel.")
            return answer.model_copy(
                update={
                    "provenance": {
                        **answer.provenance,
                        "grounding_guard": "trusted_renderer_v2",
                    }
                }
            )
        if answer.status == "INSUFFICIENT_EVIDENCE":
            if (
                answer.claims
                or answer.citations
                or answer.answer_text != "INSUFFICIENT_EVIDENCE"
                or answer.limitations != INSUFFICIENT_EVIDENCE_LIMITATIONS
            ):
                raise GroundingError("Insufficient-evidence response cannot assert visible facts.")
            return answer.model_copy(
                update={
                    "provenance": {
                        **answer.provenance,
                        "grounding_guard": "trusted_renderer_v2",
                    }
                }
            )
        if answer.status not in {"GROUNDED", "PARTIAL"} or not answer.claims:
            raise GroundingError("Grounded/partial answers require cited structured claims.")
        if answer.limitations:
            raise GroundingError(
                "Generator-supplied free-text limitations are not trusted grounded output."
            )

        trusted_claims = []
        for claim in answer.claims:
            cited = [evidence_by_id[evidence_id] for evidence_id in claim.evidence_ids]
            if _positive_probability_language(claim.text):
                if any(item.semantics in {"stress", "quantile"} for item in cited):
                    raise GroundingError(
                        "Stress/quantile evidence cannot support probability language."
                    )
                if not any(item.semantics == "probabilistic" for item in cited):
                    raise GroundingError("Probability language lacks probabilistic evidence.")
            forecast_causal = claim.causal or (
                any(item.evidence_type == "forecast_prediction" for item in cited)
                and _unsupported_forecast_causality(claim.text)
            )
            if forecast_causal and not any(
                item.evidence_type == "forecast_attribution" for item in cited
            ):
                raise GroundingError("Causal forecast claim lacks typed attribution evidence.")
            trusted = self.renderer.render(claim, cited, decision)
            trusted_claims.append(
                GroundedClaim(
                    claim_type=claim.claim_type,
                    text=trusted.text,
                    evidence_ids=trusted.evidence_ids,
                    facts=trusted.facts,
                    uses_probability_language=trusted.uses_probability_language,
                    causal=trusted.causal,
                )
            )

        trusted_text = _render(trusted_claims)
        if answer.answer_text != trusted_text:
            raise GroundingError("answer_text is not exactly composed from validated claims.")
        trusted_citations = sorted(
            {evidence_id for claim in trusted_claims for evidence_id in claim.evidence_ids}
        )
        return answer.model_copy(
            update={
                "claims": trusted_claims,
                "answer_text": trusted_text,
                "citations": trusted_citations,
                "provenance": {
                    **answer.provenance,
                    "grounding_guard": "trusted_renderer_v2",
                    "visible_text_source": "trusted_structured_claim_renderer",
                },
            }
        )
