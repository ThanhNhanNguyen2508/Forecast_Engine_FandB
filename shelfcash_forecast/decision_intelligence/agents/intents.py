from __future__ import annotations

import re

from shelfcash_forecast.decision_intelligence.agents.contracts import AgentIntent
from shelfcash_forecast.decision_intelligence.contracts import FinalDecisionPackage
from shelfcash_forecast.decision_intelligence.retrieval import boundary_entity_match
from shelfcash_forecast.decision_intelligence.what_if.contracts import (
    DemandScaleModification,
    DemandSelector,
    WhatIfDraft,
)
from shelfcash_forecast.decision_intelligence.what_if.service import draft_what_if
from shelfcash_forecast.optimization.contracts import OptimizationRequest


class IntentNormalizeAgent:
    def normalize(self, question: str) -> AgentIntent:
        text = question.casefold()
        if any(term in text for term in ("approve", "reject", "phê duyệt", "từ chối")):
            return "APPROVAL"
        if any(term in text for term in ("regret", "hối tiếc")):
            return "REGRET"
        if any(term in text for term in ("counterfactual", "điều kiện nào", "tối thiểu để")):
            return "COUNTERFACTUAL"
        if any(term in text for term in ("compare", "so sánh", "changed", "khác gì")):
            return "COMPARISON"
        if any(
            term in text
            for term in (
                "what if",
                "if demand",
                "nếu demand",
                "nếu nhu cầu",
                "giả sử",
            )
        ):
            return "WHAT_IF_DRAFT"
        return "READ_ONLY_EXPLANATION"


def _unique_match(question: str, values: set[str]) -> str | None:
    matches = sorted(
        (value for value in values if boundary_entity_match(question, value)),
        key=lambda value: (-len(value), value),
    )
    return matches[0] if len(matches) == 1 else None


class ScenarioWhatIfAgent:
    """Natural language is permitted to create a draft, never an execution request."""

    def draft(
        self,
        question: str,
        baseline_request: OptimizationRequest,
        baseline_decision: FinalDecisionPackage,
        *,
        actor: str,
        reason: str,
        idempotency_key: str,
    ) -> WhatIfDraft:
        normalized_question = question.casefold()
        unsupported = [
            label
            for label, terms in (
                ("FORECAST_RECOMPUTATION", ("forecast", "dự báo")),
                ("RECIPE_BOM_MUTATION", ("recipe", "bom", "công thức")),
                ("PRODUCT_LEVEL_DEMAND", ("product", "sản phẩm")),
            )
            if any(term in normalized_question for term in terms)
        ]
        if unsupported:
            return WhatIfDraft(
                status="NOT_SUPPORTED",
                unsupported_fields=unsupported,
                confirmation_required=True,
                warnings=["NOT_SUPPORTED_AT_CURRENT_AUTHORITY_BOUNDARY"],
            )
        stores = {
            line.store_id
            for scenario in baseline_request.demand_scenarios
            for line in scenario.lines
        }
        ingredients = {
            line.ingredient_id
            for scenario in baseline_request.demand_scenarios
            for line in scenario.lines
        }
        store = _unique_match(question, stores)
        ingredient = _unique_match(question, ingredients)
        percentage = re.search(
            r"(tăng|increase(?:s|d)?|decrease(?:s|d)?|giảm)\s*(\d+(?:\.\d+)?)\s*%",
            normalized_question,
        )
        ambiguities = []
        if store is None:
            ambiguities.append("EXACT_STORE_REQUIRED")
        if ingredient is None:
            ambiguities.append("EXACT_INGREDIENT_REQUIRED")
        if percentage is None:
            ambiguities.append("DEMAND_PERCENTAGE_REQUIRED")
        if ambiguities:
            return WhatIfDraft(
                status="NEEDS_CLARIFICATION",
                ambiguities=ambiguities,
                confirmation_required=True,
                warnings=["Natural language was not converted into executable computation."],
            )
        direction, number = percentage.groups()
        fraction = float(number) / 100
        multiplier = (
            1 - fraction
            if direction.startswith("decrease") or direction == "giảm"
            else 1 + fraction
        )
        matching = [
            line
            for scenario in baseline_request.demand_scenarios
            for line in scenario.lines
            if line.store_id == store and line.ingredient_id == ingredient
        ]
        units = {line.unit for line in matching}
        if len(units) != 1:
            return WhatIfDraft(
                status="NEEDS_CLARIFICATION",
                ambiguities=["EXACT_UNIT_REQUIRED"],
                confirmation_required=True,
            )
        modification = DemandScaleModification(
            selector=DemandSelector(
                store_id=store,
                ingredient_id=ingredient,
                unit=next(iter(units)),
                expected_matches=len(matching),
            ),
            multiplier=multiplier,
        )
        return draft_what_if(
            baseline_request,
            baseline_decision,
            [modification],
            actor=actor,
            reason=reason,
            idempotency_key=idempotency_key,
        )


__all__ = ["IntentNormalizeAgent", "ScenarioWhatIfAgent"]
