"""Bounded deterministic ingredient-demand rules; parsing never grants execution."""
from __future__ import annotations

import re
from collections import Counter
from typing import Literal
from pydantic import Field, StrictFloat
from shelfcash_forecast.decision_intelligence.contracts import StrictDecisionContract
from shelfcash_forecast.decision_intelligence.what_if.contracts import DemandScaleModification, DemandSelector
from shelfcash_forecast.optimization.contracts import OptimizationRequest


class IngredientDemandRule(StrictDecisionContract):
    operation: Literal["DEMAND_SCALE"] = "DEMAND_SCALE"
    multiplier: StrictFloat = Field(gt=0)
    scope: Literal["ALL_BASELINE_INGREDIENT_DEMAND"]


class DemandRuleParse(StrictDecisionContract):
    status: Literal["PARSED", "NEEDS_CLARIFICATION", "NOT_SUPPORTED", "INVALID"]
    rule: IngredientDemandRule | None = None
    reason_codes: list[str] = Field(default_factory=list)


_NUMBER = r"[+-]?(?:\d+(?:[.,]\d+)?|nan|inf(?:inity)?)"
_GLOBAL = "nhu cầu toàn bộ nguyên liệu trong cả kỳ kế hoạch"


def parse_demand_rule(text: str, *, scope: str | None = None) -> DemandRuleParse:
    """Accept three explicit Vietnamese forms, or demand=ratio with configured scope."""
    value = re.sub(r"\s+", " ", text.casefold().strip()).rstrip(".")
    value = re.sub(r"^what[ -]if\s*:\s*", "", value)
    if scope is not None and scope != "ALL_BASELINE_INGREDIENT_DEMAND":
        return DemandRuleParse(status="INVALID", reason_codes=["INVALID_DEMAND_SCOPE"])
    explicit = _GLOBAL in value
    if not explicit and scope is None:
        return DemandRuleParse(status="NEEDS_CLARIFICATION", reason_codes=["EXPLICIT_GLOBAL_SCOPE_REQUIRED"])
    patterns = [
        (rf"{_GLOBAL} bằng ({_NUMBER}) lần baseline", "ratio"),
        (rf"{_GLOBAL} nhân ({_NUMBER})", "ratio"),
        (rf"tăng {_GLOBAL} ({_NUMBER})\s*%", "increase"),
        (rf"giảm {_GLOBAL} ({_NUMBER})\s*%", "decrease"),
    ]
    if scope is not None:
        patterns.append((rf"demand\s*=\s*({_NUMBER})", "ratio"))
    for pattern, kind in patterns:
        match = re.fullmatch(pattern, value)
        if match:
            number = float(match.group(1).replace(",", "."))
            multiplier = 1 + number / 100 if kind == "increase" else 1 - number / 100 if kind == "decrease" else number
            try:
                rule = IngredientDemandRule(multiplier=multiplier, scope="ALL_BASELINE_INGREDIENT_DEMAND")
            except ValueError:
                return DemandRuleParse(status="INVALID", reason_codes=["FINITE_POSITIVE_MULTIPLIER_REQUIRED"])
            if kind != "ratio" and number < 0:
                return DemandRuleParse(status="INVALID", reason_codes=["NONNEGATIVE_PERCENTAGE_REQUIRED"])
            return DemandRuleParse(status="PARSED", rule=rule)
    if re.search(r"tăng|giảm", value) and "%" not in value and "lần" not in value:
        return DemandRuleParse(status="NEEDS_CLARIFICATION", reason_codes=["PERCENT_OR_MULTIPLIER_UNIT_REQUIRED"])
    return DemandRuleParse(status="NOT_SUPPORTED", reason_codes=["UNSUPPORTED_RULE_TEXT_OR_SCOPE_QUALIFIER"])


def expand_demand_rule(rule: IngredientDemandRule, baseline: OptimizationRequest) -> list[DemandScaleModification]:
    """Partition demand into disjoint store/ingredient/unit selectors with observed counts."""
    counts = Counter((l.store_id, l.ingredient_id, l.unit) for w in baseline.demand_scenarios for l in w.lines)
    full = baseline.evaluation_scenarios or baseline.demand_scenarios
    full_keys = {(l.store_id, l.ingredient_id, l.unit) for w in full for l in w.lines}
    if not counts or set(counts) != full_keys:
        raise ValueError("GLOBAL_DEMAND_KEYS_REQUIRE_OPTIMIZATION_AND_EVALUATION_COVERAGE")
    return [DemandScaleModification(selector=DemandSelector(store_id=store, ingredient_id=ingredient,
            unit=unit, expected_matches=count), multiplier=rule.multiplier)
            for (store, ingredient, unit), count in sorted(counts.items())]


def demand_rule_summary(rule: IngredientDemandRule, baseline: OptimizationRequest) -> dict:
    full = baseline.evaluation_scenarios or baseline.demand_scenarios
    keys = sorted({(l.store_id, l.ingredient_id, l.unit) for w in full for l in w.lines})
    dates = sorted({l.target_date.isoformat() for w in full for l in w.lines})
    return {**rule.model_dump(mode="json"), "percentage_change": (rule.multiplier - 1) * 100,
        "keys": [{"store_id": s, "ingredient_id": i, "unit": u} for s, i, u in keys],
        "dates": dates, "world_count": len(full), "optimization_world_count": len(baseline.demand_scenarios),
        "matched_optimization_lines": sum(len(w.lines) for w in baseline.demand_scenarios),
        "matched_evaluation_lines": sum(len(w.lines) for w in full), "solver_called": False,
        "execution_authorized": False}
