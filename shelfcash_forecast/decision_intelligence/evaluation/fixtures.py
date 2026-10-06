from __future__ import annotations

import re
from typing import Any

from shelfcash_forecast.decision_intelligence.contracts import FinalDecisionPackage
from shelfcash_forecast.decision_intelligence.evidence import EvidenceCollector
from shelfcash_forecast.decision_intelligence.graph import build_decision_graph

SCALABLE_EVIDENCE_TYPES = {
    "forecast_prediction",
    "ingredient_demand",
    "ingredient_demand_scenario",
    "inventory_daily_ledger",
    "inventory_demand_scenario",
    "inventory_key_risk",
    "inventory_key_summary",
    "inventory_scenario_result",
    "lot_consumption",
    "lot_expiry",
    "lot_waste",
    "product_demand_scenario",
    "recipe_contribution",
    "scenario_recipe_contribution",
}


def _replace_values(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {key: _replace_values(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_values(item, replacements) for item in value]
    if isinstance(value, str):
        output = value
        for source, target in sorted(
            replacements.items(), key=lambda pair: (-len(pair[0]), pair[0])
        ):
            output = re.sub(
                rf"(?<![\w]){re.escape(source)}(?![\w])",
                lambda _, replacement=target: replacement,
                output,
                flags=re.UNICODE,
            )
        return output
    return value


def build_scaled_decision_fixture(
    decision: FinalDecisionPackage,
    *,
    replicas: int = 12,
) -> FinalDecisionPackage:
    """Expand a materialized package for offline scale tests without M1-M5 computation."""

    if replicas < 1:
        raise ValueError("Scale replicas must be positive.")
    collector = EvidenceCollector(decision.request_id)
    for item in decision.evidence_package.items:
        collector.add(
            layer=item.layer,
            evidence_type=item.evidence_type,
            source_object=item.source_object,
            source_path=item.source_path,
            semantics=item.semantics,
            entities=item.entities,
            event_date=item.event_date,
            payload=item.payload,
            text=item.text,
            warnings=item.warnings,
        )
    scalable = [
        item
        for item in decision.evidence_package.items
        if item.evidence_type in SCALABLE_EVIDENCE_TYPES
    ]
    for replica in range(1, replicas + 1):
        suffix = f"{replica:02d}"
        replacements = {
            "STORE_A": f"STORE_{suffix}",
            "MILK": f"INGREDIENT_{suffix}",
            "LATTE": f"PRODUCT_{suffix}",
            "LOW": f"SCALE_LOW_{suffix}",
            "HIGH": f"SCALE_HIGH_{suffix}",
            "LOT_MAIN": f"LOT_MAIN_{suffix}",
            "LOT_EXPIRING": f"LOT_EXPIRING_{suffix}",
        }
        for item in scalable:
            collector.add(
                layer=item.layer,
                evidence_type=item.evidence_type,
                source_object=item.source_object,
                source_path=f"{item.source_path}.scale[{suffix}]",
                semantics=item.semantics,
                entities=_replace_values(item.entities, replacements),
                event_date=item.event_date,
                payload=_replace_values(item.payload, replacements),
                text=_replace_values(item.text, replacements),
                warnings=item.warnings,
            )
    evidence = collector.package()
    graph = build_decision_graph(evidence)
    payload = decision.model_dump(mode="json")
    payload["evidence_package"] = evidence.model_dump(mode="json")
    payload["decision_graph"] = graph.model_dump(mode="json")
    payload["provenance"] = {
        **payload["provenance"],
        "benchmark_scale_fixture": True,
        "benchmark_scale_replicas": replicas,
    }
    return FinalDecisionPackage.model_validate(payload)


def build_quantile_decision_fixture(
    decision: FinalDecisionPackage,
) -> FinalDecisionPackage:
    """Create a non-probabilistic quantile snapshot from already materialized evidence."""

    collector = EvidenceCollector(decision.request_id)
    weighted_risk_types = {"inventory_risk", "inventory_key_risk"}
    for item in decision.evidence_package.items:
        if item.evidence_type in weighted_risk_types:
            continue
        semantics = (
            "quantile"
            if item.layer == "M4" and item.semantics == "probabilistic"
            else item.semantics
        )
        payload = dict(item.payload)
        text = item.text
        if item.evidence_type in {
            "inventory_demand_scenario",
            "inventory_scenario_result",
        }:
            payload["probability_weight"] = None
        if item.evidence_type == "exact_simulation_package":
            payload["risk_metrics_available"] = False
            text = (
                f"Exact M4 simulation for {item.entities.get('strategy')} contains "
                f"{payload.get('scenario_count', 0)} quantile scenario result(s); "
                "weighted risk metrics are unavailable."
            )
        collector.add(
            layer=item.layer,
            evidence_type=item.evidence_type,
            source_object=item.source_object,
            source_path=item.source_path,
            semantics=semantics,
            entities=item.entities,
            event_date=item.event_date,
            payload=payload,
            text=text,
            warnings=item.warnings,
        )
    evidence = collector.package()
    graph = build_decision_graph(evidence)
    payload = decision.model_dump(mode="json")
    payload["evidence_package"] = evidence.model_dump(mode="json")
    payload["decision_graph"] = graph.model_dump(mode="json")
    payload["narrative_summary"] = None
    payload["inventory_risk_explanations"] = []
    for candidate in payload["strategy_comparison"]:
        candidate["exact_stockout_probability"] = None
    if payload["recommended_plan_summary"] is not None:
        payload["recommended_plan_summary"]["exact_stockout_probability"] = None
    payload["provenance"] = {
        **payload["provenance"],
        "benchmark_quantile_fixture": True,
        "probabilistic": False,
    }
    return FinalDecisionPackage.model_validate(payload)
