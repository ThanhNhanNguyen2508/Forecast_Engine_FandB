# Nó biến nhiều object khác nhau từ M1→M5 thành một format chung:

# EvidenceItem(
#     evidence_id=...,
#     layer="M4",
#     evidence_type="inventory_key_summary",
#     source_object="InventoryKeySummary",
#     source_path="optimization_result.evaluations.BALANCED...",
#     semantics="probabilistic",
#     entities={
#         "strategy": "BALANCED",
#         "scenario_id": "HIGH",
#         "store_id": "STORE_A",
#         "ingredient_id": "MILK",
#     },
#     payload={...},
#     text="...",
#     warnings=[...],
# )

# Mỗi evidence item có:

# nó đến từ milestone nào;
# thuộc loại evidence gì;
# object gốc nào;
# source path nào;
# entities liên quan;
# semantics;
# payload số liệu gốc;
# text ngắn để retrieval;
# warnings;
# evidence ID.

# Kết quả:

# EvidencePackage

# Đây chính là knowledge base cục bộ của một quyết định.
from __future__ import annotations

import hashlib
import json
import re
from datetime import date, datetime
from typing import Any

from pydantic import BaseModel

from shelfcash_forecast.decision_intelligence.contracts import (
    DecisionIntelligenceInput,
    EvidenceItem,
    EvidenceLayer,
    EvidencePackage,
    EvidenceSemantics,
)
from shelfcash_forecast.decision_intelligence.integrity import canonical_json
from shelfcash_forecast.inventory.contracts import (
    InventorySimulationPackage,
    InventorySimulationResult,
)
from shelfcash_forecast.optimization.strategies import default_strategy_profiles


def _json_default(value: object) -> object:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Unsupported evidence identity value: {type(value)!r}")


def stable_evidence_id(
    layer: EvidenceLayer,
    evidence_type: str,
    source_path: str,
    entities: dict[str, str] | None = None,
) -> str:
    """Return a stable locator reference, not a content-integrity digest."""

    identity = {
        "layer": layer,
        "evidence_type": evidence_type,
        "source_path": source_path,
        "entities": dict(sorted((entities or {}).items())),
    }
    encoded = json.dumps(
        identity,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()[:16]
    slug = re.sub(r"[^a-z0-9]+", "-", evidence_type.lower()).strip("-")[:32]
    return f"ev-{layer.lower()}-{slug}-{digest}"


class EvidenceCollector:
    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        self._items: dict[str, EvidenceItem] = {}

    def add(
        self,
        *,
        layer: EvidenceLayer,
        evidence_type: str,
        source_object: str,
        source_path: str,
        semantics: EvidenceSemantics,
        payload: dict[str, Any],
        text: str,
        entities: dict[str, str] | None = None,
        event_date: date | None = None,
        warnings: list[str] | None = None,
    ) -> EvidenceItem:
        normalized_entities = dict(sorted((entities or {}).items()))
        evidence_id = stable_evidence_id(layer, evidence_type, source_path, normalized_entities)
        item = EvidenceItem(
            evidence_id=evidence_id,
            layer=layer,
            evidence_type=evidence_type,
            source_object=source_object,
            source_path=source_path,
            semantics=semantics,
            entities=normalized_entities,
            event_date=event_date,
            payload=payload,
            text=text,
            warnings=sorted(set(warnings or [])),
        )
        existing = self._items.get(evidence_id)
        if existing is not None and existing != item:
            raise ValueError(f"EVIDENCE_LOCATOR_COLLISION:{evidence_id}")
        self._items[evidence_id] = item
        return item

    def package(self) -> EvidencePackage:
        items = sorted(self._items.values(), key=lambda item: item.evidence_id)
        return EvidencePackage(
            request_id=self.request_id,
            items=items,
            source_layers=sorted({item.layer for item in items}),
            provenance={
                "builder": "shelfcash_decision_evidence_v2",
                "locator_identity": (
                    "truncated_sha256_64bit_reference(layer,type,source_path,entities)"
                ),
                "content_integrity": "full_sha256_over_canonical_material_evidence",
                "package_integrity": "full_sha256_over_ordered_evidence_content_hashes",
                "read_only": True,
            },
        )


def _model_sort_key(value: BaseModel, *parts: object) -> tuple[str, ...]:
    return (*[str(part) for part in parts], canonical_json(value.model_dump(mode="json")))


def _add_coherence_evidence(
    collector: EvidenceCollector, inputs: DecisionIntelligenceInput
) -> None:
    coherence = inputs.coherence
    if coherence is None:
        raise ValueError("M6_COHERENCE_RESULT_MISSING")
    issue_codes = [issue.code for issue in coherence.issues]
    collector.add(
        layer="M6",
        evidence_type="artifact_coherence",
        source_object="ArtifactCoherenceResult",
        source_path="decision_intelligence_input.coherence",
        semantics="deterministic",
        entities={"request_id": inputs.optimization_request.request_id},
        payload=coherence.model_dump(mode="json"),
        text=(
            f"M1-M5 artifact coherence status is {coherence.status}; "
            f"issues={issue_codes or 'none'}."
        ),
        warnings=issue_codes,
    )


def _scenario_semantics(inputs: DecisionIntelligenceInput) -> EvidenceSemantics:
    scenarios = inputs.optimization_request.demand_scenarios
    if scenarios and all(item.probability_weight is not None for item in scenarios):
        return "probabilistic"
    quantile_markers = {str(item.provenance.get("quantile_field", "")) for item in scenarios}
    if quantile_markers & {"p25", "p50", "p75"} or any(
        item.provenance.get("quantile_is_probability") is False for item in scenarios
    ):
        return "quantile"
    return "deterministic"


def _add_forecast_evidence(collector: EvidenceCollector, inputs: DecisionIntelligenceInput) -> None:
    package = inputs.forecast_package
    if package is None:
        collector.add(
            layer="M6",
            evidence_type="forecast_evidence_availability",
            source_object="DecisionIntelligenceInput",
            source_path="forecast_package",
            semantics="deterministic",
            payload={"status": "UNAVAILABLE"},
            text="Forecast explanation evidence was not supplied to M6.",
        )
        return
    predictions = sorted(
        package.predictions,
        key=lambda item: _model_sort_key(item, item.target_date, item.store_id, item.product_id),
    )
    for index, prediction in enumerate(predictions):
        path = f"forecast_package.predictions[{index}]"
        entities = {
            "store_id": prediction.store_id,
            "product_id": prediction.product_id,
            "target_date": prediction.target_date.isoformat(),
        }
        collector.add(
            layer="M2",
            evidence_type="forecast_prediction",
            source_object="ForecastPrediction",
            source_path=path,
            semantics="quantile",
            entities=entities,
            event_date=prediction.target_date,
            payload={
                **prediction.model_dump(mode="json"),
                "model_version": package.model_version,
                "quantile_spread": prediction.p75 - prediction.p25,
                "interval_width": prediction.interval_upper - prediction.interval_lower,
                "delta_vs_baseline": prediction.p50 - prediction.baseline_p50,
                "derived_display_metrics": [
                    "quantile_spread",
                    "interval_width",
                    "delta_vs_baseline",
                ],
                "feature_attribution_available": False,
            },
            text=(
                f"Forecast {prediction.store_id}/{prediction.product_id} on "
                f"{prediction.target_date.isoformat()}: P50={prediction.p50:g}, "
                f"P25-P75={prediction.p25:g}-{prediction.p75:g}, calibrated interval="
                f"{prediction.interval_lower:g}-{prediction.interval_upper:g}, baseline="
                f"{prediction.baseline_p50:g}; causal feature attribution is unavailable."
            ),
            warnings=sorted({*package.warnings, *prediction.warnings}),
        )


def _add_deterministic_bom_evidence(
    collector: EvidenceCollector, inputs: DecisionIntelligenceInput
) -> None:
    package = inputs.ingredient_demand_package
    if package is None:
        collector.add(
            layer="M6",
            evidence_type="bom_evidence_availability",
            source_object="DecisionIntelligenceInput",
            source_path="ingredient_demand_package",
            semantics="deterministic",
            payload={"status": "UNAVAILABLE"},
            text="Deterministic Recipe/BOM contribution evidence was not supplied to M6.",
        )
        return
    predictions = sorted(
        package.predictions,
        key=lambda item: _model_sort_key(item, item.target_date, item.store_id, item.ingredient_id),
    )
    for index, prediction in enumerate(predictions):
        path = f"ingredient_demand_package.predictions[{index}]"
        entities = {
            "store_id": prediction.store_id,
            "ingredient_id": prediction.ingredient_id,
            "target_date": prediction.target_date.isoformat(),
        }
        collector.add(
            layer="M3",
            evidence_type="ingredient_demand",
            source_object="IngredientDemandPrediction",
            source_path=path,
            semantics="quantile",
            entities=entities,
            event_date=prediction.target_date,
            payload={
                **prediction.model_dump(mode="json", exclude={"sources"}),
                "forecast_model_version": package.forecast_model_version,
                "package_complete": package.is_complete,
            },
            text=(
                f"Ingredient demand {prediction.store_id}/{prediction.ingredient_id} "
                f"on {prediction.target_date.isoformat()} is P25={prediction.p25:g}, "
                f"P50={prediction.p50:g}, P75={prediction.p75:g} {prediction.unit}."
            ),
            warnings=sorted({*package.warnings, *prediction.warnings}),
        )
        sources = sorted(
            prediction.sources,
            key=lambda item: _model_sort_key(
                item, item.product_id, item.recipe_id, item.recipe_version
            ),
        )
        for source_index, source in enumerate(sources):
            source_path = f"{path}.sources[{source_index}]"
            source_entities = {
                **entities,
                "product_id": source.product_id,
                "recipe_id": source.recipe_id,
                "recipe_version": source.recipe_version,
            }
            collector.add(
                layer="M3",
                evidence_type="recipe_contribution",
                source_object="IngredientDemandSource",
                source_path=source_path,
                semantics="quantile",
                entities=source_entities,
                event_date=prediction.target_date,
                payload=source.model_dump(mode="json"),
                text=(
                    f"Product {source.product_id} using recipe {source.recipe_id}/"
                    f"{source.recipe_version} contributes P50={source.contribution_p50:g} "
                    f"{source.contribution_unit} to ingredient {prediction.ingredient_id} "
                    f"after materialized yield/loss allowances."
                ),
            )
    issues = sorted(
        package.issues,
        key=lambda item: _model_sort_key(item, item.code, item.message),
    )
    for index, issue in enumerate(issues):
        collector.add(
            layer="M3",
            evidence_type="bom_issue",
            source_object="BOMIssue",
            source_path=f"ingredient_demand_package.issues[{index}]",
            semantics="deterministic",
            entities={"issue_code": issue.code},
            payload=issue.model_dump(mode="json"),
            text=f"BOM issue {issue.code}: {issue.message}",
        )


def _add_scenario_evidence(collector: EvidenceCollector, inputs: DecisionIntelligenceInput) -> None:
    product_bundle = inputs.product_scenario_bundle
    if product_bundle is not None:
        product_scenarios = sorted(
            product_bundle.scenarios,
            key=lambda item: _model_sort_key(item, item.scenario_id),
        )
        for scenario_index, scenario in enumerate(product_scenarios):
            lines = sorted(
                scenario.lines,
                key=lambda item: _model_sort_key(
                    item, item.target_date, item.store_id, item.product_id
                ),
            )
            for line_index, line in enumerate(lines):
                collector.add(
                    layer="M3",
                    evidence_type="product_demand_scenario",
                    source_object="ProductDemandScenarioLine",
                    source_path=(
                        f"product_scenario_bundle.scenarios[{scenario_index}].lines[{line_index}]"
                    ),
                    semantics="probabilistic",
                    entities={
                        "scenario_id": scenario.scenario_id,
                        "store_id": line.store_id,
                        "product_id": line.product_id,
                        "target_date": line.target_date.isoformat(),
                    },
                    event_date=line.target_date,
                    payload={
                        **line.model_dump(mode="json"),
                        "probability_weight": scenario.probability_weight,
                    },
                    text=(
                        f"Probabilistic product scenario {scenario.scenario_id} assigns "
                        f"demand {line.demand_quantity:g} to {line.store_id}/"
                        f"{line.product_id} on {line.target_date.isoformat()} with explicit "
                        f"weight {scenario.probability_weight:g}."
                    ),
                )

    ingredient_bundle = inputs.ingredient_scenario_bundle
    if ingredient_bundle is None:
        return
    ingredient_scenarios = sorted(
        ingredient_bundle.scenarios,
        key=lambda item: _model_sort_key(item, item.scenario_id),
    )
    for scenario_index, scenario in enumerate(ingredient_scenarios):
        lines = sorted(
            scenario.lines,
            key=lambda item: _model_sort_key(
                item, item.target_date, item.store_id, item.ingredient_id
            ),
        )
        for line_index, line in enumerate(lines):
            path = f"ingredient_scenario_bundle.scenarios[{scenario_index}].lines[{line_index}]"
            entities = {
                "scenario_id": scenario.scenario_id,
                "store_id": line.store_id,
                "ingredient_id": line.ingredient_id,
                "target_date": line.target_date.isoformat(),
            }
            collector.add(
                layer="M3",
                evidence_type="ingredient_demand_scenario",
                source_object="IngredientDemandScenarioLine",
                source_path=path,
                semantics="probabilistic",
                entities=entities,
                event_date=line.target_date,
                payload={
                    **line.model_dump(mode="json", exclude={"contributions"}),
                    "probability_weight": scenario.probability_weight,
                    "scenario_method": ingredient_bundle.scenario_method,
                },
                text=(
                    f"Probabilistic scenario {scenario.scenario_id} requires "
                    f"{line.quantity:g} {line.unit} of ingredient {line.ingredient_id} "
                    f"at {line.store_id} on {line.target_date.isoformat()}."
                ),
                warnings=sorted(
                    {
                        *ingredient_bundle.warnings,
                        *scenario.warnings,
                        *line.warnings,
                    }
                ),
            )
            contributions = sorted(
                line.contributions,
                key=lambda item: _model_sort_key(
                    item, item.product_id, item.recipe_id, item.recipe_version
                ),
            )
            for contribution_index, contribution in enumerate(contributions):
                collector.add(
                    layer="M3",
                    evidence_type="scenario_recipe_contribution",
                    source_object="IngredientScenarioContribution",
                    source_path=f"{path}.contributions[{contribution_index}]",
                    semantics="probabilistic",
                    entities={
                        **entities,
                        "product_id": contribution.product_id,
                        "recipe_id": contribution.recipe_id,
                        "recipe_version": contribution.recipe_version,
                    },
                    event_date=line.target_date,
                    payload={
                        **contribution.model_dump(mode="json"),
                        "probability_weight": scenario.probability_weight,
                    },
                    text=(
                        f"In scenario {scenario.scenario_id}, product "
                        f"{contribution.product_id} via recipe {contribution.recipe_id}/"
                        f"{contribution.recipe_version} contributes "
                        f"{contribution.final_quantity:g} {contribution.unit} to "
                        f"ingredient {line.ingredient_id}."
                    ),
                )


def _add_request_evidence(collector: EvidenceCollector, inputs: DecisionIntelligenceInput) -> None:
    request = inputs.optimization_request
    semantics = _scenario_semantics(inputs)
    collector.add(
        layer="M5",
        evidence_type="decision_request",
        source_object="OptimizationRequest",
        source_path="optimization_request",
        semantics="deterministic",
        entities={"request_id": request.request_id},
        event_date=request.decision_date,
        payload={
            "request_id": request.request_id,
            "decision_date": request.decision_date.isoformat(),
            "planning_end_date": request.planning_end_date.isoformat(),
            "stochastic_requested": request.stochastic,
            "scenario_count": len(request.demand_scenarios),
            "scenario_semantics": semantics,
            "unknown_constraints": sorted(request.unknown_constraints),
        },
        text=(
            f"Optimization request {request.request_id} covers "
            f"{request.decision_date.isoformat()} through "
            f"{request.planning_end_date.isoformat()} with {len(request.demand_scenarios)} "
            f"{semantics} demand scenario(s)."
        ),
    )
    demand_scenarios = sorted(
        request.demand_scenarios,
        key=lambda item: _model_sort_key(item, item.scenario_id),
    )
    for index, scenario in enumerate(demand_scenarios):
        collector.add(
            layer="M4",
            evidence_type="inventory_demand_scenario",
            source_object="InventoryDemandScenario",
            source_path=f"optimization_request.demand_scenarios[{index}]",
            semantics=semantics,
            entities={"scenario_id": scenario.scenario_id},
            payload={
                "scenario_id": scenario.scenario_id,
                "probability_weight": scenario.probability_weight,
                "simulation_start_date": scenario.simulation_start_date,
                "simulation_end_date": scenario.simulation_end_date,
                "provenance": scenario.provenance,
                "line_count": len(scenario.lines),
            },
            text=(
                f"Inventory demand scenario {scenario.scenario_id} is labeled "
                f"{semantics}; probability weight="
                f"{scenario.probability_weight if scenario.probability_weight is not None else 'not provided'}."
            ),
            warnings=scenario.warnings,
        )

    profiles = {profile.name: profile for profile in default_strategy_profiles()}
    supplied_profiles = {profile.name: profile for profile in request.strategy_profiles}
    profiles.update(supplied_profiles)
    for name in ("LEAN", "BALANCED", "PROTECTED"):
        profile = profiles[name]
        supplied = supplied_profiles.get(name)
        profile_source = (
            f"optimization_request.strategy_profiles.{name}"
            if supplied is not None
            else f"optimization.default_strategy_profiles.{name}"
        )
        collector.add(
            layer="M5",
            evidence_type="strategy_profile",
            source_object="StrategyProfile",
            source_path=profile_source,
            semantics="deterministic",
            entities={"strategy": name},
            payload={
                **profile.model_dump(mode="json"),
                "profile_source": (
                    "verified_decision_input"
                    if supplied is not None
                    else "reconstructed_current_default"
                ),
                "decision_time_verified": supplied is not None,
            },
            text=(
                f"Strategy {name} {'decision-input profile' if supplied is not None else 'reconstructed current default'} "
                f"uses shortage penalty {profile.shortage_penalty:g}, "
                f"holding penalty {profile.holding_penalty:g}, cash penalty "
                f"{profile.cash_penalty:g}, and CVaR weight {profile.cvar_weight:g}; "
                f"decision-time verification={'available' if supplied is not None else 'unavailable'}."
            ),
        )
    stress_scenarios = sorted(
        request.stress_scenarios,
        key=lambda item: _model_sort_key(item, item.stress_id),
    )
    for definition in stress_scenarios:
        collector.add(
            layer="M4",
            evidence_type="stress_definition",
            source_object="StressScenarioDefinition",
            source_path=f"optimization_request.stress_scenarios.{definition.stress_id}",
            semantics="stress",
            entities={"stress_id": definition.stress_id},
            payload={
                **definition.model_dump(mode="json"),
                "probabilistic": False,
                "baseline_scenario_id": request.stress_base_scenario_id,
            },
            text=(
                f"Stress {definition.stress_id} assumes demand multiplier "
                f"{definition.demand_multiplier:g} and supplier delay "
                f"{definition.supplier_delay_days} day(s); it has no probability."
            ),
        )


def _add_simulation_evidence(
    collector: EvidenceCollector,
    package: InventorySimulationPackage,
    *,
    strategy: str,
    source_prefix: str,
    stress: bool,
) -> None:
    semantics: EvidenceSemantics = (
        "stress"
        if stress
        else ("probabilistic" if package.risk_metrics is not None else "exact_simulation")
    )
    package_type = "stress_simulation_package" if stress else "exact_simulation_package"
    collector.add(
        layer="M4",
        evidence_type=package_type,
        source_object="InventorySimulationPackage",
        source_path=source_prefix,
        semantics=semantics,
        entities={"strategy": strategy},
        payload={
            "simulation_start_date": package.simulation_start_date,
            "simulation_end_date": package.simulation_end_date,
            "scenario_count": len(package.results),
            "risk_metrics_available": package.risk_metrics is not None,
            "baseline_scenarios": sorted(package.baseline_scenarios),
            "provenance": package.provenance,
        },
        text=(
            f"{'Stress' if stress else 'Exact M4'} simulation for {strategy} contains "
            f"{len(package.results)} scenario result(s); probabilistic risk metrics "
            f"are {'available' if package.risk_metrics is not None else 'not available'}."
        ),
        warnings=package.warnings,
    )
    results = sorted(
        package.results,
        key=lambda item: _model_sort_key(item, item.scenario_id),
    )
    for result_index, result in enumerate(results):
        result_path = f"{source_prefix}.results[{result_index}]"
        _add_simulation_result_evidence(
            collector,
            result,
            strategy=strategy,
            source_path=result_path,
            semantics=semantics,
            stress=stress,
        )
    metrics = package.risk_metrics
    if metrics is None:
        return
    collector.add(
        layer="M4",
        evidence_type="inventory_risk",
        source_object="InventoryRiskMetrics",
        source_path=f"{source_prefix}.risk_metrics",
        semantics="probabilistic",
        entities={"strategy": strategy},
        payload=metrics.model_dump(mode="json", exclude={"by_key"}),
        text=(
            f"Exact M4 risk for {strategy}: any-stockout probability="
            f"{metrics.any_stockout_probability:g}, mean key fill rate="
            f"{metrics.mean_key_fill_rate:g}."
        ),
    )
    risk_by_key = sorted(
        metrics.by_key,
        key=lambda item: _model_sort_key(item, item.store_id, item.ingredient_id, item.unit),
    )
    for key_index, key_metrics in enumerate(risk_by_key):
        collector.add(
            layer="M4",
            evidence_type="inventory_key_risk",
            source_object="InventoryKeyRiskMetrics",
            source_path=f"{source_prefix}.risk_metrics.by_key[{key_index}]",
            semantics="probabilistic",
            entities={
                "strategy": strategy,
                "store_id": key_metrics.store_id,
                "ingredient_id": key_metrics.ingredient_id,
                "unit": key_metrics.unit,
            },
            payload=key_metrics.model_dump(mode="json"),
            text=(
                f"Exact probabilistic risk for {strategy} "
                f"{key_metrics.store_id}/{key_metrics.ingredient_id}: stockout probability="
                f"{key_metrics.stockout_probability:g}, expected shortage="
                f"{key_metrics.expected_shortage:g} {key_metrics.unit}, expected fill="
                f"{key_metrics.expected_fill_rate:g}."
            ),
        )


def _add_simulation_result_evidence(
    collector: EvidenceCollector,
    result: InventorySimulationResult,
    *,
    strategy: str,
    source_path: str,
    semantics: EvidenceSemantics,
    stress: bool,
) -> None:
    entities = {"strategy": strategy, "scenario_id": result.scenario_id}
    collector.add(
        layer="M4",
        evidence_type="stress_result" if stress else "inventory_scenario_result",
        source_object="InventorySimulationResult",
        source_path=source_path,
        semantics=semantics,
        entities=entities,
        payload={
            "scenario_id": result.scenario_id,
            "probability_weight": result.probability_weight,
            "accounting_valid": result.accounting_valid,
            "stockout_dates": sorted(item.isoformat() for item in result.stockout_dates),
            "provenance": result.provenance,
        },
        text=(
            f"{'Stress' if stress else 'Exact inventory'} scenario {result.scenario_id} "
            f"for {strategy} has accounting_valid={result.accounting_valid} and "
            f"{len(result.stockout_dates)} stockout date(s)."
        ),
        warnings=result.warnings,
    )
    summaries = sorted(
        result.summary.by_key,
        key=lambda item: _model_sort_key(item, item.store_id, item.ingredient_id, item.unit),
    )
    for key_index, summary in enumerate(summaries):
        key_entities = {
            **entities,
            "store_id": summary.store_id,
            "ingredient_id": summary.ingredient_id,
            "unit": summary.unit,
        }
        collector.add(
            layer="M4",
            evidence_type="stress_inventory_key" if stress else "inventory_key_summary",
            source_object="InventoryKeySummary",
            source_path=f"{source_path}.summary.by_key[{key_index}]",
            semantics=semantics,
            entities=key_entities,
            event_date=summary.projected_stockout_date,
            payload={
                **summary.model_dump(mode="json"),
                "probability_weight": result.probability_weight,
                "accounting_valid": result.accounting_valid,
                "stress": stress,
            },
            text=(
                f"{'Stress' if stress else 'Exact M4'} {strategy}/"
                f"{result.scenario_id} {summary.store_id}/{summary.ingredient_id}: "
                f"demand={summary.total_demand:g}, fulfilled={summary.fulfilled_quantity:g}, "
                f"shortage={summary.shortage_quantity:g}, expired={summary.expired_quantity:g}, "
                f"waste={summary.explicit_waste_quantity:g}, ending="
                f"{summary.ending_inventory:g} {summary.unit}, fill={summary.fill_rate:g}."
            ),
            warnings=result.warnings,
        )
    ledgers = sorted(
        result.daily_ledgers,
        key=lambda item: _model_sort_key(
            item,
            item.simulation_date,
            item.store_id,
            item.ingredient_id,
            item.unit,
        ),
    )
    for ledger_index, ledger in enumerate(ledgers):
        collector.add(
            layer="M4",
            evidence_type="inventory_daily_ledger",
            source_object="DailyInventoryLedger",
            source_path=f"{source_path}.daily_ledgers[{ledger_index}]",
            semantics=semantics,
            entities={
                **entities,
                "store_id": ledger.store_id,
                "ingredient_id": ledger.ingredient_id,
                "unit": ledger.unit,
            },
            event_date=ledger.simulation_date,
            payload=ledger.model_dump(mode="json"),
            text=(
                f"Ledger {result.scenario_id} {ledger.simulation_date.isoformat()} "
                f"{ledger.store_id}/{ledger.ingredient_id}: beginning="
                f"{ledger.beginning_quantity:g}, inbound={ledger.inbound_quantity:g}, "
                f"demand={ledger.demand_quantity:g}, shortage={ledger.shortage_quantity:g}, "
                f"ending={ledger.ending_quantity:g} {ledger.unit}."
            ),
        )
    trace_groups = (
        ("lot_consumption", "consumption_traces", result.consumption_traces, "quantity"),
        ("lot_waste", "waste_traces", result.waste_traces, "quantity"),
        ("lot_expiry", "expiry_traces", result.expiry_traces, "expired_quantity"),
    )
    for evidence_type, field_name, traces, quantity_field in trace_groups:
        ordered_traces = sorted(
            traces,
            key=lambda item: _model_sort_key(
                item,
                item.simulation_date,
                item.store_id,
                item.ingredient_id,
                item.lot_id,
            ),
        )
        for trace_index, trace in enumerate(ordered_traces):
            payload = trace.model_dump(mode="json")
            quantity = float(payload[quantity_field])
            collector.add(
                layer="M4",
                evidence_type=evidence_type,
                source_object=type(trace).__name__,
                source_path=f"{source_path}.{field_name}[{trace_index}]",
                semantics=semantics,
                entities={
                    **entities,
                    "store_id": trace.store_id,
                    "ingredient_id": trace.ingredient_id,
                    "lot_id": trace.lot_id,
                },
                event_date=trace.simulation_date,
                payload=payload,
                text=(
                    f"{evidence_type.replace('_', ' ').title()} for lot {trace.lot_id} "
                    f"on {trace.simulation_date.isoformat()}: {quantity:g} {trace.unit}."
                ),
            )


def _add_optimization_evidence(
    collector: EvidenceCollector, inputs: DecisionIntelligenceInput
) -> None:
    result = inputs.optimization_result
    supplied_rule = result.provenance.get("recommendation_rule")
    rule = (
        supplied_rule.strip() if isinstance(supplied_rule, str) and supplied_rule.strip() else None
    )
    collector.add(
        layer="M5",
        evidence_type="recommendation",
        source_object="OptimizationResult",
        source_path="optimization_result.recommended_strategy",
        semantics="critic_verdict",
        entities={
            "request_id": result.request_id,
            "strategy": result.recommended_strategy or "NONE",
        },
        payload={
            "recommended_strategy": result.recommended_strategy,
            "status": result.status,
            "recommendation_rule": rule,
            "recommendation_rule_status": "RECORDED" if rule is not None else "UNAVAILABLE",
            "candidate_passed": {
                strategy: evaluation.critic.passed
                for strategy, evaluation in sorted(result.evaluations.items())
            },
        },
        text=(
            f"M5 recommendation is {result.recommended_strategy or 'none'} with status "
            f"{result.status}; "
            + (
                f"the recorded rule is {rule}."
                if rule is not None
                else "the recommendation rule was not supplied in M5 provenance."
            )
        ),
        warnings=[
            *result.warnings,
            *(["M6_COHERENCE_RECOMMENDATION_RULE_UNAVAILABLE"] if rule is None else []),
        ],
    )
    for strategy, evaluation in sorted(result.evaluations.items()):
        path = f"optimization_result.evaluations.{strategy}"
        plan = evaluation.plan
        plan_payload = {
            "plan_id": plan.plan_id,
            "strategy": plan.strategy,
            "purchase_cost": plan.purchase_cost,
            "expected_recourse_cost": plan.expected_recourse_cost,
            "objective_value": plan.objective_value,
            "solver_status": plan.solver_status,
            "completed": plan.completed,
            "first_stage_order_count": len(plan.orders),
            "scenario_recourse_order_count": sum(
                len(orders) for orders in plan.scenario_recourse_orders.values()
            ),
            "provenance": plan.provenance,
        }
        collector.add(
            layer="M5",
            evidence_type="procurement_plan",
            source_object="ProcurementPlan",
            source_path=f"{path}.plan",
            semantics="solver_estimate",
            entities={"strategy": strategy, "plan_id": plan.plan_id},
            payload=plan_payload,
            text=(
                f"{strategy} solver candidate status={plan.solver_status}, "
                f"purchase cost={plan.purchase_cost:g}, objective="
                f"{plan.objective_value if plan.objective_value is not None else 'unavailable'}, "
                f"completed_after_critic={plan.completed}."
            ),
            warnings=plan.warnings,
        )
        first_stage_orders = sorted(
            plan.orders,
            key=lambda item: _model_sort_key(
                item,
                item.order_date,
                item.arrival_date,
                item.store_id,
                item.ingredient_id,
                item.offer_id,
            ),
        )
        for order_index, order in enumerate(first_stage_orders):
            collector.add(
                layer="M5",
                evidence_type="first_stage_order",
                source_object="ProcurementDecisionLine",
                source_path=f"{path}.plan.orders[{order_index}]",
                semantics="solver_estimate",
                entities={
                    "strategy": strategy,
                    "offer_id": order.offer_id,
                    "store_id": order.store_id,
                    "ingredient_id": order.ingredient_id,
                },
                event_date=order.order_date,
                payload={**order.model_dump(mode="json"), "decision_stage": "first_stage"},
                text=(
                    f"{strategy} first-stage order: {order.order_quantity:g} {order.unit} "
                    f"of {order.ingredient_id} from {order.supplier_id} on "
                    f"{order.order_date.isoformat()}."
                ),
            )
        for scenario_id, orders in sorted(plan.scenario_recourse_orders.items()):
            ordered_recourse = sorted(
                orders,
                key=lambda item: _model_sort_key(
                    item,
                    item.order_date,
                    item.arrival_date,
                    item.store_id,
                    item.ingredient_id,
                    item.offer_id,
                ),
            )
            for order_index, order in enumerate(ordered_recourse):
                collector.add(
                    layer="M5",
                    evidence_type="recourse_order",
                    source_object="ProcurementDecisionLine",
                    source_path=(
                        f"{path}.plan.scenario_recourse_orders.{scenario_id}[{order_index}]"
                    ),
                    semantics="solver_estimate",
                    entities={
                        "strategy": strategy,
                        "scenario_id": scenario_id,
                        "offer_id": order.offer_id,
                        "store_id": order.store_id,
                        "ingredient_id": order.ingredient_id,
                    },
                    event_date=order.order_date,
                    payload={
                        **order.model_dump(mode="json"),
                        "decision_stage": "scenario_recourse",
                        "scenario_id": scenario_id,
                    },
                    text=(
                        f"{strategy} conditional recourse in scenario {scenario_id}: "
                        f"{order.order_quantity:g} {order.unit} of {order.ingredient_id} "
                        f"from {order.supplier_id}."
                    ),
                )
        collector.add(
            layer="M5",
            evidence_type="critic_verdict",
            source_object="CriticResult",
            source_path=f"{path}.critic",
            semantics="critic_verdict",
            entities={"strategy": strategy, "plan_id": plan.plan_id},
            payload=evaluation.critic.model_dump(mode="json"),
            text=(
                f"M5 critic {'passed' if evaluation.critic.passed else 'rejected'} "
                f"{strategy}; hard violations="
                f"{evaluation.critic.hard_violations or 'none'}."
            ),
            warnings=evaluation.critic.warnings,
        )
        if evaluation.simulation is not None:
            _add_simulation_evidence(
                collector,
                evaluation.simulation,
                strategy=strategy,
                source_prefix=f"{path}.simulation",
                stress=False,
            )
        else:
            collector.add(
                layer="M4",
                evidence_type="exact_simulation_availability",
                source_object="CandidateEvaluation",
                source_path=f"{path}.simulation",
                semantics="exact_simulation",
                entities={"strategy": strategy},
                payload={"status": "UNAVAILABLE"},
                text=f"Exact M4 simulation is unavailable for {strategy}.",
            )
        if evaluation.stress_simulation is not None:
            _add_simulation_evidence(
                collector,
                evaluation.stress_simulation,
                strategy=strategy,
                source_prefix=f"{path}.stress_simulation",
                stress=True,
            )


def build_evidence_package(inputs: DecisionIntelligenceInput) -> EvidencePackage:
    """Normalize supplied M1-M5 artifacts without re-running their algorithms."""

    collector = EvidenceCollector(inputs.optimization_request.request_id)
    _add_coherence_evidence(collector, inputs)
    _add_forecast_evidence(collector, inputs)
    _add_deterministic_bom_evidence(collector, inputs)
    _add_scenario_evidence(collector, inputs)
    _add_request_evidence(collector, inputs)
    _add_optimization_evidence(collector, inputs)
    return collector.package()
