# EvidencePackage là format audit/retrieval tổng quát. Nhưng UI hoặc application cần object dễ dùng hơn.

# Explainers biến evidence thành:

# ForecastExplanation
# BOMExplanation
# InventoryKeyExplanation
# InventoryRiskExplanation
# InventoryTraceExplanation
# StressExplanation
# CandidateSummary
# ConfidenceDecomposition

# Ví dụ:

# EvidenceItem procurement_plan
# + EvidenceItem critic_verdict
# + EvidenceItem inventory_risk
# + EvidenceItem strategy_profile
#         ↓
# CandidateSummary(BALANCED)

# Layer này không tạo quyết định. Nó chỉ tạo “view có cấu trúc” từ evidence.
from __future__ import annotations

from collections import defaultdict
from typing import Any

from shelfcash_forecast.decision_intelligence.contracts import (
    BOMContributionExplanation,
    BOMExplanation,
    CandidateSummary,
    ConfidenceDecomposition,
    DecisionIntelligenceInput,
    EvidenceItem,
    EvidencePackage,
    ForecastExplanation,
    InventoryKeyExplanation,
    InventoryRiskExplanation,
    InventoryTraceExplanation,
    OrderExplanation,
    ReadinessDimension,
    StrategyProfileExplanation,
    StressExplanation,
)

# Overall readiness uses this authority ordering. UNAVAILABLE is contextual:
# optional forecast/BOM -> PARTIAL; unrequested stress -> neutral.
READINESS_SEVERITY_LATTICE = {
    "VERIFIED": 0,
    "WARNING": 1,
    "PARTIAL": 2,
    "FAILED": 3,
}


def _by_type(evidence: EvidencePackage) -> dict[str, list[EvidenceItem]]:
    output: dict[str, list[EvidenceItem]] = defaultdict(list)
    for item in evidence.items:
        output[item.evidence_type].append(item)
    return output


def build_forecast_explanations(evidence: EvidencePackage) -> list[ForecastExplanation]:
    output = []
    for item in _by_type(evidence).get("forecast_prediction", []):
        payload = item.payload
        output.append(
            ForecastExplanation(
                evidence_id=item.evidence_id,
                store_id=str(payload["store_id"]),
                product_id=str(payload["product_id"]),
                product_name=str(payload["product_name"]),
                target_date=payload["target_date"],
                model_version=str(payload["model_version"]),
                p25=float(payload["p25"]),
                p50=float(payload["p50"]),
                p75=float(payload["p75"]),
                interval_lower=float(payload["interval_lower"]),
                interval_upper=float(payload["interval_upper"]),
                quantile_spread=float(payload["quantile_spread"]),
                interval_width=float(payload["interval_width"]),
                baseline_p50=float(payload["baseline_p50"]),
                delta_vs_baseline=float(payload["delta_vs_baseline"]),
                calibration_source=str(payload["calibration_source"]),
                warnings=item.warnings,
            )
        )
    return sorted(
        output,
        key=lambda row: (row.target_date, row.store_id, row.product_id),
    )


def _matching_contributions(
    item: EvidenceItem,
    contributions: list[EvidenceItem],
) -> list[EvidenceItem]:
    keys = ("scenario_id", "store_id", "ingredient_id", "target_date")
    return [
        contribution
        for contribution in contributions
        if all(
            key not in item.entities
            or key not in contribution.entities
            or item.entities[key] == contribution.entities[key]
            for key in keys
        )
        and item.entities.get("ingredient_id") == contribution.entities.get("ingredient_id")
    ]


def build_bom_explanations(evidence: EvidencePackage) -> list[BOMExplanation]:
    types = _by_type(evidence)
    output: list[BOMExplanation] = []
    deterministic_contributions = types.get("recipe_contribution", [])
    for item in types.get("ingredient_demand", []):
        payload = item.payload
        contributions = []
        for source in _matching_contributions(item, deterministic_contributions):
            values = source.payload
            contributions.append(
                BOMContributionExplanation(
                    evidence_id=source.evidence_id,
                    product_id=str(values["product_id"]),
                    product_name=str(values["product_name"]),
                    recipe_id=str(values["recipe_id"]),
                    recipe_version=str(values["recipe_version"]),
                    unit=str(values["contribution_unit"]),
                    base_quantity=float(values["base_contribution_p50"]),
                    final_quantity=float(values["contribution_p50"]),
                    forecast_quantity=float(values["forecast_p50"]),
                    recipe_quantity=float(values["recipe_quantity"]),
                    recipe_unit=str(values["recipe_unit"]),
                    yield_quantity=float(values["yield_quantity"]),
                    yield_unit=str(values["yield_unit"]),
                    process_loss_rate=float(values["process_loss_rate"]),
                    waste_allowance_rate=float(values["waste_allowance_rate"]),
                )
            )
        output.append(
            BOMExplanation(
                evidence_id=item.evidence_id,
                status="AVAILABLE" if contributions else "PARTIAL",
                semantics="quantile",
                store_id=str(payload["store_id"]),
                ingredient_id=str(payload["ingredient_id"]),
                ingredient_name=payload.get("ingredient_name"),
                target_date=payload["target_date"],
                unit=str(payload["unit"]),
                p25=float(payload["p25"]),
                p50=float(payload["p50"]),
                p75=float(payload["p75"]),
                contributions=sorted(
                    contributions,
                    key=lambda row: (-row.final_quantity, row.product_id, row.recipe_id),
                ),
                warnings=item.warnings,
            )
        )

    scenario_contributions = types.get("scenario_recipe_contribution", [])
    for item in types.get("ingredient_demand_scenario", []):
        payload = item.payload
        contributions = []
        for source in _matching_contributions(item, scenario_contributions):
            values = source.payload
            contributions.append(
                BOMContributionExplanation(
                    evidence_id=source.evidence_id,
                    product_id=str(values["product_id"]),
                    product_name=str(values["product_name"]),
                    recipe_id=str(values["recipe_id"]),
                    recipe_version=str(values["recipe_version"]),
                    unit=str(values["unit"]),
                    base_quantity=float(values["fixed_recipe_quantity"]),
                    final_quantity=float(values["final_quantity"]),
                )
            )
        output.append(
            BOMExplanation(
                evidence_id=item.evidence_id,
                status="AVAILABLE" if contributions else "PARTIAL",
                semantics="probabilistic",
                store_id=str(payload["store_id"]),
                ingredient_id=str(payload["ingredient_id"]),
                ingredient_name=payload.get("ingredient_name"),
                target_date=payload["target_date"],
                unit=str(payload["unit"]),
                scenario_id=str(payload["scenario_id"]),
                probability_weight=float(payload["probability_weight"]),
                scenario_quantity=float(payload["quantity"]),
                contributions=sorted(
                    contributions,
                    key=lambda row: (-row.final_quantity, row.product_id, row.recipe_id),
                ),
                warnings=item.warnings,
            )
        )
    return sorted(
        output,
        key=lambda row: (
            row.target_date,
            row.store_id,
            row.ingredient_id,
            row.scenario_id or "",
        ),
    )


def _inventory_semantics(item: EvidenceItem) -> str:
    if item.semantics == "stress":
        return "stress"
    if item.semantics == "probabilistic":
        return "probabilistic"
    if item.entities.get("scenario_id") in {"LOW_P25", "MEDIAN_P50", "HIGH_P75"}:
        return "quantile"
    return "deterministic"


def _inventory_key(item: EvidenceItem) -> InventoryKeyExplanation:
    payload = item.payload
    return InventoryKeyExplanation(
        evidence_id=item.evidence_id,
        strategy=item.entities["strategy"],
        scenario_id=item.entities["scenario_id"],
        semantics=_inventory_semantics(item),
        probability_weight=payload.get("probability_weight"),
        store_id=str(payload["store_id"]),
        ingredient_id=str(payload["ingredient_id"]),
        unit=str(payload["unit"]),
        beginning_inventory=float(payload["beginning_inventory"]),
        inbound=float(payload["total_inbound"]),
        demand=float(payload["total_demand"]),
        fulfilled=float(payload["fulfilled_quantity"]),
        shortage=float(payload["shortage_quantity"]),
        expired=float(payload["expired_quantity"]),
        waste=float(payload["explicit_waste_quantity"]),
        ending_inventory=float(payload["ending_inventory"]),
        maximum_inventory=float(payload["maximum_inventory"]),
        fill_rate=float(payload["fill_rate"]),
        projected_stockout_date=payload.get("projected_stockout_date"),
        at_risk_expiry_quantity=float(payload["at_risk_expiry_quantity"]),
        capacity_violation_quantity=payload.get("capacity_violation_quantity"),
        consequence_cost=payload.get("total_consequence_cost"),
        accounting_valid=bool(payload["accounting_valid"]),
        warnings=item.warnings,
    )


def build_inventory_explanations(
    evidence: EvidencePackage,
) -> tuple[
    list[InventoryKeyExplanation],
    list[InventoryRiskExplanation],
    list[InventoryTraceExplanation],
    list[StressExplanation],
]:
    types = _by_type(evidence)
    inventory = [_inventory_key(item) for item in types.get("inventory_key_summary", [])]
    risks = []
    for item in types.get("inventory_key_risk", []):
        payload = item.payload
        risks.append(
            InventoryRiskExplanation(
                evidence_id=item.evidence_id,
                strategy=item.entities["strategy"],
                store_id=str(payload["store_id"]),
                ingredient_id=str(payload["ingredient_id"]),
                unit=str(payload["unit"]),
                stockout_probability=float(payload["stockout_probability"]),
                expected_shortage=float(payload["expected_shortage"]),
                p95_shortage=float(payload["p95_shortage"]),
                expected_fill_rate=float(payload["expected_fill_rate"]),
                expected_consequence_cost=payload.get("expected_consequence_cost"),
                p95_consequence_cost=payload.get("p95_consequence_cost"),
                cvar95_consequence_cost=payload.get("cvar95_consequence_cost"),
            )
        )

    traces = []
    trace_specs = (
        ("lot_consumption", "FEFO_CONSUMPTION", "quantity"),
        ("lot_expiry", "EXPIRY", "expired_quantity"),
        ("lot_waste", "WASTE", "quantity"),
    )
    for evidence_type, trace_type, quantity_field in trace_specs:
        for item in types.get(evidence_type, []):
            payload = item.payload
            traces.append(
                InventoryTraceExplanation(
                    evidence_id=item.evidence_id,
                    strategy=item.entities["strategy"],
                    scenario_id=item.entities["scenario_id"],
                    trace_type=trace_type,
                    simulation_date=payload["simulation_date"],
                    store_id=str(payload["store_id"]),
                    ingredient_id=str(payload["ingredient_id"]),
                    lot_id=str(payload["lot_id"]),
                    unit=str(payload["unit"]),
                    quantity=float(payload[quantity_field]),
                    event_id=payload.get("event_id"),
                    expiry_date=payload.get("expiry_date") or payload.get("lot_expiry_date"),
                )
            )

    definitions = {item.entities["stress_id"]: item for item in types.get("stress_definition", [])}
    stress_keys: dict[tuple[str, str], list[InventoryKeyExplanation]] = defaultdict(list)
    stress_item_ids: dict[tuple[str, str], str] = {}
    for item in types.get("stress_inventory_key", []):
        key = (item.entities["strategy"], item.entities["scenario_id"])
        stress_keys[key].append(_inventory_key(item))
        stress_item_ids[key] = item.evidence_id
    stresses = []
    for (strategy, stress_id), keys in sorted(stress_keys.items()):
        definition = definitions.get(stress_id)
        payload: dict[str, Any] = definition.payload if definition else {}
        stresses.append(
            StressExplanation(
                evidence_id=(
                    definition.evidence_id if definition else stress_item_ids[(strategy, stress_id)]
                ),
                strategy=strategy,
                stress_id=stress_id,
                description=payload.get("description"),
                demand_multiplier=payload.get("demand_multiplier"),
                supplier_delay_days=payload.get("supplier_delay_days"),
                supplier_ids=sorted(payload.get("supplier_ids", [])),
                inventory_keys=sorted(
                    keys,
                    key=lambda row: (row.store_id, row.ingredient_id, row.unit),
                ),
                warnings=sorted({warning for row in keys for warning in row.warnings}),
            )
        )
    return (
        sorted(
            inventory,
            key=lambda row: (
                row.strategy,
                row.scenario_id,
                row.store_id,
                row.ingredient_id,
            ),
        ),
        sorted(
            risks,
            key=lambda row: (row.strategy, row.store_id, row.ingredient_id, row.unit),
        ),
        sorted(
            traces,
            key=lambda row: (
                row.strategy,
                row.scenario_id,
                row.simulation_date,
                row.store_id,
                row.ingredient_id,
                row.lot_id,
                row.trace_type,
            ),
        ),
        stresses,
    )


def _order(item: EvidenceItem) -> OrderExplanation:
    payload = item.payload
    return OrderExplanation(
        evidence_id=item.evidence_id,
        decision_stage=payload["decision_stage"],
        scenario_id=payload.get("scenario_id"),
        offer_id=str(payload["offer_id"]),
        supplier_id=str(payload["supplier_id"]),
        store_id=str(payload["store_id"]),
        ingredient_id=str(payload["ingredient_id"]),
        unit=str(payload["unit"]),
        order_date=payload["order_date"],
        arrival_date=payload["arrival_date"],
        pack_count=int(payload["pack_count"]),
        order_quantity=float(payload["order_quantity"]),
        purchase_cost=float(payload["purchase_cost"]),
        delivery_cost=float(payload["delivery_cost"]),
        emergency=bool(payload["emergency"]),
    )


def build_candidate_summaries(evidence: EvidencePackage) -> list[CandidateSummary]:
    types = _by_type(evidence)
    plans = {item.entities["strategy"]: item for item in types.get("procurement_plan", [])}
    critics = {item.entities["strategy"]: item for item in types.get("critic_verdict", [])}
    profiles = {item.entities["strategy"]: item for item in types.get("strategy_profile", [])}
    first_stage: dict[str, list[OrderExplanation]] = defaultdict(list)
    recourse: dict[str, list[OrderExplanation]] = defaultdict(list)
    for item in types.get("first_stage_order", []):
        first_stage[item.entities["strategy"]].append(_order(item))
    for item in types.get("recourse_order", []):
        recourse[item.entities["strategy"]].append(_order(item))
    exact_risk = {item.entities["strategy"]: item for item in types.get("inventory_risk", [])}
    output = []
    for strategy, plan_item in sorted(plans.items()):
        plan = plan_item.payload
        provenance = dict(plan.get("provenance", {}))
        critic_item = critics[strategy]
        critic = critic_item.payload
        profile_item = profiles[strategy]
        profile = profile_item.payload
        risk_item = exact_risk.get(strategy)
        risk = risk_item.payload if risk_item else {}
        evidence_ids = [plan_item.evidence_id, critic_item.evidence_id, profile_item.evidence_id]
        evidence_ids.extend(
            item.evidence_id
            for item in types.get("first_stage_order", [])
            if item.entities.get("strategy") == strategy
        )
        evidence_ids.extend(
            item.evidence_id
            for item in types.get("recourse_order", [])
            if item.entities.get("strategy") == strategy
        )
        if risk_item:
            evidence_ids.append(risk_item.evidence_id)
        output.append(
            CandidateSummary(
                strategy=strategy,
                strategy_profile=StrategyProfileExplanation(
                    evidence_id=profile_item.evidence_id,
                    strategy=strategy,
                    source_status=(
                        "VERIFIED_DECISION_INPUT"
                        if profile.get("decision_time_verified")
                        else "RECONSTRUCTED_CURRENT_DEFAULT"
                    ),
                    shortage_penalty=float(profile["shortage_penalty"]),
                    holding_penalty=float(profile["holding_penalty"]),
                    waste_penalty=float(profile["waste_penalty"]),
                    cash_penalty=float(profile["cash_penalty"]),
                    cvar_weight=float(profile["cvar_weight"]),
                    cvar_alpha=float(profile["cvar_alpha"]),
                    maximum_stockout_probability=profile.get("maximum_stockout_probability"),
                    minimum_expected_fill_rate=profile.get("minimum_expected_fill_rate"),
                    minimum_fill_rate=profile.get("minimum_fill_rate"),
                    required_fill_rate_probability=profile.get("required_fill_rate_probability"),
                    minimum_acceptable_fill_rate=float(profile["minimum_acceptable_fill_rate"]),
                    maximum_acceptable_stockout_probability=float(
                        profile["maximum_acceptable_stockout_probability"]
                    ),
                    maximum_fill_rate_model_gap=float(profile["maximum_fill_rate_model_gap"]),
                    maximum_stockout_probability_model_gap=float(
                        profile["maximum_stockout_probability_model_gap"]
                    ),
                ),
                plan_id=str(plan["plan_id"]),
                solver=provenance.get("solver"),
                formulation=provenance.get("formulation"),
                solver_status=str(plan["solver_status"]),
                completed=bool(plan["completed"]),
                purchase_cost=float(plan["purchase_cost"]),
                expected_recourse_cost=float(plan["expected_recourse_cost"]),
                objective_value=plan.get("objective_value"),
                cvar_alpha=provenance.get("cvar_alpha"),
                cvar_weight=provenance.get("cvar_weight"),
                estimated_cvar=provenance.get("estimated_cvar"),
                predicted_expected_fill_rate=provenance.get("predicted_expected_fill_rate"),
                predicted_stockout_probability=provenance.get("predicted_stockout_probability"),
                exact_mean_key_fill_rate=risk.get("mean_key_fill_rate"),
                exact_stockout_probability=risk.get("any_stockout_probability"),
                first_stage_orders=sorted(
                    first_stage[strategy],
                    key=lambda row: (row.order_date, row.offer_id),
                ),
                scenario_recourse_orders=sorted(
                    recourse[strategy],
                    key=lambda row: (row.scenario_id or "", row.order_date, row.offer_id),
                ),
                critic_passed=bool(critic["passed"]),
                critic_checks=dict(critic["checks"]),
                critic_details=dict(critic["details"]),
                hard_violations=list(critic["hard_violations"]),
                warnings=sorted(
                    set(plan_item.warnings)
                    | set(critic_item.warnings)
                    | set(critic.get("warnings", []))
                ),
                evidence_ids=sorted(set(evidence_ids)),
            )
        )
    return output


def _dimension(
    status: str,
    reason: str,
    items: list[EvidenceItem],
    warnings: list[str] | None = None,
) -> ReadinessDimension:
    return ReadinessDimension(
        status=status,
        reason=reason,
        evidence_ids=sorted(item.evidence_id for item in items),
        warnings=sorted(set(warnings or [])),
    )


def build_confidence_decomposition(
    inputs: DecisionIntelligenceInput,
    evidence: EvidencePackage,
) -> ConfidenceDecomposition:
    """Report evidence readiness without inventing a confidence probability."""

    types = _by_type(evidence)
    coherence_items = types.get("artifact_coherence", [])
    coherence_result = inputs.coherence
    if coherence_result is None:
        artifact_coherence = _dimension(
            "FAILED",
            "Artifact coherence validation result is unavailable.",
            coherence_items,
        )
    else:
        artifact_coherence = _dimension(
            coherence_result.status,
            (
                "Cross-artifact identity, dates, horizons, versions, scenario lineage and "
                f"M5 authority checks completed with status {coherence_result.status}."
            ),
            coherence_items,
            [issue.code for issue in coherence_result.issues],
        )
    forecast_items = types.get("forecast_prediction", [])
    forecast_warnings = sorted({warning for item in forecast_items for warning in item.warnings})
    forecast = (
        _dimension(
            "UNAVAILABLE",
            "ForecastPackage was not supplied.",
            types.get("forecast_evidence_availability", []),
        )
        if not forecast_items
        else _dimension(
            "WARNING" if forecast_warnings else "VERIFIED",
            "Forecast quantiles, calibrated interval, baseline and calibration source are traceable.",
            forecast_items,
            forecast_warnings,
        )
    )

    scenario_items = types.get("inventory_demand_scenario", [])
    semantics = {item.semantics for item in scenario_items}
    scenario = _dimension(
        "VERIFIED" if "probabilistic" in semantics else "WARNING",
        (
            "Demand scenarios carry explicit probability weights."
            if "probabilistic" in semantics
            else "Demand scenarios are design/quantile scenarios and do not support probability claims."
        ),
        scenario_items,
    )

    bom_items = [
        *types.get("recipe_contribution", []),
        *types.get("scenario_recipe_contribution", []),
    ]
    bom_issues = types.get("bom_issue", [])
    if bom_issues:
        bom = _dimension(
            "FAILED",
            "Supplied BOM artifact contains explicit issues.",
            [*bom_items, *bom_issues],
            [item.entities.get("issue_code", "BOM_ISSUE") for item in bom_issues],
        )
    elif bom_items:
        bom = _dimension(
            "VERIFIED",
            "Product, recipe version and materialized contribution traces are available.",
            bom_items,
        )
    elif (
        inputs.ingredient_demand_package is not None
        or inputs.ingredient_scenario_bundle is not None
    ):
        bom = _dimension(
            "PARTIAL",
            "Ingredient totals are available but product/recipe contribution traces are absent.",
            [
                *types.get("ingredient_demand", []),
                *types.get("ingredient_demand_scenario", []),
            ],
        )
    else:
        bom = _dimension(
            "UNAVAILABLE",
            "No M3 artifact was supplied; recipe reasons cannot be reconstructed from M5.",
            types.get("bom_evidence_availability", []),
        )

    result = inputs.optimization_result
    recommended = result.recommended_strategy
    exact_items = [
        item
        for item in types.get("exact_simulation_package", [])
        if recommended is None or item.entities.get("strategy") == recommended
    ]
    exact_results = [
        item
        for item in types.get("inventory_scenario_result", [])
        if recommended is None or item.entities.get("strategy") == recommended
    ]
    if not exact_items:
        inventory = _dimension(
            "FAILED",
            "Exact M4 simulation is unavailable for the decision authority path.",
            types.get("exact_simulation_availability", []),
        )
    elif any(not bool(item.payload.get("accounting_valid")) for item in exact_results):
        inventory = _dimension(
            "FAILED",
            "At least one exact M4 result reports invalid accounting.",
            [*exact_items, *exact_results],
        )
    else:
        inventory_warnings = sorted(
            {warning for item in [*exact_items, *exact_results] for warning in item.warnings}
        )
        inventory = _dimension(
            "WARNING" if inventory_warnings else "VERIFIED",
            "Exact lot-level M4 simulations are present and accounting-valid.",
            [*exact_items, *exact_results],
            inventory_warnings,
        )

    critic_items = types.get("critic_verdict", [])
    recommendation_items = types.get("recommendation", [])
    if recommended is None:
        optimization = _dimension(
            "FAILED",
            "M5 returned NO_VALID_PROCUREMENT_PLAN; M6 cannot create a fallback recommendation.",
            [*recommendation_items, *critic_items],
            [
                violation
                for item in critic_items
                for violation in item.payload.get("hard_violations", [])
            ],
        )
    else:
        selected_critic = [
            item for item in critic_items if item.entities.get("strategy") == recommended
        ]
        passed = bool(selected_critic and selected_critic[0].payload.get("passed"))
        recommendation_rule_available = bool(
            recommendation_items
            and recommendation_items[0].payload.get("recommendation_rule_status") == "RECORDED"
        )
        optimization_status = (
            "FAILED" if not passed else "VERIFIED" if recommendation_rule_available else "WARNING"
        )
        optimization = _dimension(
            optimization_status,
            (
                "Recommended candidate matches M5, passed its critic, and has a recorded recommendation rule."
                if passed and recommendation_rule_available
                else "Recommended candidate passed its critic, but the M5 recommendation rule is unavailable."
                if passed
                else "Recommended candidate lacks a passing M5 critic verdict."
            ),
            [*recommendation_items, *selected_critic],
            (
                []
                if recommendation_rule_available
                else ["M6_COHERENCE_RECOMMENDATION_RULE_UNAVAILABLE"]
            ),
        )

    stress_items = [
        item
        for item in types.get("stress_inventory_key", [])
        if recommended is None or item.entities.get("strategy") == recommended
    ]
    if not inputs.optimization_request.stress_scenarios:
        stress = _dimension("UNAVAILABLE", "No stress scenarios were requested.", [])
    elif not stress_items:
        stress = _dimension(
            "FAILED",
            "Stress scenarios were requested but no stress simulation evidence is available.",
            types.get("stress_definition", []),
        )
    else:
        stress_warnings = sorted({warning for item in stress_items for warning in item.warnings})
        stress = _dimension(
            "WARNING" if stress_warnings else "VERIFIED",
            "Explicit adverse stress results are available and are not assigned probability.",
            [*types.get("stress_definition", []), *stress_items],
            stress_warnings,
        )

    authority_dimensions = (artifact_coherence, inventory, optimization)
    all_dimensions = (
        artifact_coherence,
        forecast,
        scenario,
        bom,
        inventory,
        optimization,
        stress,
    )
    if any(item.status == "FAILED" for item in authority_dimensions) or any(
        item.status == "FAILED" for item in (bom, stress)
    ):
        overall = _dimension(
            "FAILED",
            "At least one required authority or explicitly supplied validation dimension failed.",
            [
                item
                for item in evidence.items
                if item.evidence_id
                in {
                    evidence_id
                    for dimension in all_dimensions
                    if dimension.status == "FAILED"
                    for evidence_id in dimension.evidence_ids
                }
            ],
            [
                warning
                for dimension in all_dimensions
                if dimension.status == "FAILED"
                for warning in dimension.warnings
            ],
        )
    elif any(item.status in {"UNAVAILABLE", "PARTIAL"} for item in (forecast, bom)):
        overall = _dimension(
            "PARTIAL",
            "M5 decision is valid, but optional upstream explanation evidence is incomplete.",
            [
                item
                for item in evidence.items
                if item.evidence_id in set(optimization.evidence_ids + inventory.evidence_ids)
            ],
        )
    elif any(item.status == "WARNING" for item in all_dimensions):
        overall = _dimension(
            "WARNING",
            "M5 decision is valid with evidence warnings that should be presented.",
            [
                item
                for item in evidence.items
                if item.evidence_id in set(optimization.evidence_ids + inventory.evidence_ids)
            ],
        )
    else:
        overall = _dimension(
            "VERIFIED",
            "M5 recommendation, exact M4 validation and supplied upstream provenance are traceable.",
            [
                item
                for item in evidence.items
                if item.evidence_id in set(optimization.evidence_ids + inventory.evidence_ids)
            ],
        )
    return ConfidenceDecomposition(
        artifact_coherence=artifact_coherence,
        forecast_evidence=forecast,
        scenario_evidence=scenario,
        bom_traceability=bom,
        inventory_validation=inventory,
        optimization_validity=optimization,
        stress_evidence=stress,
        overall_decision_readiness=overall,
    )
