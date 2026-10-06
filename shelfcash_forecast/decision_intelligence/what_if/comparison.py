from __future__ import annotations

from collections.abc import Iterable

from shelfcash_forecast.decision_intelligence.contracts import (
    CandidateSummary,
    FinalDecisionPackage,
    OrderExplanation,
)
from shelfcash_forecast.decision_intelligence.integrity import sha256_content_hash
from shelfcash_forecast.decision_intelligence.what_if.contracts import (
    DecisionComparison,
    MetricComparison,
    MetricSnapshot,
    OrderDelta,
)


def decision_snapshot_hash(decision: FinalDecisionPackage) -> str:
    return sha256_content_hash(decision)


def _order_key(order: OrderExplanation) -> tuple[str, ...]:
    return (
        order.decision_stage,
        order.scenario_id or "",
        order.offer_id,
        order.supplier_id,
        order.store_id,
        order.ingredient_id,
        order.unit,
        order.order_date.isoformat(),
        order.arrival_date.isoformat(),
    )


def _order_deltas(
    baseline: Iterable[OrderExplanation],
    hypothetical: Iterable[OrderExplanation],
) -> list[OrderDelta]:
    left = {_order_key(order): order for order in baseline}
    right = {_order_key(order): order for order in hypothetical}
    deltas: list[OrderDelta] = []
    for key in sorted(set(left) | set(right)):
        base_quantity = left[key].order_quantity if key in left else 0.0
        hypothetical_quantity = right[key].order_quantity if key in right else 0.0
        source = right.get(key) or left[key]
        deltas.append(
            OrderDelta(
                decision_stage=source.decision_stage,
                scenario_id=source.scenario_id,
                offer_id=source.offer_id,
                supplier_id=source.supplier_id,
                store_id=source.store_id,
                ingredient_id=source.ingredient_id,
                unit=source.unit,
                order_date=source.order_date,
                arrival_date=source.arrival_date,
                baseline_quantity=base_quantity,
                hypothetical_quantity=hypothetical_quantity,
                quantity_delta=hypothetical_quantity - base_quantity,
            )
        )
    return deltas


def _candidate(decision: FinalDecisionPackage) -> CandidateSummary | None:
    return decision.recommended_plan_summary


def _metric(
    name: str,
    value: float | str | bool | None,
    *,
    unit: str | None,
    grain: str,
    semantics: str,
    evidence_refs: list[str],
) -> MetricSnapshot:
    return MetricSnapshot(
        name=name,
        value=value,
        unit=unit,
        grain=grain,
        semantics=semantics,
        status="AVAILABLE" if value is not None else "UNAVAILABLE",
        evidence_refs=sorted(evidence_refs),
    )


def _compare_metric(name: str, left: MetricSnapshot, right: MetricSnapshot) -> MetricComparison:
    if left.status != "AVAILABLE" or right.status != "AVAILABLE":
        return MetricComparison(
            name=name,
            baseline=left,
            hypothetical=right,
            status="UNAVAILABLE",
            reason="One or both typed metrics are unavailable.",
        )
    if (left.unit, left.grain, left.semantics) != (right.unit, right.grain, right.semantics):
        return MetricComparison(
            name=name,
            baseline=left,
            hypothetical=right,
            status="NOT_COMPARABLE",
            reason="Metric unit, grain, or semantics differ.",
        )
    if not isinstance(left.value, (int, float)) or not isinstance(right.value, (int, float)):
        return MetricComparison(
            name=name,
            baseline=left,
            hypothetical=right,
            status="NOT_COMPARABLE",
            reason="Metric is not numeric.",
        )
    return MetricComparison(
        name=name,
        baseline=left,
        hypothetical=right,
        delta=float(right.value) - float(left.value),
        status="COMPARABLE",
        reason="Same typed metric, unit, grain, and semantics.",
    )


def _candidate_metric(
    decision: FinalDecisionPackage,
    name: str,
    field: str,
    *,
    unit: str | None,
    semantics: str,
) -> MetricSnapshot:
    candidate = _candidate(decision)
    value = getattr(candidate, field) if candidate is not None else None
    return _metric(
        name,
        value,
        unit=unit,
        grain="recommended_candidate",
        semantics=semantics,
        evidence_refs=candidate.evidence_ids if candidate else [],
    )


def _weighted_inventory_metric(
    decision: FinalDecisionPackage,
    name: str,
    field: str,
    *,
    unit: str,
) -> MetricSnapshot:
    strategy = decision.recommended_strategy
    rows = [row for row in decision.inventory_explanations if row.strategy == strategy]
    values: list[tuple[float, float]] = []
    for row in rows:
        value = getattr(row, field)
        if value is None or row.probability_weight is None or row.semantics != "probabilistic":
            continue
        values.append((float(value), row.probability_weight))
    total_weight = sum(weight for _, weight in values)
    observed_units = {row.unit for row in rows}
    value = (
        sum(metric * weight for metric, weight in values) / total_weight
        if values and total_weight > 0 and len(observed_units) == 1
        else None
    )
    return _metric(
        name,
        value,
        unit=(f"{unit}:{next(iter(observed_units))}" if len(observed_units) == 1 else None),
        grain="recommended_candidate_weighted_inventory_keys",
        semantics="exact_m4_probabilistic",
        evidence_refs=sorted(row.evidence_id for row in rows),
    )


def _consequence_metric(decision: FinalDecisionPackage) -> MetricSnapshot:
    strategy = decision.recommended_strategy
    rows = [
        row
        for row in decision.inventory_risk_explanations
        if row.strategy == strategy and row.expected_consequence_cost is not None
    ]
    value = sum(row.expected_consequence_cost for row in rows) if rows else None
    return _metric(
        "exact_m4_expected_consequence_cost",
        value,
        unit="monetary_unit_unspecified",
        grain="recommended_candidate_inventory_keys",
        semantics="exact_m4_probabilistic",
        evidence_refs=sorted(row.evidence_id for row in rows),
    )


def _solver_exact_fill_gap(decision: FinalDecisionPackage) -> MetricSnapshot:
    candidate = _candidate(decision)
    value = None
    if (
        candidate is not None
        and candidate.predicted_expected_fill_rate is not None
        and candidate.exact_mean_key_fill_rate is not None
    ):
        value = candidate.predicted_expected_fill_rate - candidate.exact_mean_key_fill_rate
    return _metric(
        "solver_minus_exact_fill_rate_gap",
        value,
        unit="ratio",
        grain="recommended_candidate",
        semantics="solver_vs_exact_m4",
        evidence_refs=candidate.evidence_ids if candidate else [],
    )


def compare_decisions(
    baseline: FinalDecisionPackage,
    hypothetical: FinalDecisionPackage,
) -> DecisionComparison:
    """Compare only typed values with compatible unit, grain, and semantics."""

    metric_specs = (
        ("purchase_cost", "purchase_cost", "monetary_unit_unspecified", "solver_plan"),
        (
            "expected_recourse_cost",
            "expected_recourse_cost",
            "monetary_unit_unspecified",
            "solver_expectation",
        ),
        ("exact_mean_key_fill_rate", "exact_mean_key_fill_rate", "ratio", "exact_m4"),
        (
            "exact_stockout_probability",
            "exact_stockout_probability",
            "probability",
            "exact_m4_probabilistic",
        ),
    )
    metrics = [
        _compare_metric(
            name,
            _candidate_metric(baseline, name, field, unit=unit, semantics=semantics),
            _candidate_metric(hypothetical, name, field, unit=unit, semantics=semantics),
        )
        for name, field, unit, semantics in metric_specs
    ]
    metrics.extend(
        [
            _compare_metric(
                "exact_m4_expected_consequence_cost",
                _consequence_metric(baseline),
                _consequence_metric(hypothetical),
            ),
            *[
                _compare_metric(
                    name,
                    _weighted_inventory_metric(baseline, name, field, unit=unit),
                    _weighted_inventory_metric(hypothetical, name, field, unit=unit),
                )
                for name, field, unit in (
                    ("exact_m4_expected_expiry", "expired", "inventory_unit_aggregated"),
                    ("exact_m4_expected_waste", "waste", "inventory_unit_aggregated"),
                    (
                        "exact_m4_expected_capacity_violation",
                        "capacity_violation_quantity",
                        "inventory_unit_aggregated",
                    ),
                )
            ],
            _compare_metric(
                "solver_minus_exact_fill_rate_gap",
                _solver_exact_fill_gap(baseline),
                _solver_exact_fill_gap(hypothetical),
            ),
        ]
    )
    base_candidate = _candidate(baseline)
    hypothetical_candidate = _candidate(hypothetical)
    return DecisionComparison(
        baseline_decision_hash=decision_snapshot_hash(baseline),
        hypothetical_decision_hash=decision_snapshot_hash(hypothetical),
        decision_status_changed=baseline.decision_status != hypothetical.decision_status,
        baseline_decision_status=baseline.decision_status,
        hypothetical_decision_status=hypothetical.decision_status,
        strategy_changed=baseline.recommended_strategy != hypothetical.recommended_strategy,
        baseline_strategy=baseline.recommended_strategy,
        hypothetical_strategy=hypothetical.recommended_strategy,
        first_stage_order_deltas=_order_deltas(
            baseline.immediate_orders, hypothetical.immediate_orders
        ),
        recourse_order_deltas=_order_deltas(
            baseline.conditional_recourse, hypothetical.conditional_recourse
        ),
        metric_comparisons=metrics,
        baseline_hard_violations=(base_candidate.hard_violations if base_candidate else []),
        hypothetical_hard_violations=(
            hypothetical_candidate.hard_violations if hypothetical_candidate else []
        ),
        baseline_readiness=(baseline.confidence_decomposition.overall_decision_readiness.status),
        hypothetical_readiness=(
            hypothetical.confidence_decomposition.overall_decision_readiness.status
        ),
        warnings=[
            "Monetary comparisons retain the source's unspecified currency unit.",
            "Deltas are model-derived comparisons, not observed or causal effects.",
        ],
    )


__all__ = ["compare_decisions", "decision_snapshot_hash"]
