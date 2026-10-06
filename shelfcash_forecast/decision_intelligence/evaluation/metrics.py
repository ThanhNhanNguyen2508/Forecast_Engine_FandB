from __future__ import annotations

import math
from collections.abc import Iterable

from shelfcash_forecast.decision_intelligence.contracts import EvidenceItem
from shelfcash_forecast.decision_intelligence.evaluation.contracts import (
    AggregateMetricObservation,
    AggregateRetrievalMetrics,
    LatencySummary,
    MetricObservation,
    RetrievalMetricSet,
)


def safe_ratio(numerator: float, denominator: float, *, zero_value: float = 0.0) -> float:
    if denominator == 0:
        return zero_value
    return numerator / denominator


def arithmetic_mean(values: Iterable[float], *, zero_value: float = 0.0) -> float:
    materialized = list(values)
    return safe_ratio(sum(materialized), len(materialized), zero_value=zero_value)


def percentile(values: Iterable[float], percentile_value: float) -> float:
    samples = sorted(float(value) for value in values)
    if not samples:
        raise ValueError("Percentile requires at least one sample.")
    if not 0 <= percentile_value <= 100:
        raise ValueError("Percentile must be between 0 and 100.")
    position = (len(samples) - 1) * percentile_value / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return samples[lower]
    fraction = position - lower
    return samples[lower] + (samples[upper] - samples[lower]) * fraction


def latency_summary(samples_ms: list[float]) -> LatencySummary:
    samples = [float(value) for value in samples_ms]
    return LatencySummary(
        samples_ms=samples,
        sample_count=len(samples),
        minimum_ms=min(samples),
        maximum_ms=max(samples),
        p50_ms=percentile(samples, 50),
        p95_ms=percentile(samples, 95),
    )


def _observation(
    numerator: float,
    denominator: float,
    *,
    applicable: bool,
    note: str | None = None,
) -> MetricObservation:
    return MetricObservation(
        value=(numerator / denominator if applicable else None),
        applicable=applicable,
        numerator=numerator,
        denominator=denominator,
        note=note,
    )


def retrieval_metrics(
    ranking: list[EvidenceItem],
    relevant_ids: set[str],
    required_evidence_types: set[str],
    target_pairs: set[tuple[str, str]],
    relevant_graph_only_top_five_ids: set[str] | None = None,
) -> RetrievalMetricSet:
    ranked_ids = [item.evidence_id for item in ranking]
    has_gold = bool(relevant_ids)

    def recall(k: int) -> MetricObservation:
        hits = len(set(ranked_ids[:k]) & relevant_ids)
        return _observation(
            hits,
            len(relevant_ids),
            applicable=has_gold,
            note=None if has_gold else "No relevant evidence; evaluate abstention separately.",
        )

    def precision(k: int) -> MetricObservation:
        observed = ranked_ids[:k]
        hits = len(set(observed) & relevant_ids)
        applicable = has_gold and bool(observed)
        note = None
        if not has_gold:
            note = "No relevant evidence; evaluate abstention separately."
        elif not observed:
            note = "Retriever returned no items; recall captures the miss."
        return _observation(hits, len(observed), applicable=applicable, note=note)

    first_relevant_rank = next(
        (
            index
            for index, evidence_id in enumerate(ranked_ids, start=1)
            if evidence_id in relevant_ids
        ),
        None,
    )
    top_five = ranking[:5]
    observed_types = {item.evidence_type for item in top_five}
    targeting_hits = sum(
        any(item.entities.get(key) == value for item in top_five)
        for key, value in sorted(target_pairs)
    )
    graph_only = relevant_graph_only_top_five_ids or set()
    mrr_value = 0.0 if first_relevant_rank is None else 1.0 / first_relevant_rank
    return RetrievalMetricSet(
        recall_at_1=recall(1),
        recall_at_3=recall(3),
        recall_at_5=recall(5),
        precision_at_1=precision(1),
        precision_at_3=precision(3),
        precision_at_5=precision(5),
        mean_reciprocal_rank=_observation(
            mrr_value,
            1,
            applicable=has_gold,
            note=None if has_gold else "No relevant evidence; MRR is not applicable.",
        ),
        required_evidence_type_recall_at_5=_observation(
            len(observed_types & required_evidence_types),
            len(required_evidence_types),
            applicable=bool(required_evidence_types),
            note=(None if required_evidence_types else "The case has no required evidence types."),
        ),
        entity_strategy_targeting_accuracy=_observation(
            targeting_hits,
            len(target_pairs),
            applicable=bool(target_pairs),
            note=None if target_pairs else "The case has no entity/strategy target.",
        ),
        graph_relevant_contribution_at_5=_observation(
            len(graph_only),
            len(relevant_ids),
            applicable=has_gold,
            note=None if has_gold else "No relevant evidence for graph contribution.",
        ),
        relevant_evidence_count=len(relevant_ids),
    )


def aggregate_observations(
    observations: Iterable[MetricObservation],
) -> AggregateMetricObservation:
    rows = list(observations)
    applicable = [row for row in rows if row.applicable]
    if not applicable:
        return AggregateMetricObservation(
            macro_value=None,
            micro_value=None,
            numerator=0,
            denominator=0,
            applicable_case_count=0,
            excluded_case_count=len(rows),
        )
    numerator = sum(row.numerator for row in applicable)
    denominator = sum(row.denominator for row in applicable)
    return AggregateMetricObservation(
        macro_value=arithmetic_mean(row.value for row in applicable if row.value is not None),
        micro_value=numerator / denominator,
        numerator=numerator,
        denominator=denominator,
        applicable_case_count=len(applicable),
        excluded_case_count=len(rows) - len(applicable),
    )


def aggregate_retrieval_metrics(
    metrics: Iterable[RetrievalMetricSet],
) -> AggregateRetrievalMetrics:
    rows = list(metrics)
    return AggregateRetrievalMetrics(
        recall_at_1=aggregate_observations(row.recall_at_1 for row in rows),
        recall_at_3=aggregate_observations(row.recall_at_3 for row in rows),
        recall_at_5=aggregate_observations(row.recall_at_5 for row in rows),
        precision_at_1=aggregate_observations(row.precision_at_1 for row in rows),
        precision_at_3=aggregate_observations(row.precision_at_3 for row in rows),
        precision_at_5=aggregate_observations(row.precision_at_5 for row in rows),
        mean_reciprocal_rank=aggregate_observations(row.mean_reciprocal_rank for row in rows),
        required_evidence_type_recall_at_5=aggregate_observations(
            row.required_evidence_type_recall_at_5 for row in rows
        ),
        entity_strategy_targeting_accuracy=aggregate_observations(
            row.entity_strategy_targeting_accuracy for row in rows
        ),
        graph_relevant_contribution_at_5=aggregate_observations(
            row.graph_relevant_contribution_at_5 for row in rows
        ),
        evaluated_case_count=len(rows),
    )
