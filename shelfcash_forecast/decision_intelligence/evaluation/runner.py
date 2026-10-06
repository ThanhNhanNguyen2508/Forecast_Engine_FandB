from __future__ import annotations

import platform
import re
import sys
from collections import defaultdict
from collections.abc import Callable, Mapping
from itertools import pairwise
from time import perf_counter_ns

import pydantic

from shelfcash_forecast.decision_intelligence.contracts import (
    DecisionAnswer,
    DecisionGraph,
    FinalDecisionPackage,
    GroundedClaim,
    RetrievedEvidence,
)
from shelfcash_forecast.decision_intelligence.evaluation.contracts import (
    AcceptanceGate,
    AcceptanceSummary,
    AggregateGroundingMetrics,
    BenchmarkAggregateMetrics,
    BenchmarkRuntimeMetrics,
    CasePerformance,
    CategoryAggregateMetrics,
    DecisionBenchmarkCorpus,
    DecisionBenchmarkReport,
    DecisionEvaluationCase,
    DecisionPerformance,
    DeterminismMetrics,
    EvaluationCaseResult,
    GraphAblationMetrics,
    GroundingMetricSet,
    LanguageAggregateMetrics,
    OperationPerformance,
)
from shelfcash_forecast.decision_intelligence.evaluation.gold import resolve_gold_evidence
from shelfcash_forecast.decision_intelligence.evaluation.metrics import (
    aggregate_observations,
    aggregate_retrieval_metrics,
    arithmetic_mean,
    latency_summary,
    retrieval_metrics,
)
from shelfcash_forecast.decision_intelligence.evaluation.reporting import (
    benchmark_semantic_hash,
)
from shelfcash_forecast.decision_intelligence.evaluation.retrievers import (
    LexicalOnlyRetriever,
)
from shelfcash_forecast.decision_intelligence.grounding import (
    DeterministicGroundedGenerator,
    GroundedGenerator,
    GroundingError,
    GroundingGuard,
    _positive_probability_language,
    _render,
    _unsupported_forecast_causality,
)
from shelfcash_forecast.decision_intelligence.integrity import canonical_json
from shelfcash_forecast.decision_intelligence.retrieval import (
    EvidenceRetriever,
    StructuredLocalRetriever,
    build_retrieval_context,
)

PackageBuilder = Callable[[], FinalDecisionPackage]


METRIC_DEFINITIONS = {
    "recall_at_k": (
        "Relevant gold evidence retrieved in the first k results / all resolved gold evidence; "
        "not applicable when a case has no gold evidence."
    ),
    "precision_at_k": (
        "Relevant gold evidence in the first k returned results / returned results; not "
        "applicable for zero-gold abstention cases or an empty ranking."
    ),
    "mrr": (
        "Reciprocal rank of the first relevant item; not applicable when no gold evidence exists."
    ),
    "macro_micro": (
        "Macro averages exclude not-applicable cases; micro values expose summed numerators and "
        "denominators."
    ),
    "gold_evidence": (
        "Typed content selectors constrain evidence type, exact entities/strategy, source, "
        "semantics, payload and expected cardinality independently of retrieval ranking."
    ),
    "required_evidence_type_recall": "Required evidence types represented in the first five.",
    "entity_strategy_targeting": "Requested exact entity/strategy tags found in the first five.",
    "graph_relevant_gain": (
        "Relevant graph-only IDs in top five plus Recall@5/MRR delta versus the same structured "
        "retriever with graph edges removed."
    ),
    "citation_validity": "All exposed citation IDs belong to the current evidence package.",
    "citation_completeness": "Every material claim is cited and answer citations equal their union.",
    "structured_fact_fidelity": "Expected typed fact subsets occur in guard-validated claims.",
    "visible_text_consistency": (
        "Visible text is exactly the trusted renderer output for evidence-validated facts/citations."
    ),
    "authority_fidelity": (
        "Recommendation and immediate orders exactly preserve M5 and first-stage/recourse boundaries."
    ),
    "abstention": "Unsupported and insufficient-evidence cases use claim-free trusted sentinels.",
    "semantic_violations": "Unsupported probability, stress-probability and forecast-causal claims.",
    "determinism": "Repeated package, hashes, graph, ranking, answers and tie-breaks are equal.",
    "semantic_report_hash": (
        "SHA-256 over functional report content excluding timings, platform strings and paths."
    ),
    "latency": "Local perf_counter wall-clock measurements, reported separately by operation.",
}

ACCEPTANCE_TARGETS = {
    "adversarial_guard_rejection_accuracy": "=1.0",
    "causal_attribution_violation_rate": "=0.0",
    "citation_completeness": "=1.0",
    "citation_validity": "=1.0",
    "deterministic_repeatability": "=1.0",
    "failed_critical_case_count": "=0",
    "first_stage_recourse_separation_accuracy": "=1.0",
    "immediate_order_fidelity": "=1.0",
    "insufficient_evidence_abstention_accuracy": "=1.0",
    "probability_semantic_violation_rate": "=0.0",
    "recall_at_5": ">=0.95",
    "recommendation_fidelity": "=1.0",
    "required_evidence_type_recall_at_5": ">=0.95",
    "stable_tie_break_validation": "=1.0",
    "stress_as_probability_violation_rate": "=0.0",
    "structured_fact_fidelity": "=1.0",
    "unsupported_intent_abstention_accuracy": "=1.0",
    "visible_text_structured_facts_consistency": "=1.0",
}


def _elapsed_ms(start_ns: int) -> float:
    return (perf_counter_ns() - start_ns) / 1_000_000


def _facts_contain(observed: dict, expected: dict) -> bool:
    return all(key in observed and observed[key] == value for key, value in expected.items())


def _tie_break_is_deterministic(retrieved: RetrievedEvidence) -> bool:
    ranking = [item.evidence_id for item in retrieved.items]
    return all(
        retrieved.scores[left] != retrieved.scores[right] or left < right
        for left, right in pairwise(ranking)
    )


def _expected_fact_fidelity(case: DecisionEvaluationCase, answer: DecisionAnswer) -> float:
    if not case.expected_structured_facts:
        return 1.0
    return arithmetic_mean(
        float(
            any(
                claim.claim_type == expected.claim_type
                and _facts_contain(claim.facts, expected.facts)
                for claim in answer.claims
            )
        )
        for expected in case.expected_structured_facts
    )


def _semantic_violations(
    answer: DecisionAnswer,
    decision: FinalDecisionPackage,
) -> tuple[float, float, float]:
    by_id = {item.evidence_id: item for item in decision.evidence_package.items}
    probability_violation = False
    stress_violation = False
    causal_violation = False
    for claim in answer.claims:
        cited = [by_id[evidence_id] for evidence_id in claim.evidence_ids]
        if _positive_probability_language(claim.text):
            if not any(item.semantics == "probabilistic" for item in cited):
                probability_violation = True
            if any(item.semantics in {"stress", "quantile"} for item in cited):
                stress_violation = True
        if (
            claim.causal
            or (
                any(item.evidence_type == "forecast_prediction" for item in cited)
                and _unsupported_forecast_causality(claim.text)
            )
        ) and not any(item.evidence_type == "forecast_attribution" for item in cited):
            causal_violation = True
    return float(probability_violation), float(stress_violation), float(causal_violation)


def _order_fidelity(answer: DecisionAnswer, decision: FinalDecisionPackage) -> float:
    claims = [claim for claim in answer.claims if claim.claim_type == "immediate_order"]
    expected = {
        (
            order.offer_id,
            order.supplier_id,
            order.store_id,
            order.ingredient_id,
            order.order_date.isoformat(),
            order.arrival_date.isoformat(),
            order.order_quantity,
            order.unit,
        )
        for order in decision.immediate_orders
    }
    observed = {
        (
            claim.facts.get("offer_id"),
            claim.facts.get("supplier_id"),
            claim.facts.get("store_id"),
            claim.facts.get("ingredient_id"),
            claim.facts.get("order_date"),
            claim.facts.get("arrival_date"),
            claim.facts.get("order_quantity"),
            claim.facts.get("unit"),
        )
        for claim in claims
    }
    return float(observed == expected)


def _recommendation_fidelity(answer: DecisionAnswer, decision: FinalDecisionPackage) -> float:
    recommendations = [claim for claim in answer.claims if claim.claim_type == "recommendation"]
    no_valid = [claim for claim in answer.claims if claim.claim_type == "no_valid_plan"]
    if decision.recommended_strategy is None:
        return float(
            bool(no_valid) and all(claim.facts.get("strategy") is None for claim in no_valid)
        )
    return float(
        len(recommendations) == 1
        and recommendations[0].facts.get("strategy") == decision.recommended_strategy
    )


def _normal_grounding_metrics(
    case: DecisionEvaluationCase,
    answer: DecisionAnswer,
    decision: FinalDecisionPackage,
    retrieved: RetrievedEvidence,
) -> GroundingMetricSet:
    known = {item.evidence_id for item in decision.evidence_package.items}
    claim_ids = {evidence_id for claim in answer.claims for evidence_id in claim.evidence_ids}
    citation_validity = float(not ((set(answer.citations) | claim_ids) - known))
    citation_completeness = float(
        set(answer.citations) == claim_ids and all(claim.evidence_ids for claim in answer.claims)
        if answer.claims
        else not answer.citations
    )
    probability, stress, causal = _semantic_violations(answer, decision)
    trusted_again = GroundingGuard().validate(answer, decision, retrieved)
    expected_visible_text = (
        "WHAT_IF_NOT_AVAILABLE_IN_M6_PART1"
        if answer.status == "UNSUPPORTED_INTENT"
        else "INSUFFICIENT_EVIDENCE"
        if answer.status == "INSUFFICIENT_EVIDENCE"
        else _render(answer.claims)
    )
    visible_consistency = float(
        answer.answer_text == expected_visible_text
        and canonical_json(answer.model_dump(mode="json"))
        == canonical_json(trusted_again.model_dump(mode="json"))
    )
    is_recommendation = case.expected_intent == "recommendation"
    is_order = case.expected_intent == "immediate_order"
    cited_types = {
        item.evidence_type
        for item in decision.evidence_package.items
        if item.evidence_id in answer.citations
    }
    recourse_separated = float(
        not (cited_types & set(case.forbidden_evidence_types))
        and not any(
            "scenario_id" in claim.facts
            for claim in answer.claims
            if claim.claim_type == "immediate_order"
        )
    )
    return GroundingMetricSet(
        normal_guard_success=1.0,
        citation_validity=citation_validity,
        citation_completeness=citation_completeness,
        structured_fact_fidelity=_expected_fact_fidelity(case, answer),
        recommendation_fidelity=(
            _recommendation_fidelity(answer, decision) if is_recommendation else None
        ),
        immediate_order_fidelity=_order_fidelity(answer, decision) if is_order else None,
        first_stage_recourse_separation_accuracy=(recourse_separated if is_order else None),
        probability_semantic_violation=probability,
        stress_as_probability_violation=stress,
        causal_attribution_violation=causal,
        unsupported_intent_abstention_accuracy=(
            float(
                answer.status == "UNSUPPORTED_INTENT"
                and answer.answer_text == "WHAT_IF_NOT_AVAILABLE_IN_M6_PART1"
                and not answer.claims
                and not answer.citations
            )
            if case.expected_answer_status == "UNSUPPORTED_INTENT"
            else None
        ),
        insufficient_evidence_abstention_accuracy=(
            float(
                answer.status == "INSUFFICIENT_EVIDENCE"
                and answer.answer_text == "INSUFFICIENT_EVIDENCE"
                and not answer.claims
                and not answer.citations
            )
            if case.expected_answer_status == "INSUFFICIENT_EVIDENCE"
            else None
        ),
        visible_text_structured_facts_consistency=visible_consistency,
    )


def _forge_answer(
    case: DecisionEvaluationCase,
    answer: DecisionAnswer,
    retrieved: RetrievedEvidence,
) -> DecisionAnswer:
    if case.adversarial_kind == "forged_strategy":
        index = next(
            i for i, claim in enumerate(answer.claims) if claim.claim_type == "recommendation"
        )
        claim = answer.claims[index]
        recorded = str(claim.facts["strategy"])
        forged_strategy = next(
            strategy for strategy in ("BALANCED", "LEAN", "PROTECTED") if strategy != recorded
        )
        forged = claim.model_copy(update={"text": claim.text.replace(recorded, forged_strategy)})
    elif case.adversarial_kind == "forged_order_quantity":
        index = next(
            i for i, claim in enumerate(answer.claims) if claim.claim_type == "immediate_order"
        )
        claim = answer.claims[index]
        forged = claim.model_copy(
            update={
                "text": re.sub(
                    r"^Order\s+\S+",
                    "Order 999",
                    claim.text,
                )
            }
        )
    elif case.adversarial_kind == "stress_probability_vi":
        index = next(
            i for i, claim in enumerate(answer.claims) if claim.claim_type == "stress_definition"
        )
        claim = answer.claims[index]
        forged = claim.model_copy(update={"text": "Stress này có 30% nguy cơ xảy ra thiếu hàng."})
    elif case.adversarial_kind == "forecast_causality_vi":
        forecast = next(
            item for item in retrieved.items if item.evidence_type == "forecast_prediction"
        )
        text = "Forecast tăng do thời tiết và được thúc đẩy bởi khuyến mãi."
        claim = GroundedClaim(
            claim_type="retrieved_fact",
            text=text,
            evidence_ids=[forecast.evidence_id],
            facts={"evidence_type": "forecast_prediction"},
        )
        return DecisionAnswer(
            question=answer.question,
            intent=answer.intent,
            status="GROUNDED",
            answer_text=f"{text} [evidence:{forecast.evidence_id}]",
            claims=[claim],
            citations=[forecast.evidence_id],
            retrieved_evidence_ids=answer.retrieved_evidence_ids,
        )
    else:
        raise ValueError(f"Unsupported adversarial mutation: {case.adversarial_kind}")
    claims = list(answer.claims)
    claims[index] = forged
    return answer.model_copy(
        update={
            "claims": claims,
            "answer_text": answer.answer_text.replace(claim.text, forged.text),
        }
    )


def _aggregate_optional(rows: list[GroundingMetricSet], field: str, *, zero_value: float) -> float:
    values = [getattr(row, field) for row in rows if getattr(row, field) is not None]
    return arithmetic_mean(values, zero_value=zero_value)


def _aggregate_grounding(rows: list[GroundingMetricSet]) -> AggregateGroundingMetrics:
    return AggregateGroundingMetrics(
        normal_guard_success=_aggregate_optional(rows, "normal_guard_success", zero_value=1.0),
        citation_validity=_aggregate_optional(rows, "citation_validity", zero_value=1.0),
        citation_completeness=_aggregate_optional(rows, "citation_completeness", zero_value=1.0),
        structured_fact_fidelity=_aggregate_optional(
            rows, "structured_fact_fidelity", zero_value=1.0
        ),
        recommendation_fidelity=_aggregate_optional(
            rows, "recommendation_fidelity", zero_value=1.0
        ),
        immediate_order_fidelity=_aggregate_optional(
            rows, "immediate_order_fidelity", zero_value=1.0
        ),
        first_stage_recourse_separation_accuracy=_aggregate_optional(
            rows, "first_stage_recourse_separation_accuracy", zero_value=1.0
        ),
        probability_semantic_violation_rate=_aggregate_optional(
            rows, "probability_semantic_violation", zero_value=0.0
        ),
        stress_as_probability_violation_rate=_aggregate_optional(
            rows, "stress_as_probability_violation", zero_value=0.0
        ),
        causal_attribution_violation_rate=_aggregate_optional(
            rows, "causal_attribution_violation", zero_value=0.0
        ),
        unsupported_intent_abstention_accuracy=_aggregate_optional(
            rows, "unsupported_intent_abstention_accuracy", zero_value=1.0
        ),
        insufficient_evidence_abstention_accuracy=_aggregate_optional(
            rows, "insufficient_evidence_abstention_accuracy", zero_value=1.0
        ),
        visible_text_structured_facts_consistency=_aggregate_optional(
            rows, "visible_text_structured_facts_consistency", zero_value=1.0
        ),
        adversarial_guard_rejection_accuracy=_aggregate_optional(
            rows, "adversarial_guard_rejection_accuracy", zero_value=1.0
        ),
    )


def _operation(
    operation: str,
    samples: list[float],
    fixture_ids: set[str],
    description: str,
) -> OperationPerformance | None:
    if not samples:
        return None
    return OperationPerformance(
        operation=operation,
        latency=latency_summary(samples),
        fixture_ids=sorted(fixture_ids),
        description=description,
    )


def _extract_error_codes(error: Exception) -> list[str]:
    text = str(error)
    return sorted(set(re.findall(r"M6_[A-Z0-9_]+", text)))


def _metric_count(rows: list[GroundingMetricSet], field: str) -> int:
    return sum(getattr(row, field) is not None for row in rows)


def _gate(
    name: str,
    value: float | None,
    operator: str,
    target: float,
    count: int,
    note: str | None = None,
) -> AcceptanceGate:
    if value is None and note is None:
        note = "Not applicable because no corpus case evaluates this metric."
    passed = False
    if value is not None:
        passed = {"eq": value == target, "gte": value >= target, "lte": value <= target}[operator]
    return AcceptanceGate(
        metric_name=name,
        observed_value=value,
        operator=operator,
        target=target,
        passed=passed,
        evaluated_case_count=count,
        note=note,
    )


def _when_evaluated(value: float, count: int) -> float | None:
    return value if count > 0 else None


def run_decision_intelligence_benchmark(
    corpus: DecisionBenchmarkCorpus,
    decisions: Mapping[str, FinalDecisionPackage],
    *,
    package_builders: Mapping[str, PackageBuilder] | None = None,
    artifact_builders: Mapping[str, PackageBuilder] | None = None,
    scale_fixture_builders: Mapping[str, PackageBuilder] | None = None,
    artifact_errors: Mapping[str, list[str]] | None = None,
    repetitions: int = 5,
    retrieval_limit: int = 20,
    retriever: EvidenceRetriever | None = None,
    generator: GroundedGenerator | None = None,
) -> DecisionBenchmarkReport:
    """Evaluate public M6 behavior over materialized, read-only decision snapshots."""

    if repetitions < 2:
        raise ValueError("Benchmark repetitions must be at least two.")
    if artifact_errors:
        raise ValueError("Injected artifact errors are forbidden; provide artifact_builders.")
    builders = dict(package_builders or {})
    rejection_builders = dict(artifact_builders or {})
    scale_builders = dict(scale_fixture_builders or {})
    artifact_ids = {
        case.decision_id
        for case in corpus.cases
        if case.expected_answer_status == "ARTIFACT_REJECTED"
    }
    normal_ids = {case.decision_id for case in corpus.cases} - artifact_ids
    missing = normal_ids - set(decisions) - set(builders) - set(scale_builders)
    if missing:
        raise ValueError(f"Missing benchmark decision fixtures: {sorted(missing)}")

    built_decisions = dict(decisions)
    package_equalities: list[float] = []
    hash_equalities: list[float] = []
    graph_equalities: list[float] = []
    full_build_samples: list[float] = []
    full_build_ids: set[str] = set()
    for decision_id, builder in sorted(builders.items()):
        if decision_id not in normal_ids:
            continue
        outputs = []
        for _ in range(repetitions):
            start = perf_counter_ns()
            outputs.append(builder())
            full_build_samples.append(_elapsed_ms(start))
        built_decisions[decision_id] = outputs[0]
        full_build_ids.add(decision_id)
        package_equalities.append(float(len({item.model_dump_json() for item in outputs}) == 1))
        hash_equalities.append(
            float(len({item.evidence_package.package_hash for item in outputs}) == 1)
        )
        graph_equalities.append(
            float(len({item.decision_graph.model_dump_json() for item in outputs}) == 1)
        )

    scale_samples: list[float] = []
    scale_ids: set[str] = set()
    for decision_id, builder in sorted(scale_builders.items()):
        if decision_id not in normal_ids:
            continue
        outputs = []
        for _ in range(repetitions):
            start = perf_counter_ns()
            outputs.append(builder())
            scale_samples.append(_elapsed_ms(start))
        built_decisions[decision_id] = outputs[0]
        scale_ids.add(decision_id)
        package_equalities.append(float(len({item.model_dump_json() for item in outputs}) == 1))
        hash_equalities.append(
            float(len({item.evidence_package.package_hash for item in outputs}) == 1)
        )
        graph_equalities.append(
            float(len({item.decision_graph.model_dump_json() for item in outputs}) == 1)
        )

    deserialization_samples: list[float] = []
    decision_performance = []
    validated_decisions: dict[str, FinalDecisionPackage] = {}
    for decision_id in sorted(normal_ids):
        source = built_decisions[decision_id]
        outputs = []
        samples = []
        payload = source.model_dump(mode="json")
        for _ in range(repetitions):
            start = perf_counter_ns()
            outputs.append(FinalDecisionPackage.model_validate(payload))
            sample = _elapsed_ms(start)
            samples.append(sample)
            deserialization_samples.append(sample)
        first = outputs[0]
        validated_decisions[decision_id] = first
        package_equalities.append(float(len({item.model_dump_json() for item in outputs}) == 1))
        hash_equalities.append(
            float(len({item.evidence_package.package_hash for item in outputs}) == 1)
        )
        graph_equalities.append(
            float(len({item.decision_graph.model_dump_json() for item in outputs}) == 1)
        )
        decision_performance.append(
            DecisionPerformance(
                decision_id=decision_id,
                evidence_item_count=len(first.evidence_package.items),
                graph_node_count=len(first.decision_graph.nodes),
                graph_edge_count=len(first.decision_graph.edges),
                serialized_package_bytes=len(first.model_dump_json().encode("utf-8")),
                deserialization_validation_latency=latency_summary(samples),
            )
        )

    artifact_outcomes: dict[str, list[str]] = {}
    for decision_id in sorted(artifact_ids):
        builder = rejection_builders.get(decision_id)
        if builder is None:
            artifact_outcomes[decision_id] = []
            continue
        runs = []
        for _ in range(repetitions):
            try:
                builder()
            except (ValueError, pydantic.ValidationError) as error:
                runs.append(tuple(_extract_error_codes(error)))
            else:
                runs.append(())
        artifact_outcomes[decision_id] = list(runs[0])
        package_equalities.append(float(len(set(runs)) == 1))

    structured_retriever = retriever or StructuredLocalRetriever()
    lexical_retriever = LexicalOnlyRetriever()
    selected_generator = generator or DeterministicGroundedGenerator()
    guard = GroundingGuard()
    case_results = []
    ranking_equalities: list[float] = []
    answer_equalities: list[float] = []
    tie_break_checks: list[float] = []
    all_retrieval_samples: list[float] = []
    all_generation_samples: list[float] = []

    for case in corpus.cases:
        if case.expected_answer_status == "ARTIFACT_REJECTED":
            observed_codes = artifact_outcomes.get(case.decision_id, [])
            expected = set(case.expected_error_codes)
            observed_coherence = {
                code for code in observed_codes if code.startswith("M6_COHERENCE_")
            }
            failures = []
            if not observed_codes:
                failures.append("ARTIFACT_BUILDER_DID_NOT_REJECT")
            if observed_coherence != expected:
                failures.append(
                    f"ARTIFACT_ERROR_CODES:{sorted(observed_coherence)}!={sorted(expected)}"
                )
            case_results.append(
                EvaluationCaseResult(
                    case_id=case.case_id,
                    decision_id=case.decision_id,
                    category=case.category,
                    language=case.language,
                    critical=case.critical,
                    passed=not failures,
                    failures=failures,
                    observed_intent=case.expected_intent,
                    observed_answer_status=(
                        "ARTIFACT_REJECTED" if observed_codes else "NOT_REJECTED"
                    ),
                    observed_error_codes=observed_codes,
                    grounding=GroundingMetricSet(),
                    performance=CasePerformance(),
                )
            )
            continue

        decision = validated_decisions[case.decision_id]
        context = build_retrieval_context(
            case.question,
            decision.evidence_package,
            recommended_strategy=decision.recommended_strategy,
        )
        empty_graph = DecisionGraph(
            request_id=decision.request_id,
            nodes=decision.decision_graph.nodes,
            edges=[],
            provenance={"ablation": "structured_without_graph"},
        )
        structured_runs = []
        no_graph_runs = []
        lexical_runs = []
        retrieval_samples = []
        for _ in range(repetitions):
            start = perf_counter_ns()
            structured_runs.append(
                structured_retriever.retrieve(
                    case.question,
                    decision.evidence_package,
                    decision.decision_graph,
                    context=context or None,
                    limit=retrieval_limit,
                )
            )
            sample = _elapsed_ms(start)
            retrieval_samples.append(sample)
            all_retrieval_samples.append(sample)
            no_graph_runs.append(
                structured_retriever.retrieve(
                    case.question,
                    decision.evidence_package,
                    empty_graph,
                    context=context or None,
                    limit=retrieval_limit,
                )
            )
            lexical_runs.append(
                lexical_retriever.retrieve(
                    case.question,
                    decision.evidence_package,
                    decision.decision_graph,
                    context=context or None,
                    limit=retrieval_limit,
                )
            )
        structured = structured_runs[0]
        no_graph = no_graph_runs[0]
        lexical = lexical_runs[0]
        ranking_equalities.append(
            float(
                len({tuple(item.evidence_id for item in run.items) for run in structured_runs}) == 1
                and len({tuple(item.evidence_id for item in run.items) for run in no_graph_runs})
                == 1
                and len({tuple(item.evidence_id for item in run.items) for run in lexical_runs})
                == 1
            )
        )
        tie_break_checks.append(
            float(
                all(_tie_break_is_deterministic(run) for run in structured_runs)
                and all(_tie_break_is_deterministic(run) for run in no_graph_runs)
                and all(_tie_break_is_deterministic(run) for run in lexical_runs)
            )
        )

        gold = resolve_gold_evidence(
            case.gold_evidence_selectors,
            decision.evidence_package,
            strict=False,
        )
        relevant_ids = set(gold.evidence_ids)
        structured_top = {item.evidence_id for item in structured.items[:5]}
        no_graph_top = {item.evidence_id for item in no_graph.items[:5]}
        relevant_graph_only = (structured_top - no_graph_top) & relevant_ids
        target_pairs = set(case.required_entities.items())
        if case.required_strategy is not None:
            target_pairs.add(("strategy", case.required_strategy))
        structured_metrics = retrieval_metrics(
            structured.items,
            relevant_ids,
            set(case.required_evidence_types),
            target_pairs,
            relevant_graph_only,
        )
        no_graph_metrics = retrieval_metrics(
            no_graph.items,
            relevant_ids,
            set(case.required_evidence_types),
            target_pairs,
        )
        lexical_metrics = retrieval_metrics(
            lexical.items,
            relevant_ids,
            set(case.required_evidence_types),
            target_pairs,
        )

        explanation_samples = []
        answers = []
        guard_errors = []
        grounding: GroundingMetricSet
        if case.adversarial_kind is not None:
            for _ in range(repetitions):
                start = perf_counter_ns()
                base = selected_generator.generate(case.question, structured, decision)
                forged = _forge_answer(case, base, structured)
                try:
                    guard.validate(forged, decision, structured)
                except GroundingError as error:
                    guard_errors.append(str(error))
                sample = _elapsed_ms(start)
                explanation_samples.append(sample)
                all_generation_samples.append(sample)
            grounding = GroundingMetricSet(
                adversarial_guard_rejection_accuracy=float(len(guard_errors) == repetitions)
            )
            answer = None
            answer_equalities.append(float(len(set(guard_errors)) == 1))
        else:
            for _ in range(repetitions):
                start = perf_counter_ns()
                generated = selected_generator.generate(case.question, structured, decision)
                try:
                    answers.append(guard.validate(generated, decision, structured))
                except GroundingError as error:
                    guard_errors.append(str(error))
                sample = _elapsed_ms(start)
                explanation_samples.append(sample)
                all_generation_samples.append(sample)
            if guard_errors:
                answer = None
                answer_equalities.append(float(len(set(guard_errors)) == 1))
                grounding = GroundingMetricSet(
                    normal_guard_success=0.0,
                    structured_fact_fidelity=0.0,
                    visible_text_structured_facts_consistency=0.0,
                )
            else:
                answer = answers[0]
                answer_equalities.append(
                    float(len({item.model_dump_json() for item in answers}) == 1)
                )
                grounding = _normal_grounding_metrics(case, answer, decision, structured)

        failures = list(gold.errors)
        observed_intent = structured.intent
        observed_status = (
            "GUARD_REJECTED"
            if case.adversarial_kind is not None or (answer is None and guard_errors)
            else answer.status
            if answer is not None
            else "NO_ANSWER"
        )
        if observed_intent != case.expected_intent:
            failures.append(f"INTENT:{observed_intent}!={case.expected_intent}")
        if observed_status != case.expected_answer_status:
            failures.append(f"STATUS:{observed_status}!={case.expected_answer_status}")
        type_metric = structured_metrics.required_evidence_type_recall_at_5
        if type_metric.applicable and type_metric.value != 1.0:
            failures.append("REQUIRED_EVIDENCE_TYPE_MISSING_AT_5")
        if case.expected_structured_facts and grounding.structured_fact_fidelity != 1.0:
            failures.append("STRUCTURED_FACT_MISMATCH")
        if case.adversarial_kind is None and grounding.normal_guard_success != 1.0:
            failures.append("NORMAL_OUTPUT_GUARD_REJECTED")
        if grounding.visible_text_structured_facts_consistency == 0.0:
            failures.append("VISIBLE_TEXT_STRUCTURED_FACTS_MISMATCH")
        if grounding.adversarial_guard_rejection_accuracy == 0.0:
            failures.append("ADVERSARIAL_OUTPUT_NOT_REJECTED")
        if grounding.unsupported_intent_abstention_accuracy == 0.0:
            failures.append("UNSUPPORTED_INTENT_DID_NOT_ABSTAIN")
        if grounding.insufficient_evidence_abstention_accuracy == 0.0:
            failures.append("INSUFFICIENT_EVIDENCE_DID_NOT_ABSTAIN")
        if answer is not None:
            cited_types = {
                item.evidence_type
                for item in decision.evidence_package.items
                if item.evidence_id in answer.citations
            }
            if cited_types & set(case.forbidden_evidence_types):
                failures.append("FORBIDDEN_EVIDENCE_TYPE_CITED")
            if any(
                re.search(pattern, answer.answer_text, flags=re.IGNORECASE)
                for pattern in case.forbidden_claim_patterns
            ):
                failures.append("FORBIDDEN_CLAIM_PATTERN")
        case_results.append(
            EvaluationCaseResult(
                case_id=case.case_id,
                decision_id=case.decision_id,
                category=case.category,
                language=case.language,
                critical=case.critical,
                passed=not failures,
                failures=sorted(set(failures)),
                observed_intent=observed_intent,
                observed_answer_status=observed_status,
                structured_retrieval=structured_metrics,
                structured_without_graph_retrieval=no_graph_metrics,
                lexical_retrieval=lexical_metrics,
                grounding=grounding,
                structured_ranking=[item.evidence_id for item in structured.items],
                structured_without_graph_ranking=[item.evidence_id for item in no_graph.items],
                lexical_ranking=[item.evidence_id for item in lexical.items],
                relevant_evidence_ids=sorted(relevant_ids),
                relevant_graph_only_ids_at_5=sorted(relevant_graph_only),
                gold_resolution_errors=list(gold.errors),
                answer=answer,
                guard_error=guard_errors[0] if guard_errors else None,
                performance=CasePerformance(
                    retrieval=latency_summary(retrieval_samples),
                    generation_and_guard=latency_summary(explanation_samples),
                ),
            )
        )

    retrieval_results = [
        result for result in case_results if result.structured_retrieval is not None
    ]
    structured_rows = [result.structured_retrieval for result in retrieval_results]
    no_graph_rows = [result.structured_without_graph_retrieval for result in retrieval_results]
    lexical_rows = [result.lexical_retrieval for result in retrieval_results]
    structured_aggregate = aggregate_retrieval_metrics(structured_rows)
    no_graph_aggregate = aggregate_retrieval_metrics(no_graph_rows)
    lexical_aggregate = aggregate_retrieval_metrics(lexical_rows)
    grounding_rows = [result.grounding for result in case_results]
    grounding_aggregate = _aggregate_grounding(grounding_rows)
    determinism = DeterminismMetrics(
        repeated_output_equality=arithmetic_mean(package_equalities, zero_value=1.0),
        evidence_package_hash_equality=arithmetic_mean(hash_equalities, zero_value=1.0),
        graph_equality=arithmetic_mean(graph_equalities, zero_value=1.0),
        retrieval_ranking_equality=arithmetic_mean(ranking_equalities, zero_value=1.0),
        answer_equality=arithmetic_mean(answer_equalities, zero_value=1.0),
        deterministic_tie_break_validation=arithmetic_mean(tie_break_checks, zero_value=1.0),
    )

    graph_rows = [
        result
        for result in retrieval_results
        if result.structured_retrieval.recall_at_5.applicable
        and result.structured_without_graph_retrieval.recall_at_5.applicable
    ]
    recall_deltas = [
        result.structured_retrieval.recall_at_5.value
        - result.structured_without_graph_retrieval.recall_at_5.value
        for result in graph_rows
    ]
    mrr_deltas = [
        result.structured_retrieval.mean_reciprocal_rank.value
        - result.structured_without_graph_retrieval.mean_reciprocal_rank.value
        for result in graph_rows
    ]
    graph_ablation = GraphAblationMetrics(
        relevant_graph_only_id_count=sum(
            len(result.relevant_graph_only_ids_at_5) for result in graph_rows
        ),
        recall_at_5_delta=arithmetic_mean(recall_deltas),
        mrr_delta=arithmetic_mean(mrr_deltas),
        improved_case_count=sum(delta > 0 for delta in recall_deltas),
        worsened_case_count=sum(delta < 0 for delta in recall_deltas),
        unchanged_case_count=sum(delta == 0 for delta in recall_deltas),
        evaluated_case_count=len(graph_rows),
    )

    failed_critical = sum(not result.passed and result.critical for result in case_results)
    runtime = BenchmarkRuntimeMetrics(
        full_package_build=_operation(
            "full_package_build",
            full_build_samples,
            full_build_ids,
            "build_final_decision_package over supplied typed M1-M5 artifacts",
        ),
        package_deserialization_validation=_operation(
            "package_deserialization_validation",
            deserialization_samples,
            normal_ids,
            "FinalDecisionPackage model_validate over deterministic JSON-compatible payloads",
        ),
        retrieval=_operation(
            "retrieval",
            all_retrieval_samples,
            {result.decision_id for result in retrieval_results},
            "Structured retrieval with the materialized Decision Graph",
        ),
        generation_and_guard=_operation(
            "generation_and_guard",
            all_generation_samples,
            {result.decision_id for result in retrieval_results},
            "Deterministic generation followed by GroundingGuard validation",
        ),
        scale_fixture_materialization=_operation(
            "scale_fixture_materialization",
            scale_samples,
            scale_ids,
            "Synthetic evidence/graph materialization only; no M1-M5 computation",
        ),
    )
    aggregate = BenchmarkAggregateMetrics(
        structured_retrieval=structured_aggregate,
        structured_without_graph_retrieval=no_graph_aggregate,
        lexical_only_retrieval=lexical_aggregate,
        graph_ablation=graph_ablation,
        grounding=grounding_aggregate,
        determinism=determinism,
        runtime=runtime,
        passed_case_count=sum(result.passed for result in case_results),
        failed_case_count=sum(not result.passed for result in case_results),
        failed_critical_case_count=failed_critical,
    )

    failures_by_category: dict[str, list[str]] = defaultdict(list)
    category_rows: dict[str, list[EvaluationCaseResult]] = defaultdict(list)
    language_rows: dict[str, list[EvaluationCaseResult]] = defaultdict(list)
    for result in case_results:
        category_rows[result.category].append(result)
        language_rows[result.language].append(result)
        if not result.passed:
            failures_by_category[result.category].append(result.case_id)
    per_category = [
        CategoryAggregateMetrics(
            category=category,
            case_count=len(rows),
            passed_case_count=sum(row.passed for row in rows),
            pass_rate=arithmetic_mean(float(row.passed) for row in rows),
        )
        for category, rows in sorted(category_rows.items())
    ]
    per_language = []
    for language, rows in sorted(language_rows.items()):
        retrieval_for_language = [
            row.structured_retrieval for row in rows if row.structured_retrieval is not None
        ]
        abstention_values = [
            value
            for row in rows
            for value in (
                row.grounding.unsupported_intent_abstention_accuracy,
                row.grounding.insufficient_evidence_abstention_accuracy,
            )
            if value is not None
        ]
        guard_values = [
            row.grounding.normal_guard_success
            for row in rows
            if row.grounding.normal_guard_success is not None
        ]
        per_language.append(
            LanguageAggregateMetrics(
                language=language,
                case_count=len(rows),
                passed_case_count=sum(row.passed for row in rows),
                pass_rate=arithmetic_mean(float(row.passed) for row in rows),
                recall_at_5=aggregate_observations(
                    metric.recall_at_5 for metric in retrieval_for_language
                ),
                required_evidence_type_recall_at_5=aggregate_observations(
                    metric.required_evidence_type_recall_at_5 for metric in retrieval_for_language
                ),
                intent_accuracy=arithmetic_mean(
                    float(
                        row.observed_intent
                        == next(
                            case.expected_intent
                            for case in corpus.cases
                            if case.case_id == row.case_id
                        )
                    )
                    for row in rows
                ),
                abstention_accuracy=(
                    arithmetic_mean(abstention_values) if abstention_values else None
                ),
                grounding_guard_success=(arithmetic_mean(guard_values) if guard_values else None),
                answer_localization_evaluated=False,
            )
        )

    repeatability = min(
        determinism.repeated_output_equality,
        determinism.evidence_package_hash_equality,
        determinism.graph_equality,
        determinism.retrieval_ranking_equality,
        determinism.answer_equality,
    )
    grounding_counts = {
        field: _metric_count(grounding_rows, field) for field in GroundingMetricSet.model_fields
    }
    gates = sorted(
        [
            _gate(
                "failed_critical_case_count",
                float(failed_critical),
                "eq",
                0.0,
                len(case_results),
            ),
            _gate(
                "recall_at_5",
                structured_aggregate.recall_at_5.macro_value,
                "gte",
                0.95,
                structured_aggregate.recall_at_5.applicable_case_count,
            ),
            _gate(
                "required_evidence_type_recall_at_5",
                structured_aggregate.required_evidence_type_recall_at_5.macro_value,
                "gte",
                0.95,
                structured_aggregate.required_evidence_type_recall_at_5.applicable_case_count,
            ),
            _gate(
                "recommendation_fidelity",
                _when_evaluated(
                    grounding_aggregate.recommendation_fidelity,
                    grounding_counts["recommendation_fidelity"],
                ),
                "eq",
                1.0,
                grounding_counts["recommendation_fidelity"],
            ),
            _gate(
                "immediate_order_fidelity",
                _when_evaluated(
                    grounding_aggregate.immediate_order_fidelity,
                    grounding_counts["immediate_order_fidelity"],
                ),
                "eq",
                1.0,
                grounding_counts["immediate_order_fidelity"],
            ),
            _gate(
                "first_stage_recourse_separation_accuracy",
                _when_evaluated(
                    grounding_aggregate.first_stage_recourse_separation_accuracy,
                    grounding_counts["first_stage_recourse_separation_accuracy"],
                ),
                "eq",
                1.0,
                grounding_counts["first_stage_recourse_separation_accuracy"],
            ),
            _gate(
                "citation_validity",
                _when_evaluated(
                    grounding_aggregate.citation_validity,
                    grounding_counts["citation_validity"],
                ),
                "eq",
                1.0,
                grounding_counts["citation_validity"],
            ),
            _gate(
                "citation_completeness",
                _when_evaluated(
                    grounding_aggregate.citation_completeness,
                    grounding_counts["citation_completeness"],
                ),
                "eq",
                1.0,
                grounding_counts["citation_completeness"],
            ),
            _gate(
                "structured_fact_fidelity",
                _when_evaluated(
                    grounding_aggregate.structured_fact_fidelity,
                    grounding_counts["structured_fact_fidelity"],
                ),
                "eq",
                1.0,
                grounding_counts["structured_fact_fidelity"],
            ),
            _gate(
                "visible_text_structured_facts_consistency",
                _when_evaluated(
                    grounding_aggregate.visible_text_structured_facts_consistency,
                    grounding_counts["visible_text_structured_facts_consistency"],
                ),
                "eq",
                1.0,
                grounding_counts["visible_text_structured_facts_consistency"],
            ),
            _gate(
                "unsupported_intent_abstention_accuracy",
                _when_evaluated(
                    grounding_aggregate.unsupported_intent_abstention_accuracy,
                    grounding_counts["unsupported_intent_abstention_accuracy"],
                ),
                "eq",
                1.0,
                grounding_counts["unsupported_intent_abstention_accuracy"],
            ),
            _gate(
                "insufficient_evidence_abstention_accuracy",
                _when_evaluated(
                    grounding_aggregate.insufficient_evidence_abstention_accuracy,
                    grounding_counts["insufficient_evidence_abstention_accuracy"],
                ),
                "eq",
                1.0,
                grounding_counts["insufficient_evidence_abstention_accuracy"],
            ),
            _gate(
                "adversarial_guard_rejection_accuracy",
                _when_evaluated(
                    grounding_aggregate.adversarial_guard_rejection_accuracy,
                    grounding_counts["adversarial_guard_rejection_accuracy"],
                ),
                "eq",
                1.0,
                grounding_counts["adversarial_guard_rejection_accuracy"],
            ),
            _gate(
                "probability_semantic_violation_rate",
                _when_evaluated(
                    grounding_aggregate.probability_semantic_violation_rate,
                    grounding_counts["probability_semantic_violation"],
                ),
                "eq",
                0.0,
                grounding_counts["probability_semantic_violation"],
            ),
            _gate(
                "stress_as_probability_violation_rate",
                _when_evaluated(
                    grounding_aggregate.stress_as_probability_violation_rate,
                    grounding_counts["stress_as_probability_violation"],
                ),
                "eq",
                0.0,
                grounding_counts["stress_as_probability_violation"],
            ),
            _gate(
                "causal_attribution_violation_rate",
                _when_evaluated(
                    grounding_aggregate.causal_attribution_violation_rate,
                    grounding_counts["causal_attribution_violation"],
                ),
                "eq",
                0.0,
                grounding_counts["causal_attribution_violation"],
            ),
            _gate(
                "deterministic_repeatability",
                repeatability,
                "eq",
                1.0,
                len(package_equalities) + len(ranking_equalities),
            ),
            _gate(
                "stable_tie_break_validation",
                determinism.deterministic_tie_break_validation,
                "eq",
                1.0,
                len(tie_break_checks),
            ),
        ],
        key=lambda gate: gate.metric_name,
    )
    acceptance = AcceptanceSummary(gates=gates, overall_pass=all(gate.passed for gate in gates))
    report = DecisionBenchmarkReport(
        corpus_version=corpus.corpus_version,
        environment={
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "pydantic": pydantic.__version__,
            "retrieval": "offline_structured_graph_no_graph_and_lexical",
            "repetitions": str(repetitions),
        },
        metric_definitions=METRIC_DEFINITIONS,
        cases=case_results,
        aggregate=aggregate,
        failures_by_category=dict(sorted(failures_by_category.items())),
        per_category=per_category,
        per_language=per_language,
        decision_performance=decision_performance,
        acceptance=acceptance,
        acceptance_targets=ACCEPTANCE_TARGETS,
        acceptance_results={gate.metric_name: gate.passed for gate in gates},
        overall_pass=acceptance.overall_pass,
        semantic_report_hash="sha256:" + "0" * 64,
        limitations=[
            "The benchmark measures M6 retrieval/explanation, not forecast or procurement accuracy.",
            "Vietnamese cases evaluate query understanding, not localized answer quality.",
            "Latency is local wall-clock data and varies by hardware and process load.",
            "The lexical-only ablation excludes type, entity and graph bonuses.",
            "Synthetic scale replicas exercise evidence/graph size, not operational truth.",
            "No external LLM, embedding, vector database or network service is evaluated.",
        ],
    )
    return report.model_copy(update={"semantic_report_hash": benchmark_semantic_hash(report)})
