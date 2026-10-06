from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from shelfcash_forecast.decision_intelligence.contracts import (
    AnswerStatus,
    DecisionAnswer,
    EvidenceSemantics,
    StrictDecisionContract,
    _assert_finite,
)
from shelfcash_forecast.decision_intelligence.integrity import is_full_sha256

EvaluationLanguage = Literal["en", "vi"]
EvaluationExpectedStatus = AnswerStatus | Literal["GUARD_REJECTED", "ARTIFACT_REJECTED"]
AdversarialKind = Literal[
    "forged_strategy",
    "forged_order_quantity",
    "stress_probability_vi",
    "forecast_causality_vi",
]
GateOperator = Literal["eq", "gte", "lte"]

_GOLD_ENTITY_KEYS: dict[str, set[str]] = {
    "critic_verdict": {"strategy"},
    "exact_simulation_availability": {"strategy"},
    "exact_simulation_package": {"strategy"},
    "first_stage_order": {"strategy", "store_id", "ingredient_id", "supplier_id"},
    "ingredient_demand": {"store_id", "ingredient_id"},
    "inventory_daily_ledger": {"strategy", "store_id", "ingredient_id", "scenario_id"},
    "inventory_risk": {"strategy"},
    "inventory_key_risk": {"strategy", "store_id", "ingredient_id"},
    "inventory_key_summary": {"strategy", "store_id", "ingredient_id", "scenario_id"},
    "lot_consumption": {"strategy", "store_id", "ingredient_id", "scenario_id", "lot_id"},
    "lot_expiry": {"strategy", "store_id", "ingredient_id", "scenario_id", "lot_id"},
    "lot_waste": {"strategy", "store_id", "ingredient_id", "scenario_id", "lot_id"},
    "recipe_contribution": {"store_id", "product_id", "ingredient_id"},
    "recommendation": {"strategy"},
    "scenario_recipe_contribution": {
        "store_id",
        "product_id",
        "ingredient_id",
        "scenario_id",
    },
    "stress_inventory_key": {"strategy", "store_id", "ingredient_id", "scenario_id"},
}
_GOLD_MAX_MATCHES = {
    "critic_verdict": 3,
    "first_stage_order": 20,
    "inventory_daily_ledger": 50,
    "inventory_key_summary": 10,
    "lot_consumption": 20,
    "lot_expiry": 20,
    "lot_waste": 20,
    "scenario_recipe_contribution": 10,
    "stress_inventory_key": 10,
}


class ExpectedStructuredFact(StrictDecisionContract):
    claim_type: str = Field(min_length=1)
    facts: dict[str, Any]

    @field_validator("facts")
    @classmethod
    def validate_finite_facts(cls, value: dict[str, Any]) -> dict[str, Any]:
        _assert_finite(value, "facts")
        return value


class GoldEvidenceSelector(StrictDecisionContract):
    """Content-stable gold label resolved against a materialized evidence package."""

    evidence_type: str = Field(min_length=1)
    strategy: Literal["LEAN", "BALANCED", "PROTECTED"] | None = None
    entities: dict[str, str] = Field(default_factory=dict)
    source_object: str | None = None
    source_path: str | None = None
    source_path_prefix: str | None = None
    semantics: EvidenceSemantics | None = None
    payload_equals: dict[str, Any] = Field(default_factory=dict)
    payload_greater_than: dict[str, float] = Field(default_factory=dict)
    minimum_matches: int = Field(default=1, ge=0)
    maximum_matches: int | None = Field(default=1, ge=0)
    allow_missing_entity_key: bool = False

    @field_validator("entities")
    @classmethod
    def sorted_entities(cls, values: dict[str, str]) -> dict[str, str]:
        if any(not key or not value for key, value in values.items()):
            raise ValueError("Gold selector entity keys and values must be non-empty.")
        return dict(sorted(values.items()))

    @field_validator("payload_equals", "payload_greater_than")
    @classmethod
    def sorted_payload_constraints(cls, values: dict[str, Any]) -> dict[str, Any]:
        _assert_finite(values, "gold_selector_payload")
        if any(not key for key in values):
            raise ValueError("Gold selector payload keys must be non-empty.")
        return dict(sorted(values.items()))

    @model_validator(mode="after")
    def validate_cardinality(self) -> GoldEvidenceSelector:
        if self.maximum_matches is not None and self.maximum_matches < self.minimum_matches:
            raise ValueError("Gold selector maximum_matches must be >= minimum_matches.")
        if self.source_path is not None and self.source_path_prefix is not None:
            raise ValueError("Gold selector cannot combine exact and prefix source paths.")
        return self


class DecisionEvaluationCase(StrictDecisionContract):
    case_id: str = Field(min_length=1)
    decision_id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    question: str = Field(min_length=1)
    language: EvaluationLanguage
    expected_intent: str = Field(min_length=1)
    expected_answer_status: EvaluationExpectedStatus
    required_evidence_types: list[str] = Field(default_factory=list)
    gold_evidence_selectors: list[GoldEvidenceSelector] = Field(default_factory=list)
    required_entities: dict[str, str] = Field(default_factory=dict)
    required_strategy: Literal["LEAN", "BALANCED", "PROTECTED"] | None = None
    forbidden_evidence_types: list[str] = Field(default_factory=list)
    expected_structured_facts: list[ExpectedStructuredFact] = Field(default_factory=list)
    forbidden_claim_patterns: list[str] = Field(default_factory=list)
    probability_language_allowed: bool = False
    causal_language_allowed: bool = False
    adversarial_kind: AdversarialKind | None = None
    expected_error_codes: list[str] = Field(default_factory=list)
    critical: bool = True

    @field_validator(
        "required_evidence_types",
        "forbidden_evidence_types",
        "forbidden_claim_patterns",
        "expected_error_codes",
    )
    @classmethod
    def sorted_unique(cls, values: list[str]) -> list[str]:
        if any(not value for value in values):
            raise ValueError("Evaluation case lists cannot contain empty values.")
        return sorted(set(values))

    @field_validator("required_entities")
    @classmethod
    def sorted_required_entities(cls, values: dict[str, str]) -> dict[str, str]:
        if any(not key or not value for key, value in values.items()):
            raise ValueError("Required entity keys and values must be non-empty.")
        return dict(sorted(values.items()))

    @model_validator(mode="after")
    def validate_case_mode(self) -> DecisionEvaluationCase:
        if self.adversarial_kind is not None and self.expected_answer_status != "GUARD_REJECTED":
            raise ValueError("Adversarial cases must expect GUARD_REJECTED.")
        if self.expected_answer_status == "GUARD_REJECTED" and self.adversarial_kind is None:
            raise ValueError("GUARD_REJECTED requires an adversarial_kind.")
        if self.expected_answer_status == "ARTIFACT_REJECTED" and not self.expected_error_codes:
            raise ValueError("ARTIFACT_REJECTED requires expected_error_codes.")
        if self.expected_answer_status != "ARTIFACT_REJECTED" and self.expected_error_codes:
            raise ValueError("Only artifact-rejection cases may define expected_error_codes.")
        overlap = set(self.required_evidence_types) & set(self.forbidden_evidence_types)
        if overlap:
            raise ValueError(f"Evidence types cannot be required and forbidden: {sorted(overlap)}")
        if self.required_evidence_types and not self.gold_evidence_selectors:
            target_entities = dict(self.required_entities)
            if self.required_strategy is not None:
                target_entities["strategy"] = self.required_strategy
            generated = []
            for evidence_type in self.required_evidence_types:
                supported = _GOLD_ENTITY_KEYS.get(evidence_type, set())
                entities = {
                    key: value
                    for key, value in target_entities.items()
                    if key != "strategy" and key in supported
                }
                strategy = (
                    self.required_strategy
                    if self.required_strategy is not None and "strategy" in supported
                    else None
                )
                generated.append(
                    GoldEvidenceSelector(
                        evidence_type=evidence_type,
                        strategy=strategy,
                        entities=entities,
                        maximum_matches=_GOLD_MAX_MATCHES.get(evidence_type, 1),
                    )
                )
            object.__setattr__(self, "gold_evidence_selectors", generated)
        selector_types = {selector.evidence_type for selector in self.gold_evidence_selectors}
        if selector_types != set(self.required_evidence_types):
            raise ValueError(
                "Gold selector evidence types must exactly equal required_evidence_types."
            )
        return self


class DecisionBenchmarkCorpus(StrictDecisionContract):
    corpus_version: str = Field(min_length=1)
    description: str = Field(min_length=1)
    cases: list[DecisionEvaluationCase] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_cases(self) -> DecisionBenchmarkCorpus:
        identifiers = [case.case_id for case in self.cases]
        if identifiers != sorted(identifiers) or len(identifiers) != len(set(identifiers)):
            raise ValueError("Benchmark cases must have unique, sorted case_id values.")
        return self


class MetricObservation(StrictDecisionContract):
    value: float | None = Field(default=None, ge=0, le=1)
    applicable: bool
    numerator: float = Field(ge=0)
    denominator: float = Field(ge=0)
    note: str | None = None

    @model_validator(mode="after")
    def validate_applicability(self) -> MetricObservation:
        if self.applicable:
            if self.value is None or self.denominator <= 0:
                raise ValueError("Applicable metric requires a value and positive denominator.")
        elif self.value is not None:
            raise ValueError("Not-applicable metric value must be None.")
        return self


class AggregateMetricObservation(StrictDecisionContract):
    macro_value: float | None = Field(default=None, ge=0, le=1)
    micro_value: float | None = Field(default=None, ge=0, le=1)
    numerator: float = Field(ge=0)
    denominator: float = Field(ge=0)
    applicable_case_count: int = Field(ge=0)
    excluded_case_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_counts(self) -> AggregateMetricObservation:
        if self.applicable_case_count == 0:
            if self.macro_value is not None or self.micro_value is not None:
                raise ValueError("Aggregate without applicable cases must not have values.")
        elif self.macro_value is None or self.micro_value is None or self.denominator <= 0:
            raise ValueError("Aggregate with applicable cases requires macro/micro values.")
        return self


class RetrievalMetricSet(StrictDecisionContract):
    recall_at_1: MetricObservation
    recall_at_3: MetricObservation
    recall_at_5: MetricObservation
    precision_at_1: MetricObservation
    precision_at_3: MetricObservation
    precision_at_5: MetricObservation
    mean_reciprocal_rank: MetricObservation
    required_evidence_type_recall_at_5: MetricObservation
    entity_strategy_targeting_accuracy: MetricObservation
    graph_relevant_contribution_at_5: MetricObservation
    relevant_evidence_count: int = Field(ge=0)


class GroundingMetricSet(StrictDecisionContract):
    normal_guard_success: float | None = Field(default=None, ge=0, le=1)
    citation_validity: float | None = Field(default=None, ge=0, le=1)
    citation_completeness: float | None = Field(default=None, ge=0, le=1)
    structured_fact_fidelity: float | None = Field(default=None, ge=0, le=1)
    recommendation_fidelity: float | None = Field(default=None, ge=0, le=1)
    immediate_order_fidelity: float | None = Field(default=None, ge=0, le=1)
    first_stage_recourse_separation_accuracy: float | None = Field(default=None, ge=0, le=1)
    probability_semantic_violation: float | None = Field(default=None, ge=0, le=1)
    stress_as_probability_violation: float | None = Field(default=None, ge=0, le=1)
    causal_attribution_violation: float | None = Field(default=None, ge=0, le=1)
    unsupported_intent_abstention_accuracy: float | None = Field(default=None, ge=0, le=1)
    insufficient_evidence_abstention_accuracy: float | None = Field(default=None, ge=0, le=1)
    visible_text_structured_facts_consistency: float | None = Field(default=None, ge=0, le=1)
    adversarial_guard_rejection_accuracy: float | None = Field(default=None, ge=0, le=1)


class LatencySummary(StrictDecisionContract):
    samples_ms: list[float]
    sample_count: int = Field(ge=1)
    minimum_ms: float = Field(ge=0)
    maximum_ms: float = Field(ge=0)
    p50_ms: float = Field(ge=0)
    p95_ms: float = Field(ge=0)

    @field_validator("samples_ms")
    @classmethod
    def validate_samples(cls, values: list[float]) -> list[float]:
        if not values or any(value < 0 or not math.isfinite(value) for value in values):
            raise ValueError("Latency samples must be non-empty, finite and non-negative.")
        return values

    @model_validator(mode="after")
    def validate_summary(self) -> LatencySummary:
        if self.sample_count != len(self.samples_ms):
            raise ValueError("Latency sample_count mismatch.")
        if self.minimum_ms != min(self.samples_ms) or self.maximum_ms != max(self.samples_ms):
            raise ValueError("Latency min/max mismatch.")
        return self


class OperationPerformance(StrictDecisionContract):
    operation: str = Field(min_length=1)
    latency: LatencySummary
    fixture_ids: list[str] = Field(min_length=1)
    description: str = Field(min_length=1)


class CasePerformance(StrictDecisionContract):
    retrieval: LatencySummary | None = None
    generation_and_guard: LatencySummary | None = None


class EvaluationCaseResult(StrictDecisionContract):
    case_id: str
    decision_id: str
    category: str
    language: EvaluationLanguage
    critical: bool
    passed: bool
    failures: list[str]
    observed_intent: str | None = None
    observed_answer_status: str | None = None
    observed_error_codes: list[str] = Field(default_factory=list)
    structured_retrieval: RetrievalMetricSet | None = None
    structured_without_graph_retrieval: RetrievalMetricSet | None = None
    lexical_retrieval: RetrievalMetricSet | None = None
    grounding: GroundingMetricSet
    structured_ranking: list[str] = Field(default_factory=list)
    structured_without_graph_ranking: list[str] = Field(default_factory=list)
    lexical_ranking: list[str] = Field(default_factory=list)
    relevant_evidence_ids: list[str] = Field(default_factory=list)
    relevant_graph_only_ids_at_5: list[str] = Field(default_factory=list)
    gold_resolution_errors: list[str] = Field(default_factory=list)
    answer: DecisionAnswer | None = None
    guard_error: str | None = None
    performance: CasePerformance


class AggregateRetrievalMetrics(StrictDecisionContract):
    recall_at_1: AggregateMetricObservation
    recall_at_3: AggregateMetricObservation
    recall_at_5: AggregateMetricObservation
    precision_at_1: AggregateMetricObservation
    precision_at_3: AggregateMetricObservation
    precision_at_5: AggregateMetricObservation
    mean_reciprocal_rank: AggregateMetricObservation
    required_evidence_type_recall_at_5: AggregateMetricObservation
    entity_strategy_targeting_accuracy: AggregateMetricObservation
    graph_relevant_contribution_at_5: AggregateMetricObservation
    evaluated_case_count: int = Field(ge=0)


class AggregateGroundingMetrics(StrictDecisionContract):
    normal_guard_success: float = Field(ge=0, le=1)
    citation_validity: float = Field(ge=0, le=1)
    citation_completeness: float = Field(ge=0, le=1)
    structured_fact_fidelity: float = Field(ge=0, le=1)
    recommendation_fidelity: float = Field(ge=0, le=1)
    immediate_order_fidelity: float = Field(ge=0, le=1)
    first_stage_recourse_separation_accuracy: float = Field(ge=0, le=1)
    probability_semantic_violation_rate: float = Field(ge=0, le=1)
    stress_as_probability_violation_rate: float = Field(ge=0, le=1)
    causal_attribution_violation_rate: float = Field(ge=0, le=1)
    unsupported_intent_abstention_accuracy: float = Field(ge=0, le=1)
    insufficient_evidence_abstention_accuracy: float = Field(ge=0, le=1)
    visible_text_structured_facts_consistency: float = Field(ge=0, le=1)
    adversarial_guard_rejection_accuracy: float = Field(ge=0, le=1)


class DeterminismMetrics(StrictDecisionContract):
    repeated_output_equality: float = Field(ge=0, le=1)
    evidence_package_hash_equality: float = Field(ge=0, le=1)
    graph_equality: float = Field(ge=0, le=1)
    retrieval_ranking_equality: float = Field(ge=0, le=1)
    answer_equality: float = Field(ge=0, le=1)
    deterministic_tie_break_validation: float = Field(ge=0, le=1)


class GraphAblationMetrics(StrictDecisionContract):
    relevant_graph_only_id_count: int = Field(ge=0)
    recall_at_5_delta: float = Field(ge=-1, le=1)
    mrr_delta: float = Field(ge=-1, le=1)
    improved_case_count: int = Field(ge=0)
    worsened_case_count: int = Field(ge=0)
    unchanged_case_count: int = Field(ge=0)
    evaluated_case_count: int = Field(ge=0)


class BenchmarkRuntimeMetrics(StrictDecisionContract):
    full_package_build: OperationPerformance | None = None
    package_deserialization_validation: OperationPerformance
    retrieval: OperationPerformance
    generation_and_guard: OperationPerformance
    scale_fixture_materialization: OperationPerformance | None = None


class DecisionPerformance(StrictDecisionContract):
    decision_id: str
    evidence_item_count: int = Field(ge=0)
    graph_node_count: int = Field(ge=0)
    graph_edge_count: int = Field(ge=0)
    serialized_package_bytes: int = Field(ge=0)
    deserialization_validation_latency: LatencySummary


class LanguageAggregateMetrics(StrictDecisionContract):
    language: EvaluationLanguage
    case_count: int = Field(ge=0)
    passed_case_count: int = Field(ge=0)
    pass_rate: float = Field(ge=0, le=1)
    recall_at_5: AggregateMetricObservation
    required_evidence_type_recall_at_5: AggregateMetricObservation
    intent_accuracy: float = Field(ge=0, le=1)
    abstention_accuracy: float | None = Field(default=None, ge=0, le=1)
    grounding_guard_success: float | None = Field(default=None, ge=0, le=1)
    answer_localization_evaluated: bool = False


class CategoryAggregateMetrics(StrictDecisionContract):
    category: str = Field(min_length=1)
    case_count: int = Field(ge=0)
    passed_case_count: int = Field(ge=0)
    pass_rate: float = Field(ge=0, le=1)


class BenchmarkAggregateMetrics(StrictDecisionContract):
    structured_retrieval: AggregateRetrievalMetrics
    structured_without_graph_retrieval: AggregateRetrievalMetrics
    lexical_only_retrieval: AggregateRetrievalMetrics
    graph_ablation: GraphAblationMetrics
    grounding: AggregateGroundingMetrics
    determinism: DeterminismMetrics
    runtime: BenchmarkRuntimeMetrics
    passed_case_count: int = Field(ge=0)
    failed_case_count: int = Field(ge=0)
    failed_critical_case_count: int = Field(ge=0)


class AcceptanceGate(StrictDecisionContract):
    metric_name: str = Field(min_length=1)
    observed_value: float | None
    operator: GateOperator
    target: float
    passed: bool
    evaluated_case_count: int = Field(ge=0)
    note: str | None = None

    @model_validator(mode="after")
    def validate_gate(self) -> AcceptanceGate:
        if self.observed_value is None:
            if self.passed:
                raise ValueError("A not-applicable acceptance gate cannot pass.")
            return self
        expected = {
            "eq": self.observed_value == self.target,
            "gte": self.observed_value >= self.target,
            "lte": self.observed_value <= self.target,
        }[self.operator]
        if self.passed != expected:
            raise ValueError("Acceptance gate passed flag does not match its comparison.")
        return self


class AcceptanceSummary(StrictDecisionContract):
    gates: list[AcceptanceGate] = Field(min_length=1)
    overall_pass: bool

    @model_validator(mode="after")
    def validate_overall(self) -> AcceptanceSummary:
        names = [gate.metric_name for gate in self.gates]
        if names != sorted(names) or len(names) != len(set(names)):
            raise ValueError("Acceptance gates must be unique and sorted by metric_name.")
        if self.overall_pass != all(gate.passed for gate in self.gates):
            raise ValueError("overall_pass must require every acceptance gate to pass.")
        return self


class DecisionBenchmarkReport(StrictDecisionContract):
    corpus_version: str
    environment: dict[str, str]
    metric_definitions: dict[str, str]
    cases: list[EvaluationCaseResult]
    aggregate: BenchmarkAggregateMetrics
    failures_by_category: dict[str, list[str]]
    per_category: list[CategoryAggregateMetrics]
    per_language: list[LanguageAggregateMetrics]
    decision_performance: list[DecisionPerformance]
    acceptance: AcceptanceSummary
    acceptance_targets: dict[str, str]
    acceptance_results: dict[str, bool]
    overall_pass: bool
    semantic_report_hash: str
    limitations: list[str]

    @model_validator(mode="after")
    def validate_acceptance_projection(self) -> DecisionBenchmarkReport:
        expected = {gate.metric_name: gate.passed for gate in self.acceptance.gates}
        if self.acceptance_results != expected:
            raise ValueError("acceptance_results must project the typed acceptance gates.")
        if self.overall_pass != self.acceptance.overall_pass:
            raise ValueError("Report overall_pass must equal typed acceptance overall_pass.")
        if not is_full_sha256(self.semantic_report_hash):
            raise ValueError("semantic_report_hash must be a full SHA-256 digest.")
        return self
