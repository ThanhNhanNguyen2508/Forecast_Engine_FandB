from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from shelfcash_forecast.decision_intelligence.contracts import StrictDecisionContract
from shelfcash_forecast.decision_intelligence.integrity import sha256_content_hash

Part2Scenario = Literal[
    "READ_ONLY",
    "WHAT_IF_DRAFT",
    "WHAT_IF_EXECUTE",
    "MISSING_CONFIRMATION",
    "DEMAND_CHANGE",
    "SUPPLIER_CHANGE",
    "INVENTORY_CHANGE",
    "NO_VALID_PLAN",
    "PROBABILITY_SEMANTICS",
    "STRESS_SEMANTICS",
    "COMPARISON",
    "COUNTERFACTUAL",
    "REGRET",
    "APPROVAL",
    "STALE_APPROVAL",
    "UNKNOWN_ENTITY",
    "ADVERSARIAL_STRATEGY",
    "ADVERSARIAL_ORDER",
    "ADVERSARIAL_PROBABILITY",
    "ADVERSARIAL_REGRET",
    "ADVERSARIAL_CITATION",
    "CROSS_PACKAGE_CITATION",
    "MULTI_STORE_SELECTOR",
    "RECOURSE_SEPARATION",
    "NO_MUTATION",
    "DETERMINISM",
]


class Part2BenchmarkCase(StrictDecisionContract):
    case_id: str = Field(pattern=r"^[0-9]{3}-[a-z0-9-]+$")
    category: str = Field(min_length=1)
    language: Literal["en", "vi"]
    scenario: Part2Scenario
    question: str = Field(min_length=1)
    expected_status: str = Field(min_length=1)
    expected_intent: str | None = None
    expected_tool: str | None = None
    critical: bool = True


class Part2BenchmarkCorpus(StrictDecisionContract):
    corpus_version: Literal["shelfcash-m6-part2-deterministic-v1"]
    fixed_seed: int
    cases: list[Part2BenchmarkCase] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_cases(self) -> Part2BenchmarkCorpus:
        identifiers = [case.case_id for case in self.cases]
        if identifiers != sorted(identifiers) or len(identifiers) != len(set(identifiers)):
            raise ValueError("M6_PART2_CORPUS_ID_ORDER_INVALID")
        if {case.language for case in self.cases} != {"en", "vi"}:
            raise ValueError("M6_PART2_CORPUS_BILINGUAL_REQUIRED")
        return self


class Part2CaseChecks(StrictDecisionContract):
    intent_accuracy: bool | None = None
    entity_selector_accuracy: bool | None = None
    tool_routing_accuracy: bool | None = None
    unauthorized_tool_call: bool | None = None
    no_mutation_accuracy: bool | None = None
    computation_authority_fidelity: bool | None = None
    recommendation_fidelity: bool | None = None
    order_recourse_fidelity: bool | None = None
    comparison_fidelity: bool | None = None
    counterfactual_target_fidelity: bool | None = None
    regret_component_fidelity: bool | None = None
    approval_transition_fidelity: bool | None = None
    citation_validity: bool | None = None
    citation_completeness: bool | None = None
    structured_fact_fidelity: bool | None = None
    visible_text_facts_consistency: bool | None = None
    adversarial_guard_rejection: bool | None = None
    probability_violation: bool | None = None
    stress_probability_violation: bool | None = None
    causal_violation: bool | None = None
    deterministic_repeatability: bool | None = None


class Part2CaseResult(StrictDecisionContract):
    case_id: str
    category: str
    language: Literal["en", "vi"]
    observed_status: str
    observed_intent: str | None = None
    observed_tool: str | None = None
    checks: Part2CaseChecks
    passed: bool
    failures: list[str]


class Part2Rate(StrictDecisionContract):
    value: float = Field(ge=0, le=1)
    numerator: int = Field(ge=0)
    denominator: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_ratio(self) -> Part2Rate:
        expected = self.numerator / self.denominator if self.denominator else 0.0
        if abs(self.value - expected) > 1e-12:
            raise ValueError("M6_PART2_RATE_MISMATCH")
        return self


class Part2AggregateMetrics(StrictDecisionContract):
    intent_accuracy: Part2Rate
    entity_selector_accuracy: Part2Rate
    tool_routing_accuracy: Part2Rate
    unauthorized_tool_call_rate: Part2Rate
    no_mutation_accuracy: Part2Rate
    computation_authority_fidelity: Part2Rate
    recommendation_fidelity: Part2Rate
    order_recourse_fidelity: Part2Rate
    comparison_fidelity: Part2Rate
    counterfactual_target_fidelity: Part2Rate
    regret_component_fidelity: Part2Rate
    approval_transition_fidelity: Part2Rate
    citation_validity: Part2Rate
    citation_completeness: Part2Rate
    structured_fact_fidelity: Part2Rate
    visible_text_facts_consistency: Part2Rate
    adversarial_guard_rejection: Part2Rate
    probability_violation_rate: Part2Rate
    stress_as_probability_violation_rate: Part2Rate
    causal_violation_rate: Part2Rate
    deterministic_repeatability: Part2Rate


class Part2LanguageResult(StrictDecisionContract):
    language: Literal["en", "vi"]
    case_count: int
    passed_count: int
    pass_rate: float = Field(ge=0, le=1)
    intent_accuracy: float = Field(ge=0, le=1)


class Part2Latency(StrictDecisionContract):
    operation: str
    sample_count: int = Field(gt=0)
    p50_ms: float = Field(ge=0)
    p95_ms: float = Field(ge=0)
    minimum_ms: float = Field(ge=0)
    maximum_ms: float = Field(ge=0)
    description: str


class Part2AcceptanceGate(StrictDecisionContract):
    metric_name: str
    observed_value: float
    operator: Literal["EQ", "GE", "LE"]
    target: float
    passed: bool
    evaluated_case_count: int
    note: str


class Part2BenchmarkReport(StrictDecisionContract):
    corpus_version: str
    environment: dict[str, str]
    metric_definitions: dict[str, str]
    case_results: list[Part2CaseResult]
    aggregate_metrics: Part2AggregateMetrics
    per_language: list[Part2LanguageResult]
    performance: list[Part2Latency]
    size_statistics: dict[str, int]
    acceptance_gates: list[Part2AcceptanceGate]
    failed_critical_cases: list[str]
    overall_pass: bool
    limitations: list[str]
    semantic_hash: str = ""

    @model_validator(mode="after")
    def validate_report(self) -> Part2BenchmarkReport:
        expected_overall = not self.failed_critical_cases and all(
            gate.passed for gate in self.acceptance_gates
        )
        if self.overall_pass != expected_overall:
            raise ValueError("M6_PART2_ACCEPTANCE_SUMMARY_MISMATCH")
        material = self.model_dump(
            mode="json",
            exclude={"environment", "performance", "semantic_hash"},
        )
        expected = sha256_content_hash(material)
        if self.semantic_hash and self.semantic_hash != expected:
            raise ValueError("M6_PART2_SEMANTIC_HASH_MISMATCH")
        object.__setattr__(self, "semantic_hash", expected)
        return self


class Part2FixtureBundle(StrictDecisionContract):
    baseline_request: Any
    baseline_result: Any
    baseline_decision: Any
