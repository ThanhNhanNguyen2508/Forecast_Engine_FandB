# Object đầu vào trung tâm:

# DecisionIntelligenceInput

# Object đầu ra trung tâm:

# FinalDecisionPackage
from __future__ import annotations

import math
from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from shelfcash_forecast.bom.contracts import IngredientDemandPackage
from shelfcash_forecast.contracts import ForecastPackage
from shelfcash_forecast.decision_intelligence.integrity import (
    EVIDENCE_SCHEMA_VERSION,
    evidence_content_hash,
    evidence_package_hash,
    is_full_sha256,
)
from shelfcash_forecast.optimization.contracts import (
    OptimizationRequest,
    OptimizationResult,
    StrategyName,
)
from shelfcash_forecast.scenario.contracts import (
    IngredientDemandScenarioBundle,
    ProductDemandScenarioBundle,
)

EvidenceLayer = Literal["M1", "M2", "M3", "M4", "M5", "M6"]
EvidenceSemantics = Literal[
    "deterministic",
    "quantile",
    "probabilistic",
    "stress",
    "solver_estimate",
    "exact_simulation",
    "critic_verdict",
]
ReadinessStatus = Literal["VERIFIED", "WARNING", "PARTIAL", "FAILED", "UNAVAILABLE"]
AnswerStatus = Literal[
    "GROUNDED",
    "PARTIAL",
    "INSUFFICIENT_EVIDENCE",
    "UNSUPPORTED_INTENT",
]
CoherenceSeverity = Literal["WARNING", "ERROR"]


def _assert_finite(value: Any, path: str = "value") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} must not contain NaN or infinity.")
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_finite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple, set)):
        for index, item in enumerate(value):
            _assert_finite(item, f"{path}[{index}]")


class StrictDecisionContract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ArtifactCoherenceIssue(StrictDecisionContract):
    code: str = Field(min_length=1)
    severity: CoherenceSeverity
    message: str = Field(min_length=1)
    artifact_paths: list[str] = Field(default_factory=list)
    blocking: bool = False


class ArtifactCoherenceResult(StrictDecisionContract):
    status: Literal["VERIFIED", "WARNING", "FAILED"]
    issues: list[ArtifactCoherenceIssue] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_status(self) -> ArtifactCoherenceResult:
        expected = (
            "FAILED"
            if any(issue.severity == "ERROR" for issue in self.issues)
            else "WARNING"
            if self.issues
            else "VERIFIED"
        )
        if self.status != expected:
            raise ValueError("Artifact coherence status does not match issue severity.")
        codes = [issue.code for issue in self.issues]
        if codes != sorted(codes) or len(codes) != len(set(codes)):
            raise ValueError("Artifact coherence issues must have unique, sorted codes.")
        return self


class DecisionIntelligenceInput(StrictDecisionContract):
    """Read-only M1-M5 artifacts consumed by Decision Intelligence."""

    optimization_request: OptimizationRequest
    optimization_result: OptimizationResult
    forecast_package: ForecastPackage | None = None
    ingredient_demand_package: IngredientDemandPackage | None = None
    ingredient_scenario_bundle: IngredientDemandScenarioBundle | None = None
    product_scenario_bundle: ProductDemandScenarioBundle | None = None
    coherence: ArtifactCoherenceResult | None = None

    @model_validator(mode="after")
    def validate_coherence(self) -> DecisionIntelligenceInput:
        from shelfcash_forecast.decision_intelligence.coherence import (
            evaluate_artifact_coherence,
        )

        result = evaluate_artifact_coherence(self)
        object.__setattr__(self, "coherence", result)
        blocking = [issue.code for issue in result.issues if issue.blocking]
        if blocking:
            raise ValueError(f"M6_ARTIFACT_COHERENCE_FAILED:{','.join(blocking)}")
        return self


class EvidenceItem(StrictDecisionContract):
    schema_version: Literal["m6-evidence-v2"] = EVIDENCE_SCHEMA_VERSION
    evidence_id: str = Field(min_length=12)
    content_hash: str = ""
    layer: EvidenceLayer
    evidence_type: str = Field(min_length=1)
    source_object: str = Field(min_length=1)
    source_path: str = Field(min_length=1)
    semantics: EvidenceSemantics
    entities: dict[str, str] = Field(default_factory=dict)
    event_date: date | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    text: str = Field(min_length=1)
    warnings: list[str] = Field(default_factory=list)

    @field_validator("payload")
    @classmethod
    def validate_payload_finite(cls, value: dict[str, Any]) -> dict[str, Any]:
        _assert_finite(value, "payload")
        return value

    @model_validator(mode="after")
    def validate_content_hash(self) -> EvidenceItem:
        material = {
            "schema_version": self.schema_version,
            "evidence_id": self.evidence_id,
            "layer": self.layer,
            "evidence_type": self.evidence_type,
            "source_object": self.source_object,
            "source_path": self.source_path,
            "semantics": self.semantics,
            "entities": self.entities,
            "event_date": self.event_date,
            "payload": self.payload,
            "text": self.text,
            "warnings": self.warnings,
        }
        expected = evidence_content_hash(material)
        if not self.content_hash:
            object.__setattr__(self, "content_hash", expected)
        elif not is_full_sha256(self.content_hash) or self.content_hash != expected:
            raise ValueError("EVIDENCE_CONTENT_HASH_MISMATCH")
        return self


class EvidencePackage(StrictDecisionContract):
    schema_version: Literal["m6-evidence-v2"] = EVIDENCE_SCHEMA_VERSION
    request_id: str = Field(min_length=1)
    items: list[EvidenceItem]
    source_layers: list[EvidenceLayer]
    package_hash: str = ""
    provenance: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_identifiers(self) -> EvidencePackage:
        identifiers = [item.evidence_id for item in self.items]
        if identifiers != sorted(identifiers):
            raise ValueError("Evidence items must use deterministic evidence_id ordering.")
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Evidence identifiers must be unique.")
        if self.source_layers != sorted(set(self.source_layers)):
            raise ValueError("Evidence source layers must be unique and sorted.")
        expected_hash = evidence_package_hash(
            request_id=self.request_id,
            schema_version=self.schema_version,
            source_layers=self.source_layers,
            ordered_items=[(item.evidence_id, item.content_hash) for item in self.items],
        )
        if not self.package_hash:
            object.__setattr__(self, "package_hash", expected_hash)
        elif not is_full_sha256(self.package_hash) or self.package_hash != expected_hash:
            raise ValueError("EVIDENCE_PACKAGE_HASH_MISMATCH")
        return self


class GraphNode(StrictDecisionContract):
    node_id: str = Field(min_length=1)
    node_type: str = Field(min_length=1)
    label: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)


class GraphEdge(StrictDecisionContract):
    edge_id: str = Field(min_length=1)
    source_node_id: str = Field(min_length=1)
    target_node_id: str = Field(min_length=1)
    edge_type: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)


class DecisionGraph(StrictDecisionContract):
    request_id: str = Field(min_length=1)
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    provenance: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_graph(self) -> DecisionGraph:
        node_ids = [node.node_id for node in self.nodes]
        edge_ids = [edge.edge_id for edge in self.edges]
        if node_ids != sorted(node_ids) or edge_ids != sorted(edge_ids):
            raise ValueError("Decision graph nodes and edges must be deterministically sorted.")
        if len(node_ids) != len(set(node_ids)) or len(edge_ids) != len(set(edge_ids)):
            raise ValueError("Decision graph identifiers must be unique.")
        known = set(node_ids)
        if any(
            edge.source_node_id not in known or edge.target_node_id not in known
            for edge in self.edges
        ):
            raise ValueError("Decision graph edge references an unknown node.")
        return self


class OrderExplanation(StrictDecisionContract):
    evidence_id: str
    decision_stage: Literal["first_stage", "scenario_recourse"]
    scenario_id: str | None = None
    offer_id: str
    supplier_id: str
    store_id: str
    ingredient_id: str
    unit: str
    order_date: date
    arrival_date: date
    pack_count: int = Field(ge=0)
    order_quantity: float = Field(ge=0)
    purchase_cost: float = Field(ge=0)
    delivery_cost: float = Field(ge=0)
    emergency: bool

    @model_validator(mode="after")
    def validate_stage(self) -> OrderExplanation:
        if self.decision_stage == "first_stage" and self.scenario_id is not None:
            raise ValueError("First-stage order cannot carry a scenario_id.")
        if self.decision_stage == "scenario_recourse" and self.scenario_id is None:
            raise ValueError("Scenario recourse order requires a scenario_id.")
        return self


class ForecastExplanation(StrictDecisionContract):
    evidence_id: str
    store_id: str
    product_id: str
    product_name: str
    target_date: date
    model_version: str
    p25: float = Field(ge=0)
    p50: float = Field(ge=0)
    p75: float = Field(ge=0)
    interval_lower: float = Field(ge=0)
    interval_upper: float = Field(ge=0)
    quantile_spread: float = Field(ge=0)
    interval_width: float = Field(ge=0)
    baseline_p50: float = Field(ge=0)
    delta_vs_baseline: float
    calibration_source: str
    causal_attribution_status: Literal["UNAVAILABLE"] = "UNAVAILABLE"
    warnings: list[str] = Field(default_factory=list)


class BOMContributionExplanation(StrictDecisionContract):
    evidence_id: str
    product_id: str
    product_name: str
    recipe_id: str
    recipe_version: str
    unit: str
    base_quantity: float | None = Field(default=None, ge=0)
    final_quantity: float = Field(ge=0)
    forecast_quantity: float | None = Field(default=None, ge=0)
    recipe_quantity: float | None = Field(default=None, ge=0)
    recipe_unit: str | None = None
    yield_quantity: float | None = Field(default=None, gt=0)
    yield_unit: str | None = None
    process_loss_rate: float | None = Field(default=None, ge=0)
    waste_allowance_rate: float | None = Field(default=None, ge=0)


class BOMExplanation(StrictDecisionContract):
    evidence_id: str
    status: Literal["AVAILABLE", "PARTIAL"]
    semantics: Literal["quantile", "probabilistic"]
    store_id: str
    ingredient_id: str
    ingredient_name: str | None = None
    target_date: date
    unit: str
    scenario_id: str | None = None
    probability_weight: float | None = Field(default=None, ge=0, le=1)
    p25: float | None = Field(default=None, ge=0)
    p50: float | None = Field(default=None, ge=0)
    p75: float | None = Field(default=None, ge=0)
    scenario_quantity: float | None = Field(default=None, ge=0)
    contributions: list[BOMContributionExplanation] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class InventoryKeyExplanation(StrictDecisionContract):
    evidence_id: str
    strategy: StrategyName
    scenario_id: str
    semantics: Literal["deterministic", "quantile", "probabilistic", "stress"]
    probability_weight: float | None = Field(default=None, ge=0, le=1)
    store_id: str
    ingredient_id: str
    unit: str
    beginning_inventory: float = Field(ge=0)
    inbound: float = Field(ge=0)
    demand: float = Field(ge=0)
    fulfilled: float = Field(ge=0)
    shortage: float = Field(ge=0)
    expired: float = Field(ge=0)
    waste: float = Field(ge=0)
    ending_inventory: float = Field(ge=0)
    maximum_inventory: float = Field(ge=0)
    fill_rate: float = Field(ge=0, le=1)
    projected_stockout_date: date | None = None
    at_risk_expiry_quantity: float = Field(ge=0)
    capacity_violation_quantity: float | None = Field(default=None, ge=0)
    consequence_cost: float | None = Field(default=None, ge=0)
    accounting_valid: bool
    warnings: list[str] = Field(default_factory=list)


class InventoryRiskExplanation(StrictDecisionContract):
    evidence_id: str
    strategy: StrategyName
    store_id: str
    ingredient_id: str
    unit: str
    stockout_probability: float = Field(ge=0, le=1)
    expected_shortage: float = Field(ge=0)
    p95_shortage: float = Field(ge=0)
    expected_fill_rate: float = Field(ge=0, le=1)
    expected_consequence_cost: float | None = Field(default=None, ge=0)
    p95_consequence_cost: float | None = Field(default=None, ge=0)
    cvar95_consequence_cost: float | None = Field(default=None, ge=0)


class InventoryTraceExplanation(StrictDecisionContract):
    evidence_id: str
    strategy: StrategyName
    scenario_id: str
    trace_type: Literal["FEFO_CONSUMPTION", "EXPIRY", "WASTE"]
    simulation_date: date
    store_id: str
    ingredient_id: str
    lot_id: str
    unit: str
    quantity: float = Field(gt=0)
    event_id: str | None = None
    expiry_date: date | None = None


class StressExplanation(StrictDecisionContract):
    evidence_id: str
    strategy: StrategyName
    stress_id: str
    description: str | None = None
    demand_multiplier: float | None = Field(default=None, ge=0)
    supplier_delay_days: int | None = Field(default=None, ge=0)
    supplier_ids: list[str] = Field(default_factory=list)
    probabilistic: Literal[False] = False
    inventory_keys: list[InventoryKeyExplanation]
    warnings: list[str] = Field(default_factory=list)


class StrategyProfileExplanation(StrictDecisionContract):
    evidence_id: str
    strategy: StrategyName
    source_status: Literal["VERIFIED_DECISION_INPUT", "RECONSTRUCTED_CURRENT_DEFAULT"]
    shortage_penalty: float = Field(ge=0)
    holding_penalty: float = Field(ge=0)
    waste_penalty: float = Field(ge=0)
    cash_penalty: float = Field(ge=0)
    cvar_weight: float = Field(ge=0)
    cvar_alpha: float = Field(gt=0, lt=1)
    maximum_stockout_probability: float | None = Field(default=None, ge=0, le=1)
    minimum_expected_fill_rate: float | None = Field(default=None, ge=0, le=1)
    minimum_fill_rate: float | None = Field(default=None, ge=0, le=1)
    required_fill_rate_probability: float | None = Field(default=None, ge=0, le=1)
    minimum_acceptable_fill_rate: float = Field(ge=0, le=1)
    maximum_acceptable_stockout_probability: float = Field(ge=0, le=1)
    maximum_fill_rate_model_gap: float = Field(ge=0, le=1)
    maximum_stockout_probability_model_gap: float = Field(ge=0, le=1)


class CandidateSummary(StrictDecisionContract):
    strategy: StrategyName
    strategy_profile: StrategyProfileExplanation
    plan_id: str
    solver: str | None = None
    formulation: str | None = None
    solver_status: str
    completed: bool
    purchase_cost: float = Field(ge=0)
    expected_recourse_cost: float = Field(ge=0)
    objective_value: float | None = None
    cvar_alpha: float | None = Field(default=None, gt=0, lt=1)
    cvar_weight: float | None = Field(default=None, ge=0)
    estimated_cvar: float | None = Field(default=None, ge=0)
    predicted_expected_fill_rate: float | None = Field(default=None, ge=0, le=1)
    predicted_stockout_probability: float | None = Field(default=None, ge=0, le=1)
    exact_mean_key_fill_rate: float | None = Field(default=None, ge=0, le=1)
    exact_stockout_probability: float | None = Field(default=None, ge=0, le=1)
    first_stage_orders: list[OrderExplanation]
    scenario_recourse_orders: list[OrderExplanation]
    critic_passed: bool
    critic_checks: dict[str, bool]
    critic_details: dict[str, Any]
    hard_violations: list[str]
    warnings: list[str]
    evidence_ids: list[str]


class ReadinessDimension(StrictDecisionContract):
    status: ReadinessStatus
    reason: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ConfidenceDecomposition(StrictDecisionContract):
    artifact_coherence: ReadinessDimension
    forecast_evidence: ReadinessDimension
    scenario_evidence: ReadinessDimension
    bom_traceability: ReadinessDimension
    inventory_validation: ReadinessDimension
    optimization_validity: ReadinessDimension
    stress_evidence: ReadinessDimension
    overall_decision_readiness: ReadinessDimension
    probability_score: None = None

    @model_validator(mode="after")
    def validate_authority_lattice(self) -> ConfidenceDecomposition:
        required = (
            self.artifact_coherence,
            self.inventory_validation,
            self.optimization_validity,
        )
        if any(dimension.status == "FAILED" for dimension in required) and (
            self.overall_decision_readiness.status != "FAILED"
        ):
            raise ValueError("READINESS_LATTICE_AUTHORITY_FAILURE_NOT_PROPAGATED")
        return self


class GroundedClaim(StrictDecisionContract):
    claim_type: str = Field(min_length=1)
    text: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)
    facts: dict[str, Any] = Field(default_factory=dict)
    uses_probability_language: bool = False
    causal: bool = False

    @field_validator("facts")
    @classmethod
    def validate_facts_finite(cls, value: dict[str, Any]) -> dict[str, Any]:
        _assert_finite(value, "facts")
        return value


class DecisionAnswer(StrictDecisionContract):
    question: str = Field(min_length=1)
    intent: str = Field(min_length=1)
    status: AnswerStatus
    answer_text: str = Field(min_length=1)
    claims: list[GroundedClaim] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    retrieved_evidence_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)


class FinalDecisionPackage(StrictDecisionContract):
    request_id: str
    decision_date: date
    planning_end_date: date
    decision_status: str
    recommended_strategy: StrategyName | None = None
    recommended_plan_summary: CandidateSummary | None = None
    immediate_orders: list[OrderExplanation]
    conditional_recourse: list[OrderExplanation]
    strategy_comparison: list[CandidateSummary]
    forecast_explanations: list[ForecastExplanation]
    bom_explanations: list[BOMExplanation]
    inventory_explanations: list[InventoryKeyExplanation]
    inventory_risk_explanations: list[InventoryRiskExplanation]
    inventory_traces: list[InventoryTraceExplanation]
    stress_explanations: list[StressExplanation]
    confidence_decomposition: ConfidenceDecomposition
    evidence_package: EvidencePackage
    decision_graph: DecisionGraph
    narrative_summary: DecisionAnswer | None = None
    warnings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_recommendation(self) -> FinalDecisionPackage:
        if self.evidence_package.request_id != self.request_id:
            raise ValueError("Evidence package request_id mismatch.")
        if self.decision_graph.request_id != self.request_id:
            raise ValueError("Decision graph request_id mismatch.")
        recommendation_items = [
            item for item in self.evidence_package.items if item.evidence_type == "recommendation"
        ]
        if len(recommendation_items) != 1:
            raise ValueError("Final package requires exactly one M5 recommendation evidence item.")
        if recommendation_items[0].payload.get("recommended_strategy") != self.recommended_strategy:
            raise ValueError("M6 recommendation does not match M5 evidence.")
        if self.recommended_strategy is None:
            if (
                self.recommended_plan_summary is not None
                or self.immediate_orders
                or self.conditional_recourse
            ):
                raise ValueError("No-valid-plan package cannot contain a recommendation.")
        else:
            if self.recommended_plan_summary is None:
                raise ValueError("Recommended strategy requires its candidate summary.")
            if self.recommended_plan_summary.strategy != self.recommended_strategy:
                raise ValueError("Recommended plan strategy mismatch.")
            if not self.recommended_plan_summary.critic_passed:
                raise ValueError("M6 cannot recommend a critic-failed candidate.")
            if any(order.decision_stage != "first_stage" for order in self.immediate_orders):
                raise ValueError("Immediate orders must contain first-stage decisions only.")
            if any(
                order.decision_stage != "scenario_recourse" for order in self.conditional_recourse
            ):
                raise ValueError("Conditional recourse must contain scenario decisions only.")
            if self.immediate_orders != self.recommended_plan_summary.first_stage_orders:
                raise ValueError("Immediate orders must exactly match the M5 first-stage plan.")
            if self.conditional_recourse != self.recommended_plan_summary.scenario_recourse_orders:
                raise ValueError("Conditional recourse must exactly match the M5 candidate.")
        if self.narrative_summary is not None:
            known = {item.evidence_id for item in self.evidence_package.items}
            cited = set(self.narrative_summary.citations)
            cited.update(
                evidence_id
                for claim in self.narrative_summary.claims
                for evidence_id in claim.evidence_ids
            )
            if cited - known:
                raise ValueError("Narrative summary cites evidence outside this decision.")
        return self


class RetrievedEvidence(StrictDecisionContract):
    query: str
    items: list[EvidenceItem]
    scores: dict[str, float]
    intent: str
