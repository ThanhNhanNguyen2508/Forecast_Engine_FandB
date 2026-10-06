from __future__ import annotations

from datetime import date
from typing import Annotated, Any, Literal

from pydantic import Field, field_validator, model_validator

from shelfcash_forecast.decision_intelligence.contracts import (
    FinalDecisionPackage,
    StrictDecisionContract,
)
from shelfcash_forecast.decision_intelligence.integrity import (
    is_full_sha256,
    sha256_content_hash,
)
from shelfcash_forecast.inventory.contracts import InventoryLot
from shelfcash_forecast.optimization.contracts import (
    OptimizationRequest,
    OptimizationResult,
    StrategyName,
)

PackageRole = Literal["BASELINE", "HYPOTHETICAL", "DELTA", "REGRET", "APPROVAL"]
WhatIfExecutionMode = Literal["DRAFT_ONLY", "EXECUTE_HYPOTHETICAL"]


class DemandSelector(StrictDecisionContract):
    scenario_id: str | None = None
    store_id: str | None = None
    ingredient_id: str | None = None
    unit: str | None = None
    target_date: date | None = None
    expected_matches: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def require_scope(self) -> DemandSelector:
        if not any(
            (
                self.scenario_id,
                self.store_id,
                self.ingredient_id,
                self.unit,
                self.target_date,
            )
        ):
            raise ValueError("M6_WHAT_IF_SELECTOR_SCOPE_REQUIRED")
        return self


class DemandScaleModification(StrictDecisionContract):
    modification_type: Literal["DEMAND_SCALE"] = "DEMAND_SCALE"
    selector: DemandSelector
    multiplier: float = Field(gt=0)


class SupplierOfferModification(StrictDecisionContract):
    modification_type: Literal["SUPPLIER_OFFER"] = "SUPPLIER_OFFER"
    offer_id: str = Field(min_length=1)
    available: bool | None = None
    unit_price: float | None = Field(default=None, ge=0)
    delivery_cost: float | None = Field(default=None, ge=0)
    minimum_order_quantity: float | None = Field(default=None, ge=0)
    maximum_order_quantity: float | None = Field(default=None, gt=0)
    clear_maximum_order_quantity: bool = False
    lead_time_days: int | None = Field(default=None, ge=0)
    shelf_life_days: int | None = Field(default=None, ge=0)
    clear_shelf_life_days: bool = False
    order_cutoff_date: date | None = None
    clear_order_cutoff_date: bool = False
    emergency: bool | None = None

    @model_validator(mode="after")
    def require_change(self) -> SupplierOfferModification:
        values = (
            self.available,
            self.unit_price,
            self.delivery_cost,
            self.minimum_order_quantity,
            self.maximum_order_quantity,
            self.lead_time_days,
            self.shelf_life_days,
            self.order_cutoff_date,
            self.emergency,
        )
        if not any(value is not None for value in values) and not any(
            (
                self.clear_maximum_order_quantity,
                self.clear_shelf_life_days,
                self.clear_order_cutoff_date,
            )
        ):
            raise ValueError("M6_WHAT_IF_EMPTY_SUPPLIER_OFFER_CHANGE")
        return self


class InventoryLotModification(StrictDecisionContract):
    modification_type: Literal["INVENTORY_LOT"] = "INVENTORY_LOT"
    action: Literal["SET_QUANTITY", "SET_EXPIRY", "ADD", "REMOVE"]
    lot_id: str = Field(min_length=1)
    store_id: str | None = None
    ingredient_id: str | None = None
    unit: str | None = None
    quantity: float | None = Field(default=None, ge=0)
    expiry_date: date | None = None
    clear_expiry: bool = False
    lot: InventoryLot | None = None

    @model_validator(mode="after")
    def validate_action(self) -> InventoryLotModification:
        if self.action == "SET_QUANTITY" and self.quantity is None:
            raise ValueError("M6_WHAT_IF_LOT_QUANTITY_REQUIRED")
        if self.action == "SET_EXPIRY" and self.expiry_date is None and not self.clear_expiry:
            raise ValueError("M6_WHAT_IF_LOT_EXPIRY_REQUIRED")
        if self.action == "ADD":
            if self.lot is None or self.lot.lot_id != self.lot_id:
                raise ValueError("M6_WHAT_IF_ADDED_LOT_ID_MISMATCH")
            if self.store_id is not None and self.store_id != self.lot.store_id:
                raise ValueError("M6_WHAT_IF_ADDED_LOT_STORE_MISMATCH")
            if self.ingredient_id is not None and self.ingredient_id != self.lot.ingredient_id:
                raise ValueError("M6_WHAT_IF_ADDED_LOT_INGREDIENT_MISMATCH")
            if self.unit is not None and self.unit != self.lot.unit:
                raise ValueError("M6_WHAT_IF_ADDED_LOT_UNIT_MISMATCH")
        elif self.lot is not None:
            raise ValueError("M6_WHAT_IF_LOT_PAYLOAD_ONLY_ALLOWED_FOR_ADD")
        return self


class BudgetModification(StrictDecisionContract):
    modification_type: Literal["BUDGET"] = "BUDGET"
    budget: float | None = Field(default=None, ge=0)
    clear_budget: bool = False

    @model_validator(mode="after")
    def validate_budget(self) -> BudgetModification:
        if self.budget is None and not self.clear_budget:
            raise ValueError("M6_WHAT_IF_BUDGET_VALUE_REQUIRED")
        if self.budget is not None and self.clear_budget:
            raise ValueError("M6_WHAT_IF_BUDGET_CHANGE_CONFLICT")
        return self


class InventoryPolicyModification(StrictDecisionContract):
    modification_type: Literal["INVENTORY_POLICY"] = "INVENTORY_POLICY"
    expiry_inclusive: bool | None = None
    unknown_expiry: Literal["reject", "warn_and_place_last"] | None = None
    accounting_tolerance: float | None = Field(default=None, gt=0)
    at_risk_expiry_days: int | None = Field(default=None, ge=0)
    waste_threshold: float | None = Field(default=None, ge=0)
    fill_rate_target: float | None = Field(default=None, ge=0, le=1)
    trace_retention: Literal["full", "summary", "selected"] | None = None

    @model_validator(mode="after")
    def require_change(self) -> InventoryPolicyModification:
        if all(
            value is None
            for name, value in self.model_dump().items()
            if name != "modification_type"
        ):
            raise ValueError("M6_WHAT_IF_EMPTY_INVENTORY_POLICY_CHANGE")
        return self


class StrategyProfileModification(StrictDecisionContract):
    modification_type: Literal["STRATEGY_PROFILE"] = "STRATEGY_PROFILE"
    strategy: StrategyName
    shortage_penalty: float | None = Field(default=None, ge=0)
    holding_penalty: float | None = Field(default=None, ge=0)
    waste_penalty: float | None = Field(default=None, ge=0)
    cash_penalty: float | None = Field(default=None, ge=0)
    cvar_weight: float | None = Field(default=None, ge=0)
    cvar_alpha: float | None = Field(default=None, gt=0, lt=1)
    maximum_stockout_probability: float | None = Field(default=None, ge=0, le=1)
    minimum_expected_fill_rate: float | None = Field(default=None, ge=0, le=1)
    minimum_acceptable_fill_rate: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def require_change(self) -> StrategyProfileModification:
        if all(
            value is None
            for name, value in self.model_dump().items()
            if name not in {"modification_type", "strategy"}
        ):
            raise ValueError("M6_WHAT_IF_EMPTY_STRATEGY_PROFILE_CHANGE")
        return self


class ConsequenceCostModification(StrictDecisionContract):
    modification_type: Literal["CONSEQUENCE_COST"] = "CONSEQUENCE_COST"
    store_id: str = Field(min_length=1)
    ingredient_id: str = Field(min_length=1)
    unit: str = Field(min_length=1)
    holding_cost_per_unit_day: float | None = Field(default=None, ge=0)
    shortage_cost_per_unit: float | None = Field(default=None, ge=0)
    expired_cost_per_unit: float | None = Field(default=None, ge=0)
    waste_cost_per_unit: float | None = Field(default=None, ge=0)
    capacity_quantity: float | None = Field(default=None, gt=0)
    clear_capacity_quantity: bool = False

    @model_validator(mode="after")
    def require_change(self) -> ConsequenceCostModification:
        if (
            not any(
                value is not None
                for name, value in self.model_dump().items()
                if name not in {"modification_type", "store_id", "ingredient_id", "unit"}
            )
            and not self.clear_capacity_quantity
        ):
            raise ValueError("M6_WHAT_IF_EMPTY_CONSEQUENCE_COST_CHANGE")
        return self


class StressScenarioModification(StrictDecisionContract):
    modification_type: Literal["STRESS_SCENARIO"] = "STRESS_SCENARIO"
    stress_id: str = Field(min_length=1)
    demand_multiplier: float | None = Field(default=None, gt=0)
    supplier_delay_days: int | None = Field(default=None, ge=0)
    supplier_ids: set[str] | None = None
    preserve_remaining_shelf_life: bool | None = None
    description: str | None = None

    @model_validator(mode="after")
    def require_change(self) -> StressScenarioModification:
        if all(
            value is None
            for name, value in self.model_dump().items()
            if name not in {"modification_type", "stress_id"}
        ):
            raise ValueError("M6_WHAT_IF_EMPTY_STRESS_CHANGE")
        return self


WhatIfModification = Annotated[
    DemandScaleModification
    | SupplierOfferModification
    | InventoryLotModification
    | BudgetModification
    | InventoryPolicyModification
    | StrategyProfileModification
    | ConsequenceCostModification
    | StressScenarioModification,
    Field(discriminator="modification_type"),
]


def _request_material(values: dict[str, Any]) -> dict[str, Any]:
    excluded = {"what_if_id", "content_hash"}
    return {key: value for key, value in values.items() if key not in excluded}


class WhatIfRequest(StrictDecisionContract):
    what_if_id: str = ""
    content_hash: str = ""
    baseline_request_id: str = Field(min_length=1)
    baseline_decision_hash: str
    baseline_request_hash: str
    idempotency_key: str = Field(min_length=1)
    modifications: list[WhatIfModification] = Field(min_length=1)
    actor: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    execution_mode: WhatIfExecutionMode = "DRAFT_ONLY"
    confirmed: bool = False

    @field_validator("baseline_decision_hash", "baseline_request_hash")
    @classmethod
    def validate_hashes(cls, value: str) -> str:
        if not is_full_sha256(value):
            raise ValueError("M6_WHAT_IF_BASELINE_HASH_INVALID")
        return value

    @model_validator(mode="after")
    def bind_identifier_and_hash(self) -> WhatIfRequest:
        if self.execution_mode == "DRAFT_ONLY" and self.confirmed:
            raise ValueError("M6_WHAT_IF_DRAFT_CANNOT_BE_CONFIRMED")
        material = _request_material(self.model_dump(mode="json"))
        expected_hash = sha256_content_hash(material)
        expected_id = f"whatif-{expected_hash.removeprefix('sha256:')[:24]}"
        if self.content_hash and self.content_hash != expected_hash:
            raise ValueError("M6_WHAT_IF_CONTENT_HASH_MISMATCH")
        if self.what_if_id and self.what_if_id != expected_id:
            raise ValueError("M6_WHAT_IF_ID_CONTENT_MISMATCH")
        object.__setattr__(self, "content_hash", expected_hash)
        object.__setattr__(self, "what_if_id", expected_id)
        return self


class WhatIfDraft(StrictDecisionContract):
    status: Literal["DRAFT_READY", "NEEDS_CLARIFICATION", "NOT_SUPPORTED"]
    request: WhatIfRequest | None = None
    normalized_modifications: list[WhatIfModification] = Field(default_factory=list)
    ambiguities: list[str] = Field(default_factory=list)
    unsupported_fields: list[str] = Field(default_factory=list)
    confirmation_required: bool = True
    warnings: list[str] = Field(default_factory=list)


class MetricSnapshot(StrictDecisionContract):
    name: str
    value: float | str | bool | None
    unit: str | None = None
    grain: str
    semantics: str
    status: Literal["AVAILABLE", "UNAVAILABLE", "NOT_COMPARABLE"] = "AVAILABLE"
    evidence_refs: list[str] = Field(default_factory=list)


class MetricComparison(StrictDecisionContract):
    name: str
    baseline: MetricSnapshot
    hypothetical: MetricSnapshot
    delta: float | None = None
    status: Literal["COMPARABLE", "UNAVAILABLE", "NOT_COMPARABLE"]
    reason: str


class OrderDelta(StrictDecisionContract):
    decision_stage: Literal["first_stage", "scenario_recourse"]
    scenario_id: str | None = None
    offer_id: str
    supplier_id: str
    store_id: str
    ingredient_id: str
    unit: str
    order_date: date
    arrival_date: date
    baseline_quantity: float
    hypothetical_quantity: float
    quantity_delta: float


class DecisionComparison(StrictDecisionContract):
    baseline_decision_hash: str
    hypothetical_decision_hash: str
    decision_status_changed: bool
    baseline_decision_status: str
    hypothetical_decision_status: str
    strategy_changed: bool
    baseline_strategy: StrategyName | None = None
    hypothetical_strategy: StrategyName | None = None
    first_stage_order_deltas: list[OrderDelta]
    recourse_order_deltas: list[OrderDelta]
    metric_comparisons: list[MetricComparison]
    baseline_hard_violations: list[str]
    hypothetical_hard_violations: list[str]
    baseline_readiness: str
    hypothetical_readiness: str
    warnings: list[str] = Field(default_factory=list)


class ComparativeEvidenceItem(StrictDecisionContract):
    package_role: PackageRole
    package_hash: str
    evidence_id: str
    content_hash: str
    evidence_type: str
    semantics: str
    payload: dict[str, Any]
    text: str


class ComparativeEvidencePackage(StrictDecisionContract):
    items: list[ComparativeEvidenceItem]
    package_hash: str = ""

    @model_validator(mode="after")
    def validate_hash(self) -> ComparativeEvidencePackage:
        refs = [
            (item.package_role, item.package_hash, item.evidence_id, item.content_hash)
            for item in self.items
        ]
        if refs != sorted(refs):
            raise ValueError("M6_COMPARATIVE_EVIDENCE_ORDER_INVALID")
        expected = sha256_content_hash(refs)
        if self.package_hash and self.package_hash != expected:
            raise ValueError("M6_COMPARATIVE_PACKAGE_HASH_MISMATCH")
        object.__setattr__(self, "package_hash", expected)
        return self


class ComparativeGraphEdge(StrictDecisionContract):
    source: str
    target: str
    relation: Literal[
        "COUNTERFACTUAL_OF",
        "MODIFIED_BY",
        "REOPTIMIZED_AS",
        "VALIDATED_BY",
        "DIFFERS_FROM",
        "COMPARED_WITH",
        "REGRET_EVALUATED_AGAINST",
        "APPROVAL_BINDS",
        "SUPERSEDES",
        "INVALIDATED_BY",
    ]


class ComparativeDecisionGraph(StrictDecisionContract):
    nodes: list[str]
    edges: list[ComparativeGraphEdge]
    graph_hash: str = ""

    @model_validator(mode="after")
    def validate_graph(self) -> ComparativeDecisionGraph:
        if self.nodes != sorted(set(self.nodes)):
            raise ValueError("M6_COMPARATIVE_GRAPH_NODE_ORDER_INVALID")
        ordered = sorted(
            self.edges,
            key=lambda edge: (edge.source, edge.target, edge.relation),
        )
        if self.edges != ordered:
            raise ValueError("M6_COMPARATIVE_GRAPH_EDGE_ORDER_INVALID")
        expected = sha256_content_hash(
            {"nodes": self.nodes, "edges": [edge.model_dump() for edge in self.edges]}
        )
        if self.graph_hash and self.graph_hash != expected:
            raise ValueError("M6_COMPARATIVE_GRAPH_HASH_MISMATCH")
        object.__setattr__(self, "graph_hash", expected)
        return self


class ComparativeClaim(StrictDecisionContract):
    claim_type: str = Field(min_length=1)
    facts: dict[str, Any]
    citation_refs: list[str] = Field(min_length=1)
    uses_probability_language: bool = False
    causal: bool = False


class ComparativeAnswer(StrictDecisionContract):
    question: str = Field(min_length=1)
    status: Literal["GROUNDED", "PARTIAL", "INSUFFICIENT_EVIDENCE", "UNSUPPORTED_INTENT"]
    intent: str
    claims: list[ComparativeClaim] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    answer_text: str = Field(min_length=1)
    retrieved_refs: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class WhatIfAuditEvent(StrictDecisionContract):
    sequence: int = Field(ge=0)
    action: str
    reason_code: str
    input_hash: str
    output_hash: str | None = None
    status: str


class WhatIfDecisionPackage(StrictDecisionContract):
    what_if_id: str
    baseline_request_id: str
    baseline_decision_hash: str
    baseline_request_hash: str
    normalized_modifications: list[WhatIfModification]
    modified_request: OptimizationRequest
    modified_request_hash: str
    optimization_result: OptimizationResult
    hypothetical_decision: FinalDecisionPackage
    comparison: DecisionComparison
    evidence_package: ComparativeEvidencePackage
    decision_graph: ComparativeDecisionGraph
    warnings: list[str]
    limitations: list[str]
    authority_statuses: dict[str, str]
    audit_trace: list[WhatIfAuditEvent]
    provenance: dict[str, Any]
    package_hash: str = ""

    @model_validator(mode="after")
    def validate_authority_and_hash(self) -> WhatIfDecisionPackage:
        if self.optimization_result.request_id != self.modified_request.request_id:
            raise ValueError("M6_WHAT_IF_RESULT_REQUEST_ID_MISMATCH")
        if self.hypothetical_decision.request_id != self.modified_request.request_id:
            raise ValueError("M6_WHAT_IF_DECISION_REQUEST_ID_MISMATCH")
        if self.hypothetical_decision.recommended_strategy != (
            self.optimization_result.recommended_strategy
        ):
            raise ValueError("M6_WHAT_IF_M5_RECOMMENDATION_MISMATCH")
        if self.modified_request_hash != sha256_content_hash(self.modified_request):
            raise ValueError("M6_WHAT_IF_MODIFIED_REQUEST_HASH_MISMATCH")
        material = self.model_dump(mode="json", exclude={"package_hash"})
        expected = sha256_content_hash(material)
        if self.package_hash and self.package_hash != expected:
            raise ValueError("M6_WHAT_IF_PACKAGE_HASH_MISMATCH")
        object.__setattr__(self, "package_hash", expected)
        return self


class CounterfactualTarget(StrictDecisionContract):
    target_type: Literal[
        "STRATEGY",
        "DECISION_STATUS",
        "CRITIC_PASS",
        "IMMEDIATE_ORDER_THRESHOLD",
        "EXACT_METRIC_THRESHOLD",
    ]
    strategy: StrategyName | None = None
    decision_status: str | None = None
    critic_passed: bool | None = None
    offer_id: str | None = None
    metric_name: str | None = None
    operator: Literal["GT", "GE", "LT", "LE", "EQ"] = "EQ"
    threshold: float | None = None

    @model_validator(mode="after")
    def validate_target(self) -> CounterfactualTarget:
        required = {
            "STRATEGY": self.strategy,
            "DECISION_STATUS": self.decision_status,
            "CRITIC_PASS": self.critic_passed,
            "IMMEDIATE_ORDER_THRESHOLD": self.threshold,
            "EXACT_METRIC_THRESHOLD": self.threshold,
        }[self.target_type]
        if required is None:
            raise ValueError("M6_COUNTERFACTUAL_TARGET_VALUE_REQUIRED")
        if self.target_type == "IMMEDIATE_ORDER_THRESHOLD" and not self.offer_id:
            raise ValueError("M6_COUNTERFACTUAL_OFFER_REQUIRED")
        if self.target_type == "EXACT_METRIC_THRESHOLD" and not self.metric_name:
            raise ValueError("M6_COUNTERFACTUAL_METRIC_REQUIRED")
        return self


class CounterfactualSearchRequest(StrictDecisionContract):
    search_id: str = ""
    baseline_request_id: str
    baseline_decision_hash: str
    baseline_request_hash: str
    candidate_modifications: list[list[WhatIfModification]] = Field(min_length=1)
    target: CounterfactualTarget
    maximum_run_count: int = Field(gt=0)
    confirmed: bool = False
    idempotency_key: str = Field(min_length=1)
    actor: str = Field(min_length=1)

    @model_validator(mode="after")
    def bind_search_id(self) -> CounterfactualSearchRequest:
        material = self.model_dump(mode="json", exclude={"search_id"})
        expected = f"counterfactual-{sha256_content_hash(material)[7:31]}"
        if self.search_id and self.search_id != expected:
            raise ValueError("M6_COUNTERFACTUAL_ID_CONTENT_MISMATCH")
        object.__setattr__(self, "search_id", expected)
        return self


class CounterfactualRun(StrictDecisionContract):
    candidate_index: int
    modification_hash: str
    status: Literal["TARGET_MET", "TARGET_NOT_MET", "FAILED_VALIDATION", "NOT_RUN_BUDGET"]
    what_if_package_hash: str | None = None
    error_code: str | None = None


class CounterfactualSearchResult(StrictDecisionContract):
    search_id: str
    status: Literal[
        "BOUNDED_COUNTERFACTUAL_FOUND",
        "NOT_FOUND_IN_BOUNDED_SPACE",
        "RUN_BUDGET_EXHAUSTED",
        "FAILED_VALIDATION",
    ]
    target: CounterfactualTarget
    found_candidate_index: int | None = None
    runs: list[CounterfactualRun]
    candidate_count: int
    maximum_run_count: int
    bounded: Literal[True] = True
    global_minimality_claimed: Literal[False] = False
    limitations: list[str]
    result_hash: str = ""

    @model_validator(mode="after")
    def validate_hash(self) -> CounterfactualSearchResult:
        expected = sha256_content_hash(self.model_dump(mode="json", exclude={"result_hash"}))
        if self.result_hash and self.result_hash != expected:
            raise ValueError("M6_COUNTERFACTUAL_RESULT_HASH_MISMATCH")
        object.__setattr__(self, "result_hash", expected)
        return self
