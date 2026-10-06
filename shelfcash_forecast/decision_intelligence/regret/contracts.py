from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from shelfcash_forecast.decision_intelligence.contracts import StrictDecisionContract
from shelfcash_forecast.decision_intelligence.integrity import sha256_content_hash
from shelfcash_forecast.optimization.contracts import (
    OptimizationRequest,
    ProcurementPlan,
    StrategyName,
)


class DecisionRegretRequest(StrictDecisionContract):
    regret_request_id: str = ""
    baseline_request_id: str = Field(min_length=1)
    baseline_decision_hash: str
    selected_strategy: StrategyName
    selected_plan_id: str = Field(min_length=1)
    evaluation_kind: Literal["REALIZED", "HYPOTHETICAL"]
    evaluation_request: OptimizationRequest
    comparator_plans: list[ProcurementPlan] = Field(min_length=1)
    monetary_unit: str | None = None
    cost_definition: Literal["FIRST_STAGE_PLUS_APPLICABLE_RECOURSE_PLUS_EXACT_M4_CONSEQUENCE"] = (
        "FIRST_STAGE_PLUS_APPLICABLE_RECOURSE_PLUS_EXACT_M4_CONSEQUENCE"
    )
    confirmed: bool = False
    idempotency_key: str = Field(min_length=1)

    @model_validator(mode="after")
    def bind_identifier(self) -> DecisionRegretRequest:
        material = self.model_dump(mode="json", exclude={"regret_request_id"})
        expected = f"regret-{sha256_content_hash(material)[7:31]}"
        if self.regret_request_id and self.regret_request_id != expected:
            raise ValueError("M6_REGRET_ID_CONTENT_MISMATCH")
        object.__setattr__(self, "regret_request_id", expected)
        return self


class CandidateLoss(StrictDecisionContract):
    plan_id: str
    strategy: StrategyName
    first_stage_purchase_and_delivery_cost: float = Field(ge=0)
    applicable_recourse_cost: float = Field(ge=0)
    exact_m4_consequence_cost: float = Field(ge=0)
    total_exact_loss: float = Field(ge=0)
    monetary_unit: str
    exact_simulation_hash: str
    critic_passed: bool
    hard_violations: list[str]

    @model_validator(mode="after")
    def validate_total(self) -> CandidateLoss:
        expected = (
            self.first_stage_purchase_and_delivery_cost
            + self.applicable_recourse_cost
            + self.exact_m4_consequence_cost
        )
        if abs(self.total_exact_loss - expected) > 1e-8:
            raise ValueError("M6_REGRET_LOSS_COMPONENT_MISMATCH")
        return self


class DecisionRegretResult(StrictDecisionContract):
    regret_request_id: str
    status: Literal[
        "COMPUTED_REALIZED",
        "COMPUTED_HYPOTHETICAL",
        "UNAVAILABLE",
        "INCOMPARABLE",
        "FAILED_VALIDATION",
    ]
    selected_plan_id: str
    selected_exact_loss: float | None = None
    best_comparator_plan_id: str | None = None
    minimum_comparator_set_loss: float | None = None
    candidate_set_regret: float | None = None
    monetary_unit: str | None = None
    candidate_losses: list[CandidateLoss] = Field(default_factory=list)
    reason_code: str
    comparator_set_complete: bool
    global_oracle_claimed: Literal[False] = False
    limitations: list[str]
    result_hash: str = ""

    @model_validator(mode="after")
    def validate_result(self) -> DecisionRegretResult:
        computed = self.status in {"COMPUTED_REALIZED", "COMPUTED_HYPOTHETICAL"}
        values = (
            self.selected_exact_loss,
            self.minimum_comparator_set_loss,
            self.candidate_set_regret,
        )
        if computed and any(value is None for value in values):
            raise ValueError("M6_REGRET_COMPUTED_VALUES_REQUIRED")
        if not computed and any(value is not None for value in values):
            raise ValueError("M6_REGRET_UNAVAILABLE_CANNOT_HAVE_NUMERIC_RESULT")
        if self.candidate_set_regret is not None and self.candidate_set_regret < -1e-8:
            raise ValueError("M6_REGRET_NEGATIVE")
        expected = sha256_content_hash(self.model_dump(mode="json", exclude={"result_hash"}))
        if self.result_hash and self.result_hash != expected:
            raise ValueError("M6_REGRET_RESULT_HASH_MISMATCH")
        object.__setattr__(self, "result_hash", expected)
        return self
