from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from shelfcash_forecast.decision_intelligence.approval.contracts import (
    ApprovalCase,
    ApprovalState,
)
from shelfcash_forecast.decision_intelligence.contracts import (
    DecisionAnswer,
    FinalDecisionPackage,
    StrictDecisionContract,
)
from shelfcash_forecast.decision_intelligence.integrity import sha256_content_hash
from shelfcash_forecast.decision_intelligence.regret.contracts import DecisionRegretRequest
from shelfcash_forecast.decision_intelligence.what_if.contracts import (
    ComparativeAnswer,
    CounterfactualSearchRequest,
    WhatIfDecisionPackage,
    WhatIfDraft,
    WhatIfModification,
    WhatIfRequest,
)
from shelfcash_forecast.optimization.contracts import OptimizationRequest

AgentMode = Literal[
    "READ_ONLY",
    "WHAT_IF_DRAFT",
    "WHAT_IF_EXECUTE",
    "COMPARISON",
    "COUNTERFACTUAL",
    "REGRET",
    "APPROVAL",
]
AgentIntent = Literal[
    "READ_ONLY_EXPLANATION",
    "WHAT_IF_DRAFT",
    "WHAT_IF_EXECUTION",
    "COMPARISON",
    "COUNTERFACTUAL",
    "REGRET",
    "APPROVAL",
    "UNKNOWN",
]


class AgentRunRequest(StrictDecisionContract):
    run_id: str = ""
    mode: AgentMode
    question: str = Field(min_length=1)
    language: Literal["en", "vi"]
    baseline_decision: FinalDecisionPackage
    baseline_request: OptimizationRequest | None = None
    typed_modifications: list[WhatIfModification] = Field(default_factory=list)
    what_if_request: WhatIfRequest | None = None
    what_if_package: WhatIfDecisionPackage | None = None
    counterfactual_request: CounterfactualSearchRequest | None = None
    regret_request: DecisionRegretRequest | None = None
    approval_case: ApprovalCase | None = None
    approval_target_state: ApprovalState | None = None
    approval_role: str | None = None
    actor: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1)
    maximum_tool_calls: int = Field(default=4, ge=1, le=20)

    @model_validator(mode="after")
    def bind_run_id(self) -> AgentRunRequest:
        material = self.model_dump(mode="json", exclude={"run_id"})
        expected = f"agent-{sha256_content_hash(material)[7:31]}"
        if self.run_id and self.run_id != expected:
            raise ValueError("M6_AGENT_RUN_ID_CONTENT_MISMATCH")
        object.__setattr__(self, "run_id", expected)
        return self


class AgentTraceEvent(StrictDecisionContract):
    sequence: int = Field(ge=0)
    component: str
    action: str
    reason_code: str
    input_hash: str
    output_hash: str | None = None
    status: str


class AgentRunResult(StrictDecisionContract):
    run_id: str
    status: Literal[
        "COMPLETED",
        "NEEDS_CLARIFICATION",
        "INSUFFICIENT_EVIDENCE",
        "NOT_SUPPORTED_AT_CURRENT_AUTHORITY_BOUNDARY",
        "UNAUTHORIZED_TOOL_CALL",
        "TOOL_BUDGET_EXHAUSTED",
        "FAILED_VALIDATION",
    ]
    intent: AgentIntent
    answer: DecisionAnswer | ComparativeAnswer | None = None
    what_if_draft: WhatIfDraft | None = None
    what_if_package: WhatIfDecisionPackage | None = None
    result_payload: Any = None
    trace: list[AgentTraceEvent]
    tool_calls: list[str]
    error_codes: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    result_hash: str = ""

    @model_validator(mode="after")
    def validate_result(self) -> AgentRunResult:
        if len(self.tool_calls) != len(set(self.tool_calls)):
            raise ValueError("M6_AGENT_RECURSIVE_OR_DUPLICATE_TOOL_CALL")
        if [event.sequence for event in self.trace] != list(range(len(self.trace))):
            raise ValueError("M6_AGENT_TRACE_SEQUENCE_INVALID")
        expected = sha256_content_hash(self.model_dump(mode="json", exclude={"result_hash"}))
        if self.result_hash and self.result_hash != expected:
            raise ValueError("M6_AGENT_RESULT_HASH_MISMATCH")
        object.__setattr__(self, "result_hash", expected)
        return self


class AuthorityAssessment(StrictDecisionContract):
    authority_layer: Literal["M1_M2", "M3", "M4", "M5", "M6"]
    semantics: Literal[
        "deterministic",
        "probabilistic",
        "quantile",
        "stress",
        "hypothetical",
        "observed",
    ]
    may_compute: bool
    reason_code: str


class ToolCallRecord(StrictDecisionContract):
    tool_name: str
    mode: AgentMode
    input_hash: str
    output_hash: str | None = None
    status: str
    result: Any = None
