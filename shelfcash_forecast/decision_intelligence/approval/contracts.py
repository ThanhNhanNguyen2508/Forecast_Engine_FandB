from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from shelfcash_forecast.decision_intelligence.contracts import StrictDecisionContract
from shelfcash_forecast.decision_intelligence.integrity import (
    is_full_sha256,
    sha256_content_hash,
)
from shelfcash_forecast.optimization.contracts import StrategyName

ApprovalState = Literal[
    "DRAFT",
    "PENDING_APPROVAL",
    "APPROVED",
    "REJECTED",
    "CANCELLED",
    "EXPIRED",
    "STALE",
]


class ApprovalPolicy(StrictDecisionContract):
    policy_id: str = Field(min_length=1)
    submitter_roles: set[str] = Field(min_length=1)
    approver_roles: set[str] = Field(min_length=1)
    cancellation_roles: set[str] = Field(min_length=1)
    allow_self_approval: bool = False


class ApprovalOrderLine(StrictDecisionContract):
    offer_id: str
    supplier_id: str
    store_id: str
    ingredient_id: str
    unit: str
    order_date: str
    arrival_date: str
    pack_count: int = Field(ge=0)
    pack_size: float = Field(gt=0)
    order_quantity: float = Field(ge=0)
    unit_price: float = Field(ge=0)
    purchase_cost: float = Field(ge=0)
    delivery_cost: float = Field(ge=0)


class ApprovalEvent(StrictDecisionContract):
    sequence: int = Field(ge=0)
    event_id: str
    previous_event_hash: str | None = None
    event_hash: str
    actor: str
    role: str
    from_state: ApprovalState | None = None
    to_state: ApprovalState
    reason: str
    timestamp: datetime
    payload_hash: str
    idempotency_key: str


class ApprovalCase(StrictDecisionContract):
    case_id: str
    state: ApprovalState
    decision_package_hash: str
    request_id: str
    recommended_strategy: StrategyName
    orders: list[ApprovalOrderLine]
    orders_hash: str
    package_created_at: datetime
    expires_at: datetime
    policy: ApprovalPolicy
    requester_actor: str
    requester_role: str
    idempotency_key: str
    events: list[ApprovalEvent]
    limitations: list[str]
    case_hash: str = ""

    @model_validator(mode="after")
    def validate_integrity(self) -> ApprovalCase:
        if self.expires_at <= self.package_created_at:
            raise ValueError("M6_APPROVAL_EXPIRY_INVALID")
        if self.orders_hash != sha256_content_hash(self.orders):
            raise ValueError("M6_APPROVAL_ORDER_BINDING_MISMATCH")
        if not is_full_sha256(self.decision_package_hash):
            raise ValueError("M6_APPROVAL_DECISION_HASH_INVALID")
        previous = None
        for sequence, event in enumerate(self.events):
            if event.sequence != sequence or event.previous_event_hash != previous:
                raise ValueError("M6_APPROVAL_EVENT_CHAIN_BROKEN")
            material = event.model_dump(
                mode="json",
                exclude={"event_hash", "event_id"},
            )
            expected_hash = sha256_content_hash(material)
            expected_id = f"approval-event-{expected_hash[7:31]}"
            if event.event_hash != expected_hash or event.event_id != expected_id:
                raise ValueError("M6_APPROVAL_EVENT_HASH_MISMATCH")
            previous = event.event_hash
        if not self.events or self.events[-1].to_state != self.state:
            raise ValueError("M6_APPROVAL_STATE_EVENT_MISMATCH")
        material = self.model_dump(mode="json", exclude={"case_hash"})
        expected = sha256_content_hash(material)
        if self.case_hash and self.case_hash != expected:
            raise ValueError("M6_APPROVAL_CASE_HASH_MISMATCH")
        object.__setattr__(self, "case_hash", expected)
        return self
