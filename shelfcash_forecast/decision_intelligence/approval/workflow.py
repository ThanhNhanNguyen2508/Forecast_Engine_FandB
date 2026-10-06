from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Protocol

from shelfcash_forecast.decision_intelligence.approval.contracts import (
    ApprovalCase,
    ApprovalEvent,
    ApprovalOrderLine,
    ApprovalPolicy,
    ApprovalState,
)
from shelfcash_forecast.decision_intelligence.contracts import FinalDecisionPackage
from shelfcash_forecast.decision_intelligence.integrity import sha256_content_hash
from shelfcash_forecast.decision_intelligence.what_if.comparison import decision_snapshot_hash
from shelfcash_forecast.optimization.contracts import OptimizationRequest


class Clock(Protocol):
    def now(self) -> datetime: ...


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)


class ApprovalError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


_TRANSITIONS: dict[ApprovalState, set[ApprovalState]] = {
    "DRAFT": {"PENDING_APPROVAL", "CANCELLED", "EXPIRED", "STALE"},
    "PENDING_APPROVAL": {"APPROVED", "REJECTED", "CANCELLED", "EXPIRED", "STALE"},
    "APPROVED": {"STALE"},
    "REJECTED": set(),
    "CANCELLED": set(),
    "EXPIRED": set(),
    "STALE": set(),
}


def _event(
    *,
    sequence: int,
    previous_event_hash: str | None,
    actor: str,
    role: str,
    from_state: ApprovalState | None,
    to_state: ApprovalState,
    reason: str,
    timestamp: datetime,
    payload_hash: str,
    idempotency_key: str,
) -> ApprovalEvent:
    material = {
        "sequence": sequence,
        "previous_event_hash": previous_event_hash,
        "actor": actor,
        "role": role,
        "from_state": from_state,
        "to_state": to_state,
        "reason": reason,
        "timestamp": timestamp.isoformat().replace("+00:00", "Z"),
        "payload_hash": payload_hash,
        "idempotency_key": idempotency_key,
    }
    event_hash = sha256_content_hash(material)
    return ApprovalEvent(
        **material,
        event_id=f"approval-event-{event_hash[7:31]}",
        event_hash=event_hash,
    )


def _bound_orders(
    decision: FinalDecisionPackage,
    request: OptimizationRequest,
) -> list[ApprovalOrderLine]:
    offers = {offer.offer_id: offer for offer in request.supplier_offers}
    rows: list[ApprovalOrderLine] = []
    for order in decision.immediate_orders:
        offer = offers.get(order.offer_id)
        if offer is None or (
            offer.supplier_id,
            offer.store_id,
            offer.ingredient_id,
            offer.unit,
        ) != (order.supplier_id, order.store_id, order.ingredient_id, order.unit):
            raise ApprovalError("M6_APPROVAL_ORDER_OFFER_BINDING_FAILED", order.offer_id)
        if not math.isclose(
            order.order_quantity,
            order.pack_count * offer.pack_size,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ) or not math.isclose(
            order.purchase_cost,
            order.order_quantity * offer.unit_price,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ApprovalError("M6_APPROVAL_ORDER_PRICE_OR_PACK_BINDING_FAILED", order.offer_id)
        rows.append(
            ApprovalOrderLine(
                offer_id=order.offer_id,
                supplier_id=order.supplier_id,
                store_id=order.store_id,
                ingredient_id=order.ingredient_id,
                unit=order.unit,
                order_date=order.order_date.isoformat(),
                arrival_date=order.arrival_date.isoformat(),
                pack_count=order.pack_count,
                pack_size=offer.pack_size,
                order_quantity=order.order_quantity,
                unit_price=offer.unit_price,
                purchase_cost=order.purchase_cost,
                delivery_cost=order.delivery_cost,
            )
        )
    rows.sort(
        key=lambda row: (
            row.offer_id,
            row.store_id,
            row.ingredient_id,
            row.order_date,
        )
    )
    return rows


def create_approval_case(
    decision: FinalDecisionPackage,
    optimization_request: OptimizationRequest,
    *,
    policy: ApprovalPolicy,
    requester_actor: str,
    requester_role: str,
    idempotency_key: str,
    package_created_at: datetime,
    expires_at: datetime,
) -> ApprovalCase:
    if decision.request_id != optimization_request.request_id:
        raise ApprovalError("M6_APPROVAL_FOREIGN_REQUEST", "request ID mismatch")
    if decision.recommended_strategy is None:
        raise ApprovalError("M6_APPROVAL_NO_VALID_PLAN", "no recommendation to approve")
    if requester_role not in policy.submitter_roles:
        raise ApprovalError("M6_APPROVAL_UNAUTHORIZED_ROLE", requester_role)
    orders = _bound_orders(decision, optimization_request)
    orders_hash = sha256_content_hash(orders)
    decision_hash = decision_snapshot_hash(decision)
    case_material = {
        "decision_package_hash": decision_hash,
        "request_id": decision.request_id,
        "recommended_strategy": decision.recommended_strategy,
        "orders_hash": orders_hash,
        "package_created_at": package_created_at,
        "expires_at": expires_at,
        "policy": policy,
        "requester_actor": requester_actor,
        "requester_role": requester_role,
        "idempotency_key": idempotency_key,
    }
    case_id = f"approval-{sha256_content_hash(case_material)[7:31]}"
    first_event = _event(
        sequence=0,
        previous_event_hash=None,
        actor=requester_actor,
        role=requester_role,
        from_state=None,
        to_state="DRAFT",
        reason="Approval case created and bound to exact decision orders.",
        timestamp=package_created_at,
        payload_hash=orders_hash,
        idempotency_key=idempotency_key,
    )
    return ApprovalCase(
        case_id=case_id,
        state="DRAFT",
        decision_package_hash=decision_hash,
        request_id=decision.request_id,
        recommended_strategy=decision.recommended_strategy,
        orders=orders,
        orders_hash=orders_hash,
        package_created_at=package_created_at,
        expires_at=expires_at,
        policy=policy,
        requester_actor=requester_actor,
        requester_role=requester_role,
        idempotency_key=idempotency_key,
        events=[first_event],
        limitations=[
            "ACTOR_IDENTITY_IS_CALLER_ATTESTED_AND_NOT_EXTERNALLY_AUTHENTICATED",
            "APPROVED records a human decision only; it never executes supplier procurement.",
        ],
    )


def _role_allowed(case: ApprovalCase, target: ApprovalState, role: str) -> bool:
    if target == "PENDING_APPROVAL":
        return role in case.policy.submitter_roles
    if target in {"APPROVED", "REJECTED"}:
        return role in case.policy.approver_roles
    if target == "CANCELLED":
        return role in case.policy.cancellation_roles
    return role == "SYSTEM"


def transition_approval_case(
    case: ApprovalCase,
    target_state: ApprovalState,
    *,
    actor: str,
    role: str,
    reason: str,
    idempotency_key: str,
    current_decision_hash: str,
    clock: Clock | None = None,
) -> ApprovalCase:
    now = (clock or SystemClock()).now()
    duplicate = next(
        (event for event in case.events if event.idempotency_key == idempotency_key),
        None,
    )
    if duplicate is not None:
        if duplicate.to_state == target_state:
            return case
        raise ApprovalError("M6_APPROVAL_IDEMPOTENCY_CONFLICT", idempotency_key)
    effective_target = target_state
    if current_decision_hash != case.decision_package_hash:
        effective_target = "STALE"
        actor, role, reason = "SYSTEM", "SYSTEM", "Decision package hash changed."
    elif now >= case.expires_at and target_state not in {"EXPIRED", "STALE"}:
        effective_target = "EXPIRED"
        actor, role, reason = "SYSTEM", "SYSTEM", "Approval window expired."
    if effective_target not in _TRANSITIONS[case.state]:
        raise ApprovalError(
            "M6_APPROVAL_INVALID_TRANSITION",
            f"{case.state} -> {effective_target}",
        )
    if not _role_allowed(case, effective_target, role):
        raise ApprovalError("M6_APPROVAL_UNAUTHORIZED_ROLE", role)
    if (
        effective_target == "APPROVED"
        and actor == case.requester_actor
        and not case.policy.allow_self_approval
    ):
        raise ApprovalError("M6_APPROVAL_SELF_APPROVAL_FORBIDDEN", actor)
    event = _event(
        sequence=len(case.events),
        previous_event_hash=case.events[-1].event_hash,
        actor=actor,
        role=role,
        from_state=case.state,
        to_state=effective_target,
        reason=reason,
        timestamp=now,
        payload_hash=case.orders_hash,
        idempotency_key=idempotency_key,
    )
    values = case.model_dump(mode="python")
    values.update(
        {
            "state": effective_target,
            "events": [*case.events, event],
            "case_hash": "",
        }
    )
    return ApprovalCase.model_validate(values)


def inspect_approval_case(
    case: ApprovalCase,
    decision: FinalDecisionPackage,
    *,
    clock: Clock | None = None,
) -> ApprovalCase:
    now = (clock or SystemClock()).now()
    current_hash = decision_snapshot_hash(decision)
    if current_hash != case.decision_package_hash and "STALE" in _TRANSITIONS[case.state]:
        return transition_approval_case(
            case,
            "STALE",
            actor="SYSTEM",
            role="SYSTEM",
            reason="Decision package changed.",
            idempotency_key=f"system-stale-{current_hash}",
            current_decision_hash=current_hash,
            clock=clock,
        )
    if now >= case.expires_at and "EXPIRED" in _TRANSITIONS[case.state]:
        return transition_approval_case(
            case,
            "EXPIRED",
            actor="SYSTEM",
            role="SYSTEM",
            reason="Approval window expired.",
            idempotency_key=f"system-expired-{case.expires_at.isoformat()}",
            current_decision_hash=case.decision_package_hash,
            clock=clock,
        )
    return case


__all__ = [
    "ApprovalError",
    "Clock",
    "SystemClock",
    "create_approval_case",
    "inspect_approval_case",
    "transition_approval_case",
]
