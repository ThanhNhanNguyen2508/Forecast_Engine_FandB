from __future__ import annotations

from shelfcash_forecast.decision_intelligence.approval.contracts import ApprovalCase
from shelfcash_forecast.decision_intelligence.contracts import FinalDecisionPackage
from shelfcash_forecast.decision_intelligence.what_if.comparison import decision_snapshot_hash


def approval_binds_current_decision(
    case: ApprovalCase,
    decision: FinalDecisionPackage,
) -> bool:
    return (
        case.request_id == decision.request_id
        and case.recommended_strategy == decision.recommended_strategy
        and case.decision_package_hash == decision_snapshot_hash(decision)
    )


__all__ = ["approval_binds_current_decision"]
