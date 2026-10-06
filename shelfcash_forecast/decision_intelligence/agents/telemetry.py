from __future__ import annotations

from shelfcash_forecast.decision_intelligence.agents.contracts import AgentTraceEvent
from shelfcash_forecast.decision_intelligence.integrity import sha256_content_hash


def trace_event(
    sequence: int,
    component: str,
    action: str,
    reason_code: str,
    input_value,
    output_value,
    status: str,
) -> AgentTraceEvent:
    return AgentTraceEvent(
        sequence=sequence,
        component=component,
        action=action,
        reason_code=reason_code,
        input_hash=sha256_content_hash(input_value),
        output_hash=(sha256_content_hash(output_value) if output_value is not None else None),
        status=status,
    )


__all__ = ["trace_event"]
