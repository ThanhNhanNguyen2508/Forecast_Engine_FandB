from __future__ import annotations

from typing import Protocol

from pydantic import Field

from shelfcash_forecast.decision_intelligence.agents.contracts import AgentIntent
from shelfcash_forecast.decision_intelligence.contracts import (
    FinalDecisionPackage,
    StrictDecisionContract,
)
from shelfcash_forecast.decision_intelligence.what_if.contracts import WhatIfModification


class UntrustedAgentProposal(StrictDecisionContract):
    intent: AgentIntent
    candidate_entities: dict[str, str] = Field(default_factory=dict)
    proposed_modifications: list[WhatIfModification] = Field(default_factory=list)
    proposed_strategy: str | None = None
    proposed_order_quantities: dict[str, float] = Field(default_factory=dict)
    proposed_probability: float | None = Field(default=None, ge=0, le=1)
    proposed_regret: float | None = Field(default=None, ge=0)
    proposed_citations: list[str] = Field(default_factory=list)


class OptionalLLMAdapter(Protocol):
    def propose(self, question: str) -> UntrustedAgentProposal: ...


def validate_llm_proposal(
    proposal: UntrustedAgentProposal,
    decision: FinalDecisionPackage,
) -> UntrustedAgentProposal:
    """Treat every LLM numeric/authority field as untrusted and fail closed."""

    if proposal.proposed_strategy is not None and (
        proposal.proposed_strategy != decision.recommended_strategy
    ):
        raise ValueError("M6_LLM_FORGED_STRATEGY")
    orders = {order.offer_id: order.order_quantity for order in decision.immediate_orders}
    if any(
        orders.get(offer_id) != quantity
        for offer_id, quantity in proposal.proposed_order_quantities.items()
    ):
        raise ValueError("M6_LLM_FORGED_ORDER")
    if proposal.proposed_probability is not None:
        raise ValueError("M6_LLM_PROBABILITY_NOT_TRUSTED")
    if proposal.proposed_regret is not None:
        raise ValueError("M6_LLM_REGRET_NOT_TRUSTED")
    known = {item.evidence_id for item in decision.evidence_package.items}
    if set(proposal.proposed_citations) - known:
        raise ValueError("M6_LLM_UNKNOWN_CITATION")
    return proposal


__all__ = ["OptionalLLMAdapter", "UntrustedAgentProposal", "validate_llm_proposal"]
