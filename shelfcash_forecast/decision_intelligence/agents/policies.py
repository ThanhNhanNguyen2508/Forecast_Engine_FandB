from __future__ import annotations

from shelfcash_forecast.decision_intelligence.agents.contracts import AgentMode

TOOL_ALLOWLISTS: dict[AgentMode, frozenset[str]] = {
    "READ_ONLY": frozenset({"retrieve_current_decision", "explain_current_decision"}),
    "WHAT_IF_DRAFT": frozenset({"draft_what_if"}),
    "WHAT_IF_EXECUTE": frozenset({"execute_confirmed_what_if"}),
    "COMPARISON": frozenset({"compare_decisions"}),
    "COUNTERFACTUAL": frozenset({"search_bounded_counterfactual"}),
    "REGRET": frozenset({"evaluate_candidate_set_regret"}),
    "APPROVAL": frozenset(
        {"create_approval_case", "inspect_approval_case", "transition_approval_case"}
    ),
}


def tool_is_allowed(mode: AgentMode, tool_name: str) -> bool:
    return tool_name in TOOL_ALLOWLISTS[mode]


__all__ = ["TOOL_ALLOWLISTS", "tool_is_allowed"]
