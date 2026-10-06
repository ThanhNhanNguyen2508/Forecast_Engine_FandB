from __future__ import annotations

from collections.abc import Callable
from typing import Any

from shelfcash_forecast.decision_intelligence.agents.contracts import AgentMode
from shelfcash_forecast.decision_intelligence.agents.policies import tool_is_allowed


class AgentToolError(ValueError):
    pass


class DecisionToolRegistry:
    """In-memory allowlisted functions; no recursion or background execution."""

    def __init__(self) -> None:
        self._tools: dict[str, Callable[..., Any]] = {}

    def register(self, name: str, function: Callable[..., Any]) -> None:
        if name in self._tools:
            raise AgentToolError("M6_AGENT_DUPLICATE_TOOL")
        self._tools[name] = function

    def call(self, mode: AgentMode, name: str, **kwargs: Any) -> Any:
        if not tool_is_allowed(mode, name):
            raise AgentToolError(f"M6_AGENT_UNAUTHORIZED_TOOL:{mode}:{name}")
        if name not in self._tools:
            raise AgentToolError(f"M6_AGENT_TOOL_NOT_REGISTERED:{name}")
        return self._tools[name](**kwargs)


__all__ = ["AgentToolError", "DecisionToolRegistry"]
