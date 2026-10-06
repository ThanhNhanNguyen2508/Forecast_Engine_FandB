from __future__ import annotations

from typing import Protocol, runtime_checkable

from shelfcash_forecast.optimization.contracts import (
    CandidateEvaluation,
    OptimizationRequest,
    OptimizationResult,
    ProcurementPlan,
)
from shelfcash_forecast.optimization.optimizer import optimize_procurement
from shelfcash_forecast.optimization.resimulation import evaluate_candidate_plan
from shelfcash_forecast.optimization.strategies import default_strategy_profiles


@runtime_checkable
class ComputationGateway(Protocol):
    """The only M6 boundary authorized to invoke M4/M5 computation."""

    def optimize(self, request: OptimizationRequest) -> OptimizationResult: ...

    def evaluate_plan(
        self,
        plan: ProcurementPlan,
        request: OptimizationRequest,
    ) -> CandidateEvaluation: ...


class M5ComputationGateway:
    """Production gateway backed by the existing M5 and exact M4 path."""

    def optimize(self, request: OptimizationRequest) -> OptimizationResult:
        return optimize_procurement(request)

    def evaluate_plan(
        self,
        plan: ProcurementPlan,
        request: OptimizationRequest,
    ) -> CandidateEvaluation:
        profiles = {profile.name: profile for profile in default_strategy_profiles()}
        profiles.update({profile.name: profile for profile in request.strategy_profiles})
        return evaluate_candidate_plan(plan, request, profiles[plan.strategy])


__all__ = ["ComputationGateway", "M5ComputationGateway"]
