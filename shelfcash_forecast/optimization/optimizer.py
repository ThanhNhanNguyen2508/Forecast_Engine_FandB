#                    optimizer.py
#                         │
#        ┌────────────────┼─────────────────┐
#        ▼                ▼                 ▼
#      LEAN            BALANCED         PROTECTED
#        │                │                 │
#        ▼                ▼                 ▼
# stochastic/det    stochastic/det    stochastic/det
#        │                │                 │
#        ▼                ▼                 ▼
# candidate plan     candidate plan     candidate plan
#        │                │                 │
#        ▼                ▼                 ▼
# resimulation       resimulation       resimulation
#        │                │                 │
#        ▼                ▼                 ▼
# critic             critic             critic
# PASS/FAIL          PASS/FAIL          PASS/FAIL
#        └────────────────┼─────────────────┘
#                         ▼
#                  recommendation
from __future__ import annotations

from shelfcash_forecast.exceptions import OptimizationNotAvailableError
from shelfcash_forecast.optimization.contracts import (
    OptimizationRequest,
    OptimizationResult,
    StrategyProfile,
)
from shelfcash_forecast.optimization.deterministic import (
    solve_deterministic_procurement,
)
from shelfcash_forecast.optimization.resimulation import evaluate_candidate_plan
from shelfcash_forecast.optimization.stochastic import solve_stochastic_procurement
from shelfcash_forecast.optimization.strategies import default_strategy_profiles
from shelfcash_forecast.scenario.lead_time import LeadTimeModel
from shelfcash_forecast.scenario.shelf_life import ShelfLifeModel


def _profiles(request: OptimizationRequest) -> list[StrategyProfile]:
    defaults = {profile.name: profile for profile in default_strategy_profiles()}
    defaults.update({profile.name: profile for profile in request.strategy_profiles})
    return [defaults[name] for name in ("LEAN", "BALANCED", "PROTECTED")]


def optimize_procurement(
    request: OptimizationRequest,
    *,
    lead_time_model: LeadTimeModel | None = None,
    shelf_life_model: ShelfLifeModel | None = None,
) -> OptimizationResult:
    """Generate candidates with OR and accept them only after exact M4 evaluation."""

    evaluations = {}
    warnings: set[str] = set()
    has_probabilities = bool(request.demand_scenarios) and all(
        scenario.probability_weight is not None
        for scenario in request.demand_scenarios
    )
    use_stochastic = (
        request.stochastic and has_probabilities and len(request.demand_scenarios) > 1
    )
    requested_mode = "stochastic" if request.stochastic else "deterministic"
    initial_fallback_reason = None
    if request.stochastic and not use_stochastic:
        initial_fallback_reason = "STOCHASTIC_NOT_AVAILABLE_WITH_UNWEIGHTED_OR_SINGLE_SCENARIO"
        if not request.allow_mode_fallback:
            raise OptimizationNotAvailableError(
                "Stochastic mode requires at least two fully weighted scenarios.",
                code=initial_fallback_reason,
            )
        warnings.add(initial_fallback_reason)
    candidate_modes = {}
    for profile in _profiles(request):
        actual_mode = "stochastic" if use_stochastic else "deterministic"
        fallback_reason = initial_fallback_reason
        try:
            plan = (
                solve_stochastic_procurement(request, profile)
                if use_stochastic
                else solve_deterministic_procurement(request, profile)
            )
        except OptimizationNotAvailableError as exc:
            if not request.allow_mode_fallback:
                raise
            warnings.add(f"{exc.code}: {exc}")
            plan = solve_deterministic_procurement(request, profile)
            actual_mode = "deterministic"
            fallback_reason = exc.code
        evaluation = evaluate_candidate_plan(
            plan,
            request,
            profile,
            lead_time_model=lead_time_model,
            shelf_life_model=shelf_life_model,
        )
        evaluations[profile.name] = evaluation.model_copy(update={
            "requested_mode": requested_mode,
            "actual_mode": actual_mode,
            "fallback_reason": fallback_reason,
        })
        candidate_modes[profile.name] = {
            "requested_mode": requested_mode,
            "actual_mode": actual_mode,
            "fallback_reason": fallback_reason,
            "solver_status": plan.solver_status,
        }

    recommended = next(
        (
            name
            for name in ("BALANCED", "PROTECTED", "LEAN")
            if evaluations[name].critic.passed
        ),
        None,
    )
    return OptimizationResult(
        request_id=request.request_id,
        evaluations=evaluations,
        recommended_strategy=recommended,
        status=(
            "COMPLETED" if recommended is not None else "NO_VALID_PROCUREMENT_PLAN"
        ),
        provenance={
            "candidate_engine": "stochastic_saa" if use_stochastic else "deterministic_mip",
            "requested_mode": requested_mode,
            "actual_mode": (
                next(iter({item["actual_mode"] for item in candidate_modes.values()}))
                if len({item["actual_mode"] for item in candidate_modes.values()}) == 1
                else "mixed"
            ),
            "candidate_modes": candidate_modes,
            "validation_engine": "m4_lot_level_fefo_v1",
            "recommendation_rule": "BALANCED_then_PROTECTED_then_LEAN_if_valid",
            "no_valid_plan_reason": (
                None if recommended is not None else "NO_VALID_PROCUREMENT_PLAN"
            ),
            "supply_uncertainty": (
                "externally_realized"
                if lead_time_model is not None
                else "supplier_offer_fixed_lead_time"
            ),
        },
        warnings=sorted(warnings),
    )
