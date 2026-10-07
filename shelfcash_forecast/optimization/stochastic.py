"""Stochastic objective on shared lot/FEFO procurement physics."""
from shelfcash_forecast.optimization.contracts import OptimizationRequest, StrategyProfile
from shelfcash_forecast.optimization.lot_milp import solve_lot_procurement


def solve_stochastic_procurement(request: OptimizationRequest, profile: StrategyProfile):
    return solve_lot_procurement(request, profile, stochastic=True)
