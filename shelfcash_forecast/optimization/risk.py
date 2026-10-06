from __future__ import annotations

import numpy as np

from shelfcash_forecast.inventory.metrics import weighted_cvar


def calculate_discrete_cvar(
    costs: list[float],
    weights: list[float],
    *,
    alpha: float = 0.95,
) -> float:
    if not costs or len(costs) != len(weights):
        raise ValueError("Costs and weights must be non-empty and aligned.")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between zero and one.")
    weight_array = np.asarray(weights, dtype=float)
    if (weight_array < 0).any() or not np.isclose(weight_array.sum(), 1):
        raise ValueError("Non-negative weights must sum to one.")
    return weighted_cvar(np.asarray(costs, dtype=float), weight_array, alpha)
