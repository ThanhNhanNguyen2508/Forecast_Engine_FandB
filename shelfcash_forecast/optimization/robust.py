from __future__ import annotations

from typing import Protocol

from shelfcash_forecast.optimization.contracts import RobustOptimizationStatus


class AdvancedRobustOptimizer(Protocol):
    def availability(self) -> RobustOptimizationStatus: ...


def conformal_robust_status() -> RobustOptimizationStatus:
    """Report the truthful status of the not-yet-identified robust formulation."""

    return RobustOptimizationStatus(
        status="NOT_AVAILABLE",
        method="conformal_robust_procurement",
        missing_prerequisites=[
            "joint calibrated ingredient-demand uncertainty set",
            "documented uncertainty-set geometry and coverage guarantee",
            "robust counterpart for the procurement constraints",
        ],
        guarantee=None,
        details={
            "reason": (
                "Current marginal CQR intervals do not define a joint multivariate "
                "uncertainty set, so no robust guarantee is claimed."
            )
        },
    )
