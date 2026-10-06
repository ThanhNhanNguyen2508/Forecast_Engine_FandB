# Nó biến optimization từ:

# “lập plan hôm nay cho cả 7 ngày”

# thành:

# Ngày 1:
# forecast
# → optimize
# → chỉ execute order của ngày 1

# Ngày 2:
# quan sát lại inventory thật
# → forecast mới
# → optimize lại
# → chỉ execute ngày 2

# Ngày 3:
# repeat

# Tức:

# Observe
# → Forecast
# → Optimize
# → Execute first action
# → Observe again

# Nó là receding/rolling horizon.
from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import date

from shelfcash_forecast.optimization.contracts import (
    OptimizationRequest,
    OptimizationResult,
    RollingHorizonResult,
    RollingHorizonStep,
)
from shelfcash_forecast.optimization.optimizer import optimize_procurement


class RollingHorizonController:
    """Pure MPC orchestration; state observation/reforecasting stay caller-owned."""

    def run(
        self,
        decision_dates: Sequence[date],
        request_factory: Callable[[date, OptimizationResult | None], OptimizationRequest],
        *,
        optimizer: Callable[[OptimizationRequest], OptimizationResult] = optimize_procurement,
    ) -> RollingHorizonResult:
        steps: list[RollingHorizonStep] = []
        previous: OptimizationResult | None = None
        for decision_date in decision_dates:
            request = request_factory(decision_date, previous)
            if request.decision_date != decision_date:
                raise ValueError("request_factory returned the wrong decision_date.")
            result = optimizer(request)
            evaluation = (
                result.evaluations.get(result.recommended_strategy)
                if result.recommended_strategy is not None
                else None
            )
            executed = (
                [
                    order
                    for order in evaluation.plan.orders
                    if order.order_date == decision_date
                ]
                if evaluation is not None
                else []
            )
            steps.append(
                RollingHorizonStep(
                    decision_date=decision_date,
                    optimization_result=result,
                    executed_orders=executed,
                )
            )
            previous = result
        return RollingHorizonResult(
            steps=steps,
            provenance={
                "controller": "receding_horizon_v1",
                "execution_rule": "first_actionable_period_only",
            },
        )
