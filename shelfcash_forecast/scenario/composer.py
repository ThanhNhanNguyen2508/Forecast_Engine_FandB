# Nếu residuals.py trả lời:

# “Lịch sử sai số OOS của model là gì?”

# và contracts.py trả lời:

# “Scenario output phải có cấu trúc thế nào?”

# thì composer.py trả lời:

# “Với forecast hiện tại + residual history, ta sẽ dùng thuật toán nào để tạo ProductDemandScenarioBundle?”
# Historical walk-forward predictions
#         ↓
# residuals.py
#         ↓
# Residual History
#         │
#         │
# Current ForecastPackage
#         │
#         └──────────────┐
#                        ↓
#                 composer.py
#                        │
#              chọn scenario method
#                        │
#           ┌────────────┴────────────┐
#           ↓                         ↓
#  residual_bootstrap          gaussian_copula
#  bootstrap.py                   copula.py
#           │                         │
#           └────────────┬────────────┘
#                        ↓
#           ProductDemandScenarioBundle
#                        ↓
#                 Scenario BOM
#                        ↓
#        IngredientDemandScenarioBundle
from __future__ import annotations

from typing import Protocol

import pandas as pd

from shelfcash_forecast.contracts import ForecastPackage
from shelfcash_forecast.scenario.bootstrap import (
    ResidualVectorBootstrapScenarioGenerator,
)
# 2 generator implementations: residual_bootstrap, gaussian_copula
from shelfcash_forecast.scenario.contracts import ProductDemandScenarioBundle
from shelfcash_forecast.scenario.copula import GaussianCopulaScenarioGenerator


class ScenarioGenerator(Protocol):
    method: str

    def generate(
        self,
        forecast: ForecastPackage,
        residual_history: pd.DataFrame,
        *,
        n_scenarios: int,
        seed: int,
    ) -> ProductDemandScenarioBundle: ...


def generate_product_demand_scenarios(
    forecast_package: ForecastPackage,
    residual_history: pd.DataFrame,
    *,
    n_scenarios: int,
    seed: int,
    method: str = "residual_bootstrap",
) -> ProductDemandScenarioBundle:
    generators: dict[str, ScenarioGenerator] = {
        "residual_bootstrap": ResidualVectorBootstrapScenarioGenerator(),
        "gaussian_copula": GaussianCopulaScenarioGenerator(),
    }
    try:
        generator = generators[method]
    except KeyError as exc:
        raise ValueError(
            f"scenario method không được hỗ trợ: {method!r}; "
            f"chọn một trong {sorted(generators)}."
        ) from exc
    return generator.generate(
        forecast_package,
        residual_history,
        n_scenarios=n_scenarios,
        seed=seed,
    )
