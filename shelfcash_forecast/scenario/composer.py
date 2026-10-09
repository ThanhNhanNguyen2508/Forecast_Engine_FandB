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
    if any(p.forecast_method=='DECLARED_FORECAST_OVERRIDE' for p in forecast_package.predictions):
        return _declared_override_scenarios(forecast_package,residual_history,n_scenarios=n_scenarios,seed=seed,method=method)
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


def _declared_override_scenarios(forecast,residual_history,*,n_scenarios,seed,method):
    import numpy as np
    from shelfcash_forecast.pipeline.forecast_overrides import ForecastOverridePolicy
    from shelfcash_forecast.scenario.contracts import ProductDemandScenario,ProductDemandScenarioLine
    policy=ForecastOverridePolicy.model_validate(forecast.scenario_assumptions)
    cold=[p for p in forecast.predictions if p.forecast_method=='DECLARED_FORECAST_OVERRIDE']
    known=[p for p in forecast.predictions if p.forecast_method!='DECLARED_FORECAST_OVERRIDE']
    if not 1<=n_scenarios<=2000:raise ValueError('SCENARIO_COUNT_OUTSIDE_SUPPORTED_ENVELOPE')
    base=generate_product_demand_scenarios(forecast.model_copy(update={'predictions':known}),residual_history,
        n_scenarios=n_scenarios,seed=seed,method=method) if known else None
    rng=np.random.default_rng(seed);scenarios=[]
    for i in range(n_scenarios):
        world=base.scenarios[i] if base else None
        sid=world.scenario_id if world else f'scenario_{i+1:04d}'
        level=int(rng.choice(len(policy.scenario_weights),p=policy.scenario_weights))
        lines=list(world.lines) if world else []
        lines.extend(ProductDemandScenarioLine(scenario_id=sid,store_id=p.store_id,product_id=p.product_id,
            product_name=p.product_name,product_unit=p.unit,target_date=p.target_date,horizon=p.horizon,
            demand_quantity=p.p50*policy.scenario_multipliers[level],source_model_version='DECLARED_FORECAST_OVERRIDE:'+policy.scenario_evidence_id,
            scenario_method='declared_cold_start_levels') for p in cold)
        scenarios.append(ProductDemandScenario(scenario_id=sid,probability_weight=world.probability_weight if world else 1/n_scenarios,
            lines=lines,metadata={'declared_override_level':level,'scenario_evidence_id':policy.scenario_evidence_id,'quality':'NOT_CALIBRATED_OR_OOS_VALIDATED'}))
    return ProductDemandScenarioBundle(forecast_date=forecast.forecast_date,horizon=forecast.forecast_horizon,model_version=forecast.model_version,
        scenario_method='residual_bootstrap_with_declared_overrides' if known else 'declared_cold_start_levels',scenarios=scenarios,
        diagnostics={'declared_distribution':policy.model_dump(mode='json'),'known_product_count':len({p.product_id for p in known}),
            'cold_product_count':len({p.product_id for p in cold}),'fixed_scenario_method':method if known else None,
            'not_accuracy_validation':True},warnings=['COLD_START_SCENARIO_ASSUMPTIONS_NOT_EMPIRICALLY_VALIDATED'])
