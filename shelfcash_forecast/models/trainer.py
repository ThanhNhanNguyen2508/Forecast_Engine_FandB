from __future__ import annotations

import pandas as pd

from shelfcash_forecast.config import ForecastConfig
from shelfcash_forecast.features.specification import (
    CATEGORICAL_MODEL_COLUMNS,
    MODEL_FEATURES,
)
from shelfcash_forecast.models.quantile_models import (
    QuantileModelBundle,
    train_quantile_models,
)


def train_model_bundle(
    train: pd.DataFrame,
    config: ForecastConfig,
    *,
    target_column: str = "target",
    validation: pd.DataFrame | None = None,
    early_stopping_rounds: int | None = None,
    p50_objective: str = "quantile",
    target_strategy: str = "absolute",
    residual_blend_lambda: float = 1.0,
    n_estimators_by_quantile: dict[float, int] | None = None,
) -> QuantileModelBundle:
    return train_quantile_models(
        train=train,
        feature_names=list(MODEL_FEATURES),
        categorical_features=list(CATEGORICAL_MODEL_COLUMNS),
        quantiles=config.quantiles,
        base_params=config.lightgbm_params,
        target_column=target_column,
        validation=validation,
        early_stopping_rounds=early_stopping_rounds,
        p50_objective=p50_objective,
        target_strategy=target_strategy,
        residual_blend_lambda=residual_blend_lambda,
        n_estimators_by_quantile=n_estimators_by_quantile,
    )
