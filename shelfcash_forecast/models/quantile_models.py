from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import lightgbm as lgb
import numpy as np
import pandas as pd


class PredictableModel(Protocol):
    def predict(self, data: pd.DataFrame) -> np.ndarray: ...


@dataclass
class QuantileModelBundle:
    models: dict[float, PredictableModel]
    feature_names: list[str]
    categorical_features: list[str]
    target_strategy: str = "absolute"
    residual_blend_lambda: float = 1.0
    best_iterations: dict[float, int] | None = None
    objectives: dict[float, str] | None = None


def build_quantile_model(
    quantile: float,
    base_params: dict[str, object],
    *,
    objective: str = "quantile",
) -> lgb.LGBMRegressor:
    params = dict(base_params)
    params["objective"] = objective
    if objective == "quantile":
        params["alpha"] = quantile
    return lgb.LGBMRegressor(**params)


def train_quantile_models(
    train: pd.DataFrame,
    feature_names: list[str],
    categorical_features: list[str],
    quantiles: tuple[float, ...],
    base_params: dict[str, object],
    *,
    target_column: str = "target",
    validation: pd.DataFrame | None = None,
    early_stopping_rounds: int | None = None,
    p50_objective: str = "quantile",
    target_strategy: str = "absolute",
    residual_blend_lambda: float = 1.0,
    n_estimators_by_quantile: dict[float, int] | None = None,
) -> QuantileModelBundle:
    x_train = train[feature_names]
    y_train = train[target_column].astype(float)
    models: dict[float, PredictableModel] = {}
    best_iterations: dict[float, int] = {}
    objectives: dict[float, str] = {}

    for quantile in quantiles:
        objective = p50_objective if quantile == 0.50 else "quantile"
        params = dict(base_params)
        if n_estimators_by_quantile is not None:
            params["n_estimators"] = int(n_estimators_by_quantile[quantile])
        model = build_quantile_model(quantile, params, objective=objective)
        fit_kwargs: dict[str, object] = {
            "categorical_feature": categorical_features,
        }
        if validation is not None:
            fit_kwargs["eval_set"] = [
                (validation[feature_names], validation[target_column].astype(float))
            ]
            fit_kwargs["eval_metric"] = "quantile" if objective == "quantile" else "l1"
            if early_stopping_rounds is not None:
                fit_kwargs["callbacks"] = [
                    lgb.early_stopping(early_stopping_rounds, verbose=False)
                ]
        model.fit(x_train, y_train, **fit_kwargs)
        models[quantile] = model
        best = getattr(model, "best_iteration_", None)
        best_iterations[quantile] = int(best or params.get("n_estimators", 0))
        objectives[quantile] = objective

    return QuantileModelBundle(
        models=models,
        feature_names=feature_names,
        categorical_features=categorical_features,
        target_strategy=target_strategy,
        residual_blend_lambda=float(residual_blend_lambda),
        best_iterations=best_iterations,
        objectives=objectives,
    )
