from __future__ import annotations

from typing import Any

import pandas as pd

from shelfcash_forecast.baselines.ets import ets_predict_rows_detailed
from shelfcash_forecast.baselines.moving_average import moving_average_predict
from shelfcash_forecast.baselines.seasonal_naive import seasonal_naive_predict
from shelfcash_forecast.calibration.crossing import correct_quantile_crossing
from shelfcash_forecast.config import ForecastConfig
from shelfcash_forecast.evaluation.metrics import (
    comparison_diagnostics,
    evaluate_point_prediction,
    evaluate_quantile_predictions,
    metric_breakdown_by_horizon,
    model_comparison_table,
    point_metric_breakdown_by_horizon,
)
from shelfcash_forecast.evaluation.splits import generate_walk_forward_folds
from shelfcash_forecast.features.specification import CategoryEncoder
from shelfcash_forecast.models.predictor import predict_raw_quantiles
from shelfcash_forecast.models.trainer import train_model_bundle


def run_walk_forward(
    training_period: pd.DataFrame,
    config: ForecastConfig,
    panel: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    folds = generate_walk_forward_folds(
        training_period["target_date"],
        minimum_train_days=config.walk_forward_minimum_train_days,
        validation_days=config.walk_forward_validation_days,
        step_days=config.walk_forward_step_days,
        maximum_folds=config.walk_forward_maximum_folds,
    )
    if not folds:
        return pd.DataFrame(), {"status": "skipped", "reason": "INSUFFICIENT_FOLD_DATES"}

    predictions: list[pd.DataFrame] = []
    validation_counts = {
        "candidate_rows": 0,
        "eligible_rows": 0,
        "excluded_rows": 0,
        "excluded_target_ineligible_rows": 0,
        "excluded_missing_target_rows": 0,
        "excluded_insufficient_history_rows": 0,
    }
    for fold in folds:
        target_eligible = (
            training_period["target_train_eligible"].astype("boolean").fillna(False)
        )
        train_fold = training_period.loc[
            training_period["target_date"].le(fold.train_end)
            & target_eligible
            & training_period["target"].notna()
            & training_period["history_observation_count"].ge(
                config.minimum_history_observations
            )
        ].copy()
        valid_candidates = training_period.loc[
            training_period["target_date"].between(
                fold.validation_start, fold.validation_end
            )
        ].copy()
        origin_safe = valid_candidates["cutoff_date"].ge(fold.train_end)
        valid_target_eligible = (
            valid_candidates["target_train_eligible"].astype("boolean").fillna(False)
        )
        target_available = valid_candidates["target"].notna()
        history_sufficient = valid_candidates["history_observation_count"].ge(
            config.minimum_history_observations
        )
        valid_mask = (
            origin_safe
            & valid_target_eligible
            & target_available
            & history_sufficient
        )
        valid_fold = valid_candidates.loc[valid_mask].copy()

        validation_counts["candidate_rows"] += len(valid_candidates)
        validation_counts["eligible_rows"] += int(valid_mask.sum())
        validation_counts["excluded_rows"] += int((~valid_mask).sum())
        validation_counts["excluded_target_ineligible_rows"] += int(
            (~valid_target_eligible).sum()
        )
        validation_counts["excluded_missing_target_rows"] += int(
            (~target_available).sum()
        )
        validation_counts["excluded_insufficient_history_rows"] += int(
            (~history_sufficient).sum()
        )
        validation_counts.setdefault("purged_origin_unavailable_rows", 0)
        validation_counts["purged_origin_unavailable_rows"] += int(
            (~origin_safe).sum()
        )
        if train_fold.empty or valid_fold.empty:
            continue

        encoder = CategoryEncoder.fit(train_fold)
        bundle = train_model_bundle(encoder.transform(train_fold), config)
        fold_prediction = correct_quantile_crossing(
            predict_raw_quantiles(bundle, encoder.transform(valid_fold))
        )
        fold_prediction["baseline_p50"] = seasonal_naive_predict(valid_fold).to_numpy()
        fold_prediction["ma7_p50"] = moving_average_predict(valid_fold, 7).to_numpy()
        fold_prediction["ma28_p50"] = moving_average_predict(valid_fold, 28).to_numpy()
        if panel is not None:
            ets = ets_predict_rows_detailed(
                valid_fold,
                panel,
                fallback=fold_prediction["baseline_p50"],
            )
            fold_prediction["ets_p50"] = ets["prediction"].to_numpy()
            fold_prediction["ets_fallback_used"] = ets["fallback_used"].to_numpy()
            fold_prediction["ets_fallback_reason"] = ets[
                "fallback_reason"
            ].to_numpy()
            fold_prediction["ets_fit_method"] = ets["fit_method"].to_numpy()
        fold_prediction["fold_id"] = fold.fold_id
        predictions.append(fold_prediction)

    if not predictions:
        return pd.DataFrame(), {
            "status": "skipped",
            "reason": "EMPTY_FOLDS",
            "validation_eligibility": validation_counts,
        }

    combined = pd.concat(predictions, ignore_index=True)
    baseline_columns = {
        "seasonal_naive": "baseline_p50",
        "ma7": "ma7_p50",
        "ma28": "ma28_p50",
    }
    if "ets_p50" in combined:
        baseline_columns["ets"] = "ets_p50"
    comparison = model_comparison_table(combined, baseline_columns)
    metrics = {
        "status": "completed",
        "fold_count": int(combined["fold_id"].nunique()),
        "lightgbm": evaluate_quantile_predictions(combined),
        "lightgbm_by_horizon": metric_breakdown_by_horizon(combined),
        "validation_eligibility": validation_counts,
        "by_fold": {
            str(int(fold_id)): evaluate_quantile_predictions(group)
            for fold_id, group in combined.groupby("fold_id", observed=True)
        },
        "model_comparison": comparison,
        "comparison_diagnostics": comparison_diagnostics(comparison),
    }
    for name, column in baseline_columns.items():
        metrics[name] = evaluate_point_prediction(combined, column)
        metrics[f"{name}_by_horizon"] = point_metric_breakdown_by_horizon(
            combined, column
        )
    return combined, metrics
