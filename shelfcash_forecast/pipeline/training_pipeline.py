from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from shelfcash_forecast.baselines.ets import ETS_FIT_METHOD, ets_predict_rows_detailed
from shelfcash_forecast.baselines.moving_average import moving_average_predict_rows
from shelfcash_forecast.baselines.seasonal_naive import seasonal_naive_predict_rows
from shelfcash_forecast.calibration.cqr import apply_cqr_calibrator, fit_cqr_calibrator
from shelfcash_forecast.calibration.crossing import correct_quantile_crossing
from shelfcash_forecast.calibration.metrics import (
    calibration_breakdown_by_horizon,
    calibration_metrics,
)
from shelfcash_forecast.calibration.point import PointCorrector, apply_point_corrector
from shelfcash_forecast.config import ForecastConfig
from shelfcash_forecast.contracts import TrainingResult
from shelfcash_forecast.data.adapter import adapt_forecast_input
from shelfcash_forecast.data.demand_reconstruction import reconstruct_demand
from shelfcash_forecast.data.panel_builder import build_daily_panel, resolve_missing_sales
from shelfcash_forecast.data.validator import validate_calendar, validate_sales
from shelfcash_forecast.evaluation.metrics import (
    comparison_diagnostics,
    evaluate_point_prediction,
    evaluate_quantile_predictions,
    metric_breakdown_by_horizon,
    model_comparison_table,
    point_metric_breakdown_by_horizon,
)
from shelfcash_forecast.evaluation.splits import (
    apply_final_split,
    final_split_origin_audit,
    make_final_split,
)
from shelfcash_forecast.evaluation.walk_forward import run_walk_forward
from shelfcash_forecast.features.future import (
    add_calendar_future_features,
    add_deterministic_future_features,
)
from shelfcash_forecast.features.historical import add_historical_features
from shelfcash_forecast.features.specification import CategoryEncoder
from shelfcash_forecast.features.training_table import (
    add_target_seasonal_lags,
    build_training_table,
)
from shelfcash_forecast.models.predictor import predict_raw_quantiles
from shelfcash_forecast.models.trainer import train_model_bundle
from shelfcash_forecast.registry.governance import (
    FINGERPRINT_ALGORITHM,
    dataset_fingerprint,
    runtime_versions,
)
from shelfcash_forecast.registry.writer import write_artifacts
from shelfcash_forecast.scenario.residuals import build_walk_forward_residual_history


def _prepare_modelling_table(
    canonical_data: Mapping[str, pd.DataFrame],
    config: ForecastConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, dict[str, object]]:
    adapted = adapt_forecast_input(canonical_data, config)
    sales, quality_report = validate_sales(adapted.sales_history)
    calendar = validate_calendar(adapted.calendar_features, quality_report)
    panel = build_daily_panel(sales, calendar)
    panel = resolve_missing_sales(panel)
    panel = reconstruct_demand(panel, config)
    panel = add_historical_features(panel, config)
    table = build_training_table(panel, config)
    table = add_target_seasonal_lags(table, panel, config)
    table = add_deterministic_future_features(table)
    table = add_calendar_future_features(table, calendar)
    return table, panel, calendar, quality_report.to_dict()


def _eligible_rows(frame: pd.DataFrame, config: ForecastConfig) -> pd.DataFrame:
    return frame.loc[
        frame["target_train_eligible"].astype("boolean").fillna(False)
        & frame["target"].notna()
        & frame["history_observation_count"].ge(config.minimum_history_observations)
    ].copy()


def train_forecast_core(
    canonical_data: Mapping[str, pd.DataFrame],
    artifact_directory: str | Path,
    config: ForecastConfig | None = None,
    model_version: str = "forecast-core-v0.1.0",
) -> TrainingResult:
    config = config or ForecastConfig()
    table, panel, calendar, quality_report = _prepare_modelling_table(
        canonical_data, config
    )
    future_weather_used = bool(
        calendar is not None
        and not calendar.empty
        and calendar[["temperature", "rainfall"]].notna().any().any()
    )
    fingerprint_columns = [
        "date",
        "store_key",
        "product_key",
        "product_name",
        "unit",
        "observed_sales",
        "reconstructed_demand",
        "feature_demand",
        "is_stockout",
        "target_quality",
        "train_eligible",
    ]
    fingerprint = dataset_fingerprint(panel, columns=fingerprint_columns)
    split_source = table.loc[table["target"].notna()].copy()
    split = make_final_split(
        split_source,
        calibration_days=config.calibration_days,
        test_days=config.test_days,
        minimum_train_dates=config.walk_forward_minimum_train_days,
    )
    train_period, calibration_period, test_period = apply_final_split(table, split)
    origin_audit = final_split_origin_audit(table, split)
    walk_forward_predictions, walk_forward_metrics = run_walk_forward(
        train_period, config, panel
    )
    walk_forward_residuals = build_walk_forward_residual_history(
        walk_forward_predictions
    )

    train = _eligible_rows(train_period, config)
    calibration = _eligible_rows(calibration_period, config)
    test = _eligible_rows(test_period, config)
    if train.empty or calibration.empty or test.empty:
        raise ValueError(
            "Train/calibration/test is empty after eligibility filtering; check "
            "history length, stockouts, and missing sales."
        )

    encoder = CategoryEncoder.fit(train)
    encoded_train = encoder.transform(train)
    encoded_calibration = encoder.transform(calibration)
    encoded_test = encoder.transform(test)
    model_bundle = train_model_bundle(encoded_train, config)
    point_corrector = PointCorrector.identity(model_id=model_version)

    calibration_predictions = apply_point_corrector(
        correct_quantile_crossing(
            predict_raw_quantiles(model_bundle, encoded_calibration)
        ),
        point_corrector,
    )
    calibrator = fit_cqr_calibrator(
        calibration_predictions,
        desired_coverage=config.nominal_coverage,
        minimum_samples=config.minimum_calibration_samples,
    )
    test_predictions = apply_point_corrector(
        correct_quantile_crossing(
            predict_raw_quantiles(model_bundle, encoded_test)
        ),
        point_corrector,
    )
    test_predictions = apply_cqr_calibrator(test_predictions, calibrator)

    seasonal = seasonal_naive_predict_rows(test)
    ma7 = moving_average_predict_rows(test, 7)
    ma28 = moving_average_predict_rows(test, 28)
    test_predictions["baseline_p50"] = seasonal["prediction"].to_numpy()
    test_predictions["ma7_p50"] = ma7["prediction"].to_numpy()
    test_predictions["ma28_p50"] = ma28["prediction"].to_numpy()
    test_predictions["seasonal_naive_fallback_used"] = seasonal[
        "fallback_used"
    ].to_numpy()
    test_predictions["ma7_fallback_used"] = ma7["fallback_used"].to_numpy()
    test_predictions["ma28_fallback_used"] = ma28["fallback_used"].to_numpy()
    ets = ets_predict_rows_detailed(
        test,
        panel,
        fallback=test_predictions["baseline_p50"],
    )
    test_predictions["ets_p50"] = ets["prediction"].to_numpy()
    test_predictions["ets_fallback_used"] = ets["fallback_used"].to_numpy()
    test_predictions["ets_fallback_reason"] = ets["fallback_reason"].to_numpy()
    test_predictions["ets_fit_method"] = ets["fit_method"].to_numpy()

    test_metrics = {
        "overall": evaluate_quantile_predictions(test_predictions),
        "by_horizon": metric_breakdown_by_horizon(test_predictions),
    }
    baseline_columns = {
        "seasonal_naive": "baseline_p50",
        "ma7": "ma7_p50",
        "ma28": "ma28_p50",
        "ets": "ets_p50",
    }
    comparison = model_comparison_table(test_predictions, baseline_columns)
    baseline_metrics: dict[str, object] = {
        name: evaluate_point_prediction(test_predictions, column)
        for name, column in baseline_columns.items()
    }
    baseline_metrics["by_horizon"] = {
        name: point_metric_breakdown_by_horizon(test_predictions, column)
        for name, column in baseline_columns.items()
    }
    baseline_metrics["model_comparison"] = comparison
    baseline_metrics["comparison_diagnostics"] = comparison_diagnostics(comparison)
    calibration_metrics_payload = {
        "overall": calibration_metrics(test_predictions, config.nominal_coverage),
        "by_horizon": calibration_breakdown_by_horizon(
            test_predictions, config.nominal_coverage
        ),
    }

    warnings = list(quality_report.get("warnings", []))
    model_wape = test_metrics["overall"].get("wape_p50")
    baseline_wape = baseline_metrics["seasonal_naive"].get("wape")
    if (
        model_wape is not None
        and baseline_wape is not None
        and model_wape > baseline_wape
    ):
        warnings.append("MODEL_WORSE_THAN_SEASONAL_NAIVE")

    metadata = {
        "model_version": model_version,
        "target": "reconstructed_product_demand_when_supported",
        "source_target_column": "quantity_sold",
        "grain": "store_product_day",
        "horizons": list(config.horizons),
        "quantiles": list(config.quantiles),
        "nominal_interval_coverage": config.nominal_coverage,
        "censored_target_policy": (
            "use_high_or_medium_confidence_chronological_reconstruction;"
            "otherwise_exclude"
        ),
        "reconstruction_confidence_thresholds": {
            "minimum_reference_count": config.reconstruction_minimum_reference_count,
            "high_support_threshold": config.reconstruction_high_support_threshold,
            "high_recency_days": config.reconstruction_high_recency_days,
            "medium_recency_days": config.reconstruction_medium_recency_days,
            "high_relative_iqr": config.reconstruction_high_dispersion_threshold,
            "medium_relative_iqr": config.reconstruction_medium_dispersion_threshold,
        },
        "missing_stockout_policy": "use_observed_sales_with_warning",
        "future_price_used": False,
        "future_promotion_used": False,
        "future_weather_used": future_weather_used,
        "future_weather_policy": (
            "caller_supplied_values_used"
            if future_weather_used
            else "excluded_or_missing"
        ),
        "split": split.to_dict(),
        "origin_availability_audit": origin_audit,
        "dataset_fingerprint": fingerprint,
        "fingerprint_algorithm": FINGERPRINT_ALGORITHM,
        "runtime_versions": runtime_versions(),
        "artifact_fitted_at": datetime.now(timezone.utc).isoformat(),
        "training_label_cutoff": pd.Timestamp(train["target_date"].max()).date().isoformat(),
        "calibration_label_cutoff": pd.Timestamp(calibration["target_date"].max()).date().isoformat(),
        "test_label_cutoff": pd.Timestamp(test["target_date"].max()).date().isoformat(),
        "cqr_semantics_version": "cqr-order-statistic-v2",
        "promotion_status": "trained_candidate",
        "production_available_at": None,
        "warnings": sorted(set(warnings)),
    }
    training_manifest = {
        "sales_rows": int(quality_report["rows_valid"]),
        "panel_rows": len(panel),
        "training_table_rows": len(table),
        "train_rows": len(train),
        "calibration_rows": len(calibration),
        "test_rows": len(test),
        "store_count": int(panel["store_key"].nunique()),
        "product_count": int(panel["product_key"].nunique()),
        "history_start": panel["date"].min().date().isoformat(),
        "history_end": panel["date"].max().date().isoformat(),
        "walk_forward_residual_rows": len(walk_forward_residuals),
        "walk_forward_residual_source": "walk_forward_oos",
        "walk_forward_residuals_eligible_only": True,
        "walk_forward_actual_target_semantics": (
            "reconstructed_demand_when_accepted_else_eligible_observed_sales"
        ),
        "origin_availability_audit": origin_audit,
        "ets_fit_method": ETS_FIT_METHOD,
        "dataset_fingerprint": fingerprint,
        "fingerprint_algorithm": FINGERPRINT_ALGORITHM,
    }

    artifact_dir = write_artifacts(
        artifact_directory=artifact_directory,
        model_bundle=model_bundle,
        calibrator=calibrator,
        encoder=encoder,
        config=config,
        model_version=model_version,
        metadata=metadata,
        quality_report=quality_report,
        baseline_metrics=baseline_metrics,
        walk_forward_metrics=walk_forward_metrics,
        test_metrics=test_metrics,
        calibration_metrics_payload=calibration_metrics_payload,
        training_manifest=training_manifest,
        predictions={
            "walk_forward_predictions": walk_forward_predictions,
            "walk_forward_residuals": walk_forward_residuals,
            "calibration_predictions": calibration_predictions,
            "test_predictions": test_predictions,
        },
        point_corrector=point_corrector,
    )
    return TrainingResult(
        status="success",
        model_version=model_version,
        artifact_directory=str(artifact_dir),
        data_quality=quality_report,
        baseline_metrics=baseline_metrics,
        walk_forward_metrics=walk_forward_metrics,
        test_metrics=test_metrics,
        calibration_metrics=calibration_metrics_payload,
        warnings=sorted(set(warnings)),
    )
