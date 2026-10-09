from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

from shelfcash_forecast.baselines.seasonal_naive import seasonal_naive_predict
from shelfcash_forecast.calibration.cqr import apply_cqr_calibrator
from shelfcash_forecast.calibration.crossing import correct_quantile_crossing
from shelfcash_forecast.calibration.point import apply_point_corrector
from shelfcash_forecast.contracts import ForecastPackage, ForecastPrediction
from shelfcash_forecast.data.adapter import adapt_forecast_input
from shelfcash_forecast.data.demand_reconstruction import reconstruct_demand
from shelfcash_forecast.data.panel_builder import build_daily_panel, resolve_missing_sales
from shelfcash_forecast.data.validator import DataQualityReport, validate_calendar, validate_sales
from shelfcash_forecast.exceptions import InsufficientDataError
from shelfcash_forecast.features.future import (
    add_calendar_future_features,
    add_deterministic_future_features,
)
from shelfcash_forecast.features.historical import add_historical_features
from shelfcash_forecast.features.specification import validate_runtime_feature_schema
from shelfcash_forecast.features.training_table import (
    add_target_seasonal_lags,
    build_runtime_rows,
)
from shelfcash_forecast.models.predictor import predict_raw_quantiles
from shelfcash_forecast.registry.loader import LoadedArtifacts, load_artifacts


@dataclass(frozen=True)
class ForecastState:
    """In-memory checkpoint shared by the public predictor and stage runners.

    M1 contains repaired quantiles, point contains corrected quantiles, and M2
    adds decision intervals plus the final planned-closure policy. No stage fits
    or serializes a model. Keep this state in memory, not in historical artifacts.
    """

    frame: pd.DataFrame
    artifacts: LoadedArtifacts
    cutoff: pd.Timestamp
    forecast_horizon: int
    execution_mode: str
    quality_report: DataQualityReport
    stage: str = "m1"
    override_policy: object | None = None


def predict_m1(
    canonical_data: Mapping[str, pd.DataFrame],
    artifact_directory: str | Path,
    cutoff_date: str | pd.Timestamp,
    forecast_horizon: int = 7,
    *,
    execution_mode: str = "production",
    override_policy=None,
) -> ForecastState:
    """Load fixed artifacts and stop after M1 quantile crossing repair."""

    artifacts = load_artifacts(artifact_directory)
    config = artifacts.config
    cutoff = pd.Timestamp(cutoff_date).normalize()
    if execution_mode not in {"production", "backtest_replay", "demo"}:
        raise ValueError("execution_mode must be production, backtest_replay, or demo.")
    for field in ("training_label_cutoff", "calibration_label_cutoff"):
        value = artifacts.metadata.get(field)
        if value is None or pd.Timestamp(value).normalize() > cutoff:
            raise InsufficientDataError(
                f"ARTIFACT_TEMPORALLY_INELIGIBLE:{field}={value}:origin={cutoff.date()}"
            )
    if execution_mode == "production":
        available = artifacts.metadata.get("production_available_at")
        if artifacts.metadata.get("promotion_status") != "promoted" or available is None:
            raise InsufficientDataError("ARTIFACT_NOT_PROMOTED_FOR_PRODUCTION")
        if pd.Timestamp(available).tz_localize(None).normalize() > cutoff:
            raise InsufficientDataError("ARTIFACT_NOT_AVAILABLE_AT_PRODUCTION_ORIGIN")
    if forecast_horizon < 1 or forecast_horizon > (366 if override_policy else max(config.horizons)):
        raise InsufficientDataError(f"FIXED_MODEL_HORIZON_UNSUPPORTED: supported fixed range 1..{max(config.horizons)}; declared override range 1..366")

    adapted = adapt_forecast_input(canonical_data, config)
    sales, quality_report = validate_sales(adapted.sales_history)
    calendar = validate_calendar(adapted.calendar_features, quality_report)
    sales = sales.loc[sales["date"].le(cutoff)].copy()
    override_frame = None
    if override_policy is not None:
        from shelfcash_forecast.pipeline.forecast_overrides import override_rows
        override_frame=override_rows(override_policy,cutoff,forecast_horizon,calendar)
        overridden={(p.store_id,p.product_id) for p in override_policy.predictions}
        sales=sales[~pd.MultiIndex.from_frame(sales[['store_key','product_key']]).isin(overridden)]
    if sales.empty and override_frame is not None:
        return ForecastState(override_frame,artifacts,cutoff,forecast_horizon,execution_mode,quality_report,override_policy=override_policy)
    if sales.empty:
        raise InsufficientDataError("No sales_history exists at or before cutoff_date.")
    if forecast_horizon>max(config.horizons):raise InsufficientDataError('FIXED_MODEL_HORIZON_UNSUPPORTED: override every applicable product or use trained range')

    panel = build_daily_panel(sales, calendar, end_date=cutoff)
    panel = resolve_missing_sales(panel)
    panel = reconstruct_demand(panel, config)
    panel = add_historical_features(panel, config)
    runtime = build_runtime_rows(panel, cutoff, forecast_horizon)
    runtime = add_target_seasonal_lags(runtime, panel, config)
    runtime = add_deterministic_future_features(runtime)
    runtime = add_calendar_future_features(runtime, calendar)
    runtime = artifacts.encoder.transform(runtime)
    if runtime['product_code'].eq(-1).any() or runtime['store_code'].eq(-1).any() or runtime['history_observation_count'].lt(config.minimum_history_observations).any():
        raise InsufficientDataError('COLD_START_POLICY_REQUIRED: explicit forecast overrides and scenario assumptions required')
    validate_runtime_feature_schema(
        runtime,
        expected_features=artifacts.model_bundle.feature_names,
        expected_categorical_features=artifacts.model_bundle.categorical_features,
    )

    runtime = correct_quantile_crossing(
        predict_raw_quantiles(artifacts.model_bundle, runtime)
    )
    runtime['forecast_method']='FIXED_LIGHTGBM_POINT_CQR'
    if override_frame is not None:runtime=pd.concat([runtime,override_frame],ignore_index=True)
    return ForecastState(
        frame=runtime,
        artifacts=artifacts,
        cutoff=cutoff,
        forecast_horizon=forecast_horizon,
        execution_mode=execution_mode,
        quality_report=quality_report,
        override_policy=override_policy,
    )


def correct_forecast_point(state: ForecastState) -> ForecastState:
    """Apply the frozen point corrector once, before interval calibration."""
    if state.stage != "m1":
        raise ValueError("Point correction requires an M1 checkpoint.")
    mask=state.frame.get('forecast_method',pd.Series('FIXED_LIGHTGBM_POINT_CQR',index=state.frame.index)).eq('DECLARED_FORECAST_OVERRIDE')
    if mask.all():runtime=state.frame.copy()
    elif mask.any():runtime=pd.concat([apply_point_corrector(state.frame.loc[~mask],state.artifacts.point_corrector),state.frame.loc[mask]],ignore_index=True)
    else:runtime = apply_point_corrector(state.frame, state.artifacts.point_corrector)
    return replace(state, frame=runtime, stage="point")


def calibrate_forecast(state: ForecastState) -> ForecastState:
    """Apply current CQR code and finalize decision values; never fit CQR."""
    if state.stage != "point":
        raise ValueError("M2 calibration requires a point-corrected checkpoint.")
    mask=state.frame.get('forecast_method',pd.Series('FIXED_LIGHTGBM_POINT_CQR',index=state.frame.index)).eq('DECLARED_FORECAST_OVERRIDE')
    if mask.all():runtime=state.frame.copy()
    else:
        fixed=apply_cqr_calibrator(state.frame.loc[~mask],state.artifacts.calibrator)
        fixed['baseline_p50']=seasonal_naive_predict(fixed).to_numpy()
        runtime=pd.concat([fixed,state.frame.loc[mask]],ignore_index=True) if mask.any() else fixed
    closed = runtime["target_store_closed"].eq(1)
    for column in (
        "p25",
        "p50",
        "p75",
        "interval_lower",
        "interval_upper",
        "baseline_p50",
    ):
        runtime.loc[closed, column] = 0.0
    return replace(state, frame=runtime, stage="m2")


def build_forecast_package(state: ForecastState) -> ForecastPackage:
    """Build the existing public contract only after the complete M1/M2 stack."""
    if state.stage != "m2":
        raise ValueError("ForecastPackage requires a completed M2 checkpoint.")
    runtime = state.frame
    artifacts = state.artifacts
    config = artifacts.config
    quality_report = state.quality_report
    execution_mode = state.execution_mode

    global_warnings = list(quality_report.warnings) + list(artifacts.warnings)
    if execution_mode != "production":
        global_warnings.append(f"EXECUTION_MODE_{execution_mode.upper()}")
    predictions: list[ForecastPrediction] = []
    for row in runtime.itertuples(index=False):
        warnings: list[str] = []
        if getattr(row,'forecast_method','')=='DECLARED_FORECAST_OVERRIDE':
            warnings.append('COLD_START_DECLARED_DEMAND_NOT_MODEL_PREDICTION')
            warnings.append('DECLARED_INTERVAL_NOT_EMPIRICALLY_CALIBRATED')
        if row.product_code == -1:
            warnings.append("UNSEEN_PRODUCT")
        if row.calibration_source == "global":
            warnings.append("CALIBRATION_FALLBACK_GLOBAL")
        if pd.isna(row.seasonal_lag_7_target):
            warnings.append("INSUFFICIENT_SEASONAL_HISTORY")
        if row.target_store_closed == 1:
            warnings.append("STORE_PLANNED_CLOSED")
        if row.history_observation_count < config.minimum_history_observations:
            warnings.append("INSUFFICIENT_HISTORY")
        predictions.append(
            ForecastPrediction(
                store_id=str(row.store_key),
                product_id=str(row.product_key),
                product_name=str(row.product_name),
                unit=None if pd.isna(row.unit) else str(row.unit),
                target_date=pd.Timestamp(row.target_date).date(),
                horizon=int(row.horizon),
                p25=float(row.p25),
                p50=float(row.p50),
                p75=float(row.p75),
                interval_lower=float(row.interval_lower),
                interval_upper=float(row.interval_upper),
                baseline_p50=float(row.baseline_p50),
                calibration_source=str(row.calibration_source),
                warnings=warnings,
                forecast_method=getattr(row,'forecast_method','FIXED_LIGHTGBM_POINT_CQR'),
                provenance={'evidence_id':getattr(row,'forecast_evidence_id',None),'classification':getattr(row,'forecast_classification',None)} if getattr(row,'forecast_method','')=='DECLARED_FORECAST_OVERRIDE' else {},
            )
        )
    return ForecastPackage(
        forecast_date=state.cutoff.date(),
        forecast_horizon=state.forecast_horizon,
        model_version=('DECLARED_FORECAST_OVERRIDE:'+state.override_policy.scenario_evidence_id
            if predictions and all(p.forecast_method=='DECLARED_FORECAST_OVERRIDE' for p in predictions)
            else str(artifacts.metadata["model_version"])),
        predictions=predictions,
        warnings=sorted(set(global_warnings)),
        scenario_assumptions=state.override_policy.model_dump(mode='json') if state.override_policy else {},
    )


def predict_demand(
    canonical_data: Mapping[str, pd.DataFrame],
    artifact_directory: str | Path,
    cutoff_date: str | pd.Timestamp,
    forecast_horizon: int = 7,
    *,
    execution_mode: str = "production",
    override_policy=None,
) -> ForecastPackage:
    """Load immutable artifacts and forecast strictly after the inclusive cutoff.

    The existing public API and the milestone runner use these same stages so
    changes to features, point correction or calibration have one execution path.
    """
    state = predict_m1(
        canonical_data,
        artifact_directory,
        cutoff_date,
        forecast_horizon,
        execution_mode=execution_mode,
        override_policy=override_policy,
    )
    state = correct_forecast_point(state)
    state = calibrate_forecast(state)
    return build_forecast_package(state)
