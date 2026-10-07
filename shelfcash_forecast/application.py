from __future__ import annotations

import json
import os
import tempfile
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ForecastPlanConfig(BaseModel):
    """Versioned public configuration for the M1-M6 application service."""

    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["forecast-plan-v1"] = "forecast-plan-v1"
    bundle_directory: Path
    output_directory: Path
    cutoff_date: date
    horizon: int = Field(default=7, ge=1, le=7)
    seed: int = 42
    execution_mode: Literal["production", "backtest_replay", "demo"] = "production"
    train_model: bool = False
    artifact_directory: Path | None = None
    run_downstream: bool = True
    scenario_count: int = Field(default=100, ge=1, le=2000)
    scenario_method: Literal["residual_bootstrap", "gaussian_copula"] = (
        "residual_bootstrap"
    )
    yield_loss_policy: Literal["RECIPE_FIXED"] = "RECIPE_FIXED"
    optimization_mode: Literal["deterministic", "stochastic", "compare"] = "compare"
    planning_config: Path | None = None
    explanation_mode: Literal["deterministic", "llm"] = "deterministic"
    env_config: Path | None = None
    allow_llm_fallback: bool = False

    @model_validator(mode="after")
    def validate_artifact_source(self) -> "ForecastPlanConfig":
        if not self.train_model and self.artifact_directory is None:
            raise ValueError("artifact_directory is required when train_model=false")
        return self


class ForecastPlanResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["forecast-plan-result-v1"] = "forecast-plan-result-v1"
    status: Literal["COMPLETED", "PARTIAL", "BLOCKED", "FAILED"]
    execution_mode: str
    business_ready: bool
    demo_only: bool
    steps: dict[str, dict[str, Any]]
    artifacts: dict[str, str]
    limitations: list[str] = Field(default_factory=list)


def _json_default(value: object) -> object:
    if isinstance(value, (Path, date)):
        return str(value)
    if hasattr(value, "item"):
        return value.item()  # numpy scalar
    raise TypeError(type(value).__name__)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        default=_json_default,
        allow_nan=False,
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        Path(temporary_name).replace(path)
    finally:
        temporary = Path(temporary_name)
        if temporary.exists():
            temporary.unlink()


def _cost_assumptions(offers: list[Any], planning: dict[str, Any]) -> list[Any]:
    from shelfcash_forecast.optimization.planning_config import load_planning_config
    from shelfcash_forecast.optimization.planning_service import cost_assumptions

    return cost_assumptions(offers, load_planning_config(planning)[0])


def run_forecast_plan(config: ForecastPlanConfig) -> ForecastPlanResult:
    """Execute production APIs with truthful stage/mode provenance.

    The service never approves business assumptions and never sends an order.
    In demo mode, planning configuration must be explicitly labelled.
    """

    import pandas as pd

    from shelfcash_forecast import (
        ForecastConfig,
        build_final_decision_package,
        optimize_procurement,
        predict_demand,
        predict_ingredient_demand,
        predict_ingredient_demand_scenarios,
        train_forecast_core,
    )
    from shelfcash_forecast.decision_intelligence.service import explain_decision
    from shelfcash_forecast.inventory.adapters import advanced_inventory_scenarios
    from shelfcash_forecast.inventory.contracts import InventorySimulationPolicy
    from shelfcash_forecast.inventory.monte_carlo import MonteCarloInventoryRunner
    from shelfcash_forecast.optimization.contracts import OptimizationRequest
    from shelfcash_forecast.optimization.strategies import default_strategy_profiles
    from shelfcash_forecast.scenario.composer import generate_product_demand_scenarios
    from shelfcash_forecast.scenario.residuals import load_residual_history
    from shelfcash_preprocess import (
        create_forecast_input_frames,
        create_inventory_lots,
        create_supplier_offers,
        load_bundle,
        load_canonical_frames,
        validate_bundle,
    )

    output = config.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=False)
    steps: dict[str, dict[str, Any]] = {}
    artifacts: dict[str, str] = {}
    limitations: list[str] = []

    bundle = load_bundle(config.bundle_directory)
    validation = validate_bundle(config.bundle_directory)
    if bundle.manifest.context.cutoff_date != config.cutoff_date:
        raise ValueError(
            "BUNDLE_CUTOFF_MISMATCH:"
            f"bundle={bundle.manifest.context.cutoff_date}:requested={config.cutoff_date}"
        )
    steps["bundle"] = {
        "status": "VALID",
        "integrity_valid": validation["integrity_valid"],
        "schema_valid": validation["schema_valid"],
        "runtime_executed": False,
        "run_id": bundle.manifest.run_id,
    }
    frames = load_canonical_frames(config.bundle_directory)
    canonical = create_forecast_input_frames(config.bundle_directory, include_weather=False)
    canonical["recipes"] = frames["recipes"]
    canonical["ingredient_usage_history"] = frames["ingredient_usage"]

    artifact_dir = (
        (output / "artifacts")
        if config.train_model
        else Path(config.artifact_directory).resolve()  # type: ignore[arg-type]
    )
    if config.train_model:
        base = ForecastConfig()
        forecast_config = ForecastConfig(
            random_seed=config.seed,
            lightgbm_params={**base.lightgbm_params, "random_state": config.seed},
        )
        trained = train_forecast_core(
            canonical,
            artifact_dir,
            config=forecast_config,
            model_version=f"forecast-plan-{bundle.manifest.run_id}",
        )
        steps["M1_train"] = {
            "status": "EXECUTED",
            "algorithm": "LightGBM Q25/Q50/Q75",
            "model_version": trained.model_version,
            "actual_artifact": str(artifact_dir),
        }
        steps["M2"] = {
            "status": "EXECUTED",
            "algorithm": "CQR cqr-order-statistic-v2",
            "actual_artifact": str(artifact_dir / "calibrator.json"),
        }
    else:
        steps["M1_train"] = {
            "status": "LOADED",
            "actual_artifact": str(artifact_dir),
        }

    forecast = predict_demand(
        canonical,
        artifact_dir,
        config.cutoff_date.isoformat(),
        config.horizon,
        execution_mode=config.execution_mode,
    )
    forecast_path = output / "forecast_package.json"
    _write_json(forecast_path, forecast.model_dump(mode="json"))
    artifacts["forecast_package"] = str(forecast_path)
    steps["M1_inference"] = {
        "status": "EXECUTED",
        "algorithm": "artifact round-trip LightGBM inference",
        "rows": len(forecast.predictions),
        "execution_mode": config.execution_mode,
    }

    if not config.run_downstream:
        for stage in ("M3", "scenarios", "M4", "M5", "M6"):
            steps[stage] = {"status": "NOT_REQUESTED"}
        result = ForecastPlanResult(
            status="COMPLETED",
            execution_mode=config.execution_mode,
            business_ready=False,
            demo_only=config.execution_mode == "demo",
            steps=steps,
            artifacts=artifacts,
            limitations=["Downstream stages were not requested."],
        )
        _write_json(output / "application_result.json", result.model_dump(mode="json"))
        return result

    ingredient = predict_ingredient_demand(
        canonical,
        artifact_dir,
        config.cutoff_date.isoformat(),
        config.horizon,
        execution_mode=config.execution_mode,
    )
    ingredient_path = output / "ingredient_demand_package.json"
    _write_json(ingredient_path, ingredient.model_dump(mode="json"))
    artifacts["ingredient_demand_package"] = str(ingredient_path)
    steps["M3"] = {
        "status": "EXECUTED" if ingredient.is_complete else "PARTIAL",
        "algorithm": "versioned Recipe/BOM propagation",
        "rows": len(ingredient.predictions),
        "issues": [item.code for item in ingredient.issues],
    }

    residuals = load_residual_history(artifact_dir)
    product_scenarios = generate_product_demand_scenarios(
        forecast,
        residuals,
        n_scenarios=config.scenario_count,
        seed=config.seed,
        method=config.scenario_method,
    )
    from shelfcash_forecast.scenario.bom import propagate_ingredient_demand_scenarios
    from shelfcash_forecast.scenario.yield_loss import FixedRecipeYieldLossModel
    ingredient_scenarios = propagate_ingredient_demand_scenarios(
        product_scenarios, canonical['recipes'], canonical.get('unit_conversions'),
        yield_loss_model=FixedRecipeYieldLossModel(),
    )
    steps["scenarios"] = {
        "status": "EXECUTED",
        "requested_mode": config.scenario_method,
        "actual_mode": ingredient_scenarios.scenario_method,
        "count": len(ingredient_scenarios.scenarios),
        "seed": config.seed,
        "residual_as_of": config.cutoff_date.isoformat(),
        "yield_loss_source": "recipe_fixed",
        "yield_loss_model_fitted": False,
    }

    lots, snapshot = create_inventory_lots(config.bundle_directory)
    if snapshot != config.cutoff_date:
        raise ValueError(
            f"INVENTORY_SNAPSHOT_BOUNDARY_MISMATCH:{snapshot}:{config.cutoff_date}"
        )
    policy_name = bundle.manifest.context.metadata.get("unknown_expiry_policy", "reject")
    policy = InventorySimulationPolicy(unknown_expiry=policy_name)
    inventory_scenarios = advanced_inventory_scenarios(ingredient_scenarios)
    inventory = MonteCarloInventoryRunner().run(
        lots,
        inventory_scenarios,
        policy=policy,
        simulation_start_date=config.cutoff_date + timedelta(days=1),
        simulation_end_date=config.cutoff_date + timedelta(days=config.horizon),
        seed=config.seed,
    )
    inventory_path = output / "inventory_report.json"
    _write_json(inventory_path, inventory.model_dump(mode="json"))
    artifacts["inventory_report"] = str(inventory_path)
    steps["M4"] = {
        "status": "EXECUTED",
        "algorithm": "exact lot-level FEFO Monte Carlo",
        "snapshot_boundary": "EOD",
        "simulation_start": str(config.cutoff_date + timedelta(days=1)),
        "scenario_count": len(inventory_scenarios),
    }

    if config.planning_config is None:
        steps["M5"] = {
            "status": "BLOCKED",
            "reason": "PLANNING_CONFIG_REQUIRED",
        }
        steps["M6"] = {"status": "BLOCKED", "reason": "M5_OUTPUT_REQUIRED"}
        limitations.append("Business consequence-cost configuration is missing.")
        result = ForecastPlanResult(
            status="PARTIAL",
            execution_mode=config.execution_mode,
            business_ready=False,
            demo_only=config.execution_mode == "demo",
            steps=steps,
            artifacts=artifacts,
            limitations=limitations,
        )
        _write_json(output / "application_result.json", result.model_dump(mode="json"))
        return result

    from shelfcash_forecast.optimization.planning_service import run_bundle_planning
    from shelfcash_forecast.optimization.export import export_planning_run

    planning_run = run_bundle_planning(
        config.bundle_directory, planning=config.planning_config, lots=lots,
        snapshot=snapshot, policy=policy, scenarios=inventory_scenarios,
        decision_date=config.cutoff_date,
        planning_end_date=config.cutoff_date + timedelta(days=config.horizon),
        seed=config.seed, optimization_mode=config.optimization_mode,
        execution_mode=config.execution_mode,
    )
    export_planning_run(planning_run, output / "m5")
    planning = planning_run.config.model_dump(mode="json")
    selected_mode = planning_run.selected_mode
    selected_request, selected_result = planning_run.request, planning_run.result
    optimization_payload = {
        mode: {"request": request.model_dump(mode="json"),
               "result": value.model_dump(mode="json")}
        for mode, (request, value) in planning_run.runs.items()
    }
    optimization_path = output / "optimization_result.json"
    _write_json(optimization_path, optimization_payload)
    artifacts["optimization_result"] = str(optimization_path)
    artifacts["accepted_technical_plan"] = str(output / "m5/accepted_technical_plan.json")
    artifacts["accepted_technical_orders"] = str(output / "m5/accepted_technical_orders.csv")
    steps["M5"] = {
        "status": selected_result.technical_outcome,
        "requested_mode": config.optimization_mode,
        "selected_mode": selected_mode,
        "results": {mode: {"status": value.status,
                           "technical_outcome": value.technical_outcome,
                           "recommended_strategy": value.recommended_strategy}
                    for mode, (_, value) in planning_run.runs.items()},
        "optimization_scenario_count": len(selected_request.demand_scenarios),
        "evaluation_scenario_count": len(inventory_scenarios),
        "evaluation_scope": "FULL_M4_POOL",
    }

    decision = build_final_decision_package(
        selected_request,
        selected_result,
        forecast_package=forecast,
        ingredient_demand_package=ingredient,
        ingredient_scenario_bundle=ingredient_scenarios,
        product_scenario_bundle=product_scenarios,
    )
    requested_explanation = config.explanation_mode
    actual_explanation = "deterministic"
    fallback_reason = None
    if requested_explanation == "llm":
        from shelfcash_forecast.decision_intelligence.openai_generator import (
            M6LLMError,
            OpenAIGroundedGenerator,
        )

        try:
            narrative = explain_decision(
                decision,
                "Why should ShelfCash use this procurement result?",
                generator=OpenAIGroundedGenerator(
                    config_path=None
                    if config.env_config is None
                    else str(config.env_config)
                ),
            )
            decision = decision.model_copy(update={"narrative_summary": narrative})
            actual_explanation = "llm"
        except M6LLMError as exc:
            if not config.allow_llm_fallback:
                raise
            fallback_reason = exc.code
    decision_path = output / "decision_package.json"
    _write_json(decision_path, decision.model_dump(mode="json"))
    artifacts["decision_package"] = str(decision_path)
    steps["M6"] = {
        "status": "EXECUTED_WITH_DEMO_ASSUMPTIONS"
        if planning.get("label") == "DEMO_ONLY_NOT_FOR_OPERATION"
        else "EXECUTED",
        "decision_status": decision.decision_status,
        "recommended_strategy": decision.recommended_strategy,
        "order_count": len(decision.immediate_orders),
        "scheduled_order_count": len(decision.scheduled_orders),
        "technical_outcome": selected_result.technical_outcome,
        "requested_explanation_mode": requested_explanation,
        "actual_explanation_mode": actual_explanation,
        "fallback_reason": fallback_reason,
        "read_only": True,
    }

    demo_only = planning.get("label") == "DEMO_ONLY_NOT_FOR_OPERATION"
    if demo_only:
        limitations.append("Planning cost coefficients are demo-only and not approved.")
    result = ForecastPlanResult(
        status="COMPLETED",
        execution_mode=config.execution_mode,
        business_ready=selected_result.business_ready,
        demo_only=demo_only,
        steps=steps,
        artifacts=artifacts,
        limitations=limitations,
    )
    _write_json(output / "application_result.json", result.model_dump(mode="json"))
    return result
