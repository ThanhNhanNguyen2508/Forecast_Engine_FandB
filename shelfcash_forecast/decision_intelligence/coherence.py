from __future__ import annotations

import math
from collections.abc import Iterable
from datetime import timedelta
from typing import Any

from shelfcash_forecast.decision_intelligence.contracts import (
    ArtifactCoherenceIssue,
    ArtifactCoherenceResult,
    DecisionIntelligenceInput,
)


def _scenario_weights(scenarios: Iterable[Any]) -> dict[str, float | None]:
    return {str(scenario.scenario_id): scenario.probability_weight for scenario in scenarios}


def _weights_match(left: dict[str, float | None], right: dict[str, float | None]) -> bool:
    if set(left) != set(right):
        return False
    for scenario_id, left_weight in left.items():
        right_weight = right[scenario_id]
        if left_weight is None or right_weight is None:
            if left_weight is not right_weight:
                return False
        elif not math.isclose(float(left_weight), float(right_weight), rel_tol=1e-9, abs_tol=1e-9):
            return False
    return True


def _latest_exact_simulation_start(request: Any) -> Any:
    """Allow next-day transitions only for a coherent end-of-day snapshot."""

    line_dates = [
        line.target_date
        for scenario in request.demand_scenarios
        for line in scenario.lines
    ]
    explicit_starts = [
        scenario.simulation_start_date
        for scenario in request.demand_scenarios
        if scenario.simulation_start_date is not None
    ]
    next_day = request.decision_date + timedelta(days=1)
    if (
        line_dates
        and min(line_dates) >= next_day
        and explicit_starts
        and all(start == next_day for start in explicit_starts)
    ):
        return next_day
    return request.decision_date


def evaluate_artifact_coherence(
    inputs: DecisionIntelligenceInput,
) -> ArtifactCoherenceResult:
    """Validate cross-artifact lineage without running any M1-M5 computation."""

    collected: dict[str, ArtifactCoherenceIssue] = {}

    def add(
        code: str,
        severity: str,
        message: str,
        *paths: str,
        blocking: bool = True,
    ) -> None:
        existing = collected.get(code)
        merged_paths = sorted(
            set(paths) | (set(existing.artifact_paths) if existing is not None else set())
        )
        collected[code] = ArtifactCoherenceIssue(
            code=code,
            severity=severity,
            message=message,
            artifact_paths=merged_paths,
            blocking=blocking or bool(existing and existing.blocking),
        )

    request = inputs.optimization_request
    result = inputs.optimization_result
    if request.planning_mode=='SCENARIO_PREVIEW':
        for evaluation in result.evaluations.values():
            if evaluation.plan.solver_status=='OPTIMAL' and evaluation.plan.provenance.get('planning_binding')!=request.planning_binding:
                add('M6_COHERENCE_PROFILE_BINDING_MISMATCH','ERROR','Plan profile/assumptions differ from the request.',
                    'optimization_request.planning_binding','optimization_result.evaluations')
    if request.request_id != result.request_id:
        add(
            "M6_COHERENCE_REQUEST_ID_MISMATCH",
            "ERROR",
            "OptimizationRequest and OptimizationResult request_id values differ.",
            "optimization_request.request_id",
            "optimization_result.request_id",
        )

    for key, evaluation in sorted(result.evaluations.items()):
        if key != evaluation.plan.strategy:
            add(
                "M6_COHERENCE_EVALUATION_STRATEGY_MISMATCH",
                "ERROR",
                "An evaluation dictionary key differs from its plan strategy.",
                f"optimization_result.evaluations.{key}",
            )

    recommended = result.recommended_strategy
    if result.status == "NO_VALID_PROCUREMENT_PLAN" and recommended is not None:
        add(
            "M6_COHERENCE_NO_VALID_PLAN_HAS_RECOMMENDATION",
            "ERROR",
            "A no-valid-plan result cannot carry a recommended strategy.",
            "optimization_result.status",
            "optimization_result.recommended_strategy",
        )
    if recommended is not None:
        evaluation = result.evaluations.get(recommended)
        if evaluation is None:
            add(
                "M6_COHERENCE_RECOMMENDATION_NOT_EVALUATED",
                "ERROR",
                "The recommended strategy is absent from candidate evaluations.",
                "optimization_result.recommended_strategy",
                "optimization_result.evaluations",
            )
        else:
            if not evaluation.critic.passed:
                add(
                    "M6_COHERENCE_RECOMMENDED_CRITIC_FAILED",
                    "ERROR",
                    "The recommended candidate does not have a passing M5 critic.",
                    f"optimization_result.evaluations.{recommended}.critic",
                )
            if not evaluation.plan.completed:
                add(
                    "M6_COHERENCE_RECOMMENDED_PLAN_INCOMPLETE",
                    "ERROR",
                    "The recommended M5 plan is not marked completed.",
                    f"optimization_result.evaluations.{recommended}.plan.completed",
                )
            if evaluation.simulation is None:
                add(
                    "M6_COHERENCE_EXACT_M4_MISSING",
                    "ERROR",
                    "The recommended candidate lacks exact M4 simulation authority.",
                    f"optimization_result.evaluations.{recommended}.simulation",
                    blocking=False,
                )

    recommendation_rule = result.provenance.get("recommendation_rule")
    if not isinstance(recommendation_rule, str) or not recommendation_rule.strip():
        add(
            "M6_COHERENCE_RECOMMENDATION_RULE_UNAVAILABLE",
            "WARNING",
            "M5 did not materialize its recommendation rule in result provenance.",
            "optimization_result.provenance.recommendation_rule",
            blocking=False,
        )

    profile_names = [profile.name for profile in request.strategy_profiles]
    if len(profile_names) != len(set(profile_names)):
        add(
            "M6_COHERENCE_DUPLICATE_STRATEGY_PROFILE",
            "ERROR",
            "OptimizationRequest contains duplicate strategy profile names.",
            "optimization_request.strategy_profiles",
        )
    if set(profile_names) != {"LEAN", "BALANCED", "PROTECTED"}:
        add(
            "M6_COHERENCE_STRATEGY_PROFILES_RECONSTRUCTED",
            "WARNING",
            "One or more strategy profiles are reconstructed from current code defaults.",
            "optimization_request.strategy_profiles",
            blocking=False,
        )

    def validate_window(
        *, forecast_date, horizon: int, date_path: str, horizon_path: str, prefix: str
    ) -> None:
        decision_offset = (request.decision_date - forecast_date).days
        if decision_offset not in {0, 1}:
            add(
                f"M6_COHERENCE_{prefix}_DATE_MISMATCH",
                "ERROR",
                "The optimization decision date must equal the upstream forecast cutoff "
                "or the next actionable day.",
                date_path,
                "optimization_request.decision_date",
            )
        if forecast_date + timedelta(days=horizon) < request.planning_end_date:
            add(
                f"M6_COHERENCE_{prefix}_HORIZON_MISMATCH",
                "ERROR",
                "Upstream artifact horizon does not cover the optimization planning dates.",
                horizon_path,
                "optimization_request.planning_end_date",
            )

    forecast = inputs.forecast_package
    if forecast is not None:
        validate_window(
            forecast_date=forecast.forecast_date,
            horizon=forecast.forecast_horizon,
            date_path="forecast_package.forecast_date",
            horizon_path="forecast_package.forecast_horizon",
            prefix="FORECAST",
        )
        forecast_end = forecast.forecast_date + timedelta(days=forecast.forecast_horizon)
        if any(
            prediction.target_date < request.decision_date
            or prediction.target_date > forecast_end
            or prediction.horizon != (prediction.target_date - forecast.forecast_date).days
            for prediction in forecast.predictions
        ):
            add(
                "M6_COHERENCE_FORECAST_PREDICTION_HORIZON_MISMATCH",
                "ERROR",
                "Forecast prediction dates/horizons disagree with the package window.",
                "forecast_package.predictions",
            )

    ingredient = inputs.ingredient_demand_package
    if ingredient is not None:
        validate_window(
            forecast_date=ingredient.forecast_date,
            horizon=ingredient.forecast_horizon,
            date_path="ingredient_demand_package.forecast_date",
            horizon_path="ingredient_demand_package.forecast_horizon",
            prefix="BOM",
        )
        ingredient_end = ingredient.forecast_date + timedelta(days=ingredient.forecast_horizon)
        if any(
            prediction.target_date < request.decision_date
            or prediction.target_date > ingredient_end
            for prediction in ingredient.predictions
        ):
            add(
                "M6_COHERENCE_BOM_PREDICTION_HORIZON_MISMATCH",
                "ERROR",
                "BOM prediction dates must fall within the actionable planning window.",
                "ingredient_demand_package.predictions",
                "optimization_request",
            )
    if forecast is not None and ingredient is not None:
        if forecast.forecast_date != ingredient.forecast_date:
            add(
                "M6_COHERENCE_FORECAST_BOM_DATE_MISMATCH",
                "ERROR",
                "Forecast and BOM packages have different forecast dates.",
                "forecast_package.forecast_date",
                "ingredient_demand_package.forecast_date",
            )
        if forecast.forecast_horizon != ingredient.forecast_horizon:
            add(
                "M6_COHERENCE_FORECAST_BOM_HORIZON_MISMATCH",
                "ERROR",
                "Forecast and BOM packages have different horizons.",
                "forecast_package.forecast_horizon",
                "ingredient_demand_package.forecast_horizon",
            )
        if forecast.model_version != ingredient.forecast_model_version:
            add(
                "M6_COHERENCE_FORECAST_BOM_MODEL_VERSION_MISMATCH",
                "ERROR",
                "Forecast and BOM packages have different model versions.",
                "forecast_package.model_version",
                "ingredient_demand_package.forecast_model_version",
            )
        forecast_by_key = {
            (prediction.store_id, prediction.product_id, prediction.target_date): prediction
            for prediction in forecast.predictions
        }
        for prediction in ingredient.predictions:
            for source in prediction.sources:
                upstream = forecast_by_key.get(
                    (prediction.store_id, source.product_id, prediction.target_date)
                )
                if upstream is None or not all(
                    math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9)
                    for left, right in (
                        (upstream.p25, source.forecast_p25),
                        (upstream.p50, source.forecast_p50),
                        (upstream.p75, source.forecast_p75),
                    )
                ):
                    add(
                        "M6_COHERENCE_FORECAST_BOM_LINEAGE_MISMATCH",
                        "ERROR",
                        "A BOM contribution does not match its supplied product forecast.",
                        "forecast_package.predictions",
                        "ingredient_demand_package.predictions.sources",
                    )

    product_bundle = inputs.product_scenario_bundle
    if product_bundle is not None:
        validate_window(
            forecast_date=product_bundle.forecast_date,
            horizon=product_bundle.horizon,
            date_path="product_scenario_bundle.forecast_date",
            horizon_path="product_scenario_bundle.horizon",
            prefix="PRODUCT_SCENARIO",
        )
        if forecast is not None and (
            product_bundle.forecast_date != forecast.forecast_date
            or product_bundle.horizon != forecast.forecast_horizon
            or product_bundle.model_version != forecast.model_version
        ):
            add(
                "M6_COHERENCE_FORECAST_PRODUCT_SCENARIO_MISMATCH",
                "ERROR",
                "Product scenarios do not match the supplied forecast package lineage.",
                "forecast_package",
                "product_scenario_bundle",
            )
        if any(
            line.source_model_version != product_bundle.model_version
            or line.scenario_method != product_bundle.scenario_method
            for scenario in product_bundle.scenarios
            for line in scenario.lines
        ):
            add(
                "M6_COHERENCE_PRODUCT_SCENARIO_LINEAGE_MISMATCH",
                "ERROR",
                "Product scenario lines disagree with their bundle lineage metadata.",
                "product_scenario_bundle.scenarios",
            )
        product_end = product_bundle.forecast_date + timedelta(days=product_bundle.horizon)
        if any(
            line.target_date < request.decision_date
            or line.target_date > product_end
            or line.horizon != (line.target_date - product_bundle.forecast_date).days
            for scenario in product_bundle.scenarios
            for line in scenario.lines
        ):
            add(
                "M6_COHERENCE_PRODUCT_SCENARIO_HORIZON_MISMATCH",
                "ERROR",
                "Product scenario lines must fall within the actionable planning window.",
                "product_scenario_bundle.scenarios.lines",
                "optimization_request",
            )

    ingredient_bundle = inputs.ingredient_scenario_bundle
    if ingredient_bundle is not None:
        validate_window(
            forecast_date=ingredient_bundle.forecast_date,
            horizon=ingredient_bundle.horizon,
            date_path="ingredient_scenario_bundle.forecast_date",
            horizon_path="ingredient_scenario_bundle.horizon",
            prefix="INGREDIENT_SCENARIO",
        )
        expected_model = forecast.model_version if forecast is not None else None
        if (
            expected_model is not None
            and ingredient_bundle.forecast_model_version != expected_model
        ):
            add(
                "M6_COHERENCE_FORECAST_INGREDIENT_SCENARIO_MISMATCH",
                "ERROR",
                "Ingredient scenarios do not match the supplied forecast model version.",
                "forecast_package.model_version",
                "ingredient_scenario_bundle.forecast_model_version",
            )
        if ingredient is not None and (
            ingredient_bundle.forecast_date != ingredient.forecast_date
            or ingredient_bundle.horizon != ingredient.forecast_horizon
            or ingredient_bundle.forecast_model_version != ingredient.forecast_model_version
        ):
            add(
                "M6_COHERENCE_BOM_INGREDIENT_SCENARIO_MISMATCH",
                "ERROR",
                "Ingredient scenarios do not match deterministic BOM lineage metadata.",
                "ingredient_demand_package",
                "ingredient_scenario_bundle",
            )
        ingredient_scenario_end = ingredient_bundle.forecast_date + timedelta(
            days=ingredient_bundle.horizon
        )
        if any(
            line.target_date < request.decision_date or line.target_date > ingredient_scenario_end
            for scenario in ingredient_bundle.scenarios
            for line in scenario.lines
        ):
            add(
                "M6_COHERENCE_INGREDIENT_SCENARIO_HORIZON_MISMATCH",
                "ERROR",
                "Ingredient scenario lines must fall within the actionable planning window.",
                "ingredient_scenario_bundle.scenarios.lines",
                "optimization_request",
            )

    evaluation_worlds = request.evaluation_scenarios or request.demand_scenarios
    request_weights = _scenario_weights(evaluation_worlds)
    if product_bundle is not None and not _weights_match(
        _scenario_weights(product_bundle.scenarios), request_weights
    ):
        add(
            "M6_COHERENCE_PRODUCT_REQUEST_SCENARIO_IDENTITY_MISMATCH",
            "ERROR",
            "Product scenario identities/weights differ from OptimizationRequest.",
            "product_scenario_bundle.scenarios",
            "optimization_request.demand_scenarios",
        )
    if ingredient_bundle is not None and not _weights_match(
        _scenario_weights(ingredient_bundle.scenarios), request_weights
    ):
        add(
            "M6_COHERENCE_INGREDIENT_REQUEST_SCENARIO_IDENTITY_MISMATCH",
            "ERROR",
            "Ingredient scenario identities/weights differ from OptimizationRequest.",
            "ingredient_scenario_bundle.scenarios",
            "optimization_request.demand_scenarios",
        )
    if product_bundle is not None and ingredient_bundle is not None:
        if (
            product_bundle.forecast_date != ingredient_bundle.forecast_date
            or product_bundle.horizon != ingredient_bundle.horizon
            or product_bundle.model_version != ingredient_bundle.forecast_model_version
            or product_bundle.scenario_method != ingredient_bundle.scenario_method
            or not _weights_match(
                _scenario_weights(product_bundle.scenarios),
                _scenario_weights(ingredient_bundle.scenarios),
            )
        ):
            add(
                "M6_COHERENCE_PRODUCT_INGREDIENT_SCENARIO_LINEAGE_MISMATCH",
                "ERROR",
                "Product and ingredient scenario bundles have incompatible lineage.",
                "product_scenario_bundle",
                "ingredient_scenario_bundle",
            )
        product_keys = {
            (scenario.scenario_id, line.store_id, line.product_id, line.target_date)
            for scenario in product_bundle.scenarios
            for line in scenario.lines
        }
        if any(
            (
                scenario.scenario_id,
                line.store_id,
                contribution.product_id,
                line.target_date,
            )
            not in product_keys
            for scenario in ingredient_bundle.scenarios
            for line in scenario.lines
            for contribution in line.contributions
        ):
            add(
                "M6_COHERENCE_PRODUCT_INGREDIENT_CONTRIBUTION_MISMATCH",
                "ERROR",
                "An ingredient scenario contribution has no matching product scenario line.",
                "product_scenario_bundle.scenarios",
                "ingredient_scenario_bundle.scenarios.contributions",
            )

    if ingredient_bundle is not None:
        request_lines = {
            (
                scenario.scenario_id,
                line.store_id,
                line.ingredient_id,
                line.target_date,
                line.unit,
            ): line.quantity
            for scenario in evaluation_worlds
            for line in scenario.lines
        }
        if any(
            key not in request_lines
            or not math.isclose(request_lines[key], line.quantity, rel_tol=1e-9, abs_tol=1e-9)
            for scenario in ingredient_bundle.scenarios
            for line in scenario.lines
            for key in [
                (
                    scenario.scenario_id,
                    line.store_id,
                    line.ingredient_id,
                    line.target_date,
                    line.unit,
                )
            ]
        ):
            add(
                "M6_COHERENCE_INGREDIENT_REQUEST_DEMAND_LINEAGE_MISMATCH",
                "ERROR",
                "Ingredient scenarios do not match M4 demand lines in OptimizationRequest.",
                "ingredient_scenario_bundle.scenarios",
                "optimization_request.demand_scenarios",
            )

    request_scenario_ids = set(request_weights)
    latest_simulation_start = _latest_exact_simulation_start(request)
    for strategy, evaluation in sorted(result.evaluations.items()):
        simulation = evaluation.simulation
        if simulation is None:
            continue
        if not _weights_match(_scenario_weights(simulation.results), request_weights):
            add(
                "M6_COHERENCE_EXACT_M4_SCENARIO_IDENTITY_MISMATCH",
                "ERROR",
                "An exact M4 package has scenario identities foreign to the request.",
                f"optimization_result.evaluations.{strategy}.simulation.results",
                "optimization_request.demand_scenarios",
            )
        if evaluation.physics_simulation is not None:
            declared = set(evaluation.plan.provenance.get("physics_scenario_ids", []))
            if {s.scenario_id for s in evaluation.physics_simulation.results} != declared:
                add("M6_COHERENCE_MODEL_PHYSICS_SCOPE_MISMATCH", "ERROR",
                    "Exact model-validation worlds do not match declared solver physics scope.",
                    f"optimization_result.evaluations.{strategy}.physics_simulation")
            if evaluation.plan.provenance.get("mode") == "stochastic" and not declared <= request_scenario_ids:
                add("M6_COHERENCE_FOREIGN_PHYSICS_WORLD", "ERROR", "Model-validation world is foreign to the M4 pool.",
                    f"optimization_result.evaluations.{strategy}.physics_simulation")
            if evaluation.plan.provenance.get("mode") == "stochastic" and declared <= request_scenario_ids:
                mass=sum(float(request_weights[sid] or 0) for sid in declared)
                expected={sid:float(request_weights[sid])/mass for sid in declared} if mass>0 and all(
                    request_weights[sid] is not None for sid in declared) else {}
                if not _weights_match(_scenario_weights(evaluation.physics_simulation.results),expected):
                    add("M6_COHERENCE_MODEL_PHYSICS_WEIGHT_MISMATCH", "ERROR",
                        "Model-validation weights do not match the declared conditional full-pool distribution.",
                        f"optimization_result.evaluations.{strategy}.physics_simulation.results")
            if evaluation.plan.provenance.get("mode") == "deterministic" and not _weights_match(
                _scenario_weights(evaluation.physics_simulation.results),{"__DETERMINISTIC_MEAN__":1.0}):
                add("M6_COHERENCE_MODEL_PHYSICS_WEIGHT_MISMATCH", "ERROR",
                    "Deterministic physics validation requires the single unit-weight mean-demand world.",
                    f"optimization_result.evaluations.{strategy}.physics_simulation.results")
        if (
            simulation.simulation_start_date > latest_simulation_start
            or simulation.simulation_end_date < request.planning_end_date
        ):
            add(
                "M6_COHERENCE_EXACT_M4_HORIZON_MISMATCH",
                "ERROR",
                "An exact M4 package does not cover the optimization planning horizon.",
                f"optimization_result.evaluations.{strategy}.simulation",
                "optimization_request",
            )

    issues = sorted(collected.values(), key=lambda issue: issue.code)
    status = (
        "FAILED"
        if any(issue.severity == "ERROR" for issue in issues)
        else "WARNING"
        if issues
        else "VERIFIED"
    )
    return ArtifactCoherenceResult(status=status, issues=issues)
