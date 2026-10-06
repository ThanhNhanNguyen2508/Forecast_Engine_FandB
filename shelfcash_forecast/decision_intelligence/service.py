from __future__ import annotations

from shelfcash_forecast.bom.contracts import IngredientDemandPackage
from shelfcash_forecast.contracts import ForecastPackage
from shelfcash_forecast.decision_intelligence.contracts import (
    DecisionAnswer,
    DecisionIntelligenceInput,
    FinalDecisionPackage,
)
from shelfcash_forecast.decision_intelligence.evidence import build_evidence_package
from shelfcash_forecast.decision_intelligence.explainers import (
    build_bom_explanations,
    build_candidate_summaries,
    build_confidence_decomposition,
    build_forecast_explanations,
    build_inventory_explanations,
)
from shelfcash_forecast.decision_intelligence.graph import build_decision_graph
from shelfcash_forecast.decision_intelligence.grounding import (
    DeterministicGroundedGenerator,
    GroundedGenerator,
    GroundingGuard,
)
from shelfcash_forecast.decision_intelligence.retrieval import (
    EvidenceRetriever,
    StructuredLocalRetriever,
    build_retrieval_context,
)
from shelfcash_forecast.optimization.contracts import (
    OptimizationRequest,
    OptimizationResult,
)
from shelfcash_forecast.scenario.contracts import (
    IngredientDemandScenarioBundle,
    ProductDemandScenarioBundle,
)


def _build_from_input(inputs: DecisionIntelligenceInput) -> FinalDecisionPackage:
    evidence = build_evidence_package(inputs)
    graph = build_decision_graph(evidence)
    forecasts = build_forecast_explanations(evidence)
    bom = build_bom_explanations(evidence)
    inventory, risks, traces, stress = build_inventory_explanations(evidence)
    candidates = build_candidate_summaries(evidence)
    confidence = build_confidence_decomposition(inputs, evidence)
    recommended = inputs.optimization_result.recommended_strategy
    recommended_summary = next(
        (candidate for candidate in candidates if candidate.strategy == recommended),
        None,
    )
    immediate = recommended_summary.first_stage_orders if recommended_summary else []
    recourse = recommended_summary.scenario_recourse_orders if recommended_summary else []
    warnings = sorted(
        set(inputs.optimization_result.warnings)
        | {warning for candidate in candidates for warning in candidate.warnings}
        | {warning for item in evidence.items for warning in item.warnings}
    )
    limitations = [
        "M6 is read-only and does not rerun forecast, BOM, inventory, or optimization.",
        "M6 cannot override the M5 recommendation or create fallback orders.",
        "Forecast causal attribution is unavailable because M1/M2 provide no attribution artifact.",
        "What-if, decision regret, human approval, and procurement execution are outside M6 Part 1.",
    ]
    if inputs.forecast_package is None:
        limitations.append(
            "Forecast explanation is partial because ForecastPackage was not supplied."
        )
    if inputs.ingredient_demand_package is None and inputs.ingredient_scenario_bundle is None:
        limitations.append(
            "Recipe/BOM explanation is unavailable because M3 contribution artifacts were not supplied."
        )
    if inputs.coherence is not None:
        limitations.extend(
            issue.message
            for issue in inputs.coherence.issues
            if issue.code
            in {
                "M6_COHERENCE_RECOMMENDATION_RULE_UNAVAILABLE",
                "M6_COHERENCE_STRATEGY_PROFILES_RECONSTRUCTED",
                "M6_COHERENCE_EXACT_M4_MISSING",
            }
        )
    supplied_rule = inputs.optimization_result.provenance.get("recommendation_rule")
    recommendation_rule = (
        supplied_rule.strip() if isinstance(supplied_rule, str) and supplied_rule.strip() else None
    )
    package = FinalDecisionPackage(
        request_id=inputs.optimization_request.request_id,
        decision_date=inputs.optimization_request.decision_date,
        planning_end_date=inputs.optimization_request.planning_end_date,
        decision_status=inputs.optimization_result.status,
        recommended_strategy=recommended,
        recommended_plan_summary=recommended_summary,
        immediate_orders=immediate,
        conditional_recourse=recourse,
        strategy_comparison=candidates,
        forecast_explanations=forecasts,
        bom_explanations=bom,
        inventory_explanations=inventory,
        inventory_risk_explanations=risks,
        inventory_traces=traces,
        stress_explanations=stress,
        confidence_decomposition=confidence,
        evidence_package=evidence,
        decision_graph=graph,
        warnings=warnings,
        limitations=limitations,
        provenance={
            "service": "shelfcash_decision_intelligence_m6_part1_v1",
            "decision_authority": "OptimizationResult_after_exact_M4_and_critic",
            "recommendation_rule": recommendation_rule,
            "recommendation_rule_status": (
                "RECORDED" if recommendation_rule is not None else "UNAVAILABLE"
            ),
            "artifact_coherence_status": (
                inputs.coherence.status if inputs.coherence is not None else "FAILED"
            ),
            "read_only": True,
            "optimizer_called": False,
            "upstream_artifacts": {
                "forecast_package": inputs.forecast_package is not None,
                "ingredient_demand_package": inputs.ingredient_demand_package is not None,
                "ingredient_scenario_bundle": inputs.ingredient_scenario_bundle is not None,
                "product_scenario_bundle": inputs.product_scenario_bundle is not None,
            },
        },
    )
    narrative = explain_decision(package, "Why should ShelfCash buy this?")
    return package.model_copy(update={"narrative_summary": narrative})


def build_final_decision_package(
    optimization_request: OptimizationRequest,
    optimization_result: OptimizationResult,
    *,
    forecast_package: ForecastPackage | None = None,
    ingredient_demand_package: IngredientDemandPackage | None = None,
    ingredient_scenario_bundle: IngredientDemandScenarioBundle | None = None,
    product_scenario_bundle: ProductDemandScenarioBundle | None = None,
) -> FinalDecisionPackage:
    """Build the read-only Decision Intelligence package from existing artifacts."""

    inputs = DecisionIntelligenceInput(
        optimization_request=optimization_request,
        optimization_result=optimization_result,
        forecast_package=forecast_package,
        ingredient_demand_package=ingredient_demand_package,
        ingredient_scenario_bundle=ingredient_scenario_bundle,
        product_scenario_bundle=product_scenario_bundle,
    )
    return _build_from_input(inputs)


def explain_decision(
    decision: FinalDecisionPackage,
    question: str,
    *,
    retriever: EvidenceRetriever | None = None,
    generator: GroundedGenerator | None = None,
    limit: int = 20,
) -> DecisionAnswer:
    """Retrieve current-package evidence, generate, then enforce grounding."""

    selected_retriever = retriever or StructuredLocalRetriever()
    selected_generator = generator or DeterministicGroundedGenerator()
    context = build_retrieval_context(
        question,
        decision.evidence_package,
        recommended_strategy=decision.recommended_strategy,
    )
    retrieved = selected_retriever.retrieve(
        question,
        decision.evidence_package,
        decision.decision_graph,
        context=context or None,
        limit=limit,
    )
    answer = selected_generator.generate(question, retrieved, decision)
    return GroundingGuard().validate(answer, decision, retrieved)
