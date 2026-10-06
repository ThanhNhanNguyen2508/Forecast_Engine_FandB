"""Public APIs for ShelfCash prediction, consequence, and action engines."""

from shelfcash_forecast.bom.contracts import IngredientDemandPackage
from shelfcash_forecast.application import (
    ForecastPlanConfig,
    ForecastPlanResult,
    run_forecast_plan,
)
from shelfcash_forecast.bom.engine import propagate_ingredient_demand
from shelfcash_forecast.config import ForecastConfig
from shelfcash_forecast.contracts import ForecastPackage, TrainingResult
from shelfcash_forecast.decision_intelligence.contracts import (
    DecisionAnswer,
    FinalDecisionPackage,
)
from shelfcash_forecast.decision_intelligence.service import (
    build_final_decision_package,
    explain_decision,
)
from shelfcash_forecast.inventory.contracts import (
    InventoryDemandScenario,
    InventoryLot,
    InventorySimulationPackage,
    PlannedInboundDelivery,
)
from shelfcash_forecast.inventory.simulator import (
    simulate_inventory,
    simulate_inventory_scenarios,
    simulate_quantile_inventory,
)
from shelfcash_forecast.optimization.contracts import (
    OptimizationRequest,
    OptimizationResult,
)
from shelfcash_forecast.optimization.optimizer import optimize_procurement
from shelfcash_forecast.pipeline.inference_pipeline import predict_demand
from shelfcash_forecast.pipeline.ingredient_demand_pipeline import (
    predict_ingredient_demand,
)
from shelfcash_forecast.pipeline.scenario_pipeline import (
    predict_ingredient_demand_scenarios,
)
from shelfcash_forecast.pipeline.training_pipeline import train_forecast_core
from shelfcash_forecast.scenario.bom import propagate_ingredient_demand_scenarios
from shelfcash_forecast.scenario.composer import generate_product_demand_scenarios
from shelfcash_forecast.scenario.contracts import IngredientDemandScenarioBundle

__all__ = [
    "DecisionAnswer",
    "FinalDecisionPackage",
    "ForecastPlanConfig",
    "ForecastPlanResult",
    "ForecastConfig",
    "ForecastPackage",
    "IngredientDemandPackage",
    "IngredientDemandScenarioBundle",
    "InventoryDemandScenario",
    "InventoryLot",
    "InventorySimulationPackage",
    "OptimizationRequest",
    "OptimizationResult",
    "PlannedInboundDelivery",
    "TrainingResult",
    "build_final_decision_package",
    "explain_decision",
    "generate_product_demand_scenarios",
    "optimize_procurement",
    "predict_demand",
    "predict_ingredient_demand",
    "predict_ingredient_demand_scenarios",
    "propagate_ingredient_demand",
    "propagate_ingredient_demand_scenarios",
    "simulate_inventory",
    "simulate_inventory_scenarios",
    "simulate_quantile_inventory",
    "train_forecast_core",
    "run_forecast_plan",
]
