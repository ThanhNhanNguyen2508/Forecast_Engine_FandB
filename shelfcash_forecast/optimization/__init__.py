"""Risk-aware procurement action engine."""
# M1 Forecast
#    ↓
# M3 Demand scenario / BOM
#    ↓
# M4 Inventory inputs
#    ↓
# OptimizationRequest
#    ↓
# ========================
#  build_problem_data()
# ========================
#    ↓
# OptimizationProblemData
#    ↓
# ┌───────────────────────┐
# │ Deterministic MILP    │
# │ hoặc                  │
# │ Stochastic MILP       │
# └───────────────────────┘
#    ↓
# ProcurementPlan
#    ↓
# M4 exact re-simulation
#    ↓
# Critic
#    ↓
# Recommend


# M5 :
#             OptimizationRequest
#                     │
#                     ▼
#            model_data.py
#                     │
#          ┌──────────┴──────────┐
#          ▼                     ▼
#  deterministic.py        stochastic.py
#                                │
#                     Regular first stage
#                     + scenario recourse
#                                │
#          └──────────┬──────────┘
#                     ▼
#              ProcurementPlan
#              completed=False
#                     │
#                     ▼
#                adapters.py
#                     │
#                     ▼
#         PlannedInboundDelivery[]
#                     │
#                     ▼
#               resimulation.py
#                     │
#                     ▼
#              M4 EXACT FEFO
#               ┌─────┴─────┐
#               │           │
#           normal sim     stress
#               │           │
#               └─────┬─────┘
#                     ▼
#                 critic.py
#                     │
#           ┌─────────┼─────────┐
#           │         │         │
#     constraints   service    mismatch
#       check       /risk     solver vs M4
#           │         │         │
#           └─────────┼─────────┘
#                     ▼
#               PASS / FAIL
#                     │
#                     ▼
#            CandidateEvaluation
#                     │
#                     ▼
#                optimizer.py
#                     │
#         BALANCED → PROTECTED → LEAN
#                     │
#                     ▼
#             OptimizationResult
#                     │
#                     ▼
#           rolling_horizon.py
#                     │
#             execute current action
#                     │
#                     ▼
#            observe → optimize again
from shelfcash_forecast.optimization.contracts import (
    OptimizationRequest,
    OptimizationResult,
    ProcurementPlan,
    StrategyProfile,
    SupplierOffer,
)
from shelfcash_forecast.optimization.optimizer import optimize_procurement
from shelfcash_forecast.optimization.robust import conformal_robust_status
from shelfcash_forecast.optimization.rolling_horizon import RollingHorizonController

__all__ = [
    "OptimizationRequest",
    "OptimizationResult",
    "ProcurementPlan",
    "RollingHorizonController",
    "StrategyProfile",
    "SupplierOffer",
    "conformal_robust_status",
    "optimize_procurement",
]
