# optimization/contracts.py
#         ↓
# định nghĩa M5 được phép nhận / trả object gì
from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from shelfcash_forecast.bom.contracts import UnitConversionRule
from shelfcash_forecast.inventory.contracts import (
    ConsequenceCostAssumption,
    InboundDelivery,
    InventoryDemandScenario,
    InventoryLot,
    InventorySimulationPackage,
    InventorySimulationPolicy,
)
from shelfcash_forecast.inventory.stress import StressScenarioDefinition

StrategyName = Literal["LEAN", "BALANCED", "PROTECTED"] # risk appetite của procurement policy : 3 mức rủi ro với lean : ít rủi ro nhất, protected : nhiều rủi ro nhất, balanced : trung bình


class StrictOptimizationContract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class SupplierOffer(StrictOptimizationContract): # “Supplier này đang offer cho tôi mua nguyên liệu gì, với điều kiện nào?”
    offer_id: str = Field(min_length=1)
    supplier_id: str = Field(min_length=1)
    store_id: str = Field(min_length=1)
    ingredient_id: str = Field(min_length=1)
    unit: str = Field(min_length=1)
    order_date: date
    pack_size: float = Field(gt=0)
    unit_price: float = Field(ge=0)
    delivery_cost: float = Field(default=0, ge=0)
    minimum_order_quantity: float = Field(default=0, ge=0)
    maximum_order_quantity: float | None = Field(default=None, gt=0)
    lead_time_days: int = Field(ge=0)
    shelf_life_days: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Calendar-day offset from arrival_date to the inclusive expiry date; "
            "zero means usable on arrival_date only."
        ),
    )
    available: bool = True
    order_cutoff_date: date | None = None
    emergency: bool = False
    moq_basis: Literal["pack_count", "base_quantity"] = "base_quantity"
    price_basis: Literal["per_base_unit"] = "per_base_unit"
    currency: str | None = None
    source_terms: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_quantity_bounds(self) -> SupplierOffer:
        if (
            self.maximum_order_quantity is not None
            and self.maximum_order_quantity < self.minimum_order_quantity
        ):
            raise ValueError("maximum_order_quantity cannot be below MOQ.")
        return self


class SupplierConstraint(StrictOptimizationContract): # tổng tiền max mua từ supplier này trong planning horizon, hoặc tổng số lượng max mua từ supplier này trong planning horizon
    supplier_id: str = Field(min_length=1)
    store_id: str | None = None
    ingredient_id: str | None = None
    unit: str | None = None
    maximum_total_quantity: float | None = Field(default=None, ge=0)
    maximum_total_cost: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_quantity_scope(self) -> SupplierConstraint:
        if self.maximum_total_quantity is not None and (
            self.ingredient_id is None or self.unit is None
        ):
            raise ValueError(
                "A supplier quantity cap requires ingredient_id and unit."
            )
        return self


class ProcurementDecisionLine(StrictOptimizationContract):
# Đây là output nhỏ nhất của solver.
# Nếu SupplierOffer là: Có thể mua gì?
# thì ProcurementDecisionLine là: Solver quyết định muagì.
    offer_id: str
    supplier_id: str
    store_id: str
    ingredient_id: str
    unit: str
    order_date: date
    arrival_date: date
    pack_count: int = Field(ge=0)
    pack_size: float = Field(gt=0)
    order_quantity: float = Field(ge=0)
    unit_price: float = Field(ge=0)
    purchase_cost: float = Field(ge=0)
    delivery_cost: float = Field(ge=0)
    shelf_life_days: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Calendar-day offset from arrival_date to the inclusive expiry date; "
            "zero means usable on arrival_date only."
        ),
    )
    emergency: bool = False

    @model_validator(mode="after")
    def validate_derived_values(self) -> ProcurementDecisionLine:
        expected_quantity = self.pack_count * self.pack_size
        if not math.isclose(
            self.order_quantity, expected_quantity, rel_tol=1e-9, abs_tol=1e-9
        ):
            raise ValueError("order_quantity must equal pack_count * pack_size.")
        expected_cost = self.order_quantity * self.unit_price
        if not math.isclose(
            self.purchase_cost, expected_cost, rel_tol=1e-9, abs_tol=1e-9
        ):
            raise ValueError("purchase_cost must equal order_quantity * unit_price.")
        return self


class StrategyProfile(StrictOptimizationContract):
# Đây là chỗ định nghĩa:

# LEAN/BALANCED/PROTECTED thực sự khác nhau như thế nào.

# Có hai nhóm field.
    name: StrategyName
# Nhóm 1 — objective penalties
# shortage_penalty
# holding_penalty
# waste_penalty
# cash_penalty
# cvar_weight

# Ta có thể hình dung objective gần như:

# PurchaseCost+λ
# s
# 	​

# Shortage+λ
# h
# 	​

# Holding+λ
# w
# 	​

# Waste+λ
# c
# 	​

# Cash+λ
# CVaR
# 	​

# CVaR

# Strategy khác nhau chủ yếu ở các lambda này.

# Ví dụ conceptual:

# LEAN
# cash penalty cao
# holding penalty cao
# CVaR thấp

# PROTECTED
# shortage penalty cao
# CVaR cao
# service floor cao
    shortage_penalty: float = Field(ge=0)
    holding_penalty: float = Field(ge=0)
    waste_penalty: float = Field(ge=0)
    cash_penalty: float = Field(ge=0)
    cvar_weight: float = Field(default=0, ge=0)
    cvar_alpha: float = Field(default=0.95, gt=0, lt=1)
    maximum_stockout_probability: float | None = Field(default=None, ge=0, le=1)
    minimum_expected_fill_rate: float | None = Field(default=None, ge=0, le=1)
    minimum_fill_rate: float | None = Field(default=None, ge=0, le=1)
    required_fill_rate_probability: float | None = Field(default=None, ge=0, le=1)
    # Universal exact-simulation safety floors.  These apply even when the
    # candidate MILP does not carry an equivalent service constraint.
    minimum_acceptable_fill_rate: float = Field(default=0.5, ge=0, le=1)
    maximum_acceptable_stockout_probability: float = Field(
        default=0.5, ge=0, le=1
    )
    maximum_fill_rate_model_gap: float = Field(default=0.05, ge=0, le=1)
    maximum_stockout_probability_model_gap: float = Field(
        default=0.05, ge=0, le=1
    )


class ProcurementPlan(StrictOptimizationContract):
# Đây là candidate plan hoàn chỉnh.

# orders

# là first-stage decisions.

# scenario_recourse_orders

# là decision khác nhau theo scenario.

# Ví dụ:

# Regular:
#     Buy A = 50 kg

# Scenario LOW:
#     Emergency = 0

# Scenario MEDIUM:
#     Emergency = 10

# Scenario HIGH:
#     Emergency = 30
    plan_id: str = Field(min_length=1)
    strategy: StrategyName
    orders: list[ProcurementDecisionLine]
    scenario_recourse_orders: dict[str, list[ProcurementDecisionLine]] = Field(
        default_factory=dict
    )
    purchase_cost: float = Field(ge=0)
    expected_recourse_cost: float = Field(default=0, ge=0)
    objective_value: float | None = None
    solver_status: str
    completed: bool = False
    provenance: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class CriticResult(StrictOptimizationContract):
# passed
# hard_violations
# warnings
# checks
# details

# Rất dễ hiểu theo:

# passed
# → verdict

# hard_violations
# → vì sao phải reject

# warnings
# → vấn đề chưa đến mức reject

# checks
# → từng kiểm tra true/false

# details
# → numerical diagnostics
    passed: bool
    hard_violations: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    checks: dict[str, bool] = Field(default_factory=dict)
    details: dict[str, Any] = Field(default_factory=dict)


class CandidateEvaluation(StrictOptimizationContract):
# plan
# simulation
# stress_simulation
# critic

# Đây là nơi M5 ghép:

# solver world
# +
# M4 world
# +
# critic world

# Một ProcurementPlan chỉ là đề xuất.

# Một CandidateEvaluation mới là:

# đề xuất đó sau khi đã được kiểm nghiệm.
    plan: ProcurementPlan
    simulation: InventorySimulationPackage | None = None
    stress_simulation: InventorySimulationPackage | None = None
    critic: CriticResult
    requested_mode: Literal["deterministic", "stochastic"] = "deterministic"
    actual_mode: Literal["deterministic", "stochastic"] = "deterministic"
    fallback_reason: str | None = None


class OptimizationRequest(StrictOptimizationContract):
# Đây là input lớn nhất của M5.

# Nó gom tất cả những gì optimizer cần:

# decision date
# planning horizon

# inventory hiện tại
# demand scenarios
# supplier offers
# existing inbound

# supplier constraints
# budget
# cost assumptions
# unit conversions

# inventory simulation policy
# stress scenarios

# strategy profiles
# stochastic/deterministic flag
# seed
    request_id: str = Field(min_length=1)
    decision_date: date
    planning_end_date: date
    initial_inventory: list[InventoryLot]
    demand_scenarios: list[InventoryDemandScenario]
    supplier_offers: list[SupplierOffer]
    existing_inbound: list[InboundDelivery] = Field(default_factory=list)
    supplier_constraints: list[SupplierConstraint] = Field(default_factory=list)
    budget: float | None = Field(default=None, ge=0)
    cost_assumptions: list[ConsequenceCostAssumption] = Field(default_factory=list)
    unit_conversions: list[UnitConversionRule] = Field(default_factory=list)
    inventory_policy: InventorySimulationPolicy = Field(
        default_factory=InventorySimulationPolicy
    )
    stress_scenarios: list[StressScenarioDefinition] = Field(default_factory=list)
    stress_base_scenario_id: str | None = None
    strategy_profiles: list[StrategyProfile] = Field(default_factory=list)
    unknown_constraints: list[str] = Field(default_factory=list)
    stochastic: bool = True
    allow_mode_fallback: bool = True
    seed: int = 0
    inventory_snapshot_date: date | None = None
    inventory_snapshot_boundary: Literal["EOD", "BOD"] = "EOD"

    @model_validator(mode="after")
    def validate_request(self) -> OptimizationRequest:
        if self.planning_end_date < self.decision_date:
            raise ValueError("planning_end_date cannot precede decision_date.")
        if self.initial_inventory and self.inventory_snapshot_date is None:
            raise ValueError("INVENTORY_SNAPSHOT_DATE_REQUIRED")
        if (
            self.inventory_snapshot_date is not None
            and self.inventory_snapshot_boundary == "EOD"
            and self.inventory_snapshot_date != self.decision_date
        ):
            raise ValueError(
                "INVENTORY_SNAPSHOT_BOUNDARY_MISMATCH: EOD snapshot must equal decision_date"
            )
        if self.inventory_snapshot_boundary == "EOD":
            invalid_zero_lead = [
                offer.offer_id
                for offer in self.supplier_offers
                if offer.order_date == self.decision_date and offer.lead_time_days == 0
            ]
            if invalid_zero_lead:
                raise ValueError(
                    "ZERO_LEAD_UNSUPPORTED_FOR_EOD_SNAPSHOT: same-day arrival "
                    f"cannot be represented after a sealed EOD snapshot: {invalid_zero_lead}"
                )
        scenario_ids = [scenario.scenario_id for scenario in self.demand_scenarios]
        if len(scenario_ids) != len(set(scenario_ids)):
            raise ValueError("Demand scenario identifiers must be unique.")
        for scenario in self.demand_scenarios:
            if any(
                line.target_date < self.decision_date
                or line.target_date > self.planning_end_date
                for line in scenario.lines
            ):
                raise ValueError(
                    "HORIZON_MISMATCH: demand lines must fall inside the planning "
                    "horizon."
                )
            # A decision made from an end-of-day snapshot naturally has its
            # first demand transition on decision_date + 1.  Reject later gaps,
            # but do not force the snapshot date into the consumption window.
            if (
                scenario.simulation_start_date is not None
                and scenario.simulation_start_date != self.decision_date + timedelta(days=1)
            ) or (
                scenario.simulation_end_date is not None
                and scenario.simulation_end_date < self.planning_end_date
            ):
                raise ValueError(
                    "HORIZON_MISMATCH: scenario simulation window does not cover "
                    "the optimization planning horizon."
                )
        weights = [scenario.probability_weight for scenario in self.demand_scenarios]
        if any(weight is not None for weight in weights) and not all(
            weight is not None for weight in weights
        ):
            raise ValueError("Demand probability weights cannot be partially missing.")
        if weights and all(weight is not None for weight in weights):
            total = sum(float(weight) for weight in weights if weight is not None)
            if not math.isclose(total, 1, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError("Probabilistic demand weights must sum to one.")
        return self


class OptimizationResult(StrictOptimizationContract):
# Đây là output cuối cùng của M5:

# evaluations: dict[
#     StrategyName,
#     CandidateEvaluation
# ]

# có nghĩa có thể chứa:

# LEAN evaluation
# BALANCED evaluation
# PROTECTED evaluation

# Sau đó:

# recommended_strategy

# chỉ ra plan được recommend.
    request_id: str
    evaluations: dict[StrategyName, CandidateEvaluation]
    recommended_strategy: StrategyName | None = None
    status: str
    provenance: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class RobustOptimizationStatus(StrictOptimizationContract):
    status: Literal["AVAILABLE", "NOT_AVAILABLE"]
    method: str
    missing_prerequisites: list[str] = Field(default_factory=list)
    guarantee: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class RollingHorizonStep(StrictOptimizationContract):
# Chứa:

# decision date
# optimization result
# orders thực sự execute tại step đó
    decision_date: date
    optimization_result: OptimizationResult
    executed_orders: list[ProcurementDecisionLine]


class RollingHorizonResult(StrictOptimizationContract):
# Chứa toàn bộ:

# Step D1
# Step D2
# Step D3
# ...

# cho rolling-horizon controller.
    steps: list[RollingHorizonStep]
    provenance: dict[str, Any] = Field(default_factory=dict)
