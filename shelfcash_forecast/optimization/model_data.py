# Nếu contracts.py định nghĩa business objects thì model_data.py biến chúng thành:

# solver-ready data representation

# OptimizationRequest
#         ↓
# build_problem_data()
#         ↓
# OptimizationProblemData
#         ↓
# deterministic.py
# hoặc
# stochastic.py
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

from shelfcash_forecast.bom.units import UnitConverter, normalize_unit
from shelfcash_forecast.exceptions import BOMError, OptimizationError
from shelfcash_forecast.inventory.adapters import normalize_cost_assumptions
from shelfcash_forecast.inventory.contracts import ConsequenceCostAssumption
from shelfcash_forecast.optimization.contracts import (
    OptimizationRequest,
    SupplierOffer,
)

InventoryKey = tuple[str, str]


@dataclass(frozen=True)
class EligibleOffer:
#     Đây là SupplierOffer sau preprocessing.

# Ví dụ raw offer:

# unit = gram
# pack_size = 5000 gram
# lead time = 2

# target unit:

# kg

# thì EligibleOffer có:

# factor_to_target = 0.001
# pack_quantity_target = 5 kg
# arrival_date = order_date + 2 days

# Như vậy solver không cần nghĩ về gram nữa.
    offer: SupplierOffer
    arrival_date: date
    factor_to_target: float
    pack_quantity_target: float


@dataclass
class OptimizationProblemData:
# Đây là output của build_problem_data().
# key:

# (scenario_id, inventory_key, day)

    keys: list[InventoryKey] # Geometry của problem
    dates: list[date]# Geometry của problem
    target_units: dict[InventoryKey, str]# Geometry của problem
    scenario_ids: list[str]# Scenario structure
    scenario_weights: np.ndarray# Scenario structure
    probabilistic_weights: bool# Scenario structure
    demand: dict[tuple[str, InventoryKey, date], float] # demand # key: (scenario_id, inventory_key, day)
    initial_quantity: dict[InventoryKey, float]# Supply state
    existing_inbound: dict[tuple[InventoryKey, date], float]# Supply state
    regular_offers: list[EligibleOffer]# Future purchase opportunities
    emergency_offers: list[EligibleOffer]# Future purchase opportunities
    assumptions: dict[InventoryKey, ConsequenceCostAssumption]# Economics
    warnings: list[str] # Audit


def _date_range(start: date, end: date) -> list[date]:
    return [start + timedelta(days=offset) for offset in range((end - start).days + 1)]


def build_problem_data(request: OptimizationRequest) -> OptimizationProblemData:
# Nó là hàm preprocessing/compiler của M5, biến OptimizationRequest dạng business/raw data 
# thành OptimizationProblemData sạch, đồng nhất và sẵn sàng cho MILP solver.
    if not request.demand_scenarios:
        raise OptimizationError("At least one demand scenario is required.")
    try:
        converter = UnitConverter(request.unit_conversions)
    except BOMError as exc:
        raise OptimizationError(
            "Procurement unit-conversion metadata is invalid.",
            code="INVALID_PROCUREMENT_UNIT",
            details=exc.details,
        ) from exc
    target_units: dict[InventoryKey, str] = {}
    demand: dict[tuple[str, InventoryKey, date], float] = defaultdict(float)
    for scenario in request.demand_scenarios:
        for line in scenario.lines:
            if not request.decision_date <= line.target_date <= request.planning_end_date:
                continue
            key = (line.store_id, line.ingredient_id)
            normalized = normalize_unit(line.unit)
            current = target_units.get(key)
            if current is not None and current != normalized:
                raise OptimizationError(
                    "Demand units are inconsistent for one store-ingredient.",
                    details={"key": key, "units": sorted({current, normalized})},
                )
            target_units[key] = normalized
            demand[(scenario.scenario_id, key, line.target_date)] += line.quantity
    if not target_units:
        raise OptimizationError("No demand falls within the planning horizon.")

    warnings: set[str] = set()

    def factor(key: InventoryKey, unit: str) -> float:
        try:
            return converter.conversion_factor(key[1], unit, target_units[key])
        except BOMError as exc:
            raise OptimizationError(
                "Procurement inputs contain incompatible units.",
                code="INVALID_PROCUREMENT_UNIT",
                details={"key": key, **exc.details},
            ) from exc

    initial_quantity: defaultdict[InventoryKey, float] = defaultdict(float)
    for lot in request.initial_inventory:
        key = (lot.store_id, lot.ingredient_id)
        if key not in target_units:
            continue
        if lot.expiry_date is not None and lot.expiry_date < request.decision_date:
            warnings.add("AGGREGATE_MODEL_EXCLUDED_PRESTART_EXPIRED_LOT")
            continue
        if lot.expiry_date is None:
            warnings.add("AGGREGATE_MODEL_COUNTS_UNKNOWN_EXPIRY_LOT")
        elif lot.expiry_date < request.planning_end_date:
            warnings.add("AGGREGATE_MODEL_IGNORES_WITHIN_HORIZON_EXPIRY")
        initial_quantity[key] += lot.quantity_remaining * factor(key, lot.unit)

    existing_inbound: defaultdict[tuple[InventoryKey, date], float] = defaultdict(float)
    for delivery in request.existing_inbound:
        key = (delivery.store_id, delivery.ingredient_id)
        if key in target_units and request.decision_date <= delivery.arrival_date <= request.planning_end_date:
            existing_inbound[(key, delivery.arrival_date)] += delivery.quantity * factor(
                key, delivery.unit
            )

    eligible: list[EligibleOffer] = []
    for offer in request.supplier_offers:
        key = (offer.store_id, offer.ingredient_id)
        arrival = offer.order_date + timedelta(days=offer.lead_time_days)
        if key not in target_units:
            continue
        if (
            not offer.available
            or offer.order_date < request.decision_date
            or offer.order_date > request.planning_end_date
            or arrival > request.planning_end_date
            or (
                offer.order_cutoff_date is not None
                and offer.order_date > offer.order_cutoff_date
            )
        ):
            continue
        conversion = factor(key, offer.unit)
        eligible.append(
            EligibleOffer(
                offer=offer,
                arrival_date=arrival,
                factor_to_target=conversion,
                pack_quantity_target=offer.pack_size * conversion,
            )
        )

    raw_weights = [scenario.probability_weight for scenario in request.demand_scenarios]
    probabilistic = all(weight is not None for weight in raw_weights)
    if probabilistic:
        weights = np.asarray(raw_weights, dtype=float)
        weights /= weights.sum()
    else:
        weights = np.full(len(raw_weights), 1 / len(raw_weights))
        warnings.add("UNWEIGHTED_DESIGN_SCENARIOS_USE_EQUAL_CANDIDATE_WEIGHTS")

    try:
        normalized_assumptions = normalize_cost_assumptions(
            request.cost_assumptions, target_units, converter
        )
    except BOMError as exc:
        raise OptimizationError(
            "Procurement cost assumptions contain incompatible units.",
            code="INVALID_PROCUREMENT_UNIT",
            details=exc.details,
        ) from exc
    assumptions = {
        (item.store_id, item.ingredient_id): item for item in normalized_assumptions
    }
    if len(assumptions) != len(normalized_assumptions):
        raise OptimizationError("Duplicate consequence cost assumptions.")

    return OptimizationProblemData(
        keys=sorted(target_units),
        dates=_date_range(request.decision_date, request.planning_end_date),
        target_units=target_units,
        scenario_ids=[scenario.scenario_id for scenario in request.demand_scenarios],
        scenario_weights=weights,
        probabilistic_weights=probabilistic,
        demand=dict(demand),
        initial_quantity=dict(initial_quantity),
        existing_inbound=dict(existing_inbound),
        regular_offers=[item for item in eligible if not item.offer.emergency],
        emergency_offers=[item for item in eligible if item.offer.emergency],
        assumptions=assumptions,
        warnings=sorted(warnings),
    )


def expected_daily_demand(
# dùng cho deterministic solver.

# Nó tính:

# E[D] = Σ (weight × demand) cho mỗi scenario

# Ví dụ

# Có:

# LOW:
# P = 0.25
# Chicken Aug13 = 40kg

# NORMAL:
# P = 0.50
# Chicken Aug13 = 60kg

# HIGH:
# P = 0.25
# Chicken Aug13 = 100kg

# thì:

# E[D]=0.25×40+0.5×60+0.25×100
# =65kg

# Deterministic solver thấy:

# Chicken demand Aug13 = 65kg
    data: OptimizationProblemData,
) -> dict[tuple[InventoryKey, date], float]:
    output: dict[tuple[InventoryKey, date], float] = {}
    for key in data.keys:
        for day in data.dates:
            output[(key, day)] = sum(
                float(weight) * data.demand.get((scenario_id, key, day), 0)
                for scenario_id, weight in zip(
                    data.scenario_ids, data.scenario_weights, strict=True
                )
            )
    return output
