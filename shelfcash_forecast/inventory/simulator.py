from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import date, timedelta

import pandas as pd

from shelfcash_forecast.bom.contracts import IngredientDemandPackage
from shelfcash_forecast.exceptions import (
    BOMError,
    InventoryValidationError,
)
from shelfcash_forecast.inventory.accounting import validate_accounting_ledger
from shelfcash_forecast.inventory.adapters import (
    advanced_inventory_scenarios,
    demand_target_units,
    normalize_cost_assumptions,
    normalize_inbound,
    normalize_lots,
    normalize_waste_events,
    quantile_inventory_scenarios,
    unit_converter_from_frame,
)
from shelfcash_forecast.inventory.contracts import (
    ConsequenceCostAssumption,
    DailyInventoryLedger,
    EndingLotState,
    InboundDelivery,
    InventoryDemandScenario,
    InventoryLot,
    InventorySimulationPackage,
    InventorySimulationPolicy,
    InventorySimulationResult,
    LotExpiryTrace,
    LotWasteTrace,
    PlannedInboundDelivery,
    WasteEvent,
)
from shelfcash_forecast.inventory.fefo import consume_fefo, fefo_sort_key, is_expired
from shelfcash_forecast.inventory.metrics import (
    aggregate_risk_metrics,
    summarize_simulation,
)
from shelfcash_forecast.scenario.contracts import IngredientDemandScenarioBundle


def _date_range(start: date, end: date) -> Iterable[date]:
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def _validate_unique(values: Sequence[str], label: str) -> None:
    duplicates = sorted({value for value in values if values.count(value) > 1})
    if duplicates:
        raise InventoryValidationError(
            f"{label} phải duy nhất.",
            details={"duplicates": duplicates},
        )


def _apply_waste_events(
    lots: list[InventoryLot],
    events: list[WasteEvent],
    *,
    scenario_id: str,
    simulation_date: date,
    tolerance: float,
) -> tuple[list[InventoryLot], float, list[LotWasteTrace]]:
    updated = [lot.model_copy(deep=True) for lot in lots]
    traces: list[LotWasteTrace] = []
    total = 0.0
    for event in sorted(events, key=lambda item: item.event_id):
        candidates = [
            index
            for index, lot in enumerate(updated)
            if event.lot_id is None or lot.lot_id == event.lot_id
        ]
        candidates.sort(key=lambda index: fefo_sort_key(updated[index]))
        remaining = event.quantity
        for index in candidates:
            lot = updated[index]
            deducted = min(lot.quantity_remaining, remaining)
            if deducted <= 0:
                continue
            updated[index] = lot.model_copy(
                update={"quantity_remaining": lot.quantity_remaining - deducted}
            )
            remaining -= deducted
            total += deducted
            traces.append(
                LotWasteTrace(
                    scenario_id=scenario_id,
                    simulation_date=simulation_date,
                    store_id=lot.store_id,
                    ingredient_id=lot.ingredient_id,
                    lot_id=lot.lot_id,
                    event_id=event.event_id,
                    quantity=deducted,
                    unit=lot.unit,
                )
            )
        if remaining > tolerance:
            raise InventoryValidationError(
                "Waste event vượt quá ending usable inventory.",
                code="INVALID_WASTE_EVENT",
                details={
                    "event_id": event.event_id,
                    "requested_quantity": event.quantity,
                    "unapplied_quantity": remaining,
                },
            )
    return updated, total, traces


def simulate_inventory( # chạy vật lý 1 lô cho một kịch bản tương lai đã được thực hiện đầy đủ.
    initial_inventory: Sequence[InventoryLot],
    demand_scenario: InventoryDemandScenario,
    inbound: Sequence[InboundDelivery] = (),
    planned_inbound: Sequence[PlannedInboundDelivery] = (),
    waste_events: Sequence[WasteEvent] = (),
    *,
    policy: InventorySimulationPolicy | None = None,
    unit_conversions: pd.DataFrame | None = None,
    cost_assumptions: Sequence[ConsequenceCostAssumption] = (),
    simulation_start_date: date | None = None,
    simulation_end_date: date | None = None,
) -> InventorySimulationResult:
    """Run deterministic lot physics for a fully realized future scenario."""

    policy = policy or InventorySimulationPolicy()
    if not demand_scenario.lines:
        raise InventoryValidationError("Demand scenario không có lines.")
    demand_start = min(line.target_date for line in demand_scenario.lines)
    demand_end = max(line.target_date for line in demand_scenario.lines)
    start_date = (
        simulation_start_date
        or demand_scenario.simulation_start_date
        or demand_start
    )
    end_date = simulation_end_date or demand_scenario.simulation_end_date or demand_end
    if end_date < start_date or demand_start < start_date or demand_end > end_date:
        raise InventoryValidationError(
            "Demand and explicit simulation window are inconsistent.",
            code="HORIZON_MISMATCH",
            details={
                "simulation_start_date": start_date.isoformat(),
                "simulation_end_date": end_date.isoformat(),
                "demand_start_date": demand_start.isoformat(),
                "demand_end_date": demand_end.isoformat(),
            },
        )
    snapshot_date = start_date - timedelta(days=1)
    if any(
        lot.received_date is not None and lot.received_date > snapshot_date
        for lot in initial_inventory
    ):
        raise InventoryValidationError(
            "Initial inventory phải là end-of-day snapshot ngay trước simulation start.",
            details={"expected_snapshot_date": snapshot_date.isoformat()},
        )
    if any(delivery.arrival_date < start_date for delivery in (*inbound, *planned_inbound)):
        raise InventoryValidationError(
            "Inbound trước simulation start phải nằm trong initial inventory snapshot."
        )
    if any(event.event_date < start_date for event in waste_events):
        raise InventoryValidationError(
            "Waste before simulation start must already be reflected in the snapshot.",
            code="INVALID_WASTE_EVENT",
        )
    _validate_unique([lot.lot_id for lot in initial_inventory], "Initial lot_id")
    all_deliveries = [*inbound, *planned_inbound]
    _validate_unique([delivery.delivery_id for delivery in all_deliveries], "delivery_id")
    _validate_unique(
        [lot.lot_id for lot in initial_inventory]
        + [delivery.lot_id for delivery in all_deliveries],
        "lot_id across initial inventory and inbound",
    )
    _validate_unique([event.event_id for event in waste_events], "waste event_id")

    try:
        converter = unit_converter_from_frame(unit_conversions)
        target_units = demand_target_units(demand_scenario)
        lots = normalize_lots(initial_inventory, target_units, converter)
        deliveries = normalize_inbound(all_deliveries, target_units, converter)
        waste = normalize_waste_events(waste_events, target_units, converter)
        assumptions = normalize_cost_assumptions(
            cost_assumptions, target_units, converter
        )
    except BOMError as exc:
        raise InventoryValidationError(
            str(exc),
            code="INVALID_INVENTORY_UNIT",
            details=exc.details,
        ) from exc

    assumption_map: dict[tuple[str, str], ConsequenceCostAssumption] = {}
    for assumption in assumptions:
        key = (assumption.store_id, assumption.ingredient_id)
        if key in assumption_map:
            raise InventoryValidationError(
                "Duplicate consequence cost assumption.", details={"key": key}
            )
        assumption_map[key] = assumption

    demand_by_day: dict[tuple[date, str, str], float] = {}
    for line in demand_scenario.lines:
        demand_by_day[(line.target_date, line.store_id, line.ingredient_id)] = line.quantity
    inbound_by_day: defaultdict[tuple[date, str, str], list] = defaultdict(list)
    for delivery in deliveries:
        inbound_by_day[
            (delivery.arrival_date, delivery.store_id, delivery.ingredient_id)
        ].append(delivery)
    waste_by_day: defaultdict[tuple[date, str, str], list[WasteEvent]] = defaultdict(list)
    for event in waste:
        waste_by_day[(event.event_date, event.store_id, event.ingredient_id)].append(event)

    lots_by_key: defaultdict[tuple[str, str], list[InventoryLot]] = defaultdict(list)
    for lot in lots:
        lots_by_key[(lot.store_id, lot.ingredient_id)].append(lot)
    all_keys = set(lots_by_key) | set(target_units)
    all_keys.update((item.store_id, item.ingredient_id) for item in deliveries)
    all_keys.update((item.store_id, item.ingredient_id) for item in waste)
    key_units: dict[tuple[str, str], str] = dict(target_units)
    for item in [*lots, *deliveries, *waste, *assumptions]:
        key = (item.store_id, item.ingredient_id)
        existing = key_units.get(key)
        if existing is not None and existing != item.unit:
            raise InventoryValidationError(
                "Inventory inputs cannot be canonicalized to one unit.",
                code="INVALID_INVENTORY_UNIT",
                details={"key": key, "units": sorted({existing, item.unit})},
            )
        key_units[key] = item.unit
    for key, key_lots in lots_by_key.items():
        units = {lot.unit for lot in key_lots}
        if len(units) > 1:
            raise InventoryValidationError(
                "Lots không thể canonicalize về cùng unit.",
                code="INVALID_INVENTORY_UNIT",
                details={"key": key, "units": sorted(units)},
            )

    ledgers: list[DailyInventoryLedger] = []
    consumption_traces = []
    waste_traces: list[LotWasteTrace] = []
    expiry_traces: list[LotExpiryTrace] = []
    warnings = set(demand_scenario.warnings)
    if any(lot.received_date is None for lot in initial_inventory):
        warnings.add("UNKNOWN_RECEIVED_DATE_INITIAL_SNAPSHOT")
    if any(delivery.arrival_date > end_date for delivery in all_deliveries):
        warnings.add("INBOUND_AFTER_HORIZON_NOT_REALIZED")
    if any(event.event_date > end_date for event in waste):
        warnings.add("WASTE_EVENT_AFTER_HORIZON_NOT_REALIZED")
    capacity_evaluated = all(
        key in assumption_map
        and assumption_map[key].capacity_quantity is not None
        for key in all_keys
    )
    consequence_cost_evaluated = all(key in assumption_map for key in all_keys)
    if not capacity_evaluated:
        warnings.add("CAPACITY_NOT_EVALUATED")
    if not consequence_cost_evaluated:
        warnings.add("CONSEQUENCE_COST_NOT_EVALUATED")
    stockout_dates: set[date] = set()

    for simulation_date in _date_range(start_date, end_date):
# Mỗi ngày chạy qua chain : 
# 1. Beginning
# 2. Inbound
# 3. Expiry
# 4. Demand
# 5. FEFO
# 6. Waste
# 7. Ending
# 8. Cost / capacity
# 9. Accounting validation
        for store_id, ingredient_id in sorted(all_keys):
            key = (store_id, ingredient_id)
            current_lots = lots_by_key[key]
            unit = key_units.get(key)
            if unit is None:
                raise InventoryValidationError(
                    "Cannot determine an inventory unit for store-ingredient.",
                    code="INVALID_INVENTORY_UNIT",
                    details={"key": key},
                )

            beginning = sum(lot.quantity_remaining for lot in current_lots)
            todays_inbound = inbound_by_day[(simulation_date, store_id, ingredient_id)]
            inbound_quantity = sum(delivery.quantity for delivery in todays_inbound)
            for delivery in todays_inbound:
                arrived_expired = (
                    delivery.expiry_date is not None
                    and delivery.expiry_date < delivery.arrival_date
                )
                if arrived_expired:
                    warnings.add("ARRIVED_EXPIRED")
                current_lots.append(
                    InventoryLot(
                        lot_id=delivery.lot_id,
                        store_id=delivery.store_id,
                        ingredient_id=delivery.ingredient_id,
                        quantity_remaining=delivery.quantity,
                        unit=delivery.unit,
                        received_date=delivery.arrival_date,
                        expiry_date=delivery.expiry_date,
                        unit_cost=delivery.unit_cost,
                        location=delivery.location,
                        supplier_id=delivery.supplier_id,
                        source_type=(
                            "planned_inbound"
                            if isinstance(delivery, PlannedInboundDelivery)
                            else "inbound"
                        ),
                        source_reference_id=delivery.delivery_id,
                        provenance={
                            **delivery.provenance,
                            "arrival_condition": delivery.arrival_condition,
                            "purchase_order_id": delivery.purchase_order_id,
                            "arrival_date": delivery.arrival_date.isoformat(),
                        },
                    )
                )
            maximum_quantity = beginning + inbound_quantity

            usable_lots: list[InventoryLot] = []
            expired_quantity = 0.0
            for lot in current_lots:
                expired = is_expired(lot, simulation_date, policy)
                if lot.expiry_date is None:
                    warnings.add("UNKNOWN_EXPIRY_PLACED_LAST")
                if expired:
                    expired_quantity += lot.quantity_remaining
                    if lot.quantity_remaining > policy.accounting_tolerance:
                        expiry_traces.append(
                            LotExpiryTrace(
                                scenario_id=demand_scenario.scenario_id,
                                simulation_date=simulation_date,
                                store_id=lot.store_id,
                                ingredient_id=lot.ingredient_id,
                                lot_id=lot.lot_id,
                                unit=lot.unit,
                                expiry_date=lot.expiry_date,
                                expired_quantity=lot.quantity_remaining,
                                source_type=lot.source_type,
                                source_reference_id=lot.source_reference_id,
                                supplier_id=lot.supplier_id,
                                received_date=lot.received_date,
                                provenance=lot.provenance,
                            )
                        )
                else:
                    usable_lots.append(lot)

            demand_quantity = demand_by_day.get(
                (simulation_date, store_id, ingredient_id), 0.0
            )
            fefo = consume_fefo(
                usable_lots,
                demand_quantity,
                scenario_id=demand_scenario.scenario_id,
                simulation_date=simulation_date,
                policy=policy,
            )
            warnings.update(fefo.warnings)
            if fefo.shortage_quantity > policy.accounting_tolerance:
                stockout_dates.add(simulation_date)

            after_waste, waste_quantity, todays_waste_traces = _apply_waste_events(
                fefo.updated_lots,
                waste_by_day[(simulation_date, store_id, ingredient_id)],
                scenario_id=demand_scenario.scenario_id,
                simulation_date=simulation_date,
                tolerance=policy.accounting_tolerance,
            )
            lots_by_key[key] = after_waste
            ending = sum(lot.quantity_remaining for lot in after_waste)
            at_risk = sum(
                lot.quantity_remaining
                for lot in after_waste
                if lot.expiry_date is not None
                and 0 <= (lot.expiry_date - simulation_date).days <= policy.at_risk_expiry_days
            )
            assumption = assumption_map.get(key)
            capacity_violation = (
                max(0.0, maximum_quantity - assumption.capacity_quantity)
                if assumption is not None and assumption.capacity_quantity is not None
                else 0.0
            )
            ledger = DailyInventoryLedger(
                scenario_id=demand_scenario.scenario_id,
                simulation_date=simulation_date,
                store_id=store_id,
                ingredient_id=ingredient_id,
                unit=unit,
                beginning_quantity=beginning,
                inbound_quantity=inbound_quantity,
                demand_quantity=demand_quantity,
                fulfilled_quantity=fefo.fulfilled_quantity,
                shortage_quantity=fefo.shortage_quantity,
                expired_quantity=expired_quantity,
                waste_quantity=waste_quantity,
                ending_quantity=ending,
                maximum_quantity=maximum_quantity,
                at_risk_expiry_quantity=at_risk,
                capacity_violation_quantity=capacity_violation,
                holding_cost=(
                    None
                    if assumption is None
                    else ending * assumption.holding_cost_per_unit_day
                ),
                shortage_cost=(
                    None
                    if assumption is None
                    else fefo.shortage_quantity * assumption.shortage_cost_per_unit
                ),
                expiry_cost=(
                    None
                    if assumption is None
                    else expired_quantity * assumption.expired_cost_per_unit
                ),
                waste_cost=(
                    None
                    if assumption is None
                    else waste_quantity * assumption.waste_cost_per_unit
                ),
            )
#             Toàn bộ ngày được đóng lại: DailyInventoryLedger, LotExpiryTrace, LotWasteTrace, EndingLotState

# Beginning
# Inbound
# Demand
# Fulfilled
# Shortage
# Expired
# Waste
# Ending
# Maximum
# At-risk
# Capacity
# Costs

# Ví dụ:

# Beginning = 100
# Inbound = 20
# Demand = 90
# Fulfilled = 90
# Shortage = 0
# Expired = 10
# Waste = 5
# Ending = 15

            validate_accounting_ledger(
                ledger, tolerance=policy.accounting_tolerance
            )
            ledgers.append(ledger)
            consumption_traces.extend(fefo.traces)
            waste_traces.extend(todays_waste_traces)

    ending_lots = [
        EndingLotState(**lot.model_dump())
        for key in sorted(lots_by_key)
        for lot in sorted(lots_by_key[key], key=fefo_sort_key)
        if lot.quantity_remaining > policy.accounting_tolerance
    ]
    if policy.trace_retention == "summary" or (
        policy.trace_retention == "selected"
        and demand_scenario.scenario_id not in policy.trace_scenario_ids
    ):
        consumption_traces = []
        waste_traces = []
        expiry_traces = []
    capacity_evaluated_keys = {
        key
        for key, assumption in assumption_map.items()
        if assumption.capacity_quantity is not None
    }
# 6.18 Sau khi hết horizon

# Simulator có:

# DailyInventoryLedger[]
# ConsumptionTrace[]
# WasteTrace[]
# ExpiryTrace[]
# EndingLotState[]
# Stockout dates

# Nhưng chúng còn quá chi tiết.

# Nên simulator gọi:
    summary = summarize_simulation(
        ledgers,
        ending_lots,
        capacity_evaluated_keys=capacity_evaluated_keys,
    )
    
    return InventorySimulationResult(
        scenario_id=demand_scenario.scenario_id,
        probability_weight=demand_scenario.probability_weight,
        simulation_start_date=start_date,
        simulation_end_date=end_date,
        daily_ledgers=ledgers,
        consumption_traces=consumption_traces,
        waste_traces=waste_traces,
        expiry_traces=expiry_traces,
        ending_lots=ending_lots,
        stockout_dates=sorted(stockout_dates),
        summary=summary,
        accounting_valid=True,
        provenance={
            **demand_scenario.provenance,
            "initial_inventory_snapshot_date": snapshot_date.isoformat(),
            "expiry_inclusive": policy.expiry_inclusive,
            "unknown_expiry_policy": policy.unknown_expiry,
            "planned_inbound_is_supplied_action": bool(planned_inbound),
            "capacity_evaluated": capacity_evaluated,
            "consequence_cost_evaluated": consequence_cost_evaluated,
            "simulation_window_source": (
                "explicit"
                if simulation_start_date is not None
                or simulation_end_date is not None
                or demand_scenario.simulation_start_date is not None
                or demand_scenario.simulation_end_date is not None
                else "demand_inferred_backward_compatibility"
            ),
        },
        warnings=sorted(warnings),
    )

def simulate_inventory_scenarios( # apply simulate_inventory cho nhiều kịch bản demand scenario
    initial_inventory: Sequence[InventoryLot],
    demand_scenarios: Sequence[InventoryDemandScenario]
    | IngredientDemandScenarioBundle,
    inbound: Sequence[InboundDelivery] = (),
    planned_inbound: Sequence[PlannedInboundDelivery] = (),
    waste_events: Sequence[WasteEvent] = (),
    *,
    policy: InventorySimulationPolicy | None = None,
    unit_conversions: pd.DataFrame | None = None,
    cost_assumptions: Sequence[ConsequenceCostAssumption] = (),
    allow_incomplete: bool = False,
    simulation_start_date: date | None = None,
    simulation_end_date: date | None = None,
) -> InventorySimulationPackage:
    policy = policy or InventorySimulationPolicy()
    scenarios = (
        advanced_inventory_scenarios(
            demand_scenarios, allow_incomplete=allow_incomplete
        )
        if isinstance(demand_scenarios, IngredientDemandScenarioBundle)
        else list(demand_scenarios)
    )
    if not scenarios:
        raise InventoryValidationError("Không có demand scenarios để simulate.")
    identifiers = [scenario.scenario_id for scenario in scenarios]
    if len(identifiers) != len(set(identifiers)):
        raise InventoryValidationError(
            "Demand scenario identifiers must be unique.",
            code="SCENARIO_DUPLICATE_KEY",
        )
    weights = [scenario.probability_weight for scenario in scenarios]
    if any(weight is not None for weight in weights) and not all(
        weight is not None for weight in weights
    ):
        raise InventoryValidationError(
            "Scenario probability weights cannot be partially missing."
        )
    if weights and all(weight is not None for weight in weights) and not math.isclose(
        sum(float(weight) for weight in weights if weight is not None),
        1.0,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise InventoryValidationError(
            "Probabilistic scenario weights must sum to one.",
            code="INVALID_SCENARIO_WEIGHTS",
        )
    results = [
        simulate_inventory(
            initial_inventory,
            scenario,
            inbound,
            planned_inbound,
            waste_events,
            policy=policy,
            unit_conversions=unit_conversions,
            cost_assumptions=cost_assumptions,
            simulation_start_date=simulation_start_date,
            simulation_end_date=simulation_end_date,
        )
        for scenario in scenarios
    ]
    has_probability_weights = all(
        result.probability_weight is not None for result in results
    )
    return InventorySimulationPackage(
        simulation_start_date=min(result.simulation_start_date for result in results),
        simulation_end_date=max(result.simulation_end_date for result in results),
        results=results,
        risk_metrics=(
#             8.5 Lúc này mới check probabilistic hay deterministic
# all results có probability_weight?
# Nếu: YES → gọi STEP 9: aggregate_risk_metrics(...)

# Nếu: NO → risk_metrics = None
            aggregate_risk_metrics(
                results,
                waste_threshold=policy.waste_threshold,
                fill_rate_target=policy.fill_rate_target,
            )
            if has_probability_weights
            else None
        ),
        provenance={
            "simulator": "lot_level_fefo_v1",
            "probabilistic": has_probability_weights,
        },
        warnings=sorted({warning for result in results for warning in result.warnings}),
    )


def simulate_quantile_inventory(
    initial_inventory: Sequence[InventoryLot],
    ingredient_demand: IngredientDemandPackage,
    inbound: Sequence[InboundDelivery] = (),
    planned_inbound: Sequence[PlannedInboundDelivery] = (),
    waste_events: Sequence[WasteEvent] = (),
    *,
    policy: InventorySimulationPolicy | None = None,
    unit_conversions: pd.DataFrame | None = None,
    cost_assumptions: Sequence[ConsequenceCostAssumption] = (),
    allow_incomplete: bool = False,
    simulation_start_date: date | None = None,
    simulation_end_date: date | None = None,
) -> InventorySimulationPackage:
    scenarios = quantile_inventory_scenarios(
        ingredient_demand, allow_incomplete=allow_incomplete
    )
    package = simulate_inventory_scenarios(
        initial_inventory,
        scenarios,
        inbound,
        planned_inbound,
        waste_events,
        policy=policy,
        unit_conversions=unit_conversions,
        cost_assumptions=cost_assumptions,
        simulation_start_date=simulation_start_date,
        simulation_end_date=simulation_end_date,
    )
    return package.model_copy(
        update={"baseline_scenarios": [scenario.scenario_id for scenario in scenarios]}
    )
