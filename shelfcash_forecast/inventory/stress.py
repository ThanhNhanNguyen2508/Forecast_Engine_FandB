# Nếu cố tình tạo ra một tình huống bất lợi cụ thể thì inventory chịu được đến đâu?
from __future__ import annotations

from collections.abc import Sequence
from datetime import date, timedelta

import pandas as pd
from pydantic import Field

from shelfcash_forecast.inventory.contracts import (
    ConsequenceCostAssumption,
    InboundDelivery,
    InventoryDemandScenario,
    InventoryLot,
    InventorySimulationPackage,
    InventorySimulationPolicy,
    PlannedInboundDelivery,
    StrictInventoryContract,
)
from shelfcash_forecast.inventory.simulator import simulate_inventory


class StressScenarioDefinition(StrictInventoryContract):
    stress_id: str = Field(min_length=1)
    demand_multiplier: float = Field(default=1, ge=0)
    supplier_delay_days: int = Field(default=0, ge=0)
    supplier_ids: set[str] = Field(default_factory=set)
    preserve_remaining_shelf_life: bool = False
    description: str | None = None


def apply_stress(
    demand: InventoryDemandScenario,
    inbound: Sequence[InboundDelivery],
    planned_inbound: Sequence[PlannedInboundDelivery],
    definition: StressScenarioDefinition,
) -> tuple[InventoryDemandScenario, list[InboundDelivery], list[PlannedInboundDelivery]]:
    stressed_demand = demand.model_copy(
        update={
            "scenario_id": definition.stress_id,
            "probability_weight": None,
            "lines": [
                line.model_copy(
                    update={
                        "scenario_id": definition.stress_id,
                        "quantity": line.quantity * definition.demand_multiplier,
                    }
                )
                for line in demand.lines
            ],
            "provenance": {
                **demand.provenance,
                "stress_source_scenario": demand.scenario_id,
                "stress_definition": definition.model_dump(mode="json"),
            },
        }
    )

    def delay(delivery: InboundDelivery) -> InboundDelivery:
        selected = (
            not definition.supplier_ids
            or delivery.supplier_id in definition.supplier_ids
        )
        offset = definition.supplier_delay_days if selected else 0
        arrival_date = delivery.arrival_date + timedelta(days=offset)
        expiry_date = (
            delivery.expiry_date + timedelta(days=offset)
            if delivery.expiry_date is not None
            and definition.preserve_remaining_shelf_life
            else delivery.expiry_date
        )
        arrived_expired = expiry_date is not None and expiry_date < arrival_date
        values = delivery.model_dump()
        values.update(
            {
                "arrival_date": arrival_date,
                "expiry_date": expiry_date,
                "arrival_condition": (
                    "arrived_expired_realization" if arrived_expired else "normal"
                ),
                "provenance": {
                    **delivery.provenance,
                    "realization_type": "explicit_stress",
                    "stress_id": definition.stress_id,
                    "original_arrival_date": delivery.arrival_date.isoformat(),
                },
            }
        )
        return type(delivery).model_validate(values)

    return stressed_demand, [delay(item) for item in inbound], [
        delay(item) for item in planned_inbound
    ]


def run_inventory_stress_tests(
    initial_inventory: Sequence[InventoryLot],
    baseline_demand: InventoryDemandScenario,
    definitions: Sequence[StressScenarioDefinition],
    inbound: Sequence[InboundDelivery] = (),
    planned_inbound: Sequence[PlannedInboundDelivery] = (),
    *,
    policy: InventorySimulationPolicy | None = None,
    unit_conversions: pd.DataFrame | None = None,
    cost_assumptions: Sequence[ConsequenceCostAssumption] = (),
    simulation_start_date: date | None = None,
    simulation_end_date: date | None = None,
) -> InventorySimulationPackage:
    if not definitions:
        raise ValueError("At least one stress definition is required.")
    results = []
    for definition in definitions:
        demand, stressed_inbound, stressed_planned = apply_stress(
            baseline_demand, inbound, planned_inbound, definition
        )
        results.append(
            simulate_inventory(
                initial_inventory,
                demand,
                stressed_inbound,
                stressed_planned,
                policy=policy,
                unit_conversions=unit_conversions,
                cost_assumptions=cost_assumptions,
                simulation_start_date=simulation_start_date,
                simulation_end_date=simulation_end_date,
            )
        )
    return InventorySimulationPackage(
        simulation_start_date=min(item.simulation_start_date for item in results),
        simulation_end_date=max(item.simulation_end_date for item in results),
        results=results,
        risk_metrics=None,
        provenance={"runner": "explicit_stress_v1", "probabilistic": False},
        warnings=sorted({warning for result in results for warning in result.warnings}),
    )
