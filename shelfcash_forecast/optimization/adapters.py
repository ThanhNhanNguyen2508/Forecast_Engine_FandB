
# stochastic.py
#     ↓
# ProcurementPlan
# completed=False
#     ↓
# adapters.py
#     ↓
# PlannedInboundDelivery
#     ↓
# resimulation.py
#     ↓
# M4 exact FEFO simulation
#     ↓
# critic.py
#     ├── gọi constraints.py
#     ├── kiểm service/risk
#     └── kiểm model mismatch
#     ↓
# CandidateEvaluation
#     ↓
# optimizer.py
#     ↓
# chọn BALANCED / PROTECTED / LEAN

# M5 nói:
# ProcurementDecisionLine

#         ↓ adapter

# M4 hiểu:
# PlannedInboundDelivery

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, timedelta

import numpy as np

from shelfcash_forecast.inventory.contracts import PlannedInboundDelivery
from shelfcash_forecast.optimization.contracts import ProcurementDecisionLine
from shelfcash_forecast.scenario.lead_time import LeadTimeModel
from shelfcash_forecast.scenario.shelf_life import ShelfLifeModel


def _inclusive_expiry_date(
    arrival_date: date,
    shelf_life_days: int | None,
) -> date | None:
    """Convert an offset to an inclusive expiry boundary.

    A value of zero means the lot is usable on arrival_date and expires before
    the next day. A value of N means expiry_date == arrival_date + N and M4's
    default policy permits consumption through that expiry date.
    """

    return (
        None
        if shelf_life_days is None
        else arrival_date + timedelta(days=shelf_life_days)
    )


def decisions_to_planned_inbound(
    decisions: Sequence[ProcurementDecisionLine],
    *,
    plan_id: str,
    scenario_id: str | None = None,
) -> list[PlannedInboundDelivery]:
    deliveries: list[PlannedInboundDelivery] = []
    suffix = f"-{scenario_id}" if scenario_id else ""
    for index, line in enumerate(decisions):
        expiry = _inclusive_expiry_date(line.arrival_date, line.shelf_life_days)
        deliveries.append(
            PlannedInboundDelivery(
                delivery_id=f"{plan_id}{suffix}-delivery-{index:04d}",
                lot_id=f"{plan_id}{suffix}-lot-{index:04d}",
                purchase_order_id=f"{plan_id}{suffix}-po-{index:04d}",
                source_plan_id=plan_id,
                supplier_id=line.supplier_id,
                store_id=line.store_id,
                ingredient_id=line.ingredient_id,
                quantity=line.order_quantity,
                unit=line.unit,
                arrival_date=line.arrival_date,
                expiry_date=expiry,
                unit_cost=line.unit_price,
                provenance={"realization_type": "optimizer_fixed_lead_time"},
            )
        )
    return deliveries


def decisions_to_scenario_planned_inbound(
    decisions: Sequence[ProcurementDecisionLine],
    scenario_ids: Sequence[str],
    *,
    plan_id: str,
    lead_time_model: LeadTimeModel,
    shelf_life_model: ShelfLifeModel,
    seed: int,
) -> dict[str, list[PlannedInboundDelivery]]:
    """Realize external supply uncertainty before invoking inventory physics."""

    rng = np.random.default_rng(seed)
    output: dict[str, list[PlannedInboundDelivery]] = {}
    for scenario_id in scenario_ids:
        deliveries: list[PlannedInboundDelivery] = []
        for index, line in enumerate(decisions):
            lead_time = lead_time_model.realize(
                order_date=line.order_date,
                supplier_id=line.supplier_id,
                ingredient_id=line.ingredient_id,
                rng=rng,
            )
            official_expiry = _inclusive_expiry_date(
                lead_time.arrival_date, line.shelf_life_days
            )
            shelf_life = shelf_life_model.realize(
                official_expiry_date=official_expiry,
                rng=rng,
            )
            arrived_expired = (
                shelf_life.effective_expiry_date is not None
                and shelf_life.effective_expiry_date < lead_time.arrival_date
            )
            deliveries.append(
                PlannedInboundDelivery(
                    delivery_id=f"{plan_id}-{scenario_id}-base-delivery-{index:04d}",
                    lot_id=f"{plan_id}-{scenario_id}-base-lot-{index:04d}",
                    purchase_order_id=f"{plan_id}-{scenario_id}-base-po-{index:04d}",
                    source_plan_id=plan_id,
                    supplier_id=line.supplier_id,
                    store_id=line.store_id,
                    ingredient_id=line.ingredient_id,
                    quantity=line.order_quantity,
                    unit=line.unit,
                    arrival_date=lead_time.arrival_date,
                    expiry_date=shelf_life.effective_expiry_date,
                    unit_cost=line.unit_price,
                    arrival_condition=(
                        "arrived_expired_realization" if arrived_expired else "normal"
                    ),
                    provenance={
                        "realization_type": "external_supply_scenario",
                        "lead_time_source": lead_time.source,
                        "shelf_life_source": shelf_life.source,
                    },
                )
            )
        output[scenario_id] = deliveries
    return output
