# Nếu adapters.py trả lời:

# “Tất cả demand, lot, inbound, waste đã cùng unit chưa?”

# thì hai file này trả lời tiếp:

# fefo.py: “Khi có demand, lấy hàng từ lot nào trước?”
# accounting.py: “Sau khi lấy hàng, số tồn kho có còn đúng theo định luật bảo toàn hay không?”

# 30. Ví dụ end-to-end FEFO

# Giả sử ngày 12/08 có:

# LOT A
# 30 kg
# expiry 14/08

# LOT B
# 50 kg
# expiry 20/08

# LOT C
# 40 kg
# expiry 18/08

# Demand:

# 90 kg

# Sort FEFO:

# LOT A expiry 14
# LOT C expiry 18
# LOT B expiry 20

# Consume:

# LOT A:
# 30 / 30 consumed
# remaining demand = 60

# LOT C:
# 40 / 40 consumed
# remaining demand = 20

# LOT B:
# 20 / 50 consumed
# remaining demand = 0

# Output:

# fulfilled = 90
# shortage = 0

# Updated lots:

# LOT A = 0
# LOT C = 0
# LOT B = 30

# Traces:

# Trace 1 → LOT A 30kg
# Trace 2 → LOT C 40kg
# Trace 3 → LOT B 20kg

# Đây chính là FEFO physics.

from __future__ import annotations

from datetime import date

from shelfcash_forecast.exceptions import UnknownExpiryError
from shelfcash_forecast.inventory.contracts import (
    FEFOResult,
    InventoryLot,
    InventorySimulationPolicy,
    LotConsumptionTrace,
)


def is_expired(
    lot: InventoryLot,
    simulation_date: date,
    policy: InventorySimulationPolicy,
) -> bool:
    if lot.expiry_date is None:
        if policy.unknown_expiry == "reject":
            raise UnknownExpiryError(
                f"Lot {lot.lot_id} thiếu expiry_date.",
                details={"lot_id": lot.lot_id},
            )
        return False
    if policy.expiry_inclusive:
        return lot.expiry_date < simulation_date
    return lot.expiry_date <= simulation_date


def fefo_sort_key(lot: InventoryLot) -> tuple[date, int, date, str]:
    """Return a deterministic FEFO key without inventing receipt dates.

    Expiry remains the primary ordering.  For equal expiry dates, known receipt
    dates sort chronologically before unknown ones; lot_id is the stable final
    tie-break.  ``date.max`` is only an internal sort sentinel and is never
    persisted as the physical received_date.
    """

    received_unknown = int(lot.received_date is None)
    return (
        lot.expiry_date or date.max,
        received_unknown,
        lot.received_date or date.max,
        lot.lot_id,
    )


def consume_fefo(
# Nó trả lời:

# Với danh sách lots hiện tại và demand X, FEFO sẽ consume như thế nào?

# Input:

# lots
# demand quantity
# scenario
# date
# policy

# Output:

# FEFOResult

# bao gồm:

# updated_lots
# fulfilled_quantity
# shortage_quantity
# traces
# warnings
    lots: list[InventoryLot],
    demand_quantity: float,
    *,
    scenario_id: str,
    simulation_date: date,
    policy: InventorySimulationPolicy,
) -> FEFOResult:
    """Consume usable lots deterministically without changing the input list."""

    remaining_demand = float(demand_quantity)
    updated: list[InventoryLot] = []
    traces: list[LotConsumptionTrace] = []
    warnings: set[str] = set()
    for lot in sorted(lots, key=fefo_sort_key):
        if lot.expiry_date is None:
            if policy.unknown_expiry == "reject":
                raise UnknownExpiryError(
                    f"Lot {lot.lot_id} thiếu expiry_date.",
                    details={"lot_id": lot.lot_id},
                )
            warnings.add("UNKNOWN_EXPIRY_PLACED_LAST")
        if is_expired(lot, simulation_date, policy):
            updated.append(lot.model_copy(deep=True))
            continue
        consumed = min(lot.quantity_remaining, remaining_demand)
        new_quantity = lot.quantity_remaining - consumed
        if consumed > 0:
            traces.append(
                LotConsumptionTrace(
                    scenario_id=scenario_id,
                    simulation_date=simulation_date,
                    store_id=lot.store_id,
                    ingredient_id=lot.ingredient_id,
                    lot_id=lot.lot_id,
                    quantity=consumed,
                    unit=lot.unit,
                    lot_expiry_date=lot.expiry_date,
                )
            )
            remaining_demand -= consumed
        updated.append(lot.model_copy(update={"quantity_remaining": new_quantity}))

    fulfilled = demand_quantity - remaining_demand
    return FEFOResult(
        updated_lots=updated,
        fulfilled_quantity=fulfilled,
        shortage_quantity=remaining_demand,
        traces=traces,
        warnings=sorted(warnings),
    )
