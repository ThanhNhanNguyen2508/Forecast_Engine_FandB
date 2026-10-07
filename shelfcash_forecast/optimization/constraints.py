# kiểm lại xem solver có làm đúng luật mua hàng không

# Ví dụ stochastic nói
# HIGH:
# Emergency Supplier B = 40kg

# constraints.py sẽ quay về original SupplierOffer của B và hỏi:

# Offer B có tồn tại thật không?

# Supplier ID đúng không?

# Ingredient đúng không?

# Đây có thật sự là emergency offer không?

# order_date đúng không?

# lead time đúng không?

# arrival_date đúng không?

# pack size đúng không?

# unit price đúng không?

# delivery cost đúng không?

# shelf life đúng không?

# Tức nó kiểm:

# Solver output có bị “biến dạng” khỏi offer business ban đầu không?

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import timedelta

from shelfcash_forecast.bom.units import normalize_unit, UnitConverter
from shelfcash_forecast.optimization.chronology import offer_arrival
from shelfcash_forecast.optimization.shipment_fees import allocate_delivery_fees
from shelfcash_forecast.optimization.contracts import (
    ProcurementPlan,
    SupplierConstraint,
    SupplierOffer,
)


def validate_plan_constraints(
    plan: ProcurementPlan,
    offers: Sequence[SupplierOffer],
    supplier_constraints: Sequence[SupplierConstraint] = (),
    *,
    budget: float | None = None,
    tolerance: float = 1e-8,
    unit_conversions: Sequence = (),
) -> tuple[list[str], dict[str, bool]]:
    violations: list[str] = []
    offer_map = {offer.offer_id: offer for offer in offers}
    converter = UnitConverter(unit_conversions)
    all_lines = [*plan.orders]
    for lines in plan.scenario_recourse_orders.values():
        all_lines.extend(lines)
    if len({l.offer_id for l in all_lines}) != len(all_lines):
        violations.append('DUPLICATE_DECISION_OPPORTUNITY')
    expected_fees=allocate_delivery_fees([l for l in all_lines if l.offer_id in offer_map],offer_map)
    for line in all_lines:
        offer = offer_map.get(line.offer_id)
        if offer is None:
            violations.append(f"UNKNOWN_OFFER:{line.offer_id}")
            continue
        if (
            line.supplier_id != offer.supplier_id
            or line.store_id != offer.store_id
            or line.ingredient_id != offer.ingredient_id
            or line.emergency != offer.emergency
        ):
            violations.append(f"OFFER_IDENTITY:{line.offer_id}")
        if line.order_date != offer.order_date:
            violations.append(f"ORDER_DATE:{line.offer_id}")
        if line.arrival_date != offer_arrival(offer):
            violations.append(f"LEAD_TIME:{line.offer_id}")
        if not math.isclose(line.pack_size, offer.pack_size):
            violations.append(f"PACK_SIZE:{line.offer_id}")
        if not math.isclose(line.unit_price, offer.unit_price):
            violations.append(f"UNIT_PRICE:{line.offer_id}")
        if not math.isclose(line.delivery_cost, expected_fees[line.offer_id],abs_tol=tolerance):
            violations.append(f"DELIVERY_COST:{line.offer_id}")
        if line.shelf_life_days != offer.shelf_life_days:
            violations.append(f"SHELF_LIFE:{line.offer_id}")
        if line.pack_count != int(line.pack_count):
            violations.append(f"PACK_SIZE_INTEGRALITY:{line.offer_id}")
        if not math.isclose(
            line.order_quantity,
            line.pack_count * offer.pack_size,
            rel_tol=1e-9,
            abs_tol=tolerance,
        ):
            violations.append(f"PACK_SIZE:{line.offer_id}")
        if line.order_quantity + tolerance < offer.minimum_order_quantity:
            violations.append(f"MOQ:{line.offer_id}")
        if (
            offer.maximum_order_quantity is not None
            and line.order_quantity > offer.maximum_order_quantity + tolerance
        ):
            violations.append(f"SUPPLIER_OFFER_MAX:{line.offer_id}")
        if not offer.available:
            violations.append(f"SUPPLIER_UNAVAILABLE:{line.offer_id}")
        if offer.order_cutoff_date is not None and line.order_date > offer.order_cutoff_date:
            violations.append(f"ORDER_CUTOFF:{line.offer_id}")
        if normalize_unit(line.unit) != normalize_unit(offer.unit):
            violations.append(f"UNIT_COMPATIBILITY:{line.offer_id}")
    first_stage_cost = sum(
        line.purchase_cost + line.delivery_cost for line in plan.orders
    )
    if not math.isclose(plan.purchase_cost,first_stage_cost,abs_tol=tolerance):
        violations.append('COMMITTED_COST_TOTAL_MISMATCH')
    if budget is not None and first_stage_cost > budget + tolerance:
        violations.append("BUDGET")
    line_groups = [("FIRST_STAGE", plan.orders)] + [
        (scenario_id, [*plan.orders, *lines])
        for scenario_id, lines in plan.scenario_recourse_orders.items()
    ]
    for constraint in supplier_constraints:
        for group_id, lines in line_groups:
            scoped = [
                line
                for line in lines
                if line.supplier_id == constraint.supplier_id
                and (
                    constraint.ingredient_id is None
                    or line.ingredient_id == constraint.ingredient_id
                )
                and (constraint.store_id is None or line.store_id == constraint.store_id)
            ]
            if (
                constraint.maximum_total_quantity is not None
                and sum(
                    line.order_quantity * converter.conversion_factor(line.ingredient_id,line.unit,constraint.unit or "")
                    for line in scoped
                )
                > constraint.maximum_total_quantity + tolerance
            ):
                violations.append(
                    f"SUPPLIER_MAX_QUANTITY:{constraint.supplier_id}:{group_id}"
                )
            if (
                constraint.maximum_total_cost is not None
                and sum(line.purchase_cost + line.delivery_cost for line in scoped)
                > constraint.maximum_total_cost + tolerance
            ):
                violations.append(
                    f"SUPPLIER_MAX_COST:{constraint.supplier_id}:{group_id}"
                )
    unique = sorted(set(violations))
    return unique, {
        "offers": not any(item.startswith("UNKNOWN_OFFER") for item in unique),
        "pack_size": not any("PACK_SIZE" in item for item in unique),
        "moq": not any(item.startswith("MOQ") for item in unique),
        "budget": "BUDGET" not in unique,
        "supplier_capacity": not any(
            item.startswith("SUPPLIER_MAX") for item in unique
        ),
        "availability": not any(
            "UNAVAILABLE" in item or "CUTOFF" in item for item in unique
        ),
        "unit_compatibility": not any(
            item.startswith("UNIT_COMPATIBILITY") for item in unique
        ),
    }
