from __future__ import annotations

from typing import Any

from shelfcash_forecast.decision_intelligence.integrity import sha256_content_hash
from shelfcash_forecast.decision_intelligence.what_if.contracts import (
    BudgetModification,
    ConsequenceCostModification,
    DemandScaleModification,
    InventoryLotModification,
    InventoryPolicyModification,
    StrategyProfileModification,
    StressScenarioModification,
    SupplierOfferModification,
    WhatIfModification,
)
from shelfcash_forecast.optimization.contracts import OptimizationRequest
from shelfcash_forecast.optimization.strategies import default_strategy_profiles


class MutationError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


def normalize_modifications(
    modifications: list[WhatIfModification],
) -> list[WhatIfModification]:
    keyed = [(sha256_content_hash(item), item) for item in modifications]
    hashes = [key for key, _ in keyed]
    if len(hashes) != len(set(hashes)):
        raise MutationError("M6_WHAT_IF_DUPLICATE_MODIFICATION", "duplicate modification")
    return [item for _, item in sorted(keyed, key=lambda pair: pair[0])]


def _only_match(rows: list[Any], code_prefix: str) -> Any:
    if not rows:
        raise MutationError(f"{code_prefix}_ZERO_MATCH", "selector matched no artifact")
    if len(rows) > 1:
        raise MutationError(f"{code_prefix}_AMBIGUOUS_MATCH", "selector matched multiple artifacts")
    return rows[0]


def _apply_demand(data: dict[str, Any], modification: DemandScaleModification) -> None:
    selector = modification.selector
    matches: list[dict[str, Any]] = []
    for scenario in data["demand_scenarios"]:
        if selector.scenario_id is not None and scenario["scenario_id"] != selector.scenario_id:
            continue
        for line in scenario["lines"]:
            if selector.store_id is not None and line["store_id"] != selector.store_id:
                continue
            if (
                selector.ingredient_id is not None
                and line["ingredient_id"] != selector.ingredient_id
            ):
                continue
            if selector.unit is not None and line["unit"] != selector.unit:
                continue
            if selector.target_date is not None and line["target_date"] != selector.target_date:
                continue
            matches.append(line)
    if len(matches) != selector.expected_matches:
        code = "ZERO_MATCH" if not matches else "CARDINALITY_MISMATCH"
        raise MutationError(
            f"M6_WHAT_IF_DEMAND_SELECTOR_{code}",
            f"expected {selector.expected_matches}, observed {len(matches)}",
        )
    for line in matches:
        line["quantity"] *= modification.multiplier


def _apply_offer(data: dict[str, Any], modification: SupplierOfferModification) -> None:
    offer = _only_match(
        [row for row in data["supplier_offers"] if row["offer_id"] == modification.offer_id],
        "M6_WHAT_IF_OFFER",
    )
    fields = (
        "available",
        "unit_price",
        "delivery_cost",
        "minimum_order_quantity",
        "maximum_order_quantity",
        "lead_time_days",
        "shelf_life_days",
        "order_cutoff_date",
        "emergency",
    )
    for field in fields:
        value = getattr(modification, field)
        if value is not None:
            offer[field] = value
    if modification.clear_maximum_order_quantity:
        offer["maximum_order_quantity"] = None
    if modification.clear_shelf_life_days:
        offer["shelf_life_days"] = None
    if modification.clear_order_cutoff_date:
        offer["order_cutoff_date"] = None


def _lot_matches(row: dict[str, Any], modification: InventoryLotModification) -> bool:
    return (
        row["lot_id"] == modification.lot_id
        and (modification.store_id is None or row["store_id"] == modification.store_id)
        and (
            modification.ingredient_id is None or row["ingredient_id"] == modification.ingredient_id
        )
        and (modification.unit is None or row["unit"] == modification.unit)
    )


def _apply_lot(data: dict[str, Any], modification: InventoryLotModification) -> None:
    rows = data["initial_inventory"]
    matches = [row for row in rows if _lot_matches(row, modification)]
    if modification.action == "ADD":
        if matches:
            raise MutationError("M6_WHAT_IF_LOT_ALREADY_EXISTS", "lot ID already exists")
        assert modification.lot is not None
        rows.append(modification.lot.model_dump(mode="python"))
        return
    lot = _only_match(matches, "M6_WHAT_IF_LOT")
    if modification.action == "REMOVE":
        rows.remove(lot)
    elif modification.action == "SET_QUANTITY":
        lot["quantity_remaining"] = modification.quantity
    else:
        lot["expiry_date"] = None if modification.clear_expiry else modification.expiry_date


def _apply_policy(data: dict[str, Any], modification: InventoryPolicyModification) -> None:
    for name, value in modification.model_dump(mode="python").items():
        if name != "modification_type" and value is not None:
            data["inventory_policy"][name] = value


def _apply_profile(data: dict[str, Any], modification: StrategyProfileModification) -> None:
    profiles = {profile["name"]: profile for profile in data["strategy_profiles"]}
    if modification.strategy not in profiles:
        defaults = {
            profile.name: profile.model_dump(mode="python")
            for profile in default_strategy_profiles()
        }
        profiles[modification.strategy] = defaults[modification.strategy]
    target = profiles[modification.strategy]
    for name, value in modification.model_dump(mode="python").items():
        if name not in {"modification_type", "strategy"} and value is not None:
            target[name] = value
    data["strategy_profiles"] = [profiles[name] for name in sorted(profiles)]


def _apply_cost(data: dict[str, Any], modification: ConsequenceCostModification) -> None:
    target = _only_match(
        [
            row
            for row in data["cost_assumptions"]
            if row["store_id"] == modification.store_id
            and row["ingredient_id"] == modification.ingredient_id
            and row["unit"] == modification.unit
        ],
        "M6_WHAT_IF_CONSEQUENCE_COST",
    )
    for name, value in modification.model_dump(mode="python").items():
        if name not in {
            "modification_type",
            "store_id",
            "ingredient_id",
            "unit",
            "clear_capacity_quantity",
        } and (value is not None):
            target[name] = value
    if modification.clear_capacity_quantity:
        target["capacity_quantity"] = None


def _apply_stress(data: dict[str, Any], modification: StressScenarioModification) -> None:
    target = _only_match(
        [row for row in data["stress_scenarios"] if row["stress_id"] == modification.stress_id],
        "M6_WHAT_IF_STRESS",
    )
    for name, value in modification.model_dump(mode="python").items():
        if name not in {"modification_type", "stress_id"} and value is not None:
            target[name] = value


def apply_modifications(
    baseline: OptimizationRequest,
    modifications: list[WhatIfModification],
    *,
    hypothetical_request_id: str,
) -> OptimizationRequest:
    """Clone, mutate allowlisted fields, then run the original strict request validator."""

    data = baseline.model_dump(mode="python")
    data["request_id"] = hypothetical_request_id
    for modification in normalize_modifications(modifications):
        if isinstance(modification, DemandScaleModification):
            _apply_demand(data, modification)
        elif isinstance(modification, SupplierOfferModification):
            _apply_offer(data, modification)
        elif isinstance(modification, InventoryLotModification):
            _apply_lot(data, modification)
        elif isinstance(modification, BudgetModification):
            data["budget"] = None if modification.clear_budget else modification.budget
        elif isinstance(modification, InventoryPolicyModification):
            _apply_policy(data, modification)
        elif isinstance(modification, StrategyProfileModification):
            _apply_profile(data, modification)
        elif isinstance(modification, ConsequenceCostModification):
            _apply_cost(data, modification)
        elif isinstance(modification, StressScenarioModification):
            _apply_stress(data, modification)
        else:  # pragma: no cover - discriminated union is closed
            raise MutationError("M6_WHAT_IF_MODIFICATION_NOT_SUPPORTED", str(type(modification)))
    return OptimizationRequest.model_validate(data)


__all__ = ["MutationError", "apply_modifications", "normalize_modifications"]
