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
    demands = [m for m in modifications if isinstance(m, DemandScaleModification)]
    for i, left in enumerate(demands):
        for right in demands[i + 1:]:
            if all(getattr(left.selector, k) is None or getattr(right.selector, k) is None
                   or getattr(left.selector, k) == getattr(right.selector, k)
                   for k in ("scenario_id", "store_id", "ingredient_id", "unit", "target_date")):
                raise MutationError("M6_WHAT_IF_OVERLAPPING_DEMAND_CHANGES", "logical demand line may only be scaled once")
    targets = []
    for item in modifications:
        if isinstance(item, DemandScaleModification):
            continue
        identity = next((getattr(item, k) for k in ("offer_id", "lot_id", "strategy", "stress_id")
                         if hasattr(item, k)), None)
        if isinstance(item, ConsequenceCostModification):
            identity = (item.store_id, item.ingredient_id, item.unit)
        target = (item.modification_type, identity)
        if target in targets:
            raise MutationError("M6_WHAT_IF_CONFLICTING_CHANGES", "multiple changes to the same target")
        targets.append(target)
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
    if selector.scope!='ALL_APPLICABLE_DEMAND' and len(matches) != selector.expected_matches:
        code = "ZERO_MATCH" if not matches else "CARDINALITY_MISMATCH"
        raise MutationError(
            f"M6_WHAT_IF_DEMAND_SELECTOR_{code}",
            f"expected {selector.expected_matches}, observed {len(matches)}",
        )
    for line in matches:
        line["quantity"] *= modification.multiplier
    # Full evaluation retains its original weights; update the same demand selector
    # across every relevant world, including worlds omitted from optimization.
    for scenario in data.get("evaluation_scenarios", []):
        if selector.scenario_id is not None and scenario["scenario_id"] != selector.scenario_id:
            continue
        for line in scenario["lines"]:
            if all(getattr(selector, name) is None or line[name] == getattr(selector, name)
                   for name in ("store_id", "ingredient_id", "unit", "target_date")):
                line["quantity"] *= modification.multiplier


def _apply_offer(data: dict[str, Any], modification: SupplierOfferModification) -> None:
    offer = _only_match(
        [row for row in data["supplier_offers"] if row["offer_id"] == modification.offer_id],
        "M6_WHAT_IF_OFFER",
    )
    fields = (
        "pack_size",
        "calendar",
        "delivery_weekdays",
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
    if modification.pack_size is not None:
        if offer['moq_basis']=='pack_count':
            raw_moq=offer['source_terms'].get('minimum_order_quantity')
            if raw_moq is None:
                raise MutationError('M6_WHAT_IF_MOQ_SOURCE_REQUIRED','pack-count MOQ needs original scoped quantity')
            offer['minimum_order_quantity']=raw_moq*modification.pack_size
        price_mapping=offer['source_terms'].get('price_mapping')
        if price_mapping and price_mapping['basis']=='PACK':
            offer['unit_price']=offer['source_terms']['unit_price']/modification.pack_size
    if modification.lead_time_days is not None or modification.calendar is not None or modification.delivery_weekdays is not None:
        from shelfcash_forecast.optimization.chronology import offer_arrival
        from shelfcash_forecast.optimization.contracts import SupplierOffer
        offer["resolved_arrival_date"] = None
        validated = SupplierOffer.model_validate(offer)
        offer["resolved_arrival_date"] = offer_arrival(validated)


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
        target["capacity_effective_from"] = None
        target["capacity_effective_to"] = None


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

    import copy
    data = copy.deepcopy(baseline.model_dump(mode="python"))
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
    from shelfcash_forecast.optimization.planning_service import content_hash
    data['scenario_provenance']={**data.get('scenario_provenance',{}),
        'parent_request_hash':content_hash(baseline),
        'transformation_hash':content_hash([m.model_dump(mode='json') for m in normalize_modifications(modifications)]),
        'full_pool_hash':content_hash(data.get('evaluation_scenarios') or data['demand_scenarios']),
        'hypothetical':True,'not_business_approval':True}
    if baseline.planning_mode=='SCENARIO_PREVIEW':
        # Derive an explicit new scenario identity; never reuse an approval or
        # silently bind changed contracts to the original profile.
        from shelfcash_forecast.optimization.scenario_contracts import Assumption
        import copy
        modifications_json=[m.model_dump(mode='json') for m in normalize_modifications(modifications)]
        transformation=content_hash(modifications_json)
        old=baseline.planning_binding;new_id=old['profile_id']+'-WHATIF-'+transformation[:12]
        assumptions=copy.deepcopy(old['assumptions'])
        for a in assumptions:a['profile_id']=new_id
        scoped=Assumption(assumption_id=new_id+':what_if',profile_id=new_id,field_path='what_if_modifications',
            scope=assumptions[0]['scope'],value=modifications_json,unit='scoped transformed contract',
            kind='BUSINESS_POLICY_CHOICE',rationale='Explicit hypothetical what-if; parent approval invalidated; full pool and dependent references recomputed.',
            source_status='PROPOSED',source_locator='parent request:'+content_hash(baseline))
        assumptions.append(scoped.model_dump(mode='json'))
        ah=content_hash(assumptions);ph=content_hash({'parent_profile_hash':old['profile_hash'],'transformation':transformation})
        for offer in data['supplier_offers']:
            offer['source_terms'].update(profile_id=new_id,profile_hash=ph,assumption_hash=ah)
        # A changed shipment fee is one term shared by all constituent offers.
        for modification in modifications:
            if isinstance(modification,SupplierOfferModification) and modification.delivery_cost is not None:
                changed=next(o for o in data['supplier_offers'] if o['offer_id']==modification.offer_id)
                group=changed['source_terms'].get('shipment_group_id')
                if group:
                    for o in data['supplier_offers']:
                        if o['source_terms'].get('shipment_group_id')==group:o['delivery_cost']=modification.delivery_cost
        full=data.get('evaluation_scenarios') or data['demand_scenarios']
        n=(baseline.planning_end_date-baseline.decision_date).days
        for rule in data['normalized_rules']:
            rule['assumption_refs'].append(scoped.assumption_id)
            if rule['semantics']=='PURCHASE_COVER_DAYS':
                rule['reference_daily_quantity']=sum(s['probability_weight']*l['quantity']/n for s in full for l in s['lines']
                    if l['store_id']==rule['store_id'] and l['ingredient_id']==rule['ingredient_id'])
        from shelfcash_forecast.optimization.contracts import SupplierOffer
        from shelfcash_forecast.optimization.scenario_contracts import NormalizedBusinessRule
        offers=[SupplierOffer.model_validate(o).model_dump(mode='json') for o in data['supplier_offers']]
        rules=[NormalizedBusinessRule.model_validate(r).model_dump(mode='json') for r in data['normalized_rules']]
        data['planning_binding']={**old,'profile_id':new_id,'profile_version':old['profile_version']+1,
            'profile_hash':ph,'assumption_hash':ah,'assumptions':assumptions,
            'contracts_hash':content_hash({'offers':offers,'rules':rules}),
            'parent_profile_id':old['profile_id'],'parent_profile_hash':old['profile_hash'],
            'approval_binding_status':'INVALIDATED_BY_EXPLICIT_WHAT_IF_TRANSFORMATION','transformation_hash':transformation}
        data['scenario_provenance']['planning_binding']=data['planning_binding']
    return OptimizationRequest.model_validate(data)


__all__ = ["MutationError", "apply_modifications", "normalize_modifications"]
