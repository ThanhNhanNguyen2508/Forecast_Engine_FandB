from __future__ import annotations

from collections import defaultdict

import numpy as np

from shelfcash_forecast.inventory.contracts import (
    DailyInventoryLedger,
    EndingLotState,
    InventoryKeyRiskMetrics,
    InventoryKeySummary,
    InventoryRiskMetrics,
    InventorySimulationResult,
    InventorySimulationSummary,
)

InventoryKey = tuple[str, str, str]


def _sum_optional(values: list[float | None]) -> float | None:
    return float(sum(value for value in values if value is not None)) if all(
        value is not None for value in values
    ) else None


# Input:
# DailyInventoryLedger[]
# EndingLotState[]
# Output:
# InventorySimulationSummary
# Tức:
# daily detailed history
#       ↓
# business KPIs
# Ví dụ S001 sau 7 ngày:

# STORE_A × Chicken

# Total demand = 700 kg
# Fulfilled = 665 kg
# Shortage = 35 kg
# Expired = 20 kg
# Waste = 5 kg

# Fill rate = 95%
# Projected stockout = 17/08

# Ending inventory = 40 kg
# Maximum inventory = 180 kg
# Peak at-risk expiry = 30 kg

# Total consequence cost = 4.2m

# Đây chính là:

# InventoryKeySummary

# Sau đó nhiều keys thành:

# InventorySimulationSummary
def summarize_simulation( # tóm tắt hậu quả của 1 world sau 1 simulate_inventory 
    ledgers: list[DailyInventoryLedger],
    ending_lots: list[EndingLotState],
    *,
    capacity_evaluated_keys: set[tuple[str, str]],
) -> InventorySimulationSummary:
    """Summarize physical quantities only within one store/ingredient/unit key."""

    grouped_ledgers: defaultdict[InventoryKey, list[DailyInventoryLedger]] = defaultdict(
        list
    )
    for ledger in ledgers:
        grouped_ledgers[(ledger.store_id, ledger.ingredient_id, ledger.unit)].append(
            ledger
        )
    ending_by_key: defaultdict[InventoryKey, float] = defaultdict(float)
    for lot in ending_lots:
        ending_by_key[(lot.store_id, lot.ingredient_id, lot.unit)] += (
            lot.quantity_remaining
        )

    by_key: list[InventoryKeySummary] = []
    for inventory_key in sorted(grouped_ledgers):
        store_id, ingredient_id, unit = inventory_key
        key_ledgers = sorted(
            grouped_ledgers[inventory_key], key=lambda item: item.simulation_date
        )
        total_demand = sum(item.demand_quantity for item in key_ledgers)
        fulfilled = sum(item.fulfilled_quantity for item in key_ledgers)
        ending_inventory = ending_by_key[inventory_key]
        day_count = len({item.simulation_date for item in key_ledgers})
        mean_daily_demand = total_demand / day_count
        stockout_dates = [
            item.simulation_date for item in key_ledgers if item.shortage_quantity > 0
        ]
        capacity_evaluated = (store_id, ingredient_id) in capacity_evaluated_keys
        holding_cost = _sum_optional([item.holding_cost for item in key_ledgers])
        shortage_cost = _sum_optional([item.shortage_cost for item in key_ledgers])
        expiry_cost = _sum_optional([item.expiry_cost for item in key_ledgers])
        waste_cost = _sum_optional([item.waste_cost for item in key_ledgers])
        total_cost = _sum_optional(
            [holding_cost, shortage_cost, expiry_cost, waste_cost]
        )
        by_key.append(
            InventoryKeySummary(
                store_id=store_id,
                ingredient_id=ingredient_id,
                unit=unit,
                total_demand=total_demand,
                fulfilled_quantity=fulfilled,
                shortage_quantity=sum(item.shortage_quantity for item in key_ledgers),
                expired_quantity=sum(item.expired_quantity for item in key_ledgers),
                explicit_waste_quantity=sum(
                    item.waste_quantity for item in key_ledgers
                ),
                beginning_inventory=key_ledgers[0].beginning_quantity,
                total_inbound=sum(item.inbound_quantity for item in key_ledgers),
                ending_inventory=ending_inventory,
                maximum_inventory=max(item.maximum_quantity for item in key_ledgers),
                fill_rate=fulfilled / total_demand if total_demand else 1.0,
                days_of_supply=(
                    ending_inventory / mean_daily_demand
                    if mean_daily_demand > 0
                    else None
                ),
                at_risk_expiry_quantity=max(
                    item.at_risk_expiry_quantity for item in key_ledgers
                ),
                projected_stockout_date=(min(stockout_dates) if stockout_dates else None),
                stockout_event_count=len(stockout_dates),
                capacity_violation_quantity=(
                    max(item.capacity_violation_quantity for item in key_ledgers)
                    if capacity_evaluated
                    else None
                ),
                holding_cost=holding_cost,
                shortage_cost=shortage_cost,
                expiry_cost=expiry_cost,
                waste_cost=waste_cost,
                total_consequence_cost=total_cost,
            )
        )

    if not by_key:
        raise ValueError("Simulation summary requires at least one inventory key.")
    single = by_key[0] if len(by_key) == 1 else None
    global_cost = _sum_optional([item.total_consequence_cost for item in by_key])
    capacity_complete = all(
        item.capacity_violation_quantity is not None for item in by_key
    )
    return InventorySimulationSummary(
        by_key=by_key,
        inventory_key_count=len(by_key),
        number_of_stockout_events=sum(item.stockout_event_count for item in by_key),
        number_of_ingredient_keys_with_stockout=sum(
            item.stockout_event_count > 0 for item in by_key
        ),
        mean_key_fill_rate=float(np.mean([item.fill_rate for item in by_key])),
        number_of_capacity_violations=(
            sum(float(item.capacity_violation_quantity or 0) > 0 for item in by_key)
            if capacity_complete
            else None
        ),
        total_demand=single.total_demand if single else None,
        total_fulfilled=single.fulfilled_quantity if single else None,
        total_shortage=single.shortage_quantity if single else None,
        total_expired=single.expired_quantity if single else None,
        total_waste=single.explicit_waste_quantity if single else None,
        ending_inventory=single.ending_inventory if single else None,
        maximum_inventory=single.maximum_inventory if single else None,
        at_risk_expiry_quantity=(single.at_risk_expiry_quantity if single else None),
        fill_rate=single.fill_rate if single else None,
        capacity_violation_quantity=(
            single.capacity_violation_quantity if single else None
        ),
        days_of_supply=single.days_of_supply if single else None,
        consequence_cost=global_cost,
    )


def _normalized_weights(results: list[InventorySimulationResult]) -> np.ndarray:
    raw = [result.probability_weight for result in results]
    if not all(weight is not None for weight in raw):
        raise ValueError("Risk aggregation requires explicit probability weights.")
    values = np.asarray(raw, dtype=float)
    return values / values.sum()


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values)
    ordered_values = values[order]
    cumulative = np.cumsum(weights[order])
    index = min(int(np.searchsorted(cumulative, quantile, side="left")), len(values) - 1)
    return float(ordered_values[index])


def weighted_cvar(
    values: np.ndarray,
    weights: np.ndarray,
    alpha: float,
) -> float:
    order = np.argsort(values)[::-1]
    tail_mass = 1.0 - alpha
    remaining = tail_mass
    total = 0.0
    for index in order:
        used = min(float(weights[index]), remaining)
        total += float(values[index]) * used
        remaining -= used
        if remaining <= 1e-12:
            break
    return total / tail_mass


def _key_risk_metrics(
    inventory_key: InventoryKey,
    summaries: list[InventoryKeySummary],
    weights: np.ndarray,
    *,
    waste_threshold: float,
    fill_rate_target: float,
) -> InventoryKeyRiskMetrics:
    store_id, ingredient_id, unit = inventory_key

    def probability(value: float) -> float:
        # Normalized decimal weights can sum to 1 +/- a few ulps.
        return min(1.0, max(0.0, float(value)))

    def values(field: str) -> np.ndarray:
        return np.asarray([getattr(item, field) for item in summaries], dtype=float)

    shortages = values("shortage_quantity")
    expired = values("expired_quantity")
    waste = values("explicit_waste_quantity")
    fill_rate = values("fill_rate")
    ending = values("ending_inventory")
    maximum = values("maximum_inventory")
    at_risk = values("at_risk_expiry_quantity")
    stockout_dates: defaultdict[str, float] = defaultdict(float)
    for summary, weight in zip(summaries, weights, strict=True):
        if summary.projected_stockout_date is not None:
            stockout_dates[summary.projected_stockout_date.isoformat()] += float(weight)

    day_values = [item.days_of_supply for item in summaries]
    day_indices = [index for index, value in enumerate(day_values) if value is not None]
    day_array = np.asarray(
        [value for value in day_values if value is not None], dtype=float
    )
    day_weights = weights[day_indices]
    if len(day_weights):
        day_weights = day_weights / day_weights.sum()

    capacities = [item.capacity_violation_quantity for item in summaries]
    capacity_probability = None
    if all(value is not None for value in capacities):
        capacity_array = np.asarray(capacities, dtype=float)
        capacity_probability = probability(np.sum(weights[capacity_array > 0]))

    costs = [item.total_consequence_cost for item in summaries]
    cost_array = (
        np.asarray(costs, dtype=float)
        if all(value is not None for value in costs)
        else None
    )
    return InventoryKeyRiskMetrics(
        store_id=store_id,
        ingredient_id=ingredient_id,
        unit=unit,
        stockout_probability=probability(np.sum(weights[shortages > 0])),
        expected_shortage=float(np.dot(shortages, weights)),
        p50_shortage=weighted_quantile(shortages, weights, 0.5),
        p95_shortage=weighted_quantile(shortages, weights, 0.95),
        expected_expired_quantity=float(np.dot(expired, weights)),
        expected_explicit_waste=float(np.dot(waste, weights)),
        waste_threshold_exceedance_probability=probability(
            np.sum(weights[waste > waste_threshold])
        ),
        expected_fill_rate=probability(np.dot(fill_rate, weights)),
        fill_rate_below_target_probability=probability(
            np.sum(weights[fill_rate < fill_rate_target])
        ),
        expected_ending_inventory=float(np.dot(ending, weights)),
        p50_ending_inventory=weighted_quantile(ending, weights, 0.5),
        p95_ending_inventory=weighted_quantile(ending, weights, 0.95),
        expected_maximum_inventory=float(np.dot(maximum, weights)),
        p50_maximum_inventory=weighted_quantile(maximum, weights, 0.5),
        p95_maximum_inventory=weighted_quantile(maximum, weights, 0.95),
        expected_at_risk_expiry_quantity=float(np.dot(at_risk, weights)),
        p50_at_risk_expiry_quantity=weighted_quantile(at_risk, weights, 0.5),
        p95_at_risk_expiry_quantity=weighted_quantile(at_risk, weights, 0.95),
        capacity_violation_probability=capacity_probability,
        expected_days_of_supply=(
            float(np.dot(day_array, day_weights)) if len(day_array) else None
        ),
        p50_days_of_supply=(
            weighted_quantile(day_array, day_weights, 0.5) if len(day_array) else None
        ),
        p95_days_of_supply=(
            weighted_quantile(day_array, day_weights, 0.95) if len(day_array) else None
        ),
        projected_stockout_date_distribution={
            key: probability(value) for key, value in sorted(stockout_dates.items())
        },
        expected_consequence_cost=(
            float(np.dot(cost_array, weights)) if cost_array is not None else None
        ),
        p95_consequence_cost=(
            weighted_quantile(cost_array, weights, 0.95)
            if cost_array is not None
            else None
        ),
        cvar95_consequence_cost=(
            weighted_cvar(cost_array, weights, 0.95)
            if cost_array is not None
            else None
        ),
    )


def aggregate_risk_metrics(
#     Sau khi đã chạy N worlds, rủi ro tổng thể là bao nhiêu?

# Input:

# InventorySimulationResult[]

# Output:

# InventoryRiskMetrics

# Đây là:

# N deterministic consequences
#         ↓
# probability-weighted risk distribution


# aggregate_risk_metrics()
# MANY WORLDS

# Ví dụ:

# P(stockout) = 30%
# Expected shortage = 8kg
# P95 shortage = 40kg
# Expected cost = 5m
# CVaR95 = 25m

# Nó là probabilistic risk.
    results: list[InventorySimulationResult],
    *,
    waste_threshold: float,
    fill_rate_target: float,
) -> InventoryRiskMetrics:
    if not results:
        raise ValueError("At least one simulation result is required for risk metrics.")
    weights = _normalized_weights(results)
    expected_keys = {
        (item.store_id, item.ingredient_id, item.unit)
        for item in results[0].summary.by_key
    }
    summary_maps = []
    for result in results:
        mapping = {
            (item.store_id, item.ingredient_id, item.unit): item
            for item in result.summary.by_key
        }
        if set(mapping) != expected_keys:
            raise ValueError("All stochastic scenarios must cover the same inventory keys.")
        summary_maps.append(mapping)

    by_key = [
        _key_risk_metrics(
            inventory_key,
            [mapping[inventory_key] for mapping in summary_maps],
            weights,
            waste_threshold=waste_threshold,
            fill_rate_target=fill_rate_target,
        )
        for inventory_key in sorted(expected_keys)
    ]
    affected_counts = np.asarray(
        [result.summary.number_of_ingredient_keys_with_stockout for result in results],
        dtype=float,
    )
    any_stockout = affected_counts > 0
    capacity_complete = all(
        result.summary.number_of_capacity_violations is not None for result in results
    )
    any_capacity = (
        np.asarray(
            [float(result.summary.number_of_capacity_violations or 0) for result in results]
        )
        > 0
    )
    costs = [result.summary.consequence_cost for result in results]
    cost_array = (
        np.asarray(costs, dtype=float)
        if all(value is not None for value in costs)
        else None
    )
    single = by_key[0] if len(by_key) == 1 else None
    probability = lambda value: min(1.0, max(0.0, float(value)))
    return InventoryRiskMetrics(
        scenario_count=len(results),
        by_key=by_key,
        any_stockout_probability=probability(np.sum(weights[any_stockout])),
        expected_affected_key_count=float(np.dot(affected_counts, weights)),
        expected_affected_key_proportion=probability(
            np.dot(affected_counts / len(expected_keys), weights)
        ),
        mean_key_fill_rate=probability(
            np.mean([item.expected_fill_rate for item in by_key])
        ),
        any_capacity_violation_probability=(
            probability(np.sum(weights[any_capacity])) if capacity_complete else None
        ),
        expected_consequence_cost=(
            float(np.dot(cost_array, weights)) if cost_array is not None else None
        ),
        p95_consequence_cost=(
            weighted_quantile(cost_array, weights, 0.95)
            if cost_array is not None
            else None
        ),
        cvar95_consequence_cost=(
            weighted_cvar(cost_array, weights, 0.95)
            if cost_array is not None
            else None
        ),
        stockout_probability=single.stockout_probability if single else None,
        expected_shortage=single.expected_shortage if single else None,
        p50_shortage=single.p50_shortage if single else None,
        p95_shortage=single.p95_shortage if single else None,
        expected_expired_quantity=(
            single.expected_expired_quantity if single else None
        ),
        expected_explicit_waste=single.expected_explicit_waste if single else None,
        waste_threshold_exceedance_probability=(
            single.waste_threshold_exceedance_probability if single else None
        ),
        expected_fill_rate=single.expected_fill_rate if single else None,
        fill_rate_below_target_probability=(
            single.fill_rate_below_target_probability if single else None
        ),
        expected_ending_inventory=single.expected_ending_inventory if single else None,
        p50_ending_inventory=single.p50_ending_inventory if single else None,
        p95_ending_inventory=single.p95_ending_inventory if single else None,
        expected_maximum_inventory=(
            single.expected_maximum_inventory if single else None
        ),
        p50_maximum_inventory=single.p50_maximum_inventory if single else None,
        p95_maximum_inventory=single.p95_maximum_inventory if single else None,
        expected_at_risk_expiry_quantity=(
            single.expected_at_risk_expiry_quantity if single else None
        ),
        capacity_violation_probability=(
            single.capacity_violation_probability if single else None
        ),
        expected_days_of_supply=single.expected_days_of_supply if single else None,
        p50_days_of_supply=single.p50_days_of_supply if single else None,
        p95_days_of_supply=single.p95_days_of_supply if single else None,
        projected_stockout_date_distribution=(
            single.projected_stockout_date_distribution if single else None
        ),
    )
