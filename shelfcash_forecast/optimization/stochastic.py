# Nếu deterministic hỏi:

# “Với expected demand, hôm nay nên mua gì?”

# thì stochastic hỏi:

# “Khi tương lai có nhiều kịch bản demand với xác suất khác nhau, hôm nay nên mua gì trước, và nếu từng kịch bản thật sự xảy ra thì được phép phản ứng thêm bằng emergency order như thế nào, để vừa tối ưu cost vừa kiểm soát risk?”

# File của bạn triển khai đúng tư tưởng chronological two-stage stochastic SAA MILP + CVaR + chance constraints
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from shelfcash_forecast.bom.units import normalize_unit
from shelfcash_forecast.exceptions import OptimizationNotAvailableError
from shelfcash_forecast.inventory.metrics import weighted_cvar
from shelfcash_forecast.optimization.contracts import (
    OptimizationRequest,
    ProcurementDecisionLine,
    ProcurementPlan,
    StrategyProfile,
)
from shelfcash_forecast.optimization.deterministic import (
    _decision,
    _offer_upper_packs,
    _Variables,
)
from shelfcash_forecast.optimization.model_data import build_problem_data
# Input vẫn giống deterministic, nhưng có thêm scenario demand và xác suất kèm theo.
# ProcurementPlan
# │
# ├─ orders
# │    └─ regular orders hôm nay
# │
# ├─ scenario_recourse_orders
# │    ├─ LOW  → emergency orders riêng
# │    ├─ NORMAL → emergency orders riêng
# │    └─ HIGH → emergency orders riêng
# │
# ├─ predicted service/risk metrics
# ├─ CVaR
# ├─ scenario outcomes
# │
# └─ completed=False

# 3. Cách dễ nhất để hiểu stochastic: “mua trước + chữa cháy sau”

# Giả sử hôm nay chưa biết demand tương lai.

# Có 3 scenario:

# LOW
# P = 20%
# Demand = 50kg

# NORMAL
# P = 50%
# Demand = 80kg

# HIGH
# P = 30%
# Demand = 150kg

# Regular supplier:

# Supplier A
# 100k/kg
# lead time 1 ngày

# Emergency supplier:

# Supplier B
# 160k/kg
# same-day

# Stochastic solver có thể quyết định:

# FIRST STAGE:
# Hôm nay mua A = 80kg

# Sau đó:

# Nếu LOW xảy ra:
# Emergency = 0

# Nếu NORMAL xảy ra:
# Emergency = 0

# Nếu HIGH xảy ra:
# Emergency = 70kg
 # Sau khi có 3 viễn cảnh này, hiện thực tiếp các file phía sau
# Đó là bản chất của two-stage stochastic optimization.
def solve_stochastic_procurement(
    request: OptimizationRequest,
    profile: StrategyProfile,
) -> ProcurementPlan:
    """Solve a chronological two-stage SAA MILP.

    Regular orders are non-anticipative first-stage decisions. Inventory,
    lost-sales shortage, and supplied emergency recourse are indexed by
    scenario, inventory key, and day. The formulation is deliberately an
    aggregate chronological approximation; M4 remains the only exact FEFO and
    expiry implementation and must re-simulate every returned candidate.
    """

    data = build_problem_data(request)
    if not data.probabilistic_weights:
        raise OptimizationNotAvailableError(
            "Stochastic optimization requires genuine scenario probability weights.",
            details={"scenario_ids": data.scenario_ids},
        )
    maximum_demand = {
        key: max(
            sum(data.demand.get((scenario_id, key, day), 0) for day in data.dates)
            for scenario_id in data.scenario_ids
        )
        for key in data.keys
    }
    scenario_key_totals = {
        (scenario_id, key): sum(
            data.demand.get((scenario_id, key, day), 0) for day in data.dates
        )
        for scenario_id in data.scenario_ids
        for key in data.keys
    }

    variables = _Variables()
    regular_x: dict[int, int] = {}
    regular_y: dict[int, int] = {}
    regular_upper: dict[int, int] = {}
    for index, item in enumerate(data.regular_offers):
        maximum = _offer_upper_packs(item, request, maximum_demand)
        regular_upper[index] = maximum
        regular_x[index] = variables.add(
            item.offer.pack_size
            * item.offer.unit_price
            * (1 + profile.cash_penalty),
            0,
            maximum,
            True,
        )
        regular_y[index] = variables.add(
            item.offer.delivery_cost * (1 + profile.cash_penalty), 0, 1, True
        )

    inventory: dict[tuple[str, tuple[str, str], object], int] = {}
    shortage: dict[tuple[str, tuple[str, str], object], int] = {}
    emergency_x: dict[tuple[str, int], int] = {}
    emergency_y: dict[tuple[str, int], int] = {}
    emergency_upper = {
        index: _offer_upper_packs(item, request, maximum_demand)
        for index, item in enumerate(data.emergency_offers)
    }
    for scenario_id, weight in zip(
        data.scenario_ids, data.scenario_weights, strict=True
    ):
        for key in data.keys:
            assumption = data.assumptions.get(key)
            shortage_cost = profile.shortage_penalty + (
                assumption.shortage_cost_per_unit if assumption else 0
            )
            holding_cost = profile.holding_penalty + (
                assumption.holding_cost_per_unit_day if assumption else 0
            )
            for day in data.dates:
                inventory[(scenario_id, key, day)] = variables.add(
                    float(weight)
                    * (
                        holding_cost
                        + (profile.waste_penalty if day == data.dates[-1] else 0)
                    ),
                    0,
                    assumption.capacity_quantity
                    if assumption and assumption.capacity_quantity is not None
                    else np.inf,
                )
                daily_demand = data.demand.get((scenario_id, key, day), 0)
                shortage[(scenario_id, key, day)] = variables.add(
                    float(weight) * shortage_cost,
                    0,
                    daily_demand,
                )
        for index, item in enumerate(data.emergency_offers):
            emergency_x[(scenario_id, index)] = variables.add(
                float(weight) * item.offer.pack_size * item.offer.unit_price,
                0,
                emergency_upper[index],
                True,
            )
            emergency_y[(scenario_id, index)] = variables.add(
                float(weight) * item.offer.delivery_cost, 0, 1, True
            )

    eta: int | None = None
    excess: dict[str, int] = {}
    if profile.cvar_weight > 0:
        eta = variables.add(profile.cvar_weight, -np.inf, np.inf)
        for scenario_id, weight in zip(
            data.scenario_ids, data.scenario_weights, strict=True
        ):
            excess[scenario_id] = variables.add(
                profile.cvar_weight * float(weight) / (1 - profile.cvar_alpha),
                0,
                np.inf,
            )

    stockout_violation = (
        {
            scenario_id: variables.add(0, 0, 1, True)
            for scenario_id in data.scenario_ids
        }
        if profile.maximum_stockout_probability is not None
        else {}
    )
    fill_violation = (
        {
            (scenario_id, key): variables.add(0, 0, 1, True)
            for scenario_id in data.scenario_ids
            for key in data.keys
        }
        if profile.minimum_fill_rate is not None
        and profile.required_fill_rate_probability is not None
        else {}
    )

    rows: list[dict[int, float]] = []
    lower: list[float] = []
    upper: list[float] = []

    def add(coefficients: dict[int, float], low: float, high: float) -> None:
        rows.append(coefficients)
        lower.append(low)
        upper.append(high)

    for index, item in enumerate(data.regular_offers):
        minimum = math.ceil(
            item.offer.minimum_order_quantity / item.offer.pack_size - 1e-12
        )
        add(
            {regular_x[index]: 1, regular_y[index]: -regular_upper[index]},
            -np.inf,
            0,
        )
        add({regular_x[index]: -1, regular_y[index]: minimum}, -np.inf, 0)
    for scenario_id in data.scenario_ids:
        for index, item in enumerate(data.emergency_offers):
            minimum = math.ceil(
                item.offer.minimum_order_quantity / item.offer.pack_size - 1e-12
            )
            add(
                {
                    emergency_x[(scenario_id, index)]: 1,
                    emergency_y[(scenario_id, index)]: -emergency_upper[index],
                },
                -np.inf,
                0,
            )
            add(
                {
                    emergency_x[(scenario_id, index)]: -1,
                    emergency_y[(scenario_id, index)]: minimum,
                },
                -np.inf,
                0,
            )

    # Daily lost-sales balance. An offer variable enters only on its actual
    # arrival day, so inventory arriving at t+1 cannot satisfy demand at t.
    for scenario_id in data.scenario_ids:
        for key in data.keys:
            previous: int | None = None
            for day in data.dates:
                coefficients: dict[int, float] = {
                    inventory[(scenario_id, key, day)]: 1,
                    shortage[(scenario_id, key, day)]: -1,
                }
                if previous is None:
                    beginning = data.initial_quantity.get(key, 0)
                else:
                    coefficients[previous] = -1
                    beginning = 0.0
                for index, item in enumerate(data.regular_offers):
                    if (
                        (item.offer.store_id, item.offer.ingredient_id) == key
                        and item.arrival_date == day
                    ):
                        coefficients[regular_x[index]] = -item.pack_quantity_target
                for index, item in enumerate(data.emergency_offers):
                    if (
                        (item.offer.store_id, item.offer.ingredient_id) == key
                        and item.arrival_date == day
                    ):
                        coefficients[emergency_x[(scenario_id, index)]] = (
                            -item.pack_quantity_target
                        )
                rhs = (
                    beginning
                    + data.existing_inbound.get((key, day), 0)
                    - data.demand.get((scenario_id, key, day), 0)
                )
                add(coefficients, rhs, rhs)
                previous = inventory[(scenario_id, key, day)]

    if request.budget is not None:
        coefficients = {}
        for index, item in enumerate(data.regular_offers):
            coefficients[regular_x[index]] = item.offer.pack_size * item.offer.unit_price
            coefficients[regular_y[index]] = item.offer.delivery_cost
        add(coefficients, -np.inf, request.budget)

    for constraint in request.supplier_constraints:
        regular_matching = [
            (index, item)
            for index, item in enumerate(data.regular_offers)
            if item.offer.supplier_id == constraint.supplier_id
            and (constraint.store_id is None or item.offer.store_id == constraint.store_id)
            and (
                constraint.ingredient_id is None
                or item.offer.ingredient_id == constraint.ingredient_id
            )
        ]
        emergency_matching = [
            (index, item)
            for index, item in enumerate(data.emergency_offers)
            if item.offer.supplier_id == constraint.supplier_id
            and (constraint.store_id is None or item.offer.store_id == constraint.store_id)
            and (
                constraint.ingredient_id is None
                or item.offer.ingredient_id == constraint.ingredient_id
            )
        ]
        scoped_items = [item for _, item in [*regular_matching, *emergency_matching]]
        if constraint.maximum_total_quantity is not None and any(
            normalize_unit(item.offer.unit) != normalize_unit(constraint.unit or "")
            for item in scoped_items
        ):
            raise OptimizationNotAvailableError(
                "Supplier quantity cap unit does not match scoped offers.",
                code="INVALID_PROCUREMENT_UNIT",
            )
        for scenario_id in data.scenario_ids:
            if constraint.maximum_total_quantity is not None:
                coefficients = {
                    regular_x[index]: item.offer.pack_size
                    for index, item in regular_matching
                }
                coefficients.update(
                    {
                        emergency_x[(scenario_id, index)]: item.offer.pack_size
                        for index, item in emergency_matching
                    }
                )
                add(coefficients, -np.inf, constraint.maximum_total_quantity)
            if constraint.maximum_total_cost is not None:
                coefficients = {}
                for index, item in regular_matching:
                    coefficients[regular_x[index]] = (
                        item.offer.pack_size * item.offer.unit_price
                    )
                    coefficients[regular_y[index]] = item.offer.delivery_cost
                for index, item in emergency_matching:
                    coefficients[emergency_x[(scenario_id, index)]] = (
                        item.offer.pack_size * item.offer.unit_price
                    )
                    coefficients[emergency_y[(scenario_id, index)]] = (
                        item.offer.delivery_cost
                    )
                add(coefficients, -np.inf, constraint.maximum_total_cost)

    if profile.minimum_expected_fill_rate is not None:
        # Service is enforced independently for every physical inventory key.
        # A large liter demand can therefore never mask a kg/unit shortage.
        for key in data.keys:
            coefficients = {}
            for scenario_id, weight in zip(
                data.scenario_ids, data.scenario_weights, strict=True
            ):
                key_demand = scenario_key_totals[(scenario_id, key)]
                if key_demand <= 0:
                    # M4 defines a zero-demand key/scenario fill rate as one.
                    continue
                for day in data.dates:
                    coefficients[shortage[(scenario_id, key, day)]] = (
                        float(weight) / key_demand
                    )
            # E_s[fill_rate(s, key)] >= target, with scenario demand a known
            # constant, is linear and exactly matches M4's per-key definition.
            add(
                coefficients,
                -np.inf,
                1 - profile.minimum_expected_fill_rate,
            )

    if stockout_violation:
        for scenario_id in data.scenario_ids:
            for key in data.keys:
                for day in data.dates:
                    safe_big_m = data.demand.get((scenario_id, key, day), 0)
                    add(
                        {
                            shortage[(scenario_id, key, day)]: 1,
                            stockout_violation[scenario_id]: -safe_big_m,
                        },
                        -np.inf,
                        0,
                    )
        add(
            {
                stockout_violation[scenario_id]: float(weight)
                for scenario_id, weight in zip(
                    data.scenario_ids, data.scenario_weights, strict=True
                )
            },
            -np.inf,
            float(profile.maximum_stockout_probability),
        )

    if fill_violation:
        for key in data.keys:
            for scenario_id in data.scenario_ids:
                key_demand = scenario_key_totals[(scenario_id, key)]
                allowed = (1 - float(profile.minimum_fill_rate)) * key_demand
                coefficients = {
                    shortage[(scenario_id, key, day)]: 1 for day in data.dates
                }
                # Maximum excess above the allowed shortage is derived from
                # this key's scenario demand, never from mixed physical units.
                safe_big_m = key_demand - allowed
                coefficients[fill_violation[(scenario_id, key)]] = -safe_big_m
                add(coefficients, -np.inf, allowed)
            add(
                {
                    fill_violation[(scenario_id, key)]: float(weight)
                    for scenario_id, weight in zip(
                        data.scenario_ids, data.scenario_weights, strict=True
                    )
                },
                -np.inf,
                1 - float(profile.required_fill_rate_probability),
            )

    if eta is not None:
        for scenario_id in data.scenario_ids:
            coefficients: defaultdict[int, float] = defaultdict(float)
            coefficients[excess[scenario_id]] = 1
            coefficients[eta] = 1
            for index, item in enumerate(data.regular_offers):
                coefficients[regular_x[index]] -= (
                    item.offer.pack_size
                    * item.offer.unit_price
                    * (1 + profile.cash_penalty)
                )
                coefficients[regular_y[index]] -= item.offer.delivery_cost * (
                    1 + profile.cash_penalty
                )
            for key in data.keys:
                assumption = data.assumptions.get(key)
                shortage_cost = profile.shortage_penalty + (
                    assumption.shortage_cost_per_unit if assumption else 0
                )
                holding_cost = profile.holding_penalty + (
                    assumption.holding_cost_per_unit_day if assumption else 0
                )
                for day in data.dates:
                    coefficients[shortage[(scenario_id, key, day)]] -= shortage_cost
                    coefficients[inventory[(scenario_id, key, day)]] -= (
                        holding_cost
                        + (
                            profile.waste_penalty
                            if day == data.dates[-1]
                            else 0
                        )
                    )
            for index, item in enumerate(data.emergency_offers):
                coefficients[emergency_x[(scenario_id, index)]] -= (
                    item.offer.pack_size * item.offer.unit_price
                )
                coefficients[emergency_y[(scenario_id, index)]] -= item.offer.delivery_cost
            add(dict(coefficients), 0, np.inf)

    matrix = np.zeros((len(rows), len(variables.cost)))
    for row_index, coefficients in enumerate(rows):
        for column, value in coefficients.items():
            matrix[row_index, column] = value
    result = milp(
        c=np.asarray(variables.cost),
        integrality=np.asarray(variables.integrality),
        bounds=Bounds(variables.lower, variables.upper),
        constraints=LinearConstraint(matrix, lower, upper),
        options={"time_limit": 30},
    )
    statuses = {0: "OPTIMAL", 1: "LIMIT_REACHED", 2: "INFEASIBLE", 3: "UNBOUNDED"}
    status = statuses.get(result.status, "SOLVER_ERROR")
    orders: list[ProcurementDecisionLine] = []
    recourse: dict[str, list[ProcurementDecisionLine]] = {}
    if result.x is not None and result.status in {0, 1}:
        for index, item in enumerate(data.regular_offers):
            packs = round(result.x[regular_x[index]])
            if packs > 0:
                orders.append(_decision(item, packs))
        for scenario_id in data.scenario_ids:
            scenario_orders = []
            for index, item in enumerate(data.emergency_offers):
                packs = round(result.x[emergency_x[(scenario_id, index)]])
                if packs > 0:
                    scenario_orders.append(_decision(item, packs))
            if scenario_orders:
                recourse[scenario_id] = scenario_orders

    purchase_cost = sum(line.purchase_cost + line.delivery_cost for line in orders)
    first_stage_objective_cost = sum(
        (line.purchase_cost + line.delivery_cost) * (1 + profile.cash_penalty)
        for line in orders
    )
    expected_recourse_cost = 0.0
    scenario_costs: dict[str, float] = {}
    scenario_outcomes: dict[str, object] = {}
    if result.x is not None and result.status in {0, 1}:
        for scenario_id, weight in zip(
            data.scenario_ids, data.scenario_weights, strict=True
        ):
            recourse_cost = 0.0
            per_key: dict[str, dict[str, float | bool]] = {}
            for key in data.keys:
                assumption = data.assumptions.get(key)
                shortage_cost = profile.shortage_penalty + (
                    assumption.shortage_cost_per_unit if assumption else 0
                )
                holding_cost = profile.holding_penalty + (
                    assumption.holding_cost_per_unit_day if assumption else 0
                )
                key_shortage = 0.0
                key_demand = 0.0
                for day in data.dates:
                    shortage_quantity = result.x[
                        shortage[(scenario_id, key, day)]
                    ]
                    ending_quantity = result.x[inventory[(scenario_id, key, day)]]
                    key_shortage += shortage_quantity
                    key_demand += data.demand.get((scenario_id, key, day), 0)
                    recourse_cost += shortage_quantity * shortage_cost
                    recourse_cost += ending_quantity * (
                        holding_cost
                        + (
                            profile.waste_penalty
                            if day == data.dates[-1]
                            else 0
                        )
                    )
                label = f"{key[0]}|{key[1]}|{data.target_units[key]}"
                per_key[label] = {
                    "demand": float(key_demand),
                    "shortage": float(key_shortage),
                    "fill_rate": (
                        max(0.0, 1 - key_shortage / key_demand)
                        if key_demand > 0
                        else 1.0
                    ),
                    "stockout": bool(key_shortage > 1e-8),
                }
            emergency_cost = sum(
                line.purchase_cost + line.delivery_cost
                for line in recourse.get(scenario_id, [])
            )
            recourse_cost += emergency_cost
            expected_recourse_cost += float(weight) * recourse_cost
            scenario_costs[scenario_id] = first_stage_objective_cost + recourse_cost
            key_outcomes = list(per_key.values())
            single_key_outcome = key_outcomes[0] if len(key_outcomes) == 1 else None
            scenario_outcomes[scenario_id] = {
                # Deprecated physical scalar: it is available only when one
                # compatible inventory key exists in the scenario.
                "shortage": (
                    float(single_key_outcome["shortage"])
                    if single_key_outcome is not None
                    else None
                ),
                "fill_rate": float(
                    np.mean([float(item["fill_rate"]) for item in key_outcomes])
                ),
                "fill_rate_definition": "unweighted_mean_of_key_fill_rates",
                "stockout": any(bool(item["stockout"]) for item in key_outcomes),
                "per_key": per_key,
            }
    weights = [float(value) for value in data.scenario_weights]
    predicted_stockout = (
        sum(
            weight
            for scenario_id, weight in zip(data.scenario_ids, weights, strict=True)
            if bool(dict(scenario_outcomes[scenario_id]).get("stockout", False))
        )
        if scenario_outcomes
        else None
    )
    predicted_shortage_by_key: dict[str, float] = {}
    predicted_fill_by_key: dict[str, float] = {}
    for scenario_id, weight in zip(data.scenario_ids, weights, strict=True):
        outcome = dict(scenario_outcomes.get(scenario_id, {}))
        for label, raw_key_outcome in dict(outcome.get("per_key", {})).items():
            key_outcome = dict(raw_key_outcome)
            predicted_shortage_by_key[str(label)] = (
                predicted_shortage_by_key.get(str(label), 0)
                + weight * float(key_outcome.get("shortage", 0))
            )
            predicted_fill_by_key[str(label)] = (
                predicted_fill_by_key.get(str(label), 0)
                + weight * float(key_outcome.get("fill_rate", 1))
            )
    # This is exactly M4's safe package definition: scenario-weighted fill
    # within each key, then an unweighted mean across inventory keys.
    predicted_fill = (
        float(np.mean(list(predicted_fill_by_key.values())))
        if predicted_fill_by_key
        else None
    )
    cvar_value = (
        weighted_cvar(
            np.asarray([scenario_costs[item] for item in data.scenario_ids]),
            np.asarray(weights),
            profile.cvar_alpha,
        )
        if scenario_costs
        else None
    )
    return ProcurementPlan(
        plan_id=f"{request.request_id}-{profile.name.lower()}",
        strategy=profile.name,
        orders=orders,
        scenario_recourse_orders=recourse,
        purchase_cost=purchase_cost,
        expected_recourse_cost=expected_recourse_cost,
        objective_value=(float(result.fun) if result.fun is not None else None),
        solver_status=status,
        completed=False,
        provenance={
            "solver": "scipy.optimize.milp",
            "formulation": "chronological_two_stage_saa_cvar_chance_v2",
            "time_grain": "scenario_store_ingredient_day",
            "first_stage_non_anticipative": True,
            "cvar_alpha": profile.cvar_alpha,
            "cvar_weight": profile.cvar_weight,
            "estimated_cvar": cvar_value,
            "scenario_costs": scenario_costs,
            "scenario_outcomes": scenario_outcomes,
            "predicted_stockout_probability": predicted_stockout,
            "predicted_expected_fill_rate": predicted_fill,
            "predicted_expected_fill_rate_by_key": predicted_fill_by_key,
            "predicted_fill_rate_definition": (
                "mean_of_scenario_weighted_inventory_key_fill_rates"
            ),
            "minimum_expected_fill_rate_semantics": (
                "per_key_expected_scenario_fill_rate"
            ),
            "predicted_expected_shortage_by_key": predicted_shortage_by_key,
            "chance_big_m": "scenario_key_daily_demand",
            "exact_inventory_physics": False,
            "requires_m4_resimulation": True,
        },
        warnings=data.warnings,
    )
