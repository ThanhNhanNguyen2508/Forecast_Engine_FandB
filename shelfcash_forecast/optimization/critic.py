# Đây là file quyết định:

# PASS
# hay
# FAIL

# Critic không tìm plan mới.

# Nó chỉ xét:

# “Candidate hiện tại có đủ đáng tin để chấp nhận không?”

# Code critic thực sự bắt đầu bằng vali
from __future__ import annotations

from typing import Any
import math

from shelfcash_forecast.inventory.contracts import InventorySimulationPackage
from shelfcash_forecast.optimization.constraints import validate_plan_constraints
from shelfcash_forecast.optimization.contracts import (
    CriticResult,
    OptimizationRequest,
    ProcurementPlan,
    StrategyProfile,
)


def _exact_service_floors(
    simulation: InventorySimulationPackage,
) -> tuple[float, float | None, bool]:
    key_fill_rates = [
        item.fill_rate
        for result in simulation.results
        for item in result.summary.by_key
    ]
    minimum_fill = min(key_fill_rates, default=1.0)
    any_design_stockout = any(
        item.shortage_quantity > 1e-8
        for result in simulation.results
        for item in result.summary.by_key
    )
    probability = (
        simulation.risk_metrics.any_stockout_probability
        if simulation.risk_metrics is not None
        else None
    )
    return minimum_fill, probability, any_design_stockout


def _model_mismatch(
    plan: ProcurementPlan,
    simulation: InventorySimulationPackage,
    profile: StrategyProfile,
) -> tuple[bool, dict[str, Any]]:
    predicted_fill = plan.provenance.get("predicted_expected_fill_rate")
    predicted_stockout = plan.provenance.get("predicted_stockout_probability")
    metrics = simulation.risk_metrics
    if predicted_fill is None or predicted_stockout is None or metrics is None:
        return False, {"evaluated": False}
    actual_fill = metrics.mean_key_fill_rate
    actual_stockout = metrics.any_stockout_probability
    actual_shortage_by_key = {
        f"{item.store_id}|{item.ingredient_id}|{item.unit}": item.expected_shortage
        for item in metrics.by_key
    }
    actual_fill_by_key = {
        f"{item.store_id}|{item.ingredient_id}|{item.unit}": item.expected_fill_rate
        for item in metrics.by_key
    }
    predicted_fill_by_key = {
        str(key): float(value)
        for key, value in dict(
            plan.provenance.get("predicted_expected_fill_rate_by_key", {})
        ).items()
    }
    fill_gap = float(predicted_fill) - actual_fill
    key_fill_gaps = (
        {
            key: predicted_fill_by_key[key] - actual_fill_by_key[key]
            for key in actual_fill_by_key
        }
        if set(predicted_fill_by_key) == set(actual_fill_by_key)
        else {}
    )
    stockout_gap = actual_stockout - float(predicted_stockout)
    mismatch = (
        (
            max(key_fill_gaps.values(), default=fill_gap)
            > profile.maximum_fill_rate_model_gap + 1e-9
        )
        or stockout_gap
        > profile.maximum_stockout_probability_model_gap + 1e-9
    )
    return mismatch, {
        "evaluated": True,
        "predicted_fill_rate": float(predicted_fill),
        "simulated_expected_fill_rate": actual_fill,
        "fill_rate_gap": fill_gap,
        "fill_rate_definition": "mean_of_scenario_weighted_inventory_key_fill_rates",
        "predicted_expected_fill_rate_by_key": predicted_fill_by_key,
        "simulated_expected_fill_rate_by_key": actual_fill_by_key,
        "fill_rate_gap_by_key": key_fill_gaps,
        "predicted_stockout_probability": float(predicted_stockout),
        "simulated_stockout_probability": actual_stockout,
        "stockout_probability_gap": stockout_gap,
        "predicted_expected_shortage_by_key": plan.provenance.get(
            "predicted_expected_shortage_by_key", {}
        ),
        "simulated_expected_shortage_by_key": actual_shortage_by_key,
        "predicted_scenario_outcomes": plan.provenance.get("scenario_outcomes", {}),
        "simulated_per_key": [
            item.model_dump(mode="json") for item in metrics.by_key
        ],
    }

# 30. Critic layer 1 — kiểm hard constraints

# Đầu tiên:

# validate_plan_constraints(...)

# tức toàn bộ:

# offer identity
# MOQ
# pack
# budget
# supplier caps
# lead time
# availability
# cutoff
# unit

# Nếu fail:

# hard violatio
# 31. Critic layer 2 — solver phải thật sự OPTIMAL

# Code:

# if plan.solver_status != "OPTIMAL":

# → violation.

# Điểm này đáng chú ý:

# Stochastic code có thể extract candidate khi:

# LIMIT_REACHED

# nhưng critic sẽ không approve nó.

# Tức:

# LIMIT_REACHED candidate
# → có thể dùng để inspect
# → nhưng không được final recommend
# 32. Critic layer 3 — unknown constraints

# Nếu caller nói:

# “Còn một constraint business chưa encode”

# qua:

# request.unknown_constraints

# critic reject.

# Ý nghĩa:

# Không giả vờ plan an toàn nếu hệ thống biết còn luật chưa mô hình hóa.

# 33. Critic layer 4 — expiry

# Nếu policy:

# unknown_expiry = reject

# mà một order:

# shelf_life_days = None

# →:

# UNKNOWN_EXPIRY

# → fail.

# 34. Critic layer 5 — M4 simulation phải chạy thành công

# Nếu M4 lỗi:

# M4_SIMULATION_FAILED

# Thì không được nói:

# “MILP optimal nên chắc ổn.”

# Plan bị fail.

# Đây chính là ý tưởng architecture cốt lõi.

# 35. Critic layer 6 — accounting

# M4 có accounting invariant kiểu:

# beginning
# + inbound
# =
# consumed
# + waste
# + ending
# ...

# Nếu:

# accounting_valid=False

# →:

# M4_ACCOUNTING_INVALID

# → reject.

# 36. Critic layer 7 — capacity

# Ví dụ kho chứa tối đa:

# 100kg Chicken

# solver aggregate có thể nghĩ plan ổn, nhưng M4 exact run phát hiện tại D2:

# physical inventory = 120kg

# →:

# CAPACITY_CONSEQUENCE

# → reject.

# 37. Critic layer 8 — universal safety floor

# Đây rất quan trọng.

# Ngay cả khi stochastic profile không đặt constraint mạnh, critic vẫn có:

# minimum_acceptable_fill_rate
# maximum_acceptable_stockout_probability

# Ví dụ profile nói:

# minimum acceptable fill = 70%
# max acceptable stockout probability = 30%

# M4 exact chạy ra:

# một scenario/key chỉ fill 45%

# → fail.

# Code _exact_service_floors() lấy minimum fill rate qua từng scenario/key, chứ không chỉ nhìn average đẹp.

# 38. Critic layer 9 — service constraints

# Giả sử stochastic profile yêu cầu:

# expected fill Chicken >= 95%

# Solver aggregate predicted:

# 96%

# Nhưng M4 exact:

# 92%

# →:

# SERVICE_LEVEL_REQUIREMENT

# → fail.

# Đây chính là “solver tự dự đoán chưa đủ”.

# 39. Critic layer 10 — stockout risk

# Profile:

# maximum_stockout_probability = 10%

# Solver predicted:

# 8%

# Nhưng M4 exact:

# 14%

# →:

# RISK_CONSTRAINT_VIOLATION

# → fail.

def _lot_model_mismatch(plan, physics, request, profile):
    """Compare solver states with exact states on the SAME worlds, never the superset."""
    predictions = plan.provenance.get("predicted_daily_ledgers", {})
    ids = set(plan.provenance.get("physics_scenario_ids", []))
    if plan.solver_status not in {"OPTIMAL", "LIMIT_REACHED"} or not plan.provenance.get('predicted_daily_ledgers'):
        return False, {"evaluated": False, "reason": "NO_INCUMBENT_MODEL_STATES"}
    if physics is None or {s.scenario_id for s in physics.results} != ids or not ids <= set(predictions):
        return True, {"evaluated": False, "reason": "PHYSICS_SCOPE_MISSING"}
    fields = {"beginning":"beginning_quantity", "inbound":"inbound_quantity", "expired":"expired_quantity",
              "ending":"ending_quantity", "maximum":"maximum_quantity", "shortage":"shortage_quantity"}
    gaps=[];maximum=0.0
    for result in physics.results:
        traced = {}
        for trace in result.consumption_traces:
            key=(trace.store_id,trace.ingredient_id,str(trace.simulation_date),trace.unit,trace.lot_id)
            traced[key]=traced.get(key,0)+trace.quantity
        actual={(l.store_id,l.ingredient_id,str(l.simulation_date),l.unit):l for l in result.daily_ledgers}
        predicted={(l["store_id"],l["ingredient_id"],l["date"],l["unit"]):l for l in predictions[result.scenario_id]}
        if set(actual)!=set(predicted):
            gaps.append({"scenario_id":result.scenario_id,"reason":"LEDGER_IDENTITY_MISMATCH"});continue
        for key,row in predicted.items():
            if request.inventory_policy.trace_retention == "full":
                expected_allocations={(*key,lot):quantity for lot,quantity in row["consumption"].items()
                                      if quantity > request.inventory_policy.accounting_tolerance}
                actual_allocations={k:q for k,q in traced.items() if k[:4]==key}
                if set(expected_allocations)!=set(actual_allocations) or any(
                    abs(q-actual_allocations.get(k,0))>request.inventory_policy.accounting_tolerance
                    for k,q in expected_allocations.items()):
                    gaps.append({"scenario_id":result.scenario_id,"key":key,"reason":"FEFO_ALLOCATION_MISMATCH"})
            for p,a in fields.items():
                gap=abs(row[p]-getattr(actual[key],a));maximum=max(maximum,gap)
                if gap>request.inventory_policy.accounting_tolerance:
                    gaps.append({"scenario_id":result.scenario_id,"key":key,"field":p,"absolute_gap":gap})
    # Preserve the original risk/fill model-gap guards on the comparable physics
    # distribution. A full-pool sampling shift is evaluated separately.
    metrics_gap = {}
    if physics.risk_metrics is not None:
        weights={r.scenario_id:r.probability_weight for r in physics.results}
        fill_by_key={};stockout=0
        for sid in ids:
            totals={};shortages={}
            for row in predictions[sid]:
                key=f"{row['store_id']}|{row['ingredient_id']}|{row['unit']}"
                totals[key]=totals.get(key,0)+row['demand']
                shortages[key]=shortages.get(key,0)+row['shortage']
            for key,total in totals.items():
                fill_by_key[key]=fill_by_key.get(key,0)+weights[sid]*(1-shortages[key]/total if total else 1)
            if any(q>1e-8 for q in shortages.values()):stockout+=weights[sid]
        comparable=ProcurementPlan.model_validate({**plan.model_dump(),"provenance":{**plan.provenance,
            "predicted_expected_fill_rate":sum(fill_by_key.values())/len(fill_by_key) if fill_by_key else 1,
            "predicted_expected_fill_rate_by_key":fill_by_key,"predicted_stockout_probability":stockout}})
        risk_mismatch,metrics_gap=_model_mismatch(comparable,physics,profile)
        if risk_mismatch:gaps.append({"reason":"SAME_SCOPE_RISK_MODEL_GAP"})
    return bool(gaps), {"evaluated":True,"risk_model_gap":metrics_gap,"same_worlds":sorted(ids),"maximum_quantity_gap":maximum,
                        "accounting_tolerance":request.inventory_policy.accounting_tolerance,"violations":gaps[:100],
                        "violation_count":len(gaps),"scope":"solver_physics_worlds_not_full_pool_sampling_shift"}


def critique_procurement_plan(
    plan: ProcurementPlan,
    request: OptimizationRequest,
    profile: StrategyProfile,
    simulation: InventorySimulationPackage | None,
    *,
    stress_simulation: InventorySimulationPackage | None = None,
    simulation_error: str | None = None,
    physics_simulation: InventorySimulationPackage | None = None,
) -> CriticResult:
    violations, checks = validate_plan_constraints(
        plan,
        request.supplier_offers,
        request.supplier_constraints,
        budget=request.budget,
        unit_conversions=request.unit_conversions,
    )
    warnings = list(plan.warnings)
    details: dict[str, Any] = {}
    if plan.solver_status != "OPTIMAL" and not (plan.solver_status == 'LIMIT_REACHED' and plan.provenance.get('has_integer_incumbent')):
        violations.append(f"SOLVER_STATUS:{plan.solver_status}")
    elif plan.solver_status == 'LIMIT_REACHED':
        warnings.append('FEASIBLE_INCUMBENT_OPTIMALITY_NOT_PROVEN')
    cost_keys = {(a.store_id, a.ingredient_id) for a in request.cost_assumptions}
    demand_keys = {(line.store_id, line.ingredient_id) for s in request.demand_scenarios for line in s.lines}
    known_pending = {"DEMO_CONSEQUENCE_COSTS_NOT_APPROVED"} if (
        request.environment in {"DEMO", "SYNTHETIC"} and demand_keys <= cost_keys
    ) else set()
    if request.unknown_constraints:
        violations.extend(
            f"UNKNOWN_CONSTRAINT:{name}" for name in request.unknown_constraints if name not in known_pending
        )
    details["business_pending"] = sorted(set(request.business_issues) | (set(request.unknown_constraints) & known_pending))
    selected = [*plan.orders]
    for lines in plan.scenario_recourse_orders.values():
        selected.extend(lines)
    if (
        request.inventory_policy.unknown_expiry == "reject"
        and any(line.shelf_life_days is None for line in selected)
    ):
        violations.append("UNKNOWN_EXPIRY")
    if simulation is None:
        violations.append("M4_SIMULATION_FAILED")
        if simulation_error:
            warnings.append(simulation_error)
        checks.update(
            {
                "m4_accounting": False,
                "capacity": False,
                "service_level": False,
                "risk": False,
                "exact_service_floor": False,
                "candidate_model_match": False,
            }
        )
    else:
        from shelfcash_forecast.optimization.business_rules import evaluate_rules
        rules=evaluate_rules(request.normalized_rules,simulation,plan.orders)
        details['business_rule_evaluation']=rules
        checks['profile_rules']=all(r['status']!='FAIL' for r in rules)
        violations.extend('BUSINESS_RULE:'+r['rule_id'] for r in rules if r['status']=='FAIL')
        checks['profile_binding']=plan.provenance.get('planning_binding',{})==request.planning_binding
        if not checks['profile_binding']:violations.append('STALE_PLAN_PROFILE_BINDING')
        expected = {s.scenario_id: s.probability_weight for s in (request.evaluation_scenarios or request.demand_scenarios)}
        actual = {s.scenario_id: s.probability_weight for s in simulation.results}
        coverage_valid = set(expected) == set(actual) and all(
            (w is None and actual[sid] is None) or
            (w is not None and actual[sid] is not None and math.isclose(w, actual[sid], abs_tol=1e-9, rel_tol=1e-9))
            for sid, w in expected.items())
        checks["evaluation_coverage"] = coverage_valid
        if not coverage_valid:
            violations.append("EVALUATION_SCENARIO_COVERAGE_MISMATCH")
        accounting_valid = all(result.accounting_valid for result in simulation.results)
        checks["m4_accounting"] = accounting_valid
        if not accounting_valid:
            violations.append("M4_ACCOUNTING_INVALID")

        required_caps = {(a.store_id,a.ingredient_id) for a in request.cost_assumptions if a.capacity_quantity is not None}
        capacity_evaluated = all(required_caps <= {(k.store_id,k.ingredient_id) for k in result.summary.by_key
                                                  if k.capacity_violation_quantity is not None} for result in simulation.results)
        details["capacity_coverage"] = {"applicable_keys": sorted(required_caps),
                                        "rule_mapping": request.rule_coverage,
                                        "all_applicable_evaluated": capacity_evaluated}
        capacity_valid = all(
            (item.capacity_violation_quantity or 0) <= 1e-9
            for result in simulation.results
            for item in result.summary.by_key
        )
        checks["capacity"] = capacity_valid and capacity_evaluated
        if not capacity_evaluated:
            warnings.append("CAPACITY_NOT_EVALUATED")
        if not capacity_valid:
            violations.append("CAPACITY_CONSEQUENCE")

        metrics = simulation.risk_metrics
        minimum_exact_fill, exact_stockout, design_stockout = _exact_service_floors(
            simulation
        )
        details["exact_simulation"] = {
            "minimum_key_scenario_fill_rate": minimum_exact_fill,
            "any_stockout_probability": exact_stockout,
            "unweighted_design_scenario_stockout_observed": design_stockout,
        }
        floor_valid = (
            minimum_exact_fill + 1e-9
            >= profile.minimum_acceptable_fill_rate
        )
        if exact_stockout is not None:
            floor_valid = floor_valid and (
                exact_stockout
                <= profile.maximum_acceptable_stockout_probability + 1e-9
            )
        checks["exact_service_floor"] = floor_valid
        if not floor_valid:
            violations.append("EXACT_SIMULATION_SAFETY_FLOOR")

        service_valid = True
        if profile.minimum_expected_fill_rate is not None:
            service_valid = metrics is not None and all(
                item.expected_fill_rate + 1e-9
                >= profile.minimum_expected_fill_rate
                for item in metrics.by_key
            )
        if (
            profile.minimum_fill_rate is not None
            and profile.required_fill_rate_probability is not None
        ):
            if metrics is None:
                service_valid = False
                warnings.append("UNWEIGHTED_SERVICE_PROBABILITY_NOT_EVALUATED")
            else:
                weights = [float(result.probability_weight) for result in simulation.results]
                probabilities_by_key: dict[str, float] = {}
                for key_metric in metrics.by_key:
                    inventory_key = (
                        key_metric.store_id,
                        key_metric.ingredient_id,
                        key_metric.unit,
                    )
                    probability = 0.0
                    for result, weight in zip(
                        simulation.results, weights, strict=True
                    ):
                        summary_by_key = {
                            (item.store_id, item.ingredient_id, item.unit): item
                            for item in result.summary.by_key
                        }
                        if (
                            summary_by_key[inventory_key].fill_rate + 1e-9
                            >= float(profile.minimum_fill_rate)
                        ):
                            probability += weight
                    label = "|".join(inventory_key)
                    probabilities_by_key[label] = probability
                service_valid = service_valid and all(
                    probability + 1e-9
                    >= profile.required_fill_rate_probability
                    for probability in probabilities_by_key.values()
                )
                details["fill_rate_threshold_probability_by_key"] = (
                    probabilities_by_key
                )
        checks["service_level"] = service_valid
        if not service_valid:
            violations.append("SERVICE_LEVEL_REQUIREMENT")

        risk_valid = True
        if profile.maximum_stockout_probability is not None:
            risk_valid = metrics is not None and (
                metrics.any_stockout_probability
                <= profile.maximum_stockout_probability + 1e-9
            )
            if metrics is None:
                warnings.append("UNWEIGHTED_STOCKOUT_PROBABILITY_NOT_EVALUATED")
        checks["risk"] = risk_valid
        if not risk_valid:
            violations.append("RISK_CONSTRAINT_VIOLATION")

        if plan.provenance.get("formulation") == "sequential_fefo_greedy_lost_sales_v2":
            mismatch, mismatch_details = _lot_model_mismatch(plan, physics_simulation, request, profile)
        else:
            mismatch, mismatch_details = _model_mismatch(plan, simulation, profile)
# Critic còn hỏi:

# “Model approximation trong stochastic có đang quá lạc quan so với exact M4 không?”

# Ví dụ stochastic predicted:

# Expected fill = 97%
# Stockout P = 5%

# M4 exact:

# Expected fill = 89%
# Stockout P = 15%

# Ta có:

# fill optimistic gap = 8 percentage points
# stockout optimistic gap = 10 percentage points

# Nếu profile cho tolerance:

# max fill gap = 5%
# max stockout gap = 5%

# →:

# CANDIDATE_MODEL_MISMATCH

# → reject.

# Code thực sự so predicted metrics lưu trong plan.provenance với exact risk metrics từ M4, kể cả fill rate theo từng inventory key.
        details["candidate_model_mismatch"] = mismatch_details
        checks["candidate_model_match"] = not mismatch
        if mismatch:
            violations.append("CANDIDATE_MODEL_MISMATCH")

    if stress_simulation is not None:
        if not all(result.accounting_valid for result in stress_simulation.results):
            violations.append("STRESS_ACCOUNTING_INVALID")
        if any(
            item.shortage_quantity > 0
            for result in stress_simulation.results
            for item in result.summary.by_key
        ):
            warnings.append("STRESS_SHORTAGE_OBSERVED")
        if any(
            (item.capacity_violation_quantity or 0) > 0
            for result in stress_simulation.results
            for item in result.summary.by_key
        ):
            warnings.append("STRESS_CAPACITY_VIOLATION")
    unique = sorted(set(violations))
    return CriticResult(
        passed=not unique,
        hard_violations=unique,
        warnings=sorted(set(warnings)),
        checks=checks,
        details=details,
    )
