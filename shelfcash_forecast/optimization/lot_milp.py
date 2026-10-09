"""Sparse, bounded MILP with sequential FEFO min() and greedy lost-sales physics.

Regular opportunities are shared decisions. No scenario oracle or adaptive recourse.
The exact inventory engine independently checks the returned integer decisions.
"""
from __future__ import annotations

import math
import time
from collections import defaultdict
from datetime import date, timedelta
from types import SimpleNamespace

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from shelfcash_forecast.bom.units import UnitConverter, normalize_unit
from shelfcash_forecast.inventory.adapters import normalize_cost_assumptions, canonical_demand_scenarios
from shelfcash_forecast.inventory.contracts import InventoryDemandLine, InventoryDemandScenario, InventoryLot
from shelfcash_forecast.inventory.fefo import fefo_sort_key, is_expired
from shelfcash_forecast.optimization.chronology import expiry_date, offer_arrival, planned_lot_id
from shelfcash_forecast.optimization.contracts import OptimizationRequest, ProcurementDecisionLine, ProcurementPlan, StrategyProfile
from shelfcash_forecast.exceptions import OptimizationNotAvailableError
from shelfcash_forecast.optimization.shipment_fees import shipment_group, allocate_delivery_fees

# An affine expression is (sparse coefficients, constant).
Expr = tuple[dict[int, float], float]


def add(*expressions: Expr) -> Expr:
    coefficients: dict[int, float] = defaultdict(float)
    for values, _ in expressions:
        for index, value in values.items():
            coefficients[index] += value
    return {i: v for i, v in coefficients.items() if v}, sum(c for _, c in expressions)


def scale(expression: Expr, factor: float) -> Expr:
    return {i: v * factor for i, v in expression[0].items()}, expression[1] * factor


def constant(value: float) -> Expr:
    return {}, float(value)


class SparseModel:
    def __init__(self):
        self.objective_constant = 0.0
        self.cost, self.lb, self.ub, self.integer = [], [], [], []
        self.rows, self.lo, self.hi = [], [], []

    def variable(self, upper=np.inf, integer=False) -> Expr:
        index = len(self.cost)
        self.cost.append(0.0); self.lb.append(0.0); self.ub.append(upper); self.integer.append(int(integer))
        return {index: 1.0}, 0.0

    def constrain(self, expression: Expr, low=-np.inf, high=np.inf):
        self.rows.append(expression[0]); self.lo.append(low-expression[1]); self.hi.append(high-expression[1])

    def objective(self, expression: Expr, factor: float):
        self.objective_constant += expression[1] * factor
        for index, value in expression[0].items():
            self.cost[index] += value * factor

    def minimum(self, stock: Expr, remaining_demand: Expr, stock_upper: float, demand_upper: float) -> Expr:
        if stock_upper == 0 or demand_upper == 0:
            return constant(0)
        if (not remaining_demand[0] and remaining_demand[1]==0) or (not stock[0] and stock[1]==0):
            return constant(0)
        if not stock[0] and not remaining_demand[0]:
            return constant(min(stock[1],remaining_demand[1]))
        if not stock[0] and stock[1] >= demand_upper:
            return remaining_demand
        if not remaining_demand[0] and remaining_demand[1] >= stock_upper:
            return stock
        consumed = self.variable(min(stock_upper, demand_upper))
        branch = self.variable(1, integer=True)
        self.constrain(add(consumed, scale(stock, -1)), high=0)
        self.constrain(add(consumed, scale(remaining_demand, -1)), high=0)
        # z=1 selects stock; z=0 selects demand. Bounds are physical quantities.
        self.constrain(add(consumed, scale(stock, -1), scale(branch, -stock_upper)), low=-stock_upper)
        self.constrain(add(consumed, scale(remaining_demand, -1), scale(branch, demand_upper)), low=0)
        return consumed

    def solve(self, seconds):
        row_idx, col_idx, values = [], [], []
        for r, row in enumerate(self.rows):
            for c, v in row.items():
                row_idx.append(r); col_idx.append(c); values.append(v)
        matrix = coo_matrix((values, (row_idx, col_idx)), shape=(len(self.rows),len(self.cost))).tocsc()
        try:
            result = milp(np.asarray(self.cost), integrality=np.asarray(self.integer),
                          bounds=Bounds(self.lb,self.ub), constraints=LinearConstraint(matrix,self.lo,self.hi),
                          options={"time_limit":seconds,"mip_rel_gap":0.0})
        except (ValueError, RuntimeError) as exc:
            result = SimpleNamespace(status=4,x=None,fun=None,message=f"{type(exc).__name__}: {exc}")
        return result, {"variables":len(self.cost),"constraints":len(self.rows),
                        "binaries_and_integers":sum(self.integer),"sparse_nnz":matrix.nnz,
                        "sparse_storage_bytes":matrix.data.nbytes+matrix.indices.nbytes+matrix.indptr.nbytes}


def mean_world(request: OptimizationRequest) -> InventoryDemandScenario:
    """A distinct, explicitly constructed expected-demand world for deterministic physics."""
    weights = [s.probability_weight for s in request.demand_scenarios]
    if any(w is None for w in weights):
        weights = [1 / len(weights)] * len(weights)
    quantities = defaultdict(float)
    existing={s.scenario_id for s in [*request.demand_scenarios,*request.evaluation_scenarios]}
    mean_id='__DETERMINISTIC_MEAN__'
    suffix=0
    while mean_id in existing:
        suffix+=1;mean_id=f'__DETERMINISTIC_MEAN__#{suffix}'
    for scenario, weight in zip(canonical_demand_scenarios(request.demand_scenarios, UnitConverter(request.unit_conversions)),weights,strict=True):
        for line in scenario.lines:
            quantities[(line.store_id,line.ingredient_id,line.target_date,line.unit)] += float(weight)*line.quantity
    return InventoryDemandScenario(scenario_id=mean_id,probability_weight=1.0,
        simulation_start_date=request.decision_date+timedelta(days=1), simulation_end_date=request.planning_end_date,
        lines=[InventoryDemandLine(scenario_id=mean_id, store_id=k[0],ingredient_id=k[1],
                                   target_date=k[2],unit=k[3],quantity=q) for k,q in sorted(quantities.items())],
        provenance={"derivation":"weighted_mean_of_optimization_worlds","not_independent_oos":True})


def solve_lot_procurement(request: OptimizationRequest, profile: StrategyProfile, *, stochastic: bool,
                          fixed_pack_counts: dict[str,int] | None = None,
                          aggregate_nonexpiring_candidate_lots: bool = False,
                          feasibility_first: bool = False) -> ProcurementPlan:
    started = time.monotonic()
    if any(o.emergency for o in request.supplier_offers):
        raise OptimizationNotAvailableError("Adaptive recourse policy is not implemented.",code="RECOURSE_POLICY_NOT_SUPPORTED")
    model = SparseModel(); model.variable(0); converter = UnitConverter(request.unit_conversions)
    request = request.model_copy(update={
        'demand_scenarios': canonical_demand_scenarios(request.demand_scenarios, converter),
        'evaluation_scenarios': canonical_demand_scenarios(request.evaluation_scenarios, converter)})
    units = {}
    for scenario in request.demand_scenarios:
        for line in scenario.lines:
            key = (line.store_id,line.ingredient_id); unit = normalize_unit(line.unit)
            if key in units and units[key] != unit:
                raise ValueError("INCONSISTENT_DEMAND_UNITS")
            units[key] = unit
    for row in [*request.initial_inventory,*request.existing_inbound,*request.supplier_offers]:
        units.setdefault((row.store_id,row.ingredient_id),converter.canonical_unit(row.ingredient_id,row.unit))
    assumptions = {(a.store_id,a.ingredient_id):a for a in normalize_cost_assumptions(request.cost_assumptions,units,converter)}
    keys = sorted(units)
    def factor(key, unit): return converter.conversion_factor(key[1],unit,units[key])
    horizon_demand = {key:max((sum(l.quantity for l in s.lines if (l.store_id,l.ingredient_id)==key)
                              for s in request.demand_scenarios),default=0) for key in keys}
    offers = []; packs = []; activations = []; lot_specs = []
    for lot in request.initial_inventory:
        key=(lot.store_id,lot.ingredient_id); q=lot.quantity_remaining*factor(key,lot.unit)
        normalized=InventoryLot.model_validate({**lot.model_dump(),"quantity_remaining":q,"unit":units[key]})
        lot_specs.append((normalized,request.decision_date,constant(q),q))
    for delivery in request.existing_inbound:
        if delivery.arrival_date <= request.decision_date:
            raise ValueError("INBOUND_BEFORE_TRANSITION_MUST_BE_IN_SNAPSHOT")
        key=(delivery.store_id,delivery.ingredient_id);q=delivery.quantity*factor(key,delivery.unit)
        lot=InventoryLot(lot_id=delivery.lot_id,store_id=key[0],ingredient_id=key[1],unit=units[key],
            quantity_remaining=q,received_date=delivery.arrival_date,expiry_date=delivery.expiry_date,source_type="inbound")
        lot_specs.append((lot,delivery.arrival_date,constant(q),q))
    purchase=constant(0);delivery_cost=constant(0)
    for offer in request.supplier_offers:
        arrival=offer_arrival(offer);key=(offer.store_id,offer.ingredient_id)
        if key not in units or not offer.available or offer.order_date<request.decision_date or arrival>request.planning_end_date:
            continue
        if offer.order_cutoff_date is not None and offer.order_date>offer.order_cutoff_date:
            continue
        if arrival<=request.decision_date:
            raise ValueError("ARRIVAL_NOT_AFTER_EOD_SNAPSHOT")
        if fixed_pack_counts is not None and fixed_pack_counts.get(offer.offer_id,0)==0:continue
        f=factor(key,offer.unit); pack=offer.pack_size*f
        # A dominance bound retains an optimal regular lost-sales solution: buying more
        # than all demand (rounded for MOQ/pack) cannot improve service or nonnegative cost.
        reserve=max((r.target for r in request.normalized_rules if r.semantics=='END_OF_DAY_MIN' and
                     (r.store_id,r.ingredient_id)==key and r.classification in {'HARD','SOFT'}),default=0)
        upper=math.ceil(max((horizon_demand[key]+reserve)/f,offer.minimum_order_quantity,offer.pack_size)/offer.pack_size)
        peak_caps=[r.target for r in request.normalized_rules if r.classification=='HARD' and
                   r.semantics=='RECEIVING_PEAK' and (r.store_id,r.ingredient_id)==key and r.active(arrival)]
        if peak_caps:upper=min(upper,math.floor(min(peak_caps)/pack+1e-12))
        if offer.maximum_order_quantity is not None:
            upper=min(upper,math.floor(offer.maximum_order_quantity/offer.pack_size+1e-12))
        minimum=max(1,math.ceil(offer.minimum_order_quantity/offer.pack_size-1e-12))
        if fixed_pack_counts is not None:
            count=fixed_pack_counts.get(offer.offer_id,0)
            if isinstance(count,bool) or not isinstance(count,int) or count<minimum or count>upper:
                raise ValueError('INVALID_FIXED_INTEGER_COMMITMENT')
            x=constant(count);y=constant(1)
        else:
            x=model.variable(upper,integer=True);y=model.variable(1,integer=True)
            model.constrain(add(x,scale(y,-upper)),high=0)
            model.constrain(add(x,scale(y,-minimum)),low=0)
        offers.append(offer);packs.append(x);activations.append(y)
        purchase=add(purchase,scale(x,offer.pack_size*offer.unit_price))
        lot=InventoryLot(lot_id=planned_lot_id(f"{request.request_id}-{profile.name.lower()}",offer.offer_id),
            store_id=key[0],ingredient_id=key[1],unit=units[key],quantity_remaining=0,
            received_date=arrival,expiry_date=expiry_date(arrival,offer.shelf_life_days),source_type="planned_inbound")
        lot_specs.append((lot,arrival,scale(x,pack),upper*pack))
    groups=defaultdict(list)
    for i,offer in enumerate(offers): groups[shipment_group(offer)].append(i)
    shipment_activations={}
    for group,indices in groups.items():
        fees={offers[i].delivery_cost for i in indices}
        if len(fees)!=1: raise ValueError('INCONSISTENT_SHIPMENT_FEE')
        g=constant(1) if fixed_pack_counts is not None else activations[indices[0]] if len(indices)==1 else model.variable(1,integer=True)
        shipment_activations[group]=g
        if len(indices)>1 and fixed_pack_counts is None:
            for i in indices:model.constrain(add(activations[i],scale(g,-1)),high=0)
            model.constrain(add(g,scale(add(*(activations[i] for i in indices)),-1)),high=0)
        delivery_cost=add(delivery_cost,scale(g,fees.pop()))
    for rule in request.normalized_rules:
        if rule.classification!='HARD':continue
        relevant=[i for i,o in enumerate(offers) if o.store_id==rule.store_id and o.ingredient_id==rule.ingredient_id and rule.active(offer_arrival(o))]
        if rule.semantics=='PURCHASE_COVER_DAYS':
            for day in sorted({offer_arrival(offers[i]) for i in relevant}):
                model.constrain(add(*(scale(packs[i],offers[i].pack_size) for i in relevant if offer_arrival(offers[i])==day)),
                    high=rule.target*rule.reference_daily_quantity)
        elif rule.semantics=='MIN_REMAINING_LIFE_AT_RECEIVING':
            for i in relevant:
                if offers[i].shelf_life_days is None or offers[i].shelf_life_days<rule.target:
                    model.constrain(activations[i],high=0)
    cash=add(purchase,delivery_cost)
    model.objective(cash,1+profile.cash_penalty)
    if request.budget is not None:model.constrain(cash,high=request.budget)
    for cap in request.supplier_constraints:
        scoped=[i for i,o in enumerate(offers) if o.supplier_id==cap.supplier_id and
                (cap.store_id is None or o.store_id==cap.store_id) and (cap.ingredient_id is None or o.ingredient_id==cap.ingredient_id)]
        if cap.maximum_total_cost is not None:
            scoped_groups={shipment_group(offers[i]) for i in scoped}
            if any(set(groups[g])-set(scoped) for g in scoped_groups):
                raise ValueError('PARTIAL_SHIPMENT_COST_CAP_NOT_SUPPORTED')
            model.constrain(add(add(*(scale(packs[i],offers[i].pack_size*offers[i].unit_price) for i in scoped)),
                add(*(scale(shipment_activations[g],offers[groups[g][0]].delivery_cost) for g in scoped_groups))),high=cap.maximum_total_cost)
        if cap.maximum_total_quantity is not None:
            expr=add(*(scale(packs[i],offers[i].pack_size*converter.conversion_factor(offers[i].ingredient_id,offers[i].unit,cap.unit)) for i in scoped))
            model.constrain(expr,high=cap.maximum_total_quantity)
    pool_receipts={};aggregation_evidence=[]
    if aggregate_nonexpiring_candidate_lots:
        # Candidate-generation projection only. Lots that cannot expire in this
        # horizon have identical aggregate holding/capacity/greedy-fill effects.
        # Earlier-expiry lots stay separate and retain FEFO priority. The final
        # fixed-commitment model ALWAYS retains the original individual lots.
        retained=[]
        for key in keys:
            scoped=[v for v in lot_specs if (v[0].store_id,v[0].ingredient_id)==key]
            safe=[v for v in scoped if v[0].expiry_date is None or
                  (v[0].expiry_date>=request.planning_end_date if request.inventory_policy.expiry_inclusive else v[0].expiry_date>request.planning_end_date)]
            retained.extend(v for v in scoped if v not in safe)
            if not safe:continue
            pid='__CANDIDATE_NONEXPIRING_POOL__|'+key[0]+'|'+key[1]
            initial=add(*(v[2] for v in safe if v[1]<=request.decision_date))
            upper=sum(v[3] for v in safe)
            active_caps=[]
            for offset in range(1,(request.planning_end_date-request.decision_date).days+1):
                day=request.decision_date+timedelta(days=offset)
                caps=[r.target for r in request.normalized_rules if r.classification=='HARD' and
                    r.semantics=='RECEIVING_PEAK' and (r.store_id,r.ingredient_id)==key and r.active(day)]
                active_caps.append(min(caps) if caps else None)
            if active_caps and all(v is not None for v in active_caps):upper=min(upper,max(active_caps))
            pool=InventoryLot(lot_id=pid,store_id=key[0],ingredient_id=key[1],unit=units[key],quantity_remaining=0,
                received_date=request.decision_date,expiry_date=request.planning_end_date if request.inventory_policy.expiry_inclusive else request.planning_end_date+timedelta(days=1))
            retained.append((pool,request.decision_date,initial,upper))
            pool_receipts[pid]=[v for v in safe if v[1]>request.decision_date]
            aggregation_evidence.append({'pool_id':pid,'original_lot_ids':[v[0].lot_id for v in safe],
                'proof':'No expiry within horizon; earlier-expiring lots remain separate; aggregate greedy fill/peak/ending/holding invariant; individual FEFO checked by final joint fixed model and exact simulator',
                'not_a_customer_inventory_lot':True})
        lot_specs=retained
    worlds=list(request.demand_scenarios)
    physics_worlds=worlds if stochastic else [mean_world(request)]
    if not stochastic:worlds=worlds+physics_worlds
    weights={s.scenario_id:float(s.probability_weight) if s.probability_weight is not None else 1/len(request.demand_scenarios)
             for s in request.demand_scenarios}
    days=[request.decision_date+timedelta(days=i) for i in range(1,(request.planning_end_date-request.decision_date).days+1)]
    eligible_lot_days=sum(sum(arrival<=d for d in days) for _,arrival,_,_ in lot_specs)
    estimate={"keys":len(keys),"lots":len(lot_specs),"days":len(days),"worlds":len(worlds),
        "eligible_lot_days":eligible_lot_days,
        "variables_upper_estimate":1+2*len(offers)+len(groups)+len(worlds)*(3*eligible_lot_days+len(keys)*len(days)+2*len(keys)+2+len(request.normalized_rules)*len(days)),
        "constraints_upper_estimate":4*len(offers)+len(groups)+len(worlds)*(7*eligible_lot_days+3*len(keys)*len(days)+3*len(keys)+2+len(request.normalized_rules)*len(days))}
    estimate["sparse_memory_estimate_bytes"]=estimate["constraints_upper_estimate"]*12*24
    estimate["memory_estimate_basis"]="heuristic 12 nnz/row; actual sparse storage reported after assembly"
    if estimate["variables_upper_estimate"]>request.limits.max_model_variables or estimate["constraints_upper_estimate"]>request.limits.max_model_constraints:
        return ProcurementPlan(plan_id=f"{request.request_id}-{profile.name.lower()}",strategy=profile.name,orders=[],
            purchase_cost=0,expected_recourse_cost=0,solver_status="LIMIT_REACHED",provenance={
                "termination":"MODEL_SIZE_LIMIT","dimensions_preflight":estimate,"not_infeasibility_proof":True})
    predicted={};shortages={};totals={};losses={};component_expressions={};daily_by_world={}
    for world in worlds:
        sid=world.scenario_id; daily={(l.store_id,l.ingredient_id,l.target_date):l.quantity for l in world.lines}
        daily_by_world[sid] = daily
        state={}; ledgers=[]; components={k:constant(0) for k in ["holding","shortage","expiry","waste","soft_rules"]}
        for day in days:
            for key in keys:
                demand=daily.get((*key,day),0.0);remaining=constant(demand)
                specs=[v for v in lot_specs if (v[0].store_id,v[0].ingredient_id)==key and v[1]<=day]
                beginning=add(*(state.get(v[0].lot_id,v[2] if v[1]<day else constant(0)) for v in specs))
                inbound=add(*(v[2] for v in specs if v[1]==day))
                inbound=add(inbound,*(v[2] for pool in specs for v in pool_receipts.get(pool[0].lot_id,[]) if v[1]==day))
                for lot,arrival,q,upper in specs:
                    if lot.lot_id not in state and arrival<=day:state[lot.lot_id]=q
                    if lot.lot_id in pool_receipts:
                        state[lot.lot_id]=add(state[lot.lot_id],*(v[2] for v in pool_receipts[lot.lot_id] if v[1]==day))
                peak=add(*(state[v[0].lot_id] for v in specs));expired=constant(0);consumption={}
                assumption=assumptions.get(key)
                active_capacity=assumption.capacity_on(day) if assumption else None
                if active_capacity is not None:model.constrain(peak,high=active_capacity)
                for lot,arrival,q,upper in sorted(specs,key=lambda v:fefo_sort_key(v[0])):
                    stock=state[lot.lot_id]
                    if is_expired(lot,day,request.inventory_policy):
                        expired=add(expired,stock);state[lot.lot_id]=constant(0);continue
                    consumed=model.minimum(stock,remaining,upper,demand)
                    after=add(stock,scale(consumed,-1))
                    model.constrain(after,low=0,high=upper)
                    state[lot.lot_id]=after;remaining=add(remaining,scale(consumed,-1));consumption[lot.lot_id]=consumed
                ending=add(*(state[v[0].lot_id] for v in specs))
                model.constrain(remaining,low=0,high=demand)
                shortages[(sid,key,day)]=remaining
                rates={"holding":assumption.holding_cost_per_unit_day if assumption else 0,
                       "shortage":assumption.shortage_cost_per_unit if assumption else 0,
                       "expiry":assumption.expired_cost_per_unit if assumption else 0}
                for name,quantity in [("holding",ending),("shortage",remaining),("expiry",expired)]:components[name]=add(components[name],scale(quantity,rates[name]))
                ledgers.append({"store_id":key[0],"ingredient_id":key[1],"unit":units[key],"date":str(day),"demand":demand,
                                "beginning":beginning,"inbound":inbound,"expired":expired,"ending":ending,"maximum":peak,
                                "shortage":remaining,"consumption":consumption})
            for rule in request.normalized_rules:
                if not rule.active(day):continue
                scoped=[r for r in ledgers if r['date']==str(day) and r['store_id']==rule.store_id and
                        (rule.ingredient_id is None or r['ingredient_id']==rule.ingredient_id)]
                if rule.semantics=='GLOBAL_RECEIVING_PEAK':
                    coefficients={c.ingredient_id:c for c in rule.occupancy}
                    if any(r['ingredient_id'] not in coefficients or coefficients[r['ingredient_id']].base_unit!=r['unit'] for r in scoped):
                        raise ValueError('GLOBAL_OCCUPANCY_COVERAGE_OR_UNIT_MISMATCH')
                    quantity=add(*(scale(r['maximum'],coefficients[r['ingredient_id']].liters_per_base_unit) for r in scoped))
                elif rule.semantics in {'RECEIVING_PEAK','END_OF_DAY_MAX','END_OF_DAY_MIN'}:
                    quantity=add(*(r['maximum'] if rule.semantics=='RECEIVING_PEAK' else r['ending'] for r in scoped))
                else:continue
                if rule.classification=='HARD':
                    model.constrain(quantity,low=rule.target if rule.semantics=='END_OF_DAY_MIN' else -np.inf,
                                    high=np.inf if rule.semantics=='END_OF_DAY_MIN' else rule.target)
                elif rule.classification=='SOFT':
                    deficiency=model.variable(rule.target)
                    model.constrain(add(quantity,deficiency),low=rule.target)
                    components['soft_rules']=add(components['soft_rules'],scale(deficiency,rule.penalty_currency_per_unit))
        predicted[sid]=ledgers;component_expressions[sid]=components
        loss=add(scale(components["holding"],profile.holding_penalty),scale(components["shortage"],profile.shortage_penalty),
                 scale(components["expiry"],profile.waste_penalty),components['soft_rules'])
        losses[sid]=add(cash,loss)
        objective_weight=weights.get(sid,0) if stochastic else int(sid=="__DETERMINISTIC_MEAN__")
        model.objective(loss,objective_weight)
        for key in keys:
            total=sum(daily.get((*key,d),0) for d in days);totals[(sid,key)]=total
            shortage=add(*(shortages[(sid,key,d)] for d in days))
            model.constrain(shortage,high=(1-profile.minimum_acceptable_fill_rate)*total)
    for rule in request.normalized_rules:
        if rule.semantics!='PER_KEY_EXPECTED_FILL' or rule.classification!='HARD':continue
        for key in keys:
            if key[0]!=rule.store_id or (rule.ingredient_id is not None and key[1]!=rule.ingredient_id):continue
            expr=constant(0)
            for sid,w in weights.items():
                active=[d for d in days if rule.active(d)]
                total=sum(daily_by_world[sid].get((*key,d),0) for d in active)
                if total:expr=add(expr,scale(add(*(shortages[(sid,key,d)] for d in active)),w/total))
            model.constrain(expr,high=1-rule.target)
    probabilistic=all(s.probability_weight is not None for s in request.demand_scenarios)
    if probabilistic:
        stockout_vars={s.scenario_id:model.variable(1,integer=True) for s in request.demand_scenarios}
        for sid,z in stockout_vars.items():
            for key in keys:
                for day in days:model.constrain(add(shortages[(sid,key,day)],scale(z,-daily_by_world[sid].get((*key,day),0))),high=0)
        ceiling=min(profile.maximum_acceptable_stockout_probability,profile.maximum_stockout_probability if profile.maximum_stockout_probability is not None else 1)
        model.constrain(add(*(scale(stockout_vars[sid],w) for sid,w in weights.items())),high=ceiling)
        for key in keys:
            if profile.minimum_expected_fill_rate is not None:
                expr=add(*(scale(add(*(shortages[(sid,key,d)] for d in days)),w/totals[(sid,key)])
                           for sid,w in weights.items() if totals[(sid,key)]>0))
                model.constrain(expr,high=1-profile.minimum_expected_fill_rate)
            if profile.minimum_fill_rate is not None and profile.required_fill_rate_probability is not None:
                violation={sid:model.variable(1,integer=True) for sid in weights}
                for sid,v in violation.items():
                    total=totals[(sid,key)];allowed=(1-profile.minimum_fill_rate)*total
                    model.constrain(add(add(*(shortages[(sid,key,d)] for d in days)),scale(v,-(total-allowed))),high=allowed)
                model.constrain(add(*(scale(violation[sid],w) for sid,w in weights.items())),high=1-profile.required_fill_rate_probability)
    if probabilistic and profile.cvar_weight>0:
        eta=model.variable();model.objective(eta,profile.cvar_weight)
        for sid,w in weights.items():
            tail=model.variable();model.constrain(add(tail,eta,scale(losses[sid],-1)),low=0)
            model.objective(tail,profile.cvar_weight*w/(1-profile.cvar_alpha))
    solve_seconds = min(request.limits.per_solve_seconds, request.limits.total_seconds - (time.monotonic()-started))
    if solve_seconds <= 0:
        return ProcurementPlan(plan_id=f"{request.request_id}-{profile.name.lower()}",strategy=profile.name,
            orders=[],purchase_cost=0,expected_recourse_cost=0,solver_status="LIMIT_REACHED",
            provenance={"termination":"MODEL_BUILD_TOTAL_TIME_LIMIT","elapsed_seconds":time.monotonic()-started})
    if feasibility_first:
        # Preserve all physics and hard/service constraints. Objective weights
        # cannot prevent discovery of a feasible integer incumbent in this phase.
        model.cost = [0.0] * len(model.cost)
        model.objective_constant = 0.0
    result,dimensions=model.solve(solve_seconds)
    status={0:"OPTIMAL",1:"LIMIT_REACHED",2:"INFEASIBLE",3:"UNBOUNDED"}.get(result.status,"SOLVER_ERROR")
    orders=[]
    def value(expr):return expr[1]+sum(c*result.x[i] for i,c in expr[0].items())
    evidence={};cost_breakdown={}
    if result.x is not None and result.status in {0,1}:
        for offer,x in zip(offers,packs,strict=True):
            count=round(value(x))
            if count:
                q=count*offer.pack_size
                orders.append(ProcurementDecisionLine(offer_id=offer.offer_id,supplier_id=offer.supplier_id,
                    store_id=offer.store_id,ingredient_id=offer.ingredient_id,unit=offer.unit,order_date=offer.order_date,
                    arrival_date=offer_arrival(offer),pack_count=count,pack_size=offer.pack_size,order_quantity=q,
                    unit_price=offer.unit_price,purchase_cost=q*offer.unit_price,delivery_cost=offer.delivery_cost,
                    shelf_life_days=offer.shelf_life_days))
        allocations=allocate_delivery_fees(orders,{o.offer_id:o for o in offers})
        orders=[ProcurementDecisionLine.model_validate({**o.model_dump(),'delivery_cost':allocations[o.offer_id]}) for o in orders]
        for sid,rows in predicted.items():
            evidence[sid]=[{k:({lot:value(expr) for lot,expr in v.items()} if k=="consumption" else value(v) if isinstance(v,tuple) else v)
                            for k,v in row.items()} for row in rows]
        cost_breakdown={sid:{k:value(v) for k,v in comps.items()} for sid,comps in component_expressions.items()}
    cost=sum(o.purchase_cost+o.delivery_cost for o in orders)
    return ProcurementPlan(plan_id=f"{request.request_id}-{profile.name.lower()}",strategy=profile.name,orders=orders,
        purchase_cost=cost,expected_recourse_cost=0,objective_value=float(result.fun)+model.objective_constant if result.fun is not None else None,
        solver_status=status,provenance={"solver":"scipy.optimize.milp","formulation":"sequential_fefo_greedy_lost_sales_v2",
            "exact_inventory_physics":True,"requires_m4_resimulation":True,"mode":"stochastic" if stochastic else "deterministic",
            "physics_scenario_ids":[s.scenario_id for s in physics_worlds],"predicted_daily_ledgers":evidence,
            'objective_world_weights':weights,
            "cost_components_by_world":cost_breakdown,"purchase_plus_delivery":cost,"currency":request.currency,
            'planning_binding':request.planning_binding,'planning_mode':request.planning_mode,
            'shipment_count':len({shipment_group(next(o for o in offers if o.offer_id==l.offer_id)) for l in orders}),
            'optimization_scope':'FIXED_COMMITMENTS_CERTIFICATION_NOT_GLOBAL_COST_OPTIMUM' if fixed_pack_counts is not None else 'JOINT_DECLARED_OPPORTUNITY_MILP',
            'candidate_nonexpiring_lot_aggregation':aggregation_evidence,
            'individual_lot_fefo_authority':not aggregate_nonexpiring_candidate_lots,
            'has_integer_incumbent':result.x is not None and result.status in {0,1},
            'search_phase':'FEASIBILITY_FIRST' if feasibility_first else 'PREFERENCE_OPTIMIZATION',
            'mip_gap':getattr(result,'mip_gap',None),
            'mip_dual_bound':getattr(result,'mip_dual_bound',None),
            'global_optimality_claimed':result.status == 0 and fixed_pack_counts is None and not feasibility_first and not request.evaluation_scenarios,
            "cost_coverage":"COMPLETE" if set(assumptions)==set(keys) else "MISSING_CONSEQUENCE_COSTS",
            "bound_semantics":"regular_lost_sales_dominance_bound_no_terminal_minimum_stock",
            "infeasibility_scope":"declared_input_opportunities_and_profile_optimization_worlds_only",
            "dimensions":{**dimensions,"keys":len(keys),"lots":len(lot_specs),"opportunities":len(offers),"worlds":len(worlds),"days":len(days)},
            "dimensions_preflight":estimate,
            "elapsed_seconds":time.monotonic()-started,"solver_message":result.message},warnings=[])


def daily_demand(request, sid, key, day):
    return sum(l.quantity for s in request.demand_scenarios if s.scenario_id==sid for l in s.lines
               if (l.store_id,l.ingredient_id)==key and l.target_date==day)
