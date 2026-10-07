"""Bounded real per-key MILP candidate generation + joint fixed-commitment MILP.

This is a restricted search. Every final hard constraint remains in the shared
lot model and exact critic. OPTIMAL certification is not a global cost optimum.
"""
from __future__ import annotations
import time
import logging
from shelfcash_forecast.optimization.contracts import OptimizationRequest,StrategyProfile,ProcurementPlan
from shelfcash_forecast.optimization.planning_service import content_hash
from shelfcash_forecast.optimization.lot_milp import solve_lot_procurement,mean_world


def solve_decomposed(request,profile,*,stochastic):
    start=time.monotonic();full=request.evaluation_scenarios or request.demand_scenarios
    incomplete_scope={'planning_binding':request.planning_binding,'mode':'stochastic' if stochastic else 'deterministic',
        'physics_scenario_ids':[s.scenario_id for s in request.demand_scenarios] if stochastic else [mean_world(request).scenario_id]}
    keys=sorted({(l.store_id,l.ingredient_id) for s in full for l in s.lines}|
                {(l.store_id,l.ingredient_id) for l in request.initial_inventory})
    commitments={};evidence=[]
    # A union bound allocates the original package risk ceiling across keys.
    # Original universal/expected/chance floors stay intact. Independence is not
    # assumed; the final joint critic measures the actual union on the full pool.
    ceiling=min(profile.maximum_acceptable_stockout_probability,
        profile.maximum_stockout_probability if profile.maximum_stockout_probability is not None else 1)
    per_key_ceiling=ceiling/max(1,len(keys))
    generation=StrategyProfile.model_validate({**profile.model_dump(),
        'maximum_acceptable_stockout_probability':per_key_ceiling,'maximum_stockout_probability':per_key_ceiling})
    for key in keys:
        remaining=request.limits.total_seconds-(time.monotonic()-start)
        if remaining<=0:break
        data=request.model_dump()
        data['initial_inventory']=[l for l in data['initial_inventory'] if (l['store_id'],l['ingredient_id'])==key]
        data['existing_inbound']=[l for l in data['existing_inbound'] if (l['store_id'],l['ingredient_id'])==key]
        data['supplier_offers']=[o for o in data['supplier_offers'] if (o['store_id'],o['ingredient_id'])==key]
        data['cost_assumptions']=[c for c in data['cost_assumptions'] if (c['store_id'],c['ingredient_id'])==key]
        data['supplier_constraints']=[] # optimistic projection; all parent caps retained in final joint certification
        rules=[r for r in data['normalized_rules'] if r['store_id']==key[0] and r['ingredient_id'] in {None,key[1]}]
        for r in rules:
            if r['semantics']=='GLOBAL_RECEIVING_PEAK':r['occupancy']=[c for c in r['occupancy'] if c['ingredient_id']==key[1]]
            if r['semantics']=='PER_KEY_EXPECTED_FILL':r['ingredient_id']=key[1]
        data['normalized_rules']=rules
        worlds=[]
        for s in full:
            row=s.model_dump();row['lines']=[l for l in row['lines'] if (l['store_id'],l['ingredient_id'])==key]
            worlds.append(row)
        data['demand_scenarios']=worlds;data['evaluation_scenarios']=worlds
        data['budget']=None
        data['limits']['total_seconds']=remaining;data['limits']['per_solve_seconds']=min(request.limits.per_solve_seconds,remaining)
        data['planning_binding']=dict(request.planning_binding) if request.planning_binding else {}
        # Normalize JSON representation before hashing (dates and floats included).
        from shelfcash_forecast.optimization.contracts import SupplierOffer
        from shelfcash_forecast.optimization.scenario_contracts import NormalizedBusinessRule
        if request.planning_binding:
            data['planning_binding']['contracts_hash']=content_hash({'offers':[SupplierOffer.model_validate(o).model_dump(mode='json') for o in data['supplier_offers']],
                'rules':[NormalizedBusinessRule.model_validate(r).model_dump(mode='json') for r in rules]})
            data['planning_binding']['candidate_projection']={'parent_contract_hash':request.planning_binding['contracts_hash'],
                'key':list(key),'scope':'CANDIDATE_GENERATION_ONLY_NOT_A_CUSTOMER_PLAN'}
        projected=OptimizationRequest.model_validate(data)
        logging.getLogger(__name__).info('Projected %s %s worlds=%s',profile.name,key,len(full))
        plan=solve_lot_procurement(projected,generation,stochastic=stochastic,aggregate_nonexpiring_candidate_lots=True)
        logging.getLogger(__name__).info('Projected %s %s solver=%s elapsed=%.2f',profile.name,key,plan.solver_status,plan.provenance.get('elapsed_seconds',0))
        evidence.append({'key':key,'request_hash':content_hash(projected),'projection_binding':projected.planning_binding,
            'solver_status':plan.solver_status,'plan':plan.model_dump(mode='json'),
            'generation_thresholds':{'universal_floor':profile.minimum_acceptable_fill_rate,'per_key_stockout_ceiling':per_key_ceiling,
                'package_ceiling':ceiling,'union_bound':'sum of key probabilities, no independence assumption',
                'budget_and_coupled_supplier_caps':'all retained in final joint certification'}})
        if plan.solver_status!='OPTIMAL':break
        commitments.update({o.offer_id:o.pack_count for o in plan.orders})
    if len(evidence)!=len(keys) or any(e['solver_status']!='OPTIMAL' for e in evidence):
        return ProcurementPlan(plan_id=f'{request.request_id}-{profile.name.lower()}',strategy=profile.name,orders=[],purchase_cost=0,
            solver_status='LIMIT_REACHED',provenance={'termination':'RESTRICTED_DECOMPOSITION_NOT_CERTIFIED',
                'decomposition':evidence,**incomplete_scope,
                'not_infeasibility_proof':True})
    remaining=request.limits.total_seconds-(time.monotonic()-start)
    if remaining<=0:return ProcurementPlan(plan_id=f'{request.request_id}-{profile.name.lower()}',strategy=profile.name,orders=[],purchase_cost=0,
        solver_status='LIMIT_REACHED',provenance={'decomposition':evidence,**incomplete_scope,'not_infeasibility_proof':True})
    final_request=OptimizationRequest.model_validate({**request.model_dump(),'limits':{**request.limits.model_dump(),
        'per_solve_seconds':min(request.limits.per_solve_seconds,remaining),'total_seconds':remaining}})
    plan=solve_lot_procurement(final_request,profile,stochastic=stochastic,fixed_pack_counts=commitments)
    return ProcurementPlan.model_validate({**plan.model_dump(),'provenance':{**plan.provenance,
        'decomposition':evidence,'candidate_generation':'DECOMPOSED_FIXED_CERTIFICATION',
        'global_optimality_claimed':False,'elapsed_seconds':time.monotonic()-start}})
