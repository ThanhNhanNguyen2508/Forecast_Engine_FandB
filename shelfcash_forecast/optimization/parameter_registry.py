"""Resolved planning parameters with units, authority and limits of evidence."""
from __future__ import annotations
from shelfcash_forecast.optimization.strategies import default_strategy_profiles

def parameter_registry(request,config=None):
    rows=[]
    def add(name,value,unit,scope,category,path,sites,meaning,provenance='ENGINEERING_DEFAULT',source=None,valid_range=None):
        rows.append(dict(name=name,value=value,unit=unit,grain_scope=scope,category=category,provenance=provenance,
            source_evidence=source,meaning_rationale=meaning,config_override_path=path,validated_range=valid_range,
            call_sites=sites,business_calibration_evidence='NOT_AVAILABLE',sensitivity_required=category!='NUMERICAL_TOLERANCE'))
    profiles={p.name:p for p in default_strategy_profiles()};profiles.update({p.name:p for p in request.strategy_profiles})
    for strategy,p in profiles.items():
        for name,value in p.model_dump().items():
            if name=='name':continue
            ratio=any(s in name for s in ('fill_rate','probability','alpha','gap'))
            add(strategy+'.'+name,value,'ratio' if ratio else 'dimensionless multiplier on currency cost',strategy,'POLICY_PREFERENCE',
                'strategy_profiles.'+strategy+'.'+name,['lot_milp.py','critic.py','strategies.py','customer_export.py'],
                'Declared product preference/safeguard; default is not a fitted business coefficient',
                provenance='CONFIRMED_BUSINESS_POLICY' if request.environment=='PRODUCTION' and request.scenario_provenance.get('policy_evidence_id') else 'ENGINEERING_DEFAULT',
                source=request.scenario_provenance.get('policy_evidence_id'),valid_range='[0,1] ratios, alpha in (0,1), nonnegative penalty')
    add('budget',request.budget,request.currency,request.budget_scope,'BUSINESS_CONSTRAINT','budget',['lot_milp.py','constraints.py','preflight.py','customer_export.py'],
        'null removes only procurement cash cap; 0 retains a zero cap','FACT_FROM_SOURCE' if request.scenario_provenance.get('budget_evidence_id') else 'UNRESOLVED',valid_range='null or finite >=0')
    for i,c in enumerate(request.cost_assumptions):
        for field in ('holding_cost_per_unit_day','shortage_cost_per_unit','expired_cost_per_unit','waste_cost_per_unit','capacity_quantity'):
            add(f'cost.{i}.{field}',getattr(c,field),c.unit if field=='capacity_quantity' else f'{request.currency}/{c.unit}'+('/day' if 'holding' in field else ''),
                f'{c.store_id}|{c.ingredient_id}','PHYSICAL_CONSTRAINT' if field=='capacity_quantity' else 'COST_ASSUMPTION',f'cost_assumptions.{i}.{field}',
                ['lot_milp.py','inventory/simulator.py','critic.py'], 'Receiving peak capacity' if field=='capacity_quantity' else 'Currency consequence estimate, independent of objective multiplier',
                'ASSUMED_FOR_SCENARIO' if request.planning_mode=='SCENARIO_PREVIEW' or request.environment=='SYNTHETIC' else 'UNRESOLVED',valid_range='finite >=0')
    for i,o in enumerate(request.supplier_offers):
        for field in ('pack_size','minimum_order_quantity','maximum_order_quantity','unit_price','delivery_cost','lead_time_days','shelf_life_days'):
            add(f'offer.{o.offer_id}.{field}',getattr(o,field),'days offset' if field in {'lead_time_days','shelf_life_days'} else f'{request.currency}/{o.unit}' if field=='unit_price' else request.currency if field=='delivery_cost' else o.unit,
                f'{o.store_id}|{o.supplier_id}|{o.ingredient_id}|{o.offer_id}','SUPPLIER_PRODUCT_TERM',f'supplier_offers.{i}.{field}',
                ['chronology.py','lot_milp.py','constraints.py','customer_export.py'],'Scoped declared supplier opportunity, normalized base-quantity MOQ and per-base-unit price',
                'ASSUMED_FOR_SCENARIO' if o.source_terms.get('supplier_confirmation_pending') or request.environment=='SYNTHETIC' else o.source_terms.get('provenance_classification','UNRESOLVED'),
                source=o.source_terms,valid_range='pack >0; quantities/costs >=0; lead integer 0..3650; shelf offset >=0 or explicit unknown-expiry policy')
    for i,r in enumerate(request.normalized_rules):
        add('rule.'+r.rule_id,r.target,r.unit,f'{r.store_id}|{r.ingredient_id}|{r.effective_from}:{r.effective_to}',
            'POLICY_PREFERENCE' if r.classification!='HARD' else 'PHYSICAL_BUSINESS_CONSTRAINT',f'normalized_rules.{i}.target',
            ['business_rules.py','lot_milp.py','critic.py'],r.semantics+'; '+r.classification,
            'ASSUMED_FOR_SCENARIO' if request.planning_mode=='SCENARIO_PREVIEW' else str(r.source.get('classification','UNRESOLVED')),source=r.source,valid_range='finite >=0; service ratio <=1')
        add('rule.'+r.rule_id+'.soft_penalty',r.penalty_currency_per_unit,f'{request.currency}/{r.unit}',r.store_id+'|'+str(r.ingredient_id),
            'POLICY_PREFERENCE',f'normalized_rules.{i}.penalty_currency_per_unit',['lot_milp.py','business_rules.py','customer_export.py'],
            'Currency consequence of a declared soft reserve miss, separate from supplier price','ASSUMED_FOR_SCENARIO' if request.planning_mode=='SCENARIO_PREVIEW' else 'UNRESOLVED',source=r.source,valid_range='finite >=0; positive for SOFT reserve')
        for j,c in enumerate(r.occupancy):
            add('occupancy.'+r.rule_id+'.'+c.ingredient_id,c.liters_per_base_unit,'liter/'+c.base_unit,r.store_id+'|'+c.ingredient_id,
                'PHYSICAL_CONSTRAINT',f'normalized_rules.{i}.occupancy.{j}', ['lot_milp.py','business_rules.py','preflight.py'],
                'Physical storage occupancy; mass and volume are not assumed equal','ASSUMED_FOR_SCENARIO',source=c.assumption_id,
                valid_range=f'[{c.lower_bound},{c.upper_bound}]')
    for name,value in request.limits.model_dump().items():
        add('limits.'+name,value,'seconds' if 'seconds' in name else 'count' if any(s in name for s in ('variables','constraints','iterations')) else 'ratio/flag',
            request.request_id,'ENGINEERING_LIMIT','limits.'+name,['optimizer.py','decomposition.py','lot_milp.py'],'Bounded compute effort; reaching limit does not prove infeasibility',valid_range='SolverLimits schema')
    for field,value in request.inventory_policy.model_dump(mode='json').items():
        if value is not None:
            add('inventory_policy.'+field,value,'base units' if 'tolerance' in field else 'days' if 'days' in field else 'ratio/flag',request.request_id,
                'NUMERICAL_TOLERANCE' if 'tolerance' in field else 'INVENTORY_POLICY','inventory_policy.'+field,['inventory/fefo.py','inventory/simulator.py','critic.py'],
                'Explicit daily chronology/metric policy',valid_range='accounting tolerance <=1e-6; remaining bounds in contract')
    add('scenario_pool',len(request.evaluation_scenarios or request.demand_scenarios),'worlds',request.request_id,'ENGINEERING_LIMIT','evaluation_scenarios',
        ['planning_service.py','optimizer.py','critic.py'],'Full pool authority; subset is candidate generation only; not independent OOS',valid_range='1..2000')
    add('recommendation_order',['BALANCED','PROTECTED','LEAN'],'ordered strategies',request.request_id,'POLICY_PREFERENCE','optimizer._profiles',
        ['optimizer.py','customer_export.py'],'Product preference after exact acceptance; does not rank objectives with different weights')
    add('seed',request.seed,'integer',request.request_id,'ENGINEERING_LIMIT','seed',['planning_service.py','scenario/composer.py'],
        'Reproducible scenario construction/subset selection; does not make a pool independently calibrated',valid_range='integer contract')
    add('optimization_pool',len(request.demand_scenarios),'worlds',request.request_id,'ENGINEERING_LIMIT','demand_scenarios',
        ['optimizer.py','decomposition.py','lot_milp.py'],'Candidate pool; exact evaluation uses the unchanged evaluation pool',valid_range='1..2000')
    add('candidate_generation',request.candidate_generation,'enum',request.request_id,'ENGINEERING_LIMIT','candidate_generation',
        ['optimizer.py','decomposition.py'],'Warm candidate strategy, never a full feasibility proof on its own',valid_range='JOINT_MILP or DECOMPOSED_FIXED_CERTIFICATION')
    for name,value,unit,sites,meaning in [
        ('money_quantity_identity_relative',1e-9,'relative',['contracts.py','constraints.py'],'Relative tolerance on pack quantity and purchase cash identity'),
        ('money_quantity_identity_absolute',1e-9,'currency/base units',['contracts.py','constraints.py'],'Absolute tolerance on pack quantity and purchase cash identity'),
        ('probability_service_comparison',1e-9,'ratio',['critic.py','preflight.py','contracts.py'],'Probability normalization and service/risk inequality comparison'),
        ('stockout_quantity_detection',1e-8,'canonical units',['critic.py','inventory/risk.py'],'Positive shortage classification above numerical noise'),
        ('rule_violation_comparison',1e-8,'declared rule units',['business_rules.py'],'Exact normalized-rule violation classification'),
        ('capacity_comparison',1e-9,'canonical units',['critic.py'],'Exact physical capacity violation comparison'),
        ('integer_pack_rounding',1e-12,'pack count',['lot_milp.py'],'Round-off allowance in floor/ceil of pack bounds'),
        ('cvar_tail_completion',1e-12,'probability mass',['parameter_registry.py'],'Terminate independent discrete tail integration')]:
        add('numerical.'+name,value,unit,request.request_id,'NUMERICAL_TOLERANCE',None,sites,meaning,
            valid_range='fixed versioned engineering tolerance; changing it requires code review and regression evidence')
    from shelfcash_forecast.optimization.input_validation import SUPPORTED_ENVELOPE
    for name,value in SUPPORTED_ENVELOPE.items():
        add('supported_envelope.'+name,value,'canonical units' if 'quantity' in name else 'currency' if 'cost' in name else 'count',
            request.request_id,'ENGINEERING_LIMIT',None,['input_validation.py'],'Public numerical/memory envelope, no silent clamping',valid_range='fixed versioned envelope')
    if config is not None:
        payload=config.model_dump(mode='json') if hasattr(config,'model_dump') else config
        if isinstance(payload,dict) and payload.get('cost_policy') is not None:
            add('cost_policy',payload['cost_policy'],'declared per-base/day or multiplier',request.request_id,'COST_ASSUMPTION','planning.cost_policy',
                ['planning_service.py','customer_export.py'],'Source and rationale of consequence-cost reference, not a business accuracy claim',
                'ASSUMED_FOR_SCENARIO' if request.planning_mode=='SCENARIO_PREVIEW' else 'UNRESOLVED',source=payload['cost_policy'],valid_range='ConsequenceCostPolicy schema')
    return rows

def recompute_objective(request,candidate,profile):
    """Currency objective from exact states, plus discrete CVaR calculated afresh."""
    simulation=candidate.physics_simulation or candidate.simulation
    cash=sum(o.purchase_cost+o.delivery_cost for o in candidate.plan.orders);losses=[];weights=[];components=[]
    for world in simulation.results:
        sums={f:sum(getattr(l,f) or 0 for l in world.daily_ledgers) for f in ('holding_cost','shortage_cost','expiry_cost','waste_cost')}
        from shelfcash_forecast.optimization.business_rules import evaluate_rules
        one=simulation.model_copy(update={'results':[world]})
        soft=sum(r['soft_penalty_vnd'] for r in evaluate_rules(request.normalized_rules,one,candidate.plan.orders))/(world.probability_weight or 1)
        loss=sums['holding_cost']*profile.holding_penalty+sums['shortage_cost']*profile.shortage_penalty+(sums['expiry_cost']+sums['waste_cost'])*profile.waste_penalty+soft
        losses.append(cash+loss);weights.append(world.probability_weight if world.probability_weight is not None else 1/len(simulation.results));components.append({'world':world.scenario_id,**sums,'soft_penalty':soft,'weighted_consequence_loss':loss})
    cvar_losses,cvar_weights=losses,weights
    if candidate.plan.provenance.get('mode')=='deterministic':
        cvar_losses=[];cvar_weights=[]
        original=candidate.plan.provenance.get('objective_world_weights',{s.scenario_id:s.probability_weight for s in request.demand_scenarios})
        for world in candidate.simulation.results:
            if world.scenario_id not in original:continue
            sums={f:sum(getattr(l,f) or 0 for l in world.daily_ledgers) for f in ('holding_cost','shortage_cost','expiry_cost','waste_cost')}
            from shelfcash_forecast.optimization.business_rules import evaluate_rules
            one=candidate.simulation.model_copy(update={'results':[world]})
            soft=sum(r['soft_penalty_vnd'] for r in evaluate_rules(request.normalized_rules,one,candidate.plan.orders))/(world.probability_weight or 1)
            cvar_losses.append(cash+sums['holding_cost']*profile.holding_penalty+sums['shortage_cost']*profile.shortage_penalty+(sums['expiry_cost']+sums['waste_cost'])*profile.waste_penalty+soft)
            cvar_weights.append(original[world.scenario_id])
    pairs=sorted(zip(cvar_losses,cvar_weights),reverse=True);remaining=1-profile.cvar_alpha;tail=0
    for loss,weight in pairs:
        take=min(remaining,weight);tail+=take*loss;remaining-=take
        if remaining<=1e-12:break
    cvar=tail/(1-profile.cvar_alpha)
    value=profile.cash_penalty*cash+sum(l*w for l,w in zip(losses,weights))+profile.cvar_weight*cvar
    return {'recomputed_preference_objective':value,'currency':request.currency,'procurement_cash':cash,'cvar':cvar,'components':components,
        'solver_objective':candidate.plan.objective_value,'solver_search_phase':candidate.plan.provenance.get('search_phase'),
        'objective_comparable':candidate.plan.provenance.get('search_phase')!='FEASIBILITY_FIRST'}
