"""Offline catalogue, bounded coherent profiles, and accepted-only comparison."""
from __future__ import annotations
from pathlib import Path
import json
from datetime import timedelta
from shelfcash_forecast.optimization.scenario_contracts import REGISTERED_ASSUMPTION_FIELDS, semantic_hash
from shelfcash_forecast.optimization.planning_config import PlanningConfig


def semantics_catalogue():
    definitions=[
        ('U01','RECEIVING','Raw weekdays constrain receiving; lead first, then snap to next receiving day.',
         'Conservative receiving availability; clear arrival deadline.','May delay first supply beyond initial-stock coverage.', 'supplier weekdays, lead, cutoff, receiving exceptions','SUPPORTED'),
        ('U01','ORDER','Raw weekdays constrain order placement; arrival follows lead without receiving snap.',
         'Separates placement from receiving.','Receiving availability remains an assumption.','order weekdays and receiving availability','SUPPORTED'),
        ('U01','DISPATCH','Lead is order-to-dispatch; snap dispatch, then add separately declared transport once.',
         'Models dispatch semantics without double lead.','Needs separate transport evidence.','dispatch weekdays, order-to-dispatch days, transport days','SUPPORTED_UNSELECTED'),
        ('U01','NEGOTIATED','Existing supplier may offer additional receiving days under a proposed contract.',
         'Can close early/tail supply gaps.','Supplier has not agreed; extra fee conditional.','source rule, proposed receiving days, lead, MOQ, fee','SUPPORTED'),
        ('U02','CALENDAR_ORDER','Calendar-day lead measured from declared order date; inbound before demand.',
         'Simple auditable daily chronology.','EOD placement and receiving-before-demand require confirmation.','order boundary, lead, daily timing','SUPPORTED'),
        ('U02','BUSINESS_NEXT','Explicit working calendar counts lead from the next day.',
         'Exposes cutoff/weekend sensitivity.','Delays deliveries; not a Monday-Friday factual default.','supplier working weekdays and scoped holidays','SUPPORTED'),
        ('U02','AFTER_DEMAND','Receipt after demand cannot serve that day; becomes available for following demand.',
         'Captures late delivery losses.','Daily event extension not implemented; selected profiles reject it.','event chronology and receiving capacity at after-demand event','NOT_IMPLEMENTED'),
        ('U02','EXPEDITED','Proposed shorter lead, from original source rule, no backdated receipt.',
         'Can serve first-day gaps while preserving expiry offset.','Requires supplier agreement and declared surcharge.','original/proposed lead, cutoff, fee','SUPPORTED'),
        ('U03','NO_EXCEPTIONS','Assume no supplier/receiving holiday exceptions during the dated horizon.',
         'Few additional calendar assumptions.','Not confirmed availability; store holidays are not reused.','supplier scope, explicit assumption','SUPPORTED'),
        ('U03','SCOPED_WORKING','Explicit six-day supplier workweek and separately scoped exception dates.',
         'Reproducible holiday/workday sensitivity.','Needs supplier confirmation.','working weekdays and holiday scope','SUPPORTED'),
        ('U04','PEAK_GLOBAL','Per-ingredient physical receiving peaks plus all-key global liters occupancy.',
         'Checks inventory before expiry disposal and consumption, including demand-zero keys.',
         'Estimated occupancy upper coefficients can overstate capacity needs.','scoped source capacity, each normalized-unit occupancy with bounds','SUPPORTED'),
        ('U04','EOD_MAX','Maximum stock interpreted as end-of-day usable-stock target.',
         'Can analyze stock target rather than receiving space.','Does not certify physical receiving capacity; distinct interpretation.','source interpretation and effective interval','SUPPORTED_UNSELECTED'),
        ('U04','COMPARTMENTS','Separate source-backed cold/dry/packaging compartment capacities.',
         'Models noninterchangeable storage.','No measured compartment design in current data.','compartment membership, measured capacities, occupancy','NOT_IMPLEMENTED'),
        ('U05','ADVISORY_RESERVE','Source service target hard per-key expected horizon fill; safety EOD and purchase cover advisory.',
         'Prioritizes explicit service while quantifying reserve/freshness target misses.',
         'May finish below safety targets; advisory satisfaction is not claimed.','source targets, grain, effective intervals, mean demand reference','SUPPORTED'),
        ('U05','SOFT_RESERVE','Safety EOD shortfall penalized in VND/unit; unchanged service and strategy safeguards.',
         'Makes reserve tradeoff economic and measurable.','Penalty estimated, affects procurement cost.','scoped currency/unit penalties','SUPPORTED'),
        ('U05','HARD_RESERVE','Safety is minimum usable EOD stock in every world; service unchanged.',
         'Strong reserve protection.','May conflict with early arrivals, packs or peaks.','reserve checkpoints/intervals','SUPPORTED'),
        ('U05','FRESH_RECEIVING','Minimum remaining life at receipt, leaving factual expiry unchanged.',
         'Auditable freshness gate.','Distinct from purchase-cover interpretation.','remaining life threshold, supplier shelf offset','SUPPORTED_UNSELECTED'),
        ('U05','CHANCE_UNIVERSAL','Strategy chance per-key fill and universal per-world/key fill remain lower safeguards.',
         'Avoids package-mean hiding ingredient failures.','Model-derived pool probabilities do not guarantee future service.','original scenario weights and baseline floors','SUPPORTED_BASELINE'),
        ('U06','SCOPED_PACK_COUNT','MOQ is a number of each offer\'s declared packs; integer packs and sourced pack conversion.',
         'Packaging interpretation is explicit.','Requires supplier confirmation; no global pack conversion.','raw MOQ, own pack size, stable supplier/ingredient IDs','SUPPORTED'),
        ('U06','SCOPED_BASE_QUANTITY','MOQ is quantity in the declared base unit, rounded up to an integer pack requirement.',
         'Tests alternate MOQ quantity basis.','Different interpretation may increase commitment.','scoped base quantity, own pack size and evidence','SUPPORTED_UNSELECTED'),
        ('U07','BASE_PRICE_LINE_FEE','Raw VND price interpreted per normalized base unit; 50,000VND per ingredient opportunity.',
         'Transparent conservative line-fee estimate.','Raw basis ambiguity remains; missing fee is not factual free.','scoped price mappings and explicit fee assumption','SUPPORTED'),
        ('U07','BASE_PRICE_GROUP_FEE','Same price basis; 50,000VND once per supplier/store/order-day/arrival-day shipment.',
         'Analyzes conditional consolidation; equal stable VND fee allocation.',
         'Supplier grouping/fee must be confirmed; no assumed factual saving.','shipment identity and fee activation','SUPPORTED'),
        ('U07','PACK_PRICE','Raw price interpreted per pack and divided by source pack size once.',
         'Resolves alternate price basis explicitly.','Lower normalized prices are not demonstrated policy savings.','source pack size and scoped mapping','SUPPORTED_UNSELECTED'),
        ('U07','EXPLICIT_FREE','Explicit assumed free delivery, distinct from a missing fee.',
         'Can test supplier free-delivery contract.','Not selected; no source evidence that delivery is free.','scoped free-fee assumption','SUPPORTED_UNSELECTED')]
    return [{'option_id':f'{g}.{option}','semantic_group':g,'meaning':meaning,'pros':pros,'cons':cons,
        'consequences':cons+' '+pros,'required_inputs':inputs,'support_status':support,
        'compatible_options':'same versioned profile; chronological and dimensional guards required',
        'incompatible_options':'different meanings within same group cannot silently co-exist',
        'tests':'Public builder validation and M5 solver/exact differential regressions; evidence in the versioned acceptance matrix',
        'customer_confirmations':inputs+'; user/business confirmation remains pending'}
        for g,option,meaning,pros,cons,inputs,support in definitions]


class ProfileBuildError(ValueError):
    def __init__(self,diagnostics):
        self.code='BLOCKED_INPUT_SEMANTICS';self.diagnostics=diagnostics
        self.details={'diagnostics':diagnostics}
        super().__init__(self.code+': physical registry terms required')


def build_profiles(bundle, *, occupancy_registry=None, contract_source_rules=(), scenario_count=100, seed=42, version=4):
    """Six coherent scenario profiles; physical registries are always explicit."""
    from shelfcash_preprocess.engine import create_supplier_offers, load_canonical_frames
    from shelfcash_preprocess.pipeline import load_bundle
    info=load_bundle(bundle);store=info.manifest.context.store_id
    base=create_supplier_offers(bundle,info.manifest.context.cutoff_date)
    frames=load_canonical_frames(bundle)
    descriptions=[('P1_RECEIVING','Conservative normal receiving calendar; hard peaks; advisory reserve/freshness.'),
        ('P2_ORDER','Order-day interpretation sensitivity; same prices, units, physical rules and service.'),
        ('P3_WORKING','Explicit supplier Mon-Sat working days, next-day cutoff; normal receiving schedule.'),
        ('P4_CONSOLIDATED','Same normal receiving semantics with shipment-group delivery fees.'),
        ('P5_NEGOTIATED','Conditional extra receiving/expedited existing-supplier contracts, grouped fees, soft reserve.'),
        ('P6_CONTINUITY','Cumulative-prefix recovery including earlier packaging; same service, soft reserve and physical safeguards.')]
    # The caller supplies physical coefficients, IDs, units, scope and source.
    # Example registries are never imported as default customer facts.
    registry={(r.get('store_id',store),r['ingredient_id'],r['base_unit']):r for r in (occupancy_registry or [])}
    if len(registry)!=len(occupancy_registry or []):
        raise ProfileBuildError([{'reason_code':'DUPLICATE_SCOPED_OCCUPANCY','field_paths':['occupancy_registry']}])
    requires_global=any(r['rule_type']=='storage_capacity' for r in frames['business_rules'].to_dict('records'))
    missing=[o for o in base if (o.store_id,o.ingredient_id,o.unit) not in registry] if requires_global else []
    if missing:
        raise ProfileBuildError([{'reason_code':'OCCUPANCY_REQUIRED','field_paths':['occupancy_registry'],
            'store_id':o.store_id,'ingredient_id':o.ingredient_id,'unit':o.unit,
            'expected_meaning':'measured or explicit scenario liters per base unit; lower/upper bounds and source',
            'action_required':['PROVIDE_SCOPED_OCCUPANCY_WITH_PROVENANCE']} for o in missing])
    if any(not r.get('source') for r in registry.values()):
        raise ProfileBuildError([{'reason_code':'OCCUPANCY_PROVENANCE_REQUIRED','field_paths':['occupancy_registry.source']}])
    profiles={}
    for pid,description in descriptions:
        config=PlanningConfig(cost_policy={'holding_cost_rate_per_day':.001,'shortage_cost_multiplier':1.5,
            'expired_cost_multiplier':1,'waste_cost_multiplier':1,'rationale':'Explicit demo VND consequence estimates, unchanged baseline coefficients'},
            label='DEMO_ONLY_NOT_FOR_OPERATION').model_dump(mode='json')
        config.update(schema_version=3,planning_mode='SCENARIO_PREVIEW',seed=seed,scenario_count=scenario_count,
            optimization_scenario_count=min(10,scenario_count),selection_seed=seed,delivery_cost_assumption=50000,
            delivery_fee_scope='SUPPLIER_STORE_ORDER_ARRIVAL_GROUP' if pid in {'P4_CONSOLIDATED','P5_NEGOTIATED','P6_CONTINUITY'} else 'PER_INGREDIENT_ORDER_OPPORTUNITY',
            supplier_calendar={'status':'ASSUMED_FOR_SCENARIO','meaning':'ORDER' if pid=='P2_ORDER' else 'RECEIVING',
                'lead_time_basis':'BUSINESS_DAYS' if pid=='P3_WORKING' else 'CALENDAR_DAYS',
                'order_boundary':'NEXT_DAY' if pid=='P3_WORKING' else 'LEAD_FROM_ORDER_DATE',
                'arrival_before_consumption':True,'holiday_scope':'NO_HOLIDAY_EXCEPTIONS','holidays':[],
                'working_weekdays':[0,1,2,3,4,5] if pid=='P3_WORKING' else [],
                'dispatch_transport_days':None,'evidence':f'{pid}: explicit unconfirmed scenario; supplier/receiving only'},
            limits={'per_solve_seconds':60,'total_seconds':300,'max_refinement_iterations':2,
                    'max_model_variables':250000,'max_model_constraints':1000000})
        config['price_basis_mappings']={o.source_terms['source_rule_id']:{'basis':'BASE_UNIT','evidence':f'{pid}: assumption, raw transform default not confirmation'} for o in base}
        config['packaging_mappings']={o.source_terms['source_rule_id']+'|'+o.source_terms['order_unit']:'PACK_COUNT'
            for o in base if o.source_terms['order_unit'] not in {'pack',o.unit}}
        config['occupancy_coefficients']=[{'ingredient_id':key[1],'base_unit':key[2],
            **{k:r[k] for k in ('liters_per_base_unit','lower_bound','upper_bound')},
            'assumption_id':pid+':occupancy_coefficients'} for key,r in sorted(registry.items()) if key[0]==store]
        config['rule_mappings']=[]
        for row in frames['business_rules'].to_dict('records'):
            import pandas as pd
            ingredient=None if pd.isna(row['ingredient_id']) else str(row['ingredient_id'])
            operators={'storage_capacity':'GLOBAL_RECEIVING_PEAK','maximum_stock':'RECEIVING_PEAK',
                'safety_stock':'END_OF_DAY_MIN','service_level_target':'PER_KEY_EXPECTED_FILL',
                'shelf_life_target':'PURCHASE_COVER_DAYS'}
            if row['rule_type'] not in operators:
                raise ProfileBuildError([{'reason_code':'UNSUPPORTED_RULE_OPERATOR','field_paths':['business_rules.rule_type'],
                    'value':row['rule_type'],'expected_meaning':list(operators)}])
            semantics=operators[row['rule_type']]
            classification='ADVISORY' if row['rule_type'] in {'safety_stock','shelf_life_target'} else 'HARD'
            penalty=0
            if pid in {'P5_NEGOTIATED','P6_CONTINUITY'} and row['rule_type']=='safety_stock':
                classification='SOFT';o=next(o for o in base if o.ingredient_id==ingredient)
                penalty=o.unit_price*.02
            config['rule_mappings'].append({'rule_type':row['rule_type'],'ingredient_id':ingredient,'unit':row['unit'],
                'semantics':semantics,'classification':classification,'penalty_currency_per_unit':penalty,
                'evidence':f'{pid}: service explicit business aim; peaks conservative physical interpretation; reserve is an EOD target rather than absolute no-sales minimum; cover advisory uses full-pool horizon daily mean'})
        if pid in {'P5_NEGOTIATED','P6_CONTINUITY'}:
            for o in base:
                if o.source_terms['source_rule_id'] in contract_source_rules:
                    config['contract_options'].append({'contract_id':'PROPOSED_DAILY_LEAD1_'+o.ingredient_id,
                        'source_rule_id':o.source_terms['source_rule_id'],'proposed_lead_days':1,
                        'receiving_weekdays':list(range(7)),'extra_fee':100000,
                        'assumption_id':pid+':contract_options','classification':'CONTRACT_OPTION_REQUIRES_SUPPLIER_CONFIRMATION'})
        rationale={'supplier_calendar':'Interpretation chosen for chronology sensitivity; no supplier approval inferred.',
            'supplier_calendars':'No unregistered supplier overrides.',
            'packaging_mappings':'Explicit scenario interpretation: source order label is pack count; size remains each scoped source pack. Requires confirmation.',
            'price_basis_mappings':'Raw prices interpreted per base unit; alternative pack basis is not a saving.',
            'rule_mappings':'Hard per-key service uses each mapped source target and effective scope; physical peaks retain mapped limits. EOD reserve/freshness misses measured; soft reserve variants use declared currency/unit penalties.',
            'delivery_cost_assumption':'No factual fee supplied: estimated 50,000VND per declared fee group.',
            'delivery_fee_scope':'Line versus shipment grouping is a business assumption, not a proven supplier discount.',
            'cost_policy':'Unchanged baseline demo holding/shortage/expiry coefficients; no business approval.',
            'occupancy_coefficients':'Bounded engineering storage envelopes for packaged liquids, fruit void space, dry materials and nested cups; enforce upper bounds. No measurements supplied.',
            'contract_options':'Only source-backed existing suppliers with diagnostics-driven early/tail gaps; proposed daily receiving/lead1 +100,000VND shipment surcharge. Not existing inbound.',
            'currency':'VND accounting matches source numeric price context; no other currency supported.'}
        from pydantic import TypeAdapter
        for k in REGISTERED_ASSUMPTION_FIELDS:
            adapter=TypeAdapter(PlanningConfig.model_fields[k].annotation)
            config[k]=adapter.dump_python(adapter.validate_python(config[k]),mode='json')
        behavior={k:config[k] for k in sorted(REGISTERED_ASSUMPTION_FIELDS)}
        assumptions=[]
        for k,value in behavior.items():
            kind='HYPOTHETICAL_SUPPLIER_CONTRACT' if k=='contract_options' else 'ENGINEERING_ESTIMATE' if k in {'occupancy_coefficients','delivery_cost_assumption','cost_policy'} else 'BUSINESS_POLICY_CHOICE' if k in {'rule_mappings','delivery_fee_scope'} else 'AMBIGUOUS_SOURCE_INTERPRETATION'
            assumptions.append({'assumption_id':pid+':'+k,'profile_id':pid,'field_path':k,'scope':store,
                'value':value,'unit':'VND/group' if k=='delivery_cost_assumption' else 'scoped typed contract',
                'kind':kind,'rationale':rationale[k],'source_status':'PROPOSED' if kind=='HYPOTHETICAL_SUPPLIER_CONTRACT' else 'ESTIMATED' if kind=='ENGINEERING_ESTIMATE' else 'UNCONFIRMED',
                'source_locator':json.dumps([r['source'] for r in registry.values()],ensure_ascii=False) if k=='occupancy_coefficients' else 'sealed canonical supplier_rules/business_rules; explicit task-authorized assumptions',
                'confirmation_required':True,'sensitivity_alternatives':['catalogue alternate basis/calendar/rule; occupancy lower versus upper envelope']})
        profile={'schema_version':1,'profile_id':pid,'version':version,'description':description,
            'planning_context':'SCENARIO_PREVIEW','store_id':store,
            'selected_options':{'U01':'ORDER' if pid=='P2_ORDER' else 'NEGOTIATED' if pid in {'P5_NEGOTIATED','P6_CONTINUITY'} else 'RECEIVING',
                'U02':'BUSINESS_NEXT' if pid=='P3_WORKING' else 'EXPEDITED' if pid in {'P5_NEGOTIATED','P6_CONTINUITY'} else 'CALENDAR_ORDER',
                'U03':'SCOPED_WORKING' if pid=='P3_WORKING' else 'NO_EXCEPTIONS','U04':'PEAK_GLOBAL',
                'U05':'SOFT_RESERVE' if pid in {'P5_NEGOTIATED','P6_CONTINUITY'} else 'ADVISORY_RESERVE','U06':'SCOPED_PACK_COUNT',
                'U07':'BASE_PRICE_GROUP_FEE' if pid in {'P4_CONSOLIDATED','P5_NEGOTIATED','P6_CONTINUITY'} else 'BASE_PRICE_LINE_FEE'},
            'default_rationale':'P1 is default independently of cost/acceptance: preserves normal lead, conservative receiving peaks, same baseline service/risk floors, few supplier contract hypotheses.',
            'changed_from_shared_baseline':[description],'assumptions':assumptions,
            'configuration_binding_hash':semantic_hash(behavior)}
        config['scenario_profile']=profile
        profiles[pid]=PlanningConfig.model_validate_json(json.dumps(config,allow_nan=False))
    return profiles


def compare_profile_runs(runs):
    rows=[]
    for pid,budget,run in runs:
        candidate=run.result.evaluations.get(run.result.recommended_strategy)
        metrics=candidate.simulation.risk_metrics if candidate else None
        orders=candidate.plan.orders if candidate else []
        rows.append({'profile_id':pid,'budget_case':budget,'technical_outcome':run.result.technical_outcome,
            'profile_version':run.config.scenario_profile.version,
            'accepted':candidate is not None,'critic_passed':candidate.critic.passed if candidate else False,
            'order_count':len(orders),'purchase_delivery_vnd':sum(o.purchase_cost+o.delivery_cost for o in orders) if candidate else None,
            'mean_key_fill':metrics.mean_key_fill_rate if metrics else None,'any_stockout_probability':metrics.any_stockout_probability if metrics else None,
            'required_contract_count':len(run.config.contract_options),'assumption_count':len(run.config.scenario_profile.assumptions),
            'conditional_comparison_only':True,'mode':run.selected_mode,'strategy':run.result.recommended_strategy,
            'rule_evaluations':candidate.critic.details.get('business_rule_evaluation') if candidate else None,
            'per_key_risk_metrics':[k.model_dump(mode='json') for k in metrics.by_key] if metrics else None,
            'shipment_count':candidate.plan.provenance.get('shipment_count') if candidate else None,
            'per_key_capacity_peaks':{key:max(l.maximum_quantity for w in candidate.simulation.results for l in w.daily_ledgers
                if '|'.join((l.store_id,l.ingredient_id,l.unit))==key) for key in
                {'|'.join((l.store_id,l.ingredient_id,l.unit)) for w in candidate.simulation.results for l in w.daily_ledgers}} if candidate else None,
            'modes':{mode:{'outcome':r.technical_outcome,'candidates':{s:{'solver_status':e.plan.solver_status,
                'critic_passed':e.critic.passed,'violations':e.critic.hard_violations,'dimensions':e.plan.provenance.get('dimensions'),
                'runtime':e.plan.provenance.get('elapsed_seconds')} for s,e in r.evaluations.items()}} for mode,(_,r) in run.runs.items()}})
    accepted=sorted([r for r in rows if r['accepted'] and r['order_count']>0],key=lambda r:(r['required_contract_count'],r['assumption_count'],
        r['any_stockout_probability'],r['purchase_delivery_vnd'],r['profile_id'],r['budget_case']))
    return {'default_profile':'P1_RECEIVING','ranking_policy':'accepted nonzero only; fewer hypothetical contracts, assumptions, stockout, then conditional procurement cash',
        'comparison_limits':'All prices same base-unit interpretation; fee grouping and negotiated costs remain conditional.',
        'runs':rows,'recommended':accepted[0] if accepted else None}
