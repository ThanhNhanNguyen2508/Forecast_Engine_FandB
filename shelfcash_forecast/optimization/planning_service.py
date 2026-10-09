"""One public M5 boundary for pipeline, application, staged CLI and bundle API."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from datetime import date
from dataclasses import dataclass
from datetime import timedelta

from shelfcash_forecast.inventory.contracts import ConsequenceCostAssumption
from shelfcash_forecast.optimization.chronology import offer_arrival, parse_weekdays, resolve_arrival
from shelfcash_forecast.optimization.contracts import OptimizationRequest, ProcurementDiagnostic, SupplierOffer
from shelfcash_forecast.optimization.planning_config import PlanningConfig, load_planning_config
from shelfcash_forecast.optimization.strategies import default_strategy_profiles


def content_hash(value):
    if hasattr(value,"model_dump"):value=value.model_dump(mode="json")
    if isinstance(value,list):value=[v.model_dump(mode='json') if hasattr(v,'model_dump') else v for v in value]
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(",",":"),allow_nan=False,default=str).encode()).hexdigest()


def select_scenarios(full, count, seed):
    import numpy as np
    ordered=sorted(full,key=lambda s:s.scenario_id)
    if count>=len(ordered):return list(full),1.0
    indices=sorted(np.random.default_rng(seed).choice(len(ordered),size=count,replace=False).tolist())
    selected=[ordered[i] for i in indices]
    if any(s.probability_weight is None for s in selected):raise ValueError("WEIGHTED_OPTIMIZATION_POOL_REQUIRED")
    mass=sum(s.probability_weight for s in selected)
    if mass<=0:raise ValueError("ZERO_SELECTED_PROBABILITY_MASS")
    return [type(s).model_validate({**s.model_dump(),"probability_weight":s.probability_weight/mass}) for s in selected],mass


def cost_assumptions(offers, config: PlanningConfig):
    """Explicit demo reference prices, stable under supplier/opportunity ordering."""
    grouped={}
    for o in offers:
        key=(o.store_id,o.ingredient_id,o.unit)
        grouped[key]=min(grouped.get(key,float("inf")),o.unit_price)
    p=config.cost_policy
    return [ConsequenceCostAssumption(store_id=k[0],ingredient_id=k[1],unit=k[2],
        holding_cost_per_unit_day=price*p.holding_cost_rate_per_day,shortage_cost_per_unit=price*p.shortage_cost_multiplier,
        expired_cost_per_unit=price*p.expired_cost_multiplier,waste_cost_per_unit=price*p.waste_cost_multiplier)
        for k,price in sorted(grouped.items())]


def resolved_price(base,config):
    mapping=config.price_basis_mappings.get(base.source_terms['source_rule_id'])
    return base.unit_price if mapping is None else base.source_terms['unit_price']/(base.pack_size if mapping.basis=='PACK' else 1)


def equivalent_opportunity_reduction(offers):
    """Certified dominance only: unlimited identical receipt, packs/MOQ/cost/expiry.

    Combining integer quantities into the earliest valid placement preserves all
    supplier-horizon caps and physical states, pays a nonnegative fee no more
    often, and has no order-date holding/discount/cutoff distinction. Grouped fees
    and offer caps retain distinct decisions because this proof does not apply.
    """
    # Shipment-date alternatives can be collapsed only as whole identical offer
    # vectors. All constituent ingredients must have the same valid order-date
    # set and unlimited per-opportunity quantities; otherwise preserve grouping.
    shipment_vectors={}
    for o in offers:
        group=o.source_terms.get('shipment_group_id')
        if group:shipment_vectors.setdefault(group,[]).append(o)
    vector_equivalence={}
    for group,rows in shipment_vectors.items():
        if any(o.maximum_order_quantity is not None or o.order_cutoff_date is not None for o in rows):continue
        signature=tuple(sorted((o.source_terms.get('source_rule_id'),o.source_terms.get('contract_id') or '',
            o.supplier_id,o.store_id,o.ingredient_id,o.unit,str(offer_arrival(o)),o.pack_size,o.minimum_order_quantity,
            o.unit_price,o.delivery_cost,str(o.shelf_life_days),o.available,o.emergency) for o in rows))
        vector_equivalence.setdefault(signature,[]).append(group)
    remove=set();replace={}
    for groupset in vector_equivalence.values():
        if len(groupset)<2:continue
        chosen=min(groupset,key=lambda g:(shipment_vectors[g][0].order_date,g))
        dates=sorted({str(shipment_vectors[g][0].order_date) for g in groupset})
        for g in groupset:
            if g!=chosen:remove.update(o.offer_id for o in shipment_vectors[g])
        for o in shipment_vectors[chosen]:
            replace[o.offer_id]=SupplierOffer.model_validate({**o.model_dump(),'source_terms':{**o.source_terms,
                'equivalent_order_dates':dates,'opportunity_reduction_proof':'whole identical unlimited shipment vectors; combining packs preserves peaks/expiry/supplier-horizon caps; earliest common order date; nonnegative fee charged once'}})
    offers=[replace.get(o.offer_id,o) for o in offers if o.offer_id not in remove]
    groups={};output=[]
    for o in offers:
        if o.maximum_order_quantity is not None or o.source_terms.get('shipment_group_id') or o.order_cutoff_date is not None:
            output.append(o);continue
        signature=(o.source_terms.get('source_rule_id'),o.source_terms.get('contract_id'),o.supplier_id,o.store_id,
            o.ingredient_id,o.unit,offer_arrival(o),o.pack_size,o.minimum_order_quantity,o.unit_price,o.delivery_cost,
            o.shelf_life_days,o.available,o.emergency)
        groups.setdefault(signature,[]).append(o)
    for rows in groups.values():
        first=min(rows,key=lambda o:(o.order_date,o.offer_id))
        if len(rows)>1:
            first=SupplierOffer.model_validate({**first.model_dump(),'source_terms':{**first.source_terms,
                'equivalent_order_dates':[str(o.order_date) for o in sorted(rows,key=lambda o:o.order_date)],
                'opportunity_reduction_proof':'unlimited identical lot/pack/MOQ/price/nonnegative per-line fee; earliest placement dominates; no cutoff/order-date discount; no shipment group'}})
        output.append(first)
    return sorted(output,key=lambda o:o.offer_id)


def create_order_opportunities(base_offers, end, config, environment):
    output=[];issues=[];used_mappings=set()
    preview = config.planning_mode == 'SCENARIO_PREVIEW'
    if preview and environment == 'PRODUCTION':
        raise ValueError('PRODUCTION_REJECTS_HYPOTHETICAL_SCENARIO')
    for base in base_offers:
        source=base.source_terms;rule=source["source_rule_id"]
        if config.delivery_cost_assumption is None and source.get('delivery_cost_status')=='NOT_SUPPLIED':
            issues.append(ProcurementDiagnostic(reason_code='DELIVERY_FEE_REQUIRED',proof_status='BLOCKED',
                store_id=base.store_id,ingredient_id=base.ingredient_id,field_paths=['delivery_cost_assumption'],
                source=source,expected_meaning='explicit zero/free fee or fee amount and scope; missing is not free',
                action_required=['CONFIRM_DELIVERY_FEE_OR_DECLARE_SCENARIO_ASSUMPTION']))
        price_mapping=config.price_basis_mappings.get(rule)
        price_blocked=source.get("price_basis_confirmation_required",False) and price_mapping is None
        if price_blocked:
            issues.append(ProcurementDiagnostic(reason_code="PRICE_BASIS_CONFIRMATION_REQUIRED",proof_status="BLOCKED",
                ingredient_id=base.ingredient_id,source=source,details={"field":"price_basis_mappings","source_rule_id":rule,
                    "reason":"canonical transform defaults price_basis=base_unit; not factual raw evidence"},
                action_required=["CONFIRM_SOURCE_PRICE_IS_PER_BASE_UNIT_OR_PER_PACK"]))
        calendar=config.supplier_calendars.get(base.supplier_id,config.supplier_calendar)
        if calendar.status=="UNRESOLVED" or (calendar.status=="SYNTHETIC" and environment!="SYNTHETIC"):
            issues.append(ProcurementDiagnostic(reason_code="CALENDAR_SEMANTICS_REQUIRED",proof_status="BLOCKED",
                ingredient_id=base.ingredient_id,store_id=base.store_id,source=source,
                details={"field":"supplier_calendar","supplier_id":base.supplier_id,"delivery_schedule":source.get("delivery_schedule")},
                action_required=["CONFIRM_RECEIVING_ORDER_OR_DISPATCH_DAYS_LEAD_CUTOFF_HOLIDAY"]))
        raw_unit=source["order_unit"];mapping_key=rule+"|"+raw_unit
        basis="PACK_COUNT" if raw_unit=="pack" else "BASE_QUANTITY" if raw_unit==base.unit else config.packaging_mappings.get(mapping_key)
        if basis is None:
            issues.append(ProcurementDiagnostic(reason_code="MOQ_PACKAGING_MAPPING_REQUIRED",proof_status="BLOCKED",
                ingredient_id=base.ingredient_id,source=source,details={"field":"packaging_mappings","mapping_key":mapping_key},
                action_required=["CONFIRM_PACK_COUNT_OR_BASE_QUANTITY"]))
        if mapping_key in config.packaging_mappings:used_mappings.add(mapping_key)
        if calendar.status=="UNRESOLVED" or basis is None or price_blocked or (calendar.status=="SYNTHETIC" and environment!="SYNTHETIC"):continue
        weekdays=parse_weekdays(source["delivery_schedule"] or "")
        moq=source["minimum_order_quantity"]*(base.pack_size if basis=="PACK_COUNT" else 1)
        for offset in range((end-base.order_date).days+1):
            order=base.order_date+timedelta(days=offset);arrival=resolve_arrival(order,base.lead_time_days,weekdays,calendar)
            if arrival is None or arrival>end or arrival<=base.order_date:continue
            data=base.model_dump();data.update(offer_id=f"{rule}|{order}|{arrival}",order_date=order,
                unit_price=resolved_price(base,config),
                resolved_arrival_date=arrival,calendar=calendar,delivery_weekdays=weekdays,
                minimum_order_quantity=moq,moq_basis="pack_count" if basis=="PACK_COUNT" else "base_quantity",
                currency=config.currency,delivery_cost=config.delivery_cost_assumption or 0,
                source_terms={**source,"calendar_confirmation_required":False,
                    "price_basis_confirmation_required":False,
                    "price_mapping":price_mapping.model_dump(mode="json") if price_mapping else None,
                    "delivery_fee_scope":config.delivery_fee_scope,
                    "currency_basis":"explicit_planning_currency_pending_business_validation",
                    "delivery_cost_status":"EXPLICIT_DEMO_ASSUMPTION" if config.delivery_cost_assumption is not None else "NOT_SUPPLIED"})
            if preview:
                data['source_terms'].update(profile_id=config.scenario_profile.profile_id,
                    profile_hash=content_hash(config.scenario_profile),
                    assumption_hash=content_hash(config.scenario_profile.assumptions),
                    supplier_confirmation_pending=True,original_lead_time_days=base.lead_time_days,
                    classification='REGULAR_TERMS_UNDER_ASSUMED_INTERPRETATION',
                    assumption_refs=[a.assumption_id for a in config.scenario_profile.assumptions])
            if config.delivery_fee_scope == 'SUPPLIER_STORE_ORDER_ARRIVAL_GROUP':
                data['source_terms']['shipment_group_id']=f'{base.supplier_id}|{base.store_id}|{order}|{arrival}|regular'
            output.append(SupplierOffer.model_validate(data))
        if preview:
            for option in [c for c in config.contract_options if c.source_rule_id==rule]:
                proposed_calendar=type(calendar).model_validate({**calendar.model_dump(),'meaning':'RECEIVING',
                    'lead_time_basis':'CALENDAR_DAYS','order_boundary':'LEAD_FROM_ORDER_DATE',
                    'holiday_scope':'NO_HOLIDAY_EXCEPTIONS','holidays':[],
                    'evidence':'proposed contract:'+option.contract_id})
                for offset in range((end-base.order_date).days+1):
                    order=base.order_date+timedelta(days=offset)
                    arrival=resolve_arrival(order,option.proposed_lead_days,option.receiving_weekdays,proposed_calendar)
                    if arrival is None or arrival>end:continue
                    if option.receiving_dates and arrival not in option.receiving_dates:continue
                    fee=(config.delivery_cost_assumption or 0)+option.extra_fee
                    terms={**source,'calendar_confirmation_required':False,'price_basis_confirmation_required':False,
                        'price_mapping':price_mapping.model_dump(mode='json') if price_mapping else None,
                        'delivery_fee_scope':config.delivery_fee_scope,'delivery_cost_status':'PROPOSED_NOT_CONFIRMED',
                        'original_lead_time_days':base.lead_time_days,'proposed_lead_time_days':option.proposed_lead_days,
                        'original_delivery_schedule':source.get('delivery_schedule'),'proposed_receiving_weekdays':option.receiving_weekdays,
                        'proposed_receiving_dates':[str(d) for d in option.receiving_dates],
                        'contract_id':option.contract_id,'classification':option.classification,
                        'profile_id':config.scenario_profile.profile_id,'profile_hash':content_hash(config.scenario_profile),
                        'assumption_hash':content_hash(config.scenario_profile.assumptions),
                        'assumption_refs':[option.assumption_id],'supplier_confirmation_pending':True}
                    if config.delivery_fee_scope=='SUPPLIER_STORE_ORDER_ARRIVAL_GROUP':
                        terms['shipment_group_id']=f'{base.supplier_id}|{base.store_id}|{order}|{arrival}|contract|fee={fee}'
                    output.append(SupplierOffer.model_validate({**base.model_dump(),
                        'offer_id':f'{rule}|{option.contract_id}|{order}|{arrival}',
                        'order_date':order,'lead_time_days':option.proposed_lead_days,'resolved_arrival_date':arrival,
                        'calendar':proposed_calendar,'delivery_weekdays':option.receiving_weekdays,
                        'minimum_order_quantity':moq,'moq_basis':'pack_count' if basis=='PACK_COUNT' else 'base_quantity',
                        'unit_price':resolved_price(base,config),'delivery_cost':fee,'currency':config.currency,'source_terms':terms}))
    unknown=set(config.packaging_mappings)-used_mappings
    if unknown:raise ValueError("UNKNOWN_PACKAGING_MAPPING_TARGET:"+",".join(sorted(unknown)))
    if config.delivery_cost_assumption is None:
        issues.append(ProcurementDiagnostic(reason_code="DELIVERY_COST_INPUT_REQUIRED",proof_status="BLOCKED",
            details={"field":"delivery_cost_assumption","not_assumed_free":True},action_required=["PROVIDE_FACTUAL_FEE_OR_EXPLICIT_DEMO_ASSUMPTION"]))
    unknown_suppliers=set(config.supplier_calendars)-{o.supplier_id for o in base_offers}
    if unknown_suppliers:raise ValueError("UNKNOWN_SUPPLIER_CALENDAR_TARGET:"+",".join(sorted(unknown_suppliers)))
    unknown_prices=set(config.price_basis_mappings)-{o.source_terms['source_rule_id'] for o in base_offers}
    if unknown_prices:raise ValueError("UNKNOWN_PRICE_MAPPING_TARGET:"+",".join(sorted(unknown_prices)))
    if set(c.source_rule_id for c in config.contract_options)-{o.source_terms['source_rule_id'] for o in base_offers}:
        raise ValueError('UNKNOWN_CONTRACT_SOURCE_RULE')
    return equivalent_opportunity_reduction(output),issues


def map_business_rules(frames, config, costs, store_id, source_path,decision_date=None,planning_end_date=None):
    import pandas as pd
    from shelfcash_forecast.bom.units import UnitConverter
    coverage=[];issues=[];used=set();enforced=set();updated=list(costs)
    source_hash=hashlib.sha256(Path(source_path).read_bytes()).hexdigest() if Path(source_path).is_file() else None
    for index,row in enumerate(frames.get("business_rules",pd.DataFrame()).to_dict("records"),2):
        ingredient=None if pd.isna(row.get("ingredient_id")) else str(row["ingredient_id"])
        source={"file":str(source_path),"sha256":source_hash,"row":index,"rule_type":row["rule_type"],"ingredient_id":ingredient,
                "value":row["value"],"unit":row["unit"],"application_status":row["application_status"],
                "effective_from":str(row.get("effective_from")),"store_scope":"sealed_bundle_context:"+store_id}
        mapping=next((m for m in config.rule_mappings if m.rule_type==row["rule_type"] and m.ingredient_id==ingredient),None)
        status="UNRESOLVED_MAPPING";reason="BUSINESS_RULE_SEMANTICS_REQUIRED"
        if mapping is not None:
            used.add((mapping.rule_type,mapping.ingredient_id))
            effective=None if pd.isna(row.get('effective_from')) else date.fromisoformat(str(row['effective_from'])[:10])
            if effective is not None and planning_end_date is not None and effective>planning_end_date:
                status='NOT_APPLICABLE_AFTER_HORIZON'
                coverage.append({**source,'mapping_status':status,'business_validated':False})
                continue
            if row["rule_type"]!="maximum_stock" or ingredient is None:
                raise ValueError("UNSUPPORTED_RULE_MAPPING:"+row["rule_type"])
            if ingredient in enforced:
                raise ValueError("MULTIPLE_CAPACITY_RULES_REQUIRE_INTERVAL_MAPPING:"+ingredient)
            enforced.add(ingredient)
            if mapping.unit!=row["unit"]:raise ValueError("RULE_MAPPING_UNIT_MISMATCH")
            matched=[i for i,c in enumerate(updated) if c.store_id==store_id and c.ingredient_id==ingredient]
            if not matched:raise ValueError("RULE_MAPPING_TARGET_NOT_FOUND")
            for i in matched:
                c=updated[i];q=float(row["value"])*UnitConverter().conversion_factor(ingredient,row["unit"],c.unit)
                updated[i]=ConsequenceCostAssumption.model_validate({**c.model_dump(),"capacity_quantity":q,
                    "capacity_effective_from":effective,
                    "capacity_effective_to":None if pd.isna(row.get('effective_to')) else date.fromisoformat(str(row['effective_to'])[:10])})
            status="ENFORCED_RECEIVING_PEAK_PENDING_BUSINESS_VALIDATION";reason=None
        coverage.append({**source,"mapping_status":status,"business_validated":False,"semantics":mapping.semantics if mapping else None})
        if reason:issues.append(ProcurementDiagnostic(reason_code=reason,proof_status="BLOCKED",ingredient_id=ingredient,
            source=source,details={"field":"rule_mappings","mapping_status":status},action_required=["CONFIRM_RULE_SCOPE_UNIT_EFFECTIVE_DATE_AND_GRAIN"]))
    if {(m.rule_type,m.ingredient_id) for m in config.rule_mappings}-used:raise ValueError("UNUSED_RULE_MAPPING")
    return updated,coverage,issues


def prepare_bundle_requests(bundle, *, planning, lots, snapshot, policy, scenarios, decision_date,
                            planning_end_date, seed, optimization_mode, execution_mode):
    from shelfcash_preprocess.engine import create_supplier_offers,load_canonical_frames
    from shelfcash_preprocess.pipeline import load_bundle
    config,migration=load_planning_config(planning,scenario_count=len(scenarios),seed=seed)
    if execution_mode=='synthetic':
        raise ValueError('ACTUAL_BUNDLE_CANNOT_BE_RELABELLED_SYNTHETIC')
    declared_methods={s.provenance.get('scenario_method') for s in scenarios if s.provenance.get('scenario_method')}
    if declared_methods and declared_methods != {config.scenario_method}:
        raise ValueError('M4_SCENARIO_METHOD_CONFIG_MISMATCH:scenario_method')
    if execution_mode=="demo" and config.label!="DEMO_ONLY_NOT_FOR_OPERATION":raise ValueError("DEMO_PLANNING_CONFIG_LABEL_REQUIRED")
    count=min(config.optimization_scenario_count,len(scenarios))
    if count<2 and optimization_mode in {"stochastic","compare"}:raise ValueError("STOCHASTIC_OPTIMIZATION_REQUIRES_AT_LEAST_TWO_SCENARIOS")
    selected,mass=select_scenarios(scenarios,count,config.selection_seed)
    environment={"demo":"DEMO","backtest_replay":"BACKTEST","production":"PRODUCTION","synthetic":"SYNTHETIC"}[execution_mode]
    base=create_supplier_offers(bundle,decision_date);offers,issues=create_order_opportunities(base,planning_end_date,config,environment)
    # Cost references use the same normalized price basis as actual opportunities.
    costs=cost_assumptions([SupplierOffer.model_validate({**o.model_dump(),'unit_price':resolved_price(o,config)}) for o in base],config)
    frames=load_canonical_frames(bundle);info=load_bundle(bundle)
    rules=[];binding={}
    if config.planning_mode=='SCENARIO_PREVIEW':
        from shelfcash_forecast.optimization.business_rules import normalize_preview_rules
        rules,coverage=normalize_preview_rules(frames,config,info.manifest.context.store_id,
            info.canonical_files['business_rules'],scenarios,planning_end_date)
        rule_issues=[]
        binding={'profile_id':config.scenario_profile.profile_id,'profile_version':config.scenario_profile.version,
            'profile_hash':content_hash(config.scenario_profile),'assumption_hash':content_hash(config.scenario_profile.assumptions),
            'assumptions':[a.model_dump(mode='json') for a in config.scenario_profile.assumptions],
            'contracts_hash':content_hash({'offers':[o.model_dump(mode='json') for o in offers],
                                           'rules':[r.model_dump(mode='json') for r in rules]})}
    else:
        costs,coverage,rule_issues=map_business_rules(frames,config,costs,info.manifest.context.store_id,
                                                    info.canonical_files.get("business_rules",""),decision_date,planning_end_date)
    issues+=rule_issues
    profiles=config.strategy_profiles or default_strategy_profiles()
    conversions=[]
    if "unit_conversions" in frames:
        from shelfcash_forecast.bom.contracts import UnitConversionRule
        conversions=[UnitConversionRule.model_validate(row) for row in frames["unit_conversions"].to_dict("records")]
    provenance={"full_pool_hash":content_hash([s.model_dump(mode="json") for s in scenarios]),
        "optimization_ids":[s.scenario_id for s in selected],"evaluation_ids":[s.scenario_id for s in scenarios],
        "original_evaluation_weights":{s.scenario_id:s.probability_weight for s in scenarios},
        "optimization_weights":{s.scenario_id:s.probability_weight for s in selected},"selected_original_mass":mass,
        "selection_seed":config.selection_seed,"selection":"uniform_without_replacement_conditional_weights",
        "evaluation":"FULL_M4_POOL_MODEL_DERIVED_NOT_INDEPENDENT_OOS","M4_seed":seed}
    provenance['planning_binding']=binding
    modes=["deterministic","stochastic"] if optimization_mode=="compare" else [optimization_mode]
    requests={mode:OptimizationRequest(request_id=f"{info.manifest.run_id}-{mode}",decision_date=decision_date,
        planning_end_date=planning_end_date,initial_inventory=lots,inventory_snapshot_date=snapshot,inventory_snapshot_boundary="EOD",
        demand_scenarios=selected,evaluation_scenarios=scenarios,supplier_offers=offers,cost_assumptions=costs,unit_conversions=conversions,
        inventory_policy=policy,strategy_profiles=profiles,budget=config.budget,budget_scope=config.budget_scope,currency=config.currency,
        seed=seed,limits=config.limits,stochastic=mode=="stochastic",allow_mode_fallback=False,
        business_issues=["DEMO_CONSEQUENCE_COSTS_NOT_APPROVED","BUSINESS_RULES_PENDING_VALIDATION","MODEL_NOT_PROMOTED_FOR_PRODUCTION"],
        blocked_issues=issues,rule_coverage=coverage,environment=environment,scenario_provenance=provenance,
        entity_display_names={'ingredients':{str(r['ingredient_id']):str(r['ingredient_name']) for name in ['inventory_snapshot','supplier_rules','recipes'] if name in frames
            for r in frames[name].to_dict('records')}},
        planning_mode=config.planning_mode,planning_binding=binding,normalized_rules=rules,
        candidate_generation=config.candidate_generation,
        stress_scenarios=config.stress_scenarios,stress_base_scenario_id=config.stress_base_scenario_id) for mode in modes}
    return requests,config,{"migration":migration,"rule_mapping":coverage,"scenario_provenance":provenance,
                           "config_hash":content_hash(config),"source_bundle":str(bundle),
                           "resolved_sources":{"scenario_count":"caller_M4_actual_pool","seed":"caller_M4_seed","budget":"validated_planning_config"}}


@dataclass
class PlanningRun:
    selected_mode: str
    request: OptimizationRequest
    result: object
    runs: dict
    config: PlanningConfig
    provenance: dict


def run_bundle_planning(bundle, **kwargs):
    import os,logging
    if os.environ.get('SHELFCASH_PLANNING_PROGRESS')=='1':
        logging.basicConfig(level=logging.INFO,format='%(message)s')
    from shelfcash_forecast.optimization.optimizer import optimize_procurement
    requests,config,provenance=prepare_bundle_requests(bundle,**kwargs)
    runs={mode:(r,optimize_procurement(r,structured_errors=True)) for mode,r in requests.items()}
    selected=next((mode for mode in ("stochastic","deterministic") if mode in runs and runs[mode][1].recommended_strategy is not None),
                  "stochastic" if "stochastic" in runs else next(iter(runs)))
    provenance["compare_selection_policy"]="accepted_stochastic_then_accepted_deterministic_else_stochastic_diagnostics"
    r,result=runs[selected]
    return PlanningRun(selected,r,result,runs,config,provenance)


def run_typed_planning(request, *, destination=None):
    """Public BE/Python/CLI path; natural-language adapters are optional."""
    from shelfcash_forecast.optimization.optimizer import optimize_procurement
    from shelfcash_forecast.optimization.export import export_planning_run
    from shelfcash_forecast.decision_intelligence.service import build_final_decision_package
    from shelfcash_forecast.json_output import write_json
    result=optimize_procurement(request,structured_errors=True)
    if result.technical_outcome=='INVALID_INPUT':
        if destination is not None:
            folder=Path(destination);folder.mkdir(parents=True,exist_ok=False)
            write_json(folder/'optimization_result.json',result)
            write_json(folder/'customer_diagnostic_package.json',{'schema_version':2,'technical_outcome':result.technical_outcome,
                'orders':[],'accepted_zero_purchase':False,'diagnostics':result.diagnostics,'business_ready':False,'execution_authorized':False})
            (folder/'CUSTOMER_DIAGNOSTICS_VI.md').write_text('INVALID_INPUT; chưa lập plan.\n'+ '\n'.join(d.reason_code+': '+str(d.field_paths)+'; '+str(d.expected_meaning) for d in result.diagnostics),encoding='utf-8')
            from shelfcash_forecast.optimization.customer_export import export_invalid_customer_diagnostics
            try:
                status=export_invalid_customer_diagnostics(result,folder)
            except Exception as exc:
                status={'status':'EXPORT_FAILED','customer_export_complete':False,'technical_result_preserved':True,
                    'error':{'code':'CUSTOMER_EXPORT_FAILED','type':type(exc).__name__,'message':str(exc)}}
            write_json(folder/'customer_export_status.json',status)
        return result
    typed=OptimizationRequest.model_validate(request)
    mode=result.provenance.get('actual_mode','stochastic' if typed.stochastic else 'deterministic')
    run=PlanningRun(mode,typed,result,{mode:(typed,result)},typed,{'compare_selection_policy':'BALANCED_then_PROTECTED_then_LEAN_if_valid','data_origin':'TYPED_REQUEST_DECLARED_INPUT'})
    if destination is not None:
        export_planning_run(run,destination)
        decision=build_final_decision_package(typed,result)
        write_json(Path(destination)/'decision_package.json',decision)
    return run
