"""Accepted-only technical procurement tables, independent of M6 explanations."""
from __future__ import annotations
import csv
import json
from pathlib import Path

from shelfcash_forecast.optimization.chronology import expiry_date
from shelfcash_forecast.optimization.planning_service import content_hash
from shelfcash_forecast.json_output import write_json as _write_json


def write_json(path, value):
    _write_json(path,value,fallback_str=True)


ORDER_FIELDS=["store_id","ingredient_id","ingredient_name","supplier_id","supplier_name","source_rule_id",
    "opportunity_id","mode","strategy","decision_stage","order_date","arrival_date","expiry_date","quantity","unit",
    "pack_size","pack_count","MOQ_source_basis","MOQ_normalized","unit_price","price_basis","currency",
    "purchase_cost","allocated_delivery_cost","total_procurement_cost","technical_status","business_ready",
    "execution_authorized","operational_status","provenance_ref"]


def _csv(path,fields,rows):
    with Path(path).open("w",encoding="utf-8-sig",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=fields,extrasaction="ignore");writer.writeheader()
        for row in rows:
            writer.writerow({k:"'"+v if isinstance(v,str) and v.startswith(('=','+','-','@')) else v for k,v in row.items()})


def export_planning_run(run,destination, *, include_full_result=True, compact_attempts=False):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    request,result=run.request,run.result
    selected=result.evaluations.get(result.recommended_strategy) if result.recommended_strategy else None
    rows=[];offers={o.offer_id:o for o in request.supplier_offers}
    if selected:
        if not selected.critic.passed or not selected.plan.completed or selected.simulation is None:
            raise ValueError("EXPORT_REQUIRES_EXACT_CRITIC_ACCEPTANCE")
        for line in selected.plan.orders:
            offer=offers[line.offer_id]
            rows.append({"store_id":line.store_id,"ingredient_id":line.ingredient_id,
                "ingredient_name":request.entity_display_names.get('ingredients',{}).get(f'{line.store_id}|{line.ingredient_id}',request.entity_display_names.get('ingredients',{}).get(line.ingredient_id,offer.source_terms.get("ingredient_name",line.ingredient_id))),"supplier_id":line.supplier_id,
                "supplier_name":request.entity_display_names.get('suppliers',{}).get(f'{line.store_id}|{line.supplier_id}',request.entity_display_names.get('suppliers',{}).get(line.supplier_id,offer.source_terms.get("supplier_name",line.supplier_id))),"source_rule_id":offer.source_terms.get("source_rule_id"),
                "opportunity_id":line.offer_id,"mode":run.selected_mode,"strategy":line.strategy if hasattr(line,"strategy") else selected.plan.strategy,
                "decision_stage":"IMMEDIATE_ADVICE" if line.order_date<=request.decision_date else "SCHEDULED_COMMITMENT",
                "order_date":str(line.order_date),"arrival_date":str(line.arrival_date),
                "expiry_date":str(expiry_date(line.arrival_date,line.shelf_life_days)) if line.shelf_life_days is not None else None,
                "quantity":line.order_quantity,"unit":line.unit,"pack_size":line.pack_size,"pack_count":line.pack_count,
                "MOQ_source_basis":offer.moq_basis,"MOQ_normalized":offer.minimum_order_quantity,"unit_price":line.unit_price,
                "price_basis":offer.price_basis,"currency":request.currency or offer.currency,"purchase_cost":line.purchase_cost,
                "allocated_delivery_cost":line.delivery_cost,"total_procurement_cost":line.purchase_cost+line.delivery_cost,
                "technical_status":result.technical_outcome,"business_ready":result.business_ready,"execution_authorized":False,
                "operational_status":result.operational_status,"provenance_ref":"scenario_provenance.json"})
    diagnostics=[d.model_dump(mode="json") for d in result.diagnostics]
    for mode,(_,mode_result) in run.runs.items():
        for strategy,e in mode_result.evaluations.items():
            if e.critic.passed:continue
            if e.simulation:
                for world in e.simulation.results:
                    for ledger in world.daily_ledgers:
                        if ledger.shortage_quantity>request.inventory_policy.accounting_tolerance or ledger.capacity_violation_quantity>0:
                            diagnostics.append({"reason_code":"EXACT_SHORTAGE" if ledger.shortage_quantity>0 else "EXACT_CAPACITY_VIOLATION",
                                "proof_status":"OBSERVED","mode":mode,"strategy":strategy,"store_id":ledger.store_id,
                                "ingredient_id":ledger.ingredient_id,"unit":ledger.unit,"target_date":str(ledger.simulation_date),
                                "scenario_id":world.scenario_id,"shortage":ledger.shortage_quantity,
                                "capacity_violation":ledger.capacity_violation_quantity,"critic_violations":e.critic.hard_violations})
    accepted={"schema_version":2,"request_id":request.request_id,"selected_mode":run.selected_mode,
        "technical_outcome":result.technical_outcome,"technical_feasible":result.technical_feasible,
        "business_ready":result.business_ready,"execution_authorized":False,"environment":request.environment,
        "operational_status":result.operational_status,"selected_plan":selected.plan.model_dump(mode="json") if selected else None,
        "accepted_zero_purchase":bool(selected is not None and not rows),"orders":rows,"budget":request.budget,
        "currency":request.currency,"budget_scope":request.budget_scope,"request_hash":content_hash(request),
        "config_hash":content_hash(run.config),"input_scope":request.scenario_provenance,
        "rule_coverage":request.rule_coverage,"business_pending":request.business_issues,
        "critic":selected.critic.model_dump(mode="json") if selected else None,
        "evaluation_metrics":selected.simulation.risk_metrics.model_dump(mode="json") if selected and selected.simulation.risk_metrics else None,
        "recommendation_rule":result.provenance.get("recommendation_rule"),"selection":run.provenance.get("compare_selection_policy"),
        "missing_display_names":"null means source has no distinct display-name field"}
    write_json(destination/"accepted_technical_plan.json",accepted);_csv(destination/"accepted_technical_orders.csv",ORDER_FIELDS,rows)
    write_json(destination/"procurement_diagnostics.json",{"outcome":result.technical_outcome,"diagnostics":diagnostics})
    fields=sorted({k for d in diagnostics for k in d}) or ["reason_code","proof_status","ingredient_id","target_date"]
    _csv(destination/"procurement_diagnostics.csv",fields,[{k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in d.items()} for d in diagnostics])
    if include_full_result:
        write_json(destination/"optimization_result.json",{mode:{"request":r,"result":v} for mode,(r,v) in run.runs.items()})
    write_json(destination/"resolved_planning_config.json",{"config":run.config.model_dump(mode="json"),"provenance":run.provenance})
    write_json(destination/"planning_config.json",run.config.model_dump(mode="json"))
    write_json(destination/"rule_mapping_report.json",request.rule_coverage)
    write_json(destination/"scenario_provenance.json",request.scenario_provenance)
    write_json(destination/"preflight.json",{mode:v.provenance.get("preflight") for mode,(_,v) in run.runs.items()})
    for mode,(_,v) in run.runs.items():
        for strategy,e in v.evaluations.items():
            folder=destination/"attempts"/mode/strategy;folder.mkdir(parents=True,exist_ok=True)
            for attempt in e.attempts:
                artifact={k:val for k,val in attempt.items() if k not in {'physics_simulation','evaluation_simulation'}} if compact_attempts else attempt
                write_json(folder/(f"attempt_{attempt['attempt']:03d}.json"),artifact)
    write_json(destination/"summary.json",{"requested_mode":"compare" if len(run.runs)>1 else run.selected_mode,
        "selected_mode":run.selected_mode,"technical_outcome":result.technical_outcome,"technical_feasible":result.technical_feasible,
        "optimization_scenario_count":len(request.demand_scenarios),"m4_diagnostic_scenario_count":len(request.evaluation_scenarios or request.demand_scenarios),
        "evaluation_scenario_count":len(selected.simulation.results) if selected else None,"optimizer_called":result.provenance.get("optimizer_called"),
        "business_ready":False,"execution_authorized":False,"accepted_order_count":len(rows),"accepted_zero_purchase":accepted["accepted_zero_purchase"],
        "results":{m:{"status":v.status,"recommended_strategy":v.recommended_strategy,"technical_outcome":v.technical_outcome,"actual_mode":v.provenance.get("actual_mode"),"warnings":v.warnings} for m,(_,v) in run.runs.items()}})
    from shelfcash_forecast.optimization.customer_export import export_customer_plan
    try:
        customer=export_customer_plan(run,destination,accepted)
        export_status={'status':'COMPLETE','customer_export_complete':True,'schema_version':customer.schema_version}
    except (ValueError,OSError,KeyError,TypeError) as exc:
        export_status={'status':'EXPORT_FAILED','customer_export_complete':False,'error':{'code':'CUSTOMER_EXPORT_FAILED','type':type(exc).__name__,'message':str(exc)},
            'technical_result_preserved':True,'rerun':'reuse bound request/result and export into a fresh destination; no optimizer/order execution required'}
    write_json(destination/'customer_export_status.json',export_status)
    accepted['customer_export']=export_status
    write_json(destination/'accepted_technical_plan.json',accepted)
    if request.planning_mode=='SCENARIO_PREVIEW':
        from shelfcash_forecast.optimization.scenario_contracts import ProfileRunResult
        record=ProfileRunResult(profile_id=request.planning_binding['profile_id'],profile_version=request.planning_binding['profile_version'],
            input_hash=request.scenario_provenance.get('full_pool_hash',content_hash(request.evaluation_scenarios or request.demand_scenarios)),config_hash=content_hash(run.config),
            assumption_hash=request.planning_binding['assumption_hash'],budget_case='null' if request.budget is None else str(request.budget),
            technical_outcome=result.technical_outcome,selected_plan_id=selected.plan.plan_id if selected else None,
            critic_passed=selected.critic.passed if selected else False,evaluation_coverage=len(selected.simulation.results) if selected else 0,
            evidence_directory=str(destination))
        write_json(destination/'profile_run_result.json',record.model_dump(mode='json'))
    return accepted
