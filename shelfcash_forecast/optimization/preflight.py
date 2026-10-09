"""Sound optimistic supply bounds. Passing these checks is not a feasible certificate."""
from __future__ import annotations
from collections import defaultdict
from datetime import timedelta

from shelfcash_forecast.bom.units import UnitConverter
from shelfcash_forecast.optimization.chronology import offer_arrival, expiry_date
from shelfcash_forecast.optimization.contracts import ProcurementDiagnostic
from shelfcash_forecast.exceptions import BOMError


def preflight(request, profiles):
    diagnostics=list(request.blocked_issues)
    if request.normalized_rules and any(s.probability_weight is None for s in (request.evaluation_scenarios or request.demand_scenarios)):
        diagnostics.append(ProcurementDiagnostic(reason_code='NORMALIZED_RULES_REQUIRE_DECLARED_WORLD_WEIGHTS',proof_status='BLOCKED',
            field_paths=['evaluation_scenarios.probability_weight','demand_scenarios.probability_weight'],
            expected_meaning='normalized rule reports and expected-fill/soft losses require explicit nonnegative world weights summing to one',
            action_required=['DECLARE_SCENARIO_WEIGHT_SEMANTICS']))
    if not request.demand_scenarios:
        diagnostics.append(ProcurementDiagnostic(reason_code="DEMAND_SCOPE_REQUIRED",proof_status="BLOCKED"))
    if request.stress_base_scenario_id is not None and request.stress_base_scenario_id not in {
        s.scenario_id for s in (request.evaluation_scenarios or request.demand_scenarios)}:
        diagnostics.append(ProcurementDiagnostic(reason_code="STRESS_BASE_SCENARIO_NOT_FOUND",proof_status="BLOCKED"))
    if any(d.arrival_date <= request.decision_date for d in request.existing_inbound):
        diagnostics.append(ProcurementDiagnostic(reason_code="INBOUND_BEFORE_TRANSITION_MUST_BE_IN_SNAPSHOT",proof_status="BLOCKED"))
    if any(o.emergency for o in request.supplier_offers):
        diagnostics.append(ProcurementDiagnostic(reason_code="RECOURSE_POLICY_NOT_SUPPORTED",proof_status="BLOCKED",
            action_required=["PROVIDE_SUPPORTED_CAUSAL_RECOURSE_POLICY"]))
    if request.inventory_snapshot_boundary != "EOD":
        diagnostics.append(ProcurementDiagnostic(reason_code="BOD_SNAPSHOT_NOT_SUPPORTED",proof_status="BLOCKED"))
    for offer in request.supplier_offers:
        if offer.source_terms.get('price_basis_confirmation_required'):
            diagnostics.append(ProcurementDiagnostic(reason_code='PRICE_BASIS_CONFIRMATION_REQUIRED',proof_status='BLOCKED',
                source={'offer_id':offer.offer_id},action_required=['CONFIRM_SOURCE_PRICE_BASIS']))
        if offer.source_terms.get("delivery_fee_scope", "PER_INGREDIENT_ORDER_OPPORTUNITY") not in {'PER_INGREDIENT_ORDER_OPPORTUNITY','SUPPLIER_STORE_ORDER_ARRIVAL_GROUP'}:
            diagnostics.append(ProcurementDiagnostic(reason_code="UNSUPPORTED_DELIVERY_FEE_SCOPE",proof_status="BLOCKED",
                source={"offer_id":offer.offer_id},action_required=["PROVIDE_SUPPORTED_FEE_SCOPE_OR_IMPLEMENT_SHIPMENT_GROUPING"]))
        if offer.source_terms.get("calendar_confirmation_required") and (offer.calendar is None or offer.calendar.status=="UNRESOLVED"):
            diagnostics.append(ProcurementDiagnostic(reason_code="CALENDAR_SEMANTICS_REQUIRED",proof_status="BLOCKED",
                ingredient_id=offer.ingredient_id,source={"offer_id":offer.offer_id},action_required=["CONFIRM_SUPPLIER_CALENDAR"]))
    for lot in request.initial_inventory:
        if lot.received_date is not None and lot.received_date>request.decision_date:
            diagnostics.append(ProcurementDiagnostic(reason_code="INITIAL_LOT_RECEIVED_AFTER_SNAPSHOT",proof_status="BLOCKED",source={"lot_id":lot.lot_id}))
    if request.inventory_policy.unknown_expiry=="reject":
        if any(l.expiry_date is None for l in request.initial_inventory) or any(d.expiry_date is None for d in request.existing_inbound):
            diagnostics.append(ProcurementDiagnostic(reason_code="UNKNOWN_EXPIRY_REQUIRED",proof_status="BLOCKED",action_required=["PROVIDE_FACTUAL_EXPIRY_OR_EXPLICIT_POLICY"]))
    if request.unknown_constraints:
        cost_keys={(c.store_id,c.ingredient_id) for c in request.cost_assumptions}
        demand_keys={(l.store_id,l.ingredient_id) for s in request.demand_scenarios for l in s.lines}
        for name in request.unknown_constraints:
            if name=="DEMO_CONSEQUENCE_COSTS_NOT_APPROVED" and request.environment in {"DEMO","SYNTHETIC"} and demand_keys<=cost_keys:
                continue
            diagnostics.append(ProcurementDiagnostic(reason_code="UNKNOWN_CONSTRAINT:"+name,proof_status="BLOCKED"))
    targets={(l.store_id,l.ingredient_id):l.unit for s in request.demand_scenarios for l in s.lines}
    converter=UnitConverter(request.unit_conversions)
    for row in [*request.initial_inventory,*request.existing_inbound,*request.supplier_offers]:
        targets.setdefault((row.store_id,row.ingredient_id),converter.canonical_unit(row.ingredient_id,row.unit))
    for rule in request.normalized_rules:
        if rule.semantics=='GLOBAL_RECEIVING_PEAK':
            coefficients={c.ingredient_id:c for c in rule.occupancy}
            scoped={k:u for k,u in targets.items() if k[0]==rule.store_id}
            missing=[k[1] for k,u in scoped.items() if k[1] not in coefficients or
                     coefficients[k[1]].base_unit!=converter.canonical_unit(k[1],u)]
            if missing:diagnostics.append(ProcurementDiagnostic(reason_code='GLOBAL_OCCUPANCY_COVERAGE_OR_UNIT_MISMATCH',proof_status='BLOCKED',
                field_paths=['normalized_rules.'+rule.rule_id+'.occupancy'],store_id=rule.store_id,
                expected_meaning='all storage keys, including zero-demand initial/inbound inventory, require scoped liters/base-unit',
                details={'rule_id':rule.rule_id,'missing_keys':missing}))
    for row in [*request.initial_inventory,*request.existing_inbound,*request.supplier_offers,*request.cost_assumptions]:
        key=(row.store_id,row.ingredient_id)
        if key not in targets:targets[key]=converter.canonical_unit(row.ingredient_id,row.unit)
        try:
            converter.conversion_factor(row.ingredient_id,row.unit,targets[key])
        except BOMError as exc:
            diagnostics.append(ProcurementDiagnostic(reason_code="UNSUPPORTED_UNIT",proof_status="BLOCKED",
                store_id=row.store_id,ingredient_id=row.ingredient_id,unit=row.unit,
                details={"target_unit":targets[key],"error":str(exc)},action_required=["PROVIDE_SCOPED_UNIT_CONVERSION"]))
    if any(d.proof_status=="BLOCKED" for d in diagnostics):
        fields={'CALENDAR_SEMANTICS_REQUIRED':'supplier_calendar','PRICE_BASIS_CONFIRMATION_REQUIRED':'price_basis_mappings',
            'DELIVERY_FEE_MISSING':'delivery_cost_assumption','MOQ_MAPPING_REQUIRED':'packaging_mappings',
            'UNKNOWN_EXPIRY_REQUIRED':'initial_inventory.expiry_date','BOD_SNAPSHOT_NOT_SUPPORTED':'inventory_snapshot_boundary',
            'RECOURSE_POLICY_NOT_SUPPORTED':'supplier_offers.emergency','DEMAND_SCOPE_REQUIRED':'demand_scenarios',
            'INBOUND_BEFORE_TRANSITION_MUST_BE_IN_SNAPSHOT':'existing_inbound.arrival_date'}
        diagnostics=[d.model_copy(update={'field_paths':d.field_paths or [fields.get(d.reason_code,'blocked_issues.'+str(i))],
            'expected_meaning':d.expected_meaning or '; '.join(d.action_required) or d.reason_code}) for i,d in enumerate(diagnostics)]
        return {"status":"BLOCKED_INPUT_SEMANTICS","diagnostics":diagnostics,"infeasible_profiles":[],"proof_scope":"unresolved_input_not_physical_infeasibility"}
    converter=UnitConverter(request.unit_conversions);worlds=request.evaluation_scenarios or request.demand_scenarios
    bounds=defaultdict(float);totals=defaultdict(float);stockout_worlds=set()
    for world in worlds:
        for line in world.lines:
            key=(line.store_id,line.ingredient_id);day=line.target_date
            def convert(q,u):return q*converter.conversion_factor(line.ingredient_id,u,line.unit)
            def usable(expiry):return expiry is None or expiry>=day if request.inventory_policy.expiry_inclusive else expiry is None or expiry>day
            base=sum(convert(l.quantity_remaining,l.unit) for l in request.initial_inventory if (l.store_id,l.ingredient_id)==key and usable(l.expiry_date))
            base+=sum(convert(d.quantity,d.unit) for d in request.existing_inbound if (d.store_id,d.ingredient_id)==key and d.arrival_date<=day and usable(d.expiry_date))
            opportunities=[];arrivals=[];expiries=[];unbounded=False
            for o in request.supplier_offers:
                if (o.store_id,o.ingredient_id)!=key or not o.available or o.order_date<request.decision_date or (o.order_cutoff_date is not None and o.order_date>o.order_cutoff_date):continue
                arrival=offer_arrival(o);expiry=expiry_date(arrival,o.shelf_life_days)
                arrivals.append(arrival)
                if expiry is not None:expiries.append(expiry)
                if arrival<=day and usable(expiry):
                    if o.shelf_life_days is None and request.inventory_policy.unknown_expiry=="reject":
                        diagnostics.append(ProcurementDiagnostic(reason_code="SUPPLIER_EXPIRY_REQUIRED",proof_status="BLOCKED",ingredient_id=line.ingredient_id,source={"offer_id":o.offer_id}));continue
                    opportunities.append(o)
                    if o.maximum_order_quantity is None:unbounded=True
                    else:base+=convert(o.maximum_order_quantity,o.unit)
            lower=0 if unbounded else max(0,line.quantity-base)
            totals[(world.scenario_id,key)]+=line.quantity;bounds[(world.scenario_id,key)]+=lower
            if lower>request.inventory_policy.accounting_tolerance:
                stockout_worlds.add(world.scenario_id)
                diagnostics.append(ProcurementDiagnostic(reason_code="DEMAND_BEFORE_FIRST_ARRIVAL" if arrivals and day<min(arrivals) else "NO_USABLE_LOT",
                    proof_status="SOUND_BOUND",store_id=key[0],ingredient_id=key[1],unit=line.unit,target_date=day,scenario_id=world.scenario_id,
                    required_quantity=line.quantity,usable_quantity_upper_bound=base,shortfall_lower_bound=lower,
                    earliest_valid_arrival=min(arrivals,default=None),last_usable_expiry=max(expiries,default=None),
                    next_valid_arrival=min((a for a in arrivals if a>day),default=None),
                    action_required=["CONFIRMED_LATER_DELIVERY_OR_EXISTING_INBOUND_OR_REAL_SOURCE"],
                    details={"bound":"optimistic ignores earlier consumption, packs and supplier coupling","not_global_infeasibility":True}))
    # A daily comparison against the same snapshot can miss cumulative depletion
    # before the first receipt. This optimistic prefix bound counts initial stock
    # once, ignores its expiry, and grants all factual inbound before the checkpoint.
    # It is valid even though pack/capacity/expiry constraints are relaxed.
    for world in worlds:
        world_keys={(l.store_id,l.ingredient_id,l.unit) for l in world.lines}
        for store,ingredient,unit in world_keys:
            key=(store,ingredient)
            valid=[o for o in request.supplier_offers if (o.store_id,o.ingredient_id)==key and o.available and
                o.order_date>=request.decision_date and (o.order_cutoff_date is None or o.order_date<=o.order_cutoff_date)]
            earliest=min((offer_arrival(o) for o in valid),default=request.planning_end_date+timedelta(days=1))
            prefix=[l for l in world.lines if (l.store_id,l.ingredient_id)==key and l.target_date<earliest]
            if not prefix:continue
            checkpoint=max(l.target_date for l in prefix)
            q=sum(l.quantity for l in prefix)
            stock=sum(l.quantity_remaining*converter.conversion_factor(ingredient,l.unit,unit)
                for l in request.initial_inventory if (l.store_id,l.ingredient_id)==key)
            stock+=sum(d.quantity*converter.conversion_factor(ingredient,d.unit,unit) for d in request.existing_inbound
                if (d.store_id,d.ingredient_id)==key and d.arrival_date<=checkpoint)
            lower=max(0,q-stock)
            if lower>request.inventory_policy.accounting_tolerance:
                bounds[(world.scenario_id,key)]=max(bounds[(world.scenario_id,key)],lower)
                stockout_worlds.add(world.scenario_id)
                diagnostics.append(ProcurementDiagnostic(reason_code='CUMULATIVE_DEMAND_BEFORE_FIRST_ARRIVAL',proof_status='SOUND_BOUND',
                    store_id=store,ingredient_id=ingredient,unit=unit,target_date=checkpoint,scenario_id=world.scenario_id,
                    required_quantity=q,usable_quantity_upper_bound=stock,shortfall_lower_bound=lower,
                    earliest_valid_arrival=earliest,details={'bound':'initial counted once; expiry/pack/capacity relaxed',
                    'scope':'declared opportunities and prefix, not global supplier impossibility'},
                    action_required=['EARLIER_SOURCE_BACKED_DELIVERY_REQUIRES_SUPPLIER_CONFIRMATION']))
    infeasible=[];budget_bounds={}
    weighted=all(s.probability_weight is not None for s in worlds)
    lower_probability=sum(float(s.probability_weight) for s in worlds if s.scenario_id in stockout_worlds) if weighted else None
    for p in profiles:
        violated=any(1-bounds[k]/total+1e-9<p.minimum_acceptable_fill_rate for k,total in totals.items() if total>0)
        ceiling=min(p.maximum_acceptable_stockout_probability,p.maximum_stockout_probability if p.maximum_stockout_probability is not None else 1)
        if lower_probability is not None and lower_probability>ceiling+1e-9:violated=True
        if p.minimum_expected_fill_rate is not None and weighted:
            for key in {k[1] for k in totals}:
                upper_fill=sum(float(s.probability_weight)*(1-bounds[(s.scenario_id,key)]/totals[(s.scenario_id,key)] if totals[(s.scenario_id,key)] else 1) for s in worlds)
                if upper_fill+1e-9<p.minimum_expected_fill_rate:violated=True
        if request.budget is not None:
            # Optimistic necessary purchase-cost bound: allow all starting/inbound
            # stock regardless of expiry, ignore packs, MOQ, fees and capacity.
            # Universal per-world/key fill floor supplies a valid minimum quantity.
            lower_cash=0.0;derivation=[]
            for key in {k[1] for k in totals}:
                lines=[l for s in worlds for l in s.lines if (l.store_id,l.ingredient_id)==key]
                unit=lines[0].unit
                factual=sum(l.quantity_remaining*converter.conversion_factor(key[1],l.unit,unit)
                            for l in request.initial_inventory if (l.store_id,l.ingredient_id)==key)
                factual+=sum(d.quantity*converter.conversion_factor(key[1],d.unit,unit)
                             for d in request.existing_inbound if (d.store_id,d.ingredient_id)==key and d.arrival_date<=request.planning_end_date)
                quantity=max(0,max(totals[(s.scenario_id,key)]*p.minimum_acceptable_fill_rate for s in worlds)-factual)
                prices=[o.unit_price/converter.conversion_factor(key[1],o.unit,unit) for o in request.supplier_offers
                        if (o.store_id,o.ingredient_id)==key and o.available and offer_arrival(o)<=request.planning_end_date]
                if quantity and not prices:continue  # separately diagnosed structural supply gap
                cost=quantity*min(prices,default=0);lower_cash+=cost
                derivation.append({"key":key,"unit":unit,"minimum_quantity":quantity,"minimum_unit_price":min(prices,default=None),"cost_bound":cost})
            budget_bounds[p.name]={"lower_bound":lower_cash,"currency":request.currency,"scope":request.budget_scope,
                                   "proof":"optimistic_purchase_only_relaxation","derivation":derivation}
            if lower_cash>request.budget+request.inventory_policy.accounting_tolerance:
                violated=True
                diagnostics.append(ProcurementDiagnostic(reason_code="BUDGET_LOWER_BOUND_EXCEEDS_LIMIT",proof_status="SOUND_BOUND",
                    action_required=["BUSINESS_REVIEW_BUDGET_OR_CONFIRMED_COST_INPUTS"],
                    details={"strategy":p.name,"lower_bound":lower_cash,"budget":request.budget,
                             "necessary_shortfall_lower_bound":lower_cash-request.budget,"derivation":derivation,
                             "scope":request.budget_scope,"currency":request.currency,
                             "ignored_constraints":"expiry/pack/MOQ/fees/capacity; optimistic relaxation"}))
        if violated:infeasible.append(p.name)
    status="BLOCKED_INPUT_SEMANTICS" if any(d.proof_status=="BLOCKED" for d in diagnostics) else "PROVEN_INFEASIBLE" if len(infeasible)==len(profiles) else "NOT_RULED_OUT"
    return {"status":status,"diagnostics":diagnostics,"infeasible_profiles":infeasible,
            "stockout_probability_lower_bound":lower_probability,"budget_bounds":budget_bounds,"proof_scope":"declared_supplier_opportunities_full_evaluation_worlds_and_resolved_profiles",
            "bound_semantics":"necessary_conditions_only; pass_is_not_feasible"}
