"""Declared engineering envelope and structured diagnostics at the public boundary."""
from __future__ import annotations
from shelfcash_forecast.optimization.contracts import OptimizationResult, ProcurementDiagnostic

SUPPORTED_ENVELOPE = {'horizon_days':366, 'worlds':2000, 'store_ingredient_keys':200,
    'offers':10000, 'lots':10000, 'canonical_quantity_upper':1e9,
    'positive_canonical_quantity_lower':1e-6, 'currency_cost_upper':1e12, 'text_characters':16384}


def error_result(request, outcome, exc):
    request_id = request.get('request_id','INVALID_REQUEST') if isinstance(request,dict) else getattr(request,'request_id','INVALID_REQUEST')
    errors = exc.errors(include_url=False) if hasattr(exc,'errors') else [{'loc':(), 'msg':str(exc),'type':getattr(exc,'code',type(exc).__name__)}]
    diagnostics=[]
    for error in errors:
        value=error.get('input')
        # JSON itself cannot represent NaN/Infinity. Preserve the submitted
        # meaning in diagnostic text instead of emitting invalid JSON numbers.
        if not isinstance(value,(str,int,bool,type(None))): value=repr(value)[:2000]
        diagnostics.append(ProcurementDiagnostic(reason_code=str(error['type']),proof_status='BLOCKED',
            field_paths=['.'.join(map(str,error.get('loc',()))) or '$'],value=value,
            expected_meaning=error['msg'], action_required=['CORRECT_INPUT_OR_PROVIDE_REQUIRED_TERMS']))
    return OptimizationResult(request_id=str(request_id),evaluations={},status='NO_VALID_PROCUREMENT_PLAN',
        technical_outcome=outcome,diagnostics=diagnostics)


def validate_supported_request(request):
    from shelfcash_forecast.bom.units import UnitConverter
    issues=[]
    def block(code,path,value,expected):
        issues.append(ProcurementDiagnostic(reason_code=code,proof_status='BLOCKED',field_paths=[path],
            value=value,expected_meaning=expected,action_required=['USE_SUPPORTED_CONTRACT_OR_INCREASE_DECLARED_RESOURCE_LIMITS']))
    def check_text(value, path='$'):
        if isinstance(value,str) and len(value)>SUPPORTED_ENVELOPE['text_characters']:
            block('SUPPORTED_TEXT_LIMIT',path,len(value),'text <=16384 characters; bounded workbook cell and diagnostic memory')
        elif isinstance(value,dict):
            for key,item in value.items():
                check_text(str(key),path+'.<key>');check_text(item,path+'.'+str(key))
        elif isinstance(value,list):
            for index,item in enumerate(value):check_text(item,f'{path}.{index}')
    check_text(request.model_dump(mode='json'))
    sizes={'horizon_days':(request.planning_end_date-request.decision_date).days,
        'worlds':len(request.evaluation_scenarios or request.demand_scenarios),
        'store_ingredient_keys':len({(l.store_id,l.ingredient_id) for s in (request.evaluation_scenarios or request.demand_scenarios) for l in s.lines}),
        'offers':len(request.supplier_offers),'lots':len(request.initial_inventory)+len(request.existing_inbound)}
    for field,value in sizes.items():
        if value > SUPPORTED_ENVELOPE[field] or field=='horizon_days' and value<1:
            block('SUPPORTED_SIZE_LIMIT',field,value,f'1..{SUPPORTED_ENVELOPE[field]} for horizon; bounded sparse daily MILP memory')
    if request.inventory_policy.accounting_tolerance > 1e-6:
        block('UNSUPPORTED_ACCOUNTING_TOLERANCE','inventory_policy.accounting_tolerance',request.inventory_policy.accounting_tolerance,'0 < tolerance <= 1e-6 canonical units; no acceptance relaxation')
    for i,rule in enumerate(request.normalized_rules):
        if rule.unit not in {'ratio','day','liter','kg','unit'}:
            block('NORMALIZED_RULE_UNIT_REQUIRED',f'normalized_rules.{i}.unit',rule.unit,'base unit for normalized rule; raw mappings convert before solve')
    from shelfcash_forecast.optimization.chronology import offer_arrival
    for i,offer in enumerate(request.supplier_offers):
        if max(offer.unit_price,offer.delivery_cost)>SUPPORTED_ENVELOPE['currency_cost_upper']:
            block('SUPPORTED_COST_LIMIT',f'supplier_offers.{i}',offer.offer_id,'individual currency cost/rate <=1e12')
        try:
            offer_arrival(offer)
        except ValueError as exc:
            block('SUPPLY_CHRONOLOGY_REQUIRED',f'supplier_offers.{i}.calendar',str(exc),
                'declared admissible order date, lead, schedule and resolved arrival; ambiguous calendars require explicit terms')
    try:
        converter=UnitConverter(request.unit_conversions)
        def check_quantity(ingredient,unit,value,path):
            base=converter.canonical_unit(ingredient,unit)
            q=converter.convert(value,unit,base,ingredient_id=ingredient)
            if q>SUPPORTED_ENVELOPE['canonical_quantity_upper'] or 0<q<SUPPORTED_ENVELOPE['positive_canonical_quantity_lower']:
                block('SUPPORTED_NUMERIC_LIMIT',path,q,'zero or 1e-6..1e9 canonical units; bounded Big-M and quantities above exact/solver absolute tolerance')
        for i,lot in enumerate(request.initial_inventory):
            check_quantity(lot.ingredient_id,lot.unit,lot.quantity_remaining,f'initial_inventory.{i}.quantity_remaining')
        for i,delivery in enumerate(request.existing_inbound):
            check_quantity(delivery.ingredient_id,delivery.unit,delivery.quantity,f'existing_inbound.{i}.quantity')
        for i,offer in enumerate(request.supplier_offers):
            for field in ('pack_size','minimum_order_quantity','maximum_order_quantity'):
                value=getattr(offer,field)
                if value is not None:check_quantity(offer.ingredient_id,offer.unit,value,f'supplier_offers.{i}.{field}')
        for i,s in enumerate(request.evaluation_scenarios or request.demand_scenarios):
            for j,l in enumerate(s.lines):
                unit=converter.canonical_unit(l.ingredient_id,l.unit)
                if unit not in {'kg','liter','unit'}:
                    block('UNRECOGNIZED_BASE_UNIT',f'demand_scenarios.{i}.lines.{j}.unit',l.unit,'kg/g, liter/ml, piece/unit, or sourced ingredient conversion to a base unit')
                check_quantity(l.ingredient_id,l.unit,l.quantity,f'demand_scenarios.{i}.lines.{j}.quantity')
    except Exception as exc:
        from shelfcash_forecast.exceptions import BOMError
        if not isinstance(exc,BOMError): raise
        block('UNIT_CONVERSION_REQUIRED','unit_conversions',str(exc),'explicit consistent ingredient-scoped physical conversion')
    return issues
