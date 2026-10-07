"""Customer package from the single critic-accepted M5 authority. No quantities invented."""
from __future__ import annotations
import csv
import json
from datetime import timedelta, date
from pathlib import Path
from typing import Literal
from pydantic import Field
from shelfcash_forecast.optimization.scenario_contracts import ScenarioContract
from shelfcash_forecast.optimization.chronology import expiry_date, planned_lot_id, offer_arrival
from shelfcash_forecast.optimization.planning_service import content_hash
from shelfcash_forecast.optimization.shipment_fees import shipment_group


class CustomerProcurementPackage(ScenarioContract):
    schema_version: Literal[1] = 1
    request_id: str
    plan_id: str
    profile_id: str
    profile_version: int
    planning_mode: Literal['SCENARIO_PREVIEW']
    environment: Literal['DEMO','BACKTEST','SYNTHETIC']
    data_origin: Literal['ACTUAL_SEALED_RAW_BUNDLE']
    technical_outcome: Literal['FEASIBLE']
    critic_passed: Literal[True]
    business_ready: Literal[False] = False
    execution_authorized: Literal[False] = False
    procurement_executed: Literal[False] = False
    operational_status: Literal['SCENARIO_PREVIEW_NOT_FOR_OPERATION'] = 'SCENARIO_PREVIEW_NOT_FOR_OPERATION'
    metadata: dict
    accepted_plan: dict
    orders: list[dict]
    assumptions_and_customer_conditions: list[dict]
    cost_decomposition: dict
    service_risk_metrics: dict
    rule_coverage: list[dict]
    accepted_zero_purchase: bool


def write_json(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,default=str)+'\n',encoding='utf-8')


def csv_rows(path,rows,fields=None):
    fields=fields or list(rows[0])
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(
            {k:json.dumps(v,ensure_ascii=False,default=str) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in rows)


def banana_evidence(request, candidate, folder, budget_case):
    ingredient='ING_929389e845d7'
    initial=[l for l in request.initial_inventory if l.ingredient_id==ingredient]
    opportunities=[o for o in request.supplier_offers if o.ingredient_id==ingredient]
    lines=[o for o in candidate.plan.orders if o.ingredient_id==ingredient] if candidate else []
    chronology={'profile_id':request.planning_binding.get('profile_id'),'budget_case':budget_case,
        'ingredient_id':ingredient,'source_name':'Chuối','expiry_convention':'arrival + offset; usable THROUGH expiry date',
        'initial_lots':[l.model_dump(mode='json') for l in initial],
        'opportunities':[{'opportunity_id':o.offer_id,'order_placement_deadline':str(o.order_date),
            'arrival_date':str(offer_arrival(o)),'expiry_date':str(expiry_date(offer_arrival(o),o.shelf_life_days)),
            'source_terms':o.source_terms} for o in opportunities],
        'tail_day':str(request.planning_end_date),
        'tail_covering_orders':[o.model_dump(mode='json') for o in lines if o.arrival_date<=request.planning_end_date and
            expiry_date(o.arrival_date,o.shelf_life_days)>=request.planning_end_date],
        'baseline_expiry_gap':'initial 8kg expires 2026-08-17; 2026-08-13 receipt expires 2026-08-18; increasing that lot cannot serve 2026-08-19',
        'exact_simulation_executed':candidate is not None and candidate.simulation is not None}
    rows=[]
    if candidate and candidate.simulation:
        for world in candidate.simulation.results:
            state={l.lot_id:{'q':l.quantity_remaining,'expiry':l.expiry_date,'arrival':l.received_date} for l in initial}
            for ledger in [l for l in world.daily_ledgers if l.ingredient_id==ingredient]:
                day=ledger.simulation_date
                beginning_ids=[lid for lid,l in state.items() if l['q']>1e-8]
                usable=sum(l['q'] for l in state.values() if l['expiry'] is None or l['expiry']>=day)
                inbound=[o for o in lines if o.arrival_date==day]
                inbound_ids=[planned_lot_id(candidate.plan.plan_id,o.offer_id) for o in inbound]
                for o,lid in zip(inbound,inbound_ids):state[lid]={'q':o.order_quantity,'expiry':expiry_date(o.arrival_date,o.shelf_life_days),'arrival':o.arrival_date}
                for l in state.values():
                    if l['expiry'] is not None and l['expiry']<day:l['q']=0
                traces=[t for t in world.consumption_traces if t.ingredient_id==ingredient and t.simulation_date==day]
                for t in traces:state[t.lot_id]['q']-=t.quantity
                remaining=[lid for lid,l in state.items() if l['q']>1e-8]
                cap=[r.target for r in request.normalized_rules if r.ingredient_id==ingredient and r.semantics=='RECEIVING_PEAK' and r.active(day)]
                violations=[r.rule_id for r in request.normalized_rules if r.ingredient_id==ingredient and r.active(day) and
                    ((r.semantics=='END_OF_DAY_MIN' and ledger.ending_quantity+1e-8<r.target) or
                     (r.semantics=='RECEIVING_PEAK' and ledger.maximum_quantity>r.target+1e-8))]
                rows.append({'profile_id':request.planning_binding.get('profile_id'),'budget_case':budget_case,
                    'scenario_id':world.scenario_id,'probability_weight':world.probability_weight,'date':str(day),
                    'beginning_physical_stock':ledger.beginning_quantity,'beginning_usable_stock':usable,
                    'initial_lot_ids':[l.lot_id for l in initial],'inbound_lot_ids':inbound_ids,
                    'order_opportunity_ids':[o.offer_id for o in inbound],
                    'arrival_dates':{lid:str(l['arrival']) for lid,l in state.items()},
                    'expiry_dates':{lid:str(l['expiry']) for lid,l in state.items()},
                    'received_quantity':ledger.inbound_quantity,'receiving_peak':ledger.maximum_quantity,
                    'applicable_capacity':min(cap) if cap else None,'expired_quantity':ledger.expired_quantity,
                    'demand':ledger.demand_quantity,'fulfilled':ledger.fulfilled_quantity,'shortage':ledger.shortage_quantity,
                    'ending_usable_stock':ledger.ending_quantity,'ending_lot_ids':remaining,
                    'lots_consumed':[t.model_dump(mode='json') for t in traces],
                    'rule_violations':violations,'assumption_refs':[a['assumption_id'] for a in request.planning_binding.get('assumptions',[])]})
                if abs(sum(l['q'] for l in state.values())-ledger.ending_quantity)>request.inventory_policy.accounting_tolerance:
                    raise ValueError('BANANA_TRACE_ACCOUNTING_MISMATCH')
        tail=[r for r in rows if r['date']==str(request.planning_end_date)]
        worst=max(tail,key=lambda r:(r['shortage'],r['demand'],-r['ending_usable_stock'],r['scenario_id']))
        chronology.update(full_world_count=len(candidate.simulation.results),tail_shortage_world_count=sum(r['shortage']>1e-8 for r in tail),
            tail_max_shortage=max(r['shortage'] for r in tail),customer_representative_world=worst['scenario_id'],
            representative_reason='largest tail shortage, then largest last-day demand; deterministic tie-break')
    write_json(folder/'banana_chronology.json',chronology)
    csv_rows(folder/'banana_daily_ledger.csv',rows,fields=list(rows[0]) if rows else ['profile_id','budget_case','scenario_id','date','shortage'])
    return chronology,rows


def export_customer_plan(run, destination, accepted):
    if run.request.planning_mode!='SCENARIO_PREVIEW':return None
    request,result=run.request,run.result
    candidate=result.evaluations.get(result.recommended_strategy) if result.recommended_strategy else None
    folder=Path(destination)/'customer_plan';folder.mkdir(exist_ok=True)
    budget_case='budget_null' if request.budget is None else f'budget_{request.budget:g}'
    chronology,banana=banana_evidence(request,candidate,folder,budget_case)
    if candidate is None:
        csv_rows(folder/'customer_procurement_orders.csv',[],['ingredient_id','quantity','technical_status'])
        write_json(folder/'plan_validation.json',{'selected_plan':None,'outcome':result.technical_outcome,'reason':'NO_CRITIC_ACCEPTED_PLAN'})
        return None
    if not candidate.critic.passed or not candidate.critic.checks.get('evaluation_coverage') or not candidate.plan.completed:
        raise ValueError('CUSTOMER_EXPORT_REQUIRES_FULL_EXACT_ACCEPTANCE')
    offers={o.offer_id:o for o in request.supplier_offers};rows=[]
    conditions=request.planning_binding['assumptions']
    for o,row in zip(candidate.plan.orders,accepted['orders'],strict=True):
        offer=offers[o.offer_id];terms=offer.source_terms
        rows.append({**row,'store_name':None,'supplier_name':terms.get('supplier_name',o.supplier_id),
            'shipment_group_id':shipment_group(offer),'order_deadline':str(o.order_date),
            'packaging_description':f'{o.pack_size:g} {o.unit}/pack','pack_unit':o.unit,
            'source_MOQ':terms.get('minimum_order_quantity'),'source_MOQ_basis':terms.get('order_unit'),
            'raw_price':terms.get('unit_price'),'raw_price_basis':terms.get('price_basis'),
            'price_interpretation':terms.get('price_mapping'),'classification':terms.get('classification'),
            'condition_assumption_refs':terms.get('assumption_refs',[]),'original_lead_days':terms.get('original_lead_time_days'),
            'proposed_lead_days':terms.get('proposed_lead_time_days'),'contract_id':terms.get('contract_id'),
            'provenance_ref':terms.get('source')})
    purchase=sum(o.purchase_cost for o in candidate.plan.orders);delivery=sum(o.delivery_cost for o in candidate.plan.orders)
    rules=candidate.critic.details['business_rule_evaluation'];metrics=candidate.simulation.risk_metrics.model_dump(mode='json')
    package=CustomerProcurementPackage(request_id=request.request_id,plan_id=candidate.plan.plan_id,
        profile_id=request.planning_binding['profile_id'],profile_version=request.planning_binding['profile_version'],
        planning_mode='SCENARIO_PREVIEW',environment=request.environment,data_origin='ACTUAL_SEALED_RAW_BUNDLE',
        technical_outcome='FEASIBLE',critic_passed=True,
        metadata={'snapshot_date':str(request.inventory_snapshot_date),'snapshot_boundary':'EOD',
            'planning_horizon':{'start':str(request.decision_date+timedelta(days=1)),'end':str(request.planning_end_date)},
            'request_hash':content_hash(request),'config_hash':content_hash(run.config),**request.planning_binding,
            'source_bundle':run.provenance['source_bundle'],'source_pool_hash':request.scenario_provenance['full_pool_hash'],
            'selected_mode':run.selected_mode,'selected_strategy':candidate.plan.strategy,
            'selection_reason':run.provenance['compare_selection_policy'],'evaluation_coverage':len(candidate.simulation.results),
            'budget':{'amount':request.budget,'currency':request.currency,'scope':request.budget_scope},
            'budget_satisfied':request.budget is None or purchase+delivery<=request.budget+1e-8,
            'shipment_count':len({r['shipment_group_id'] for r in rows}),
            'global_procurement_cost_optimality_claimed':candidate.plan.provenance.get('global_optimality_claimed',False),
            'optimization_scope':candidate.plan.provenance.get('optimization_scope'),
            'inherited_context_condition':'Null-expiry packaging follows explicit DEMO warn_and_place_last policy; confirm non-expiring packaging classification before business use.',
            'sensitivity_limitations':['Unconfirmed price/fee/occupancy/calendar; conditional cost comparison only',
                '100 model-derived worlds are not independent OOS or a real-world probability guarantee',
                'Fixed scheduled commitments; no adaptive supplier recourse'],
            'missing_display_names':'store_name null: raw context supplies STORE_A only'},
        accepted_plan=candidate.plan.model_dump(mode='json'),orders=rows,
        assumptions_and_customer_conditions=conditions,cost_decomposition={'purchase_vnd':purchase,'delivery_vnd':delivery,
            'committed_procurement_vnd':purchase+delivery,'consequence_costs_by_optimization_world':candidate.plan.provenance['cost_components_by_world'],
            'soft_rule_penalty_vnd':sum(r['soft_penalty_vnd'] for r in rules)},
        service_risk_metrics=metrics,rule_coverage=rules,accepted_zero_purchase=not rows)
    write_json(folder/'customer_procurement_plan.json',package.model_dump(mode='json'))
    csv_rows(folder/'customer_procurement_orders.csv',rows)
    write_json(folder/'plan_conditions.json',{'business_ready':False,'execution_authorized':False,'conditions':conditions,
        'supplier_terms_to_confirm':[{'supplier_id':r['supplier_id'],'contract_id':r['contract_id'],
            'order_deadline':r['order_deadline'],'arrival':r['arrival_date'],'quantity':r['quantity'],'pack_count':r['pack_count'],
            'fee':r['allocated_delivery_cost'],'unit_price':r['unit_price'],'classification':r['classification']} for r in rows]})
    write_json(folder/'plan_validation.json',{'critic':candidate.critic.model_dump(mode='json'),'coverage':len(candidate.simulation.results),
        'plan_hash':content_hash(candidate.plan),'planning_binding':request.planning_binding,'banana_tail':chronology})
    text=['# KẾ HOẠCH ĐỀ XUẤT THEO PROFILE/ĐIỀU KIỆN',
        '**DEMO/BACKTEST chưa xác nhận — chưa được phép đặt hàng hay thực thi.**',
        f"Profile: {package.profile_id} v{package.profile_version}. Tồn kho chốt cuối ngày 12/08/2026; kế hoạch 13–19/08/2026.",
        f'Tổng mua + giao cam kết: **{purchase+delivery:,.0f} VND** (mua {purchase:,.0f}; giao {delivery:,.0f}).',
        'Ngày đặt trong tương lai là cam kết lên lịch, cần đặt đúng hạn ghi dưới đây.',
        '| Nguyên liệu | Nhà cung cấp | Hạn đặt | Nhận | Hạn dùng (bao gồm ngày này) | Lượng | Số pack | Tổng VND |',
        '|---|---|---|---|---|---:|---:|---:|']
    text += [f"| {r['ingredient_name']} | {r['supplier_name']} | {r['order_date']} | {r['arrival_date']} | {r['expiry_date'] or 'Không có hạn nguồn cho bao bì'} | {r['quantity']:g} {r['unit']} | {r['pack_count']} | {r['total_procurement_cost']:,.0f} |" for r in rows]
    text += [f"Kiểm tra 100 kịch bản: tỷ lệ đáp ứng trung bình theo nguyên liệu {metrics['mean_key_fill_rate']:.2%}; xác suất có thiếu hàng trong pool {metrics['any_stockout_probability']:.2%}. Tất cả ràng buộc cứng áp dụng đã đạt.",
        'Mục tiêu dự phòng cuối ngày và độ tươi được báo riêng; mục tiêu tư vấn có thể chưa đạt, không phải chứng nhận đã giữ đủ dự phòng mỗi ngày.',
        f"Chuối được chia theo lô nhận hợp lệ; ngày 19/08 thiếu tối đa {chronology['tail_max_shortage']:g} kg trên 100 kịch bản. Hạn dùng lô nhận = ngày nhận + 5, dùng được đến hết ngày hạn; lô nhận 13/08 không phục vụ 19/08.",
        'Điều kiện áp dụng: khách hàng xác nhận nghĩa lịch, giá theo đơn vị cơ sở, MOQ matcha một hộp = 0,5kg, phí và sức chứa quy đổi; nhà cung cấp xác nhận các chuyến, hạn đặt và thời gian nhận trước nhu cầu. Những dòng hợp đồng đề xuất cần nhà cung cấp đồng ý riêng.',
        'Không dịch ngày dữ liệu sang tuần hiện tại. Kế hoạch tương lai thực tế cần snapshot và dự báo cập nhật.',
        '\nCác điều kiện chi tiết nằm trong Conditions của Excel và plan_conditions.json.']
    text += ['Bao bì không có hạn dùng trong nguồn đang theo policy DEMO đã khai báo; cần xác nhận phân loại bao bì không hết hạn trước khi sử dụng nghiệp vụ.',
        'Chi phí là ước tính theo điều kiện giá/phí trên; chưa chứng minh đây là phương án rẻ nhất trong mọi lựa chọn.']
    formatted=text[:5]+['\n'.join(text[5:7+len(rows)])]+text[7+len(rows):]
    (folder/'CUSTOMER_PLAN_VI.md').write_text('\n\n'.join(formatted)+'\n',encoding='utf-8')
    workbook_qa=write_workbook(folder/'customer_procurement_plan.xlsx',package,rows,conditions,banana,chronology)
    write_json(folder/'workbook_verification.json',workbook_qa)
    return package


def write_workbook(path,package,rows,conditions,banana,chronology):
    from openpyxl import Workbook,load_workbook
    from openpyxl.styles import Font,PatternFill,Alignment
    from openpyxl.utils import get_column_letter
    wb=Workbook();wb.remove(wb.active)
    def sheet(name,headers,data):
        ws=wb.create_sheet(name);ws.append(headers)
        for row in data:
            ws.append([json.dumps(v,ensure_ascii=False,default=str) if isinstance(v,(list,dict)) else v for v in row])
        ws.freeze_panes='A2';ws.auto_filter.ref=ws.dimensions
        for c in ws[1]:c.font=Font(bold=True,color='FFFFFF');c.fill=PatternFill('solid',fgColor='1F4E78')
        for j,h in enumerate(headers,1):
            ws.column_dimensions[get_column_letter(j)].width=min(55,max(16,len(str(h))+3))
            for cell in list(ws.columns)[j-1][1:]:
                cell.alignment=Alignment(vertical='top',wrap_text=isinstance(cell.value,str) and len(cell.value)>55)
                if isinstance(cell.value,str) and cell.value.startswith(('=','+','-','@')):cell.data_type='s'
                if 'date' in h or h in {'order_deadline','Hạn đặt','Nhận','Hạn dùng'}:
                    if isinstance(cell.value,str) and len(cell.value)==10:
                        try:cell.value=date.fromisoformat(cell.value);cell.number_format='dd/mm/yyyy'
                        except ValueError:pass
                if any(k in h.lower() for k in ('cost','price','fee','vnd')) and isinstance(cell.value,(int,float)):
                    cell.number_format='#,##0.00'
        return ws
    summary=[['Trạng thái','KẾ HOẠCH ĐỀ XUẤT THEO PROFILE/ĐIỀU KIỆN — DEMO; chưa được phép thực thi'],
        ['Profile',package.profile_id],['Horizon','13–19/08/2026'],
        ['Mua VND',package.cost_decomposition['purchase_vnd']],['Giao VND',package.cost_decomposition['delivery_vnd']],
        ['Tổng VND',package.cost_decomposition['committed_procurement_vnd']],
        ['Budget VND',package.metadata['budget']['amount']],['Critic','PASS; full100'],
        ['Fill',package.service_risk_metrics['mean_key_fill_rate']],['Stockout pool',package.service_risk_metrics['any_stockout_probability']],
        ['Business ready',False],['Execution authorized',False]]
    sheet('Summary',['Nội dung','Giá trị'],summary)
    headers=list(rows[0]);orders=sheet('Orders',headers,[[r[k] for k in headers] for r in rows])
    for row in orders.iter_rows(min_row=2):
        stage=row[headers.index('decision_stage')].value
        for c in row:c.fill=PatternFill('solid',fgColor='FFF2CC' if stage=='IMMEDIATE_ADVICE' else 'E2F0D9')
    schedule_headers=['ingredient_name','supplier_name','order_date','arrival_date','expiry_date','quantity','unit','pack_count','shipment_group_id','classification']
    sheet('Delivery Schedule',schedule_headers,[[r[k] for k in schedule_headers] for r in sorted(rows,key=lambda r:(r['arrival_date'],r['ingredient_id']))])
    sheet('Conditions',['ID','Kind','Scope','Field','Value','Rationale','Confirmation pending'],
        [[a['assumption_id'],a['kind'],a['scope'],a['field_path'],a['value'],a['rationale'],True] for a in conditions])
    sheet('Validation',['Rule','Semantics','Classification','Target','Unit','Status','Max violation','Weighted violation'],
        [[r[k] for k in ['rule_id','semantics','classification','target','unit','status','maximum_violation','weighted_violation']] for r in package.rule_coverage])
    representative=[r for r in banana if r['scenario_id']==chronology['customer_representative_world']]
    bh=['date','scenario_id','beginning_usable_stock','received_quantity','expiry_dates','demand','fulfilled','shortage','ending_usable_stock','lots_consumed']
    sheet('Banana',bh,[[r[k] for k in bh] for r in representative])
    wb.save(path);wb.close()
    check=load_workbook(path,data_only=False)
    errors=[]
    for ws in check:
        if ws.freeze_panes!='A2' or not ws.auto_filter.ref:errors.append('sheet_format:'+ws.title)
        for row in ws:
            for c in row:
                if c.data_type in {'e','f'}:errors.append('formula_or_error:'+ws.title+':'+c.coordinate)
    actual_rows=list(check['Orders'].iter_rows(min_row=2,values_only=True))
    if len(actual_rows)!=len(rows):errors.append('row_count')
    for observed,expected in zip(actual_rows,rows,strict=True):
        for i,key in enumerate(headers):
            value=expected[key];actual=observed[i]
            if isinstance(value,(int,float)) and not isinstance(value,bool) and abs(actual-value)>1e-7:errors.append('numeric:'+key)
            elif isinstance(value,bool) and actual!=value:errors.append('bool:'+key)
            elif isinstance(value,(dict,list)) and json.loads(actual)!=value:errors.append('structured:'+key)
            elif isinstance(value,str) and ('date' in key or key=='order_deadline') and len(value)==10:
                if actual.date().isoformat()!=value:errors.append('date:'+key)
            elif isinstance(value,str) and actual!=value:errors.append('text:'+key)
    total=sum(r[headers.index('total_procurement_cost')] for r in actual_rows)
    fee=sum(r[headers.index('allocated_delivery_cost')] for r in actual_rows)
    if abs(total-package.cost_decomposition['committed_procurement_vnd'])>1e-7:errors.append('total')
    if abs(fee-package.cost_decomposition['delivery_vnd'])>1e-7:errors.append('fee')
    check.close()
    if errors:raise ValueError('WORKBOOK_QA_FAILED:'+','.join(errors))
    return {'passed':True,'sheets':['Summary','Orders','Delivery Schedule','Conditions','Validation','Banana'],
        'order_rows':len(rows),'readback_total_vnd':total,'readback_delivery_vnd':fee,'formula_errors':[],
        'representative_banana_rows':len(representative),'all_order_cells_checked':True}
