"""Source-scoped normalization and independent scoring of exact inventory ledgers."""
from __future__ import annotations
from datetime import date
from collections import defaultdict
import hashlib
from pathlib import Path
from shelfcash_forecast.optimization.scenario_contracts import NormalizedBusinessRule


ALLOWED = {'maximum_stock': {'RECEIVING_PEAK','END_OF_DAY_MAX'},
    'storage_capacity': {'GLOBAL_RECEIVING_PEAK'}, 'safety_stock': {'END_OF_DAY_MIN'},
    'service_level_target': {'PER_KEY_EXPECTED_FILL'},
    'shelf_life_target': {'PURCHASE_COVER_DAYS','MIN_REMAINING_LIFE_AT_RECEIVING'}}


def normalize_preview_rules(frames, config, store_id, source_path, worlds, end):
    import pandas as pd
    from shelfcash_forecast.bom.units import UnitConverter
    profile = config.scenario_profile
    if profile.store_id != store_id:
        raise ValueError('PROFILE_STORE_SCOPE_MISMATCH')
    mapping = {(m.rule_type,m.ingredient_id):m for m in config.rule_mappings}
    units = {(l.store_id,l.ingredient_id):l.unit for s in worlds for l in s.lines}
    occupancy = {c.ingredient_id:c for c in config.occupancy_coefficients}
    source_hash = hashlib.sha256(Path(source_path).read_bytes()).hexdigest()
    daily = defaultdict(float)
    days = (end - worlds[0].simulation_start_date).days + 1
    for s in worlds:
        for l in s.lines:
            daily[l.ingredient_id] += s.probability_weight*l.quantity/days
    output=[];coverage=[];used=set()
    aid = next(a.assumption_id for a in profile.assumptions if a.field_path == 'rule_mappings')
    for index,row in enumerate(frames['business_rules'].to_dict('records'),2):
        ingredient = None if pd.isna(row['ingredient_id']) else str(row['ingredient_id'])
        m=mapping.get((row['rule_type'],ingredient))
        if m is None:
            raise ValueError('PREVIEW_UNMAPPED_SOURCE_RULE:' + str(index))
        used.add((m.rule_type,m.ingredient_id))
        if m.semantics not in ALLOWED.get(row['rule_type'],set()) or m.unit != row['unit']:
            raise ValueError('RULE_TYPE_SEMANTICS_OR_UNIT_MISMATCH')
        source={'file':str(source_path),'sha256':source_hash,'row':index,'rule_type':row['rule_type'],
                'value':float(row['value']),'unit':row['unit'],'note':str(row.get('note',''))}
        target=float(row['value']);unit=row['unit']
        if ingredient and unit not in {'day','ratio'}:
            if (store_id,ingredient) not in units:
                # Demand-zero inventory keys still require a scoped unit from occupancy.
                if ingredient not in occupancy: raise ValueError('RULE_TARGET_UNIT_REQUIRED')
                normalized=occupancy[ingredient].base_unit
            else: normalized=units[(store_id,ingredient)]
            target *= UnitConverter().conversion_factor(ingredient,unit,normalized);unit=normalized
        rule=NormalizedBusinessRule(rule_id=f'business_rules:{index}',source=source,store_id=store_id,
            ingredient_id=ingredient,unit=unit,target=target,semantics=m.semantics,classification=m.classification,
            effective_from=None if pd.isna(row.get('effective_from')) else date.fromisoformat(str(row['effective_from'])[:10]),
            effective_to=None if pd.isna(row.get('effective_to')) else date.fromisoformat(str(row['effective_to'])[:10]),
            penalty_currency_per_unit=m.penalty_currency_per_unit,assumption_refs=[aid],
            occupancy=config.occupancy_coefficients if m.semantics=='GLOBAL_RECEIVING_PEAK' else [],
            reference_daily_quantity=daily[ingredient] if m.semantics=='PURCHASE_COVER_DAYS' else None)
        output.append(rule)
        coverage.append({**source,'rule_id':rule.rule_id,'ingredient_id':ingredient,'mapping_status':'EXPLICIT_SCENARIO_INTERPRETATION',
            'semantics':m.semantics,'classification':m.classification,'business_validated':False,
            'effective_from':str(rule.effective_from),'effective_to':str(rule.effective_to),
            'applicable':rule.effective_from is None or rule.effective_from<=end,'assumption_refs':[aid]})
    if used != set(mapping): raise ValueError('UNUSED_RULE_MAPPING')
    return output,coverage


def evaluate_rules(rules, simulation, orders):
    """Exact results only; weights retained; hard misses are critic violations."""
    evaluations=[]
    for rule in rules:
        observations=[];target=rule.target
        if rule.semantics in {'PURCHASE_COVER_DAYS','MIN_REMAINING_LIFE_AT_RECEIVING'}:
            grouped=defaultdict(float)
            for o in orders:
                if o.store_id==rule.store_id and o.ingredient_id==rule.ingredient_id and rule.active(o.arrival_date):
                    if rule.semantics=='PURCHASE_COVER_DAYS': grouped[str(o.arrival_date)] += o.order_quantity
                    else: grouped[o.offer_id] = o.shelf_life_days if o.shelf_life_days is not None else -1
            for identity,q in grouped.items():
                if rule.semantics=='PURCHASE_COVER_DAYS' and rule.reference_daily_quantity==0:
                    # Cross-multiplication gives quantity <= target_days * 0.
                    # Coverage days is undefined; it must never be an infinite fact.
                    observations.append({'identity':identity,'observed':None,'violation':q,'weight':1.0,
                        'violation_measure':'base quantity above zero purchase bound',
                        'required_maximum_base_quantity':0,'purchased_base_quantity':q})
                    continue
                value=q/rule.reference_daily_quantity if rule.semantics=='PURCHASE_COVER_DAYS' else q
                gap=max(0,value-target) if rule.semantics=='PURCHASE_COVER_DAYS' else max(0,target-value)
                observations.append({'identity':identity,'observed':value,'violation':gap,'weight':1.0})
        elif rule.semantics=='PER_KEY_EXPECTED_FILL':
            keys={(l.store_id,l.ingredient_id) for w in simulation.results for l in w.daily_ledgers
                  if l.store_id==rule.store_id and (rule.ingredient_id is None or l.ingredient_id==rule.ingredient_id)}
            for key in sorted(keys):
                fill=0
                for world in simulation.results:
                    rows=[l for l in world.daily_ledgers if (l.store_id,l.ingredient_id)==key and rule.active(l.simulation_date)]
                    demand=sum(l.demand_quantity for l in rows);shortage=sum(l.shortage_quantity for l in rows)
                    fill += world.probability_weight*(1-shortage/demand if demand else 1)
                observations.append({'identity':'|'.join(key),'observed':fill,'violation':max(0,target-fill),'weight':1.0})
        else:
            coeff={c.ingredient_id:c for c in rule.occupancy}
            for world in simulation.results:
                dayrows=defaultdict(list)
                for l in world.daily_ledgers:
                    if l.store_id==rule.store_id and rule.active(l.simulation_date) and (rule.ingredient_id is None or l.ingredient_id==rule.ingredient_id):
                        dayrows[l.simulation_date].append(l)
                for day,rows in dayrows.items():
                    if rule.semantics=='GLOBAL_RECEIVING_PEAK':
                        if any(l.ingredient_id not in coeff or coeff[l.ingredient_id].base_unit!=l.unit for l in rows):
                            raise ValueError('GLOBAL_OCCUPANCY_COVERAGE_OR_UNIT_MISMATCH')
                        value=sum(l.maximum_quantity*coeff[l.ingredient_id].liters_per_base_unit for l in rows)
                    else:
                        value=sum(l.maximum_quantity if rule.semantics=='RECEIVING_PEAK' else l.ending_quantity for l in rows)
                    gap=max(0,target-value) if rule.semantics=='END_OF_DAY_MIN' else max(0,value-target)
                    observations.append({'scenario_id':world.scenario_id,'date':str(day),'observed':value,'violation':gap,'weight':world.probability_weight})
        maximum=max((o['violation'] for o in observations),default=0)
        weighted=sum(o['weight']*o['violation'] for o in observations)
        evaluations.append({'rule_id':rule.rule_id,'source':rule.source,'semantics':rule.semantics,
            'classification':rule.classification,'target':target,'unit':rule.unit,
            'status':'PASS' if maximum<=1e-8 else 'FAIL' if rule.classification=='HARD' else 'TARGET_MISS',
            'maximum_violation':maximum,'weighted_violation':weighted,
            'violation_unit':next((l.unit for w in simulation.results for l in w.daily_ledgers
                if (l.store_id,l.ingredient_id)==(rule.store_id,rule.ingredient_id)), 'base quantity')
                if rule.semantics=='PURCHASE_COVER_DAYS' and rule.reference_daily_quantity==0 else rule.unit,
            'soft_penalty_vnd':weighted*rule.penalty_currency_per_unit,'evaluation_count':len(observations),
            'maximum_observed':max((o['observed'] for o in observations if o['observed'] is not None),default=None),
            'worst_observations':sorted(observations,key=lambda o:-o['violation'])[:10],
            'assumption_refs':rule.assumption_refs})
    return evaluations
