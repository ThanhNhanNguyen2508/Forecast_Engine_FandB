"""Explicit cold-start forecasts and scenario assumptions; never borrowed accuracy."""
from __future__ import annotations
from datetime import date,timedelta
from pydantic import BaseModel,ConfigDict,Field,model_validator
import pandas as pd

class DeclaredForecast(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    store_id:str=Field(min_length=1)
    product_id:str=Field(min_length=1)
    product_name:str=Field(min_length=1)
    unit:str=Field(min_length=1)
    target_date:date
    p25:float=Field(ge=0)
    p50:float=Field(ge=0)
    p75:float=Field(ge=0)
    evidence_id:str=Field(min_length=1)
    classification:str=Field(pattern='^(FACT_FROM_SOURCE|ASSUMED_FOR_SCENARIO|CONFIRMED_BUSINESS_POLICY)$')
    @model_validator(mode='after')
    def ordering(self):
        if not self.p25<=self.p50<=self.p75:raise ValueError('OVERRIDE_QUANTILE_ORDERING')
        return self

class ForecastOverridePolicy(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    schema_version:int=1
    method:str=Field(pattern='^DECLARED_FORECAST_OVERRIDE$')
    predictions:list[DeclaredForecast]=Field(min_length=1)
    scenario_multipliers:list[float]=Field(min_length=1)
    scenario_weights:list[float]=Field(min_length=1)
    scenario_evidence_id:str=Field(min_length=1)
    dependence:str=Field(pattern='^COMMON_DECLARED_LEVEL_ACROSS_OVERRIDE_KEYS$')
    @model_validator(mode='after')
    def domain(self):
        import math
        if len(self.scenario_weights)!=len(self.scenario_multipliers) or any(v<0 for v in self.scenario_weights+self.scenario_multipliers):raise ValueError('INVALID_DECLARED_SCENARIO_LEVELS')
        if not math.isclose(sum(self.scenario_weights),1,abs_tol=1e-9):raise ValueError('DECLARED_SCENARIO_WEIGHTS_MUST_SUM_ONE')
        keys=[(p.store_id,p.product_id,p.target_date) for p in self.predictions]
        if len(set(keys))!=len(keys):raise ValueError('DUPLICATE_OVERRIDE_KEY')
        return self

def override_rows(policy,cutoff,horizon,calendar):
    keys={(p.store_id,p.product_id) for p in policy.predictions}
    expected={(s,p,cutoff.date()+timedelta(days=d)) for s,p in keys for d in range(1,horizon+1)}
    actual={(p.store_id,p.product_id,p.target_date) for p in policy.predictions}
    if actual!=expected:raise ValueError('OVERRIDE_REQUIRES_COMPLETE_DECLARED_HORIZON')
    if calendar is None or calendar.empty:raise ValueError('OVERRIDE_REQUIRES_EXPLICIT_STORE_CALENDAR')
    rows=[]
    for p in policy.predictions:
        match=calendar[pd.to_datetime(calendar['date']).dt.date==p.target_date]
        if 'store_id' in match:match=match[match['store_id'].isna()|match['store_id'].eq(p.store_id)]
        if len(match)!=1 or pd.isna(match.iloc[0].get('is_store_closed')):raise ValueError('OVERRIDE_CALENDAR_SCOPE_REQUIRED')
        closed=int(bool(match.iloc[0]['is_store_closed']))
        rows.append({'store_key':p.store_id,'product_key':p.product_id,'product_name':p.product_name,'unit':p.unit,
            'target_date':pd.Timestamp(p.target_date),'cutoff_date':cutoff,'horizon':(p.target_date-cutoff.date()).days,
            'p25':p.p25,'p50':p.p50,'p75':p.p75,'p25_raw':p.p25,'p50_raw':p.p50,'p75_raw':p.p75,
            'interval_lower':p.p25,'interval_upper':p.p75,'baseline_p50':p.p50,'product_code':-1,'store_code':-1,
            'seasonal_lag_7_target':float('nan'),'history_observation_count':0,'target_store_closed':closed,
            'calibration_source':'NOT_CALIBRATED_DECLARED_INTERVAL','forecast_method':policy.method,
            'forecast_evidence_id':p.evidence_id,'forecast_classification':p.classification})
    return pd.DataFrame(rows)
