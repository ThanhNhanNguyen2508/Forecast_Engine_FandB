"""Versioned, explicitly conditional planning semantics; never supplier confirmation."""
from __future__ import annotations
import hashlib
import json
from datetime import date
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, JsonValue, field_validator, model_validator


def semantic_hash(value):
    if hasattr(value, 'model_dump'):
        value = value.model_dump(mode='json')
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


class ScenarioContract(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)

    @field_validator('*', mode='before')
    @classmethod
    def numeric_bool(cls, value, info):
        if isinstance(value, bool) and any(t in str(cls.model_fields[info.field_name].annotation) for t in ('float', 'int')):
            raise ValueError('BOOLEAN_NOT_NUMERIC:' + info.field_name)
        return value


REGISTERED_ASSUMPTION_FIELDS = {'supplier_calendar', 'supplier_calendars', 'packaging_mappings',
    'price_basis_mappings', 'rule_mappings', 'delivery_cost_assumption', 'delivery_fee_scope',
    'cost_policy', 'occupancy_coefficients', 'contract_options', 'currency'}

KNOWN_OPTIONS={'U01':{'RECEIVING','ORDER','DISPATCH','NEGOTIATED'},
    'U02':{'CALENDAR_ORDER','BUSINESS_NEXT','AFTER_DEMAND','EXPEDITED'},
    'U03':{'NO_EXCEPTIONS','SCOPED_WORKING'},'U04':{'PEAK_GLOBAL','EOD_MAX','COMPARTMENTS'},
    'U05':{'ADVISORY_RESERVE','SOFT_RESERVE','HARD_RESERVE','FRESH_RECEIVING','CHANCE_UNIVERSAL'},
    'U06':{'MATCHA_PACK','MATCHA_KG'},'U07':{'BASE_PRICE_LINE_FEE','BASE_PRICE_GROUP_FEE','PACK_PRICE','EXPLICIT_FREE'}}


class SemanticsOption(ScenarioContract):
    option_id: str
    semantic_group: Literal['U01','U02','U03','U04','U05','U06','U07']
    meaning: str
    pros: str
    cons: str
    consequences: str
    required_inputs: str
    support_status: str
    compatible_options: str
    incompatible_options: str
    tests: str
    customer_confirmations: str


class ProfileRunResult(ScenarioContract):
    schema_version: Literal[1] = 1
    profile_id: str
    profile_version: StrictInt
    input_hash: str
    config_hash: str
    assumption_hash: str
    budget_case: str
    technical_outcome: str
    selected_plan_id: str | None
    critic_passed: StrictBool
    evaluation_coverage: StrictInt = Field(ge=0)
    evidence_directory: str
    business_ready: Literal[False] = False
    execution_authorized: Literal[False] = False


class Assumption(ScenarioContract):
    assumption_id: str = Field(min_length=1)
    profile_id: str = Field(min_length=1)
    field_path: str
    scope: str = Field(min_length=1)
    value: JsonValue
    unit: str = Field(min_length=1)
    kind: Literal['BUSINESS_POLICY_CHOICE', 'AMBIGUOUS_SOURCE_INTERPRETATION',
                  'HYPOTHETICAL_SUPPLIER_CONTRACT', 'ENGINEERING_ESTIMATE']
    rationale: str = Field(min_length=1)
    source_status: Literal['UNCONFIRMED', 'PROPOSED', 'ESTIMATED']
    source_locator: str = Field(min_length=1)
    confirmation_required: Literal[True] = True
    sensitivity_alternatives: list[JsonValue] = Field(default_factory=list)

    @model_validator(mode='after')
    def registered(self):
        if self.field_path not in REGISTERED_ASSUMPTION_FIELDS | {'what_if_modifications'}:
            raise ValueError('UNREGISTERED_ASSUMPTION_FIELD:' + self.field_path)
        json.dumps(self.value, allow_nan=False)
        return self


class OccupancyCoefficient(ScenarioContract):
    ingredient_id: str
    base_unit: Literal['kg', 'liter', 'unit']
    liters_per_base_unit: float = Field(gt=0)
    lower_bound: float = Field(gt=0)
    upper_bound: float = Field(gt=0)
    assumption_id: str

    @model_validator(mode='after')
    def bounds(self):
        if not self.lower_bound <= self.liters_per_base_unit <= self.upper_bound:
            raise ValueError('OCCUPANCY_OUTSIDE_DECLARED_BOUNDS')
        return self


class ContractOption(ScenarioContract):
    contract_id: str = Field(min_length=1)
    source_rule_id: str = Field(min_length=1)
    proposed_lead_days: StrictInt = Field(ge=1, le=365)
    receiving_weekdays: list[StrictInt] = Field(min_length=1)
    receiving_dates: list[date] = Field(default_factory=list, exclude_if=lambda v: not v)
    extra_fee: float = Field(ge=0)
    assumption_id: str
    classification: Literal['CONTRACT_OPTION_REQUIRES_SUPPLIER_CONFIRMATION'] = 'CONTRACT_OPTION_REQUIRES_SUPPLIER_CONFIRMATION'

    @model_validator(mode='after')
    def weekdays(self):
        if any(d not in range(7) for d in self.receiving_weekdays):
            raise ValueError('INVALID_CONTRACT_WEEKDAY')
        return self


class ScenarioProfile(ScenarioContract):
    schema_version: Literal[1] = 1
    profile_id: str
    version: StrictInt = Field(ge=1)
    description: str
    planning_context: Literal['SCENARIO_PREVIEW'] = 'SCENARIO_PREVIEW'
    store_id: str
    selected_options: dict[str, str]
    default_rationale: str
    changed_from_shared_baseline: list[str]
    assumptions: list[Assumption]
    configuration_binding_hash: str = Field(pattern=r'^[0-9a-f]{64}$')

    @model_validator(mode='after')
    def binding(self):
        if set(self.selected_options) != {'U01','U02','U03','U04','U05','U06','U07'}:
            raise ValueError('SEVEN_SEMANTIC_GROUPS_REQUIRED')
        if any(v not in KNOWN_OPTIONS[k] for k,v in self.selected_options.items()):
            raise ValueError('UNKNOWN_SELECTED_SEMANTICS_OPTION')
        if len({a.assumption_id for a in self.assumptions}) != len(self.assumptions):
            raise ValueError('DUPLICATE_ASSUMPTION_ID')
        if len({a.field_path for a in self.assumptions}) != len(self.assumptions):
            raise ValueError('DUPLICATE_ASSUMPTION_FIELD')
        if any(a.profile_id != self.profile_id or a.scope != self.store_id for a in self.assumptions):
            raise ValueError('FOREIGN_ASSUMPTION_BINDING')
        return self


class NormalizedBusinessRule(ScenarioContract):
    rule_id: str
    source: dict[str, JsonValue]
    store_id: str
    ingredient_id: str | None
    unit: str
    target: float = Field(ge=0)
    semantics: Literal['RECEIVING_PEAK', 'END_OF_DAY_MAX', 'GLOBAL_RECEIVING_PEAK',
        'END_OF_DAY_MIN', 'PER_KEY_EXPECTED_FILL', 'PURCHASE_COVER_DAYS', 'MIN_REMAINING_LIFE_AT_RECEIVING']
    classification: Literal['HARD', 'SOFT', 'ADVISORY']
    effective_from: date | None = None
    effective_to: date | None = None
    penalty_currency_per_unit: float = Field(default=0, ge=0)
    assumption_refs: list[str]
    occupancy: list[OccupancyCoefficient] = Field(default_factory=list)
    demand_reference: Literal['FULL_POOL_WEIGHTED_HORIZON_DAILY_MEAN'] = 'FULL_POOL_WEIGHTED_HORIZON_DAILY_MEAN'
    reference_daily_quantity: float | None = Field(default=None, ge=0)

    @model_validator(mode='after')
    def valid_dimensions(self):
        if self.effective_to and self.effective_from and self.effective_to < self.effective_from:
            raise ValueError('INVALID_RULE_INTERVAL')
        if self.semantics == 'PER_KEY_EXPECTED_FILL' and (self.unit != 'ratio' or self.target > 1):
            raise ValueError('INVALID_SERVICE_RATIO')
        if self.semantics == 'GLOBAL_RECEIVING_PEAK' and (self.unit != 'liter' or not self.occupancy):
            raise ValueError('GLOBAL_OCCUPANCY_REQUIRED')
        if len({c.ingredient_id for c in self.occupancy}) != len(self.occupancy):
            raise ValueError('DUPLICATE_OCCUPANCY_KEY')
        if self.classification == 'SOFT' and (self.semantics != 'END_OF_DAY_MIN' or self.penalty_currency_per_unit <= 0):
            raise ValueError('SUPPORTED_SOFT_RULE_REQUIRES_CURRENCY_PER_UNIT')
        if self.semantics == 'PURCHASE_COVER_DAYS' and self.reference_daily_quantity is None:
            raise ValueError('COVER_DEMAND_REFERENCE_REQUIRED')
        if self.semantics in {'PURCHASE_COVER_DAYS','MIN_REMAINING_LIFE_AT_RECEIVING'} and self.unit != 'day':
            raise ValueError('FRESHNESS_UNIT_MUST_BE_DAY')
        return self

    def active(self, day):
        return (self.effective_from is None or day >= self.effective_from) and (self.effective_to is None or day <= self.effective_to)
