"""Validated outer planning JSON. Legacy migration is explicit and never approval."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, field_validator, model_validator

from shelfcash_forecast.inventory.stress import StressScenarioDefinition
from shelfcash_forecast.optimization.contracts import SolverLimits, StrategyProfile, SupplyCalendar
from shelfcash_forecast.optimization.scenario_contracts import (ScenarioProfile, OccupancyCoefficient, ContractOption,
    REGISTERED_ASSUMPTION_FIELDS, semantic_hash)


class ConfigContract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    @field_validator("*", mode="before")
    @classmethod
    def reject_bool_numbers(cls, value, info):
        annotation = str(cls.model_fields[info.field_name].annotation)
        if isinstance(value, bool) and ("float" in annotation or "int" in annotation):
            raise ValueError("BOOLEAN_NOT_NUMERIC:" + info.field_name)
        return value


class CostPolicy(ConfigContract):
    holding_cost_rate_per_day: float = Field(ge=0)
    shortage_cost_multiplier: float = Field(ge=0)
    expired_cost_multiplier: float = Field(ge=0)
    waste_cost_multiplier: float = Field(ge=0)
    reference_price: Literal["MIN_NORMALIZED_OFFER_PRICE_DEMO"] = "MIN_NORMALIZED_OFFER_PRICE_DEMO"
    purchase_price_source: str | None = None
    units: str | None = None
    rationale: str | None = None


class RuleMapping(ConfigContract):
    rule_type: str
    ingredient_id: str | None = None
    unit: str
    semantics: Literal['RECEIVING_PEAK', 'END_OF_DAY_MAX', 'GLOBAL_RECEIVING_PEAK', 'END_OF_DAY_MIN',
                       'PER_KEY_EXPECTED_FILL', 'PURCHASE_COVER_DAYS', 'MIN_REMAINING_LIFE_AT_RECEIVING']
    classification: Literal['HARD', 'SOFT', 'ADVISORY'] = 'HARD'
    penalty_currency_per_unit: float = Field(default=0, ge=0)
    evidence: str


class PriceBasisMapping(ConfigContract):
    basis: Literal["BASE_UNIT", "PACK"]
    evidence: str = Field(min_length=1)


class PlanningConfig(ConfigContract):
    schema_version: Literal[2, 3] = 2
    planning_mode: Literal['BASELINE', 'SCENARIO_PREVIEW'] = 'BASELINE'
    candidate_generation: Literal['JOINT_MILP','DECOMPOSED_FIXED_CERTIFICATION'] = 'JOINT_MILP'
    scenario_profile: ScenarioProfile | None = None
    occupancy_coefficients: list[OccupancyCoefficient] = Field(default_factory=list)
    contract_options: list[ContractOption] = Field(default_factory=list)
    label: Literal["DEMO_ONLY_NOT_FOR_OPERATION", "PENDING_BUSINESS_INPUT"]
    description: str | None = None
    seed: StrictInt | None = None
    scenario_count: StrictInt | None = Field(default=None, ge=1, le=2000)
    optimization_scenario_count: StrictInt = Field(default=100, ge=1, le=2000)
    selection_seed: StrictInt = 42
    scenario_method: Literal["residual_bootstrap","residual_bootstrap_with_declared_overrides","declared_cold_start_levels"] = "residual_bootstrap"
    evaluation: Literal["FULL_M4_POOL"] = "FULL_M4_POOL"
    budget: float | None = Field(default=None, ge=0)
    budget_scope: Literal["ALL_COMMITTED_REGULAR_PURCHASE_PLUS_DELIVERY"] = "ALL_COMMITTED_REGULAR_PURCHASE_PLUS_DELIVERY"
    currency: str = "VND"
    cost_policy: CostPolicy
    strategy_profiles: list[StrategyProfile] = Field(default_factory=list)
    supplier_calendar: SupplyCalendar = Field(default_factory=SupplyCalendar)
    supplier_calendars: dict[str, SupplyCalendar] = Field(default_factory=dict)
    packaging_mappings: dict[str, Literal["PACK_COUNT", "BASE_QUANTITY"]] = Field(default_factory=dict)
    price_basis_mappings: dict[str, PriceBasisMapping] = Field(default_factory=dict)
    rule_mappings: list[RuleMapping] = Field(default_factory=list)
    stress_scenarios: list[StressScenarioDefinition] = Field(default_factory=list)
    stress_policy: Literal["DIAGNOSTIC_ONLY"] = "DIAGNOSTIC_ONLY"
    stress_base_scenario_id: str | None = None
    limits: SolverLimits = Field(default_factory=SolverLimits)
    delivery_cost_assumption: float | None = Field(default=None, ge=0)
    delivery_fee_scope: Literal['PER_INGREDIENT_ORDER_OPPORTUNITY', 'SUPPLIER_STORE_ORDER_ARRIVAL_GROUP'] = 'PER_INGREDIENT_ORDER_OPPORTUNITY'
    metadata: dict[str, str | list[str] | StrictBool] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_supported(self):
        if self.currency != "VND":
            raise ValueError("UNSUPPORTED_CURRENCY:only explicit VND accounting is supported")
        mappings = [(m.rule_type, m.ingredient_id) for m in self.rule_mappings]
        if len(set(mappings)) != len(mappings):
            raise ValueError("DUPLICATE_RULE_MAPPING")
        if self.planning_mode == 'SCENARIO_PREVIEW':
            if self.schema_version != 3 or self.scenario_profile is None:
                raise ValueError('VERSIONED_PREVIEW_PROFILE_REQUIRED')
            from shelfcash_forecast.optimization.strategies import default_strategy_profiles
            floors={p.name:p for p in default_strategy_profiles()}
            if any(p.minimum_acceptable_fill_rate<floors[p.name].minimum_acceptable_fill_rate or
                   p.maximum_acceptable_stockout_probability>floors[p.name].maximum_acceptable_stockout_probability
                   for p in self.strategy_profiles):
                raise ValueError('PREVIEW_CANNOT_WEAKEN_BASELINE_SAFEGUARDS')
            fields = self.model_dump(mode='json')
            behavior = {k: fields[k] for k in sorted(REGISTERED_ASSUMPTION_FIELDS)}
            if semantic_hash(behavior) != self.scenario_profile.configuration_binding_hash:
                raise ValueError('STALE_PROFILE_CONFIGURATION_BINDING')
            assumptions = {a.field_path: a for a in self.scenario_profile.assumptions}
            if set(assumptions) != REGISTERED_ASSUMPTION_FIELDS:
                raise ValueError('COMPLETE_REGISTERED_ASSUMPTIONS_REQUIRED')
            for k, value in behavior.items():
                if semantic_hash(assumptions[k].value) != semantic_hash(value):
                    raise ValueError('ASSUMPTION_VALUE_MISMATCH:' + k)
            ids = {a.assumption_id for a in assumptions.values()}
            if any(c.assumption_id not in ids for c in [*self.occupancy_coefficients, *self.contract_options]):
                raise ValueError('UNBOUND_BEHAVIORAL_ASSUMPTION')
            if any(c.status != 'ASSUMED_FOR_SCENARIO' for c in [self.supplier_calendar, *self.supplier_calendars.values()]):
                raise ValueError('PREVIEW_CALENDAR_MUST_RETAIN_ASSUMED_STATUS')
        elif self.scenario_profile or self.contract_options or self.occupancy_coefficients or self.supplier_calendar.status == 'ASSUMED_FOR_SCENARIO':
            raise ValueError('SCENARIO_TERMS_REQUIRE_PREVIEW_CONTEXT')
        return self


LEGACY_KEYS = {
    "label", "description", "seed", "scenario_count", "optimization_scenario_count",
    "optimization_scenario_note", "scenario_method", "strategy_profiles", "budget", "budget_semantics",
    "cost_policy", "capacity_policy", "stress", "production_readiness", "required_replacements_before_operation",
}


def load_planning_config(source: Path | dict, *, scenario_count: int | None = None,
                         seed: int | None = None) -> tuple[PlanningConfig, list[dict]]:
    raw = json.loads(source.read_text(encoding="utf-8-sig")) if isinstance(source, Path) else dict(source)
    migration: list[dict] = []
    legacy = "schema_version" not in raw
    if legacy:
        unknown = set(raw) - LEGACY_KEYS
        if unknown:
            raise ValueError("UNKNOWN_LEGACY_PLANNING_FIELDS:" + ",".join(sorted(unknown)))
        profiles = raw.get("strategy_profiles", {})
        if set(profiles) - {"source", "names"} or profiles.get("names", ["LEAN", "BALANCED", "PROTECTED"]) != ["LEAN", "BALANCED", "PROTECTED"]:
            raise ValueError("UNSUPPORTED_LEGACY_STRATEGY_PROFILES")
        if profiles.get("source", "shelfcash_forecast.optimization.strategies.default_strategy_profiles") != "shelfcash_forecast.optimization.strategies.default_strategy_profiles":
            raise ValueError("UNSUPPORTED_LEGACY_STRATEGY_SOURCE")
        capacity = raw.get("capacity_policy", {})
        if set(capacity) - {"applied", "reason"} or capacity.get("applied", False) is not False:
            raise ValueError("LEGACY_CAPACITY_MAPPING_REQUIRED")
        stress = raw.get("stress", {})
        if set(stress) - {"demand_multiplier", "supplier_delay_days", "semantics"}:
            raise ValueError("UNKNOWN_LEGACY_STRESS_FIELDS")
        metadata = {k: raw[k] for k in ("optimization_scenario_note", "budget_semantics", "production_readiness", "required_replacements_before_operation") if k in raw}
        data = {k: raw[k] for k in ("label", "description", "seed", "optimization_scenario_count", "scenario_method", "budget", "cost_policy") if k in raw}
        data.update(schema_version=2, strategy_profiles=[], metadata=metadata,
                    stress_scenarios=([{"stress_id": "LEGACY_EXPLICIT_STRESS", "demand_multiplier": stress.get("demand_multiplier", 1),
                                        "supplier_delay_days": stress.get("supplier_delay_days", 0), "description": stress.get("semantics")}]
                                      if stress else []))
        # These old controls never reached M4 in this caller. The override is now recorded.
        data["scenario_count"] = scenario_count if scenario_count is not None else raw.get("scenario_count")
        migration.append({"from_version": 1, "to_version": 2, "legacy_scenario_count": raw.get("scenario_count"),
                          "resolved_scenario_count": data["scenario_count"], "authority": "explicit_caller_M4_count",
                          "calendar": "UNRESOLVED", "capacity": "PER_RULE_MAPPING_REQUIRED",
                          "stress": "IMPLEMENTED_DIAGNOSTIC_ONLY", "business_approval": "NOT_GRANTED"})
    else:
        data = raw
    config = PlanningConfig.model_validate_json(json.dumps(data, allow_nan=False))
    if scenario_count is not None and config.scenario_count is not None and config.scenario_count != scenario_count:
        raise ValueError("CONFIG_CLI_CONFLICT:scenario_count")
    if seed is not None and config.seed is not None and config.seed != seed:
        raise ValueError("CONFIG_CLI_CONFLICT:seed")
    return config, migration
