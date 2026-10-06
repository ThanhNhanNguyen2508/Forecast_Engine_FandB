from __future__ import annotations

import math
from datetime import date
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from shelfcash_forecast.bom.contracts import BOMIssue


class StrictScenarioContract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ProductDemandScenarioLine(StrictScenarioContract):
    scenario_id: str = Field(min_length=1)
    store_id: str = Field(min_length=1)
    product_id: str = Field(min_length=1)
    product_name: str = Field(min_length=1)
    product_unit: str | None = None
    target_date: date
    horizon: int = Field(ge=1)
    demand_quantity: float = Field(ge=0)
    source_model_version: str = Field(min_length=1)
    scenario_method: str = Field(min_length=1)
# Đây là một dòng product demand trong một scenario.

# Hãy tưởng tượng table:

# Scenario	Store	Product	Date	Demand
# BASE	S01	LATTE	15/08	100
# BASE	S01	MOCHA	15/08	80
# HIGH	S01	LATTE	15/08	130

# Mỗi row tương ứng một:

# ProductDemandScenarioLine

class ProductDemandScenario(StrictScenarioContract):
    scenario_id: str = Field(min_length=1)
    probability_weight: float = Field(ge=0)
    lines: list[ProductDemandScenarioLine]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_lines(self) -> ProductDemandScenario:
        keys: set[tuple[str, str, date]] = set()
        for line in self.lines:
            if line.scenario_id != self.scenario_id:
                raise ValueError("Scenario line có scenario_id không khớp container.")
            key = (line.store_id, line.product_id, line.target_date)
            if key in keys:
                raise ValueError(f"SCENARIO_DUPLICATE_KEY: {key!r}")
            keys.add(key)
        return self
# BASE scenario

# probability = 0.50

# lines:
#  S01 LATTE  = 100
#  S01 MOCHA  = 80
#  S02 LATTE  = 120
#  ...

class ProductDemandScenarioBundle(StrictScenarioContract): # Đây là container cấp cao nhất của product scenarios.
    forecast_date: date
    horizon: int = Field(ge=1)
    model_version: str = Field(min_length=1)
    scenario_method: str = Field(min_length=1)
    scenarios: list[ProductDemandScenario]
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_scenarios(self) -> ProductDemandScenarioBundle:
        identifiers = [scenario.scenario_id for scenario in self.scenarios]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Scenario IDs phải duy nhất trong bundle.")
        if self.scenarios:
            total = sum(scenario.probability_weight for scenario in self.scenarios)
            if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError("Scenario probability weights phải có tổng bằng 1.")
        return self
# ProductDemandScenarioBundle
# │
# ├── Scenario LOW
# │    └── Product lines
# │
# ├── Scenario BASE
# │    └── Product lines
# │
# └── Scenario HIGH
#      └── Product lines


# nửa sau tương tự nhưng là đổi thành Ingredient từ product rồi
class IngredientScenarioContribution(StrictScenarioContract):
    product_id: str = Field(min_length=1)
    product_name: str = Field(min_length=1)
    recipe_id: str = Field(min_length=1)
    recipe_version: str = Field(min_length=1)
    fixed_recipe_quantity: float = Field(ge=0)
    final_quantity: float = Field(ge=0)
    unit: str = Field(min_length=1)


class IngredientDemandScenarioLine(StrictScenarioContract):
    scenario_id: str = Field(min_length=1)
    store_id: str = Field(min_length=1)
    ingredient_id: str = Field(min_length=1)
    ingredient_name: str | None = None
    target_date: date
    quantity: float = Field(ge=0)
    unit: str = Field(min_length=1)
    yield_loss_source: str = Field(min_length=1)
    yield_loss_multiplier: float = Field(gt=0)
    contributions: list[IngredientScenarioContribution] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class IngredientDemandScenario(StrictScenarioContract):
    scenario_id: str = Field(min_length=1)
    probability_weight: float = Field(ge=0)
    lines: list[IngredientDemandScenarioLine]
    issues: list[BOMIssue] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    is_complete: bool = True

    @model_validator(mode="after")
    def validate_lines(self) -> IngredientDemandScenario:
        keys: set[tuple[str, str, date]] = set()
        for line in self.lines:
            if line.scenario_id != self.scenario_id:
                raise ValueError("Ingredient line có scenario_id không khớp container.")
            key = (line.store_id, line.ingredient_id, line.target_date)
            if key in keys:
                raise ValueError(f"SCENARIO_DUPLICATE_KEY: {key!r}")
            keys.add(key)
        if self.is_complete == bool(self.issues):
            raise ValueError("is_complete phải phản ánh việc scenario có issues hay không.")
        return self


class IngredientDemandScenarioBundle(StrictScenarioContract):
    forecast_date: date
    horizon: int = Field(ge=1)
    forecast_model_version: str = Field(min_length=1)
    scenario_method: str = Field(min_length=1)
    scenarios: list[IngredientDemandScenario]
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    is_complete: bool = True

    @model_validator(mode="after")
    def validate_scenarios(self) -> IngredientDemandScenarioBundle:
        identifiers = [scenario.scenario_id for scenario in self.scenarios]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Ingredient scenario IDs phải duy nhất.")
        if self.scenarios:
            total = sum(scenario.probability_weight for scenario in self.scenarios)
            if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError("Ingredient scenario weights phải có tổng bằng 1.")
        if self.is_complete != all(scenario.is_complete for scenario in self.scenarios):
            raise ValueError("Bundle is_complete không khớp scenario completeness.")
        return self
