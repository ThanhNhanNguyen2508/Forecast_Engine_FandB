from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictBOMContract(BaseModel):
    """Strict, finite-number contract shared by Recipe/BOM payloads."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False) # Disallow extra fields and infinite/nan values


class RecipeRecord(StrictBOMContract): # Một dòng nguyên liệu trong một version của công thức của một sản phẩm.
    recipe_id: str = Field(min_length=1) # 
    product_id: str = Field(min_length=1)
    ingredient_id: str = Field(min_length=1)
    ingredient_name: str | None = None # ên của nguyên liệu, có thể None nếu không cần thiết
    ingredient_quantity: float = Field(ge=0)
    ingredient_unit: str = Field(min_length=1) #
    yield_quantity: float = Field(gt=0) # 
    yield_unit: str = Field(min_length=1)
    process_loss_rate: float = Field(default=0, ge=0) # hao hụt trong quá trình chế biến, ví dụ 0.05 = 5% hao hụt
    waste_allowance_rate: float = Field(default=0, ge=0) # tỷ lệ hao hụt cho phép, ví dụ 0.02 = 2% hao hụt
    recipe_version: str = Field(min_length=1) 
    effective_from: date
    effective_to: date | None = None
#     RecipeRecord(
#     recipe_id="CF_SUA_V2",
#     product_id="CAFE_SUA",
#     ingredient_id="COFFEE_BEAN",
#     ingredient_name="Coffee bean",
#     ingredient_quantity=20,
#     ingredient_unit="g",
#     yield_quantity=1,
#     yield_unit="cup",
#     process_loss_rate=0.05,
#     waste_allowance_rate=0.02,
#     recipe_version="v2",
#     effective_from=date(2026, 1, 1),
# )

    @field_validator(
        "recipe_id",
        "product_id",
        "ingredient_id",
        "ingredient_unit",
        "yield_unit",
        "recipe_version",
        mode="before",
    )
    @classmethod
    def strip_required_text(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @model_validator(mode="after")
    def validate_effective_range(self) -> RecipeRecord:
        if self.effective_to is not None and self.effective_to < self.effective_from:
            raise ValueError("effective_to không được trước effective_from.")
        return self


class UnitConversionRule(StrictBOMContract): # đổi đơn vị 
    ingredient_id: str = Field(min_length=1) # 
    from_unit: str = Field(min_length=1)
    to_unit: str = Field(min_length=1)
    factor: float = Field(gt=0)


class BOMIssue(StrictBOMContract): # structured error/warning object của M3.
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    details: dict[str, Any] = Field(default_factory=dict)
    recoverable: bool
    suggested_action: str | None = None
# BOMIssue(
#     code="MISSING_RECIPE",
#     message="Không tìm thấy recipe cho product A",
#     details={
#         "product_id": "A",
#         "target_date": "2026-08-15",
#     },
#     recoverable=True,
#     suggested_action="Add recipe before rerunning."
# )

class IngredientDemandSource(StrictBOMContract): # Một product đã đóng góp bao nhiêu nhu cầu vào một ingredient cụ thể.
    # Nhóm 1 : demand nguyên liệu này bắt nguồn từ nhóm product (sp) nào :
    product_id: str
    product_name: str
    product_unit: str | None = None
    # Nhóm 2 : demand nguyên liệu này bắt nguồn từ recipe nào của product đó :
    recipe_id: str
    recipe_version: str
    # Nhóm 3 : demand nguyên liệu này bắt nguồn từ forecast nào của product đó :
    forecast_p25: float = Field(ge=0)
    forecast_p50: float = Field(ge=0)
    forecast_p75: float = Field(ge=0)
    # Nhóm 4 : recipe parameter
    recipe_quantity: float = Field(ge=0)
    recipe_unit: str
    yield_quantity: float = Field(gt=0)
    yield_unit: str
    # Nhóm 5 : loss para
    process_loss_rate: float = Field(ge=0)
    waste_allowance_rate: float = Field(ge=0)
    # Nhóm 6 : contribution trước loss
    base_contribution_p25: float = Field(ge=0)
    base_contribution_p50: float = Field(ge=0)
    base_contribution_p75: float = Field(ge=0)
    # Nhóm 7 : contribution sau loss
    contribution_p25: float = Field(ge=0)
    contribution_p50: float = Field(ge=0)
    contribution_p75: float = Field(ge=0)
    contribution_unit: str

    @model_validator(mode="after")
    def validate_quantile_ordering(self) -> IngredientDemandSource:
        groups = (
            (self.forecast_p25, self.forecast_p50, self.forecast_p75),
            (
                self.base_contribution_p25,
                self.base_contribution_p50,
                self.base_contribution_p75,
            ),
            (self.contribution_p25, self.contribution_p50, self.contribution_p75),
        )
        if any(not low <= median <= high for low, median, high in groups):
            raise ValueError("BOM source quantiles phải thỏa P25 <= P50 <= P75.")
        return self


class IngredientDemandPrediction(StrictBOMContract): # Tổng demand cuối cùng của một ingredient, tại một store, vào một target date.
    store_id: str
    ingredient_id: str
    ingredient_name: str | None = None
    target_date: date
    p25: float = Field(ge=0)
    p50: float = Field(ge=0)
    p75: float = Field(ge=0)
    unit: str = Field(min_length=1)
    sources: list[IngredientDemandSource] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_ordering(self) -> IngredientDemandPrediction:
        if not self.p25 <= self.p50 <= self.p75:
            raise ValueError("Ingredient quantiles phải thỏa P25 <= P50 <= P75.")
        return self


class IngredientDemandPackage(StrictBOMContract): # Gói dữ liệu dự báo nhu cầu nguyên liệu
    forecast_date: date
    forecast_horizon: int = Field(ge=1)
    forecast_model_version: str
    predictions: list[IngredientDemandPrediction]
    issues: list[BOMIssue] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    is_complete: bool = True
