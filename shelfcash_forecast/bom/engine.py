# M1 / M2
# Sales History
#     ↓
# Forecast Engine
#     ↓
# ForecastPackage
#     │
#     │  Product forecast:
#     │  P25 / P50 / P75
#     │
#     ▼
# ===============================
# M3 - BOM / Ingredient Planning
# ===============================
#     │
#     ├── adapter.py
#     │     đọc + validate recipe/conversion table
#     │
#     ├── recipe_resolver.py
#     │     chọn recipe version active
#     │
#     ├── units.py
#     │     chuẩn hóa / convert unit
#     │
#     └── engine.py      ← FILE NÀY
#           ↓
# IngredientDemandPackage
#           ↓
# M4 / M5
# Purchasing / inventory / cash...

# Nói ngắn gọn:

# recipe_resolver.py trả lời “dùng công thức nào?”
# units.py trả lời “đổi đơn vị thế nào?”
# engine.py trả lời “vậy cuối cùng cần bao nhiêu nguyên liệu?”
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from shelfcash_forecast.bom.adapter import (
    adapt_recipes,
    adapt_unit_conversions,
    validate_forecast_product_unit_consistency,
)
from shelfcash_forecast.bom.contracts import (
    BOMIssue,
    IngredientDemandPackage,
    IngredientDemandPrediction,
    IngredientDemandSource,
    RecipeRecord,
)
from shelfcash_forecast.bom.recipe_resolver import RecipeResolver
from shelfcash_forecast.bom.units import (
    UnitConverter,
    convert_product_quantity,
    normalize_unit,
)
from shelfcash_forecast.contracts import ForecastPackage, ForecastPrediction
from shelfcash_forecast.exceptions import UnitConversionError


def ingredient_adjustment_factor( # logic thêm lượng nguyên liệu dự phòng do: process loss+waste allowance
    process_loss_rate: float,
    waste_allowance_rate: float,
) -> float:
    """Return the documented MVP multiplicative loss/waste factor."""

    return (1.0 + process_loss_rate) * (1.0 + waste_allowance_rate)


@dataclass
class _IngredientAggregate:
    store_id: str
    ingredient_id: str
    target_date: date
    unit: str
    ingredient_name: str | None = None
    p25: float = 0.0
    p50: float = 0.0
    p75: float = 0.0
    sources: list[IngredientDemandSource] = field(default_factory=list)
    warnings: set[str] = field(default_factory=set)


def _convert_forecast_to_yield_unit( # đồng nhất units forecast đang xài với qui đổi chuẩn
    prediction: ForecastPrediction,
    recipe: RecipeRecord,
    issues: list[BOMIssue],
) -> tuple[float, float, float, str | None]:
    product_unit = prediction.unit
    if product_unit is None or not product_unit.strip():
        issue_key = (
            prediction.store_id,
            prediction.product_id,
            prediction.target_date.isoformat(),
        )
        already_reported = any(
            issue.code == "PRODUCT_UNIT_MISSING"
            and (
                issue.details.get("store_id"),
                issue.details.get("product_id"),
                issue.details.get("target_date"),
            )
            == issue_key
            for issue in issues
        )
        if not already_reported:
            issues.append(
                BOMIssue(
                    code="PRODUCT_UNIT_MISSING",
                    message=(
                        f"Product {prediction.product_id} không có unit; BOM dùng "
                        f"yield_unit={recipe.yield_unit} làm working unit."
                    ),
                    details={
                        "store_id": prediction.store_id,
                        "product_id": prediction.product_id,
                        "target_date": prediction.target_date.isoformat(),
                        "yield_unit": recipe.yield_unit,
                    },
                    recoverable=True,
                    suggested_action="Bổ sung unit nhất quán trong sales_history.",
                )
            )
        return prediction.p25, prediction.p50, prediction.p75, None

    try:
        converted = tuple(
            convert_product_quantity(value, product_unit, recipe.yield_unit)
            for value in (prediction.p25, prediction.p50, prediction.p75)
        )
    except UnitConversionError as exc:
        raise UnitConversionError(
            str(exc),
            details={
                **exc.details,
                "store_id": prediction.store_id,
                "product_id": prediction.product_id,
                "target_date": prediction.target_date.isoformat(),
                "recipe_id": recipe.recipe_id,
                "recipe_version": recipe.recipe_version,
            },
        ) from exc
    return converted[0], converted[1], converted[2], normalize_unit(product_unit)


def _build_source( #propagate_ingredient_demand() là orchestrator, thì _build_source() chính là: calculator cho một product × một recipe ingredient row.
    prediction: ForecastPrediction,
    recipe: RecipeRecord,
    converter: UnitConverter,
    issues: list[BOMIssue],
) -> IngredientDemandSource:
    forecast_in_yield_unit = _convert_forecast_to_yield_unit(
        prediction, recipe, issues
    )
    forecast_values = forecast_in_yield_unit[:3]
    product_unit = forecast_in_yield_unit[3]
    base_in_recipe_unit = tuple(
        value * recipe.ingredient_quantity / recipe.yield_quantity
        for value in forecast_values
    )
    adjustment = ingredient_adjustment_factor(
        recipe.process_loss_rate,
        recipe.waste_allowance_rate,
    )
    adjusted_in_recipe_unit = tuple(
        value * adjustment for value in base_in_recipe_unit
    )

    output_unit = converter.canonical_unit(
        recipe.ingredient_id, recipe.ingredient_unit
    )
    base_output = tuple(
        converter.convert(
            value,
            recipe.ingredient_unit,
            output_unit,
            ingredient_id=recipe.ingredient_id,
        )
        for value in base_in_recipe_unit
    )
    adjusted_output = tuple(
        converter.convert(
            value,
            recipe.ingredient_unit,
            output_unit,
            ingredient_id=recipe.ingredient_id,
        )
        for value in adjusted_in_recipe_unit
    )

    return IngredientDemandSource(
        product_id=prediction.product_id,
        product_name=prediction.product_name,
        product_unit=product_unit,
        recipe_id=recipe.recipe_id,
        recipe_version=recipe.recipe_version,
        forecast_p25=prediction.p25,
        forecast_p50=prediction.p50,
        forecast_p75=prediction.p75,
        recipe_quantity=recipe.ingredient_quantity,
        recipe_unit=recipe.ingredient_unit,
        yield_quantity=recipe.yield_quantity,
        yield_unit=recipe.yield_unit,
        process_loss_rate=recipe.process_loss_rate,
        waste_allowance_rate=recipe.waste_allowance_rate,
        base_contribution_p25=base_output[0],
        base_contribution_p50=base_output[1],
        base_contribution_p75=base_output[2],
        contribution_p25=adjusted_output[0],
        contribution_p50=adjusted_output[1],
        contribution_p75=adjusted_output[2],
        contribution_unit=output_unit,
    )


def propagate_ingredient_demand(
    forecast: ForecastPackage, # Store-product-date và P25->75 từ các milestone trước
    recipes: pd.DataFrame, # recipe_id, recipe_version, ingredient_id, ingredient_quantity, ingredient_unit, yield_quantity, yield_unit, process_loss_rate, waste_allowance_rate
    unit_conversions: pd.DataFrame | None = None, # unit đổi
) -> IngredientDemandPackage:
    # output là all nguyên liệu :
# 2026-08-15
# Store S01

# Coffee:
# P25 = 1.76 kg
# P50 = 2.20 kg
# P75 = 2.64 kg

# Milk:
# P25 = ...
# P50 = ...
# P75 = ...
    """Deterministically propagate product quantiles through active recipes.

    Formula per source row and scenario::

        base = product_forecast * ingredient_quantity / yield_quantity
        adjusted = base * (1 + process_loss_rate) * (1 + waste_allowance_rate)

    Product P25/P50/P75 are propagated directly. CQR interval bounds are not
    used by this service.
    """

    validate_forecast_product_unit_consistency(forecast)
    recipe_records = adapt_recipes(recipes)
    conversion_rules = adapt_unit_conversions(unit_conversions)
    resolver = RecipeResolver(recipe_records)
    converter = UnitConverter(conversion_rules)

    aggregates: dict[tuple[str, str, date], _IngredientAggregate] = {}
    issues: list[BOMIssue] = []
    warning_codes = set(forecast.warnings)

    ordered_predictions = sorted(
        forecast.predictions,
        key=lambda item: (
            item.target_date,
            item.store_id,
            item.product_id,
            item.horizon,
        ),
    )
    for prediction in ordered_predictions:
        resolution = resolver.resolve(
            prediction.product_id,
            prediction.target_date,
            store_id=prediction.store_id,
        )
        if not resolution.found:
            assert resolution.issue is not None
            issues.append(resolution.issue)
            warning_codes.add(resolution.issue.code)
            continue

        issue_count_before = len(issues)
        for recipe in resolution.records:
            source = _build_source(prediction, recipe, converter, issues)
            key = (
                prediction.store_id,
                recipe.ingredient_id,
                prediction.target_date,
            )
            aggregate = aggregates.get(key)
            if aggregate is None:
                aggregate = _IngredientAggregate(
                    store_id=prediction.store_id,
                    ingredient_id=recipe.ingredient_id,
                    ingredient_name=recipe.ingredient_name,
                    target_date=prediction.target_date,
                    unit=source.contribution_unit,
                )
                aggregates[key] = aggregate
            elif aggregate.unit != source.contribution_unit:
                raise UnitConversionError(
                    "Không thể aggregate ingredient contributions khác unit.",
                    details={
                        "store_id": prediction.store_id,
                        "ingredient_id": recipe.ingredient_id,
                        "target_date": prediction.target_date.isoformat(),
                        "existing_unit": aggregate.unit,
                        "incoming_unit": source.contribution_unit,
                    },
                )

            if (
                aggregate.ingredient_name is not None
                and recipe.ingredient_name is not None
                and aggregate.ingredient_name != recipe.ingredient_name
            ):
                aggregate.ingredient_name = None
                aggregate.warnings.add("INCONSISTENT_INGREDIENT_NAME")
                warning_codes.add("INCONSISTENT_INGREDIENT_NAME")
            elif (
                aggregate.ingredient_name is None
                and recipe.ingredient_name is not None
                and "INCONSISTENT_INGREDIENT_NAME" not in aggregate.warnings
            ):
                aggregate.ingredient_name = recipe.ingredient_name

            aggregate.p25 += source.contribution_p25
            aggregate.p50 += source.contribution_p50
            aggregate.p75 += source.contribution_p75
            aggregate.sources.append(source)
            aggregate.warnings.update(prediction.warnings)

        if len(issues) > issue_count_before:
            warning_codes.add("PRODUCT_UNIT_MISSING")

    predictions = [
        IngredientDemandPrediction(
            store_id=aggregate.store_id,
            ingredient_id=aggregate.ingredient_id,
            ingredient_name=aggregate.ingredient_name,
            target_date=aggregate.target_date,
            p25=aggregate.p25,
            p50=aggregate.p50,
            p75=aggregate.p75,
            unit=aggregate.unit,
            sources=sorted(
                aggregate.sources,
                key=lambda source: (
                    source.product_id,
                    source.recipe_id,
                    source.recipe_version,
                    source.recipe_unit,
                ),
            ),
            warnings=sorted(aggregate.warnings),
        )
        for aggregate in sorted(
            aggregates.values(),
            key=lambda item: (
                item.target_date,
                item.store_id,
                item.ingredient_id,
            ),
        )
    ]

    return IngredientDemandPackage(
        forecast_date=forecast.forecast_date,
        forecast_horizon=forecast.forecast_horizon,
        forecast_model_version=forecast.model_version,
        predictions=predictions,
        issues=issues,
        warnings=sorted(warning_codes),
        is_complete=not issues,
    )
