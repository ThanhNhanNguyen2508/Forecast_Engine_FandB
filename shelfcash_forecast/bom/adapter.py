from __future__ import annotations

from collections import defaultdict
from datetime import date

import pandas as pd
from pydantic import ValidationError

from shelfcash_forecast.bom.contracts import RecipeRecord, UnitConversionRule
from shelfcash_forecast.bom.units import normalize_unit
from shelfcash_forecast.contracts import ForecastPackage
from shelfcash_forecast.exceptions import (
    InvalidUnitConversionError,
    ProductUnitConsistencyError,
    RecipeValidationError,
)

RECIPE_REQUIRED_COLUMNS = { # shcema bắt buộc input phải đưa
    "recipe_id",
    "product_id",
    "ingredient_id",
    "ingredient_quantity",
    "ingredient_unit",
    "yield_quantity",
    "yield_unit",
    "recipe_version",
    "effective_from",
}
CONVERSION_REQUIRED_COLUMNS = { # schema bắt buộc đổi đơn vị phải đưa
    "ingredient_id",
    "from_unit",
    "to_unit",
    "factor",
}


def _missing_text(value: object) -> bool:
    return pd.isna(value) or not str(value).strip()


def _parse_date(value: object) -> date | None: # Phân tích ngày tháng
    if pd.isna(value) or (isinstance(value, str) and not value.strip()):
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed).date()


def _parse_number(value: object) -> float | None: # Phân tích số
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(parsed):
        return None
    return float(parsed)


def _parse_optional_rate(value: object) -> float | None: # Phân tích tỷ lệ tùy chọn
    if pd.isna(value) or (isinstance(value, str) and not value.strip()):
        return 0.0
    return _parse_number(value)


def _raise_product_unit_conflicts( # Đẩy lỗi khi có nhiều đơn vị chuẩn hóa trong một chuỗi store-product
    grouped_units: dict[tuple[str, str], set[str]],
) -> None:
    conflicts = [
        {
            "store_id": store_id,
            "product_id": product_id,
            "normalized_units": sorted(units),
        }
        for (store_id, product_id), units in sorted(grouped_units.items())
        if len(units) > 1
    ]
    if conflicts:
        raise ProductUnitConsistencyError(
            "Một store-product series không được có nhiều normalized product units.",
            details={"conflicts": conflicts},
        ) # 1 product trong 1 store chỉ được có 1 đơn vị chuẩn hóa

# Với cùng một (store, product), lịch sử sales có dùng nhất quán một normalized unit hay không? :
def validate_sales_product_unit_consistency( # 
    sales: pd.DataFrame,
    cutoff_date: str | pd.Timestamp | None = None,
) -> None:
    """Reject multiple normalized units in relevant store-product history."""

    required_columns = {
        "date",
        "store_key",
        "product_key",
        "product_name",
        "quantity_sold",
        "unit",
    }
    missing_columns = required_columns - set(sales.columns)
    if missing_columns:
        raise ProductUnitConsistencyError(
            "Không thể validate product unit vì canonical sales thiếu cột.",
            details={"missing_columns": sorted(missing_columns)},
        )

    valid = sales.loc[
        sales["date"].notna()
        & sales["product_name"].notna()
        & sales["product_name"].ne("")
        & sales["quantity_sold"].notna()
        & sales["quantity_sold"].ge(0)
    ]
    if cutoff_date is not None:
        cutoff = pd.Timestamp(cutoff_date).normalize()
        valid = valid.loc[valid["date"].le(cutoff)]

    grouped_units: dict[tuple[str, str], set[str]] = defaultdict(set) # gom normalized unit theo store-product
    for row in valid.itertuples(index=False):
        if _missing_text(row.unit):
            continue
        grouped_units[(str(row.store_key), str(row.product_key))].add(
            normalize_unit(str(row.unit))
        )
    _raise_product_unit_conflicts(grouped_units)

# tương tự trên nhưng dùng cho output từ M1,M2
def validate_forecast_product_unit_consistency(forecast: ForecastPackage) -> None: # 
    """Reject multiple normalized units in a ForecastPackage store-product series."""

    grouped_units: dict[tuple[str, str], set[str]] = defaultdict(set)
    for prediction in forecast.predictions:
        if prediction.unit is None or not prediction.unit.strip():
            continue
        grouped_units[(prediction.store_id, prediction.product_id)].add(
            normalize_unit(prediction.unit)
        )
    _raise_product_unit_conflicts(grouped_units)


def adapt_recipes(recipes: pd.DataFrame) -> list[RecipeRecord]: # chuyển từ raw input thành recipe record
    """Validate a canonical recipe table and return normalized records."""

    if not isinstance(recipes, pd.DataFrame):
        raise RecipeValidationError(
            "recipes phải là pandas DataFrame.",
            details={"received_type": type(recipes).__name__},
        )
    missing_columns = RECIPE_REQUIRED_COLUMNS - set(recipes.columns)
    if missing_columns:
        raise RecipeValidationError(
            f"recipes thiếu cột bắt buộc: {sorted(missing_columns)}",
            details={"missing_columns": sorted(missing_columns)},
        )
    if recipes.empty:
        raise RecipeValidationError("recipes không được rỗng.")

    frame = recipes.copy()
    if "process_loss_rate" not in frame.columns:
        frame["process_loss_rate"] = 0.0
    if "waste_allowance_rate" not in frame.columns:
        frame["waste_allowance_rate"] = 0.0
    if "effective_to" not in frame.columns:
        frame["effective_to"] = pd.NaT
    if "ingredient_name" not in frame.columns:
        frame["ingredient_name"] = pd.NA

    records: list[RecipeRecord] = []
    errors: list[dict[str, object]] = []
    for row_number, row in frame.reset_index(drop=True).iterrows(): # mỗi recipe row -> 1 recipe record
        row_errors: list[str] = []
        for column in (
            "recipe_id",
            "product_id",
            "ingredient_id",
            "ingredient_unit",
            "yield_unit",
            "recipe_version",
        ):
            if _missing_text(row[column]):
                row_errors.append(f"{column} is required")

        ingredient_quantity = _parse_number(row["ingredient_quantity"])
        yield_quantity = _parse_number(row["yield_quantity"])
        process_loss_rate = _parse_optional_rate(row["process_loss_rate"])
        waste_allowance_rate = _parse_optional_rate(row["waste_allowance_rate"])
        effective_from = _parse_date(row["effective_from"])
        effective_to = _parse_date(row["effective_to"])

        if ingredient_quantity is None or ingredient_quantity < 0:
            row_errors.append("ingredient_quantity must be >= 0")
        if yield_quantity is None or yield_quantity <= 0:
            row_errors.append("yield_quantity must be > 0")
        if process_loss_rate is None or process_loss_rate < 0:
            row_errors.append("process_loss_rate must be >= 0")
        if waste_allowance_rate is None or waste_allowance_rate < 0:
            row_errors.append("waste_allowance_rate must be >= 0")
        if effective_from is None:
            row_errors.append("effective_from must be a valid date")
        raw_effective_to = row["effective_to"]
        if (
            not pd.isna(raw_effective_to)
            and str(raw_effective_to).strip()
            and effective_to is None
        ):
            row_errors.append("effective_to must be a valid date or null")
        if (
            effective_from is not None
            and effective_to is not None
            and effective_to < effective_from
        ):
            row_errors.append("effective_to must be >= effective_from") # effective_to là ngày kết thúc hiệu lực, effective_from là ngày bắt đầu hiệu lực, effective_to phải lớn hơn hoặc bằng effective_from

        if row_errors:
            errors.append({"row": int(row_number), "errors": row_errors})
            continue

        try:
            records.append(
                RecipeRecord(
                    recipe_id=str(row["recipe_id"]).strip(),
                    product_id=str(row["product_id"]).strip(),
                    ingredient_id=str(row["ingredient_id"]).strip(),
                    ingredient_name=(
                        None
                        if _missing_text(row["ingredient_name"])
                        else str(row["ingredient_name"]).strip()
                    ),
                    ingredient_quantity=ingredient_quantity,
                    ingredient_unit=normalize_unit(str(row["ingredient_unit"])),
                    yield_quantity=yield_quantity,
                    yield_unit=normalize_unit(str(row["yield_unit"])),
                    process_loss_rate=process_loss_rate,
                    waste_allowance_rate=waste_allowance_rate,
                    recipe_version=str(row["recipe_version"]).strip(),
                    effective_from=effective_from,
                    effective_to=effective_to,
                )
            )
        except (ValidationError, ValueError) as exc:
            errors.append({"row": int(row_number), "errors": [str(exc)]})

    if errors:
        raise RecipeValidationError(
            "recipes chứa dữ liệu không hợp lệ.",
            details={"row_errors": errors},
        )

    _validate_recipe_metadata_consistency(records)
    return records


def _validate_recipe_metadata_consistency(records: list[RecipeRecord]) -> None: # gom ingredient của cùng 1 product vào chugn với nhau
    grouped: dict[tuple[str, str], list[RecipeRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.recipe_id, record.recipe_version)].append(record)

    inconsistent: list[dict[str, object]] = []
    metadata_fields = (
        "product_id",
        "recipe_id",
        "recipe_version",
        "yield_quantity",
        "yield_unit",
        "effective_from",
        "effective_to",
    )
    for (recipe_id, recipe_version), group in grouped.items():
        fields = [
            field_name
            for field_name in metadata_fields
            if len({getattr(record, field_name) for record in group}) > 1
        ]
        if fields:
            inconsistent.append(
                {
                    "recipe_id": recipe_id,
                    "recipe_version": recipe_version,
                    "inconsistent_fields": fields,
                }
            )
    if inconsistent:
        raise RecipeValidationError(
            "Metadata trong cùng recipe version không nhất quán.",
            details={"inconsistent_versions": inconsistent},
        )


def adapt_unit_conversions( # func dùng để validate metadata đổi đơn vị nguyên liệu
    unit_conversions: pd.DataFrame | None,
) -> list[UnitConversionRule]:
    """Validate optional ingredient-specific conversion metadata."""

    if unit_conversions is None:
        return []
    if not isinstance(unit_conversions, pd.DataFrame):
        raise InvalidUnitConversionError(
            "unit_conversions phải là pandas DataFrame.",
            details={"received_type": type(unit_conversions).__name__},
        )
    if unit_conversions.empty:
        return []
    missing_columns = CONVERSION_REQUIRED_COLUMNS - set(unit_conversions.columns)
    if missing_columns:
        raise InvalidUnitConversionError(
            f"unit_conversions thiếu cột bắt buộc: {sorted(missing_columns)}",
            details={"missing_columns": sorted(missing_columns)},
        )

    rules: list[UnitConversionRule] = []
    errors: list[dict[str, object]] = []
    seen: dict[tuple[str, str, str], float] = {}
    for row_number, row in unit_conversions.reset_index(drop=True).iterrows():
        row_errors: list[str] = []
        for column in ("ingredient_id", "from_unit", "to_unit"):
            if _missing_text(row[column]):
                row_errors.append(f"{column} is required")
        factor = _parse_number(row["factor"])
        if factor is None or factor <= 0:
            row_errors.append("factor must be > 0")
        if row_errors:
            errors.append({"row": int(row_number), "errors": row_errors})
            continue

        ingredient_id = str(row["ingredient_id"]).strip()
        from_unit = normalize_unit(str(row["from_unit"]))
        to_unit = normalize_unit(str(row["to_unit"]))
        key = (ingredient_id, from_unit, to_unit)
        previous = seen.get(key)
        if previous is not None and previous != factor:
            errors.append(
                {
                    "row": int(row_number),
                    "errors": ["conflicting conversion factor for the same rule"],
                }
            )
            continue
        seen[key] = factor
        rules.append(
            UnitConversionRule(
                ingredient_id=ingredient_id,
                from_unit=from_unit,
                to_unit=to_unit,
                factor=factor,
            )
        )

    if errors:
        raise InvalidUnitConversionError(
            "unit_conversions chứa dữ liệu không hợp lệ.",
            details={"row_errors": errors},
        )
    return rules
#                          adapter.py
#                              │
#           ┌──────────────────┼──────────────────┐
#           │                  │                  │
#           ▼                  ▼                  ▼
#    SALES HISTORY       FORECAST PACKAGE       RECIPES
#           │                  │                  │
#           ▼                  ▼                  ▼
# validate_sales_     validate_forecast_     adapt_recipes()
# product_unit_       product_unit_               │
# consistency()       consistency()                │
#           │                  │                  ▼
#           │                  │          Parse text/date/number
#           │                  │                  │
#           │                  │                  ▼
#           │                  │          Normalize units
#           │                  │                  │
#           │                  │                  ▼
#           │                  │          RecipeRecord
#           │                  │                  │
#           │                  │                  ▼
#           │                  │       Metadata consistency
#           │                  │
#           └────────────┬─────┘
#                        │
#                        │
#             UNIT CONVERSIONS
#                        │
#                        ▼
#          adapt_unit_conversions()
#                        │
#                        ▼
#               normalize units
#                        │
#                        ▼
#            detect conflicting rule
#                        │
#                        ▼
#          UnitConversionRule objects