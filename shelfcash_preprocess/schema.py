from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from shelfcash_preprocess.models import SCHEMA_VERSION


Kind = Literal["string", "float", "int", "boolean", "date"]


@dataclass(frozen=True)
class ColumnSpec:
    kind: Kind
    nullable: bool = True


@dataclass(frozen=True)
class TableSchema:
    columns: dict[str, ColumnSpec]
    grain: tuple[str, ...] = ()
    allow_empty: bool = True


S = ColumnSpec
TABLE_SCHEMAS: dict[str, TableSchema] = {
    "sales": TableSchema({
        "date": S("date", False), "store_id": S("string", False),
        "product_id": S("string", False), "product_name": S("string", False),
        "quantity_sold": S("float", False), "unit": S("string"),
        "selling_price": S("float"), "revenue": S("float"),
        "is_stockout": S("boolean"), "promotion_name": S("string"),
    }, ("date", "store_id", "product_id")),
    "calendar": TableSchema({
        "date": S("date", False), "store_id": S("string"),
        "is_weekend": S("boolean"), "is_holiday": S("boolean"),
        "is_store_closed": S("boolean"), "is_promotion": S("boolean"),
        "promotion_name": S("string"), "temperature": S("float"),
        "rainfall": S("float"), "available_as_of": S("date"),
        "weather_kind": S("string"), "weather_issued_at": S("date"),
        "weather_available_at": S("date"),
    }, ("date", "store_id")),
    "menu": TableSchema({
        "product_id": S("string", False), "product_name": S("string", False),
        "product_type": S("string"), "components": S("string"),
        "unit": S("string"), "selling_price": S("float"), "status": S("string"),
    }, ("product_id",)),
    "recipes": TableSchema({
        "recipe_id": S("string", False), "product_id": S("string", False),
        "product_name": S("string", False), "ingredient_id": S("string", False),
        "ingredient_name": S("string", False), "ingredient_quantity": S("float", False),
        "ingredient_unit": S("string", False), "yield_quantity": S("float", False),
        "yield_unit": S("string", False), "process_loss_rate": S("float", False),
        "waste_allowance_rate": S("float", False), "recipe_version": S("string", False),
        "effective_from": S("date", False), "effective_to": S("date"),
    }, ("recipe_id", "ingredient_id", "effective_from")),
    "ingredient_usage": TableSchema({
        "date": S("date", False), "store_id": S("string", False),
        "ingredient_id": S("string", False), "ingredient_name": S("string", False),
        "actual_usage_quantity": S("float", False), "unit": S("string", False),
        "waste_quantity": S("float"), "source": S("string"),
    }, ("date", "store_id", "ingredient_id")),
    "inventory_snapshot": TableSchema({
        "snapshot_date": S("date", False), "store_id": S("string", False),
        "ingredient_id": S("string", False), "ingredient_name": S("string", False),
        "quantity_remaining": S("float", False), "unit": S("string", False),
        "expiry_date": S("date"), "lot_id": S("string", False),
        "received_date": S("date"), "location": S("string"),
        "source_type": S("string", False),
    }, ("store_id", "lot_id")),
    "purchase_history": TableSchema({
        "document_date": S("date"), "received_date": S("date", False),
        "ingredient_id": S("string", False), "ingredient_name": S("string", False),
        "quantity": S("float", False), "unit": S("string", False),
        "unit_price": S("float"), "line_amount": S("float"),
        "supplier_id": S("string"), "expiry_date": S("date"),
        "lot_id": S("string"), "purchase_order_id": S("string"),
        "simulation_policy": S("string", False),
    }),
    "supplier_rules": TableSchema({
        "supplier_id": S("string", False), "ingredient_id": S("string", False),
        "ingredient_name": S("string", False),
        "minimum_order_quantity": S("float"), "order_unit": S("string"),
        "pack_size": S("float", False), "unit": S("string", False),
        "lead_time_days": S("int", False), "unit_price": S("float", False),
        "price_basis": S("string", False), "delivery_schedule": S("string"),
        "shelf_life_days": S("int"),
    }, ("supplier_id", "ingredient_id")),
    "business_rules": TableSchema({
        "rule_type": S("string", False), "ingredient_id": S("string"),
        "ingredient_name": S("string"), "value": S("float", False),
        "unit": S("string", False), "currency": S("string"),
        "effective_from": S("date"), "note": S("string"),
        "application_status": S("string", False),
    }),
}


def schema_document() -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "na_policy": "empty CSV field is missing; literal NA/NULL identifiers are preserved",
        "tables": {
            name: {
                "columns": {
                    column: {"kind": spec.kind, "nullable": spec.nullable}
                    for column, spec in schema.columns.items()
                },
                "grain": list(schema.grain),
                "allow_empty": schema.allow_empty,
            }
            for name, schema in TABLE_SCHEMAS.items()
        },
    }


def _coerce_boolean(series: pd.Series, table: str, column: str) -> pd.Series:
    text = series.astype("string")
    normalized = text.str.strip().str.casefold()
    mapping = {"true": True, "false": False, "1": True, "0": False}
    invalid = normalized.notna() & normalized.ne("") & ~normalized.isin(mapping)
    if invalid.any():
        values = sorted(normalized.loc[invalid].dropna().unique().tolist())[:5]
        raise ValueError(f"CANONICAL_SCHEMA_INVALID_BOOLEAN:{table}.{column}:{values}")
    return normalized.map(mapping).astype("boolean")


def coerce_frame(table: str, frame: pd.DataFrame) -> pd.DataFrame:
    try:
        schema = TABLE_SCHEMAS[table]
    except KeyError as exc:
        raise ValueError(f"CANONICAL_SCHEMA_UNKNOWN_TABLE:{table}") from exc
    missing = [column for column in schema.columns if column not in frame.columns]
    unexpected = [column for column in frame.columns if column not in schema.columns]
    if missing or unexpected:
        raise ValueError(
            f"CANONICAL_SCHEMA_COLUMNS:{table}:missing={missing}:unexpected={unexpected}"
        )
    output = frame[list(schema.columns)].copy()
    for column, spec in schema.columns.items():
        raw = output[column]
        if spec.kind == "string":
            value = raw.astype("string").str.strip()
            value = value.mask(value.eq(""), pd.NA)
        elif spec.kind in {"float", "int"}:
            blank = raw.astype("string").str.strip().eq("")
            value = pd.to_numeric(raw.mask(blank, pd.NA), errors="coerce")
            invalid = raw.notna() & ~blank & value.isna()
            if invalid.any():
                raise ValueError(f"CANONICAL_SCHEMA_INVALID_NUMBER:{table}.{column}")
            value = value.astype("Float64" if spec.kind == "float" else "Int64")
        elif spec.kind == "boolean":
            value = _coerce_boolean(raw, table, column)
        else:
            text = raw.astype("string").str.strip()
            text = text.mask(text.eq(""), pd.NA)
            bad_format = text.notna() & ~text.str.match(r"^\d{4}-\d{2}-\d{2}$")
            if bad_format.any():
                raise ValueError(f"CANONICAL_SCHEMA_DATE_NOT_ISO:{table}.{column}")
            value = pd.to_datetime(text, format="%Y-%m-%d", errors="coerce")
            if (text.notna() & value.isna()).any():
                raise ValueError(f"CANONICAL_SCHEMA_INVALID_DATE:{table}.{column}")
        if not spec.nullable and value.isna().any():
            raise ValueError(f"CANONICAL_SCHEMA_NULL_REQUIRED:{table}.{column}")
        output[column] = value
    if not schema.allow_empty and output.empty:
        raise ValueError(f"CANONICAL_SCHEMA_EMPTY:{table}")
    if schema.grain and not output.empty and output.duplicated(list(schema.grain)).any():
        raise ValueError(f"CANONICAL_SCHEMA_DUPLICATE_GRAIN:{table}:{schema.grain}")
    return output


def write_frame(table: str, frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    typed = coerce_frame(table, frame)
    serial = typed.copy()
    for column, spec in TABLE_SCHEMAS[table].columns.items():
        if spec.kind == "date":
            serial[column] = typed[column].dt.strftime("%Y-%m-%d")
        elif spec.kind == "boolean":
            serial[column] = typed[column].map({True: "true", False: "false"}).astype("string")
    path.parent.mkdir(parents=True, exist_ok=True)
    serial.to_csv(path, index=False, na_rep="", lineterminator="\n")
    return typed


def read_frame(table: str, path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, dtype="string", keep_default_na=False, na_filter=False)
    return coerce_frame(table, raw)
