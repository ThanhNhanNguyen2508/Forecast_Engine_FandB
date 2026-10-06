from __future__ import annotations

from dataclasses import replace
from typing import Any

import pandas as pd

from shelfcash_preprocess.discovery import RegionData
from shelfcash_preprocess.models import MappingProposal, TransformOperation
from shelfcash_preprocess.values import parse_boolean, parse_date, parse_number


OPERATION_SUPPORT: dict[str, str] = {
    "rename": "supported_metadata",
    "trim": "supported",
    "cast": "supported",
    "parse_date": "supported",
    "parse_number": "supported",
    "parse_boolean": "supported",
    "filter_total_rows": "supported",
    "unpivot": "supported_by_canonical_wide_path",
    "lookup_id": "supported_exact_mapping_only",
    "unit_conversion": "supported_explicit_factor",
    "split": "unsupported_requires_review",
    "combine": "unsupported_requires_review",
    "aggregate": "unsupported_requires_review",
}


class OperationPlanError(ValueError):
    pass


def operation_problem(operation: TransformOperation) -> str | None:
    name = operation.operation
    status = OPERATION_SUPPORT[name]
    args = operation.arguments
    if status.startswith("unsupported"):
        return f"UNSUPPORTED_OPERATION:{name}"
    if name == "cast" and args.get("dtype") not in {"string", "float", "int", "boolean"}:
        return "UNSUPPORTED_OPERATION_VARIANT:cast"
    if name == "parse_date" and args.get("locale") not in {None, "DMY", "MDY", "YMD"}:
        return "UNSUPPORTED_OPERATION_VARIANT:parse_date"
    if name == "lookup_id" and not isinstance(args.get("mapping"), dict):
        return "UNSUPPORTED_OPERATION_VARIANT:lookup_id_requires_exact_mapping"
    if name == "unit_conversion":
        try:
            factor = float(args["factor"])
        except (KeyError, TypeError, ValueError):
            return "UNSUPPORTED_OPERATION_VARIANT:unit_conversion_requires_factor"
        if factor <= 0 or not str(args.get("to_unit", "")).strip():
            return "UNSUPPORTED_OPERATION_VARIANT:unit_conversion_arguments"
        if not str(args.get("unit_column", "")).strip():
            return "UNSUPPORTED_OPERATION_VARIANT:unit_conversion_requires_unit_column"
    if name == "unpivot" and not isinstance(args.get("date_columns"), list):
        return "UNSUPPORTED_OPERATION_VARIANT:unpivot"
    return None


def validate_operation_plan(proposal: MappingProposal) -> list[str]:
    return [
        problem
        for field in proposal.field_mappings
        for operation in field.operations
        if (problem := operation_problem(operation)) is not None
    ]


def _apply_series(series: pd.Series, operation: TransformOperation) -> pd.Series:
    args = operation.arguments
    if operation.operation == "trim":
        return series.map(lambda value: value.strip() if isinstance(value, str) else value)
    if operation.operation == "parse_number":
        return series.map(parse_number)
    if operation.operation == "parse_boolean":
        return series.map(parse_boolean)
    if operation.operation == "parse_date":
        locale = args.get("locale")
        return series.map(lambda value: parse_date(value, locale))
    if operation.operation == "cast":
        dtype = args["dtype"]
        if dtype == "string":
            return series.astype("string")
        if dtype == "float":
            return pd.to_numeric(series, errors="raise").astype(float)
        if dtype == "int":
            return pd.to_numeric(series, errors="raise").astype("Int64")
        return series.map(parse_boolean).astype("boolean")
    if operation.operation == "lookup_id":
        mapping = {str(key): value for key, value in args["mapping"].items()}
        missing = sorted({str(value) for value in series.dropna()} - set(mapping))
        if missing:
            raise OperationPlanError(f"LOOKUP_ID_UNRESOLVED:{missing[:5]}")
        return series.map(lambda value: mapping.get(str(value)) if pd.notna(value) else value)
    if operation.operation == "unit_conversion":
        return pd.to_numeric(series, errors="raise") * float(args["factor"])
    return series


def execute_operations(region: RegionData, proposal: MappingProposal) -> RegionData:
    problems = validate_operation_plan(proposal)
    if problems:
        raise OperationPlanError(";".join(problems))
    frame = region.frame.copy()
    for field in proposal.field_mappings:
        if field.source_column not in frame.columns:
            continue
        for operation in field.operations:
            if operation.operation in {"rename", "unpivot"}:
                continue
            if operation.operation == "filter_total_rows":
                markers = {
                    str(item).strip().casefold()
                    for item in operation.arguments.get("markers", ["total", "subtotal", "grand total", "tổng"])
                }
                frame = frame.loc[
                    ~frame[field.source_column].astype("string").str.strip().str.casefold().isin(markers)
                ].copy()
                continue
            frame[field.source_column] = _apply_series(frame[field.source_column], operation)
            if operation.operation == "unit_conversion":
                unit_column = str(operation.arguments["unit_column"])
                if unit_column not in frame.columns:
                    raise OperationPlanError(f"UNIT_CONVERSION_UNIT_COLUMN_MISSING:{unit_column}")
                frame[unit_column] = str(operation.arguments["to_unit"])
    # Filtering preserves original DataFrame indices.  Convert those indices
    # back to exact source rows instead of using positional modulo arithmetic.
    source_by_index = dict(zip(region.frame.index, region.source_rows, strict=True))
    source_rows = [source_by_index[index] for index in frame.index]
    return replace(region, frame=frame.reset_index(drop=True), source_rows=source_rows)
