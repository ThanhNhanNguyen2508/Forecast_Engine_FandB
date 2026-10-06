"""Executable, immutable import-revision semantics.

The preprocess pipeline publishes immutable bundles.  This module defines how a
persistence layer may construct the next accepted table revision without
pretending that an ``append``/``upsert``/``replace`` label alone changed data.
It deliberately refuses overlapping summaries and ambiguous replace scopes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

import pandas as pd


ImportMode = Literal["append", "upsert", "replace"]


@dataclass(frozen=True)
class ImportRevision:
    frame: pd.DataFrame
    mode: ImportMode
    key_columns: tuple[str, ...]
    existing_rows: int
    incoming_rows: int
    output_rows: int
    inserted_rows: int
    updated_rows: int
    removed_rows: int
    idempotent_rows: int
    replace_scope: dict[str, tuple[Any, ...]] | None


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], *, name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"IMPORT_MISSING_{name.upper()}_COLUMNS:{missing}")


def _validate_unique(frame: pd.DataFrame, keys: list[str], *, name: str) -> None:
    duplicated = frame.duplicated(keys, keep=False)
    if bool(duplicated.any()):
        examples = frame.loc[duplicated, keys].head(5).to_dict(orient="records")
        raise ValueError(f"IMPORT_DUPLICATE_{name.upper()}_GRAIN:{examples}")


def _key_tuples(frame: pd.DataFrame, keys: list[str]) -> pd.Series:
    return frame[keys].astype("string").fillna("<NULL>").agg("\x1f".join, axis=1)


def _records_equal(left: pd.Series, right: pd.Series, columns: list[str]) -> bool:
    for column in columns:
        a, b = left.get(column, pd.NA), right.get(column, pd.NA)
        if pd.isna(a) and pd.isna(b):
            continue
        if a != b:
            return False
    return True


def apply_import_revision(
    existing: pd.DataFrame,
    incoming: pd.DataFrame,
    *,
    mode: ImportMode,
    key_columns: Sequence[str],
    replace_scope: Mapping[str, Any | Sequence[Any]] | None = None,
) -> ImportRevision:
    """Return a new table revision and an audit summary.

    ``append`` accepts new grains and silently skips only byte-semantically
    equivalent records at an existing key.  Conflicting overlap is rejected so
    transaction detail and daily summaries cannot be double-counted.
    ``upsert`` lets incoming records replace the same explicit keys.
    ``replace`` requires a caller-provided scope and replaces only that scope.
    """

    if mode not in {"append", "upsert", "replace"}:
        raise ValueError(f"IMPORT_MODE_UNSUPPORTED:{mode}")
    keys = list(key_columns)
    if not keys:
        raise ValueError("IMPORT_GRAIN_KEYS_REQUIRED")
    current = existing.copy(deep=True)
    new = incoming.copy(deep=True)
    _require_columns(current, keys, name="key")
    _require_columns(new, keys, name="key")
    _validate_unique(current, keys, name="existing")
    _validate_unique(new, keys, name="incoming")
    if set(current.columns) != set(new.columns):
        raise ValueError(
            "IMPORT_SCHEMA_MISMATCH: existing and incoming columns must match exactly"
        )
    columns = list(current.columns)
    new = new.loc[:, columns]
    current_keys = _key_tuples(current, keys)
    incoming_keys = _key_tuples(new, keys)
    current_index = {value: index for index, value in current_keys.items()}
    incoming_index = {value: index for index, value in incoming_keys.items()}
    overlap = sorted(set(current_index) & set(incoming_index))
    idempotent = 0
    updated = 0
    removed = 0
    inserted = 0
    normalized_scope: dict[str, tuple[Any, ...]] | None = None

    if mode == "append":
        conflicts: list[dict[str, Any]] = []
        skip_indices: set[Any] = set()
        for key in overlap:
            old_row = current.loc[current_index[key]]
            new_row = new.loc[incoming_index[key]]
            if _records_equal(old_row, new_row, columns):
                idempotent += 1
                skip_indices.add(incoming_index[key])
            else:
                conflicts.append({column: new_row[column] for column in keys})
        if conflicts:
            raise ValueError(
                "IMPORT_APPEND_OVERLAP_REQUIRES_PRECEDENCE_OR_REVIEW:"
                f"{conflicts[:5]}"
            )
        additions = new.drop(index=list(skip_indices))
        inserted = len(additions)
        output = pd.concat([current, additions], ignore_index=True)
    elif mode == "upsert":
        kept = current.loc[~current_keys.isin(set(incoming_index))]
        inserted = len(set(incoming_index) - set(current_index))
        updated = len(overlap)
        removed = len(overlap)
        output = pd.concat([kept, new], ignore_index=True)
    else:
        if not replace_scope:
            raise ValueError("IMPORT_REPLACE_SCOPE_REQUIRED")
        scope_masks: list[pd.Series] = []
        normalized_scope = {}
        for column, raw_values in replace_scope.items():
            _require_columns(current, [column], name="scope")
            _require_columns(new, [column], name="scope")
            values = (
                tuple(raw_values)
                if isinstance(raw_values, Sequence) and not isinstance(raw_values, (str, bytes))
                else (raw_values,)
            )
            if not values:
                raise ValueError(f"IMPORT_REPLACE_SCOPE_EMPTY:{column}")
            normalized_scope[column] = values
            scope_masks.append(current[column].isin(values))
            if not bool(new[column].isin(values).all()):
                raise ValueError(f"IMPORT_REPLACE_INCOMING_OUTSIDE_SCOPE:{column}")
        mask = pd.Series(True, index=current.index)
        for item in scope_masks:
            mask &= item
        removed = int(mask.sum())
        inserted = len(new)
        output = pd.concat([current.loc[~mask], new], ignore_index=True)

    _validate_unique(output, keys, name="output")
    return ImportRevision(
        frame=output.reset_index(drop=True),
        mode=mode,
        key_columns=tuple(keys),
        existing_rows=len(existing),
        incoming_rows=len(incoming),
        output_rows=len(output),
        inserted_rows=inserted,
        updated_rows=updated,
        removed_rows=removed,
        idempotent_rows=idempotent,
        replace_scope=normalized_scope,
    )
