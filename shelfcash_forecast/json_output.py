"""Write runtime JSON without materializing the entire model graph and text.

Containers are written incrementally. Each Pydantic model uses its native UTF-8
serializer, preserving aliases, serializers and JSON-mode types. Peak temporary
memory is bounded by the largest compact model, not the whole compare envelope.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, BinaryIO

from pydantic import BaseModel


def _finite_model_values(value: Any, active: set[int] | None = None) -> None:
    """Keep the existing allow_nan=False guard for nested Any/model fields."""
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON output cannot contain NaN or infinity")
        return
    if not isinstance(value, (BaseModel, dict, list, tuple, set, frozenset)):
        return
    active = set() if active is None else active
    identity = id(value)
    if identity in active:
        raise ValueError("Circular reference in JSON output")
    active.add(identity)
    try:
        if isinstance(value, BaseModel):
            children = value.__dict__.values()
        elif isinstance(value, dict):
            children = value.values()
        else:
            children = value
        for child in children:
            _finite_model_values(child, active)
        if isinstance(value, BaseModel) and value.__pydantic_extra__:
            for child in value.__pydantic_extra__.values():
                _finite_model_values(child, active)
    finally:
        active.remove(identity)


def _write(handle: BinaryIO, value: Any, fallback_str: bool) -> None:
    if isinstance(value, BaseModel):
        # Avoid model_dump -> recursively duplicated dicts -> giant Unicode str.
        # Native serialization retains the model's JSON contract exactly.
        handle.write(value.__pydantic_serializer__.to_json(value))
    elif isinstance(value, dict):
        handle.write(b"{")
        for index, (key, item) in enumerate(value.items()):
            if index:
                handle.write(b",")
            # JSON encoder key coercion must match the former json.dumps writer.
            if isinstance(key, str):
                name = key
            elif key is True:
                name = "true"
            elif key is False:
                name = "false"
            elif key is None:
                name = "null"
            elif isinstance(key, (int, float)):
                name = json.dumps(key, allow_nan=False)
            else:
                raise TypeError("JSON object keys must be strings or scalar keys")
            handle.write(json.dumps(name, ensure_ascii=False).encode("utf-8"))
            handle.write(b":")
            _write(handle, item, fallback_str)
        handle.write(b"}")
    elif isinstance(value, (list, tuple)):
        handle.write(b"[")
        for index, item in enumerate(value):
            if index:
                handle.write(b",")
            _write(handle, item, fallback_str)
        handle.write(b"]")
    else:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False,
                                default=str if fallback_str else None).encode("utf-8"))


def write_json(path: str | Path, value: Any, *, fallback_str: bool = False) -> None:
    """UTF-8 JSON; no full-envelope model_dump or json.dumps allocation.

Outputs are compact machine-readable JSON; customer Markdown/XLSX presentation
is unchanged. Runtime callers provide their existing owned/path-guarded targets.
"""
    _finite_model_values(value)
    with Path(path).open("wb", buffering=1024 * 1024) as handle:
        _write(handle, value, fallback_str)
        handle.write(b"\n")
