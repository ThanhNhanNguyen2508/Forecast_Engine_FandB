from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from typing import Any

from pydantic import BaseModel

EVIDENCE_SCHEMA_VERSION = "m6-evidence-v2"
HASH_PREFIX = "sha256:"


def _canonical_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _canonical_value(value.model_dump(mode="json"))
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (set, frozenset)):
        normalized = [_canonical_value(item) for item in value]
        return sorted(normalized, key=canonical_json)
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Canonical evidence cannot contain NaN or infinity.")
    return value


def canonical_json(value: Any) -> str:
    """Serialize evidence material deterministically without runtime metadata."""

    return json.dumps(
        _canonical_value(value),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_content_hash(value: Any) -> str:
    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    return f"{HASH_PREFIX}{digest}"


def evidence_content_hash(material: Mapping[str, Any]) -> str:
    return sha256_content_hash(material)


def evidence_package_hash(
    *,
    request_id: str,
    schema_version: str,
    source_layers: Sequence[str],
    ordered_items: Sequence[tuple[str, str]],
) -> str:
    return sha256_content_hash(
        {
            "request_id": request_id,
            "schema_version": schema_version,
            "source_layers": list(source_layers),
            "ordered_items": [
                {"evidence_id": evidence_id, "content_hash": content_hash}
                for evidence_id, content_hash in ordered_items
            ],
        }
    )


def is_full_sha256(value: str) -> bool:
    if not value.startswith(HASH_PREFIX):
        return False
    digest = value.removeprefix(HASH_PREFIX)
    return len(digest) == 64 and all(character in "0123456789abcdef" for character in digest)
