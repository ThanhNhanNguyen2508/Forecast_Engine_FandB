from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ARTIFACT_SCHEMA_VERSION = "3.0"
SUPPORTED_ARTIFACT_SCHEMA_VERSIONS = frozenset({"2.0", ARTIFACT_SCHEMA_VERSION})
FINGERPRINT_ALGORITHM = "sha256"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_scalar(value: object) -> object:
    if pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return format(number, ".17g")
    return str(value)


def dataset_fingerprint(
    frame: pd.DataFrame,
    *,
    columns: list[str] | tuple[str, ...] | None = None,
    schema_version: str = ARTIFACT_SCHEMA_VERSION,
) -> str:
    """Hash canonical rows independent of incoming DataFrame row order."""

    selected = list(columns or sorted(frame.columns))
    missing = [column for column in selected if column not in frame]
    if missing:
        raise ValueError(f"Fingerprint columns are missing: {missing}")
    records = [
        [_normalize_scalar(value) for value in row]
        for row in frame[selected].itertuples(index=False, name=None)
    ]
    records.sort(key=lambda row: json.dumps(row, ensure_ascii=False, separators=(",", ":")))
    payload: dict[str, Any] = {
        "schema_version": schema_version,
        "columns": selected,
        "row_count": len(records),
        "rows": records,
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def runtime_versions() -> dict[str, str | None]:
    def version(distribution: str) -> str | None:
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            return None

    return {
        "python_version": platform.python_version(),
        "package_version": version("shelfcash-forecast-core"),
        "lightgbm_version": version("lightgbm"),
        "pandas_version": version("pandas"),
        "numpy_version": version("numpy"),
        "scikit_learn_version": version("scikit-learn"),
        "statsmodels_version": version("statsmodels"),
    }
