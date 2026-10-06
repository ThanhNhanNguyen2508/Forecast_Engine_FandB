from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _file_record(path: Path, package_root: Path) -> dict[str, Any]:
    content = path.read_bytes()
    return {
        "path": path.relative_to(package_root).as_posix(),
        "sha256": hashlib.sha256(content).hexdigest(),
        "size": len(content),
    }


def current_source_records(repository_root: str | Path) -> dict[str, list[dict[str, Any]]]:
    root = Path(repository_root).resolve()
    package = root / "shelfcash_forecast"
    all_python = sorted(package.rglob("*.py"))
    m1_m5 = [
        _file_record(path, package)
        for path in all_python
        if "decision_intelligence" not in path.relative_to(package).parts
    ]
    m6_core = [
        _file_record(path, package)
        for path in sorted((package / "decision_intelligence").glob("*.py"))
    ]
    evaluation = [
        _file_record(path, package)
        for path in sorted((package / "decision_intelligence" / "evaluation").rglob("*.py"))
    ]
    return {"m1_m5": m1_m5, "m6_part1_1": m6_core, "evaluation": evaluation}


def _changes(
    before: list[Mapping[str, Any]],
    after: list[Mapping[str, Any]],
) -> list[dict[str, str | None]]:
    left = {str(item["path"]): str(item["sha256"]) for item in before}
    right = {str(item["path"]): str(item["sha256"]) for item in after}
    return [
        {
            "path": path,
            "preflight_sha256": left.get(path),
            "postflight_sha256": right.get(path),
        }
        for path in sorted(set(left) | set(right))
        if left.get(path) != right.get(path)
    ]


def build_source_manifest(
    repository_root: str | Path,
    *,
    preflight: Mapping[str, list[Mapping[str, Any]]],
) -> dict[str, Any]:
    current = current_source_records(repository_root)
    m1_changes = _changes(preflight["m1_m5"], current["m1_m5"])
    m6_changes = _changes(preflight["m6_part1_1"], current["m6_part1_1"])
    return {
        "schema_version": "shelfcash-source-manifest-v1",
        "path_basis": "relative_to_shelfcash_forecast_package",
        "m1_m5": {
            "preflight": list(preflight["m1_m5"]),
            "postflight": current["m1_m5"],
            "changed_files": m1_changes,
            "unchanged": not m1_changes,
        },
        "m6_part1_1": {
            "preflight": list(preflight["m6_part1_1"]),
            "postflight": current["m6_part1_1"],
            "changed_files": m6_changes,
        },
        "evaluation_postflight": current["evaluation"],
    }


def source_manifest_json(manifest: Mapping[str, Any]) -> str:
    return json.dumps(
        manifest,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    )


def write_source_manifest(
    manifest: Mapping[str, Any],
    *,
    json_path: str | Path,
) -> Path:
    target = Path(json_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source_manifest_json(manifest), encoding="utf-8")
    return target
