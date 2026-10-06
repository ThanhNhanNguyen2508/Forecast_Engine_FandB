from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath

import pandas as pd

from shelfcash_preprocess.models import ArtifactRecord, RunManifest, SCHEMA_VERSION


SEALED_ROOT_FILES = {
    "source_inventory.json",
    "tables.json",
    "mapping_plan.json",
    "review_required.json",
    "review_decisions.example.json",
    "quality_report.json",
    "canonical_schema.json",
    "summary.md",
}
SEALED_DIRECTORIES = {"canonical", "lineage", "quarantine", "engine_inputs", "raw"}
SUPPLEMENTAL_ROOT_FILES = {"engine_smoke_report.json", "validation_report.json"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _role(relative: str) -> str:
    head = PurePosixPath(relative).parts[0]
    return {
        "canonical": "canonical_table",
        "lineage": "lineage",
        "quarantine": "quarantine",
        "engine_inputs": "engine_input",
        "raw": "raw_snapshot",
    }.get(head, "bundle_metadata")


def sealed_files(directory: Path) -> list[Path]:
    files = [directory / name for name in sorted(SEALED_ROOT_FILES)]
    for name in sorted(SEALED_DIRECTORIES):
        child = directory / name
        if child.exists():
            files.extend(sorted(path for path in child.rglob("*") if path.is_file()))
    return files


def build_artifact_records(directory: Path) -> list[ArtifactRecord]:
    records: list[ArtifactRecord] = []
    for path in sealed_files(directory):
        if not path.is_file():
            raise ValueError(f"BUNDLE_ARTIFACT_MISSING_BEFORE_PUBLISH:{path.name}")
        relative = path.relative_to(directory).as_posix()
        row_count = None
        if relative.startswith("canonical/") and path.suffix == ".csv":
            row_count = len(pd.read_csv(path, dtype="string", keep_default_na=False))
        elif relative == "lineage/records.json":
            row_count = len(json.loads(path.read_text(encoding="utf-8")))
        records.append(ArtifactRecord(
            path=relative,
            role=_role(relative),
            size_bytes=path.stat().st_size,
            sha256=sha256_file(path),
            row_count=row_count,
            schema_version=SCHEMA_VERSION if relative.startswith("canonical/") else None,
        ))
    return records


def _safe_artifact_path(directory: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts:
        raise ValueError(f"BUNDLE_UNSAFE_ARTIFACT_PATH:{relative}")
    path = (directory / Path(*pure.parts)).resolve()
    root = directory.resolve()
    if root not in path.parents:
        raise ValueError(f"BUNDLE_UNSAFE_ARTIFACT_PATH:{relative}")
    return path


def verify_bundle_artifacts(directory: Path, manifest: RunManifest) -> None:
    if manifest.schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"BUNDLE_SCHEMA_INCOMPATIBLE:{manifest.schema_version}:expected={SCHEMA_VERSION}"
        )
    if manifest.artifact_manifest_version != "bundle-artifacts-v2":
        raise ValueError("BUNDLE_ARTIFACT_MANIFEST_INCOMPATIBLE")
    by_name: dict[str, ArtifactRecord] = {}
    for record in manifest.artifacts:
        if record.path in by_name:
            raise ValueError(f"BUNDLE_DUPLICATE_ARTIFACT_PATH:{record.path}")
        by_name[record.path] = record
        path = _safe_artifact_path(directory, record.path)
        if path.is_symlink():
            raise ValueError(f"BUNDLE_SYMLINK_FORBIDDEN:{record.path}")
        if not path.is_file():
            raise ValueError(f"BUNDLE_ARTIFACT_MISSING:{record.path}")
        if path.stat().st_size != record.size_bytes:
            raise ValueError(f"BUNDLE_ARTIFACT_SIZE_MISMATCH:{record.path}")
        if sha256_file(path) != record.sha256:
            raise ValueError(f"BUNDLE_ARTIFACT_HASH_MISMATCH:{record.path}")
    actual = {path.relative_to(directory).as_posix() for path in sealed_files(directory) if path.is_file()}
    expected = set(by_name)
    missing_manifest_entries = actual - expected
    stale_manifest_entries = expected - actual
    if missing_manifest_entries or stale_manifest_entries:
        raise ValueError(
            "BUNDLE_ARTIFACT_SET_MISMATCH:"
            f"unsealed={sorted(missing_manifest_entries)}:missing={sorted(stale_manifest_entries)}"
        )
    for child in directory.iterdir():
        if child.is_symlink():
            raise ValueError(f"BUNDLE_SYMLINK_FORBIDDEN:{child.name}")
        if child.is_file() and child.name not in SEALED_ROOT_FILES | SUPPLEMENTAL_ROOT_FILES | {"manifest.json"}:
            raise ValueError(f"BUNDLE_UNEXPECTED_CRITICAL_FILE:{child.name}")
        if child.is_dir() and child.name not in SEALED_DIRECTORIES:
            raise ValueError(f"BUNDLE_UNEXPECTED_CRITICAL_DIRECTORY:{child.name}")
