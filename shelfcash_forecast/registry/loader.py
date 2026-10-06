from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb

from shelfcash_forecast.calibration.cqr import CQRCalibrator
from shelfcash_forecast.calibration.point import PointCorrector
from shelfcash_forecast.config import ForecastConfig
from shelfcash_forecast.exceptions import ArtifactError
from shelfcash_forecast.features.specification import (
    CATEGORICAL_MODEL_COLUMNS,
    MODEL_FEATURES,
    CategoryEncoder,
)
from shelfcash_forecast.models.quantile_models import QuantileModelBundle
from shelfcash_forecast.registry.governance import (
    SUPPORTED_ARTIFACT_SCHEMA_VERSIONS,
    sha256_file,
)


@dataclass(frozen=True)
class LoadedArtifacts:
    model_bundle: QuantileModelBundle
    calibrator: CQRCalibrator
    point_corrector: PointCorrector
    encoder: CategoryEncoder
    config: ForecastConfig
    metadata: dict[str, Any]
    warnings: tuple[str, ...] = ()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ArtifactError(
            f"Missing artifact: {path.name}",
            code="ARTIFACT_MISSING",
            details={"file": path.name},
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_checksums(artifact_dir: Path) -> tuple[str, ...]:
    checksum_path = artifact_dir / "artifact_checksums.json"
    if not checksum_path.exists():
        return ("ARTIFACT_CHECKSUMS_MISSING",)
    payload = _read_json(checksum_path)
    if payload.get("algorithm") != "sha256" or not isinstance(payload.get("files"), dict):
        raise ArtifactError(
            "Invalid artifact checksum manifest.",
            code="ARTIFACT_CHECKSUM_MANIFEST_INVALID",
        )
    root = artifact_dir.resolve()
    for relative_name, expected in payload["files"].items():
        path = (artifact_dir / relative_name).resolve()
        if root not in path.parents:
            raise ArtifactError(
                "Checksum manifest references a file outside the artifact directory.",
                code="ARTIFACT_CHECKSUM_MANIFEST_INVALID",
                details={"file": relative_name},
            )
        if not path.is_file():
            raise ArtifactError(
                f"Missing checksummed artifact: {relative_name}",
                code="ARTIFACT_CHECKSUM_MISMATCH",
                details={"file": relative_name, "reason": "missing"},
            )
        actual = sha256_file(path)
        if actual != expected:
            raise ArtifactError(
                f"Checksum mismatch for artifact: {relative_name}",
                code="ARTIFACT_CHECKSUM_MISMATCH",
                details={"file": relative_name, "expected": expected, "actual": actual},
            )
    return ()


def _validate_feature_schema(feature_schema: dict[str, Any]) -> None:
    version = str(feature_schema.get("schema_version", ""))
    if version not in SUPPORTED_ARTIFACT_SCHEMA_VERSIONS:
        raise ArtifactError(
            f"Unsupported artifact schema version: {version or '<missing>'}",
            code="ARTIFACT_SCHEMA_INCOMPATIBLE",
            details={
                "artifact_schema_version": version or None,
                "supported_versions": sorted(SUPPORTED_ARTIFACT_SCHEMA_VERSIONS),
            },
        )
    features = list(feature_schema.get("features", []))
    categorical = list(feature_schema.get("categorical_features", []))
    if features != list(MODEL_FEATURES) or categorical != list(CATEGORICAL_MODEL_COLUMNS):
        raise ArtifactError(
            "Artifact feature schema is incompatible with this runtime.",
            code="ARTIFACT_FEATURE_SCHEMA_INCOMPATIBLE",
            details={
                "expected_features": list(MODEL_FEATURES),
                "artifact_features": features,
                "expected_categorical_features": list(CATEGORICAL_MODEL_COLUMNS),
                "artifact_categorical_features": categorical,
            },
        )


def load_artifacts(artifact_directory: str | Path) -> LoadedArtifacts:
    artifact_dir = Path(artifact_directory)
    if not artifact_dir.is_dir():
        raise ArtifactError(
            f"Artifact directory does not exist: {artifact_dir}",
            code="ARTIFACT_DIRECTORY_MISSING",
        )
    warnings = _verify_checksums(artifact_dir)
    feature_schema = _read_json(artifact_dir / "feature_schema.json")
    _validate_feature_schema(feature_schema)
    config = ForecastConfig.from_dict(_read_json(artifact_dir / "preprocessing_config.json"))
    metadata = _read_json(artifact_dir / "model_metadata.json")
    encoder = CategoryEncoder.from_dict(_read_json(artifact_dir / "category_mappings.json"))
    calibrator = CQRCalibrator.from_dict(_read_json(artifact_dir / "calibrator.json"))
    if calibrator.semantics_version != "cqr-order-statistic-v2":
        raise ArtifactError(
            "Artifact uses incompatible CQR order-statistic semantics.",
            code="ARTIFACT_CALIBRATOR_INCOMPATIBLE",
            details={"semantics_version": calibrator.semantics_version},
        )
    schema_version = str(feature_schema.get("schema_version"))
    point_path = artifact_dir / "point_corrector.json"
    if point_path.exists():
        try:
            point_corrector = PointCorrector.from_dict(_read_json(point_path))
        except ValueError as exc:
            raise ArtifactError(
                str(exc),
                code="ARTIFACT_POINT_CORRECTOR_INCOMPATIBLE",
            ) from exc
    elif schema_version == "2.0":
        point_corrector = PointCorrector.identity(
            model_id=str(metadata.get("model_version", "legacy-v2"))
        )
        warnings = (*warnings, "LEGACY_V2_IDENTITY_POINT_CORRECTOR")
    else:
        raise ArtifactError(
            "Missing artifact: point_corrector.json",
            code="ARTIFACT_MISSING",
            details={"file": "point_corrector.json"},
        )
    model_paths = {
        0.25: artifact_dir / "model_q25.txt",
        0.50: artifact_dir / "model_q50.txt",
        0.75: artifact_dir / "model_q75.txt",
    }
    for path in model_paths.values():
        if not path.exists():
            raise ArtifactError(
                f"Missing artifact: {path.name}",
                code="ARTIFACT_MISSING",
                details={"file": path.name},
            )
    bundle = QuantileModelBundle(
        models={
            quantile: lgb.Booster(model_file=str(path))
            for quantile, path in model_paths.items()
        },
        feature_names=list(feature_schema["features"]),
        categorical_features=list(feature_schema["categorical_features"]),
        target_strategy=str(feature_schema.get("target_strategy", "absolute")),
        residual_blend_lambda=float(feature_schema.get("residual_blend_lambda", 1.0)),
        best_iterations={
            float(quantile): int(iteration)
            for quantile, iteration in feature_schema.get("best_iterations", {}).items()
        },
        objectives={
            float(quantile): str(objective)
            for quantile, objective in feature_schema.get("objectives", {}).items()
        },
    )
    return LoadedArtifacts(
        model_bundle=bundle,
        calibrator=calibrator,
        point_corrector=point_corrector,
        encoder=encoder,
        config=config,
        metadata=metadata,
        warnings=warnings,
    )
