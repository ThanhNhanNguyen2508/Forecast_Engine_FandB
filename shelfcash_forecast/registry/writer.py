from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any
from uuid import uuid4

import lightgbm as lgb
import numpy as np
import pandas as pd

from shelfcash_forecast.calibration.cqr import CQRCalibrator
from shelfcash_forecast.calibration.point import PointCorrector
from shelfcash_forecast.config import ForecastConfig
from shelfcash_forecast.features.specification import (
    CATEGORICAL_MODEL_COLUMNS,
    MODEL_FEATURES,
    CategoryEncoder,
)
from shelfcash_forecast.models.quantile_models import QuantileModelBundle
from shelfcash_forecast.registry.governance import (
    ARTIFACT_SCHEMA_VERSION,
    sha256_file,
)

REQUIRED_ARTIFACTS = {
    "model_q25.txt",
    "model_q50.txt",
    "model_q75.txt",
    "calibrator.json",
    "point_corrector.json",
    "category_mappings.json",
    "feature_schema.json",
    "preprocessing_config.json",
    "model_metadata.json",
    "training_manifest.json",
}


def _json_default(value: object) -> object:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot JSON serialize {type(value)!r}")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _save_model(model: object, path: Path) -> None:
    if isinstance(model, lgb.Booster):
        model.save_model(str(path))
        return
    booster = getattr(model, "booster_", None)
    if booster is None:
        raise TypeError(f"Model {type(model)!r} does not expose booster_ for saving.")
    booster.save_model(str(path))


def _write_staged_artifacts(
    artifact_dir: Path,
    model_bundle: QuantileModelBundle,
    calibrator: CQRCalibrator,
    encoder: CategoryEncoder,
    config: ForecastConfig,
    metadata: dict[str, Any],
    quality_report: dict[str, Any],
    baseline_metrics: dict[str, Any],
    walk_forward_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    calibration_metrics_payload: dict[str, Any],
    training_manifest: dict[str, Any],
    predictions: dict[str, pd.DataFrame] | None,
    point_corrector: PointCorrector | None,
) -> None:
    artifact_dir.mkdir(parents=False, exist_ok=False)
    _save_model(model_bundle.models[0.25], artifact_dir / "model_q25.txt")
    _save_model(model_bundle.models[0.50], artifact_dir / "model_q50.txt")
    _save_model(model_bundle.models[0.75], artifact_dir / "model_q75.txt")
    write_json(artifact_dir / "calibrator.json", calibrator.to_dict())
    write_json(
        artifact_dir / "point_corrector.json",
        (point_corrector or PointCorrector.identity()).to_dict(),
    )
    write_json(artifact_dir / "category_mappings.json", encoder.to_dict())
    write_json(
        artifact_dir / "feature_schema.json",
        {
            "features": list(MODEL_FEATURES),
            "categorical_features": list(CATEGORICAL_MODEL_COLUMNS),
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "target_strategy": model_bundle.target_strategy,
            "residual_blend_lambda": model_bundle.residual_blend_lambda,
            "best_iterations": {
                str(quantile): iteration
                for quantile, iteration in (model_bundle.best_iterations or {}).items()
            },
            "objectives": {
                str(quantile): objective
                for quantile, objective in (model_bundle.objectives or {}).items()
            },
        },
    )
    write_json(artifact_dir / "preprocessing_config.json", config.to_dict())
    write_json(artifact_dir / "model_metadata.json", metadata)
    write_json(artifact_dir / "quality_report.json", quality_report)
    write_json(artifact_dir / "baseline_metrics.json", baseline_metrics)
    write_json(artifact_dir / "walk_forward_metrics.json", walk_forward_metrics)
    write_json(artifact_dir / "test_metrics.json", test_metrics)
    write_json(artifact_dir / "calibration_metrics.json", calibration_metrics_payload)
    write_json(artifact_dir / "training_manifest.json", training_manifest)

    if predictions:
        for name, frame in predictions.items():
            if frame.empty:
                continue
            try:
                frame.to_parquet(artifact_dir / f"{name}.parquet", index=False)
            except ImportError:
                frame.to_csv(artifact_dir / f"{name}.csv", index=False)

    present = {path.name for path in artifact_dir.iterdir() if path.is_file()}
    missing = REQUIRED_ARTIFACTS - present
    if missing:
        raise RuntimeError(f"Incomplete staged artifact set: {sorted(missing)}")
    checksums = {
        path.relative_to(artifact_dir).as_posix(): sha256_file(path)
        for path in sorted(artifact_dir.rglob("*"))
        if path.is_file() and path.name != "artifact_checksums.json"
    }
    write_json(
        artifact_dir / "artifact_checksums.json",
        {"algorithm": "sha256", "files": checksums},
    )


def _publish_staged_directory(staging: Path, target: Path) -> None:
    backup = target.parent / f".{target.name}.backup-{uuid4().hex}"
    moved_existing = False
    try:
        if target.exists():
            target.replace(backup)
            moved_existing = True
        try:
            staging.replace(target)
        except PermissionError:
            # Some Windows filesystems/endpoint scanners temporarily deny a
            # directory rename even though all staged files are readable.  A
            # target-absent copy preserves the complete-set check and the
            # checksums written above; existing targets still use backup/restore.
            if target.exists():
                raise
            shutil.copytree(staging, target)
    except Exception:
        if moved_existing and backup.exists() and not target.exists():
            backup.replace(target)
        raise
    else:
        if backup.exists():
            shutil.rmtree(backup)


def write_artifacts(
    artifact_directory: str | Path,
    model_bundle: QuantileModelBundle,
    calibrator: CQRCalibrator,
    encoder: CategoryEncoder,
    config: ForecastConfig,
    model_version: str,
    metadata: dict[str, Any],
    quality_report: dict[str, Any],
    baseline_metrics: dict[str, Any],
    walk_forward_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    calibration_metrics_payload: dict[str, Any],
    training_manifest: dict[str, Any],
    predictions: dict[str, pd.DataFrame] | None = None,
    point_corrector: PointCorrector | None = None,
) -> Path:
    del model_version  # retained in the public signature for backward compatibility
    artifact_dir = Path(artifact_directory).resolve()
    artifact_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = artifact_dir.parent / f".{artifact_dir.name}.staging-{uuid4().hex}"
    try:
        _write_staged_artifacts(
            staging,
            model_bundle,
            calibrator,
            encoder,
            config,
            metadata,
            quality_report,
            baseline_metrics,
            walk_forward_metrics,
            test_metrics,
            calibration_metrics_payload,
            training_manifest,
            predictions,
            point_corrector,
        )
        _publish_staged_directory(staging, artifact_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return artifact_dir
