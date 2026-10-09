"""Run options and output helpers; computational code stays in engine packages."""
from __future__ import annotations

import json
import os
import shutil
import stat
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from shelfcash_forecast.json_output import write_json as _write_json


SOURCE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_LAYOUT = (SOURCE_ROOT / "shelfcash.config.json").is_file() and (SOURCE_ROOT / "pyproject.toml").is_file()
ENGINE_ROOT = SOURCE_ROOT if REPOSITORY_LAYOUT else SOURCE_ROOT.parent
# Deprecated example-path exports for historical readers/tests only. Neither
# RunOptions nor a generic launcher uses them as missing-input defaults.
DEFAULT_INPUT = ENGINE_ROOT / "demo/input" if REPOSITORY_LAYOUT else ENGINE_ROOT / "codex_tests/Demo data"
DEFAULT_ARTIFACTS = (ENGINE_ROOT / "demo/artifacts" if REPOSITORY_LAYOUT else
                    ENGINE_ROOT / "codex_tests/runs/m1_m2_research_20261004T054706Z/artifacts")
MILESTONES = ("preprocess", "m1", "point", "m2", "m3", "m4", "m5", "m6")
OWNER_FILE = ".shelfcash_pipeline.json"
LOCK_FILE = ".pipeline.lock"


@dataclass(frozen=True)
class RunOptions:
    output_dir: Path
    input_path: Path | None = None
    bundle_path: Path | None = None
    artifacts_path: Path | None = None
    stop_after: str = "m2"
    cutoff_date: date | None = None
    horizon: int = 7
    execution_mode: str = "demo"
    store_id: str | None = None
    date_locale: str | None = None
    context_metadata: Path | None = None
    planning_config: Path | None = None
    scenario_count: int = 100
    seed: int = 42
    optimization_mode: str = "compare"
    workspace_root: Path | None = None
    forecast_overrides: Path | None = None


def default_output_path(stop_after: str) -> Path:
    if stop_after not in MILESTONES:
        raise ValueError(f"Unknown milestone: {stop_after}")
    # Preserve the advanced CLI/API default. The one-file runner explicitly uses
    # pipeline.output_root (outputs/ in a standalone checkout).
    return ENGINE_ROOT / "codex_tests/runs" / f"pipeline_until_{stop_after}"


def _reject_links(root: Path) -> None:
    """Never traverse a symlink/junction while replacing a managed run."""
    for directory, dirs, files in os.walk(root, followlinks=False):
        for path in [Path(directory), *(Path(directory) / name for name in dirs + files)]:
            attrs = getattr(path.lstat(), "st_file_attributes", 0)
            if path.is_symlink() or attrs & stat.FILE_ATTRIBUTE_REPARSE_POINT:
                raise ValueError(f"Output contains a symlink/junction; refusing replacement: {path}")


def release_output(output: Path) -> None:
    (output / LOCK_FILE).unlink(missing_ok=True)


def reserve_output(options: RunOptions) -> Path:
    """Replace only our owned checkpoint folder, with checked absolute targets."""
    requested = options.output_dir.absolute()
    # resolve() follows root junctions: check the original path first as well.
    for ancestor in (requested, *requested.parents):
        if ancestor.is_symlink():
            raise ValueError(f"Output path traverses a symlink/junction: {ancestor}")
        if ancestor.exists():
            attrs = getattr(ancestor.lstat(), "st_file_attributes", 0)
            if attrs & stat.FILE_ATTRIBUTE_REPARSE_POINT:
                raise ValueError(f"Output path traverses a symlink/junction: {ancestor}")
    output = options.output_dir.resolve()
    if options.stop_after not in MILESTONES:
        raise ValueError(f"Unknown milestone: {options.stop_after}")
    if (options.input_path is None) == (options.bundle_path is None):
        raise ValueError("Specify exactly one raw input path or existing bundle path.")
    if options.cutoff_date is None:
        raise ValueError('PIPELINE_CUTOFF_REQUIRED: Provide cutoff_date or an existing bundle with a declared cutoff date')
    if options.horizon < 1:
        raise ValueError("horizon must be positive")
    if options.execution_mode not in {"demo", "production", "backtest_replay"}:
        raise ValueError("Unsupported execution mode")
    if not 1 <= options.scenario_count <= 2000:
        raise ValueError("scenario_count must be in 1..2000")
    if options.optimization_mode not in {"deterministic", "stochastic", "compare"}:
        raise ValueError("Unsupported optimization mode")
    if options.bundle_path is not None and options.context_metadata is not None:
        raise ValueError("Context metadata applies to raw preprocessing only; existing bundles are read-only.")
    for config in (options.context_metadata, options.planning_config,options.forecast_overrides):
        if config is not None and not config.is_file():
            raise FileNotFoundError(f"Configuration not found: {config}")
    input_path = options.bundle_path or options.input_path
    assert input_path is not None
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")
    if options.stop_after != "preprocess" and (options.artifacts_path is None or not options.artifacts_path.is_dir()):
        raise FileNotFoundError(f"Fixed artifacts not found: {options.artifacts_path}")
    workspace = options.workspace_root or ENGINE_ROOT
    source_paths = ([workspace / name for name in
                     ("shelfcash_forecast", "shelfcash_preprocess", "shelfcash_pipeline", "scripts", "tests", ".git")]
                    if (workspace / "pyproject.toml").is_file() and (workspace / "shelfcash_pipeline").is_dir()
                    else [workspace / "source_code"])
    protected = [*source_paths, input_path, *([options.artifacts_path] if options.artifacts_path is not None else [])]
    protected.extend(config for config in (options.context_metadata, options.planning_config) if config is not None)
    for path in protected:
        resolved = path.resolve()
        if output.is_relative_to(resolved) or resolved.is_relative_to(output):
            raise ValueError(f"Output must be outside source/input/artifacts: {path}")
    # A new child inside an existing run still changes that run's evidence.
    for runs in (workspace / "codex_tests/runs", workspace / "outputs"):
        if output.is_relative_to(runs.resolve()):
            relative = output.relative_to(runs.resolve())
            if len(relative.parts) > 1 and (runs / relative.parts[0]).exists():
                raise ValueError("Output must not be nested inside any existing run.")
    owner = {
        "owner": "shelfcash_pipeline",
        "schema_version": 1,
        "stop_after": options.stop_after,
        "output_dir": str(output),
    }
    existed = output.exists()
    if existed:
        _reject_links(output)
        marker = output / OWNER_FILE
        try:
            recorded_owner = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise FileExistsError(f"Output is not owned by this runner; refusing overwrite: {output}") from exc
        if recorded_owner != owner:
            raise FileExistsError(f"Output ownership/milestone does not match; refusing overwrite: {output}")
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / OWNER_FILE, owner)
    try:
        with (output / LOCK_FILE).open("x", encoding="utf-8") as handle:
            handle.write(str(os.getpid()))
    except FileExistsError as exc:
        raise RuntimeError(f"OUTPUT_IN_USE: {output}; another run holds {LOCK_FILE}") from exc
    try:
        if existed:
            for child in output.iterdir():
                if child.name in {OWNER_FILE, LOCK_FILE}:
                    continue
                target = child.resolve()
                if target.parent != output or not target.is_relative_to(output):
                    raise ValueError(f"Refusing delete outside the owned output: {target}")
                if child.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
    except Exception:
        release_output(output)
        raise
    return output


def write_json(path: Path, value: Any) -> None:
    _write_json(path,value)


def save_checkpoint(output: Path, stage: str, frame: pd.DataFrame) -> dict[str, Any]:
    """Save features and prediction columns outside the sealed canonical bundle."""
    keys = ["store_key", "product_key", "target_date", "horizon"]
    columns = ["p25", "p50", "p75"]
    values = frame[columns].to_numpy(dtype=float)
    summary: dict[str, Any] = {
        "stage": stage,
        "rows": len(frame),
        "stores": int(frame["store_key"].nunique()),
        "products": int(frame["product_key"].nunique()),
        "horizons": sorted(int(value) for value in frame["horizon"].unique()),
        "duplicate_keys": int(frame.duplicated(keys).sum()),
        "nonfinite_quantile_rows": int((~np.isfinite(values).all(axis=1)).sum()),
        "negative_quantile_rows": int((values < 0).any(axis=1).sum()),
        "quantile_crossing_rows": int(
            ((values[:, 0] > values[:, 1]) | (values[:, 1] > values[:, 2])).sum()
        ),
        "point_correction_applied": stage in {"point", "m2"},
        "cqr_applied": stage == "m2",
        "planned_closure_policy_applied": stage == "m2",
        "accuracy_and_coverage": "NOT_MEASURED: future actuals are required",
    }
    if stage == "m2":
        intervals = frame[["interval_lower", "interval_upper"]].to_numpy(dtype=float)
        summary["nonfinite_interval_rows"] = int((~np.isfinite(intervals).all(axis=1)).sum())
        summary["invalid_interval_rows"] = int(
            ((intervals[:, 0] < 0) | (intervals[:, 0] > values[:, 1])
             | (values[:, 1] > intervals[:, 1])).sum()
        )
        summary["mean_interval_width"] = float((intervals[:, 1] - intervals[:, 0]).mean())
    invalid = any(summary[key] for key in (
        "duplicate_keys", "nonfinite_quantile_rows", "negative_quantile_rows",
        "quantile_crossing_rows", "nonfinite_interval_rows", "invalid_interval_rows",
    ) if key in summary)
    if frame.empty or invalid:
        raise ValueError(f"Invalid {stage} prediction checkpoint: {summary}")
    destination = output / stage
    destination.mkdir(exist_ok=False)
    frame.to_csv(destination / "forecast_rows.csv", index=False, encoding="utf-8-sig")
    write_json(destination / "summary.json", summary)
    return summary
