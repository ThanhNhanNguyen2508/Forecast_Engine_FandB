"""Single-file configuration entry point; computation stays in run_pipeline."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
import sys
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

from shelfcash_forecast.optimization.planning_config import PlanningConfig, load_planning_config
from shelfcash_pipeline.context import ENGINE_ROOT, MILESTONES, RunOptions
from shelfcash_pipeline.run import run_pipeline


class PipelineParameters(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    input_path: str | None = "demo/input"
    bundle_path: str | None = None
    artifacts_path: str = "demo/artifacts"
    output_root: str = "outputs"
    output_prefix: str = "pipeline_until_"
    cutoff_date: date = date(2026, 8, 12)
    horizon: StrictInt = Field(default=7, ge=1, le=7)
    execution_mode: Literal["demo", "backtest_replay", "production"] = "demo"
    store_id: str = Field(default="STORE_A", min_length=1)
    date_locale: Literal["DMY", "MDY", "YMD"] = "DMY"
    scenario_count: StrictInt = Field(default=100, ge=1, le=2000)
    seed: StrictInt = 42
    optimization_mode: Literal["deterministic", "stochastic", "compare"] = "compare"

    @field_validator("cutoff_date", mode="before")
    @classmethod
    def iso_date(cls, value):
        if isinstance(value, str):
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
                raise ValueError("cutoff_date must be YYYY-MM-DD")
            return date.fromisoformat(value)
        if type(value) is not date:
            raise ValueError("cutoff_date must be an ISO date")
        return value

    @field_validator("output_prefix")
    @classmethod
    def safe_prefix(cls, value):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value):
            raise ValueError("output_prefix must be a filename prefix without separators")
        return value

    @model_validator(mode="after")
    def input_choice(self):
        if (self.input_path is None) == (self.bundle_path is None):
            raise ValueError("Specify exactly one input_path or bundle_path")
        for value in (self.input_path, self.bundle_path, self.artifacts_path, self.output_root):
            if value is not None and not value.strip():
                raise ValueError("Paths must not be empty")
        return self


class RunConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    schema_version: Literal[1] = 1
    pipeline: PipelineParameters
    context_metadata: dict = Field(default_factory=dict)
    planning: PlanningConfig

    @model_validator(mode="after")
    def consistent_context(self):
        load_planning_config(self.planning.model_dump(mode="json"),
                             scenario_count=self.pipeline.scenario_count, seed=self.pipeline.seed)
        if self.pipeline.bundle_path is not None and self.context_metadata:
            raise ValueError("Existing bundles are read-only; clear context_metadata")
        if self.pipeline.execution_mode == "production" and self.planning.planning_mode == "SCENARIO_PREVIEW":
            raise ValueError("PRODUCTION_REJECTS_SCENARIO_PREVIEW")
        return self


def load_configuration(path: Path) -> RunConfiguration:
    # JSON-mode validation preserves the existing profile/assumption validators.
    return RunConfiguration.model_validate_json(path.read_text(encoding="utf-8-sig"))


def _path(value: str | None, engine_root: Path) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else engine_root / path


def _reject_reparse(path: Path) -> None:
    for ancestor in (path.absolute(), *path.absolute().parents):
        if ancestor.is_symlink() or (ancestor.exists() and
                getattr(ancestor.lstat(), "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT):
            raise ValueError(f"Configuration path traverses a symlink/junction: {ancestor}")


def materialize_configuration(config: RunConfiguration, engine_root: Path) -> tuple[Path | None, Path]:
    """Store validated, content-addressed effective inputs; never edit source settings."""
    payload = json.dumps(config.model_dump(mode="json"), ensure_ascii=False,
                         sort_keys=True, separators=(",", ":"), allow_nan=False)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    portable = (engine_root / "pyproject.toml").is_file() and (engine_root / "shelfcash_pipeline").is_dir()
    directory = engine_root / (".runtime/configs" if portable else "codex_tests/configs/resolved_runs") / digest
    _reject_reparse(directory)
    expected = {
        ".shelfcash_config.json": json.dumps({"owner": "shelfcash_single_file_runner", "sha256": digest}),
        "planning.json": json.dumps(config.planning.model_dump(mode="json"), ensure_ascii=False,
                                    indent=2, allow_nan=False),
    }
    if config.context_metadata:
        expected["context.json"] = json.dumps(config.context_metadata, ensure_ascii=False,
                                               indent=2, allow_nan=False)
    if directory.exists():
        if {p.name for p in directory.iterdir()} != set(expected):
            raise ValueError("Resolved configuration folder is incomplete or not owned")
        for name, text in expected.items():
            target = directory / name
            _reject_reparse(target)
            if target.read_text(encoding="utf-8") != text:
                raise ValueError("Resolved configuration was modified: " + str(target))
    else:
        directory.mkdir(parents=True, exist_ok=False)
        for name, text in expected.items():
            with (directory / name).open("x", encoding="utf-8") as handle:
                handle.write(text)
    return (directory / "context.json" if config.context_metadata else None, directory / "planning.json")


def configured_options(config: RunConfiguration, stop_after: str, *, engine_root: Path = ENGINE_ROOT) -> RunOptions:
    if stop_after not in MILESTONES:
        raise ValueError("Unknown milestone: " + stop_after)
    params = config.pipeline
    output = _path(params.output_root, engine_root) / (params.output_prefix + stop_after)
    _reject_reparse(output)
    if not output.resolve().is_relative_to(engine_root.resolve()):
        raise ValueError("Configured output must stay inside the workspace")
    context, planning = materialize_configuration(config, engine_root)
    return RunOptions(
        output_dir=output, input_path=_path(params.input_path, engine_root),
        bundle_path=_path(params.bundle_path, engine_root),
        artifacts_path=_path(params.artifacts_path, engine_root), stop_after=stop_after,
        cutoff_date=params.cutoff_date, horizon=params.horizon,
        execution_mode=params.execution_mode, store_id=params.store_id, date_locale=params.date_locale,
        context_metadata=context, planning_config=planning,
        scenario_count=params.scenario_count, seed=params.seed, optimization_mode=params.optimization_mode,
        workspace_root=engine_root,
    )


def configuration_root(config_path: Path) -> Path:
    """Support a standalone checkout and the existing outer engine config explicitly."""
    for candidate in config_path.resolve().parents:
        if (candidate / "pyproject.toml").is_file() and (candidate / "shelfcash_pipeline").is_dir():
            return candidate
        if (candidate / "source_code/shelfcash_pipeline").is_dir():
            return candidate
    return ENGINE_ROOT


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ShelfCash milestones using one configuration file.")
    parser.add_argument("--config", type=Path, default=ENGINE_ROOT / "shelfcash.config.json")
    parser.add_argument("--stop-after", choices=MILESTONES, default="m2")
    parser.add_argument("--validate-only", action="store_true", help="Validate settings without writing or running stages")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    try:
        config = load_configuration(args.config)
        if args.validate_only:
            print(json.dumps({"status": "VALID", "config": str(args.config.resolve()),
                              "budget": config.planning.budget, "planning_mode": config.planning.planning_mode}))
            return 0
        options = configured_options(config, args.stop_after, engine_root=configuration_root(args.config))
        print("CONFIG_FILE=" + str(args.config.resolve()), flush=True)
        output = run_pipeline(options)
        print("OUTPUT_DIR=" + str(output))
        return 0
    except Exception as exc:
        print(f"CONFIGURED_PIPELINE_FAILED:{type(exc).__name__}:{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
