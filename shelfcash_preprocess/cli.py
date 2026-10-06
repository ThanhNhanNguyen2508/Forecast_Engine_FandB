from __future__ import annotations

import importlib.util
import json
import tempfile
from datetime import date
from pathlib import Path
from typing import Annotated

import typer

from shelfcash_preprocess.config import load_settings, project_root
from shelfcash_preprocess.engine import engine_smoke as run_engine_smoke
from shelfcash_preprocess.engine import validate_bundle
from shelfcash_preprocess.llm import LLMError, live_probe
from shelfcash_preprocess.models import Readiness, RunContext
from shelfcash_preprocess.pipeline import PreprocessService


app = typer.Typer(no_args_is_help=True, help="Auditable multi-format preprocessing for ShelfCash M1-M6 inputs.")

EXIT_SUCCESS = 0
EXIT_NEEDS_REVIEW = 2
EXIT_INVALID_INPUT = 3
EXIT_DEPENDENCY_OR_CONFIG = 4
EXIT_LIVE_API = 5


def _emit(payload: dict) -> None:
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def _service(config: Path | None) -> PreprocessService:
    return PreprocessService.from_config(config)


@app.command()
def doctor(
    live: Annotated[bool, typer.Option("--live", help="Send one small real Structured Outputs request.")] = False,
    config: Annotated[Path | None, typer.Option("--config", help="Override engine-root .env.preprocess.")] = None,
) -> None:
    """Check paths, dependencies, configuration, imports, and optionally real API access."""
    settings = load_settings(config)
    dependencies = {name: importlib.util.find_spec(name) is not None for name in ("pandas", "pydantic", "openpyxl", "openai")}
    optional = {name: importlib.util.find_spec(name) is not None for name in ("pdfplumber", "pyarrow", "xlrd", "pytesseract", "lightgbm")}
    try:
        probe_dir = Path(tempfile.mkdtemp(prefix="shelfcash-doctor-", dir=project_root()))
        writable = probe_dir.exists()
        probe_dir.rmdir()
    except OSError:
        writable = False
    report = {"status": "OK", "project_root": str(project_root()), "config_path": str(settings.config_path),
              "api_key": settings.key_status, "model": settings.model, "reasoning_effort": settings.reasoning_effort,
              "dependencies": dependencies, "optional_dependencies": optional, "output_writable": writable,
              "live_called": False}
    if not all(dependencies.values()) or not writable:
        report["status"] = "DEPENDENCY_OR_PATH_ERROR"; _emit(report); raise typer.Exit(EXIT_DEPENDENCY_OR_CONFIG)
    if live:
        report["live_called"] = True
        if settings.key_status != "SET":
            report["status"] = "LIVE_NOT_VERIFIED_NO_KEY"; _emit(report); raise typer.Exit(EXIT_DEPENDENCY_OR_CONFIG)
        try:
            report["live_probe"] = live_probe(settings)
        except LLMError as exc:
            report["status"] = getattr(exc, "code", "LIVE_API_FAILED")
            report["error"] = str(exc)  # Exception is deliberately sanitized by the LLM layer.
            _emit(report); raise typer.Exit(EXIT_LIVE_API)
    _emit(report)


@app.command()
def inspect(
    input: Annotated[Path, typer.Option("--input", exists=True, readable=True, help="File, directory, or ZIP.")],
    output: Annotated[Path, typer.Option("--output", help="Directory for inspection.json and raw archive workspace.")],
    config: Annotated[Path | None, typer.Option("--config")] = None,
) -> None:
    """Inventory, read, discover and profile input without calling any API."""
    try:
        report = _service(config).inspect(input, output)
        _emit({"status": "INSPECTED", "report": str(report)})
    except Exception as exc:
        _emit({"status": "INVALID_INPUT", "error": f"{type(exc).__name__}: {exc}"})
        raise typer.Exit(EXIT_INVALID_INPUT)


@app.command()
def run(
    input: Annotated[Path, typer.Option("--input", exists=True, readable=True, help="File, directory, or ZIP.")],
    output: Annotated[Path, typer.Option("--output", help="Parent directory; one atomic run directory is created below it.")],
    store_id: Annotated[str | None, typer.Option("--store-id", help="Required when source tables have no store column.")] = None,
    cutoff_date: Annotated[str | None, typer.Option("--cutoff-date", help="Business cutoff YYYY-MM-DD; never inferred from wall clock.")] = None,
    date_locale: Annotated[str | None, typer.Option("--date-locale", help="DMY, MDY or YMD for ambiguous text dates.")] = None,
    llm_mode: Annotated[str, typer.Option("--llm-mode", help="live or offline. Live never falls back to mock.")] = "live",
    tenant_id: Annotated[str | None, typer.Option("--tenant-id")] = None,
    config: Annotated[Path | None, typer.Option("--config")] = None,
) -> None:
    """Run inventory, discovery, mapping, canonical bundle, and readiness."""
    if llm_mode not in {"live", "offline"}:
        _emit({"status": "INVALID_ARGUMENT", "error": "--llm-mode must be live or offline"}); raise typer.Exit(EXIT_INVALID_INPUT)
    if date_locale and date_locale not in {"DMY", "MDY", "YMD"}:
        _emit({"status": "INVALID_ARGUMENT", "error": "--date-locale must be DMY, MDY or YMD"}); raise typer.Exit(EXIT_INVALID_INPUT)
    try:
        parsed_cutoff = date.fromisoformat(cutoff_date) if cutoff_date else None
    except ValueError:
        _emit({"status": "INVALID_ARGUMENT", "error": "--cutoff-date must be YYYY-MM-DD"}); raise typer.Exit(EXIT_INVALID_INPUT)
    try:
        bundle = _service(config).run(input, output, context=RunContext(store_id=store_id, cutoff_date=parsed_cutoff,
                                                                        date_locale=date_locale, tenant_id=tenant_id), llm_mode=llm_mode)
    except LLMError as exc:
        _emit({"status": getattr(exc, "code", "LIVE_API_FAILED"), "error": str(exc)}); raise typer.Exit(EXIT_LIVE_API)
    except ImportError as exc:
        _emit({"status": "MISSING_DEPENDENCY", "error": str(exc)}); raise typer.Exit(EXIT_DEPENDENCY_OR_CONFIG)
    except Exception as exc:
        _emit({"status": "INVALID_INPUT", "error": f"{type(exc).__name__}: {exc}"}); raise typer.Exit(EXIT_INVALID_INPUT)
    states = {key: value.status.value for key, value in bundle.manifest.readiness.items()}
    _emit({"status": "BUNDLE_CREATED", "run_dir": str(bundle.run_dir), "readiness": states,
           "review_file": str(bundle.run_dir / "review_required.json"), "exit_code": EXIT_NEEDS_REVIEW if any(x == Readiness.NEEDS_REVIEW.value for x in states.values()) else 0})
    if any(x == Readiness.NEEDS_REVIEW.value for x in states.values()):
        raise typer.Exit(EXIT_NEEDS_REVIEW)


@app.command("apply-review")
def apply_review(
    run_dir: Annotated[Path, typer.Option("--run-dir", exists=True, file_okay=False)],
    review_file: Annotated[Path, typer.Option("--review-file", exists=True, dir_okay=False)],
    config: Annotated[Path | None, typer.Option("--config")] = None,
) -> None:
    """Validate hash-bound review decisions and resume into a new atomic bundle."""
    try:
        bundle = _service(config).apply_review(run_dir, review_file)
        _emit({"status": "REVIEW_APPLIED", "run_dir": str(bundle.run_dir),
               "readiness": {key: value.status.value for key, value in bundle.manifest.readiness.items()}})
    except Exception as exc:
        _emit({"status": "REVIEW_REJECTED", "error": f"{type(exc).__name__}: {exc}"}); raise typer.Exit(EXIT_INVALID_INPUT)


@app.command()
def validate(bundle: Annotated[Path, typer.Option("--bundle", exists=True, file_okay=False)]) -> None:
    """Round-trip load and validate a completed canonical bundle."""
    try:
        _emit(validate_bundle(bundle))
    except Exception as exc:
        _emit({"status": "INVALID_BUNDLE", "error": f"{type(exc).__name__}: {exc}"}); raise typer.Exit(EXIT_INVALID_INPUT)


@app.command("engine-smoke")
def engine_smoke(
    bundle: Annotated[Path, typer.Option("--bundle", exists=True, file_okay=False)],
    artifact_dir: Annotated[Path | None, typer.Option("--artifact-dir", help="Existing M1/M2 artifact directory; never fabricated.")] = None,
) -> None:
    """Invoke the forecast engine's real adapters/contracts for M1-M5 input compatibility."""
    try:
        report = run_engine_smoke(bundle, artifact_dir)
        _emit(report)
        if report["status"] == "FAIL":
            raise typer.Exit(EXIT_INVALID_INPUT)
    except typer.Exit:
        raise
    except ImportError as exc:
        _emit({"status": "MISSING_FORECAST_DEPENDENCY", "error": str(exc), "install": ".venv-preprocess\\Scripts\\python.exe -m pip install -e \".[forecast]\""})
        raise typer.Exit(EXIT_DEPENDENCY_OR_CONFIG)
    except Exception as exc:
        _emit({"status": "ENGINE_SMOKE_FAILED", "error": f"{type(exc).__name__}: {exc}"}); raise typer.Exit(EXIT_INVALID_INPUT)
