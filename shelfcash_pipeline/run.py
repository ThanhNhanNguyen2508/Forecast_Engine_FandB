"""Read this first: the complete preprocess -> M1 -> point -> M2 -> M3..M6 graph."""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, replace
from datetime import date, datetime, timezone
from pathlib import Path

from shelfcash_forecast.pipeline.inference_pipeline import build_forecast_package
from shelfcash_pipeline import m1, m2, m3, m4, m5, m6, point_correction, preprocess
from shelfcash_pipeline.context import (
    MILESTONES, RunOptions,
    default_output_path, release_output, reserve_output, write_json,
)


def run_pipeline(options: RunOptions) -> Path:
    if options.cutoff_date is None and options.bundle_path is not None:
        from shelfcash_preprocess.pipeline import load_bundle
        declared=load_bundle(options.bundle_path).manifest.context.cutoff_date
        if declared is not None:
            options=replace(options,cutoff_date=declared)
    output = reserve_output(options)
    manifest = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "options": {
            key: str(value) if isinstance(value, (Path, date)) else value
            for key, value in asdict(options).items()
        },
        "completed_milestones": [],
        "training": False,
        "llm_mode": "offline",
        "explanation_mode": "deterministic",
        "output_directory": str(output),
        "output_policy": "replace_owned_milestone_folder",
        "business_ready": False,
    }

    def complete(stage: str) -> None:
        manifest["completed_milestones"].append(stage)
        write_json(output / "run_manifest.json", manifest)
        print(f"COMPLETED={stage}", flush=True)

    print(f"WRITES / OVERWRITES MANAGED OUTPUT: {output}", flush=True)
    try:
        write_json(output / "run_manifest.json", manifest)
        for stage in MILESTONES[:MILESTONES.index(options.stop_after) + 1]:
            manifest["active_milestone"] = stage
            write_json(output / "run_manifest.json", manifest)
            if stage == "preprocess":
                bundle = preprocess.run(options, output)
                manifest["bundle_path"] = str(bundle.resolve())
            elif stage == "m1":
                state = m1.run(options, output, bundle)
            elif stage == "point":
                state = point_correction.run(output, state)
            elif stage == "m2":
                state = m2.run(output, state)
                forecast = build_forecast_package(state)
            elif stage == "m3":
                ingredient = m3.run(output, bundle, forecast, state.artifacts.config)
            elif stage == "m4":
                inventory = m4.run(options, output, bundle, forecast, ingredient)
            elif stage == "m5":
                optimization = m5.run(options, output, bundle, inventory)
            elif stage == "m6":
                m6.run(output, forecast, ingredient, optimization)
            complete(stage)

        manifest["status"] = "completed"
        manifest["stopped_after"] = options.stop_after
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = {"type": type(exc).__name__, "message": str(exc)}
        if hasattr(exc, "code"):
            manifest["error"]["code"] = exc.code
        if hasattr(exc, "details"):
            manifest["error"]["details"] = exc.details
        from shelfcash_forecast.exceptions import ForecastCoreError
        from pydantic import ValidationError
        code=getattr(exc,'code',str(exc).split(':',1)[0] if isinstance(exc,ForecastCoreError) else type(exc).__name__)
        status='INVALID_INPUT' if isinstance(exc,ValidationError) else 'BLOCKED_INPUT_SEMANTICS' if isinstance(exc,ForecastCoreError) else 'SOLVER_ERROR'
        if str(code).startswith(('FIXED_MODEL_HORIZON_UNSUPPORTED','SUPPORTED_HORIZON_REQUIRED')):status='UNSUPPORTED_INPUT'
        manifest['technical_outcome']=status
        manifest['error'].update(reason_code=code,field_paths=[manifest.get('active_milestone','pipeline')],
            expected_meaning='Provide the declared stage prerequisites; see error details and supported input contract',
            completed_milestones=list(manifest['completed_milestones']),accepted_orders=[],accepted_zero_purchase=False)
        failed_stage = manifest.get("active_milestone")
        if failed_stage is not None:
            destination = output / failed_stage
            destination.mkdir(exist_ok=True)
            write_json(destination / "failure.json", manifest["error"])
        raise
    finally:
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        try:
            write_json(output / "run_manifest.json", manifest)
        finally:
            release_output(output)
    return output


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline preprocess -> fixed-artifact M1 -> point -> M2 -> M3 -> M4 -> M5 -> M6. Owned checkpoint folders are overwritten. No training or live API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="Explicit raw input path")
    source.add_argument("--bundle", type=Path, help="Existing sealed bundle; read-only, skips raw preprocessing")
    parser.add_argument("--stop-after", choices=MILESTONES, default="m2", help="Last stage to execute")
    parser.add_argument("--artifacts", type=Path, help="Explicit fixed artifacts to LOAD; required beyond preprocessing")
    parser.add_argument("--output-dir", type=Path, help="Default: codex_tests/runs/pipeline_until_<stop-after>; only runner-owned folders can be overwritten")
    parser.add_argument("--cutoff-date", type=date.fromisoformat, required=True, help="Inclusive EOD origin, YYYY-MM-DD")
    parser.add_argument("--horizon", type=int, default=7, help="Forecast horizon; bounded by artifact config")
    parser.add_argument('--forecast-overrides',type=Path,help='Explicit cold-start forecast and scenario policy; required outside fixed trained range')
    parser.add_argument("--execution-mode", choices=("demo", "backtest_replay", "production"), default="demo", help="Existing engine promotion checks are retained")
    parser.add_argument("--store-id", help="Explicit single-store mapping if source rows omit store identity")
    parser.add_argument("--date-locale", choices=("DMY", "MDY", "YMD"), help="Explicit locale mapping; ambiguity requires review")
    parser.add_argument("--context-metadata", type=Path, help="Explicit JSON metadata for RAW preprocessing (expiry policy/classification); never inferred or auto-approved")
    parser.add_argument("--planning-config", type=Path, help="Explicit existing application planning JSON; required for M5/M6")
    parser.add_argument("--scenario-count", type=int, default=100, help="M4 bootstrap scenario count in 1..2000; no yield-loss fitting")
    parser.add_argument("--seed", type=int, default=42, help="Scenario/optimization seed")
    parser.add_argument("--optimization-mode", choices=("deterministic", "stochastic", "compare"), default="compare", help="M5 solver mode; compare uses both modes on identical scenarios")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    options = RunOptions(
        output_dir=args.output_dir or default_output_path(args.stop_after),
        input_path=None if args.bundle is not None else args.input,
        bundle_path=args.bundle,
        artifacts_path=args.artifacts,
        stop_after=args.stop_after,
        cutoff_date=args.cutoff_date,
        horizon=args.horizon,
        execution_mode=args.execution_mode,
        store_id=args.store_id,
        date_locale=args.date_locale,
        context_metadata=args.context_metadata,
        planning_config=args.planning_config,
        scenario_count=args.scenario_count,
        seed=args.seed,
        optimization_mode=args.optimization_mode,
        forecast_overrides=args.forecast_overrides,
    )
    try:
        output = run_pipeline(options)
    except Exception as exc:
        print(f"PIPELINE FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"OUTPUT_DIR={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
