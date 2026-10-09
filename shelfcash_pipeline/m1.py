"""Milestone 2: canonical bundle -> fixed model -> repaired M1 quantiles."""
from __future__ import annotations

from pathlib import Path

from shelfcash_forecast.pipeline import inference_pipeline
from shelfcash_preprocess.engine import create_forecast_input_frames
from shelfcash_pipeline.context import RunOptions, save_checkpoint, write_json


def run(options: RunOptions, output: Path, bundle: Path) -> inference_pipeline.ForecastState:
    frames = create_forecast_input_frames(
        bundle, include_weather=False, as_of_date=options.cutoff_date,
    )
    policy=None
    if options.forecast_overrides:
        from shelfcash_forecast.pipeline.forecast_overrides import ForecastOverridePolicy
        policy=ForecastOverridePolicy.model_validate_json(options.forecast_overrides.read_text(encoding='utf-8-sig'))
        from shelfcash_preprocess.engine import load_canonical_frames
        menu=load_canonical_frames(bundle).get('menu')
        if menu is None:raise ValueError('COLD_START_MENU_REQUIRED')
        registry={str(r['product_id']):str(r['unit']) for r in menu.to_dict('records')}
        for p in policy.predictions:
            if p.product_id not in registry or p.unit!=registry[p.product_id]:raise ValueError('COLD_START_MENU_ID_OR_UNIT_MISMATCH:'+p.product_id)
    state = inference_pipeline.predict_m1(
        frames, options.artifacts_path, options.cutoff_date, options.horizon,
        execution_mode=options.execution_mode,
        override_policy=policy,
    )
    # Save the adapter's ACTUAL inputs (including units filled from menu).
    destination = output / "engine_inputs"
    destination.mkdir(exist_ok=False)
    for name, frame in frames.items():
        frame.to_csv(destination / f"{name}.csv", index=False, encoding="utf-8-sig")
    save_checkpoint(output, "m1", state.frame)
    write_json(output / "m1" / "model_context.json", {
        "artifacts_path": str(options.artifacts_path.resolve()),
        "metadata": state.artifacts.metadata,
        "artifact_warnings": list(state.artifacts.warnings),
        "execution_mode": state.execution_mode,
        "data_quality": state.quality_report.to_dict(),
    })
    return state
