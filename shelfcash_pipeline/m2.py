"""Milestone 3: apply loaded CQR with current engine code -> public forecast."""
from __future__ import annotations

from pathlib import Path

from shelfcash_forecast.pipeline import inference_pipeline
from shelfcash_pipeline.context import save_checkpoint, write_json


def run(output: Path, state: inference_pipeline.ForecastState) -> inference_pipeline.ForecastState:
    calibrated = inference_pipeline.calibrate_forecast(state)
    package = inference_pipeline.build_forecast_package(calibrated)
    save_checkpoint(output, "m2", calibrated.frame)
    write_json(output / "m2" / "forecast.json", package.model_dump(mode="json"))
    return calibrated
