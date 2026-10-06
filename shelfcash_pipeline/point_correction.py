"""Optional stopping milestone: apply the loaded point corrector, without fitting."""
from __future__ import annotations

from pathlib import Path

from shelfcash_forecast.pipeline import inference_pipeline
from shelfcash_pipeline.context import save_checkpoint


def run(output: Path, state: inference_pipeline.ForecastState) -> inference_pipeline.ForecastState:
    corrected = inference_pipeline.correct_forecast_point(state)
    save_checkpoint(output, "point", corrected.frame)
    return corrected
