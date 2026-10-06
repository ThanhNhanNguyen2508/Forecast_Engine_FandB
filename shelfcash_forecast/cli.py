from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from shelfcash_forecast import (
    ForecastConfig,
    ForecastPlanConfig,
    predict_demand,
    run_forecast_plan,
    train_forecast_core,
)
from shelfcash_forecast.data.adapter import adapt_forecast_input
from shelfcash_forecast.data.validator import validate_calendar, validate_sales

app = typer.Typer(help="ShelfCash Forecast Core command-line interface.")


def _load_table(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise typer.BadParameter(f"Không hỗ trợ định dạng: {suffix}")


def _canonical_data(sales: Path, calendar: Path | None) -> dict[str, pd.DataFrame]:
    sales_frame = _load_table(sales)
    assert sales_frame is not None
    payload = {"sales_history": sales_frame}
    calendar_frame = _load_table(calendar)
    if calendar_frame is not None:
        payload["calendar_features"] = calendar_frame
    return payload


@app.command()
def validate(
    sales: Path = typer.Option(..., exists=True, readable=True),
    calendar: Optional[Path] = typer.Option(None, exists=True, readable=True),
) -> None:
    """Validate canonical input without training."""

    config = ForecastConfig()
    adapted = adapt_forecast_input(_canonical_data(sales, calendar), config)
    _, report = validate_sales(adapted.sales_history)
    validate_calendar(adapted.calendar_features, report)
    typer.echo(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))


@app.command()
def train(
    sales: Path = typer.Option(..., exists=True, readable=True),
    artifact_dir: Path = typer.Option(...),
    calendar: Optional[Path] = typer.Option(None, exists=True, readable=True),
    model_version: str = typer.Option("forecast-core-v0.1.0"),
    calibration_days: int = typer.Option(28, min=7),
    test_days: int = typer.Option(28, min=7),
) -> None:
    """Train, calibrate, evaluate and save artifacts."""

    config = ForecastConfig(
        calibration_days=calibration_days,
        test_days=test_days,
    )
    result = train_forecast_core(
        canonical_data=_canonical_data(sales, calendar),
        artifact_directory=artifact_dir,
        config=config,
        model_version=model_version,
    )
    typer.echo(result.model_dump_json(indent=2))


@app.command()
def predict(
    sales: Path = typer.Option(..., exists=True, readable=True),
    artifact_dir: Path = typer.Option(..., exists=True, file_okay=False),
    cutoff_date: str = typer.Option(...),
    horizon: int = typer.Option(7, min=1, max=7),
    calendar: Optional[Path] = typer.Option(None, exists=True, readable=True),
    output: Optional[Path] = typer.Option(None),
    execution_mode: str = typer.Option("production"),
) -> None:
    """Load artifacts and forecast future product demand."""

    package = predict_demand(
        canonical_data=_canonical_data(sales, calendar),
        artifact_directory=artifact_dir,
        cutoff_date=cutoff_date,
        forecast_horizon=horizon,
        execution_mode=execution_mode,
    )
    text = package.model_dump_json(indent=2)
    if output is None:
        typer.echo(text)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        typer.echo(str(output))


@app.command("run-plan")
def run_plan(
    bundle: Path = typer.Option(..., exists=True, file_okay=False),
    output: Path = typer.Option(...),
    cutoff_date: str = typer.Option(...),
    artifact_dir: Path = typer.Option(..., exists=True, file_okay=False),
    horizon: int = typer.Option(7, min=1, max=7),
    seed: int = typer.Option(42),
    scenario_count: int = typer.Option(100, min=1, max=2000),
    execution_mode: str = typer.Option("production"),
    optimization_mode: str = typer.Option("compare"),
    planning_config: Optional[Path] = typer.Option(None, exists=True, dir_okay=False),
    explanation_mode: str = typer.Option("deterministic"),
    env_config: Optional[Path] = typer.Option(None, exists=True, dir_okay=False),
) -> None:
    """Run the public bundle-to-M1-M6 application service."""

    config = ForecastPlanConfig(
        bundle_directory=bundle,
        output_directory=output,
        cutoff_date=cutoff_date,
        horizon=horizon,
        seed=seed,
        scenario_count=scenario_count,
        execution_mode=execution_mode,
        train_model=False,
        artifact_directory=artifact_dir,
        run_downstream=True,
        optimization_mode=optimization_mode,
        planning_config=planning_config,
        explanation_mode=explanation_mode,
        env_config=env_config,
    )
    result = run_forecast_plan(config)
    typer.echo(result.model_dump_json(indent=2))


if __name__ == "__main__":
    app()
