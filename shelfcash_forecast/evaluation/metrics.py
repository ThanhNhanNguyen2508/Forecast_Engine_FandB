from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    error = y_true - y_pred
    return float(np.mean(np.maximum(quantile * error, (quantile - 1.0) * error)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean(prediction - actual); positive values mean over-forecasting."""

    return float(np.mean(y_pred - y_true))


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    denominator = float(np.sum(np.abs(y_true)))
    if denominator == 0:
        return None
    return float(np.sum(np.abs(y_true - y_pred)) / denominator)


def quantile_crossing_rate(
    frame: pd.DataFrame,
    columns: tuple[str, str, str],
) -> float | None:
    if not set(columns).issubset(frame.columns) or frame.empty:
        return None
    values = frame[list(columns)].to_numpy(dtype=float)
    crossing = (values[:, 0] > values[:, 1]) | (values[:, 1] > values[:, 2])
    return float(np.mean(crossing))


def evaluate_quantile_predictions(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {}
    y_true = frame["target"].to_numpy(dtype=float)
    p50 = frame["p50"].to_numpy(dtype=float)
    return {
        "row_count": len(frame),
        "pinball_q25": pinball_loss(y_true, frame["p25"].to_numpy(dtype=float), 0.25),
        "pinball_q50": pinball_loss(y_true, p50, 0.50),
        "pinball_q75": pinball_loss(y_true, frame["p75"].to_numpy(dtype=float), 0.75),
        "mae_p50": mae(y_true, p50),
        "wape_p50": wape(y_true, p50),
        "bias_p50": bias(y_true, p50),
        "raw_crossing_rate": quantile_crossing_rate(
            frame, ("p25_raw", "p50_raw", "p75_raw")
        ),
        "corrected_crossing_rate": quantile_crossing_rate(frame, ("p25", "p50", "p75")),
    }


def evaluate_point_prediction(
    frame: pd.DataFrame,
    prediction_column: str,
) -> dict[str, Any]:
    if frame.empty:
        return {}
    y_true = frame["target"].to_numpy(dtype=float)
    prediction = frame[prediction_column].to_numpy(dtype=float)
    return {
        "row_count": len(frame),
        "mae": mae(y_true, prediction),
        "wape": wape(y_true, prediction),
        "bias": bias(y_true, prediction),
    }


def metric_breakdown_by_horizon(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if "horizon" not in frame:
        return {}
    return {
        str(int(horizon)): evaluate_quantile_predictions(group)
        for horizon, group in frame.groupby("horizon", observed=True)
    }


def point_metric_breakdown_by_horizon(
    frame: pd.DataFrame,
    prediction_column: str,
) -> dict[str, dict[str, Any]]:
    if "horizon" not in frame:
        return {}
    return {
        str(int(horizon)): evaluate_point_prediction(group, prediction_column)
        for horizon, group in frame.groupby("horizon", observed=True)
    }


def model_comparison_table(
    frame: pd.DataFrame,
    baseline_columns: dict[str, str],
) -> list[dict[str, Any]]:
    """Create machine-readable overall and per-horizon comparison rows."""

    rows: list[dict[str, Any]] = []
    groups: list[tuple[str | int, pd.DataFrame]] = [("overall", frame)]
    if "horizon" in frame:
        groups.extend(
            (int(h), group) for h, group in frame.groupby("horizon", observed=True)
        )
    for horizon, group in groups:
        lightgbm = evaluate_quantile_predictions(group)
        rows.append(
            {
                "model": "lightgbm_quantile",
                "horizon": horizon,
                "mae": lightgbm.get("mae_p50"),
                "wape": lightgbm.get("wape_p50"),
                "bias": lightgbm.get("bias_p50"),
                "pinball_q25": lightgbm.get("pinball_q25"),
                "pinball_q50": lightgbm.get("pinball_q50"),
                "pinball_q75": lightgbm.get("pinball_q75"),
            }
        )
        for model, column in baseline_columns.items():
            point = evaluate_point_prediction(group, column)
            rows.append(
                {
                    "model": model,
                    "horizon": horizon,
                    "mae": point.get("mae"),
                    "wape": point.get("wape"),
                    "bias": point.get("bias"),
                    "pinball_q25": None,
                    "pinball_q50": None,
                    "pinball_q75": None,
                }
            )
    return rows


def comparison_diagnostics(table: list[dict[str, Any]]) -> dict[str, Any]:
    overall = [row for row in table if row["horizon"] == "overall"]
    baselines = [
        row for row in overall if row["model"] != "lightgbm_quantile" and row["wape"] is not None
    ]
    lightgbm = next((row for row in overall if row["model"] == "lightgbm_quantile"), None)
    if not baselines or lightgbm is None or lightgbm["wape"] is None:
        return {"best_baseline_by_wape": None, "lightgbm_vs_best_baseline_wape_delta": None}
    best = min(baselines, key=lambda row: float(row["wape"]))
    return {
        "best_baseline_by_wape": best["model"],
        "lightgbm_vs_best_baseline_wape_delta": float(lightgbm["wape"] - best["wape"]),
    }
