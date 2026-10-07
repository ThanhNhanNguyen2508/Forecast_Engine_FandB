from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from datetime import date

from shelfcash_forecast.calibration.cqr import (
    apply_cqr_calibrator,
    fit_cqr_calibrator,
)
from shelfcash_forecast.calibration.crossing import correct_quantile_crossing
from shelfcash_forecast.calibration.metrics import calibration_metrics
from shelfcash_forecast.baselines.ets import forecast_ets_path
from shelfcash_forecast.config import ForecastConfig
from shelfcash_forecast.evaluation.splits import (
    DateSplit,
    apply_final_split,
    final_split_origin_audit,
)
from shelfcash_forecast.features.historical import add_historical_features
from shelfcash_forecast.features.training_table import add_target_seasonal_lags
from shelfcash_forecast.inventory.contracts import (
    InventoryDemandLine,
    InventoryDemandScenario,
)
from shelfcash_forecast.optimization.contracts import OptimizationRequest
from shelfcash_preprocess.engine import _fill_sales_units_from_menu
from shelfcash_forecast.scenario.yield_loss import build_theoretical_usage_history
from shelfcash_forecast.decision_intelligence.coherence import (
    _latest_exact_simulation_start,
)


def test_final_split_purges_multihorizon_rows_before_fit_is_available() -> None:
    split = DateSplit(
        train_end=pd.Timestamp("2026-06-16"),
        calibration_start=pd.Timestamp("2026-06-17"),
        calibration_end=pd.Timestamp("2026-07-14"),
        test_start=pd.Timestamp("2026-07-15"),
        test_end=pd.Timestamp("2026-08-11"),
    )
    rows = []
    for target_date in pd.date_range("2026-01-01", "2026-08-11"):
        for horizon in range(1, 8):
            rows.append(
                {
                    "target_date": target_date,
                    "cutoff_date": target_date - pd.Timedelta(days=horizon),
                    "horizon": horizon,
                }
            )
    table = pd.DataFrame(rows)

    _train, calibration, test = apply_final_split(table, split)
    audit = final_split_origin_audit(table, split)

    assert calibration["cutoff_date"].min() == split.train_end
    assert test["cutoff_date"].min() == split.calibration_end
    # At each boundary 1+2+...+6 rows are unavailable across H1..H7.
    assert audit["calibration_purged_rows"] == 21
    assert audit["test_purged_rows"] == 21


def test_future_demand_change_does_not_change_features_at_earlier_origin() -> None:
    config = ForecastConfig()
    dates = pd.date_range("2026-01-01", periods=50)
    base = pd.DataFrame(
        {
            "date": dates,
            "store_key": "STORE_A",
            "product_key": "P1",
            "product_name": "Product",
            "unit": "unit",
            "feature_demand": np.arange(50, dtype=float),
            "demand_proxy": np.arange(50, dtype=float),
            "is_stockout": False,
        }
    )
    changed = base.copy()
    changed.loc[changed["date"] > pd.Timestamp("2026-02-09"), "feature_demand"] = 9999
    before = add_historical_features(base, config)
    after = add_historical_features(changed, config)
    cutoff = pd.Timestamp("2026-02-09")
    feature_columns = [
        column
        for column in before.columns
        if column.startswith(("cutoff_lag_", "rolling_", "stockout_"))
        or column
        in {
            "last_observed_demand",
            "history_observation_count",
            "mean_last_7_minus_previous_7",
        }
    ]
    pd.testing.assert_series_equal(
        before.loc[before["date"].eq(cutoff), feature_columns].iloc[0],
        after.loc[after["date"].eq(cutoff), feature_columns].iloc[0],
    )

    rows = pd.DataFrame(
        {
            "cutoff_date": [cutoff],
            "target_date": [cutoff + pd.Timedelta(days=7)],
            "store_key": ["STORE_A"],
            "product_key": ["P1"],
        }
    )
    before_lags = add_target_seasonal_lags(rows, before, config)
    after_lags = add_target_seasonal_lags(rows, after, config)
    pd.testing.assert_series_equal(
        before_lags.filter(like="seasonal_lag").iloc[0],
        after_lags.filter(like="seasonal_lag").iloc[0],
    )


def test_cqr_round_trip_keeps_median_and_reports_computed_crossing() -> None:
    calibration = pd.DataFrame(
        {
            "target": np.arange(30, dtype=float),
            "horizon": [1] * 30,
            "p25_raw": np.arange(30, dtype=float) - 2,
            "p50_raw": np.arange(30, dtype=float),
            "p75_raw": np.arange(30, dtype=float) + 2,
        }
    )
    corrected = correct_quantile_crossing(calibration)
    calibrator = fit_cqr_calibrator(corrected, 0.5, 20)
    output = apply_cqr_calibrator(corrected, calibrator)
    metrics = calibration_metrics(output, 0.5)

    np.testing.assert_allclose(output["p50"], corrected["p50"])
    assert (output["interval_lower"] <= output["p50"]).all()
    assert (output["interval_upper"] >= output["p50"]).all()
    assert metrics["corrected_crossing_rate"] == 0.0
    assert metrics["calibrated_interval_invalid_rate"] == 0.0
    assert metrics["calibrated_interval_excludes_median_rate"] == 0.0


def test_optimization_accepts_day_after_eod_snapshot_as_scenario_start() -> None:
    scenario = InventoryDemandScenario(
        scenario_id="S1",
        probability_weight=1.0,
        simulation_start_date=date(2026, 8, 13),
        simulation_end_date=date(2026, 8, 19),
        lines=[
            InventoryDemandLine(
                scenario_id="S1",
                store_id="STORE_A",
                ingredient_id="ING_A",
                target_date=date(2026, 8, 13),
                quantity=1.0,
                unit="kg",
            )
        ],
    )
    request = OptimizationRequest(
        request_id="eod-boundary",
        decision_date=date(2026, 8, 12),
        planning_end_date=date(2026, 8, 19),
        initial_inventory=[],
        demand_scenarios=[scenario],
        supplier_offers=[],
    )
    assert request.demand_scenarios[0].simulation_start_date == date(2026, 8, 13)


def test_ets_uses_recent_contiguous_suffix_after_an_old_gap() -> None:
    dates = pd.date_range("2026-01-01", periods=45)
    history = pd.Series(10.0 + np.sin(np.arange(45) * 2 * np.pi / 7), index=dates)
    history.iloc[5] = np.nan

    path = forecast_ets_path(history, maximum_horizon=7)

    assert path is not None
    assert len(path) == 7
    assert np.isfinite(path).all()


def test_forecast_bridge_fills_missing_unit_from_menu_without_changing_ids() -> None:
    sales = pd.DataFrame(
        {
            "product_id": ["P1", "P2"],
            "unit": [pd.NA, "box"],
            "quantity_sold": [1.0, 2.0],
        }
    )
    menu = pd.DataFrame(
        {"product_id": ["P1", "P2"], "unit": ["unit", "should_not_replace"]}
    )

    result = _fill_sales_units_from_menu({"sales": sales, "menu": menu})

    assert result["product_id"].tolist() == ["P1", "P2"]
    assert result["quantity_sold"].tolist() == [1.0, 2.0]
    assert result["unit"].tolist() == ["unit", "box"]


def test_yield_loss_history_rejects_partial_pre_effective_denominator() -> None:
    sales = pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-06-01"],
            "store_id": ["STORE_A", "STORE_A"],
            "product_id": ["P1", "P1"],
            "product_name": ["Product", "Product"],
            "quantity_sold": [2.0, 3.0],
            "unit": ["unit", "unit"],
        }
    )
    recipes = pd.DataFrame(
        {
            "recipe_id": ["R1"],
            "product_id": ["P1"],
            "ingredient_id": ["I1"],
            "ingredient_quantity": [0.5],
            "ingredient_unit": ["kg"],
            "yield_quantity": [1.0],
            "yield_unit": ["unit"],
            "recipe_version": ["v1"],
            "effective_from": ["2026-06-01"],
        }
    )

    with pytest.raises(Exception, match="complete theoretical usage history"):
        build_theoretical_usage_history(sales, recipes)


def test_m6_coherence_accepts_next_day_eod_inventory_transition() -> None:
    scenario = InventoryDemandScenario(
        scenario_id="S1",
        probability_weight=1.0,
        simulation_start_date=date(2026, 8, 13),
        simulation_end_date=date(2026, 8, 19),
        lines=[
            InventoryDemandLine(
                scenario_id="S1",
                store_id="STORE_A",
                ingredient_id="ING_A",
                target_date=date(2026, 8, 13),
                quantity=1.0,
                unit="kg",
            )
        ],
    )
    request = OptimizationRequest(
        request_id="eod-coherence",
        decision_date=date(2026, 8, 12),
        planning_end_date=date(2026, 8, 19),
        initial_inventory=[],
        demand_scenarios=[scenario],
        supplier_offers=[],
    )
    assert _latest_exact_simulation_start(request) == date(2026, 8, 13)
