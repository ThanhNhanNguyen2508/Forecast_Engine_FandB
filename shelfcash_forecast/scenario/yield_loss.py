from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Protocol

import numpy as np
import pandas as pd

from shelfcash_forecast.bom.adapter import (
    adapt_unit_conversions,
    validate_sales_product_unit_consistency,
)
from shelfcash_forecast.bom.contracts import UnitConversionRule
from shelfcash_forecast.bom.engine import propagate_ingredient_demand
from shelfcash_forecast.bom.units import UnitConverter
from shelfcash_forecast.config import ForecastConfig
from shelfcash_forecast.contracts import ForecastPackage, ForecastPrediction
from shelfcash_forecast.data.adapter import adapt_sales_history
from shelfcash_forecast.data.panel_builder import build_daily_panel
from shelfcash_forecast.data.validator import validate_sales
from shelfcash_forecast.exceptions import (
    ScenarioDataInsufficiencyError,
    ScenarioValidationError,
)


class YieldLossModel(Protocol):
    source: str

    def multiplier(
        self,
        *,
        store_id: str,
        ingredient_id: str,
        target_date: date,
        theoretical_quantity: float,
    ) -> float: ...


@dataclass(frozen=True)
class FixedRecipeYieldLossModel:
    """Preserve the fixed recipe yield/loss/waste calculation exactly."""

    source: str = "recipe_fixed"

    def multiplier(
        self,
        *,
        store_id: str,
        ingredient_id: str,
        target_date: date,
        theoretical_quantity: float,
    ) -> float:
        del store_id, ingredient_id, target_date, theoretical_quantity
        return 1.0


@dataclass(frozen=True)
class EmpiricalUsageResidualYieldLossModel:
    """Auditable hierarchical median model for actual/theoretical usage residuals."""

    by_store_ingredient: Mapping[tuple[str, str], float]
    by_ingredient: Mapping[str, float]
    by_store: Mapping[str, float]
    global_multiplier: float
    sample_count: int
    source: str = "usage_residual_empirical"

    @classmethod
    def fit(
        cls,
        ingredient_usage_history: pd.DataFrame,
        theoretical_usage_history: pd.DataFrame,
        *,
        conversion_rules: list[UnitConversionRule] | None = None,
        minimum_samples: int = 3,
        epsilon: float = 1e-6,
    ) -> EmpiricalUsageResidualYieldLossModel:
        actual_required = {
            "date",
            "store_id",
            "ingredient_id",
            "actual_usage_quantity",
            "unit",
        }
        theoretical_required = {
            "date",
            "store_id",
            "ingredient_id",
            "theoretical_usage_quantity",
            "unit",
        }
        actual_missing = actual_required - set(ingredient_usage_history.columns)
        theoretical_missing = theoretical_required - set(theoretical_usage_history.columns)
        if actual_missing or theoretical_missing:
            raise ScenarioValidationError(
                "Usage history thiếu cột để fit yield/loss residual model.",
                details={
                    "actual_missing": sorted(actual_missing),
                    "theoretical_missing": sorted(theoretical_missing),
                },
            )

        actual = ingredient_usage_history[list(actual_required)].copy()
        theoretical = theoretical_usage_history[list(theoretical_required)].copy()
        actual["date"] = pd.to_datetime(actual["date"], errors="coerce").dt.normalize()
        theoretical["date"] = pd.to_datetime(
            theoretical["date"], errors="coerce"
        ).dt.normalize()
        actual["actual_usage_quantity"] = pd.to_numeric(
            actual["actual_usage_quantity"], errors="coerce"
        )
        theoretical["theoretical_usage_quantity"] = pd.to_numeric(
            theoretical["theoretical_usage_quantity"], errors="coerce"
        )
        merged = actual.merge(
            theoretical,
            on=["date", "store_id", "ingredient_id"],
            suffixes=("_actual", "_theoretical"),
            how="inner",
            validate="many_to_one",
        )
        converter = UnitConverter(conversion_rules)
        converted_actual: list[float] = []
        for row in merged.itertuples(index=False):
            converted_actual.append(
                converter.convert(
                    float(row.actual_usage_quantity),
                    str(row.unit_actual),
                    str(row.unit_theoretical),
                    ingredient_id=str(row.ingredient_id),
                )
            )
        merged["actual_converted"] = converted_actual
        valid = (
            merged["date"].notna()
            & merged["actual_converted"].ge(0)
            & merged["theoretical_usage_quantity"].gt(epsilon)
            & np.isfinite(
                merged[["actual_converted", "theoretical_usage_quantity"]]
            ).all(axis=1)
        )
        merged = merged.loc[valid].copy()
        if len(merged) < minimum_samples:
            raise ScenarioDataInsufficiencyError(
                "Không đủ actual ingredient usage để fit yield/loss residual model.",
                details={
                    "sample_count": len(merged),
                    "minimum_samples": minimum_samples,
                },
            )
        merged["multiplier"] = (
            merged["actual_converted"] / merged["theoretical_usage_quantity"]
        ).clip(lower=epsilon)

        def grouped_medians(columns: list[str]) -> dict[object, float]:
            output: dict[object, float] = {}
            for key, group in merged.groupby(columns, observed=True):
                if len(group) < minimum_samples:
                    continue
                normalized_key: object = key
                if len(columns) == 1 and isinstance(key, tuple):
                    normalized_key = key[0]
                output[normalized_key] = float(group["multiplier"].median())
            return output

        return cls(
            by_store_ingredient=grouped_medians(["store_id", "ingredient_id"]),
            by_ingredient=grouped_medians(["ingredient_id"]),
            by_store=grouped_medians(["store_id"]),
            global_multiplier=float(merged["multiplier"].median()),
            sample_count=len(merged),
        )

    def multiplier(
        self,
        *,
        store_id: str,
        ingredient_id: str,
        target_date: date,
        theoretical_quantity: float,
    ) -> float:
        del target_date, theoretical_quantity
        return float(
            self.by_store_ingredient.get(
                (store_id, ingredient_id),
                self.by_ingredient.get(
                    ingredient_id,
                    self.by_store.get(store_id, self.global_multiplier),
                ),
            )
        )


def build_theoretical_usage_history(
    sales_history: pd.DataFrame,
    recipes: pd.DataFrame,
    unit_conversions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Apply fixed Recipe/BOM assumptions to actual product sales for training labels."""

    sales = adapt_sales_history(sales_history, ForecastConfig())
    validate_sales_product_unit_consistency(sales)
    sales, _report = validate_sales(sales)
    panel = build_daily_panel(sales)
    observed = panel.loc[panel["row_observed"]].copy()
    predictions = [
        ForecastPrediction(
            store_id=str(row.store_key),
            product_id=str(row.product_key),
            product_name=str(row.product_name),
            unit=None if pd.isna(row.unit) else str(row.unit),
            target_date=pd.Timestamp(row.date).date(),
            horizon=1,
            p25=float(row.quantity_sold),
            p50=float(row.quantity_sold),
            p75=float(row.quantity_sold),
            interval_lower=float(row.quantity_sold),
            interval_upper=float(row.quantity_sold),
            baseline_p50=float(row.quantity_sold),
            calibration_source="historical_actual",
        )
        for row in observed.itertuples(index=False)
    ]
    if not predictions:
        raise ScenarioDataInsufficiencyError(
            "Không có product sales hợp lệ để tạo theoretical usage history."
        )
    package = propagate_ingredient_demand(
        ForecastPackage(
            forecast_date=(observed["date"].min() - pd.Timedelta(days=1)).date(),
            forecast_horizon=1,
            model_version="historical-actual-recipe-fixed",
            predictions=predictions,
        ),
        recipes,
        unit_conversions,
    )
    # A partial denominator is unsafe: a sold product without an effective
    # recipe may contain any ingredient.  Fitting on resolved products alone
    # would falsely call the store/ingredient/day group complete.
    if package.issues or not package.predictions:
        raise ScenarioValidationError(
            "Cannot build complete theoretical usage history.",
            details={
                "issues": [
                    issue.model_dump(mode="json")
                    for issue in package.issues
                ],
                "coverage_status": "INCOMPLETE",
                "resolved_prediction_count": len(package.predictions),
            },
        )
    return pd.DataFrame(
        [
            {
                "date": prediction.target_date,
                "store_id": prediction.store_id,
                "ingredient_id": prediction.ingredient_id,
                "theoretical_usage_quantity": prediction.p50,
                "unit": prediction.unit,
            }
            for prediction in package.predictions
        ]
    )


def fit_usage_residual_yield_loss_model(
    ingredient_usage_history: pd.DataFrame,
    sales_history: pd.DataFrame,
    recipes: pd.DataFrame,
    unit_conversions: pd.DataFrame | None = None,
    *,
    minimum_samples: int = 3,
    cutoff_date: str | pd.Timestamp | None = None,
) -> EmpiricalUsageResidualYieldLossModel:
    usage_history = ingredient_usage_history.copy()
    product_sales = sales_history.copy()
    if cutoff_date is not None:
        cutoff = pd.Timestamp(cutoff_date).normalize()
        if "date" not in usage_history.columns or "date" not in product_sales.columns:
            raise ScenarioValidationError(
                "Usage and sales history require date for cutoff-safe fitting."
            )
        usage_dates = pd.to_datetime(usage_history["date"], errors="coerce").dt.normalize()
        sales_dates = pd.to_datetime(product_sales["date"], errors="coerce").dt.normalize()
        usage_history = usage_history.loc[usage_dates.le(cutoff)].copy()
        product_sales = product_sales.loc[sales_dates.le(cutoff)].copy()
    theoretical = build_theoretical_usage_history(
        product_sales,
        recipes,
        unit_conversions,
    )
    rules = adapt_unit_conversions(unit_conversions)
    return EmpiricalUsageResidualYieldLossModel.fit(
        usage_history,
        theoretical,
        conversion_rules=rules,
        minimum_samples=minimum_samples,
    )
