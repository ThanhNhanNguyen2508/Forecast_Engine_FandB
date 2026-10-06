from __future__ import annotations

import numpy as np
import pandas as pd

from shelfcash_forecast.baselines.seasonal_naive import seasonal_naive_predict_rows
from shelfcash_forecast.models.quantile_models import QuantileModelBundle


def predict_raw_quantiles(
    bundle: QuantileModelBundle,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    result = frame.copy()
    x = result[bundle.feature_names]
    raw_model = {
        quantile: np.asarray(bundle.models[quantile].predict(x), dtype=float)
        for quantile in (0.25, 0.50, 0.75)
    }
    if bundle.target_strategy == "absolute":
        for quantile, label in ((0.25, "25"), (0.50, "50"), (0.75, "75")):
            result[f"model_q{label}_raw"] = raw_model[quantile]
            result[f"p{label}_raw"] = np.maximum(0.0, raw_model[quantile])
        result["target_strategy"] = "absolute"
        result["residual_blend_lambda"] = 1.0
        return result
    if bundle.target_strategy != "seasonal_residual_signed":
        raise ValueError(f"Unsupported target strategy: {bundle.target_strategy}")

    anchor = seasonal_naive_predict_rows(result)
    anchor_value = anchor["prediction"].to_numpy(dtype=float)
    if "seasonal_reference_date_7" in result:
        unsafe = pd.to_datetime(result["seasonal_reference_date_7"]).gt(
            pd.to_datetime(result["cutoff_date"])
        )
        if unsafe.any():
            raise ValueError("Seasonal residual anchor references data after origin")
    result["seasonal_anchor"] = anchor_value
    result["seasonal_anchor_fallback_used"] = anchor["fallback_used"].to_numpy()
    result["seasonal_anchor_warning"] = anchor["warnings"].astype(str).to_numpy()
    result["target_strategy"] = bundle.target_strategy
    result["residual_blend_lambda"] = float(bundle.residual_blend_lambda)
    for quantile, label in ((0.25, "25"), (0.50, "50"), (0.75, "75")):
        # Residuals are intentionally signed here.  Clipping happens only after
        # the residual has been added to the causal seasonal anchor.
        result[f"residual_q{label}_raw"] = raw_model[quantile]
        result[f"model_q{label}_raw"] = raw_model[quantile]
        result[f"p{label}_raw"] = np.maximum(
            0.0,
            anchor_value + bundle.residual_blend_lambda * raw_model[quantile],
        )
    return result
