from __future__ import annotations

import numpy as np
import pandas as pd


def correct_quantile_crossing(
    frame: pd.DataFrame,
    source_columns: tuple[str, str, str] = ("p25_raw", "p50_raw", "p75_raw"),
) -> pd.DataFrame:
    result = frame.copy()
    raw = result[list(source_columns)].to_numpy(dtype=float)
    corrected = np.sort(raw, axis=1)

    result["p25"] = np.maximum(0.0, corrected[:, 0])
    result["p50"] = np.maximum(0.0, corrected[:, 1])
    result["p75"] = np.maximum(0.0, corrected[:, 2])
    result["had_quantile_crossing"] = (
        (raw[:, 0] > raw[:, 1])
        | (raw[:, 1] > raw[:, 2])
        | (raw[:, 0] > raw[:, 2])
    )
    return result
