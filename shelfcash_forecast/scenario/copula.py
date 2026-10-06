# Nếu bootstrap.py lấy trực tiếp các residual lịch sử theo block rồi resample, thì copula.py cố học cấu trúc phụ thuộc/correlation giữa nhiều product–store–horizon dimensions, sau đó sinh ra các residual mới có joint structure tương tự lịch sử.

# Điểm cốt lõi nhất của file này là:

# Mỗi dimension vẫn giữ empirical distribution của scaled_residual, nhưng quan hệ giữa các dimensions được mô hình hóa bằng Gaussian Copula.

# Historical residual matrix
#         ↓
# convert từng dimension sang Gaussian ranks
#         ↓
# estimate correlation matrix
#         ↓
# sample multivariate Gaussian
#         ↓
# convert Gaussian samples → uniforms
#         ↓
# map uniforms về empirical residual distribution
#         ↓
# apply vào current forecast

# Do đó copula cố tách hai thứ:

# Marginal distribution
# =
# residual distribution riêng của từng product/store/horizon

# Dependence structure
# =
# các dimensions cùng tăng/giảm với nhau như thế nào
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

from shelfcash_forecast.contracts import ForecastPackage
from shelfcash_forecast.scenario.bootstrap import (
    ResidualVectorBootstrapScenarioGenerator,
)
from shelfcash_forecast.scenario.contracts import (
    ProductDemandScenario,
    ProductDemandScenarioBundle,
    ProductDemandScenarioLine,
)
from shelfcash_forecast.scenario.residuals import validate_residual_history
from shelfcash_forecast.scenario.validation import scenario_reproduction_diagnostics


class GaussianCopulaScenarioGenerator:
    """Gaussian-copula challenger over empirical scaled-residual marginals."""

    method = "gaussian_copula"

    def __init__(
        self,
        *,
        minimum_samples: int = 20, # cần nhiều sample hơn bootstrap vì copula phải ước lượng correlation matrix
        regularization: float = 0.05, # Dùng để làm correlation/covariance matrix ổn định hơn.
        fallback: ResidualVectorBootstrapScenarioGenerator | None = None,
    ) -> None:
        self.minimum_samples = minimum_samples
        self.regularization = regularization
        self.fallback = fallback or ResidualVectorBootstrapScenarioGenerator()

    def _fallback_bundle(
        self,
        forecast: ForecastPackage,
        residuals: pd.DataFrame,
        n_scenarios: int,
        seed: int,
        reason: str,
    ) -> ProductDemandScenarioBundle:
        bundle = self.fallback.generate(
            forecast,
            residuals,
            n_scenarios=n_scenarios,
            seed=seed,
        )
        return bundle.model_copy(
            update={
                "warnings": sorted(
                    set(bundle.warnings + ["COPULA_FALLBACK_RESIDUAL_BOOTSTRAP"])
                ),
                "diagnostics": {
                    **bundle.diagnostics,
                    "requested_method": self.method,
                    "copula_fallback_reason": reason,
                },
            }
        )

    def generate(
        self,
        forecast: ForecastPackage,
        residual_history: pd.DataFrame,
        *,
        n_scenarios: int,
        seed: int,
    ) -> ProductDemandScenarioBundle:
        if n_scenarios < 1:
            raise ValueError("n_scenarios phải >= 1.")
        residuals = validate_residual_history(
            residual_history,
            as_of=forecast.forecast_date,
        )
        predictions = sorted( # Mục tiêu vẫn là deterministic ordering.
            forecast.predictions,
            key=lambda row: (row.store_id, row.product_id, row.target_date, row.horizon),
        )
        dimensions = [
            (prediction.store_id, prediction.product_id, prediction.horizon)
            for prediction in predictions
        ]
        if len(dimensions) != len(set(dimensions)):
            return self._fallback_bundle(
                forecast, residuals, n_scenarios, seed, "DUPLICATE_DIMENSIONS"
            )
#             Mỗi dimension được định nghĩa bởi:

#               Store
#               × Product
#               × Horizon

#               Ví dụ:

#               (S01, Latte, H1)
#               (S01, Mocha, H1)
#               (S01, Latte, H2)
#               (S02, Latte, H1)
# Mỗi tuple này trở thành một biến ngẫu nhiên trong Copula.

        column_series: dict[tuple[str, str, int], pd.Series] = {} # tạo historical marginal cho từng dimension
        for dimension in dimensions:
            store_id, product_id, horizon = dimension
            values = residuals.loc[
                residuals["store_id"].eq(store_id)
                & residuals["product_id"].eq(product_id)
                & residuals["horizon"].eq(horizon),
                ["forecast_origin", "scaled_residual"],
            ].drop_duplicates("forecast_origin")
            if len(values) < self.minimum_samples:
                return self._fallback_bundle(
                    forecast,
                    residuals,
                    n_scenarios,
                    seed,
                    f"INSUFFICIENT_MARGINAL:{dimension!r}",
                )
            column_series[dimension] = values.set_index("forecast_origin")[
                "scaled_residual"
            ]

        matrix = pd.DataFrame(column_series).sort_index().dropna()
#         Origin       Latte H1   Mocha H1   Tea H1
# ------------------------------------------------
# Jun 1          +0.2       +0.3      -0.1
# Jun 2          -0.4       -0.2      -0.3
# Jun 3          +0.6       +0.5      +0.2
# ...

# Mỗi row:

# một historical forecast origin

# Mỗi column:

# một current scenario dimension
        if len(matrix) < self.minimum_samples:
            return self._fallback_bundle(
                forecast, residuals, n_scenarios, seed, "INSUFFICIENT_SHARED_ORIGINS"
            )
        gaussian = np.column_stack(
            [
                norm.ppf(
                    np.clip(
                        (rankdata(matrix[column].to_numpy()) - 0.5) / len(matrix), 
# Ví dụ residuals:
# -0.8
# -0.2
# +0.1
# +0.7

# Ranks:
# 1
# 2
# 3
# 4
# Copula quan tâm thứ tự/rank hơn exact magnitude khi học dependence.
                        1e-6,
                        1 - 1e-6,
                    )
                )
                for column in matrix.columns
            ]
        )
#         28. Convert rank thành uniform percentile
# (rankdata(...) - 0.5) / len(matrix)

# Nếu có N observations: Ui = (ranki - 0.5) / N
# 	​
# Ví dụ N=4:
# rank 1 → 0.125
# rank 2 → 0.375
# rank 3 → 0.625
# rank 4 → 0.875
# Đây là pseudo-uniform empirical probabilities

        covariance = np.corrcoef(gaussian, rowvar=False)
# 34. Correlation matrix
# covariance = np.corrcoef(
#     gaussian,
#     rowvar=False
# )

# Tên variable là covariance, nhưng np.corrcoef() thực ra trả correlation matrix.

# Ví dụ:

#                Latte    Mocha    Tea
# Latte           1.0      0.8    0.3
# Mocha           0.8      1.0    0.4
# Tea             0.3      0.4    1.0

# Interpretation:

# Latte và Mocha residual có tendency cùng rank cao/thấp.
        if covariance.ndim == 0:
            covariance = np.array([[1.0]])
        covariance = (1 - self.regularization) * covariance + self.regularization * np.eye(
            len(dimensions)
        )
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        covariance = eigenvectors @ np.diag(np.clip(eigenvalues, 1e-8, None)) @ eigenvectors.T

        rng = np.random.default_rng(seed)
        gaussian_samples = rng.multivariate_normal(
            np.zeros(len(dimensions)), covariance, size=n_scenarios
        )
#         Đây là bước sinh joint latent shocks.

# Ví dụ dimensions:

# Latte
# Mocha
# Tea

# Một sampled row có thể là:

# [+1.2, +0.9, +0.4]

# Nếu historical dependence Latte–Mocha cao, generated samples cũng có xu hướng cùng chiều.
        uniforms = norm.cdf(gaussian_samples) 
#         43. Gaussian → Uniform
# uniforms = norm.cdf(gaussian_samples)

# Mỗi latent Gaussian value được chuyển về:

# 0 < u < 1

# Ví dụ:

# z = 0     → u=0.5
# z = +1.28 → u≈0.90
# z = -1.28 → u≈0.10

# Ta có correlated uniforms.

        empirical_samples = np.column_stack(
            [
                np.quantile(
                    matrix[dimension].to_numpy(dtype=float),
                    uniforms[:, index],
                    method="linear",
                )
                for index, dimension in enumerate(dimensions)
            ]
        )

        scenarios: list[ProductDemandScenario] = []
        sampled_records: list[dict[str, Any]] = []
        clipped_count = 0
        for scenario_index in range(n_scenarios):
            scenario_id = f"scenario_{scenario_index + 1:04d}"
            lines: list[ProductDemandScenarioLine] = []
            for dimension_index, prediction in enumerate(predictions):
                scaled_residual = float(empirical_samples[scenario_index, dimension_index])
                spread = max(prediction.p75 - prediction.p25, 1e-6)
                raw_quantity = prediction.p50 + scaled_residual * spread
                clipped_count += int(raw_quantity < 0)
                sampled_records.append(
                    {
                        "scenario_id": scenario_id,
                        "store_id": prediction.store_id,
                        "product_id": prediction.product_id,
                        "horizon": prediction.horizon,
                        "scaled_residual": scaled_residual,
                    }
                )
                lines.append(
                    ProductDemandScenarioLine(
                        scenario_id=scenario_id,
                        store_id=prediction.store_id,
                        product_id=prediction.product_id,
                        product_name=prediction.product_name,
                        product_unit=prediction.unit,
                        target_date=prediction.target_date,
                        horizon=prediction.horizon,
                        demand_quantity=max(0.0, raw_quantity),
                        source_model_version=forecast.model_version,
                        scenario_method=self.method,
                    )
                )
            scenarios.append(
                ProductDemandScenario(
                    scenario_id=scenario_id,
                    probability_weight=1.0 / n_scenarios,
                    lines=lines,
                    metadata={"seed": seed, "copula_regularization": self.regularization},
                )
            )

        diagnostics = scenario_reproduction_diagnostics(
            residuals,
            pd.DataFrame(sampled_records),
            clipped_count=clipped_count,
            total_count=len(sampled_records),
        )
        diagnostics.update(
            {
                "seed": seed,
                "scenario_count": n_scenarios,
                "residual_row_count": len(residuals),
                "minimum_samples": self.minimum_samples,
                "regularization": self.regularization,
            }
        )
        return ProductDemandScenarioBundle(
            forecast_date=forecast.forecast_date,
            horizon=forecast.forecast_horizon,
            model_version=forecast.model_version,
            scenario_method=self.method,
            scenarios=scenarios,
            diagnostics=diagnostics,
        )
