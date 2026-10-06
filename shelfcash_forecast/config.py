from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ForecastConfig:
    """Configuration shared by training and inference."""

    horizons: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7) # cutoff_day -> cutoff_day + horizon = forecast_day
    quantiles: tuple[float, ...] = (0.25, 0.50, 0.75) # 3 quantiles: P25, P50, P75

    minimum_history_observations: int = 28 # min number of days of history to be eligible for training
    reconstruction_minimum_reference_count: int = 3
    reconstruction_lookback_days: int = 84
    reconstruction_high_support_threshold: int = 6
    reconstruction_high_recency_days: int = 14
    reconstruction_medium_recency_days: int = 28
    reconstruction_high_dispersion_threshold: float = 0.25
    reconstruction_medium_dispersion_threshold: float = 0.50
    calibration_days: int = 28 # number of days to use for calibration
    test_days: int = 28 # number of days to use for testing

    nominal_coverage: float = 0.50 # mong muốn bao phủ đc 50% trong thực tế
    minimum_calibration_samples: int = 20 # số lượng mẫu tối thiểu để thực hiện calibration

    lag_days: tuple[int, ...] = (1, 2, 7, 14, 28) # tính từ ngày cutoff, lấy các giá trị lag của target để làm feature
    rolling_windows: tuple[int, ...] = (7, 14, 28) # `rolling_mean_7` là nhu cầu trung bình trong cửa sổ bảy ngày gần cutoff.
    target_seasonal_lags: tuple[int, ...] = (7, 14, 28) # tính từ target_day

    # Walk-forward dùng để kiểm tra model qua nhiều giai đoạn thời gian, thay vì chỉ dựa vào một test window duy nhất.
    walk_forward_minimum_train_days: int = 84 # số ngày tối thiểu để huấn luyện trong mỗi bước walk-forward
    walk_forward_validation_days: int = 14 # số ngày để kiểm tra validation trong mỗi bước walk-forward
    walk_forward_step_days: int = 14 # số ngày bước nhảy trong walk-forward
    walk_forward_maximum_folds: int = 3 # số lượng fold tối đa trong walk-forward

    random_seed: int = 42
    default_store_key: str = "STORE_DEFAULT"

    lightgbm_params: dict[str, Any] = field(
        default_factory=lambda: {
            "learning_rate": 0.04, # tốc độ học của mô hình LightGBM
            "n_estimators": 350, # số lượng cây quyết định trong mô hình LightGBM
            "num_leaves": 31, # số lượng lá tối đa trong mỗi cây quyết định
            "min_child_samples": 20, # số lượng mẫu tối thiểu trong mỗi nút lá của cây quyết định
            "subsample": 0.9, # tỷ lệ mẫu được sử dụng để huấn luyện mỗi cây quyết định
            "subsample_freq": 1, # tần suất thực hiện subsample trong quá trình huấn luyện
            "colsample_bytree": 0.9, # tỷ lệ đặc trưng được sử dụng để huấn luyện mỗi cây quyết định
            "reg_alpha": 0.05, # hệ số phạt L1
            "reg_lambda": 0.5, # hệ số phạt L2
            "random_state": 42, # seed ngẫu nhiên để đảm bảo kết quả có thể tái lập
            "n_jobs": -1, # số lượng luồng song song để huấn luyện mô hình LightGBM (-1 nghĩa là sử dụng tất cả các luồng có sẵn)
            "verbosity": -1, # mức độ chi tiết của thông báo trong quá trình huấn luyện (-1 nghĩa là không hiển thị thông báo)
        }
    )

    def __post_init__(self) -> None:
        if not self.horizons or any(horizon < 1 for horizon in self.horizons):
            raise ValueError("horizons phải chứa các số nguyên dương.")
        if tuple(sorted(set(self.horizons))) != self.horizons:
            raise ValueError("horizons phải tăng dần và không trùng nhau.")
        if self.quantiles != (0.25, 0.50, 0.75):
            raise ValueError("Forecast Core v0.1 chỉ hỗ trợ P25/P50/P75.")
        if not 0 < self.nominal_coverage < 1:
            raise ValueError("nominal_coverage phải nằm trong (0, 1).")
        if self.minimum_history_observations < max(self.rolling_windows):
            raise ValueError(
                "minimum_history_observations phải >= rolling window lớn nhất."
            )
        if self.reconstruction_minimum_reference_count < 1:
            raise ValueError("reconstruction_minimum_reference_count must be positive.")
        if self.reconstruction_lookback_days < 1:
            raise ValueError("reconstruction_lookback_days must be positive.")
        if (
            self.reconstruction_high_support_threshold
            < self.reconstruction_minimum_reference_count
        ):
            raise ValueError(
                "reconstruction_high_support_threshold must be at least the minimum "
                "reference count."
            )
        if not 0 <= self.reconstruction_high_recency_days <= self.reconstruction_medium_recency_days:
            raise ValueError("Reconstruction recency thresholds must be ordered.")
        if not (
            0
            <= self.reconstruction_high_dispersion_threshold
            <= self.reconstruction_medium_dispersion_threshold
        ):
            raise ValueError("Reconstruction dispersion thresholds must be ordered.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ForecastConfig:
        data = dict(payload)
        tuple_fields = {
            "horizons",
            "quantiles",
            "lag_days",
            "rolling_windows",
            "target_seasonal_lags",
        }
        for field_name in tuple_fields:
            if field_name in data:
                data[field_name] = tuple(data[field_name])
        return cls(**data)
