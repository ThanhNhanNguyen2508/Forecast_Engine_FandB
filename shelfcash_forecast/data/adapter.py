from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from typing import Mapping

import pandas as pd

from shelfcash_forecast.config import ForecastConfig
from shelfcash_forecast.exceptions import DataValidationError


@dataclass(frozen=True) # dùng để ko cần def __init__() 
class ForecastInput: # output của adapter
    sales_history: pd.DataFrame # pandas dataframe chứa dữ liệu lịch sử bán hàng
    calendar_features: pd.DataFrame | None # dataframe : dữ liệu mảng 2 chiều , ví dụ date là 1 chiều gồm các ngày , product name 1 chiều gồm cf , trà,...
# import pandas as pd
# sales = pd.DataFrame(
#     {
#         "date": ["2026-07-01", "2026-07-02"],
#         "product_name": ["Trà sữa", "Cà phê"],
#         "quantity_sold": [10, 20],
#     }
# )


def normalize_text(value: object) -> str:
    text = "" if pd.isna(value) else str(value) # Kiểm tra None, NaN, pd.NA, NaT
    text = unicodedata.normalize("NFKD", text.strip().lower()) # strip() loại bỏ khoảng trắng đầu và cuối, lower() chuyển thành chữ thường, NFKD chuẩn hóa unicode
    # normalize() : à => "a" + dấu, NFKD : chuẩn hóa unicode thành dạng phân tách ký tự và dấu
    text = "".join(character for character in text if not unicodedata.combining(character)) # combining() : kiểm tra ký tự có phải là ký tự kết hợp (dấu) hay không, nếu là ký tự kết hợp thì bỏ đi
    # dòng trên bỏ dấu
    text = re.sub(r"[^a-z0-9]+", "_", text) # sub : hàm find & replace = regular expressio, thay các ký tụ ko hợp lệ ( ngoài a-z,0-9 ) bằng _ vào text
    return text.strip("_") # bỏ _ đầu cuối


def stable_entity_key(prefix: str, value: object) -> str: # Mục đích của hàm là tạo một mã định danh ổn định từ tên sản phẩm hoặc một giá trị bất kỳ
    normalized = normalize_text(value)
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12] # túi mù tạo riêng , lấy 12 ký tự đầu
    # cùng input → luôn tạo cùng hash
    # khác input → thông thường tạo hash khác
    return f"{prefix}_{digest}" # trả về mã định danh ổn định với prefix + hash
    # prefix là loại domain như product , quantities,...


def _coerce_nullable_bool(series: pd.Series) -> pd.Series: # Hàm để chuyển đổi một cột thành kiểu boolean có thể chứa giá trị NA
    if str(series.dtype) == "boolean":
        return series

    truthy = {"1", "true", "yes", "y", "co", "có", "x"}
    falsy = {"0", "false", "no", "n", "khong", "không"}

    def convert(value: object) -> object:
        if pd.isna(value):
            return pd.NA
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            if value == 1:
                return True
            if value == 0:
                return False
        normalized = normalize_text(value)
        if normalized in truthy:
            return True
        if normalized in falsy:
            return False
        return pd.NA

    return series.map(convert).astype("boolean") # trả yes,1,true,co,... -> True ; chuỗi đang qui ước loạn -> chuỗi qui ước thành True/False thôi


def adapt_sales_history(
    sales: pd.DataFrame,
    config: ForecastConfig,
) -> pd.DataFrame:
    required = {"date", "product_name", "quantity_sold"}
    missing = required - set(sales.columns)
    if missing:
        raise DataValidationError(
            f"sales_history thiếu cột bắt buộc: {sorted(missing)}"
        )

    frame = sales.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize() # to_datetime() : chuyển đổi chuỗi thành datetime, errors="coerce" : nếu lỗi thì trả về NaT, dt.normalize() : chuẩn hóa ngày tháng về 00:00:00
    frame["product_name"] = frame["product_name"].astype("string").str.strip() # astype("string") : chuyển đổi cột thành kiểu string, str.strip() : loại bỏ khoảng trắng đầu và cuối
    frame["quantity_sold"] = pd.to_numeric(frame["quantity_sold"], errors="coerce") # to_numeric() : chuyển đổi cột thành kiểu số, errors="coerce" : nếu lỗi thì trả về NaN

    if "store_id" in frame.columns:
        store = frame["store_id"].astype("string").str.strip()
        frame["store_key"] = store.fillna(config.default_store_key) # fillna() : điền giá trị mặc định nếu store_id là NaN
        frame.loc[frame["store_key"].eq(""), "store_key"] = config.default_store_key # loc[] : truy cập vào các hàng thỏa mãn điều kiện, eq("") : kiểm tra giá trị bằng chuỗi rỗng, gán giá trị mặc định nếu store_key là chuỗi rỗng
    else:
        frame["store_key"] = config.default_store_key

    if "product_id" in frame.columns:
        product_id = frame["product_id"].astype("string").str.strip()
        generated = frame["product_name"].map(lambda value: stable_entity_key("PRD", value)) # map() : áp dụng hàm stable_entity_key cho từng giá trị trong cột product_name, stable_entity_key("PRD", value) : tạo mã định danh ổn định từ tên sản phẩm với prefix "PRD"
        frame["product_key"] = product_id.where(product_id.notna() & product_id.ne(""), generated)
    else:
        frame["product_key"] = frame["product_name"].map(
            lambda value: stable_entity_key("PRD", value)
        )

    if "is_stockout" not in frame.columns:
        frame["is_stockout"] = pd.Series(pd.NA, index=frame.index, dtype="boolean")
    else:
        frame["is_stockout"] = _coerce_nullable_bool(frame["is_stockout"])

    if "selling_price" not in frame.columns:
        frame["selling_price"] = pd.NA
    frame["selling_price"] = pd.to_numeric(frame["selling_price"], errors="coerce")

    if "revenue" not in frame.columns:
        frame["revenue"] = pd.NA
    frame["revenue"] = pd.to_numeric(frame["revenue"], errors="coerce")

    if "unit" not in frame.columns:
        frame["unit"] = pd.NA
    frame["unit"] = frame["unit"].astype("string")

    if "promotion_name" not in frame.columns:
        frame["promotion_name"] = pd.NA
    frame["promotion_name"] = frame["promotion_name"].astype("string")

    columns = [
        "date", # ngày bán hàng
        "store_key", # mã định danh cửa hàng
        "product_key", # mã định danh sản phẩm
        "product_name", # tên sản phẩm
        "quantity_sold", # số lượng bán
        "unit", # đơn vị
        "selling_price", # giá bán
        "revenue", # doanh thu
        "is_stockout", # hết hàng ko ?
        "promotion_name", # tên chương trình khuyến mãi
    ]
    return frame[columns]
# Ví dụ toàn bộ adapt_sales_history()
# Input:
# date                 = "2026-07-30 15:20"
# product_name         = "  Trà sữa  "
# quantity_sold        = "12"
# store_id             = None
# product_id           = ""
# is_stockout          = "YES"
# selling_price        = "35000"
# extra_column         = "ignored"

# Output ý tưởng:
# date                 = 2026-07-30 00:00:00
# store_key            = "STORE_DEFAULT"
# product_key          = "PRD_<hash từ tra_sua>"
# product_name         = "Trà sữa"
# quantity_sold        = 12
# unit                 = <NA>
# selling_price        = 35000
# revenue              = NaN
# is_stockout          = True
# promotion_name       = <NA>

def adapt_calendar(calendar: pd.DataFrame | None) -> pd.DataFrame | None:
    if calendar is None or calendar.empty:
        return None
    if "date" not in calendar.columns:
        raise DataValidationError("calendar_features thiếu cột date.")

    frame = calendar.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()

    for column in ("is_weekend", "is_holiday", "is_store_closed", "is_promotion"):
        if column not in frame.columns:
            frame[column] = pd.Series(pd.NA, index=frame.index, dtype="boolean")
        else:
            frame[column] = _coerce_nullable_bool(frame[column])

    for column in ("temperature", "rainfall"):
        if column not in frame.columns:
            frame[column] = pd.NA
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if "promotion_name" not in frame.columns:
        frame["promotion_name"] = pd.NA
    frame["promotion_name"] = frame["promotion_name"].astype("string")

    if "weather_kind" not in frame.columns:
        frame["weather_kind"] = pd.NA
    frame["weather_kind"] = frame["weather_kind"].astype("string").str.lower()
    for column in ("weather_issued_at", "weather_available_at"):
        if column not in frame.columns:
            frame[column] = pd.NaT
        frame[column] = pd.to_datetime(frame[column], errors="coerce")

    return frame[
        [
            "date", # ngày áp dụng thông tin calendar
            "is_weekend", # là cuối tuần không?
            "is_holiday", # là ngày lễ không?
            "is_store_closed", # cửa hàng đóng cửa không?
            "is_promotion", # là chương trình khuyến mãi không?
            "promotion_name", # tên chương trình khuyến mãi
            "temperature", # nhiệt độ
            "rainfall", # lượng mưa
            "weather_kind",
            "weather_issued_at",
            "weather_available_at",
        ]
    ]


def adapt_forecast_input(
    canonical_data: Mapping[str, pd.DataFrame],
    config: ForecastConfig,
) -> ForecastInput:
    if "sales_history" not in canonical_data:
        raise DataValidationError("canonical_data bắt buộc có sales_history.")

    return ForecastInput(
        sales_history=adapt_sales_history(canonical_data["sales_history"], config),
        calendar_features=adapt_calendar(canonical_data.get("calendar_features")),
    )
