from __future__ import annotations

import pandas as pd


def _first_non_missing_unit(series: pd.Series) -> object:
    observed = series.astype("string").str.strip()
    observed = observed.loc[observed.notna() & observed.ne("")]
    if observed.empty:
        return pd.NA
    return observed.iloc[0]


def build_daily_panel(
    sales: pd.DataFrame,
    calendar: pd.DataFrame | None = None,
    end_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame: # Tạo một dòng cho mọi ngày của mọi cặp cửa hàng–sản phẩm, kể cả ngày không có sales record.
    """Create one row per day and observed store-product series."""

    series = sales[
        ["store_key", "product_key", "product_name", "unit"]
    ].copy()
    series["unit"] = series.groupby(
        ["store_key", "product_key"],
        observed=True,
    )["unit"].transform(_first_non_missing_unit)
    series = series.drop_duplicates(
        ["store_key", "product_key"]
    )  # giữ một dòng đại diện cho mỗi cặp cửa hàng–sản phẩm.

    panel_end = sales["date"].max() # ngày cúi record đc
    if end_date is not None:
        panel_end = max(panel_end, pd.Timestamp(end_date).normalize())

    date_frame = pd.DataFrame( # tạo danh sách chuỗi ngày từ start -> end
        {
            "date": pd.date_range(
                sales["date"].min(), # ngày đầu tiên có record bán hàng
                panel_end,
                freq="D", # bước nhảy 1 ngày
            )
        }
    )

    panel = (
        series.assign(_join_key=1) # thêm 1 cột vào series với giá trị 1 để join với date_frame
        .merge(date_frame.assign(_join_key=1), on="_join_key", how="inner") # merge xong sẽ gồm store - product - date
        .drop(columns="_join_key") # xoá join_key vì nó chỉ phục vụ cross join 
    ) # join_key để trung gian

    observed_columns = [
        "date", # mốc để link vào panel
        "store_key", # mốc để link vào panel
        "product_key",
        "quantity_sold",
        "selling_price",
        "revenue",
        "is_stockout",
        "promotion_name",
    ]
# panel trước khi merge sales có năm cột:
# store_key
# product_key
# product_name
# unit
# date
    panel = panel.merge(
        sales[observed_columns], # chứa all các cột quan sát được từ sales : data real
        on=["date", "store_key", "product_key"],
        how="left",
        validate="one_to_one",
    )
    # Sau merge : thêm các cột ch có trong observed+columns vào panel + data real
    panel["row_observed"] = panel["quantity_sold"].notna() # cột này đánh dấu xem ngày đó có bán hàng hay không, nếu có bán thì quantity_sold sẽ khác NaN

    if calendar is not None and not calendar.empty:
        panel = panel.merge(calendar, on="date", how="left", validate="many_to_one") # thêm calendar đặc tả vào 1 ngày tương ứng trong panel date hiện tại

    if "is_store_closed" not in panel.columns:
        panel["is_store_closed"] = False
    panel["is_store_closed"] = panel["is_store_closed"].fillna(False).astype(bool)
    panel["store_open"] = ~panel["is_store_closed"]

    if "is_holiday" not in panel.columns:
        panel["is_holiday"] = False
    if "temperature" not in panel.columns:
        panel["temperature"] = pd.NA
    if "rainfall" not in panel.columns:
        panel["rainfall"] = pd.NA

    return panel.sort_values(
        ["store_key", "product_key", "date"]
    ).reset_index(drop=True)
# panel gồm ngày và full data của store-product-date, kể cả ngày không có bán hàng, có thể có missing sales, stockout, holiday, temperature, rainfall.

def resolve_missing_sales(panel: pd.DataFrame) -> pd.DataFrame: # Phân biệt ngày thiếu sales vì cửa hàng đóng và ngày cửa hàng mở nhưng không có record.
    frame = panel.copy()
    frame["sales_missing_reason"] = pd.Series(pd.NA, index=frame.index, dtype="string") # tạo thêm 1 cột misssiong reason

    closed = ~frame["store_open"]
    frame.loc[closed, "quantity_sold"] = pd.NA # nếu cửa hàng đóng thì quantity_sold = NaN
    frame.loc[closed & ~frame["row_observed"], "sales_missing_reason"] = "STORE_CLOSED" # nếu cửa hàng đóng và không có record bán hàng thì sales_missing_reason = STORE_CLOSED

    missing_open = frame["store_open"] & ~frame["row_observed"] # nếu cửa hàng mở nhưng không có record bán hàng thì sales_missing_reason = MISSING_OPEN_DAY_RECORD
    frame.loc[missing_open, "sales_missing_reason"] = "MISSING_OPEN_DAY_RECORD"

    return frame
