# Với product X cần forecast cho ngày Y, chính xác recipe version nào phải được dùng?
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date

from shelfcash_forecast.bom.contracts import BOMIssue, RecipeRecord
from shelfcash_forecast.exceptions import RecipeVersionError


@dataclass(frozen=True) # nội bộ của file này
class RecipeResolution:
    records: tuple[RecipeRecord, ...] # records chứa tất cả ingredient lines thuộc recipe version được chọn
    issue: BOMIssue | None = None
# Trả tuple do 1 product đc cấu thành từ nhiều ingre ( nguyên liệu ) khác nhau, mỗi ingre có 1 record riêng. Nếu không tìm thấy recipe version nào active, issue sẽ được set.
    @property
    def found(self) -> bool:
        return bool(self.records)


class RecipeResolver:
    """Resolve exactly one active recipe version for each product target date."""

    def __init__(self, records: list[RecipeRecord]) -> None: # input là output của adapt_recipe_records()
        by_product: dict[str, list[RecipeRecord]] = defaultdict(list)
        for record in records:
            by_product[record.product_id].append(record) # sắp xếp danh sách lại theo dạng :
# {
#     "COFFEE": [...],
#     "BURGER": [...],
#     "PIZZA": [...],
# }
        self._by_product = {
            product_id: tuple(product_records)
            for product_id, product_records in by_product.items()
        }
# 8. Sau __init__, dữ liệu có dạng gì?

# Giả sử đầu vào:

# P1 / R1 / v1 / ingredient A
# P1 / R1 / v1 / ingredient B
# P1 / R1 / v2 / ingredient A
# P1 / R1 / v2 / ingredient B
# P2 / R2 / v1 / ingredient C

# Sau grouping:

# _by_product
# │
# ├── P1
# │   ├── R1/v1/A
# │   ├── R1/v1/B
# │   ├── R1/v2/A
# │   └── R1/v2/B
# │
# └── P2
#     └── R2/v1/C

# Resolver vẫn chưa chọn version ở bước này.

# Nó chỉ tạo index để lookup nhanh.
    def resolve(
        self,
        product_id: str,
        target_date: date,
        *,
        store_id: str | None = None,
    ) -> RecipeResolution:
        active = [
            record
            for record in self._by_product.get(product_id, ())
            if record.effective_from <= target_date
            and (record.effective_to is None or target_date <= record.effective_to)
        ]
        if not active:
            details: dict[str, object] = {
                "product_id": product_id,
                "target_date": target_date.isoformat(),
            }
            if store_id is not None:
                details["store_id"] = store_id
            return RecipeResolution(
                records=(),
                issue=BOMIssue(
                    code="MISSING_RECIPE",
                    message=(
                        f"Không có recipe active cho product={product_id} "
                        f"tại {target_date.isoformat()}."
                    ),
                    details=details,
                    recoverable=True,
                    suggested_action="Bổ sung recipe version bao phủ target_date.",
                ),
            )

        versions = {
            (
                record.recipe_id,
                record.recipe_version,
                record.effective_from,
                record.effective_to,
            )
            for record in active
        }
# Mỗi active record được collapse thành identity:

# recipe_id
# recipe_version
# effective_from
# effective_to

# Ví dụ active:

# R1 v2 Coffee  2026-01-01 → None
# R1 v2 Milk    2026-01-01 → None
# R1 v2 Sugar   2026-01-01 → None

# Set trở thành:

# {
#     ("R1", "v2", 2026-01-01, None)
# }
        if len(versions) != 1:# overlap nh recipe versions
            raise RecipeVersionError(
                f"Có nhiều recipe versions active cho product={product_id} "
                f"tại {target_date.isoformat()}.",
                details={
                    "product_id": product_id,
                    "target_date": target_date.isoformat(),
                    "store_id": store_id,
                    "active_versions": [
                        {
                            "recipe_id": recipe_id,
                            "recipe_version": version,
                            "effective_from": effective_from.isoformat(),
                            "effective_to": (
                                effective_to.isoformat()
                                if effective_to is not None
                                else None
                            ),
                        }
                        for recipe_id, version, effective_from, effective_to in sorted(
                            versions,
                            key=lambda item: (
                                item[0],
                                item[1],
                                item[2],
                                item[3] or date.max,
                            ),
                        )
                    ],
                },
            )

        ordered = tuple(
            sorted(
                active,
                key=lambda record: (
                    record.ingredient_id,
                    record.ingredient_unit,
                    record.ingredient_quantity,
                ),
            )
        )
        return RecipeResolution(records=ordered)
# Recipe master:

# Product LATTE

# R1 / v1 / COFFEE
# 20 g
# 2025-01-01 → 2025-12-31

# R1 / v1 / MILK
# 200 ml
# 2025-01-01 → 2025-12-31


# R1 / v2 / COFFEE
# 22 g
# 2026-01-01 → None

# R1 / v2 / MILK
# 180 ml
# 2026-01-01 → None

# Gọi:

# resolver.resolve(
#     product_id="LATTE",
#     target_date=date(2026, 8, 15),
# )

# Filter active:

# v1 → false
# v2 COFFEE → true
# v2 MILK → true

# versions:

# {
#     ("R1", "v2", date(2026, 1, 1), None)
# }

# length = 1.

# Return:

# RecipeResolution
# │
# ├── R1/v2/COFFEE 22 g
# └── R1/v2/MILK 180 ml