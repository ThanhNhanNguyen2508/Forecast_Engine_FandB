from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from typing import Iterable

from shelfcash_preprocess.discovery import RegionData
from shelfcash_preprocess.models import (
    FieldMapping,
    ImportProfile,
    MappingProposal,
    Role,
    TransformOperation,
)


def normalize_label(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value)).strip().casefold()
    text = text.replace("đ", "d")
    text = "".join(
        char for char in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(char)
    )
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


ALIASES: dict[Role, dict[str, set[str]]] = {
    Role.SALES: {
        "date": {"date", "ngay", "ngay_gd", "transaction_date", "sales_date"},
        "product_id": {"product_id", "sku", "ma_mon", "ma_san_pham"},
        "product_name": {"product_name", "ten_mon", "ten_mon_sku", "sku_name", "item_name"},
        "quantity_sold": {"quantity_sold", "slx", "so_luong_ban", "qty", "sales_qty"},
        "unit": {"unit", "dvt", "dv", "uom"},
        "selling_price": {"selling_price", "don_gia_ban", "gia_ban", "unit_price"},
        "revenue": {"revenue", "doanh_thu", "line_amount", "amount"},
        "is_stockout": {"is_stockout", "het_mon", "het_mon_", "stockout"},
        "promotion_name": {"promotion_name", "ctkm", "campaign", "khuyen_mai"},
        "store_id": {"store_id", "ma_cua_hang", "chi_nhanh", "store"},
        "transaction_id": {"transaction_id", "invoice_id", "order_id", "receipt_id", "line_id"},
    },
    Role.CALENDAR: {
        "date": {"date", "ngay"},
        "store_id": {"store_id", "store", "ma_cua_hang"},
        "is_weekend": {"is_weekend", "weekend", "cuoi_tuan"},
        "is_holiday": {"is_holiday", "holiday", "ngay_le"},
        "is_store_closed": {"is_store_closed", "store_closed", "dong_cua"},
        "is_promotion": {"is_promotion", "promo", "promotion"},
        "promotion_name": {"promotion_name", "campaign", "ctkm"},
        "temperature": {"temperature", "temp_c", "nhiet_do"},
        "rainfall": {"rainfall", "rain_mm", "luong_mua"},
    },
    Role.MENU: {
        "product_id": {"product_id", "ma_mon", "sku"},
        "product_type": {"product_type", "loai", "type"},
        "product_name": {"product_name", "ten_mon_combo", "ten_mon", "name"},
        "components": {"components", "thanh_phan_combo"},
        "unit": {"unit", "dvt", "uom"},
        "selling_price": {"selling_price", "gia_ban"},
        "status": {"status", "trang_thai"},
    },
    Role.RECIPES: {
        "product_name": {"product_name", "mon_ban"},
        "product_id": {"product_id", "ma_mon"},
        "ingredient_name": {"ingredient_name", "thanh_phan", "ten_nl"},
        "ingredient_id": {"ingredient_id", "ma_nl"},
        "ingredient_quantity": {"ingredient_quantity", "luong_1_sp", "quantity_per_product"},
        "ingredient_unit": {"ingredient_unit", "don_vi_nl", "dvt_nl"},
        "yield_quantity": {"yield_quantity", "yield"},
        "yield_unit": {"yield_unit", "don_vi_sp"},
        "recipe_version": {"recipe_version", "ver", "version"},
        "effective_from": {"effective_from", "ap_dung_tu", "start_date"},
        "effective_to": {"effective_to", "ap_dung_den", "end_date"},
        "process_loss_rate": {"process_loss_rate"},
        "waste_allowance_rate": {"waste_allowance_rate"},
        "recipe_id": {"recipe_id", "ma_cong_thuc"},
    },
    Role.INGREDIENT_USAGE: {
        "date": {"date", "ngay_su_dung"},
        "store_id": {"store_id", "ma_cua_hang"},
        "ingredient_name": {"ingredient_name", "ten_nl", "material"},
        "ingredient_id": {"ingredient_id", "ma_nl"},
        "actual_usage_quantity": {"actual_usage_quantity", "luong_thuc_dung"},
        "unit": {"unit", "dvt"},
        "waste_quantity": {"waste_quantity", "hao_hut"},
        "source": {"source", "nguon"},
    },
    Role.INVENTORY_SNAPSHOT: {
        "snapshot_date": {"snapshot_date", "ngay_kiem_ke", "as_of_date"},
        "store_id": {"store_id", "ma_cua_hang"},
        "ingredient_name": {"ingredient_name", "mat_hang_ton", "material"},
        "ingredient_id": {"ingredient_id", "ma_nl"},
        "quantity_remaining": {"quantity_remaining", "sl_cuoi_ngay", "quantity"},
        "unit": {"unit", "dv", "dvt"},
        "expiry_date": {"expiry_date", "hsd", "han_dung"},
        "lot_id": {"lot_id", "ma_lo", "batch"},
        "received_date": {"received_date", "ngay_nhap_hang", "ngay_nhan"},
        "location": {"location", "vi_tri"},
    },
    Role.PURCHASE_HISTORY: {
        "document_date": {"document_date", "nct", "ngay_chung_tu"},
        "ingredient_name": {"ingredient_name", "ten_hang", "material"},
        "ingredient_id": {"ingredient_id", "ma_nl"},
        "quantity": {"quantity", "sln", "so_luong_nhap"},
        "unit": {"unit", "dvt"},
        "unit_price": {"unit_price", "gia_nhap"},
        "line_amount": {"line_amount", "thanh_tien"},
        "supplier_id": {"supplier_id", "ncc", "vendor"},
        "expiry_date": {"expiry_date", "han_dung", "hsd"},
        "lot_id": {"lot_id", "batch", "ma_lo"},
        "purchase_order_id": {"purchase_order_id", "so_po", "po"},
        "received_date": {"received_date", "ngay_nhap_hang", "arrival_date"},
    },
    Role.SUPPLIER_RULES: {
        "supplier_id": {"supplier_id", "vendor", "ncc"},
        "ingredient_name": {"ingredient_name", "material", "ten_hang"},
        "ingredient_id": {"ingredient_id", "ma_nl"},
        "minimum_order_quantity": {"minimum_order_quantity", "moq"},
        "order_unit": {"order_unit", "order_uom"},
        "pack_size": {"pack_size"},
        "unit": {"unit", "base_uom"},
        "lead_time_days": {"lead_time_days", "lead_time_days_"},
        "unit_price": {"unit_price", "gia_mua"},
        "delivery_schedule": {"delivery_schedule", "lich_giao"},
        "shelf_life_days": {"shelf_life_days"},
    },
    Role.BUSINESS_RULES: {
        "rule_type": {"rule_type", "loai_dieu_kien"},
        "ingredient_name": {"ingredient_name", "ap_dung_cho_nl"},
        "value": {"value", "gia_tri"},
        "unit": {"unit"},
        "currency": {"currency"},
        "effective_from": {"effective_from", "bat_dau"},
        "note": {"note", "ghi_chu"},
    },
}

ROLE_REQUIRED = {
    Role.SALES: {"date", "product_name", "quantity_sold"},
    Role.CALENDAR: {"date"},
    Role.MENU: {"product_id", "product_name"},
    Role.RECIPES: {"product_name", "ingredient_name", "ingredient_quantity", "ingredient_unit", "yield_quantity", "yield_unit", "recipe_version", "effective_from"},
    Role.INGREDIENT_USAGE: {"date", "ingredient_name", "actual_usage_quantity", "unit"},
    Role.INVENTORY_SNAPSHOT: {"snapshot_date", "ingredient_name", "quantity_remaining", "unit", "lot_id"},
    Role.PURCHASE_HISTORY: {"ingredient_name", "quantity", "unit", "received_date"},
    Role.SUPPLIER_RULES: {"supplier_id", "ingredient_name", "pack_size", "unit", "lead_time_days", "unit_price"},
    Role.BUSINESS_RULES: {"rule_type", "value", "unit"},
}


def schema_fingerprint(
    headers: Iterable[str],
    *,
    inferred_types: dict[str, str] | None = None,
    semantic_version: str = "mapping-semantics-v2",
) -> str:
    signature: dict[str, object] = {
        "headers": sorted(normalize_label(header) for header in headers),
        "inferred_types": {
            normalize_label(key): value
            for key, value in sorted((inferred_types or {}).items())
        },
        "semantic_version": semantic_version,
    }
    return hashlib.sha256(json.dumps(signature).encode()).hexdigest()


def region_schema_fingerprint(region: RegionData) -> str:
    unit_signatures = {
        normalize_label(column): sorted(
            str(value).strip().casefold()
            for value in region.frame[column].dropna().unique().tolist()
        )[:50]
        for column in region.frame.columns
        if normalize_label(column) in {
            "unit", "dvt", "dv", "uom", "order_uom", "base_uom",
            "don_vi_nl", "don_vi_sp",
        }
    }
    payload = {
        "base": schema_fingerprint(
            region.frame.columns,
            inferred_types=region.region.profile.inferred_types,
        ),
        "unit_signatures": unit_signatures,
        "row_shape": "empty" if region.frame.empty else "nonempty",
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _matches(headers: list[str], role: Role) -> tuple[dict[str, str], float]:
    normalized: dict[str, str] = {}
    for header in headers:
        # Multilingual exports often join equivalent labels with '/'.  Keep
        # those aliases for role discovery; the same-source identity/name
        # conflict is removed below and product IDs are then resolved against
        # the versioned menu master instead of trusting the header token.
        candidates = [str(header), *str(header).split("/")]
        for candidate in candidates:
            label = normalize_label(candidate)
            if label:
                normalized[label] = header
    mapped: dict[str, str] = {}
    for target, aliases in ALIASES[role].items():
        hits = sorted({normalized[alias] for alias in aliases if alias in normalized})
        if len(hits) == 1:
            mapped[target] = hits[0]
    required = ROLE_REQUIRED[role]
    coverage = len(required & set(mapped)) / max(len(required), 1)
    optional = len(set(mapped) - required) / max(len(ALIASES[role]) - len(required), 1)
    return mapped, min(1.0, 0.85 * coverage + 0.15 * optional)


def infer_mapping(region: RegionData) -> MappingProposal:
    date_columns: list[str] = []
    for header in region.frame.columns:
        try:
            parsed = __import__("pandas").to_datetime(str(header), errors="raise")
            if 2000 <= parsed.year <= 2100:
                date_columns.append(header)
        except Exception:
            pass
    product_columns = [header for header in region.frame.columns if normalize_label(header) in ALIASES[Role.SALES]["product_name"]]
    if len(date_columns) >= 2 and len(product_columns) == 1:
        return MappingProposal(
            region_id=region.region.region_id, role=Role.SALES, role_confidence=0.9,
            field_mappings=[
                FieldMapping(source_column=product_columns[0], target_field="product_name", confidence=0.99,
                             evidence=["wide_table_identifier"]),
                FieldMapping(source_column="__wide_date_columns__", target_field="quantity_sold", confidence=0.95,
                             evidence=["date_headers"], operations=[TransformOperation(operation="unpivot", arguments={"date_columns": date_columns, "target_date": "date", "target_value": "quantity_sold"})]),
            ], unresolved_fields=[x for x in region.frame.columns if x not in date_columns + product_columns],
            issues=[], evidence_references=["wide_date_columns"], needs_review=False, origin="deterministic",
        )
    candidates = []
    for role in ALIASES:
        mapped, score = _matches(list(region.frame.columns), role)
        candidates.append((score, role, mapped))
    score, role, mapped = max(candidates, key=lambda item: item[0])
    if score < 0.52:
        container_label = normalize_label(region.region.container)
        documented_non_data_sheets = {
            "readme",
            "huong_dan",
            "instructions",
            "notes",
            "ghi_chu",
        }
        if container_label in documented_non_data_sheets:
            return MappingProposal(
                region_id=region.region.region_id,
                role=Role.UNKNOWN,
                role_confidence=score,
                unresolved_fields=list(region.frame.columns),
                issues=["AUTO_IGNORED_DOCUMENTATION_SHEET"],
                evidence_references=[
                    *region.region.evidence,
                    f"container={region.region.container}",
                ],
                needs_review=False,
                origin="deterministic",
            )
        return MappingProposal(
            region_id=region.region.region_id,
            role=Role.UNKNOWN,
            role_confidence=score,
            unresolved_fields=list(region.frame.columns),
            issues=["ROLE_UNRESOLVED"],
            evidence_references=region.region.evidence,
            needs_review=True,
            origin="deterministic",
        )
    required_missing = sorted(ROLE_REQUIRED[role] - set(mapped))
    mappings = [
        FieldMapping(
            source_column=source,
            target_field=target,
            confidence=0.99,
            evidence=[f"normalized_alias:{normalize_label(source)}"],
            operations=[TransformOperation(operation="rename", arguments={"to": target})],
        )
        for target, source in sorted(mapped.items())
    ]
    # Never allow one ambiguous source column to become both identity and
    # display name.  A unique menu master is resolved later by EntityRegistry.
    by_source: dict[str, list[FieldMapping]] = {}
    for item in mappings:
        by_source.setdefault(item.source_column, []).append(item)
    for source, items in by_source.items():
        targets = {item.target_field for item in items}
        if {"product_id", "product_name"} <= targets:
            mappings = [
                item
                for item in mappings
                if not (item.source_column == source and item.target_field == "product_id")
            ]
            mapped.pop("product_id", None)
    unresolved = [column for column in region.frame.columns if column not in mapped.values()]
    return MappingProposal(
        region_id=region.region.region_id,
        role=role,
        role_confidence=score,
        field_mappings=mappings,
        unresolved_fields=unresolved,
        issues=[f"MISSING_REQUIRED:{field}" for field in required_missing],
        evidence_references=region.region.evidence,
        needs_review=bool(required_missing) or score < 0.75,
        origin="deterministic",
    )


def proposal_from_profile(region: RegionData, profiles: list[ImportProfile], tenant_id: str | None = None) -> MappingProposal | None:
    from shelfcash_preprocess.models import PROFILE_VERSION

    fingerprint = region_schema_fingerprint(region)
    compatible = [
        profile
        for profile in profiles
        if profile.profile_version == PROFILE_VERSION
        and profile.schema_fingerprint == fingerprint
        and profile.tenant_id == tenant_id
        and profile.verified_by in {"deterministic", "review"}
        and not profile.mapping.needs_review
    ]
    if len(compatible) != 1:
        return None
    proposal = compatible[0].mapping.model_copy(deep=True)
    targets_by_source: dict[str, set[str | None]] = {}
    for item in proposal.field_mappings:
        targets_by_source.setdefault(item.source_column, set()).add(item.target_field)
    if any({"product_id", "product_name"} <= targets for targets in targets_by_source.values()):
        # Explicitly invalidate the v1 semantic bug even if a hand-edited
        # profile carries a v2 version string.
        return None
    proposal.region_id = region.region.region_id
    proposal.origin = "profile"
    return proposal
