from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd

from shelfcash_preprocess.discovery import RegionData
from shelfcash_preprocess.mapping import normalize_label
from shelfcash_preprocess.models import Issue, IssueScope, MappingProposal, Role, RunContext, Severity
from shelfcash_preprocess.values import AmbiguousValue, normalize_unit, parse_boolean, parse_date, parse_number


def _stable_id(prefix: str, text: object) -> str:
    digest = hashlib.sha256(_entity_key(text).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _entity_key(value: object) -> str:
    """Conservative exact-name key; punctuation remains identity-significant."""

    text = unicodedata.normalize("NFKC", str(value)).strip().casefold()
    return re.sub(r"\s+", " ", text)


def _none(value: Any) -> Any:
    return None if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)) else value


def _text(value: Any) -> str | None:
    value = _none(value)
    result = "" if value is None else str(value).strip()
    return result or None


@dataclass
class CanonicalResult:
    tables: dict[str, pd.DataFrame]
    lineage: list[dict[str, Any]]
    issues: list[Issue]
    quarantine: list[dict[str, Any]] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


class EntityRegistry:
    """One registry shared by sales/menu/recipe/inventory/supplier data."""

    def __init__(self) -> None:
        self.products: dict[str, str] = {}
        self.ingredients: dict[str, str] = {}
        self.conflicts: list[str] = []

    def _add(self, collection: dict[str, str], prefix: str, name: Any, identifier: Any = None) -> str:
        key = _entity_key(name)
        value = _text(identifier) or _stable_id(prefix, name)
        previous = collection.get(key)
        if previous and previous != value:
            self.conflicts.append(f"{prefix}_ID_CONFLICT:{name}:{previous}:{value}")
        else:
            collection[key] = value
        return collection.get(key, value)

    def add_product(self, name: Any, identifier: Any = None) -> str:
        return self._add(self.products, "PRD", name, identifier)

    def product(self, name: Any) -> str:
        return self.products.get(_entity_key(name)) or self.add_product(name)

    def add_ingredient(self, name: Any, identifier: Any = None) -> str:
        return self._add(self.ingredients, "ING", name, identifier)

    def ingredient(self, name: Any) -> str:
        return self.ingredients.get(_entity_key(name)) or self.add_ingredient(name)


class CanonicalTransformer:
    def __init__(self, context: RunContext):
        self.context = context
        self.issues: list[Issue] = []
        self.lineage: list[dict[str, Any]] = []
        self.quarantine: list[dict[str, Any]] = []
        self.registry = EntityRegistry()
        self._issue_counter = 0
        self._active_role: Role | None = None

    @staticmethod
    def _role_capabilities(role: Role | None) -> list[str]:
        return {
            Role.SALES: ["forecast_core", "ingredient_demand"],
            Role.CALENDAR: ["forecast_core"],
            Role.MENU: ["forecast_core", "ingredient_demand"],
            Role.RECIPES: ["ingredient_demand", "inventory_simulation", "procurement_optimization"],
            Role.INGREDIENT_USAGE: ["inventory_simulation", "procurement_optimization"],
            Role.INVENTORY_SNAPSHOT: ["inventory_simulation", "procurement_optimization"],
            Role.PURCHASE_HISTORY: ["inventory_simulation", "procurement_optimization"],
            Role.SUPPLIER_RULES: ["procurement_optimization"],
            Role.BUSINESS_RULES: ["procurement_optimization"],
        }.get(role, [
            "forecast_core", "ingredient_demand", "inventory_simulation",
            "procurement_optimization", "decision_intelligence_input",
        ])

    def issue(self, code: str, severity: Severity, message: str, region: RegionData | None = None,
              *, row: int | None = None, capability: str | None = None,
              details: dict[str, Any] | None = None, suggested_action: str | None = None,
              field: str | None = None, scope: IssueScope | None = None) -> None:
        self._issue_counter += 1
        locator: dict[str, Any] = {}
        if region:
            locator = {"source_path": region.region.source_path, "container": region.region.container}
            if row is not None:
                locator["row"] = row
        issue_id = f"issue_{self._issue_counter:04d}"
        affected = [capability] if capability else self._role_capabilities(self._active_role)
        self.issues.append(Issue(
            issue_id=issue_id, code=code, severity=severity,
            message=message, region_id=region.region.region_id if region else None,
            source_locator=locator, details=details or {}, suggested_action=suggested_action,
            scope=scope or (IssueScope.FIELD if field else IssueScope.TABLE if region else IssueScope.GLOBAL),
            table=self._active_role.value if self._active_role else None,
            field=field,
            affected_capabilities=affected,
            capability=capability,
        ))
        if region is not None and row is not None and severity in {Severity.ERROR, Severity.BLOCKING}:
            try:
                position = region.source_rows.index(row)
                raw = {
                    str(key): (None if _none(value) is None else str(value))
                    for key, value in region.frame.iloc[position].to_dict().items()
                }
            except (ValueError, IndexError):
                raw = {}
            self.quarantine.append({
                "issue_id": issue_id,
                "reason_code": code,
                "field": field or (details or {}).get("field"),
                "source_locator": locator,
                "raw_row": raw,
                "action": "excluded_pending_review",
                "review_decision_references": list(
                    self.context.metadata.get("review_decision_ids", [])
                ),
            })

    def _emit(
        self,
        records: list[dict[str, Any]],
        data: dict[str, Any],
        region: RegionData,
        source_row: int,
        *,
        source_columns: list[str] | None = None,
        contributors: list[dict[str, Any]] | None = None,
    ) -> None:
        data["__lineage__"] = {
            "row_locator": source_row,
            "column_locators": source_columns or [],
            "contributors": contributors or [],
        }
        records.append(data)

    @staticmethod
    def _mapping(proposal: MappingProposal) -> dict[str, str]:
        return {item.target_field: item.source_column for item in proposal.field_mappings if item.target_field}

    @staticmethod
    def _value(row: pd.Series, mapping: dict[str, str], field: str) -> Any:
        column = mapping.get(field)
        return None if column is None else _none(row.get(column))

    def _date(self, value: Any, region: RegionData, row: int, field: str) -> date | None:
        if _none(value) is None:
            return None
        try:
            return parse_date(value, self.context.date_locale)
        except AmbiguousValue as exc:
            self.issue("AMBIGUOUS_DATE", Severity.BLOCKING, str(exc), region, row=row,
                       details={"field": field, "raw_value": str(value)},
                       suggested_action="Choose DMY, MDY, or YMD with select_date_locale.", field=field)
        except (TypeError, ValueError) as exc:
            self.issue("INVALID_DATE", Severity.ERROR, str(exc), region, row=row,
                       details={"field": field, "raw_value": str(value)}, field=field)
        return None

    def _number(self, value: Any, region: RegionData, row: int, field: str) -> float | None:
        if _none(value) is None:
            return None
        try:
            return parse_number(value)
        except AmbiguousValue as exc:
            self.issue("AMBIGUOUS_NUMBER", Severity.BLOCKING, str(exc), region, row=row,
                       details={"field": field, "raw_value": str(value)}, field=field)
        except (TypeError, ValueError) as exc:
            self.issue("INVALID_NUMBER", Severity.ERROR, str(exc), region, row=row,
                       details={"field": field, "raw_value": str(value)}, field=field)
        return None

    def transform(self, regions: list[RegionData], proposals: list[MappingProposal]) -> CanonicalResult:
        by_id = {item.region.region_id: item for item in regions}
        work = [
            (by_id[p.region_id], p)
            for p in proposals
            if p.region_id in by_id and p.role != Role.UNKNOWN and not p.needs_review
        ]
        for override in self.context.metadata.get("entity_overrides", []):
            if override.get("entity_type") == "product":
                self.registry.add_product(override.get("source_name"), override.get("target_id"))
            elif override.get("entity_type") == "ingredient":
                self.registry.add_ingredient(override.get("source_name"), override.get("target_id"))
            else:
                self.issue("INVALID_ENTITY_OVERRIDE", Severity.BLOCKING,
                           "entity_type must be product or ingredient", details={"override": override})
        for region, proposal in work:
            self._active_role = proposal.role
            mapping = self._mapping(proposal)
            if proposal.role == Role.MENU:
                for _, row in region.frame.iterrows():
                    name = self._value(row, mapping, "product_name")
                    if _text(name):
                        self.registry.add_product(name, self._value(row, mapping, "product_id"))
        for region, proposal in work:
            self._active_role = proposal.role
            mapping = self._mapping(proposal)
            for _, row in region.frame.iterrows():
                name = self._value(row, mapping, "ingredient_name")
                if _text(name):
                    self.registry.add_ingredient(name, self._value(row, mapping, "ingredient_id"))

        output: dict[str, list[dict[str, Any]]] = {role.value: [] for role in Role if role != Role.UNKNOWN}
        for region, proposal in work:
            self._active_role = proposal.role
            records = getattr(self, f"_transform_{proposal.role.value}")(region, self._mapping(proposal))
            start = len(output[proposal.role.value])
            clean_records: list[dict[str, Any]] = []
            for offset, record in enumerate(records):
                lineage = record.pop("__lineage__")
                clean_records.append(record)
                self.lineage.append({
                    "canonical_table": proposal.role.value, "canonical_row": start + offset,
                    "source_id": region.region.source_id,
                    "source_file_sha256": region.region.metadata.get("source_sha256"),
                    "source_path": region.region.source_path,
                    "container": region.region.container, "table_id": region.region.region_id,
                    "row_locator": lineage["row_locator"],
                    "column_locators": lineage["column_locators"],
                    "contributors": lineage["contributors"],
                    "transformation_ids": ["mapping-v2", "canonical-v2"],
                    "review_decision_references": list(
                        self.context.metadata.get("review_decision_ids", [])
                    ),
                })
            output[proposal.role.value].extend(clean_records)
        self._active_role = None
        for conflict in self.registry.conflicts:
            self.issue("ENTITY_CONFLICT", Severity.BLOCKING, conflict)
        tables = {name: pd.DataFrame(rows) for name, rows in output.items() if rows}
        self._cross_checks(tables)
        return CanonicalResult(
            tables=tables,
            lineage=self.lineage,
            issues=self.issues,
            quarantine=self.quarantine,
            stats={
                "rows": {name: len(frame) for name, frame in tables.items()},
                "quarantined_rows": len(self.quarantine),
            },
        )

    def _transform_sales(self, region: RegionData, m: dict[str, str]) -> list[dict[str, Any]]:
        records = []
        if not self.context.store_id and "store_id" not in m:
            self.issue("STORE_CONTEXT_REQUIRED", Severity.BLOCKING,
                       "Sales has no store_id and the run has no --store-id.", region,
                       capability="forecast_core", suggested_action="Rerun with --store-id or set store metadata in review.")
        working = region.frame.copy()
        working["__source_row"] = region.source_rows
        if m.get("quantity_sold") == "__wide_date_columns__":
            id_columns = [column for column in working.columns if column == m.get("product_name") or column == m.get("store_id")]
            date_columns = [column for column in working.columns if column not in id_columns + ["__source_row"]]
            working = working.melt(id_vars=id_columns + ["__source_row"], value_vars=date_columns, var_name="__wide_date", value_name="__wide_value")
            m = dict(m); m["date"] = "__wide_date"; m["quantity_sold"] = "__wide_value"
        for _, row in working.iterrows():
            sr = int(row["__source_row"])
            name = _text(self._value(row, m, "product_name"))
            dt = self._date(self._value(row, m, "date"), region, sr, "date")
            qty = self._number(self._value(row, m, "quantity_sold"), region, sr, "quantity_sold")
            store = _text(self._value(row, m, "store_id")) or self.context.store_id
            if name is None or dt is None or qty is None or store is None:
                continue
            if qty < 0:
                self.issue("RETURN_OR_CANCELLATION", Severity.WARNING,
                           "Negative event kept in raw/staging and excluded from non-negative forecast input.",
                           region, row=sr, capability="forecast_core")
                continue
            price = self._number(self._value(row, m, "selling_price"), region, sr, "selling_price")
            revenue = self._number(self._value(row, m, "revenue"), region, sr, "revenue")
            if price is not None and revenue is not None and abs(revenue - qty * price) > max(1.0, abs(revenue) * 1e-6):
                self.issue("REVENUE_MISMATCH", Severity.WARNING,
                           "Revenue differs from quantity × price; both source values were preserved.", region, row=sr)
            raw_stockout = self._value(row, m, "is_stockout")
            try:
                stockout = parse_boolean(raw_stockout) if _none(raw_stockout) is not None else None
            except ValueError:
                stockout = None
                self.issue("INVALID_BOOLEAN", Severity.ERROR, "Invalid stockout value.", region, row=sr)
            self._emit(records, {
                "date": dt, "store_id": store,
                "product_id": _text(self._value(row, m, "product_id")) or self.registry.product(name),
                "product_name": name, "quantity_sold": qty,
                "unit": normalize_unit(self._value(row, m, "unit")),
                "selling_price": price, "revenue": revenue, "is_stockout": stockout,
                "promotion_name": _text(self._value(row, m, "promotion_name")),
            }, region, sr, source_columns=[
                column for column in m.values() if column in region.frame.columns
            ] + ([str(row.get("__wide_date"))] if "__wide_date" in row else []))
        frame = pd.DataFrame([{k: v for k, v in item.items() if k != "__lineage__"} for item in records])
        if not frame.empty:
            duplicate_rows = int(frame.duplicated(["date", "store_id", "product_id"], keep=False).sum())
            if duplicate_rows and "transaction_id" not in m:
                self.issue("DAILY_GRAIN_OVERLAP", Severity.BLOCKING,
                           "Repeated daily product grain has no transaction key; overlap semantics need review.",
                           region, capability="forecast_core", details={"rows": duplicate_rows},
                           suggested_action="Set append/upsert/replace only after checking export overlap.")
        return records

    def _transform_calendar(self, region: RegionData, m: dict[str, str]) -> list[dict[str, Any]]:
        records = []
        for pos, (_, row) in enumerate(region.frame.iterrows()):
            sr = region.source_rows[pos]
            dt = self._date(self._value(row, m, "date"), region, sr, "date")
            if not dt:
                continue
            item: dict[str, Any] = {"date": dt, "store_id": _text(self._value(row, m, "store_id"))}
            for field in ("is_weekend", "is_holiday", "is_store_closed", "is_promotion"):
                raw = self._value(row, m, field)
                try:
                    item[field] = parse_boolean(raw) if _none(raw) is not None else None
                except ValueError:
                    item[field] = None
                    self.issue("INVALID_BOOLEAN", Severity.ERROR, f"Invalid {field} value.", region, row=sr)
            item.update({
                "promotion_name": _text(self._value(row, m, "promotion_name")),
                "temperature": self._number(self._value(row, m, "temperature"), region, sr, "temperature"),
                "rainfall": self._number(self._value(row, m, "rainfall"), region, sr, "rainfall"),
                # Availability is source evidence, never inferred from the run
                # cutoff.  Weather without issued/available provenance is
                # conservatively excluded by the forecast adapter.
                "available_as_of": self._date(self._value(row, m, "available_as_of"), region, sr, "available_as_of"),
                "weather_kind": _text(self._value(row, m, "weather_kind")),
                "weather_issued_at": self._date(self._value(row, m, "weather_issued_at"), region, sr, "weather_issued_at"),
                "weather_available_at": self._date(self._value(row, m, "weather_available_at"), region, sr, "weather_available_at"),
            })
            self._emit(records, item, region, sr, source_columns=list(m.values()))
        return records

    def _transform_menu(self, region: RegionData, m: dict[str, str]) -> list[dict[str, Any]]:
        records = []
        for pos, (_, row) in enumerate(region.frame.iterrows()):
            sr = region.source_rows[pos]
            name = _text(self._value(row, m, "product_name"))
            if name:
                self._emit(records, {
                    "product_id": self.registry.add_product(name, self._value(row, m, "product_id")),
                    "product_name": name, "product_type": _text(self._value(row, m, "product_type")),
                    "components": _text(self._value(row, m, "components")),
                    "unit": normalize_unit(self._value(row, m, "unit")),
                    "selling_price": self._number(self._value(row, m, "selling_price"), region, sr, "selling_price"),
                    "status": _text(self._value(row, m, "status")),
                }, region, sr, source_columns=list(m.values()))
        combo_active = any(normalize_label(x.get("product_type")) == "combo" and normalize_label(x.get("status")) in {"dang_ban", "active"} for x in records)
        if combo_active:
            context_text = " ".join(
                str(value) for row in region.region.metadata.get("context_rows", []) for value in row if value
            )
            if "tạm dừng" in context_text.casefold() or "tam_dung" in normalize_label(context_text):
                self.issue("MENU_STATUS_CONFLICT", Severity.WARNING,
                           "The sheet note says combos are paused while row status says active.",
                           region, capability="ingredient_demand",
                           suggested_action="Confirm the authoritative combo status in review.")
            self.issue("MENU_COMBO_RECIPE_COVERAGE", Severity.WARNING,
                       "Active combo rows are kept in master; combos without observed sales/recipes are not synthesized.",
                       region, capability="ingredient_demand")
        return records

    def _transform_recipes(self, region: RegionData, m: dict[str, str]) -> list[dict[str, Any]]:
        records = []
        for pos, (_, row) in enumerate(region.frame.iterrows()):
            sr = region.source_rows[pos]
            product, ingredient = _text(self._value(row, m, "product_name")), _text(self._value(row, m, "ingredient_name"))
            version = _text(self._value(row, m, "recipe_version"))
            start = self._date(self._value(row, m, "effective_from"), region, sr, "effective_from")
            qty = self._number(self._value(row, m, "ingredient_quantity"), region, sr, "ingredient_quantity")
            yqty = self._number(self._value(row, m, "yield_quantity"), region, sr, "yield_quantity")
            if not product or not ingredient or not version or not start or qty is None or yqty is None:
                continue
            product_id, ingredient_id = self.registry.product(product), self.registry.ingredient(ingredient)
            self._emit(records, {
                "recipe_id": _text(self._value(row, m, "recipe_id")) or _stable_id("RCP", f"{product_id}|{version}|{start}"),
                "product_id": product_id, "product_name": product, "ingredient_id": ingredient_id,
                "ingredient_name": ingredient, "ingredient_quantity": qty,
                "ingredient_unit": normalize_unit(self._value(row, m, "ingredient_unit")),
                "yield_quantity": yqty, "yield_unit": normalize_unit(self._value(row, m, "yield_unit")),
                "process_loss_rate": self._number(self._value(row, m, "process_loss_rate"), region, sr, "process_loss_rate") or 0.0,
                "waste_allowance_rate": self._number(self._value(row, m, "waste_allowance_rate"), region, sr, "waste_allowance_rate") or 0.0,
                "recipe_version": version, "effective_from": start,
                "effective_to": self._date(self._value(row, m, "effective_to"), region, sr, "effective_to"),
            }, region, sr, source_columns=list(m.values()))
        return records

    def _transform_ingredient_usage(self, region: RegionData, m: dict[str, str]) -> list[dict[str, Any]]:
        records = []
        if not self.context.store_id and "store_id" not in m:
            self.issue("STORE_CONTEXT_REQUIRED", Severity.BLOCKING, "Ingredient usage needs a store context.", region, capability="ingredient_demand")
        for pos, (_, row) in enumerate(region.frame.iterrows()):
            sr = region.source_rows[pos]
            name = _text(self._value(row, m, "ingredient_name")); dt = self._date(self._value(row, m, "date"), region, sr, "date")
            qty = self._number(self._value(row, m, "actual_usage_quantity"), region, sr, "actual_usage_quantity")
            store = _text(self._value(row, m, "store_id")) or self.context.store_id
            if name and dt and qty is not None and store:
                self._emit(records, {
                    "date": dt, "store_id": store,
                    "ingredient_id": self.registry.ingredient(name),
                    "ingredient_name": name, "actual_usage_quantity": qty,
                    "unit": normalize_unit(self._value(row, m, "unit")),
                    "waste_quantity": self._number(self._value(row, m, "waste_quantity"), region, sr, "waste_quantity"),
                    "source": _text(self._value(row, m, "source")),
                }, region, sr, source_columns=list(m.values()))
        return records

    def _transform_inventory_snapshot(self, region: RegionData, m: dict[str, str]) -> list[dict[str, Any]]:
        records = []
        if not self.context.store_id and "store_id" not in m:
            self.issue("STORE_CONTEXT_REQUIRED", Severity.BLOCKING, "Inventory snapshot needs a store context.", region, capability="inventory_simulation")
        for pos, (_, row) in enumerate(region.frame.iterrows()):
            sr = region.source_rows[pos]
            name = _text(self._value(row, m, "ingredient_name")); snap = self._date(self._value(row, m, "snapshot_date"), region, sr, "snapshot_date")
            qty = self._number(self._value(row, m, "quantity_remaining"), region, sr, "quantity_remaining")
            store = _text(self._value(row, m, "store_id")) or self.context.store_id
            if name and snap and qty is not None and store:
                self._emit(records, {
                    "snapshot_date": snap, "store_id": store,
                    "ingredient_id": self.registry.ingredient(name),
                    "ingredient_name": name, "quantity_remaining": qty,
                    "unit": normalize_unit(self._value(row, m, "unit")),
                    "expiry_date": self._date(self._value(row, m, "expiry_date"), region, sr, "expiry_date"),
                    "lot_id": _text(self._value(row, m, "lot_id")),
                    "received_date": self._date(self._value(row, m, "received_date"), region, sr, "received_date"),
                    "location": _text(self._value(row, m, "location")),
                    "source_type": "initial_inventory",
                }, region, sr, source_columns=list(m.values()))
        return records

    def _transform_purchase_history(self, region: RegionData, m: dict[str, str]) -> list[dict[str, Any]]:
        records = []
        for pos, (_, row) in enumerate(region.frame.iterrows()):
            sr = region.source_rows[pos]
            name = _text(self._value(row, m, "ingredient_name")); received = self._date(self._value(row, m, "received_date"), region, sr, "received_date")
            qty = self._number(self._value(row, m, "quantity"), region, sr, "quantity")
            if name and received and qty is not None:
                self._emit(records, {
                    "document_date": self._date(self._value(row, m, "document_date"), region, sr, "document_date"),
                    "received_date": received,
                    "ingredient_id": self.registry.ingredient(name), "ingredient_name": name,
                    "quantity": qty, "unit": normalize_unit(self._value(row, m, "unit")),
                    "unit_price": self._number(self._value(row, m, "unit_price"), region, sr, "unit_price"),
                    "line_amount": self._number(self._value(row, m, "line_amount"), region, sr, "line_amount"),
                    "supplier_id": _text(self._value(row, m, "supplier_id")),
                    "expiry_date": self._date(self._value(row, m, "expiry_date"), region, sr, "expiry_date"),
                    "lot_id": _text(self._value(row, m, "lot_id")),
                    "purchase_order_id": _text(self._value(row, m, "purchase_order_id")),
                    "simulation_policy": "record_only_historical",
                }, region, sr, source_columns=list(m.values()))
        return records

    def _transform_supplier_rules(self, region: RegionData, m: dict[str, str]) -> list[dict[str, Any]]:
        records = []
        for pos, (_, row) in enumerate(region.frame.iterrows()):
            sr = region.source_rows[pos]; name = _text(self._value(row, m, "ingredient_name"))
            if name:
                self._emit(records, {
                    "supplier_id": _text(self._value(row, m, "supplier_id")),
                    "ingredient_id": self.registry.ingredient(name), "ingredient_name": name,
                    "minimum_order_quantity": self._number(self._value(row, m, "minimum_order_quantity"), region, sr, "minimum_order_quantity"),
                    "order_unit": normalize_unit(self._value(row, m, "order_unit")),
                    "pack_size": self._number(self._value(row, m, "pack_size"), region, sr, "pack_size"),
                    "unit": normalize_unit(self._value(row, m, "unit")),
                    "lead_time_days": self._number(self._value(row, m, "lead_time_days"), region, sr, "lead_time_days"),
                    "unit_price": self._number(self._value(row, m, "unit_price"), region, sr, "unit_price"),
                    "price_basis": "base_unit",
                    "delivery_schedule": _text(self._value(row, m, "delivery_schedule")),
                    "shelf_life_days": self._number(self._value(row, m, "shelf_life_days"), region, sr, "shelf_life_days"),
                }, region, sr, source_columns=list(m.values()))
        return records

    def _transform_business_rules(self, region: RegionData, m: dict[str, str]) -> list[dict[str, Any]]:
        records = []
        for pos, (_, row) in enumerate(region.frame.iterrows()):
            sr = region.source_rows[pos]; ingredient = _text(self._value(row, m, "ingredient_name"))
            self._emit(records, {
                "rule_type": _text(self._value(row, m, "rule_type")),
                "ingredient_id": self.registry.ingredient(ingredient) if ingredient else None,
                "ingredient_name": ingredient,
                "value": self._number(self._value(row, m, "value"), region, sr, "value"),
                "unit": normalize_unit(self._value(row, m, "unit")),
                "currency": _text(self._value(row, m, "currency")),
                "effective_from": self._date(self._value(row, m, "effective_from"), region, sr, "effective_from"),
                "note": _text(self._value(row, m, "note")),
                "application_status": "staged_requires_engine_equivalence_check",
            }, region, sr, source_columns=list(m.values()))
        return records

    def _cross_checks(self, tables: dict[str, pd.DataFrame]) -> None:
        inventory, purchases = tables.get("inventory_snapshot"), tables.get("purchase_history")
        if inventory is not None and purchases is not None and not inventory.empty and not purchases.empty:
            known = purchases.dropna(subset=["lot_id", "received_date"])
            known = known[~known.duplicated("lot_id", keep=False)]
            receipt_by_lot = dict(zip(known["lot_id"], known["received_date"], strict=False))
            for index, row in inventory.iterrows():
                if pd.isna(row.get("received_date")) and row.get("lot_id") in receipt_by_lot:
                    inventory.at[index, "received_date"] = receipt_by_lot[row["lot_id"]]
            missing = int(inventory["received_date"].isna().sum())
            if missing:
                self.issue("INITIAL_RECEIVED_DATE_UNKNOWN", Severity.WARNING,
                           f"{missing} initial lots keep unknown physical received_date; snapshot boundary is authoritative.",
                           capability="inventory_simulation")
            invalid = inventory[inventory["received_date"].notna() & (pd.to_datetime(inventory["received_date"]) > pd.to_datetime(inventory["snapshot_date"]))]
            if len(invalid):
                self.issue("INVENTORY_CHRONOLOGY", Severity.BLOCKING, "Known received_date is after snapshot boundary.",
                           capability="inventory_simulation", details={"rows": int(len(invalid))})
        sales, recipes = tables.get("sales"), tables.get("recipes")
        if sales is not None and not sales.empty:
            duplicate_rows = int(sales.duplicated(["date", "store_id", "product_id"], keep=False).sum())
            if duplicate_rows:
                self.issue("CROSS_EXPORT_OVERLAP", Severity.BLOCKING,
                           "Multiple source regions overlap at daily product grain without a proven transaction key.",
                           capability="forecast_core", details={"rows": duplicate_rows},
                           suggested_action="Review source precedence and import semantics; do not sum overlapping summary/detail exports.")
        if sales is not None and recipes is not None:
            missing = sorted(set(sales["product_id"]) - set(recipes["product_id"]))
            if missing:
                self.issue("MISSING_RECIPE_FOR_SOLD_PRODUCT", Severity.BLOCKING,
                           "Observed products are missing recipes.", capability="ingredient_demand",
                           details={"product_ids": missing})
