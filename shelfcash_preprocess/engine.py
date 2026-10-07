from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from shelfcash_preprocess.pipeline import load_bundle
from shelfcash_preprocess.schema import read_frame


def _require_data_readiness(bundle: Any, capability: str) -> None:
    readiness = bundle.manifest.readiness.get(capability)
    if readiness is None or not readiness.data_validated:
        status = None if readiness is None else readiness.status.value
        issues = [] if readiness is None else readiness.issues
        raise ValueError(
            f"BUNDLE_CAPABILITY_NOT_READY:{capability}:status={status}:issues={issues}"
        )


def load_canonical_frames(
    bundle_dir: str | Path,
    *,
    required_capability: str | None = None,
) -> dict[str, pd.DataFrame]:
    bundle = load_bundle(bundle_dir)
    if required_capability is not None:
        _require_data_readiness(bundle, required_capability)
    frames: dict[str, pd.DataFrame] = {}
    for name, path in bundle.canonical_files.items():
        frames[name] = read_frame(name, path)
    return frames


def _fill_sales_units_from_menu(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Fill only missing sales units from the canonical product master."""

    sales = frames["sales"].copy()
    if "menu" not in frames or "product_id" not in sales.columns:
        return sales
    menu = frames["menu"][["product_id", "unit"]].copy()
    menu["product_id"] = menu["product_id"].astype("string").str.strip()
    menu["unit"] = menu["unit"].astype("string").str.strip()
    usable = menu["product_id"].notna() & menu["product_id"].ne("")
    menu = menu.loc[usable]
    conflicts = menu.groupby("product_id", observed=True)["unit"].nunique(dropna=True)
    if conflicts.gt(1).any():
        raise ValueError(
            "MENU_PRODUCT_UNIT_CONFLICT: one product_id has multiple canonical units"
        )
    lookup = menu.drop_duplicates("product_id").rename(columns={"unit": "_menu_unit"})
    sales = sales.merge(lookup, on="product_id", how="left", validate="many_to_one")
    if "unit" not in sales.columns:
        sales["unit"] = pd.Series(pd.NA, index=sales.index, dtype="string")
    else:
        sales["unit"] = sales["unit"].astype("string")
    missing = sales["unit"].isna() | sales["unit"].str.strip().eq("")
    sales.loc[missing, "unit"] = sales.loc[missing, "_menu_unit"]
    return sales.drop(columns="_menu_unit")


def create_forecast_input_frames(
    bundle_dir: str | Path,
    *,
    include_weather: bool = False,
    as_of_date: date | str | pd.Timestamp | None = None,
) -> dict[str, pd.DataFrame]:
    """Build raw public forecast frames while preserving canonical IDs."""

    bundle = load_bundle(bundle_dir)
    _require_data_readiness(bundle, "forecast_core")
    frames = load_canonical_frames(bundle_dir)
    payload: dict[str, pd.DataFrame] = {}
    if "sales" in frames:
        payload["sales_history"] = _fill_sales_units_from_menu(frames)
    if "calendar" in frames:
        calendar = frames["calendar"]
        stores = calendar["store_id"].dropna().astype(str).unique() if "store_id" in calendar else []
        if len(stores) > 1:
            raise ValueError(
                "MULTI_STORE_CALENDAR_UNSUPPORTED: the engine calendar adapter "
                "cannot safely collapse store-specific rows"
            )
        calendar = calendar.copy()
        cutoff = pd.Timestamp(
            as_of_date or bundle.manifest.context.cutoff_date
        ).normalize() if (as_of_date is not None or bundle.manifest.context.cutoff_date is not None) else None
        if include_weather:
            if cutoff is None:
                raise ValueError("WEATHER_AS_OF_REQUIRED")
            kind = calendar.get("weather_kind", pd.Series(pd.NA, index=calendar.index)).astype("string")
            issued = pd.to_datetime(calendar.get("weather_issued_at"), errors="coerce")
            available = pd.to_datetime(calendar.get("weather_available_at"), errors="coerce")
            target = pd.to_datetime(calendar["date"], errors="coerce")
            valid_weather = (
                kind.str.casefold().eq("forecast")
                & issued.notna() & issued.le(cutoff)
                & available.notna() & available.le(cutoff)
                & target.ge(cutoff)
            )
            calendar.loc[~valid_weather, ["temperature", "rainfall"]] = pd.NA
        dropped = [
            "store_id", "available_as_of", "weather_kind",
            "weather_issued_at", "weather_available_at",
        ]
        if not include_weather:
            dropped.extend(["temperature", "rainfall"])
        payload["calendar_features"] = calendar.drop(columns=dropped, errors="ignore")
    return payload


def create_forecast_input(bundle_dir: str | Path) -> Any:
    """Load the bundle through the forecast engine's real public adapter."""
    from shelfcash_forecast.config import ForecastConfig
    from shelfcash_forecast.data.adapter import adapt_forecast_input

    payload = create_forecast_input_frames(bundle_dir)
    return adapt_forecast_input(payload, ForecastConfig())


def create_recipe_records(bundle_dir: str | Path) -> list[Any]:
    from shelfcash_forecast.bom.adapter import adapt_recipes

    frames = load_canonical_frames(bundle_dir, required_capability="ingredient_demand")
    if "recipes" not in frames:
        raise ValueError("Bundle has no canonical recipes table")
    return adapt_recipes(frames["recipes"])


def create_inventory_lots(bundle_dir: str | Path) -> tuple[list[Any], date]:
    from shelfcash_forecast.inventory.contracts import InventoryLot

    frames = load_canonical_frames(bundle_dir, required_capability="inventory_simulation")
    if "inventory_snapshot" not in frames:
        raise ValueError("Bundle has no canonical inventory snapshot")
    frame = frames["inventory_snapshot"]
    snapshots = pd.to_datetime(frame["snapshot_date"], errors="raise").dt.date.unique()
    if len(snapshots) != 1:
        raise ValueError("Inventory snapshot must have exactly one as-of boundary")
    lots = []
    for row in frame.to_dict(orient="records"):
        def parsed(name: str) -> date | None:
            value = row.get(name)
            return None if pd.isna(value) else pd.Timestamp(value).date()
        lots.append(InventoryLot(
            lot_id=str(row["lot_id"]), store_id=str(row["store_id"]), ingredient_id=str(row["ingredient_id"]),
            quantity_remaining=float(row["quantity_remaining"]), unit=str(row["unit"]),
            received_date=parsed("received_date"), expiry_date=parsed("expiry_date"),
            location=None if pd.isna(row.get("location")) else str(row["location"]),
            source_type="initial_inventory", provenance={"snapshot_date": str(snapshots[0]), "source": "shelfcash_preprocess"},
        ))
    return lots, snapshots[0]


def create_supplier_offers(bundle_dir: str | Path, order_date: date | None = None) -> list[Any]:
    from shelfcash_forecast.optimization.contracts import SupplierOffer

    bundle = load_bundle(bundle_dir)
    frames = load_canonical_frames(bundle_dir, required_capability="procurement_optimization")
    if "supplier_rules" not in frames:
        raise ValueError("Bundle has no canonical supplier rules")
    decision_date = order_date or bundle.manifest.context.cutoff_date
    if decision_date is None:
        raise ValueError("Supplier offers require an explicit order_date or bundle cutoff_date")
    store = bundle.manifest.context.store_id
    if not store:
        raise ValueError("Supplier offers require store_id")
    offers = []
    for index, row in enumerate(frames["supplier_rules"].to_dict(orient="records"), 1):
        pack_size = float(row["pack_size"])
        raw_moq = float(row.get("minimum_order_quantity") or 0)
        order_unit = str(row.get("order_unit") or "")
        base_unit = str(row["unit"])
        moq_basis = "pack_count" if order_unit == "pack" else "base_quantity"
        minimum_quantity = raw_moq * pack_size if moq_basis == "pack_count" else raw_moq
        price_basis = str(row.get("price_basis") or "")
        if price_basis not in {"base_unit", "pack"}:
            raise ValueError(f"SUPPLIER_PRICE_BASIS_UNSUPPORTED:{price_basis}")
        source_price = float(row["unit_price"])
        normalized_price = source_price / pack_size if price_basis == "pack" else source_price
        shelf = row.get("shelf_life_days")
        offers.append(SupplierOffer(
            offer_id=f"{row['supplier_id']}|{row['ingredient_id']}|{decision_date}|{index}",
            supplier_id=str(row["supplier_id"]), store_id=store, ingredient_id=str(row["ingredient_id"]),
            unit=base_unit, order_date=decision_date, pack_size=pack_size,
            unit_price=normalized_price, minimum_order_quantity=minimum_quantity,
            lead_time_days=int(float(row["lead_time_days"])),
            shelf_life_days=None if pd.isna(shelf) else int(float(shelf)),
            moq_basis=moq_basis,
            price_basis="per_base_unit",
            source_terms={
                "minimum_order_quantity": raw_moq,
                "order_unit": order_unit,
                "unit_price": source_price,
                "price_basis": price_basis,
                "delivery_schedule": None if pd.isna(row.get("delivery_schedule")) else str(row["delivery_schedule"]),
                "ingredient_name": str(row["ingredient_name"]),
                "calendar_confirmation_required": True,
                "source_rule_id": f"{row['supplier_id']}|{row['ingredient_id']}",
                "canonical_row": index + 1,
                "source": str(bundle.canonical_files["supplier_rules"]),
                "delivery_cost_status": "NOT_SUPPLIED",
                "price_basis_confirmation_required": True,
            },
        ))
    return offers


def create_optimization_request(
    bundle_dir: str | Path,
    *,
    planning_end_date: date,
    demand_scenarios: Iterable[Any],
    strategy_profiles: Iterable[Any] = (),
    cost_assumptions: Iterable[Any] = (),
    budget: float | None = None,
    existing_inbound: Iterable[Any] = (),
    planning_config: Any = None,
    stochastic: bool = False,
    seed: int = 42,
    execution_mode: str = "demo",
) -> Any:
    """Build a real M5 contract only from caller-supplied planning assumptions."""
    from shelfcash_forecast.inventory.contracts import InventorySimulationPolicy
    from shelfcash_forecast.optimization.contracts import OptimizationRequest
    from shelfcash_forecast.optimization.contracts import ProcurementDiagnostic

    bundle = load_bundle(bundle_dir)
    decision_date = bundle.manifest.context.cutoff_date
    if decision_date is None:
        raise ValueError("Optimization requires bundle cutoff_date")
    lots, snapshot = create_inventory_lots(bundle_dir)
    if snapshot != decision_date:
        raise ValueError(
            f"INVENTORY_SNAPSHOT_BOUNDARY_MISMATCH:snapshot={snapshot}:decision={decision_date}"
        )
    policy_name = bundle.manifest.context.metadata.get("unknown_expiry_policy", "reject")
    if planning_config is not None:
        from shelfcash_forecast.optimization.planning_service import prepare_bundle_requests
        mode = "stochastic" if stochastic else "deterministic"
        requests, _, _ = prepare_bundle_requests(bundle_dir, planning=planning_config,
            lots=lots, snapshot=snapshot, policy=InventorySimulationPolicy(unknown_expiry=policy_name),
            scenarios=list(demand_scenarios), decision_date=decision_date,
            planning_end_date=planning_end_date, seed=seed, optimization_mode=mode,
            execution_mode=execution_mode)
        data = requests[mode].model_dump()
        data["existing_inbound"] = list(existing_inbound)
        if list(strategy_profiles) or list(cost_assumptions) or budget is not None:
            raise ValueError("DIRECT_API_CONFIG_CONFLICT:use versioned planning_config controls")
        return OptimizationRequest.model_validate(data)
    return OptimizationRequest(
        request_id=f"preprocess-{bundle.manifest.run_id}", decision_date=decision_date,
        planning_end_date=planning_end_date, initial_inventory=lots,
        demand_scenarios=list(demand_scenarios), supplier_offers=create_supplier_offers(bundle_dir, decision_date),
        existing_inbound=list(existing_inbound), cost_assumptions=list(cost_assumptions),
        strategy_profiles=list(strategy_profiles), budget=budget,
        inventory_policy=InventorySimulationPolicy(unknown_expiry=policy_name),
        inventory_snapshot_date=snapshot,
        inventory_snapshot_boundary="EOD",
        blocked_issues=[ProcurementDiagnostic(reason_code="VERSIONED_PLANNING_CONFIG_REQUIRED",proof_status="BLOCKED",
            details={"field":"planning_config","legacy_api":"caller costs alone cannot resolve staged business rules or calendar"},
            action_required=["PROVIDE_VERSIONED_PLANNING_CONFIG_WITH_EXPLICIT_OR_UNRESOLVED_SEMANTICS"])],
    )


def validate_bundle(bundle_dir: str | Path) -> dict[str, Any]:
    bundle = load_bundle(bundle_dir)
    frames = load_canonical_frames(bundle_dir)
    row_counts = {name: len(frame) for name, frame in frames.items()}
    return {"status": "VALID", "integrity_valid": True, "schema_valid": True,
            "runtime_executed": False,
            "run_id": bundle.manifest.run_id, "schema_version": bundle.manifest.schema_version,
            "canonical_tables": sorted(frames), "row_counts": row_counts,
            "readiness": {key: value.status.value for key, value in bundle.manifest.readiness.items()}}


def engine_smoke(bundle_dir: str | Path, artifact_dir: str | Path | None = None) -> dict[str, Any]:
    """Call real M1/M3/M4/M5 adapters/contracts; never duplicate their validation."""
    report: dict[str, Any] = {"checks": {}, "artifact_dir": str(artifact_dir) if artifact_dir else None}
    try:
        forecast_input = create_forecast_input(bundle_dir)
        from shelfcash_forecast.data.validator import validate_calendar, validate_sales
        sales, quality = validate_sales(forecast_input.sales_history)
        calendar = validate_calendar(forecast_input.calendar_features, quality)
        report["checks"]["M1_M2_INPUT"] = {"status": "PASS", "sales_rows": len(sales),
                                                   "calendar_rows": 0 if calendar is None else len(calendar),
                                                   "quality": quality.to_dict()}
    except Exception as exc:
        report["checks"]["M1_M2_INPUT"] = {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}
    try:
        recipes = create_recipe_records(bundle_dir)
        report["checks"]["M3_RECIPES"] = {"status": "PASS", "records": len(recipes)}
    except Exception as exc:
        report["checks"]["M3_RECIPES"] = {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}
    try:
        lots, snapshot = create_inventory_lots(bundle_dir)
        from shelfcash_forecast.inventory.fefo import fefo_sort_key
        sorted(lots, key=fefo_sort_key)
        report["checks"]["M4_INITIAL_LOTS"] = {"status": "PASS", "records": len(lots), "snapshot": str(snapshot)}
    except Exception as exc:
        report["checks"]["M4_INITIAL_LOTS"] = {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}
    try:
        offers = create_supplier_offers(bundle_dir)
        report["checks"]["M5_SUPPLIER_OFFERS"] = {"status": "PASS", "records": len(offers),
                                                        "optimization_request": "NOT_BUILT_MISSING_DEMAND_STRATEGIES_COSTS"}
    except Exception as exc:
        report["checks"]["M5_SUPPLIER_OFFERS"] = {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}
    report["checks"]["M6_INPUT"] = {"status": "BLOCKED", "reason": "Requires evaluated M5 output; preprocess does not fabricate it."}
    report["status"] = "PASS_WITH_BLOCKERS" if all(x["status"] == "PASS" for key, x in report["checks"].items() if key != "M6_INPUT") else "FAIL"
    if artifact_dir is not None:
        target = Path(artifact_dir)
        target.mkdir(parents=True, exist_ok=True)
        (target / "engine_smoke_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return report
