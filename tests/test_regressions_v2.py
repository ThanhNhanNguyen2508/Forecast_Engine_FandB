from __future__ import annotations

from datetime import date
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import sqlite3
from types import SimpleNamespace
import zipfile

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from shelfcash_forecast.calibration.cqr import conformal_quantile
from shelfcash_forecast.cli import app as forecast_cli
from shelfcash_preprocess.cli import app as preprocess_cli
from shelfcash_forecast.features.future import add_calendar_future_features
from shelfcash_forecast.inventory.contracts import (
    InventoryDemandLine,
    InventoryDemandScenario,
    InventoryLot,
)
from shelfcash_forecast.optimization.contracts import (
    OptimizationRequest,
    SupplierOffer,
)
from shelfcash_forecast.optimization.optimizer import optimize_procurement
from shelfcash_forecast.exceptions import OptimizationNotAvailableError
from shelfcash_forecast.decision_intelligence.grounding import GroundingError
from shelfcash_forecast.decision_intelligence.openai_generator import (
    M6LLMError,
    OpenAIGroundedGenerator,
    _GeneratedClaim,
    _GeneratedExplanation,
)
from shelfcash_forecast.decision_intelligence.service import (
    build_final_decision_package,
    explain_decision,
)
from shelfcash_forecast.scenario.residuals import validate_residual_history
from shelfcash_preprocess.discovery import RegionData
from shelfcash_preprocess.models import (
    CellRange,
    FieldMapping,
    MappingProposal,
    Role,
    TableProfile,
    TableRegion,
    TransformOperation,
    ReviewDecision,
    ReviewFile,
    RunContext,
)
from shelfcash_preprocess.operations import execute_operations, validate_operation_plan
from shelfcash_preprocess.config import Settings
from shelfcash_preprocess.pipeline import PreprocessService, load_bundle
from shelfcash_preprocess.schema import read_frame, write_frame
from shelfcash_preprocess.import_semantics import apply_import_revision
from shelfcash_preprocess.readers import ReaderError, inventory_input, read_file


def test_r07_identifier_and_nullable_values_round_trip(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "date": [date(2026, 1, 1)] * 5,
            "store_id": ["001", "0007", "NA", "NULL", "A-" + "9" * 40],
            "product_id": ["P001", "P007", "NA", "NULL", "Ab-C_9"],
            "product_name": ["a", "b", "c", "d", "e"],
            "quantity_sold": [1, 2, 3, 4, 5],
            "unit": [pd.NA, "unit", "unit", "unit", "unit"],
            "selling_price": [pd.NA] * 5,
            "revenue": [pd.NA] * 5,
            "is_stockout": [False, True, pd.NA, False, True],
            "promotion_name": [pd.NA] * 5,
        }
    )
    path = tmp_path / "sales.csv"
    write_frame("sales", frame, path)
    loaded = read_frame("sales", path)
    assert loaded["store_id"].tolist() == frame["store_id"].tolist()
    assert loaded["product_id"].tolist() == frame["product_id"].tolist()
    assert loaded["is_stockout"].tolist() == [False, True, pd.NA, False, True]
    assert b"001" in path.read_bytes() and b"NULL" in path.read_bytes()


def test_r08_schema_rejects_invalid_boolean(tmp_path: Path) -> None:
    path = tmp_path / "sales.csv"
    path.write_text(
        "date,store_id,product_id,product_name,quantity_sold,unit,selling_price,revenue,is_stockout,promotion_name\n"
        "2026-01-01,S,P,N,1,unit,,,sometimes,\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="CANONICAL_SCHEMA_INVALID_BOOLEAN"):
        read_frame("sales", path)


def test_r16_residual_as_of_filters_future_actual_known_at() -> None:
    rows = []
    for target, known in (("2026-01-02", "2026-01-02"), ("2030-01-02", "2030-01-02")):
        target_ts = pd.Timestamp(target)
        rows.append(
            {
                "forecast_origin": target_ts - pd.Timedelta(days=1),
                "target_date": target_ts,
                "actual_known_at": known,
                "horizon": 1,
                "store_id": "S",
                "product_id": "P",
                "actual": 2.0,
                "p25": 1.0,
                "p50": 2.0,
                "p75": 3.0,
                "raw_residual": 0.0,
                "scaled_residual": 0.0,
                "target_train_eligible": True,
                "residual_source": "walk_forward_oos",
            }
        )
    result = validate_residual_history(pd.DataFrame(rows), as_of="2026-08-12")
    assert result["target_date"].tolist() == [pd.Timestamp("2026-01-02")]


def test_r17_weather_requires_forecast_vintage_available_at_origin() -> None:
    tasks = pd.DataFrame(
        {
            "cutoff_date": pd.to_datetime(["2026-10-03"]),
            "target_date": pd.to_datetime(["2026-10-04"]),
        }
    )
    unsafe = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-10-04"]),
            "is_holiday": [False],
            "is_store_closed": [False],
            "temperature": [31.0],
            "rainfall": [3.0],
            "weather_kind": ["forecast"],
            "weather_issued_at": pd.to_datetime(["2026-10-03"]),
            "weather_available_at": pd.to_datetime(["2026-10-05"]),
        }
    )
    blocked = add_calendar_future_features(tasks, unsafe)
    assert pd.isna(blocked.loc[0, "target_temperature"])
    safe = unsafe.assign(weather_available_at=pd.Timestamp("2026-10-03"))
    accepted = add_calendar_future_features(tasks, safe)
    assert accepted.loc[0, "target_temperature"] == 31.0
    realized = safe.assign(weather_kind="realized")
    assert pd.isna(add_calendar_future_features(tasks, realized).loc[0, "target_temperature"])


def _region(frame: pd.DataFrame) -> RegionData:
    profile = TableProfile(
        row_count=len(frame),
        column_count=len(frame.columns),
        columns=list(frame.columns),
        inferred_types={column: "string" for column in frame},
        null_ratio={column: 0.0 for column in frame},
        unique_count={column: int(frame[column].nunique()) for column in frame},
    )
    metadata = TableRegion(
        region_id="r1",
        source_id="s1",
        source_path="fixture.csv",
        container="sheet",
        container_kind="csv",
        bounds=CellRange(start_row=1, end_row=len(frame) + 1, start_col=1, end_col=len(frame.columns)),
        header_rows=[1],
        proposed_header=list(frame.columns),
        confidence=1.0,
        profile=profile,
    )
    return RegionData(metadata, frame, list(range(2, len(frame) + 2)))


def test_r24_unit_conversion_executes_value_and_unit() -> None:
    region = _region(pd.DataFrame({"quantity": [1000.0], "unit": ["g"]}))
    proposal = MappingProposal(
        region_id="r1",
        role=Role.INVENTORY_SNAPSHOT,
        role_confidence=1.0,
        field_mappings=[
            FieldMapping(
                source_column="quantity",
                target_field="quantity_remaining",
                confidence=1.0,
                operations=[
                    TransformOperation(
                        operation="unit_conversion",
                        arguments={"factor": 0.001, "unit_column": "unit", "to_unit": "kg"},
                    )
                ],
            )
        ],
        origin="deterministic",
    )
    transformed = execute_operations(region, proposal)
    assert transformed.frame.loc[0, "quantity"] == 1.0
    assert transformed.frame.loc[0, "unit"] == "kg"
    unsupported = proposal.model_copy(deep=True)
    unsupported.field_mappings[0].operations = [
        TransformOperation(operation="aggregate", arguments={})
    ]
    assert validate_operation_plan(unsupported) == ["UNSUPPORTED_OPERATION:aggregate"]


def test_r29_eod_zero_lead_rejected_before_solver() -> None:
    scenario = InventoryDemandScenario(
        scenario_id="s1",
        probability_weight=1.0,
        simulation_start_date=date(2026, 8, 13),
        simulation_end_date=date(2026, 8, 13),
        lines=[
            InventoryDemandLine(
                scenario_id="s1",
                store_id="S",
                ingredient_id="I",
                target_date=date(2026, 8, 13),
                quantity=1,
                unit="kg",
            )
        ],
    )
    lot = InventoryLot(
        lot_id="L",
        store_id="S",
        ingredient_id="I",
        quantity_remaining=1,
        unit="kg",
        received_date=date(2026, 8, 1),
        expiry_date=date(2026, 8, 30),
    )
    offer = SupplierOffer(
        offer_id="O",
        supplier_id="SUP",
        store_id="S",
        ingredient_id="I",
        unit="kg",
        order_date=date(2026, 8, 12),
        pack_size=1,
        unit_price=1,
        lead_time_days=0,
    )
    with pytest.raises(ValueError, match="ZERO_LEAD_UNSUPPORTED_FOR_EOD_SNAPSHOT"):
        OptimizationRequest(
            request_id="r",
            decision_date=date(2026, 8, 12),
            planning_end_date=date(2026, 8, 13),
            initial_inventory=[lot],
            inventory_snapshot_date=date(2026, 8, 12),
            inventory_snapshot_boundary="EOD",
            demand_scenarios=[scenario],
            supplier_offers=[offer],
        )


def _tiny_optimization_request(*, stochastic: bool, allow_fallback: bool) -> OptimizationRequest:
    scenarios = [
        InventoryDemandScenario(
            scenario_id="s1",
            probability_weight=1.0,
            simulation_start_date=date(2026, 8, 13),
            simulation_end_date=date(2026, 8, 13),
            lines=[
                InventoryDemandLine(
                    scenario_id="s1",
                    store_id="S",
                    ingredient_id="I",
                    target_date=date(2026, 8, 13),
                    quantity=1,
                    unit="kg",
                )
            ],
        )
    ]
    return OptimizationRequest(
        request_id="tiny-real-scipy",
        decision_date=date(2026, 8, 12),
        planning_end_date=date(2026, 8, 13),
        inventory_snapshot_date=date(2026, 8, 12),
        inventory_snapshot_boundary="EOD",
        initial_inventory=[
            InventoryLot(
                lot_id="L",
                store_id="S",
                ingredient_id="I",
                quantity_remaining=10,
                unit="kg",
                received_date=date(2026, 8, 1),
                expiry_date=date(2026, 8, 30),
            )
        ],
        demand_scenarios=scenarios,
        supplier_offers=[
            SupplierOffer(
                offer_id="O",
                supplier_id="SUP",
                store_id="S",
                ingredient_id="I",
                unit="kg",
                order_date=date(2026, 8, 12),
                pack_size=1,
                unit_price=1,
                lead_time_days=1,
                shelf_life_days=30,
            )
        ],
        stochastic=stochastic,
        allow_mode_fallback=allow_fallback,
        seed=42,
    )


def test_r34_real_scipy_candidate_passes_exact_m4_and_critic() -> None:
    result = optimize_procurement(
        _tiny_optimization_request(stochastic=False, allow_fallback=False)
    )
    assert result.status == "COMPLETED"
    assert result.recommended_strategy is not None
    assert result.provenance["actual_mode"] == "deterministic"
    evaluation = result.evaluations[result.recommended_strategy]
    assert evaluation.plan.solver_status == "OPTIMAL"
    assert evaluation.simulation is not None
    assert evaluation.critic.passed


def test_r32_strict_stochastic_does_not_hide_single_scenario_fallback() -> None:
    with pytest.raises(
        OptimizationNotAvailableError,
        match="at least two fully weighted scenarios",
    ):
        optimize_procurement(
            _tiny_optimization_request(stochastic=True, allow_fallback=False)
        )


def test_r25_append_upsert_replace_have_executable_scoped_semantics() -> None:
    existing = pd.DataFrame(
        {
            "store_id": ["S", "S"],
            "date": ["2026-01-01", "2026-01-02"],
            "product_id": ["P", "P"],
            "quantity": [1.0, 2.0],
        }
    )
    keys = ["store_id", "date", "product_id"]
    identical = existing.iloc[[1]].copy()
    appended = apply_import_revision(
        existing, identical, mode="append", key_columns=keys
    )
    assert appended.frame.equals(existing)
    assert appended.idempotent_rows == 1

    conflicting = identical.assign(quantity=20.0)
    with pytest.raises(ValueError, match="OVERLAP_REQUIRES_PRECEDENCE_OR_REVIEW"):
        apply_import_revision(existing, conflicting, mode="append", key_columns=keys)

    upserted = apply_import_revision(
        existing, conflicting, mode="upsert", key_columns=keys
    )
    assert upserted.updated_rows == 1
    assert upserted.frame.loc[upserted.frame["date"] == "2026-01-02", "quantity"].item() == 20

    replacement = pd.DataFrame(
        {
            "store_id": ["S"],
            "date": ["2026-01-03"],
            "product_id": ["P"],
            "quantity": [3.0],
        }
    )
    replaced = apply_import_revision(
        existing,
        replacement,
        mode="replace",
        key_columns=keys,
        replace_scope={"store_id": "S"},
    )
    assert replaced.removed_rows == 2
    assert replaced.frame.to_dict(orient="records") == replacement.to_dict(orient="records")
    assert existing["quantity"].tolist() == [1.0, 2.0]


def test_r30_cqr_uses_exact_finite_sample_order_statistic() -> None:
    correction, level = conformal_quantile(np.arange(20, dtype=float), 0.5)
    assert correction == 10.0
    assert level == 11 / 20
    with pytest.raises(ValueError, match="finite"):
        conformal_quantile(np.array([1.0, np.nan]), 0.5)
    with pytest.raises(Exception, match="rank exceeds"):
        conformal_quantile(np.array([1.0]), 0.9)


def _offline_service(tmp_path: Path) -> PreprocessService:
    return PreprocessService(
        Settings(
            config_path=tmp_path / "missing.env",
            api_key=None,
            model="",
            llm_mode="offline",
        )
    )


def test_r01_r09_blocking_parse_issue_blocks_readiness_and_keeps_locator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SHELFCASH_PREPROCESS_STATE_DIR", str(tmp_path / "state"))
    source = tmp_path / "sales.csv"
    source.write_text(
        "Ngày GD,Tên món,SLX\n2026-01-01,A,bad\n2026-01-02,A,3\n",
        encoding="utf-8",
    )
    bundle = _offline_service(tmp_path).run(
        source,
        tmp_path / "runs",
        context=RunContext(
            store_id="S", cutoff_date=date(2026, 1, 2), date_locale="YMD", tenant_id="t"
        ),
        llm_mode="offline",
    )
    readiness = bundle.manifest.readiness["forecast_core"]
    assert not readiness.data_validated
    assert "AMBIGUOUS_NUMBER" in readiness.issues
    lineage = __import__("json").loads(
        (bundle.run_dir / "lineage" / "records.json").read_text(encoding="utf-8")
    )
    quarantine = __import__("json").loads(
        (bundle.run_dir / "quarantine" / "records.json").read_text(encoding="utf-8")
    )
    assert lineage[0]["row_locator"] == 3
    assert quarantine[0]["source_locator"]["row"] == 2
    assert quarantine[0]["raw_row"]["SLX"] == "bad"


def test_r05_review_replays_snapshot_not_mutated_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SHELFCASH_PREPROCESS_STATE_DIR", str(tmp_path / "state"))
    source = tmp_path / "unknown.csv"
    source.write_text("when,item,qty\n2026-01-01,A,7\n", encoding="utf-8")
    service = _offline_service(tmp_path)
    parent = service.run(
        source,
        tmp_path / "runs",
        context=RunContext(
            store_id="S", cutoff_date=date(2026, 1, 1), date_locale="YMD", tenant_id="t"
        ),
        llm_mode="offline",
    )
    plan = __import__("json").loads(
        (parent.run_dir / "mapping_plan.json").read_text(encoding="utf-8")
    )
    proposal = plan["proposals"][0]
    issue_id = f"mapping_{proposal['region_id']}"
    decisions = [
        ReviewDecision(
            issue_id=issue_id,
            action="set_role",
            value={"region_id": proposal["region_id"], "role": "sales"},
        ),
        *[
            ReviewDecision(
                issue_id=issue_id,
                action="map_field",
                value={
                    "region_id": proposal["region_id"],
                    "source_column": source_column,
                    "target_field": target,
                },
            )
            for source_column, target in (
                ("when", "date"),
                ("item", "product_name"),
                ("qty", "quantity_sold"),
            )
        ],
        ReviewDecision(
            issue_id=issue_id,
            action="approve_mapping",
            value={"region_id": proposal["region_id"]},
        ),
    ]
    review = ReviewFile(
        run_id=parent.manifest.run_id,
        source_inventory_hash=parent.manifest.source_inventory_hash,
        mapping_plan_hash=parent.manifest.mapping_plan_hash,
        decisions=decisions,
    )
    review_path = tmp_path / "review.json"
    review_path.write_text(review.model_dump_json(indent=2), encoding="utf-8")
    source.write_text("when,item,qty\n2026-01-01,A,97\n", encoding="utf-8")
    child = service.apply_review(parent.run_dir, review_path)
    sales = read_frame("sales", child.run_dir / "canonical" / "sales.csv")
    assert sales["quantity_sold"].tolist() == [7.0]
    assert child.manifest.parent_run_id == parent.manifest.run_id
    assert child.manifest.review_file_sha256


def test_r06_tampered_canonical_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SHELFCASH_PREPROCESS_STATE_DIR", str(tmp_path / "state"))
    source = tmp_path / "sales.csv"
    source.write_text(
        "Ngày GD,Tên món,SLX\n2026-01-01,A,7\n", encoding="utf-8"
    )
    bundle = _offline_service(tmp_path).run(
        source,
        tmp_path / "runs",
        context=RunContext(
            store_id="S", cutoff_date=date(2026, 1, 1), date_locale="YMD", tenant_id="t"
        ),
        llm_mode="offline",
    )
    sales_path = bundle.run_dir / "canonical" / "sales.csv"
    sales_path.write_bytes(sales_path.read_bytes().replace(b",7.0,", b",97.0,"))
    with pytest.raises(ValueError, match="BUNDLE_ARTIFACT_(SIZE|HASH)_MISMATCH"):
        load_bundle(bundle.run_dir)


def test_r27_profiles_are_tenant_namespaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = tmp_path / "state"
    monkeypatch.setenv("SHELFCASH_PREPROCESS_STATE_DIR", str(state))
    source = tmp_path / "sales.csv"
    source.write_text(
        "Ngày GD,Tên món,SLX\n2026-01-01,A,7\n", encoding="utf-8"
    )
    service = _offline_service(tmp_path)
    for tenant in ("tenant-a", "tenant-b"):
        service.run(
            source,
            tmp_path / "runs",
            context=RunContext(
                store_id="S",
                cutoff_date=date(2026, 1, 1),
                date_locale="YMD",
                tenant_id=tenant,
            ),
            llm_mode="offline",
        )
    profiles = list((state / "profiles_v2").glob("*.json"))
    assert len(profiles) == 2
    assert len({path.name.split("_")[1] for path in profiles}) == 2


def test_r28_same_tenant_interpretation_is_idempotent_in_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SHELFCASH_PREPROCESS_STATE_DIR", str(tmp_path / "state"))
    source = tmp_path / "sales.csv"
    source.write_text(
        "NgÃ y GD,TÃªn mÃ³n,SLX\n2026-01-01,A,7\n", encoding="utf-8"
    )
    service = _offline_service(tmp_path)
    context = RunContext(
        store_id="S",
        cutoff_date=date(2026, 1, 1),
        date_locale="YMD",
        tenant_id="tenant-idempotent",
    )
    first = service.run(source, tmp_path / "runs", context=context, llm_mode="offline")
    second = service.run(source, tmp_path / "runs", context=context, llm_mode="offline")
    assert first.run_dir == second.run_dir
    published = [path for path in (tmp_path / "runs").iterdir() if path.is_dir()]
    assert published == [first.run_dir]


def test_r28_concurrent_import_has_one_published_ledger_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = tmp_path / "state"
    monkeypatch.setenv("SHELFCASH_PREPROCESS_STATE_DIR", str(state))
    source = tmp_path / "sales.csv"
    source.write_text(
        "NgÃ y GD,TÃªn mÃ³n,SLX\n2026-01-01,A,7\n", encoding="utf-8"
    )
    context = RunContext(
        store_id="S",
        cutoff_date=date(2026, 1, 1),
        date_locale="YMD",
        tenant_id="tenant-concurrent",
    )

    def execute() -> tuple[str, str]:
        try:
            bundle = _offline_service(tmp_path).run(
                source, tmp_path / "runs", context=context, llm_mode="offline"
            )
            return "ok", str(bundle.run_dir)
        except RuntimeError as exc:
            return "busy", str(exc)

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _: execute(), range(2)))
    successes = [value for status, value in outcomes if status == "ok"]
    failures = [value for status, value in outcomes if status == "busy"]
    assert successes
    assert len(set(successes)) == 1
    assert not failures or all("IMPORT_ALREADY_PROCESSING" in item for item in failures)
    with sqlite3.connect(state / "import_ledger_v2.sqlite") as connection:
        rows = connection.execute(
            "SELECT status, COUNT(*) FROM imports WHERE tenant_id=? GROUP BY status",
            ("tenant-concurrent",),
        ).fetchall()
    assert rows == [("published", 1)]


def test_r38_installed_cli_help_is_callable_and_windows_safe() -> None:
    runner = CliRunner()
    for application in (preprocess_cli, forecast_cli):
        result = runner.invoke(application, ["--help"])
        assert result.exit_code == 0, result.output
        assert "Usage" in result.output
    # Rich help on a legacy Windows code page must not require Vietnamese
    # glyph support merely to discover the production commands.
    assert (forecast_cli.info.help or "").isascii()


class _FakeResponses:
    def __init__(self, *, parsed: object | None = None, error: Exception | None = None) -> None:
        self.parsed = parsed
        self.error = error

    def parse(self, **_: object) -> object:
        if self.error is not None:
            raise self.error
        return SimpleNamespace(output_parsed=self.parsed, usage=None)


def _m6_fake_settings(tmp_path: Path) -> Settings:
    return Settings(
        config_path=tmp_path / "unused.env",
        api_key=None,
        model="",
        m6_llm_enabled=True,
        m6_model="test-only-model-id",
    )


def _tiny_decision_package():
    request = _tiny_optimization_request(stochastic=False, allow_fallback=False)
    result = optimize_procurement(request)
    return build_final_decision_package(request, result)


def test_r36_llm_invented_evidence_is_rejected_by_grounding_guard(tmp_path: Path) -> None:
    malicious = _GeneratedExplanation(
        claims=[
            _GeneratedClaim(
                claim_type="recommendation",
                text="Invented recommendation.",
                evidence_ids=["evidence-that-does-not-exist"],
                facts={"order_quantity": 999999},
            )
        ]
    )
    generator = OpenAIGroundedGenerator(
        settings=_m6_fake_settings(tmp_path),
        client=SimpleNamespace(responses=_FakeResponses(parsed=malicious)),
    )
    with pytest.raises(GroundingError, match="Unknown evidence citation"):
        explain_decision(_tiny_decision_package(), "Why?", generator=generator)


@pytest.mark.parametrize(
    ("responses", "code"),
    [
        (_FakeResponses(parsed=None), "M6_LLM_MALFORMED_OUTPUT"),
        (_FakeResponses(error=TimeoutError("secret-free timeout")), "M6_LLM_TIMEOUT"),
    ],
)
def test_r37_llm_malformed_and_timeout_are_typed(
    tmp_path: Path, responses: _FakeResponses, code: str
) -> None:
    generator = OpenAIGroundedGenerator(
        settings=_m6_fake_settings(tmp_path),
        client=SimpleNamespace(responses=responses),
    )
    with pytest.raises(M6LLMError) as captured:
        explain_decision(_tiny_decision_package(), "Why?", generator=generator)
    assert captured.value.code == code


def test_r43_reader_budgets_apply_across_archives_and_before_table_growth(
    tmp_path: Path,
) -> None:
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    for index in (1, 2):
        with zipfile.ZipFile(uploads / f"part{index}.zip", "w") as archive:
            archive.writestr(f"part{index}.csv", b"x" * 60)
    settings = Settings(
        config_path=tmp_path / "none.env",
        api_key=None,
        max_files=10,
        max_file_bytes=1_000,
        max_total_input_bytes=10_000,
        max_archive_bytes=100,
    )
    with pytest.raises(ReaderError, match="ACROSS_INPUT_SET"):
        inventory_input(uploads, tmp_path / "workspace", settings)

    nested = tmp_path / "nested.zip"
    with zipfile.ZipFile(nested, "w") as archive:
        archive.writestr("inside.zip", b"not-a-real-nested-archive")
    with pytest.raises(ReaderError, match="NESTED_ARCHIVE_FORBIDDEN"):
        inventory_input(nested, tmp_path / "nested-workspace", settings)

    csv_path = tmp_path / "too_many.csv"
    csv_path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    row_limited = Settings(
        config_path=tmp_path / "none.env",
        api_key=None,
        max_table_rows=2,
    )
    with pytest.raises(ReaderError, match="TABLE_ROW_LIMIT_EXCEEDED"):
        read_file(csv_path, "source", row_limited)
