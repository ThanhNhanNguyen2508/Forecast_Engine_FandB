from __future__ import annotations

import json
import tracemalloc
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict, Field, field_serializer

from shelfcash_forecast.json_output import write_json
from shelfcash_forecast.optimization.contracts import OptimizationRequest


class Payload(BaseModel):
    model_config = ConfigDict(extra="forbid", serialize_by_alias=True)
    text: str
    date: date
    timestamp: datetime
    price: Decimal
    internal: str = Field(serialization_alias="public_name")
    hidden: int = Field(exclude=True, default=1)


def test_native_json_matches_validated_json_mode_and_aliases(tmp_path):
    value = Payload(text="Chuối • sữa", date=date(2026, 8, 19),
                    timestamp=datetime(2026, 8, 12, tzinfo=timezone.utc), price=Decimal("15.50"), internal="name")
    request = OptimizationRequest(request_id="stream-boundary", decision_date=date(2026, 8, 12),
                                  planning_end_date=date(2026, 8, 19), initial_inventory=[],
                                  supplier_offers=[], demand_scenarios=[])
    path = tmp_path / "result.json"
    write_json(path, {"model": value, "request": request, "rows": [value], "ordinary": (None, True, 0, 1.5)})
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "model": value.model_dump(mode="json"), "request": request.model_dump(mode="json"),
        "rows": [value.model_dump(mode="json")], "ordinary": [None, True, 0, 1.5]}
    assert path.read_bytes().endswith(b"\n")


def test_custom_serializer_uses_native_pydantic_contract(tmp_path):
    class Serialized(BaseModel):
        value: int

        @field_serializer("value")
        def stringify(self, value):
            return "value=" + str(value)

    value = Serialized(value=42)
    write_json(tmp_path / "custom.json", value)
    assert json.loads((tmp_path / "custom.json").read_bytes()) == value.model_dump(mode="json")


def test_writer_does_not_call_recursive_model_dump(tmp_path, monkeypatch):
    class Model(BaseModel):
        data: list[int]

    models = [Model(data=[1, 2, 3]), Model(data=[4])]
    expected = [model.model_dump(mode="json") for model in models]

    def forbidden(*args, **kwargs):
        raise AssertionError("Entire graph must not be materialized with model_dump")

    monkeypatch.setattr(BaseModel, "model_dump", forbidden)
    write_json(tmp_path / "models.json", models)
    assert json.loads((tmp_path / "models.json").read_bytes()) == expected


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("inside_model", [False, True])
def test_nonfinite_any_fields_still_rejected(tmp_path, value, inside_model):
    class OpenValues(BaseModel):
        payload: Any

    payload = {"value": [value]}
    if inside_model:
        payload = OpenValues(payload=payload)
    with pytest.raises(ValueError, match="NaN or infinity"):
        write_json(tmp_path / "invalid.json", payload)
    assert not (tmp_path / "invalid.json").exists()


def test_cycle_rejected_without_creating_partial_file(tmp_path):
    value = {}
    value["cycle"] = value
    with pytest.raises(ValueError, match="Circular"):
        write_json(tmp_path / "cycle.json", value)


def test_scalar_key_coercion_and_explicit_string_fallback(tmp_path):
    value = {None: "none", False: "false", 2.5: "number", "path": Path("display")}
    write_json(tmp_path / "keys.json", value, fallback_str=True)
    assert json.loads((tmp_path / "keys.json").read_bytes()) == json.loads(json.dumps(value, default=str))
    with pytest.raises(TypeError):
        write_json(tmp_path / "strict.json", {"unsupported": object()})


def test_large_sequence_uses_bounded_temporary_memory(tmp_path):
    class Row(BaseModel):
        value: str

    rows = [Row(value="x" * 8192) for _ in range(2000)]
    path = tmp_path / "large.json"
    tracemalloc.start()
    try:
        write_json(path, rows)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert path.stat().st_size > 16_000_000
    # A regression to a full dict/string envelope allocates tens of MB here.
    assert peak < 4_000_000, peak
    loaded = json.loads(path.read_bytes())
    assert len(loaded) == 2000 and loaded[-1]["value"] == rows[-1].value
