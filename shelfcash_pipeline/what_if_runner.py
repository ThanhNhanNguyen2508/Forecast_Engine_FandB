"""Portable What-if CLI around the existing typed mutation and M5 authority APIs."""
from __future__ import annotations

import argparse
import gc
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from shelfcash_forecast.decision_intelligence.contracts import FinalDecisionPackage
from shelfcash_forecast.decision_intelligence.what_if import (
    confirm_what_if, draft_what_if, explain_what_if, run_what_if,
)
from shelfcash_forecast.decision_intelligence.what_if.contracts import WhatIfModification
from shelfcash_forecast.optimization.contracts import OptimizationRequest
from shelfcash_pipeline.config_runner import _reject_reparse
from shelfcash_pipeline.context import ENGINE_ROOT, write_json


class WhatIfConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    label: Literal["DEMO_ONLY_NOT_FOR_OPERATION"]
    actor: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1)
    question: str = "Giả định này thay đổi kế hoạch nhập hàng thế nào?"
    modifications: list[WhatIfModification] = Field(min_length=1)


def load_what_if_configuration(path: Path) -> WhatIfConfiguration:
    return WhatIfConfiguration.model_validate_json(path.read_text(encoding="utf-8-sig"))


def run_from_pipeline(config: WhatIfConfiguration, baseline: Path, *, output_root: Path,
                      workspace_root: Path = ENGINE_ROOT) -> Path:
    _reject_reparse(output_root)
    if not output_root.resolve().is_relative_to(workspace_root.resolve()):
        raise ValueError("What-if output must stay inside the workspace")
    if output_root.resolve().is_relative_to(baseline.resolve()):
        raise ValueError("What-if must not write inside the baseline run")
    for relative in ("m6/summary.json", "m5/optimization_result.json", "m6/decision_package.json"):
        _reject_reparse(baseline / relative)
    summary = json.loads((baseline / "m6/summary.json").read_text(encoding="utf-8-sig"))
    mode = summary["selected_optimization_mode"]
    payload = json.loads((baseline / "m5/optimization_result.json").read_text(encoding="utf-8-sig"))
    runs = payload.get("runs", payload)
    request = OptimizationRequest.model_validate(runs[mode]["request"])
    del payload, runs
    gc.collect()
    decision = FinalDecisionPackage.model_validate_json(
        (baseline / "m6/decision_package.json").read_text(encoding="utf-8-sig"))
    if request.environment not in {"DEMO", "BACKTEST"}:
        raise ValueError("What-if runner requires a DEMO/BACKTEST baseline")
    draft = draft_what_if(request, decision, config.modifications, actor=config.actor,
                         reason=config.reason, idempotency_key=config.idempotency_key)
    package = run_what_if(request, decision, confirm_what_if(draft))
    answer = explain_what_if(package, config.question)
    name = "what_if_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    destination = output_root / name
    _reject_reparse(destination)
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / "what_if_package.json", package.model_dump(mode="json"))
    write_json(destination / "what_if_answer.json", answer.model_dump(mode="json"))
    write_json(destination / "what_if_result.json", {
        "what_if_id": package.what_if_id, "baseline": str(baseline.resolve()),
        "hypothetical_status": package.hypothetical_decision.decision_status,
        "technical_outcome": package.optimization_result.technical_outcome,
        "supplier_orders_executed": False, "llm_api_called": False,
        "execution_authorized": False, "business_ready": False,
    })
    return destination


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run typed What-if against an existing M5/M6 pipeline baseline.")
    parser.add_argument("--config", type=Path, default=ENGINE_ROOT / "configs/what_if_budget.example.json")
    parser.add_argument("--baseline", type=Path, default=ENGINE_ROOT / "outputs/pipeline_until_m6")
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    try:
        config = load_what_if_configuration(args.config)
        if args.validate_only:
            print(json.dumps({"status": "VALID", "modifications": len(config.modifications)}))
            return 0
        destination = run_from_pipeline(config, args.baseline, output_root=ENGINE_ROOT / "outputs")
        print("OUTPUT_DIR=" + str(destination))
        return 0
    except Exception as exc:
        print(f"WHAT_IF_FAILED:{type(exc).__name__}:{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
