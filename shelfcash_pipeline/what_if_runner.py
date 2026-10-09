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
from shelfcash_forecast.optimization.contracts import OptimizationRequest, OptimizationResult
from shelfcash_forecast.decision_intelligence.service import build_final_decision_package
from shelfcash_forecast.optimization.planning_service import content_hash
from shelfcash_pipeline.config_runner import _reject_reparse
from shelfcash_pipeline.context import ENGINE_ROOT, write_json


class WhatIfConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    label: Literal["DEMO_ONLY_NOT_FOR_OPERATION", "SCENARIO_ONLY_NOT_FOR_OPERATION"]
    actor: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1)
    question: str = "Giả định này thay đổi kế hoạch nhập hàng thế nào?"
    modifications: list[WhatIfModification] = Field(min_length=1)
    output_root: Path | None = None


def load_what_if_configuration(path: Path) -> WhatIfConfiguration:
    return WhatIfConfiguration.model_validate_json(path.read_text(encoding="utf-8-sig"))


def load_baseline(baseline: Path) -> tuple[OptimizationRequest, OptimizationResult, FinalDecisionPackage, str]:
    """Load the declared selected mode, checking M5 acceptance before read-only M6."""
    summary_path = baseline / "m5/summary.json"
    if not summary_path.exists():
        summary_path = baseline / "m6/summary.json"
    _reject_reparse(summary_path)
    _reject_reparse(baseline / "m5/optimization_result.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    mode = summary.get("selected_mode") or summary["selected_optimization_mode"]
    payload = json.loads((baseline / "m5/optimization_result.json").read_text(encoding="utf-8-sig"))
    runs = payload.get("runs", payload)
    request = OptimizationRequest.model_validate(runs[mode]["request"])
    result = OptimizationResult.model_validate(runs[mode]["result"])
    del payload, runs
    gc.collect()
    selected = result.evaluations.get(result.recommended_strategy)
    if (result.request_id != request.request_id or selected is None or not selected.plan.completed
            or not selected.critic.passed or selected.simulation is None
            or not selected.critic.checks.get("evaluation_coverage")):
        raise ValueError("BASELINE_REQUIRES_EXACT_CRITIC_ACCEPTED_PLAN")
    full = request.evaluation_scenarios or request.demand_scenarios
    if (len(selected.simulation.results) != len(full)
            or {w.scenario_id: w.probability_weight for w in selected.simulation.results}
            != {w.scenario_id: w.probability_weight for w in full}):
        raise ValueError("BASELINE_FULL_EVALUATION_COVERAGE_MISMATCH")
    accepted_path = baseline / "m5/accepted_technical_plan.json"
    if accepted_path.exists():
        _reject_reparse(accepted_path)
        accepted = json.loads(accepted_path.read_text(encoding="utf-8-sig"))
        if (accepted["request_hash"] != content_hash(request)
                or content_hash(accepted["selected_plan"]) != content_hash(selected.plan)
                or not accepted["critic"]["passed"]):
            raise ValueError("BASELINE_ACCEPTED_ARTIFACT_HASH_MISMATCH")
    decision_path = baseline / "m6/decision_package.json"
    if decision_path.exists():
        _reject_reparse(decision_path)
        decision = FinalDecisionPackage.model_validate_json(decision_path.read_text(encoding="utf-8-sig"))
        summary = decision.recommended_plan_summary
        physical = [(o.offer_id, o.order_quantity, o.pack_count, o.order_date, o.arrival_date,
                     o.purchase_cost, o.delivery_cost) for o in selected.plan.orders]
        reported = [(o.offer_id, o.order_quantity, o.pack_count, o.order_date, o.arrival_date,
                     o.purchase_cost, o.delivery_cost) for o in [*decision.immediate_orders, *decision.scheduled_orders]]
        if (decision.request_id != request.request_id or decision.recommended_strategy != result.recommended_strategy
                or summary is None or summary.plan_id != selected.plan.plan_id
                or sorted(physical) != sorted(reported)):
            raise ValueError("BASELINE_M6_DISAGREES_WITH_M5_PLAN")
    else:
        decision = build_final_decision_package(request, result)
    if not decision.provenance.get('currency'):
        decision=decision.model_copy(update={'provenance':{**decision.provenance,
            'historical_readonly_decision_hash':content_hash(decision),
            'currency':request.currency,'currency_source':'VALIDATED_BOUND_REQUEST_READER_ADAPTER',
            'currency_request_hash':content_hash(request)}})
    return request, result, decision, mode


def run_from_pipeline(config: WhatIfConfiguration, baseline: Path, *, output_root: Path,
                      workspace_root: Path = ENGINE_ROOT) -> Path:
    _reject_reparse(output_root)
    if not output_root.resolve().is_relative_to(workspace_root.resolve()):
        raise ValueError("What-if output must stay inside the workspace")
    if output_root.resolve().is_relative_to(baseline.resolve()):
        raise ValueError("What-if must not write inside the baseline run")
    request, result, decision, mode = load_baseline(baseline)
    del result
    gc.collect()
    draft = draft_what_if(request, decision, config.modifications, actor=config.actor,
                         reason=config.reason, idempotency_key=config.idempotency_key)
    package = run_what_if(request, decision, confirm_what_if(draft))
    answer = explain_what_if(package, config.question)
    name = "what_if_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    destination = output_root / name
    _reject_reparse(destination)
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / "what_if_package.json", package)
    write_json(destination / "what_if_answer.json", answer)
    from shelfcash_forecast.optimization.planning_config import load_planning_config
    from shelfcash_forecast.decision_intelligence.what_if.export import export_what_if_planning
    parent_config=load_planning_config(baseline/'m5/planning_config.json')[0]
    manifest=json.loads((baseline/'run_manifest.json').read_text(encoding='utf-8-sig')) if (baseline/'run_manifest.json').exists() else {}
    export_what_if_planning(package,parent_config,destination/'hypothetical',source_bundle=manifest.get('bundle_path'),baseline_path=str(baseline.resolve()))
    write_json(destination/'three_way_comparison.json',package.provenance.get('three_way_comparison'))
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
    parser.add_argument('--output-root',type=Path,help='Fresh What-if packages outside the parent run, within the workspace')
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    try:
        config = load_what_if_configuration(args.config)
        if args.validate_only:
            print(json.dumps({"status": "VALID", "modifications": len(config.modifications)}))
            return 0
        from shelfcash_pipeline.config_runner import configuration_root
        import shelfcash_forecast
        print("PYTHON="+sys.executable,flush=True)
        print("WHAT_IF_RUNNER_ORIGIN="+str(Path(__file__).resolve()),flush=True)
        print("FORECAST_ORIGIN="+str(shelfcash_forecast.__file__),flush=True)
        destination = run_from_pipeline(config, args.baseline,
            output_root=args.output_root or config.output_root or ENGINE_ROOT/'outputs',workspace_root=configuration_root(args.config))
        print("OUTPUT_DIR=" + str(destination))
        return 0
    except Exception as exc:
        print(f"WHAT_IF_FAILED:{type(exc).__name__}:{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
