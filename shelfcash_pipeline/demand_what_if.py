"""Reusable rule-based demand What-if CLI and public orchestrator/export integration."""
from __future__ import annotations

import argparse
import gc
import inspect
import json
import logging
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from shelfcash_forecast.decision_intelligence.agents.contracts import AgentRunRequest
from shelfcash_forecast.decision_intelligence.agents.orchestrator import DecisionOrchestrator
from shelfcash_forecast.decision_intelligence.computation_gateway import M5ComputationGateway
from shelfcash_forecast.decision_intelligence.integrity import sha256_content_hash
from shelfcash_forecast.decision_intelligence.what_if.rules import IngredientDemandRule, expand_demand_rule, demand_rule_summary, parse_demand_rule
from shelfcash_forecast.decision_intelligence.what_if.service import confirm_what_if
from shelfcash_forecast.decision_intelligence.what_if.export import export_what_if_planning
from shelfcash_forecast.optimization.planning_config import load_planning_config
from shelfcash_forecast.optimization.planning_service import content_hash
from shelfcash_forecast.optimization.contracts import ProcurementPlan
from shelfcash_pipeline.context import ENGINE_ROOT, write_json
from shelfcash_pipeline.config_runner import _reject_reparse
from shelfcash_pipeline.what_if_runner import load_baseline


class TracedM5Gateway(M5ComputationGateway):
    """Record real inherited computation calls; no replacement solver or verdict."""
    def __init__(self):
        self.calls = []

    def optimize(self, request):
        started = time.monotonic()
        logging.info("REAL_M5_GATEWAY optimize request=%s full_worlds=%s", request.request_id,
                     len(request.evaluation_scenarios or request.demand_scenarios))
        result = super().optimize(request)
        self.calls.append({"operation": "optimize", "implementation": inspect.getfile(M5ComputationGateway.optimize),
            "request_hash": content_hash(request), "result_hash": content_hash(result),
            "elapsed_seconds": time.monotonic() - started, "real_computation": True,
            "outcome": result.technical_outcome, "recommended_strategy": result.recommended_strategy})
        return result

    def evaluate_plan(self, plan, request):
        started = time.monotonic()
        logging.info("REAL_M5_GATEWAY exact evaluate fixed plan=%s full_worlds=%s", plan.plan_id,
                     len(request.evaluation_scenarios or request.demand_scenarios))
        result = super().evaluate_plan(plan, request)
        self.calls.append({"operation": "evaluate_plan", "implementation": inspect.getfile(M5ComputationGateway.evaluate_plan),
            "request_hash": content_hash(request), "input_plan_hash": content_hash(plan),
            "result_hash": content_hash(result), "elapsed_seconds": time.monotonic() - started,
            "real_computation": True, "optimizer_called": False, "critic_passed": result.critic.passed})
        return result


def _trace(result):
    return result.model_dump(mode="json", exclude={"what_if_package", "what_if_draft", "answer", "result_payload"})


def run_demand_what_if(rule: IngredientDemandRule, baseline: Path, destination: Path, *, actor: str, reason: str,
                      idempotency_key: str, execute_hypothetical: bool = False, workspace_root: Path = ENGINE_ROOT):
    """Draft, explicitly confirm, execute and export through registered public tools."""
    _reject_reparse(destination)
    if not destination.resolve().is_relative_to(workspace_root.resolve()) or destination.resolve().is_relative_to(baseline.resolve()):
        raise ValueError("WHAT_IF_OUTPUT_MUST_BE_NEW_AND_OUTSIDE_BASELINE_WITHIN_WORKSPACE")
    destination.mkdir(parents=True, exist_ok=False)
    logging.info("LOAD_BASELINE %s", baseline)
    request, result, decision, mode = load_baseline(baseline)
    selected = result.evaluations[result.recommended_strategy]
    request_before, result_before, decision_before = content_hash(request), content_hash(result), sha256_content_hash(decision)
    write_json(destination / "baseline_references.json", {"path": str(baseline.resolve()), "selected_mode": mode,
        "strategy": result.recommended_strategy, "request_id": request.request_id, "plan_id": selected.plan.plan_id,
        "profile": request.planning_binding, "request_hash": request_before, "result_hash": result_before,
        "plan_hash": content_hash(selected.plan), "decision_hash": decision_before,
        "full_pool_hash": content_hash(request.evaluation_scenarios or request.demand_scenarios),
        "source_files": [str(baseline / p) for p in ("m5/summary.json", "m5/optimization_result.json", "m5/accepted_technical_plan.json")],
        "m6_context": "existing_read_only" if (baseline / "m6/decision_package.json").exists() else "public_read_only_builder"})
    # One selected evaluation is enough for A. Do not copy all historical attempts.
    write_json(destination / "baseline_evaluation.json", selected.model_dump(mode="json", exclude={"attempts", "physics_simulation"}))
    write_json(destination / "baseline_request.json", request)
    write_json(destination / "baseline_decision.json", decision)
    gateway = TracedM5Gateway()
    orchestrator = DecisionOrchestrator(gateway=gateway)
    question = "What if: nhu cầu toàn bộ nguyên liệu trong cả kỳ kế hoạch bằng " + str(rule.multiplier) + " lần baseline."
    common = {"language": "vi", "baseline_request": request, "baseline_decision": decision,
              "actor": actor, "reason": reason, "idempotency_key": idempotency_key}
    draft_run = orchestrator.run(AgentRunRequest(mode="WHAT_IF_DRAFT", question=question,
                                typed_modifications=expand_demand_rule(rule, request), **common))
    if draft_run.what_if_draft is None or draft_run.what_if_draft.status != "DRAFT_READY":
        raise ValueError("PUBLIC_ORCHESTRATOR_DRAFT_FAILED:" + str(draft_run.error_codes))
    draft = draft_run.what_if_draft
    assert not gateway.calls
    write_json(destination / "rulebased_input.json", {**rule.model_dump(mode="json"), "question": question,
               "case_id": "GLOBAL_INGREDIENT_DEMAND_X1_1" if rule.multiplier == 1.1 else "GLOBAL_INGREDIENT_DEMAND_SCALE"})
    write_json(destination / "draft_summary.json", demand_rule_summary(rule, request))
    write_json(destination / "what_if_draft.json", draft)
    write_json(destination / "normalized_modifications.json", draft.normalized_modifications)
    write_json(destination / "draft_trace.json", _trace(draft_run))
    if not execute_hypothetical:
        return None
    confirmed = confirm_what_if(draft)
    write_json(destination / "what_if_confirmed_request.json", confirmed)
    logging.info("PUBLIC_ORCHESTRATOR WHAT_IF_EXECUTE confirmed=%s", confirmed.what_if_id)
    execution = orchestrator.run(AgentRunRequest(mode="WHAT_IF_EXECUTE", question=question,
                                what_if_request=confirmed, **common))
    if execution.what_if_package is None:
        write_json(destination / "execution_trace.json", _trace(execution))
        raise ValueError("PUBLIC_ORCHESTRATOR_EXECUTION_FAILED:" + str(execution.error_codes))
    package = execution.what_if_package
    write_json(destination / "execution_trace.json", _trace(execution))
    write_json(destination / "what_if_decision_package.json", package)
    write_json(destination / "hypothetical_request.json", package.modified_request)
    write_json(destination / "comparison.json", package.comparison)
    # Comparative explanation goes through the registered deterministic comparison tool.
    explanation = orchestrator.run(AgentRunRequest(mode="COMPARISON", question="So sánh orders và rủi ro What if",
        what_if_package=package, **common))
    write_json(destination / "comparison_trace.json", _trace(explanation))
    write_json(destination / "deterministic_answer.json", explanation.answer)
    planning_path = baseline / "m5/planning_config.json"
    _reject_reparse(planning_path)
    parent_config = load_planning_config(planning_path)[0]
    manifest = json.loads((baseline / "run_manifest.json").read_text(encoding="utf-8-sig")) if (baseline / "run_manifest.json").exists() else {}
    export_what_if_planning(package, parent_config, destination / "hypothetical",
                           source_bundle=manifest.get("bundle_path"), baseline_path=str(baseline.resolve()))
    # B rebinds certification identity only. Every physical P0 order and cost is retained.
    transformed = package.modified_request
    data = selected.plan.model_dump(mode="python")
    data.update(plan_id=transformed.request_id + "-P0-FIXED-EVALUATION", completed=False)
    data["provenance"].update(planning_binding=transformed.planning_binding,
        evaluation_lineage={"parent_plan_id": selected.plan.plan_id, "parent_plan_hash": content_hash(selected.plan),
                            "purpose": "P0_FIXED_ON_D1_NO_OPTIMIZATION", "physical_orders_unchanged": True})
    rebound = ProcurementPlan.model_validate(data)
    old = gateway.evaluate_plan(rebound, transformed)
    folder = destination / "old_plan_on_increased_demand"
    folder.mkdir()
    write_json(folder / "evaluation.json", old)
    write_json(folder / "input_plan.json", rebound)
    write_json(folder / "lineage.json", {"parent_plan_hash": content_hash(selected.plan), "rebound_plan_hash": content_hash(rebound),
        "physical_orders_hash": content_hash(rebound.orders), "parent_physical_orders_hash": content_hash(selected.plan.orders),
        "identity_fields_rebound": ["plan_id", "completed", "provenance.planning_binding", "provenance.evaluation_lineage"],
        "optimizer_called": False})
    assert content_hash(rebound.orders) == content_hash(selected.plan.orders)
    assert (content_hash(request), content_hash(result), sha256_content_hash(decision)) == (request_before, result_before, decision_before)
    write_json(destination / "gateway_trace.json", gateway.calls)
    write_json(destination / "runtime_result.json", {"status": package.optimization_result.status,
        "technical_outcome": package.optimization_result.technical_outcome, "recommended_strategy": package.optimization_result.recommended_strategy,
        "selected_mode": mode, "what_if_id": package.what_if_id, "package_hash": package.package_hash,
        "baseline_objects_unchanged": True, "hypothetical_computation_confirmed": True,
        "business_ready": False, "execution_authorized": False, "procurement_executed": False,
        "explain_mode": "deterministic", "llm_api_called": False})
    logging.info("WHAT_IF_FINISHED outcome=%s strategy=%s B_critic=%s", package.optimization_result.technical_outcome,
                 package.optimization_result.recommended_strategy, old.critic.passed)
    return package


def main(argv=None):
    p = argparse.ArgumentParser(description="Rule-based global ingredient demand What-if; no LLM")
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workspace-root", type=Path, default=ENGINE_ROOT)
    p.add_argument("--multiplier", type=float)
    p.add_argument("--scope", choices=["ALL_BASELINE_INGREDIENT_DEMAND"])
    p.add_argument("--text")
    p.add_argument("--execute-hypothetical", action="store_true")
    p.add_argument("--deny-network", action="store_true")
    p.add_argument("--actor", required=True)
    p.add_argument("--reason", required=True)
    p.add_argument("--idempotency-key", required=True)
    args = p.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
    try:
        if args.text and args.multiplier is not None:
            raise ValueError("CHOOSE_CANONICAL_INPUT_OR_TEXT_NOT_BOTH")
        if args.text:
            parsed = parse_demand_rule(args.text, scope=args.scope)
            if parsed.rule is None:
                print(parsed.model_dump_json()); return 2
            rule = parsed.rule
        else:
            rule = IngredientDemandRule(multiplier=args.multiplier, scope=args.scope)
        attempts = []
        if args.deny_network:
            def guard(event, values):
                if event in {"socket.connect", "socket.getaddrinfo", "socket.sendto"}:
                    attempts.append(event)
                    raise RuntimeError("NETWORK_DISABLED_FOR_RULEBASED_WHAT_IF:" + event)
            sys.addaudithook(guard)
        import shelfcash_forecast, shelfcash_pipeline
        print(json.dumps({"python": sys.executable, "forecast": shelfcash_forecast.__file__,
                          "pipeline": shelfcash_pipeline.__file__, "runtime": __file__,
                          "dont_write_bytecode": sys.dont_write_bytecode}), flush=True)
        run_demand_what_if(rule, args.baseline, args.output, actor=args.actor, reason=args.reason,
            idempotency_key=args.idempotency_key, execute_hypothetical=args.execute_hypothetical, workspace_root=args.workspace_root)
        write_json(args.output / "network_guard.json", {"enabled": args.deny_network, "attempts": attempts,
            "llm_api_called": False, "mechanism": "Python audit hook rejects outbound sockets when --deny-network"})
        return 0
    except Exception:
        logging.exception("RULEBASED_WHAT_IF_FAILED")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
