from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from shelfcash_forecast.decision_intelligence.evaluation.contracts import (
    AggregateMetricObservation,
    DecisionBenchmarkReport,
    OperationPerformance,
)
from shelfcash_forecast.decision_intelligence.integrity import sha256_content_hash


def benchmark_report_json(report: DecisionBenchmarkReport) -> str:
    return json.dumps(
        report.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    )


def benchmark_semantic_hash(report: DecisionBenchmarkReport) -> str:
    """Hash functional results while excluding machine-dependent runtime metadata."""

    material = report.model_dump(mode="json")
    material.pop("semantic_report_hash", None)
    material.pop("environment", None)
    material["aggregate"].pop("runtime", None)
    for case in material["cases"]:
        case.pop("performance", None)
    for performance in material["decision_performance"]:
        performance.pop("deserialization_validation_latency", None)
    return sha256_content_hash(material)


def baseline_vs_v2_markdown(
    baseline: Mapping[str, Any],
    current: DecisionBenchmarkReport,
) -> str:
    """Render a factual schema-aware v1/v2 comparison without changing either report."""

    v1 = baseline["aggregate"]
    v1_structured = v1["structured_retrieval"]
    v1_grounding = v1["grounding"]
    v2_structured = current.aggregate.structured_retrieval
    v2_grounding = current.aggregate.grounding
    rows = [
        ("Corpus cases", len(baseline["cases"]), len(current.cases)),
        ("Passed cases", v1["passed_case_count"], current.aggregate.passed_case_count),
        ("Failed cases", v1["failed_case_count"], current.aggregate.failed_case_count),
        ("Recall@5", v1_structured["recall_at_5"], v2_structured.recall_at_5.macro_value),
        (
            "Required-type Recall@5",
            v1_structured["required_evidence_type_recall_at_5"],
            v2_structured.required_evidence_type_recall_at_5.macro_value,
        ),
        (
            "Insufficient-evidence abstention",
            v1_grounding["insufficient_evidence_abstention_accuracy"],
            v2_grounding.insufficient_evidence_abstention_accuracy,
        ),
        (
            "Visible text/facts consistency",
            v1_grounding["visible_text_structured_facts_consistency"],
            v2_grounding.visible_text_structured_facts_consistency,
        ),
    ]
    lines = [
        "# ShelfCash M6 Part 1.2 Baseline versus Part 1.2.1",
        "",
        f"- Baseline corpus: `{baseline['corpus_version']}`",
        f"- Current corpus: `{current.corpus_version}`",
        "- Metrics compare retrieval/explanation quality, not forecast or procurement accuracy.",
        "",
        "| Metric | v1 | v2 |",
        "|---|---:|---:|",
    ]
    lines.extend(f"| {name} | {old} | {new} |" for name, old, new in rows)
    lines.extend(
        [
            "",
            "## Methodology changes",
            "",
            "- Zero-gold Recall/MRR are excluded as NOT_APPLICABLE.",
            "- Gold labels use typed content selectors with cardinality validation.",
            "- Artifact rejection executes the public coherence/build path.",
            "- Structured-with-graph is compared with structured-without-graph and lexical-only.",
            "- Latency is separated by operation.",
            "",
            f"Current overall pass: **{'PASS' if current.overall_pass else 'FAIL'}**.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_baseline_comparison(
    baseline: Mapping[str, Any],
    current: DecisionBenchmarkReport,
    *,
    markdown_path: str | Path,
) -> Path:
    target = Path(markdown_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(baseline_vs_v2_markdown(baseline, current), encoding="utf-8")
    return target


def _metric(value: AggregateMetricObservation) -> str:
    if value.macro_value is None:
        return "N/A"
    return f"{value.macro_value:.4f} ({value.applicable_case_count} applicable)"


def _operation_lines(performance: OperationPerformance | None) -> list[str]:
    if performance is None:
        return ["| Not measured | N/A | N/A | N/A | N/A | N/A |"]
    latency = performance.latency
    return [
        (
            f"| {performance.operation} | {latency.sample_count} | "
            f"{latency.minimum_ms:.3f} | {latency.p50_ms:.3f} | "
            f"{latency.p95_ms:.3f} | {latency.maximum_ms:.3f} |"
        )
    ]


def benchmark_report_markdown(report: DecisionBenchmarkReport) -> str:
    aggregate = report.aggregate
    structured = aggregate.structured_retrieval
    no_graph = aggregate.structured_without_graph_retrieval
    lexical = aggregate.lexical_only_retrieval
    grounding = aggregate.grounding
    lines = [
        "# ShelfCash M6 Decision Intelligence Benchmark",
        "",
        f"- Corpus: `{report.corpus_version}`",
        f"- Passed cases: {aggregate.passed_case_count}",
        f"- Failed cases: {aggregate.failed_case_count}",
        f"- Failed critical cases: {aggregate.failed_critical_case_count}",
        f"- Overall pass: **{'PASS' if report.overall_pass else 'FAIL'}**",
        f"- Semantic report SHA-256: `{report.semantic_report_hash}`",
        "",
        "## Retrieval",
        "",
        "| Metric | Structured + graph | Structured without graph | Lexical only |",
        "|---|---:|---:|---:|",
        f"| Recall@1 | {_metric(structured.recall_at_1)} | {_metric(no_graph.recall_at_1)} | {_metric(lexical.recall_at_1)} |",
        f"| Recall@3 | {_metric(structured.recall_at_3)} | {_metric(no_graph.recall_at_3)} | {_metric(lexical.recall_at_3)} |",
        f"| Recall@5 | {_metric(structured.recall_at_5)} | {_metric(no_graph.recall_at_5)} | {_metric(lexical.recall_at_5)} |",
        f"| Precision@5 | {_metric(structured.precision_at_5)} | {_metric(no_graph.precision_at_5)} | {_metric(lexical.precision_at_5)} |",
        f"| MRR | {_metric(structured.mean_reciprocal_rank)} | {_metric(no_graph.mean_reciprocal_rank)} | {_metric(lexical.mean_reciprocal_rank)} |",
        f"| Required-type Recall@5 | {_metric(structured.required_evidence_type_recall_at_5)} | {_metric(no_graph.required_evidence_type_recall_at_5)} | {_metric(lexical.required_evidence_type_recall_at_5)} |",
        "",
        "Recall values exclude zero-gold abstention cases; numerator/denominator and excluded counts are present in JSON.",
        "",
        "## Graph ablation",
        "",
        f"- Relevant graph-only IDs in top-5: {aggregate.graph_ablation.relevant_graph_only_id_count}",
        f"- Macro Recall@5 delta: {aggregate.graph_ablation.recall_at_5_delta:.4f}",
        f"- Macro MRR delta: {aggregate.graph_ablation.mrr_delta:.4f}",
        f"- Improved/worsened/unchanged cases: {aggregate.graph_ablation.improved_case_count} / {aggregate.graph_ablation.worsened_case_count} / {aggregate.graph_ablation.unchanged_case_count}",
        "",
        "## Grounding and authority",
        "",
        f"- Normal guard success: {grounding.normal_guard_success:.4f}",
        f"- Citation validity/completeness: {grounding.citation_validity:.4f} / {grounding.citation_completeness:.4f}",
        f"- Structured fact fidelity: {grounding.structured_fact_fidelity:.4f}",
        f"- Visible text/facts consistency: {grounding.visible_text_structured_facts_consistency:.4f}",
        f"- Recommendation/order/recourse fidelity: {grounding.recommendation_fidelity:.4f} / {grounding.immediate_order_fidelity:.4f} / {grounding.first_stage_recourse_separation_accuracy:.4f}",
        f"- Unsupported/insufficient abstention: {grounding.unsupported_intent_abstention_accuracy:.4f} / {grounding.insufficient_evidence_abstention_accuracy:.4f}",
        f"- Probability/stress/causal violation rates: {grounding.probability_semantic_violation_rate:.4f} / {grounding.stress_as_probability_violation_rate:.4f} / {grounding.causal_attribution_violation_rate:.4f}",
        "",
        "## Per-language results",
        "",
        "| Language | Cases | Pass rate | Recall@5 | Required-type Recall@5 | Intent | Abstention | Guard | Localization |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for language in report.per_language:
        lines.append(
            f"| {language.language} | {language.case_count} | {language.pass_rate:.4f} | "
            f"{_metric(language.recall_at_5)} | "
            f"{_metric(language.required_evidence_type_recall_at_5)} | "
            f"{language.intent_accuracy:.4f} | "
            f"{language.abstention_accuracy if language.abstention_accuracy is not None else 'N/A'} | "
            f"{language.grounding_guard_success if language.grounding_guard_success is not None else 'N/A'} | "
            "Not evaluated |"
        )
    lines.extend(
        [
            "",
            "## Per-case results",
            "",
            "| Case | Language | Category | Result | Failures |",
            "|---|---|---|---|---|",
        ]
    )
    for case in report.cases:
        failures = "; ".join(case.failures) if case.failures else "—"
        lines.append(
            f"| `{case.case_id}` | {case.language} | {case.category} | "
            f"{'PASS' if case.passed else 'FAIL'} | {failures} |"
        )
    lines.extend(
        [
            "",
            "## Performance",
            "",
            "| Operation | Samples | Min ms | p50 ms | p95 ms | Max ms |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    runtime = aggregate.runtime
    for operation in (
        runtime.full_package_build,
        runtime.package_deserialization_validation,
        runtime.retrieval,
        runtime.generation_and_guard,
        runtime.scale_fixture_materialization,
    ):
        lines.extend(_operation_lines(operation))
    lines.extend(["", "## Acceptance gates", ""])
    for gate in report.acceptance.gates:
        observed = "N/A" if gate.observed_value is None else f"{gate.observed_value:g}"
        lines.append(
            f"- {gate.metric_name}: {'PASS' if gate.passed else 'FAIL'} "
            f"(observed {observed}, {gate.operator} {gate.target:g}, n={gate.evaluated_case_count})"
        )
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {limitation}" for limitation in report.limitations)
    return "\n".join(lines) + "\n"


def write_benchmark_reports(
    report: DecisionBenchmarkReport,
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> tuple[Path, Path]:
    """Write reports only when explicitly requested by the benchmark caller."""

    json_target = Path(json_path)
    markdown_target = Path(markdown_path)
    json_target.parent.mkdir(parents=True, exist_ok=True)
    markdown_target.parent.mkdir(parents=True, exist_ok=True)
    json_target.write_text(benchmark_report_json(report), encoding="utf-8")
    markdown_target.write_text(benchmark_report_markdown(report), encoding="utf-8")
    return json_target, markdown_target
