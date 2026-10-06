from __future__ import annotations

from pathlib import Path

from shelfcash_forecast.decision_intelligence.evaluation_part2.contracts import (
    Part2BenchmarkReport,
)
from shelfcash_forecast.decision_intelligence.integrity import sha256_content_hash


def part2_semantic_hash(report: Part2BenchmarkReport) -> str:
    return sha256_content_hash(
        report.model_dump(
            mode="json",
            exclude={"environment", "performance", "semantic_hash"},
        )
    )


def part2_report_json(report: Part2BenchmarkReport) -> str:
    return report.model_dump_json(indent=2)


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def part2_report_markdown(report: Part2BenchmarkReport) -> str:
    metrics = report.aggregate_metrics
    lines = [
        "# ShelfCash M6 Part 2 deterministic evaluation",
        "",
        f"- Corpus: `{report.corpus_version}`",
        f"- Cases: {len(report.case_results)}",
        f"- Passed: {sum(row.passed for row in report.case_results)}",
        f"- Failed critical: {len(report.failed_critical_cases)}",
        f"- Overall pass: `{report.overall_pass}`",
        f"- Semantic hash: `{report.semantic_hash}`",
        "",
        "## Authority and grounding metrics",
        "",
        f"- Computation authority fidelity: {_pct(metrics.computation_authority_fidelity.value)}",
        f"- Recommendation fidelity: {_pct(metrics.recommendation_fidelity.value)}",
        f"- Order/recourse fidelity: {_pct(metrics.order_recourse_fidelity.value)}",
        f"- No-mutation accuracy: {_pct(metrics.no_mutation_accuracy.value)}",
        f"- Approval transition/binding: {_pct(metrics.approval_transition_fidelity.value)}",
        f"- Citation validity: {_pct(metrics.citation_validity.value)}",
        f"- Citation completeness: {_pct(metrics.citation_completeness.value)}",
        f"- Adversarial guard rejection: {_pct(metrics.adversarial_guard_rejection.value)}",
        f"- Probability violation rate: {_pct(metrics.probability_violation_rate.value)}",
        f"- Stress-as-probability rate: {_pct(metrics.stress_as_probability_violation_rate.value)}",
        f"- Causal violation rate: {_pct(metrics.causal_violation_rate.value)}",
        "",
        "## Per-language query routing",
        "",
        "| Language | Cases | Passed | Pass rate | Intent accuracy |",
        "|---|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {row.language} | {row.case_count} | {row.passed_count} | "
        f"{_pct(row.pass_rate)} | {_pct(row.intent_accuracy)} |"
        for row in report.per_language
    )
    lines.extend(
        [
            "",
            "The renderer is English-only; Vietnamese cases evaluate query understanding, not answer localization.",
            "",
            "## Cases",
            "",
            "| Case | Category | Language | Status | Pass | Failures |",
            "|---|---|---|---|---|---|",
        ]
    )
    lines.extend(
        f"| {row.case_id} | {row.category} | {row.language} | {row.observed_status} | "
        f"{row.passed} | {', '.join(row.failures) or '-'} |"
        for row in report.case_results
    )
    lines.extend(
        [
            "",
            "## Acceptance gates",
            "",
            "| Metric | Observed | Target | Pass |",
            "|---|---:|---:|---|",
        ]
    )
    lines.extend(
        f"| {gate.metric_name} | {gate.observed_value:.6f} | "
        f"{gate.operator} {gate.target:.6f} | {gate.passed} |"
        for gate in report.acceptance_gates
    )
    lines.extend(["", "## Operation-specific latency", ""])
    lines.extend(
        f"- {row.operation}: n={row.sample_count}, p50={row.p50_ms:.3f} ms, "
        f"p95={row.p95_ms:.3f} ms ({row.description})"
        for row in report.performance
    )
    lines.extend(["", "## Explicit limitations", ""])
    lines.extend(f"- {item}" for item in report.limitations)
    return "\n".join(lines) + "\n"


def write_part2_reports(
    report: Part2BenchmarkReport,
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    Path(json_path).write_text(part2_report_json(report) + "\n", encoding="utf-8")
    Path(markdown_path).write_text(part2_report_markdown(report), encoding="utf-8")


__all__ = [
    "part2_report_json",
    "part2_report_markdown",
    "part2_semantic_hash",
    "write_part2_reports",
]
