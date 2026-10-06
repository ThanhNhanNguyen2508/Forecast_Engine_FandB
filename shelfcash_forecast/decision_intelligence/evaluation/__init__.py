"""Offline deterministic evaluation tools for M6 Decision Intelligence."""

from shelfcash_forecast.decision_intelligence.evaluation.contracts import (
    AcceptanceGate,
    AcceptanceSummary,
    DecisionBenchmarkCorpus,
    DecisionBenchmarkReport,
    DecisionEvaluationCase,
    ExpectedStructuredFact,
    GoldEvidenceSelector,
    MetricObservation,
)
from shelfcash_forecast.decision_intelligence.evaluation.corpus import (
    CURATED_CORPUS_V1_VERSION,
    CURATED_CORPUS_VERSION,
    curated_benchmark_corpus,
    curated_benchmark_corpus_v1,
)
from shelfcash_forecast.decision_intelligence.evaluation.fixtures import (
    build_quantile_decision_fixture,
    build_scaled_decision_fixture,
)
from shelfcash_forecast.decision_intelligence.evaluation.gold import (
    GoldEvidenceResolutionError,
    resolve_gold_evidence,
)
from shelfcash_forecast.decision_intelligence.evaluation.manifest import (
    build_source_manifest,
    current_source_records,
    source_manifest_json,
    write_source_manifest,
)
from shelfcash_forecast.decision_intelligence.evaluation.metrics import (
    aggregate_observations,
    aggregate_retrieval_metrics,
    latency_summary,
    percentile,
    retrieval_metrics,
    safe_ratio,
)
from shelfcash_forecast.decision_intelligence.evaluation.reporting import (
    baseline_vs_v2_markdown,
    benchmark_report_json,
    benchmark_report_markdown,
    benchmark_semantic_hash,
    write_baseline_comparison,
    write_benchmark_reports,
)
from shelfcash_forecast.decision_intelligence.evaluation.retrievers import (
    LexicalOnlyRetriever,
)
from shelfcash_forecast.decision_intelligence.evaluation.runner import (
    run_decision_intelligence_benchmark,
)

__all__ = [
    "CURATED_CORPUS_V1_VERSION",
    "CURATED_CORPUS_VERSION",
    "AcceptanceGate",
    "AcceptanceSummary",
    "DecisionBenchmarkCorpus",
    "DecisionBenchmarkReport",
    "DecisionEvaluationCase",
    "ExpectedStructuredFact",
    "GoldEvidenceResolutionError",
    "GoldEvidenceSelector",
    "LexicalOnlyRetriever",
    "MetricObservation",
    "aggregate_observations",
    "aggregate_retrieval_metrics",
    "baseline_vs_v2_markdown",
    "benchmark_report_json",
    "benchmark_report_markdown",
    "benchmark_semantic_hash",
    "build_quantile_decision_fixture",
    "build_scaled_decision_fixture",
    "build_source_manifest",
    "curated_benchmark_corpus",
    "curated_benchmark_corpus_v1",
    "current_source_records",
    "latency_summary",
    "percentile",
    "resolve_gold_evidence",
    "retrieval_metrics",
    "run_decision_intelligence_benchmark",
    "safe_ratio",
    "source_manifest_json",
    "write_baseline_comparison",
    "write_benchmark_reports",
    "write_source_manifest",
]
