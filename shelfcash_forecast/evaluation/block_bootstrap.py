from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


BOOTSTRAP_METHOD = "circular_fixed_length_target_date_blocks_v1"


@dataclass(frozen=True)
class ClusterBootstrapPlan:
    """A reproducible set of sampled cluster positions.

    Every source block has exactly ``block_length`` positions. Circular wrap
    avoids the shortened tail blocks produced by ordinary slicing. A replicate
    is trimmed only after concatenating complete blocks to the requested
    number of clusters.
    """

    method: str
    cluster_labels: tuple[str, ...]
    block_length: int
    replications: int
    seed: int
    sampled_cluster_positions: np.ndarray

    @property
    def cluster_count(self) -> int:
        return len(self.cluster_labels)


def normalized_target_dates(values: Iterable[object]) -> pd.DatetimeIndex:
    parsed = pd.to_datetime(pd.Index(values), errors="coerce")
    if parsed.isna().any():
        raise ValueError("Bootstrap target_date clusters must all be valid")
    return pd.DatetimeIndex(parsed).normalize()


def circular_fixed_length_blocks(
    cluster_count: int,
    block_length: int,
) -> np.ndarray:
    """Return all circular blocks; every row is exactly ``block_length``."""

    if cluster_count < 1:
        raise ValueError("cluster_count must be positive")
    if block_length < 1:
        raise ValueError("block_length must be positive")
    starts = np.arange(cluster_count, dtype=np.int64)[:, None]
    offsets = np.arange(block_length, dtype=np.int64)[None, :]
    return (starts + offsets) % cluster_count


def build_cluster_bootstrap_plan(
    cluster_labels: Iterable[object],
    *,
    block_length: int,
    replications: int,
    seed: int,
) -> ClusterBootstrapPlan:
    dates = normalized_target_dates(cluster_labels)
    unique_dates = pd.DatetimeIndex(sorted(dates.unique()))
    if replications < 1:
        raise ValueError("replications must be positive")
    blocks = circular_fixed_length_blocks(len(unique_dates), block_length)
    blocks_per_replication = int(np.ceil(len(unique_dates) / block_length))
    rng = np.random.default_rng(seed)
    starts = rng.integers(
        0,
        len(unique_dates),
        size=(replications, blocks_per_replication),
    )
    sampled = blocks[starts].reshape(replications, -1)[:, : len(unique_dates)]
    return ClusterBootstrapPlan(
        method=BOOTSTRAP_METHOD,
        cluster_labels=tuple(date.date().isoformat() for date in unique_dates),
        block_length=block_length,
        replications=replications,
        seed=seed,
        sampled_cluster_positions=sampled,
    )


def cluster_row_indices(
    target_dates: Iterable[object],
) -> tuple[tuple[str, ...], Mapping[int, np.ndarray]]:
    dates = normalized_target_dates(target_dates)
    unique_dates = pd.DatetimeIndex(sorted(dates.unique()))
    labels = tuple(date.date().isoformat() for date in unique_dates)
    rows = {
        position: np.flatnonzero(dates == date)
        for position, date in enumerate(unique_dates)
    }
    return labels, rows


def sampled_row_indices(
    target_dates: Iterable[object],
    plan: ClusterBootstrapPlan,
) -> Iterable[np.ndarray]:
    labels, by_position = cluster_row_indices(target_dates)
    if labels != plan.cluster_labels:
        raise ValueError(
            "Bootstrap plan clusters do not match target_date clusters: "
            f"plan={plan.cluster_labels}, data={labels}"
        )
    for sampled_positions in plan.sampled_cluster_positions:
        yield np.concatenate(
            [by_position[int(position)] for position in sampled_positions]
        )


def validate_common_key_matrix(
    long_predictions: pd.DataFrame,
    *,
    key_columns: list[str],
    candidate_method: str,
    baseline_methods: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    required = set(key_columns) | {"method", "prediction", "actual"}
    missing = sorted(required - set(long_predictions.columns))
    if missing:
        raise ValueError(f"Uncertainty input columns are missing: {missing}")
    methods = [candidate_method, *baseline_methods]
    scoped = long_predictions.loc[long_predictions["method"].isin(methods)].copy()
    duplicates = scoped.duplicated(key_columns + ["method"], keep=False)
    if duplicates.any():
        raise ValueError("Uncertainty input has duplicate method/task keys")
    actual_counts = scoped.groupby(key_columns, observed=True)["actual"].nunique(
        dropna=False
    )
    if actual_counts.ne(1).any():
        raise ValueError("Actual values disagree across methods on common keys")
    pivot = scoped.pivot(index=key_columns, columns="method", values="prediction")
    absent_methods = sorted(set(methods) - set(pivot.columns))
    if absent_methods:
        raise ValueError(f"Uncertainty methods are missing: {absent_methods}")
    pivot = pivot[methods]
    if pivot.isna().any().any():
        raise ValueError("Uncertainty pivot is not a complete common-key matrix")
    actual = scoped.groupby(key_columns, observed=True)["actual"].first()
    actual = actual.reindex(pivot.index)
    if actual.isna().any() or not pivot.index.equals(actual.index):
        raise ValueError("Uncertainty actuals do not align to common keys")
    return pivot, actual


def paired_wape_block_bootstrap(
    long_predictions: pd.DataFrame,
    *,
    key_columns: list[str],
    candidate_method: str,
    baseline_methods: list[str],
    plan: ClusterBootstrapPlan,
    familywise_alpha: float = 0.05,
) -> list[dict[str, object]]:
    if not baseline_methods:
        raise ValueError("At least one baseline comparison is required")
    pivot, actual = validate_common_key_matrix(
        long_predictions,
        key_columns=key_columns,
        candidate_method=candidate_method,
        baseline_methods=baseline_methods,
    )
    task_dates = pivot.index.get_level_values("target_date")
    row_samples = list(sampled_row_indices(task_dates, plan))
    candidate = pivot[candidate_method].to_numpy(dtype=float)
    actual_values = actual.to_numpy(dtype=float)
    individual_alpha = familywise_alpha / len(baseline_methods)
    output: list[dict[str, object]] = []
    for baseline_method in baseline_methods:
        baseline = pivot[baseline_method].to_numpy(dtype=float)
        estimates = np.empty(plan.replications, dtype=float)
        for replication, indices in enumerate(row_samples):
            denominator = np.abs(actual_values[indices]).sum()
            if not np.isfinite(denominator) or denominator <= 0:
                raise ValueError("Bootstrap WAPE denominator must be positive")
            candidate_wape = (
                np.abs(candidate[indices] - actual_values[indices]).sum()
                / denominator
            )
            baseline_wape = (
                np.abs(baseline[indices] - actual_values[indices]).sum()
                / denominator
            )
            estimates[replication] = baseline_wape - candidate_wape
        point_denominator = np.abs(actual_values).sum()
        point = (
            np.abs(baseline - actual_values).sum()
            - np.abs(candidate - actual_values).sum()
        ) / point_denominator
        output.append(
            {
                "candidate_method": candidate_method,
                "baseline_method": baseline_method,
                "sampler": plan.method,
                "block_days": plan.block_length,
                "replications": plan.replications,
                "seed": plan.seed,
                "target_date_clusters": plan.cluster_count,
                "tasks": len(pivot),
                "wape_baseline_minus_candidate": float(point),
                "familywise_alpha": familywise_alpha,
                "comparison_count": len(baseline_methods),
                "bonferroni_individual_two_sided_alpha": individual_alpha,
                "ci_lower": float(np.quantile(estimates, individual_alpha / 2)),
                "ci_upper": float(
                    np.quantile(estimates, 1 - individual_alpha / 2)
                ),
                "evidence_scope": "DESCRIPTIVE_EXPOSED_HOLDOUT_ONLY",
            }
        )
    return output


def interval_block_confidence(
    *,
    target_dates: Iterable[object],
    coverage_indicator: np.ndarray,
    candidate_interval_scores: np.ndarray,
    reference_interval_scores: np.ndarray,
    plan: ClusterBootstrapPlan,
    alpha: float = 0.05,
) -> dict[str, object]:
    coverage = np.asarray(coverage_indicator, dtype=float)
    candidate_scores = np.asarray(candidate_interval_scores, dtype=float)
    reference_scores = np.asarray(reference_interval_scores, dtype=float)
    if not (
        len(coverage) == len(candidate_scores) == len(reference_scores)
    ):
        raise ValueError("Interval bootstrap arrays must have equal lengths")
    samples = list(sampled_row_indices(target_dates, plan))
    coverage_estimates = np.empty(plan.replications, dtype=float)
    score_differences = np.empty(plan.replications, dtype=float)
    for replication, indices in enumerate(samples):
        coverage_estimates[replication] = coverage[indices].mean()
        score_differences[replication] = (
            candidate_scores[indices].mean() - reference_scores[indices].mean()
        )
    return {
        "sampler": plan.method,
        "block_days": plan.block_length,
        "replications": plan.replications,
        "seed": plan.seed,
        "target_date_clusters": plan.cluster_count,
        "individual_two_sided_alpha": alpha,
        "candidate_coverage_ci_lower": float(
            np.quantile(coverage_estimates, alpha / 2)
        ),
        "candidate_coverage_ci_upper": float(
            np.quantile(coverage_estimates, 1 - alpha / 2)
        ),
        "candidate_minus_reference_interval_score_ci_lower": float(
            np.quantile(score_differences, alpha / 2)
        ),
        "candidate_minus_reference_interval_score_ci_upper": float(
            np.quantile(score_differences, 1 - alpha / 2)
        ),
        "evidence_scope": "DESCRIPTIVE_EXPOSED_HOLDOUT_ONLY",
    }
