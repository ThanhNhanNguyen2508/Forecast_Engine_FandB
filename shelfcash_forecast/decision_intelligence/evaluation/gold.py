from __future__ import annotations

from dataclasses import dataclass

from shelfcash_forecast.decision_intelligence.contracts import EvidenceItem, EvidencePackage
from shelfcash_forecast.decision_intelligence.evaluation.contracts import GoldEvidenceSelector


@dataclass(frozen=True)
class GoldResolution:
    evidence_ids: tuple[str, ...]
    errors: tuple[str, ...]


class GoldEvidenceResolutionError(ValueError):
    """Raised when a content selector is missing or labels an ambiguous evidence set."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = tuple(sorted(set(errors)))
        super().__init__(";".join(self.errors))


def _entity_matches(
    item: EvidenceItem,
    key: str,
    expected: str,
    *,
    allow_missing: bool,
) -> bool:
    actual = item.entities.get(key)
    if actual is None:
        return allow_missing
    return actual == expected


def selector_matches(item: EvidenceItem, selector: GoldEvidenceSelector) -> bool:
    if item.evidence_type != selector.evidence_type:
        return False
    if selector.strategy is not None and not _entity_matches(
        item,
        "strategy",
        selector.strategy,
        allow_missing=selector.allow_missing_entity_key,
    ):
        return False
    if any(
        not _entity_matches(
            item,
            key,
            expected,
            allow_missing=selector.allow_missing_entity_key,
        )
        for key, expected in selector.entities.items()
    ):
        return False
    if selector.source_object is not None and item.source_object != selector.source_object:
        return False
    if selector.source_path is not None and item.source_path != selector.source_path:
        return False
    if selector.source_path_prefix is not None and not item.source_path.startswith(
        selector.source_path_prefix
    ):
        return False
    if selector.semantics is not None and item.semantics != selector.semantics:
        return False
    if any(item.payload.get(key) != value for key, value in selector.payload_equals.items()):
        return False
    for key, threshold in selector.payload_greater_than.items():
        value = item.payload.get(key)
        if value is None or not isinstance(value, int | float) or float(value) <= threshold:
            return False
    return True


def resolve_gold_evidence(
    selectors: list[GoldEvidenceSelector],
    evidence: EvidencePackage,
    *,
    strict: bool = True,
) -> GoldResolution:
    resolved: set[str] = set()
    errors: list[str] = []
    for index, selector in enumerate(selectors):
        matches = sorted(
            item.evidence_id for item in evidence.items if selector_matches(item, selector)
        )
        count = len(matches)
        if count < selector.minimum_matches:
            errors.append(
                f"GOLD_SELECTOR_ZERO_OR_TOO_FEW:{index}:{selector.evidence_type}:"
                f"{count}<{selector.minimum_matches}"
            )
        if selector.maximum_matches is not None and count > selector.maximum_matches:
            errors.append(
                f"GOLD_SELECTOR_AMBIGUOUS:{index}:{selector.evidence_type}:"
                f"{count}>{selector.maximum_matches}"
            )
        resolved.update(matches)
    if errors and strict:
        raise GoldEvidenceResolutionError(errors)
    return GoldResolution(tuple(sorted(resolved)), tuple(sorted(errors)))
