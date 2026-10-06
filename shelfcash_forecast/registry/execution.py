from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


FULL_RUN_PHASES = (
    "load_canonical_bundle",
    "prepare_causal_modelling_table",
    "bounded_development_search",
    "final_refit_point_cqr_and_retrospective_benchmark",
    "independent_metric_recomputation",
    "fresh_process_roundtrip_and_future_forecast",
    "offline_verification",
    "report_and_review_package",
)


def append_execution_event(path: Path, event: dict[str, Any]) -> None:
    """Append an immutable execution event without rewriting prior attempts."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(event, ensure_ascii=False, sort_keys=True))
        stream.write("\n")


def load_execution_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def source_tree_hash(roots: Iterable[Path], *, base: Path) -> str:
    """Hash relative paths and file bytes for a deterministic source identity."""

    digest = hashlib.sha256()
    paths: list[Path] = []
    for root in roots:
        if root.is_file():
            paths.append(root)
        elif root.is_dir():
            paths.extend(
                path
                for path in root.rglob("*")
                if path.is_file()
                and "__pycache__" not in path.parts
                and path.suffix.lower() not in {".pyc", ".pyo"}
            )
    for path in sorted(set(paths), key=lambda item: item.relative_to(base).as_posix()):
        relative = path.relative_to(base).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        payload = path.read_bytes()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def derive_attempt_verification(
    events: list[dict[str, Any]],
    phase_records: list[dict[str, Any]],
    attempt_id: str,
) -> dict[str, Any]:
    attempt_events = [
        event for event in events if event.get("attempt_id") == attempt_id
    ]
    starts = [event for event in attempt_events if event.get("event") == "started"]
    finishes = [event for event in attempt_events if event.get("event") == "finished"]
    mode = starts[-1].get("mode") if starts else None
    exit_code = finishes[-1].get("exit_code") if finishes else None
    completed = {
        record.get("phase")
        for record in phase_records
        if record.get("attempt_id") == attempt_id
        and record.get("exit_code") == 0
        and record.get("status") != "REUSED_FROZEN_OUTPUTS"
    }
    missing = sorted(set(FULL_RUN_PHASES) - completed)
    full_verified = mode == "full" and exit_code == 0 and not missing
    if full_verified:
        status = "FULL_RUN_EXECUTED_VERIFIED"
    elif mode == "resume" and exit_code == 0:
        status = "RESUME_COMPLETED_NOT_FULL_RUN_VERIFIED"
    elif exit_code is None:
        status = "INCOMPLETE_NO_FINISH_EVENT"
    else:
        status = "FAILED_OR_INCOMPLETE"
    return {
        "attempt_id": attempt_id,
        "status": status,
        "mode": mode,
        "exit_code": exit_code,
        "required_full_run_phases": list(FULL_RUN_PHASES),
        "completed_non_reused_phases": sorted(completed),
        "missing_full_run_phases": missing,
        "derived_only_from_execution_journal": True,
    }
