"""Milestone 1: raw files -> a fresh sealed bundle, or validate an existing one."""
from __future__ import annotations

import os
import json
from pathlib import Path

from shelfcash_preprocess.models import RunContext
from shelfcash_preprocess.pipeline import PreprocessService, load_bundle
from shelfcash_pipeline.context import RunOptions, write_json


def run(options: RunOptions, output: Path) -> Path:
    if options.bundle_path is not None:
        bundle = load_bundle(options.bundle_path)
        source = "existing_bundle_read_only"
    else:
        assert options.input_path is not None
        # Import profiles/SQLite ledger belong to this new run, never old state.
        previous = os.environ.get("SHELFCASH_PREPROCESS_STATE_DIR")
        os.environ["SHELFCASH_PREPROCESS_STATE_DIR"] = str(output / "state")
        try:
            metadata = (
                {} if options.context_metadata is None
                else json.loads(options.context_metadata.read_text(encoding="utf-8-sig"))
            )
            if not isinstance(metadata, dict):
                raise ValueError("Context metadata must be a JSON object.")
            bundle = PreprocessService.from_config().run(
                options.input_path,
                output / "preprocess" / "bundle",
                context=RunContext(
                    store_id=options.store_id,
                    tenant_id=output.name,
                    cutoff_date=options.cutoff_date,
                    date_locale=options.date_locale,
                    metadata=metadata,
                ),
                llm_mode="offline",
            )
        finally:
            if previous is None:
                os.environ.pop("SHELFCASH_PREPROCESS_STATE_DIR", None)
            else:
                os.environ["SHELFCASH_PREPROCESS_STATE_DIR"] = previous
        source = "fresh_raw_preprocess_offline"
    write_json(output / "preprocess_summary.json", {
        "source": source,
        "bundle_path": str(bundle.run_dir.resolve()),
        "bundle_complete": bundle.manifest.bundle_complete,
        "readiness": {
            name: value.model_dump(mode="json")
            for name, value in bundle.manifest.readiness.items()
        },
        "warnings": bundle.manifest.warnings,
        "review_applied": False,
        "context_metadata_file": None if options.context_metadata is None else str(options.context_metadata.resolve()),
        "context_metadata": bundle.manifest.context.metadata,
    })
    return bundle.run_dir
