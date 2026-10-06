from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def project_root() -> Path:
    override = os.environ.get("SHELFCASH_ENGINE_ROOT")
    if override:
        return Path(override).resolve()
    source_root = Path(__file__).resolve().parent.parent
    editable_engine_root = source_root.parent
    if source_root.name == "source_code":
        return editable_engine_root
    return Path.cwd().resolve()


def default_env_path() -> Path:
    return project_root() / ".env.preprocess"


def _read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


@dataclass(frozen=True)
class Settings:
    config_path: Path
    api_key: str | None
    model: str = ""
    reasoning_effort: str = "medium"
    llm_mode: str = "offline"
    m6_llm_enabled: bool = False
    m6_model: str = ""
    m6_fallback_to_deterministic: bool = False
    timeout_seconds: float = 45.0
    max_retries: int = 2
    max_output_tokens: int = 6000
    max_files: int = 250
    max_file_bytes: int = 100 * 1024 * 1024
    max_total_input_bytes: int = 1024 * 1024 * 1024
    max_archive_bytes: int = 500 * 1024 * 1024
    max_table_rows: int = 1_000_000
    max_table_columns: int = 10_000
    max_table_cells: int = 10_000_000
    max_pdf_pages: int = 500
    max_image_pixels: int = 50_000_000

    @property
    def key_status(self) -> str:
        if not self.api_key:
            return "MISSING"
        if self.api_key == "YOUR_OPENAI_API_KEY" or len(self.api_key) < 20:
            return "PLACEHOLDER_INVALID"
        return "SET"


def load_settings(config_path: str | Path | None = None) -> Settings:
    path = Path(config_path).resolve() if config_path else default_env_path()
    file_values = _read_env(path)

    def value(name: str, default: str | None = None) -> str | None:
        return os.environ.get(name, file_values.get(name, default))

    return Settings(
        config_path=path,
        api_key=value("OPENAI_API_KEY"),
        model=str(value("SHELFCASH_PREPROCESS_MODEL", "")),
        reasoning_effort=str(
            value("SHELFCASH_PREPROCESS_REASONING_EFFORT", "medium")
        ),
        llm_mode=str(value("SHELFCASH_PREPROCESS_LLM_MODE", "offline")),
        m6_llm_enabled=str(value("SHELFCASH_M6_LLM_ENABLED", "false")).casefold()
        in {"1", "true", "yes"},
        m6_model=str(value("SHELFCASH_M6_MODEL", "")),
        m6_fallback_to_deterministic=str(
            value("SHELFCASH_M6_FALLBACK_TO_DETERMINISTIC", "false")
        ).casefold()
        in {"1", "true", "yes"},
        timeout_seconds=float(value("SHELFCASH_PREPROCESS_TIMEOUT_SECONDS", "45")),
        max_retries=int(value("SHELFCASH_PREPROCESS_MAX_RETRIES", "2")),
        max_output_tokens=int(
            value("SHELFCASH_PREPROCESS_MAX_OUTPUT_TOKENS", "6000")
        ),
        max_files=int(value("SHELFCASH_PREPROCESS_MAX_FILES", "250")),
        max_file_bytes=int(
            value("SHELFCASH_PREPROCESS_MAX_FILE_BYTES", str(100 * 1024 * 1024))
        ),
        max_total_input_bytes=int(
            value(
                "SHELFCASH_PREPROCESS_MAX_TOTAL_INPUT_BYTES",
                str(1024 * 1024 * 1024),
            )
        ),
        max_archive_bytes=int(
            value(
                "SHELFCASH_PREPROCESS_MAX_ARCHIVE_BYTES",
                str(500 * 1024 * 1024),
            )
        ),
        max_table_rows=int(
            value("SHELFCASH_PREPROCESS_MAX_TABLE_ROWS", "1000000")
        ),
        max_table_columns=int(
            value("SHELFCASH_PREPROCESS_MAX_TABLE_COLUMNS", "10000")
        ),
        max_table_cells=int(
            value("SHELFCASH_PREPROCESS_MAX_TABLE_CELLS", "10000000")
        ),
        max_pdf_pages=int(
            value("SHELFCASH_PREPROCESS_MAX_PDF_PAGES", "500")
        ),
        max_image_pixels=int(
            value("SHELFCASH_PREPROCESS_MAX_IMAGE_PIXELS", "50000000")
        ),
    )
