from __future__ import annotations

from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "2.0.0"
PROMPT_VERSION = "mapping-v2"
PROFILE_VERSION = "2.0"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Role(str, Enum):
    SALES = "sales"
    INGREDIENT_USAGE = "ingredient_usage"
    INVENTORY_SNAPSHOT = "inventory_snapshot"
    PURCHASE_HISTORY = "purchase_history"
    SUPPLIER_RULES = "supplier_rules"
    RECIPES = "recipes"
    BUSINESS_RULES = "business_rules"
    MENU = "menu"
    CALENDAR = "calendar"
    UNKNOWN = "unknown"


class Readiness(str, Enum):
    READY = "READY"
    READY_WITH_WARNINGS = "READY_WITH_WARNINGS"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    BLOCKED = "BLOCKED"


class Severity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    BLOCKING = "BLOCKING"


class IssueScope(str, Enum):
    GLOBAL = "global"
    TABLE = "table"
    ENTITY = "entity"
    FIELD = "field"


class SourceFile(StrictModel):
    source_id: str
    relative_path: str
    sha256: str
    size_bytes: int
    media_type: str
    archive_parent: str | None = None
    status: str = "DISCOVERED"
    notes: list[str] = Field(default_factory=list)


class CellRange(StrictModel):
    start_row: int
    end_row: int
    start_col: int
    end_col: int


class TableProfile(StrictModel):
    row_count: int
    column_count: int
    columns: list[str]
    inferred_types: dict[str, str]
    null_ratio: dict[str, float]
    unique_count: dict[str, int]
    numeric_ranges: dict[str, dict[str, float | None]] = Field(default_factory=dict)
    date_ranges: dict[str, dict[str, str | None]] = Field(default_factory=dict)
    samples: list[dict[str, Any]] = Field(default_factory=list)
    anomalies: list[str] = Field(default_factory=list)


class TableRegion(StrictModel):
    region_id: str
    source_id: str
    source_path: str
    container: str
    container_kind: str
    hidden: bool = False
    bounds: CellRange
    header_rows: list[int]
    proposed_header: list[str]
    confidence: float = Field(ge=0, le=1)
    evidence: list[str] = Field(default_factory=list)
    profile: TableProfile
    extraction_status: str = "EXTRACTED"
    metadata: dict[str, Any] = Field(default_factory=dict)


class TransformOperation(StrictModel):
    operation: Literal[
        "rename",
        "cast",
        "parse_date",
        "parse_number",
        "parse_boolean",
        "trim",
        "filter_total_rows",
        "split",
        "combine",
        "unpivot",
        "lookup_id",
        "unit_conversion",
        "aggregate",
    ]
    arguments: dict[str, Any] = Field(default_factory=dict)


class FieldMapping(StrictModel):
    source_column: str
    target_field: str | None
    confidence: float = Field(ge=0, le=1)
    evidence: list[str] = Field(default_factory=list)
    operations: list[TransformOperation] = Field(default_factory=list)
    ambiguous: bool = False


class MappingProposal(StrictModel):
    region_id: str
    role: Role
    role_confidence: float = Field(ge=0, le=1)
    field_mappings: list[FieldMapping] = Field(default_factory=list)
    unresolved_fields: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    evidence_references: list[str] = Field(default_factory=list)
    needs_review: bool = False
    origin: Literal["deterministic", "live", "profile", "review", "fake_test"]


class MappingPlan(StrictModel):
    schema_version: str = SCHEMA_VERSION
    proposals: list[MappingProposal]
    model: str | None = None
    reasoning_effort: str | None = None
    prompt_version: str = PROMPT_VERSION
    origin: Literal["offline", "live", "mixed", "fake_test"]
    usage: dict[str, int] = Field(default_factory=dict)


class Issue(StrictModel):
    issue_id: str
    code: str
    severity: Severity
    message: str
    region_id: str | None = None
    source_locator: dict[str, Any] = Field(default_factory=dict)
    details: dict[str, Any] = Field(default_factory=dict)
    suggested_action: str | None = None
    scope: IssueScope = IssueScope.GLOBAL
    table: str | None = None
    entity_key: str | None = None
    field: str | None = None
    affected_capabilities: list[str] = Field(default_factory=list)
    review_status: Literal["pending", "approved", "rejected", "not_required"] = "pending"
    review_decision_references: list[str] = Field(default_factory=list)
    # Kept for bundle-v1 diagnostics.  Bundle-v2 readiness uses
    # affected_capabilities and never treats a missing capability as harmless.
    capability: str | None = None


class ReviewDecision(StrictModel):
    issue_id: str = Field(min_length=1)
    action: Literal[
        "approve_mapping",
        "set_role",
        "map_field",
        "map_entity",
        "set_metadata",
        "ignore_region",
        "select_date_locale",
        "select_import_semantics",
    ]
    value: Any = None
    note: str | None = None

    @model_validator(mode="after")
    def validate_action_value(self) -> "ReviewDecision":
        if self.action == "select_date_locale" and self.value not in {"DMY", "MDY", "YMD"}:
            raise ValueError("select_date_locale value must be DMY, MDY, or YMD")
        if self.action == "select_import_semantics" and self.value not in {"append", "replace", "upsert"}:
            raise ValueError("select_import_semantics value must be append, replace, or upsert")
        if self.action in {"set_role", "map_field", "map_entity", "set_metadata", "ignore_region", "approve_mapping"} and not isinstance(self.value, dict):
            raise ValueError(f"{self.action} requires an object value")
        return self


class ReviewFile(StrictModel):
    schema_version: str = SCHEMA_VERSION
    run_id: str
    source_inventory_hash: str
    mapping_plan_hash: str
    decisions: list[ReviewDecision]


class CapabilityReadiness(StrictModel):
    status: Readiness
    data_validated: bool = False
    runtime_validated: bool = False
    business_status: Literal["NOT_APPLICABLE", "READY", "NEEDS_BUSINESS_INPUT"] = (
        "NOT_APPLICABLE"
    )
    issues: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class RunContext(StrictModel):
    store_id: str | None = None
    cutoff_date: date | None = None
    date_locale: Literal["DMY", "MDY", "YMD"] | None = None
    tenant_id: str | None = None
    import_semantics: Literal["append", "replace", "upsert"] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RunManifest(StrictModel):
    schema_version: str = SCHEMA_VERSION
    run_id: str
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    completed_at: str | None = None
    input_path: str
    output_path: str
    context: RunContext
    source_inventory_hash: str
    mapping_plan_hash: str
    llm_mode: Literal["offline", "live", "fake_test"]
    model: str
    reasoning_effort: str
    prompt_version: str = PROMPT_VERSION
    profile_version: str = PROFILE_VERSION
    steps_completed: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    readiness: dict[str, CapabilityReadiness]
    bundle_complete: bool = False
    usage: dict[str, int] = Field(default_factory=dict)
    artifact_manifest_version: str = "bundle-artifacts-v2"
    artifacts: list["ArtifactRecord"] = Field(default_factory=list)
    parent_run_id: str | None = None
    parent_manifest_sha256: str | None = None
    review_file_sha256: str | None = None
    review_decision_ids: list[str] = Field(default_factory=list)


class ImportProfile(StrictModel):
    profile_version: str = PROFILE_VERSION
    profile_id: str
    tenant_id: str | None = None
    role: Role
    schema_fingerprint: str
    header_signature: list[str]
    mapping: MappingProposal
    source_hashes: list[str] = Field(default_factory=list)
    verified_by: Literal["deterministic", "review"]
    semantic_signature: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ArtifactRecord(StrictModel):
    path: str
    role: str
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    row_count: int | None = Field(default=None, ge=0)
    schema_version: str | None = None


class BundleInfo(StrictModel):
    run_dir: Path
    manifest: RunManifest
    canonical_files: dict[str, Path]


class SemanticPlanEnvelope(StrictModel):
    """Structured Outputs schema for one or more unknown regions."""

    proposals: list[MappingProposal]

    @model_validator(mode="after")
    def unique_regions(self) -> SemanticPlanEnvelope:
        ids = [item.region_id for item in self.proposals]
        if len(ids) != len(set(ids)):
            raise ValueError("region_id must be unique")
        return self
