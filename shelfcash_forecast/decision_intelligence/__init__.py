"""Read-only, evidence-grounded Decision Intelligence for M1-M5 outputs."""
# 4. M6 apply RAG kiểu nào?

# M6 hiện dùng:

# Structured-first + lexical + graph-neighborhood retrieval, chạy hoàn toàn offline.

# Nó không phải RAG truyền thống kiểu:

# Documents
# → chunking
# → embeddings
# → vector database
# → similarity search
# → LLM

# M6 hiện tại là:

# Typed artifacts
# → deterministic evidence
# → typed graph
# → intent/entity retrieval
# → grounded deterministic generation
# → guard
from shelfcash_forecast.decision_intelligence.agents import (
    AgentRunRequest,
    AgentRunResult,
    DecisionOrchestrator,
    run_decision_agent,
)
from shelfcash_forecast.decision_intelligence.approval import (
    ApprovalCase,
    ApprovalPolicy,
    create_approval_case,
    inspect_approval_case,
    transition_approval_case,
)
from shelfcash_forecast.decision_intelligence.computation_gateway import (
    ComputationGateway,
    M5ComputationGateway,
)
from shelfcash_forecast.decision_intelligence.contracts import (
    ArtifactCoherenceIssue,
    ArtifactCoherenceResult,
    DecisionAnswer,
    DecisionGraph,
    DecisionIntelligenceInput,
    EvidenceItem,
    EvidencePackage,
    FinalDecisionPackage,
)
from shelfcash_forecast.decision_intelligence.evidence import (
    build_evidence_package,
    stable_evidence_id,
)
from shelfcash_forecast.decision_intelligence.graph import build_decision_graph
from shelfcash_forecast.decision_intelligence.grounding import (
    DeterministicGroundedGenerator,
    GroundedGenerator,
    GroundingError,
    GroundingGuard,
)
from shelfcash_forecast.decision_intelligence.regret import (
    DecisionRegretRequest,
    DecisionRegretResult,
    evaluate_decision_regret,
)
from shelfcash_forecast.decision_intelligence.retrieval import (
    EvidenceRetriever,
    StructuredLocalRetriever,
)
from shelfcash_forecast.decision_intelligence.service import (
    build_final_decision_package,
    explain_decision,
)
from shelfcash_forecast.decision_intelligence.what_if import (
    CounterfactualSearchRequest,
    CounterfactualSearchResult,
    WhatIfDecisionPackage,
    WhatIfDraft,
    WhatIfRequest,
    compare_decisions,
    confirm_what_if,
    draft_what_if,
    explain_what_if,
    run_what_if,
    search_counterfactuals,
)

__all__ = [
    "AgentRunRequest",
    "AgentRunResult",
    "ApprovalCase",
    "ApprovalPolicy",
    "ArtifactCoherenceIssue",
    "ArtifactCoherenceResult",
    "ComputationGateway",
    "CounterfactualSearchRequest",
    "CounterfactualSearchResult",
    "DecisionAnswer",
    "DecisionGraph",
    "DecisionIntelligenceInput",
    "DecisionOrchestrator",
    "DecisionRegretRequest",
    "DecisionRegretResult",
    "DeterministicGroundedGenerator",
    "EvidenceItem",
    "EvidencePackage",
    "EvidenceRetriever",
    "FinalDecisionPackage",
    "GroundedGenerator",
    "GroundingError",
    "GroundingGuard",
    "M5ComputationGateway",
    "StructuredLocalRetriever",
    "WhatIfDecisionPackage",
    "WhatIfDraft",
    "WhatIfRequest",
    "build_decision_graph",
    "build_evidence_package",
    "build_final_decision_package",
    "compare_decisions",
    "confirm_what_if",
    "create_approval_case",
    "draft_what_if",
    "evaluate_decision_regret",
    "explain_decision",
    "explain_what_if",
    "inspect_approval_case",
    "run_decision_agent",
    "run_what_if",
    "search_counterfactuals",
    "stable_evidence_id",
    "transition_approval_case",
]
from shelfcash_forecast.decision_intelligence.openai_generator import (
    M6LLMError,
    OpenAIGroundedGenerator,
)

__all__ += ["M6LLMError", "OpenAIGroundedGenerator"]
