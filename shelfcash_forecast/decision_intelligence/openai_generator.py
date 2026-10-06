from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from shelfcash_forecast.decision_intelligence.contracts import (
    DecisionAnswer,
    FinalDecisionPackage,
    GroundedClaim,
    RetrievedEvidence,
)
from shelfcash_preprocess.config import Settings, load_settings


class M6LLMError(RuntimeError):
    """Sanitized, typed failure from the optional M6 explanation branch."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class _StrictOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _GeneratedClaim(_StrictOutput):
    claim_type: str = Field(min_length=1)
    text: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)
    facts: dict[str, Any] = Field(default_factory=dict)
    uses_probability_language: bool = False
    causal: bool = False


class _GeneratedExplanation(_StrictOutput):
    claims: list[_GeneratedClaim]
    limitations: list[str] = Field(default_factory=list)


def _classify_error(exc: Exception) -> str:
    name = type(exc).__name__.casefold()
    status = getattr(exc, "status_code", None)
    if status in {401, 403} or "authentication" in name or "permission" in name:
        return "M6_LLM_AUTH_OR_ACCESS_ERROR"
    if status == 429 or "ratelimit" in name:
        return "M6_LLM_RATE_LIMIT"
    if "timeout" in name:
        return "M6_LLM_TIMEOUT"
    if status == 404 or "notfound" in name:
        return "M6_LLM_MODEL_NOT_AVAILABLE"
    return "M6_LLM_API_ERROR"


class OpenAIGroundedGenerator:
    """Opt-in structured explanation generator; it never changes M5 decisions."""

    def __init__(
        self,
        *,
        config_path: str | None = None,
        settings: Settings | None = None,
        client: Any | None = None,
    ) -> None:
        self.settings = settings or load_settings(config_path)
        self.model = self.settings.m6_model
        if client is None:
            if not self.settings.m6_llm_enabled:
                raise M6LLMError("M6_LLM_DISABLED", "M6 LLM mode is not enabled.")
            if self.settings.key_status != "SET":
                raise M6LLMError(
                    "M6_LLM_KEY_UNAVAILABLE",
                    f"M6 API key status is {self.settings.key_status}.",
                )
            if not self.model or self.model.startswith("SET_SUPPORTED_"):
                raise M6LLMError(
                    "M6_LLM_MODEL_UNCONFIGURED",
                    "Set an OpenAI API model ID verified for this account.",
                )
            from openai import OpenAI

            client = OpenAI(
                api_key=self.settings.api_key,
                timeout=self.settings.timeout_seconds,
                max_retries=self.settings.max_retries,
            )
        self.client = client

    def generate(
        self,
        question: str,
        retrieved: RetrievedEvidence,
        decision: FinalDecisionPackage,
    ) -> DecisionAnswer:
        evidence = [
            {
                "evidence_id": item.evidence_id,
                "evidence_type": item.evidence_type,
                "entities": item.entities,
                "payload": item.payload,
            }
            for item in retrieved.items[:20]
        ]
        protected = {
            "decision_status": decision.decision_status,
            "recommended_strategy": decision.recommended_strategy,
            "immediate_orders": [
                order.model_dump(mode="json") for order in decision.immediate_orders
            ],
        }
        prompt = (
            "Explain only the supplied decision evidence. Every claim must cite one or "
            "more supplied evidence_id values. Do not create or alter strategy, order, "
            "quantity, cost, approval, probability, or operational status. Copy numeric "
            "facts exactly from evidence. If evidence is insufficient, return no claims "
            "and state a limitation.\n\n"
            + json.dumps(
                {
                    "question": question,
                    "protected_decision": protected,
                    "retrieved_evidence": evidence,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
        try:
            response = self.client.responses.parse(
                model=self.model,
                input=[
                    {
                        "role": "system",
                        "content": "You are a read-only ShelfCash evidence explainer.",
                    },
                    {"role": "user", "content": prompt},
                ],
                text_format=_GeneratedExplanation,
                max_output_tokens=min(self.settings.max_output_tokens, 3000),
            )
            parsed = response.output_parsed
            if parsed is None:
                raise M6LLMError(
                    "M6_LLM_MALFORMED_OUTPUT",
                    "The API returned no parsed structured explanation.",
                )
        except M6LLMError:
            raise
        except Exception as exc:
            raise M6LLMError(
                _classify_error(exc),
                f"OpenAI M6 explanation failed ({type(exc).__name__}).",
            ) from exc

        claims = [GroundedClaim.model_validate(item.model_dump()) for item in parsed.claims]
        citations = sorted(
            {evidence_id for claim in claims for evidence_id in claim.evidence_ids}
        )
        answer_text = "\n".join(
            f"{claim.text} "
            + " ".join(f"[evidence:{item}]" for item in claim.evidence_ids)
            for claim in claims
        )
        if not answer_text:
            answer_text = "INSUFFICIENT_EVIDENCE"
        usage = getattr(response, "usage", None)
        return DecisionAnswer(
            question=question,
            intent=retrieved.intent,
            status="GROUNDED" if claims else "INSUFFICIENT_EVIDENCE",
            answer_text=answer_text,
            claims=claims,
            citations=citations,
            retrieved_evidence_ids=[item.evidence_id for item in retrieved.items],
            limitations=list(parsed.limitations),
            provenance={
                "generator": "openai_structured_grounded_generator_v1",
                "requested_mode": "llm",
                "actual_mode": "llm",
                "model": self.model,
                "usage": None
                if usage is None
                else {
                    "input_tokens": getattr(usage, "input_tokens", None),
                    "output_tokens": getattr(usage, "output_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                },
            },
        )
