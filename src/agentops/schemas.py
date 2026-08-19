from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class AgentChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=2000)
    remember: bool = True

    @field_validator("session_id", "message")
    @classmethod
    def _strip_nonempty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must not be blank")
        return cleaned


class AgentCitation(BaseModel):
    document: str
    source_id: str
    chunk_id: str | int
    excerpt: str
    score: float


class AgentChatResponse(BaseModel):
    response_id: str
    response: str
    grounding_status: Literal["grounded", "abstained"]
    citations: list[AgentCitation] = Field(default_factory=list)
    agent_path: list[str] = Field(default_factory=list)
    memory_ids: list[str] = Field(default_factory=list)
    trace_id: str | None = None
    latency_ms: float


class AgentHealthResponse(BaseModel):
    status: Literal["healthy"] = "healthy"
    agent_ready: bool = False
    memory_backend: str = "local"
    eval_backend: str = "local"
    vertex_ready: bool = False
    vertex_issues: list[str] = Field(default_factory=list)


class MemoryCandidate(BaseModel):
    content: str = Field(min_length=1, max_length=2000)
    provenance: str = "response"


class ResponseCandidate(BaseModel):
    response: str
    citation_ids: list[str] = Field(default_factory=list)
    memory_candidate: MemoryCandidate | None = None


class SafetyVerdict(BaseModel):
    verdict: Literal["PASS", "FAIL"]
    reason: str = Field(min_length=1, max_length=240)
    schema_ok: bool = True
    grounded: bool = True


class FeedbackRequest(BaseModel):
    response_id: str = Field(min_length=1, max_length=128)
    session_id: str = Field(min_length=1, max_length=128)
    rating: int = Field(ge=1, le=5)
    accepted: bool
    reason: str | None = Field(default=None, max_length=500)

    @field_validator("response_id", "session_id")
    @classmethod
    def _strip_ids(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must not be blank")
        return cleaned

    @field_validator("reason")
    @classmethod
    def _strip_reason(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None


class FeedbackResponse(BaseModel):
    response_id: str
    session_id: str
    rating: int
    accepted: bool
    updated: bool
