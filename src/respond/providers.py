from __future__ import annotations

import os
from typing import Any, Protocol


XAI_DEFAULT_BASE_URL = "https://api.x.ai/v1"
XAI_DEFAULT_MODEL = "grok-4.20-0309-non-reasoning"


def _response_template():
    """Build the prompt lazily so importing the provider stays dependency-light."""
    from langchain_core.prompts import ChatPromptTemplate

    return ChatPromptTemplate.from_template(
        "You are a thoughtful Orlando trip advisor.\n"
        "Answer using ONLY verified facts in Context. If unsure, say so.\n\n"
        "Context:\n{context}\n\nQuestion: {question}\n"
    )


class ChatProvider(Protocol):
    def generate(self, *, question: str, context: str) -> str: ...


class OpenAIProvider:
    """Lazy OpenAI adapter; importing API code never requires an API key."""

    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.3):
        self.model_name = model_name
        self.temperature = temperature
        self._model: Any | None = None

    def generate(self, *, question: str, context: str) -> str:
        if self._model is None:
            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY is required for OpenAI response generation")
            from langchain_openai import ChatOpenAI

            self._model = ChatOpenAI(model=self.model_name, temperature=self.temperature)
        template = _response_template()
        return str((template | self._model).invoke({"question": question, "context": context}).content)


class XAIProvider:
    """Lazy xAI/Grok adapter using xAI's OpenAI-compatible API."""

    def __init__(
        self,
        model_name: str | None = None,
        temperature: float = 0.3,
        base_url: str | None = None,
    ):
        self.model_name = model_name or os.getenv("XAI_MODEL") or XAI_DEFAULT_MODEL
        self.temperature = temperature
        self.base_url = base_url or os.getenv("XAI_BASE_URL") or XAI_DEFAULT_BASE_URL
        self._model: Any | None = None

    def generate(self, *, question: str, context: str) -> str:
        if self._model is None:
            api_key = os.getenv("XAI_API_KEY")
            if not api_key:
                raise RuntimeError("XAI_API_KEY is required for Grok response generation")
            from langchain_openai import ChatOpenAI

            self._model = ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                api_key=api_key,
                base_url=self.base_url,
            )
        template = _response_template()
        return str((template | self._model).invoke({"question": question, "context": context}).content)


class FakeProvider:
    """Deterministic provider for unit tests; never calls an external model."""

    def generate(self, *, question: str, context: str) -> str:
        return f"Fixture response for: {question}. Context: {context[:120]}"
