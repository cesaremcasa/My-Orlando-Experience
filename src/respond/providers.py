from __future__ import annotations

import os
from typing import Protocol


class ChatProvider(Protocol):
    def generate(self, *, question: str, context: str) -> str: ...


class OpenAIProvider:
    """Lazy OpenAI adapter; importing API code never requires an API key."""

    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.3):
        self.model_name = model_name
        self.temperature = temperature
        self._model = None

    def generate(self, *, question: str, context: str) -> str:
        if self._model is None:
            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY is required for OpenAI response generation")
            from langchain_openai import ChatOpenAI

            self._model = ChatOpenAI(model=self.model_name, temperature=self.temperature)
        from langchain_core.prompts import ChatPromptTemplate

        template = ChatPromptTemplate.from_template(
            "You are a thoughtful Orlando trip advisor.\n"
            "Answer using ONLY verified facts in Context. If unsure, say so.\n\n"
            "Context:\n{context}\n\nQuestion: {question}\n"
        )
        return str((template | self._model).invoke({"question": question, "context": context}).content)


class FakeProvider:
    """Deterministic provider for unit tests; never calls an external model."""

    def generate(self, *, question: str, context: str) -> str:
        return f"Fixture response for: {question}. Context: {context[:120]}"
