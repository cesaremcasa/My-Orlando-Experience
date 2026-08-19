from __future__ import annotations

import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from src.respond.providers import (
    XAI_DEFAULT_BASE_URL,
    XAI_DEFAULT_MODEL,
    XAIProvider,
)


def test_xai_provider_uses_validated_defaults(monkeypatch):
    monkeypatch.delenv("XAI_BASE_URL", raising=False)
    monkeypatch.delenv("XAI_MODEL", raising=False)

    provider = XAIProvider()

    assert provider.base_url == XAI_DEFAULT_BASE_URL
    assert provider.model_name == XAI_DEFAULT_MODEL


def test_api_selects_xai_provider_by_default():
    from src.api import main

    assert isinstance(main.provider, XAIProvider)


def test_xai_provider_reads_base_url_and_model_overrides(monkeypatch):
    monkeypatch.setenv("XAI_BASE_URL", "https://xai.example.test/v1")
    monkeypatch.setenv("XAI_MODEL", "grok-test")

    provider = XAIProvider()

    assert provider.base_url == "https://xai.example.test/v1"
    assert provider.model_name == "grok-test"


def test_xai_provider_requires_key_only_when_generating(monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)

    provider = XAIProvider()
    assert provider._model is None

    with pytest.raises(RuntimeError, match="XAI_API_KEY"):
        provider.generate(question="q", context="c")

    assert provider._model is None


def test_xai_provider_configures_compatible_client_without_network(monkeypatch):
    captured: dict[str, object] = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class FakeTemplate:
        def __or__(self, model):
            assert model is not None
            return self

        def invoke(self, _payload):
            return SimpleNamespace(content="synthetic response")

    class FakePromptTemplate:
        @classmethod
        def from_template(cls, _template):
            return FakeTemplate()

    langchain_openai = ModuleType("langchain_openai")
    langchain_openai.ChatOpenAI = FakeChatOpenAI  # type: ignore[attr-defined]
    langchain_core_prompts = ModuleType("langchain_core.prompts")
    langchain_core_prompts.ChatPromptTemplate = FakePromptTemplate  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langchain_openai", langchain_openai)
    monkeypatch.setitem(sys.modules, "langchain_core.prompts", langchain_core_prompts)
    monkeypatch.setenv("XAI_API_KEY", "test-key-not-a-real-secret")
    monkeypatch.setenv("XAI_BASE_URL", "https://xai.example.test/v1")
    monkeypatch.setenv("XAI_MODEL", "grok-test")

    provider = XAIProvider(temperature=0.1)

    assert provider.generate(question="q", context="c") == "synthetic response"
    assert captured == {
        "model": "grok-test",
        "temperature": 0.1,
        "api_key": "test-key-not-a-real-secret",
        "base_url": "https://xai.example.test/v1",
    }


def test_provider_canary_is_pending_without_key(monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    script = Path(__file__).resolve().parents[1] / "scripts" / "provider_canary.py"

    with pytest.raises(SystemExit) as raised:
        runpy.run_path(str(script), run_name="__main__")

    assert raised.value.code == "XAI_API_KEY is required; canary is pending"
