from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

_SENSITIVE_KEY_PARTS = (
    "input",
    "output",
    "prompt",
    "message",
    "reason",
    "api_key",
    "authorization",
    "invocation",
    "document",
    "excerpt",
    "body",
    "value",
)

from src.agentops.settings import (
    phoenix_collector_endpoint,
    trace_content_enabled,
    trace_exporter_name,
)

_initialized = False
_active = False
_memory_exporter: Any = None
_instrumented = False
_provider: Any = None
_tracer: Any = None


def tracing_active() -> bool:
    return _active


def memory_exporter() -> Any:
    return _memory_exporter


def reset_tracing() -> None:
    global _initialized, _active, _memory_exporter, _instrumented, _provider, _tracer
    if _instrumented:
        try:
            from openinference.instrumentation.google_adk import GoogleADKInstrumentor

            instrumentor = GoogleADKInstrumentor()
            if instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.uninstrument()
        except Exception:
            pass
    _initialized = False
    _active = False
    _memory_exporter = None
    _instrumented = False
    _provider = None
    _tracer = None


def ensure_tracing() -> None:
    global _initialized, _active, _memory_exporter, _instrumented, _provider, _tracer
    if _initialized:
        return
    _initialized = True
    exporter_name = trace_exporter_name()
    endpoint = phoenix_collector_endpoint()
    if exporter_name == "noop" or (exporter_name == "otlp" and not endpoint):
        _active = False
        _provider, _tracer = _install_noop()
        return
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    provider = TracerProvider()
    if exporter_name == "memory":
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        _memory_exporter = InMemorySpanExporter()
        sanitizing: Any = _SanitizingExporter(_memory_exporter)
        provider.add_span_processor(SimpleSpanProcessor(sanitizing))
        _active = True
    else:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        otlp: Any = _SanitizingExporter(OTLPSpanExporter(endpoint=endpoint))
        provider.add_span_processor(SimpleSpanProcessor(otlp))
        _active = True
    _provider = provider
    _tracer = provider.get_tracer("orlando.agentops")
    _instrument_adk(provider)


def force_flush(timeout_millis: int = 5000) -> bool:
    ensure_tracing()
    if _provider is None:
        return True
    flush = getattr(_provider, "force_flush", None)
    if callable(flush):
        return bool(flush(timeout_millis))
    return True


def current_trace_id() -> str | None:
    if not _active:
        return None
    from opentelemetry import trace

    context = trace.get_current_span().get_span_context()
    if context is None or not context.is_valid:
        return None
    return format(context.trace_id, "032x")


@contextmanager
def start_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[Any]:
    ensure_tracing()
    tracer = _tracer
    if tracer is None:
        from opentelemetry import trace

        tracer = trace.get_tracer("orlando.agentops")
    with tracer.start_as_current_span(name) as span:
        for key, value in _safe_attributes(attributes).items():
            span.set_attribute(key, value)
        yield span


class _SanitizingExporter:  # duck-typed SpanExporter; SDK reads export/shutdown/force_flush
    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def export(self, spans: Any) -> Any:
        for span in spans:
            _redact_span(span)
        return self._inner.export(spans)

    def shutdown(self) -> None:
        shutdown = getattr(self._inner, "shutdown", None)
        if callable(shutdown):
            shutdown()

    def force_flush(self, timeout_millis: int = 0) -> bool:
        flush = getattr(self._inner, "force_flush", None)
        if callable(flush):
            return bool(flush(timeout_millis))
        return True


def _install_noop() -> tuple[Any, Any]:
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

    class _NoOpExporter(SpanExporter):
        def export(self, spans: Any) -> SpanExportResult:
            del spans
            return SpanExportResult.SUCCESS

        def shutdown(self) -> None:
            return None

    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(_NoOpExporter()))
    return provider, provider.get_tracer("orlando.agentops")


def _instrument_adk(provider: Any) -> None:
    global _instrumented
    if _instrumented:
        return
    from openinference.instrumentation import TraceConfig
    from openinference.instrumentation.google_adk import GoogleADKInstrumentor

    hide = not trace_content_enabled()
    config = TraceConfig(
        hide_inputs=hide,
        hide_outputs=hide,
        hide_input_messages=hide,
        hide_output_messages=hide,
        hide_input_text=hide,
        hide_output_text=hide,
        hide_llm_invocation_parameters=hide,
        hide_prompts=hide,
    )
    instrumentor = GoogleADKInstrumentor()
    if not instrumentor.is_instrumented_by_opentelemetry:
        instrumentor.instrument(tracer_provider=provider, config=config)
    _instrumented = True


def _redact_span(span: Any) -> None:
    raw = getattr(span, "attributes", None) or {}
    kept: dict[str, Any] = {}
    for key, value in dict(raw).items():
        lowered = str(key).lower()
        if any(part in lowered for part in _SENSITIVE_KEY_PARTS):
            continue
        if isinstance(value, str) and len(value) > 128:
            continue
        kept[key] = value
    try:
        span._attributes = kept
    except Exception:
        pass
    events = getattr(span, "_events", None)
    if events:
        try:
            events.clear()
        except Exception:
            pass


def _safe_attributes(attributes: dict[str, Any] | None) -> dict[str, Any]:
    allowed = {
        "session_id",
        "response_id",
        "beta_user",
        "status",
        "latency_ms",
        "chunk_count",
        "hit_count",
        "citation_count",
        "score",
        "verdict",
        "schema_ok",
        "grounded",
        "rating",
        "accepted",
        "agent",
        "content_hash",
        "size",
        "grounding_status",
        "updated",
    }
    out: dict[str, Any] = {}
    for key, value in (attributes or {}).items():
        if key not in allowed or value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            out[key] = value
    return out
