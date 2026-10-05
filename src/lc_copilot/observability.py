"""Lightweight run tracing.

Every agent node, LLM call, retrieval and tool call becomes a span with latency, token usage
and cost. Spans always go to ``runs/traces.jsonl`` (grep-able, diff-able in CI) and, when
``LC_OTEL_ENABLED=true``, are mirrored to OpenTelemetry — which Langfuse, LangSmith and
Datadog all ingest over OTLP. That keeps vendor choice a deployment decision, not a code change.
"""

from __future__ import annotations

import contextvars
import json
import logging
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import get_settings

log = logging.getLogger("lc_copilot")

_trace_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("trace_id", default=None)
_parent: contextvars.ContextVar[str | None] = contextvars.ContextVar("parent_span", default=None)
_collector: contextvars.ContextVar[list[Span] | None] = contextvars.ContextVar("collector", default=None)


@dataclass
class Span:
    name: str
    trace_id: str
    span_id: str
    parent_id: str | None
    start: float
    end: float | None = None
    attrs: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"

    @property
    def latency_ms(self) -> float:
        return round(((self.end or time.time()) - self.start) * 1000, 2)

    def set(self, **attrs: Any) -> None:
        self.attrs.update(attrs)


_otel_tracer = None


def _otel():
    global _otel_tracer
    if _otel_tracer is None and get_settings().otel_enabled:
        try:  # optional extra: pip install .[otel]
            from opentelemetry import trace
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            provider = TracerProvider()
            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))  # OTEL_EXPORTER_OTLP_* env
            trace.set_tracer_provider(provider)
            _otel_tracer = trace.get_tracer("lc_copilot")
        except ImportError:  # pragma: no cover
            log.warning("LC_OTEL_ENABLED=true but opentelemetry is not installed")
            _otel_tracer = False
    return _otel_tracer or None


def _export(span: Span) -> None:
    path: Path = get_settings().trace_dir
    path.mkdir(parents=True, exist_ok=True)
    record = asdict(span) | {"latency_ms": span.latency_ms}
    with (path / "traces.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str) + "\n")


@contextmanager
def span(name: str, **attrs: Any) -> Iterator[Span]:
    trace_id = _trace_id.get() or uuid.uuid4().hex
    s = Span(name, trace_id, uuid.uuid4().hex[:16], _parent.get(), time.time(), attrs=dict(attrs))
    t_tok, p_tok = _trace_id.set(trace_id), _parent.set(s.span_id)
    otel = _otel()
    otel_cm = otel.start_as_current_span(name) if otel else None
    otel_span = otel_cm.__enter__() if otel_cm else None
    try:
        yield s
    except Exception as exc:
        s.status = f"error:{type(exc).__name__}"
        raise
    finally:
        s.end = time.time()
        _trace_id.reset(t_tok)
        _parent.reset(p_tok)
        if otel_span is not None:
            for k, v in s.attrs.items():
                otel_span.set_attribute(f"lc.{k}", v if isinstance(v, (str, int, float, bool)) else str(v))
            otel_cm.__exit__(None, None, None)
        if (bucket := _collector.get()) is not None:
            bucket.append(s)
        _export(s)


@contextmanager
def collect_spans() -> Iterator[list[Span]]:
    """Collect every span emitted inside the block (used by the eval harness and the API)."""
    bucket: list[Span] = []
    tok = _collector.set(bucket)
    try:
        yield bucket
    finally:
        _collector.reset(tok)


def current_trace_id() -> str | None:
    return _trace_id.get()


def usage_totals(spans: list[Span]) -> dict[str, float]:
    s = get_settings()
    tin = sum(int(x.attrs.get("tokens_in", 0)) for x in spans)
    tout = sum(int(x.attrs.get("tokens_out", 0)) for x in spans)
    return {
        "tokens_in": tin,
        "tokens_out": tout,
        "cost_usd": round(tin / 1e6 * s.price_in_per_m + tout / 1e6 * s.price_out_per_m, 6),
        "llm_calls": sum(1 for x in spans if x.name.startswith("llm.")),
    }
