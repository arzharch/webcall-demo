"""
OpenTelemetry instrumentation for observability.
Provides tracing, metrics, and cost tracking for LLM calls.
"""
import time
import logging
import functools
import json
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager
from threading import Lock

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    tiktoken = None

# OpenTelemetry imports with graceful fallback
try:
    from opentelemetry import trace, metrics
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import ConsoleMetricExporter, PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.semconv.resource import ResourceAttributes
    from opentelemetry.trace import Status, StatusCode
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = None
    metrics = None

from infra.config import config

logger = logging.getLogger(__name__)


@dataclass
class LLMCallMetrics:
    """Metrics for a single LLM call."""
    model: str
    operation: str
    input_tokens: int
    output_tokens: int
    duration_ms: float
    cost_usd: float
    cached: bool = False
    error: Optional[str] = None


@dataclass
class SessionMetrics:
    """Accumulated metrics for a session."""
    session_id: str
    total_turns: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    total_duration_ms: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    errors: int = 0


class TokenCounter:
    """
    Counts tokens for cost estimation using tiktoken.
    Falls back to character-based estimation if tiktoken unavailable.
    """
    
    def __init__(self):
        self._encoders: Dict[str, Any] = {}
        self._lock = Lock()
    
    def _get_encoder(self, model: str):
        """Get or create encoder for a model."""
        if not TIKTOKEN_AVAILABLE:
            return None
        
        with self._lock:
            if model not in self._encoders:
                try:
                    # Map model names to encoding
                    encoding_name = "cl100k_base"  # Default for GPT-3.5/4
                    self._encoders[model] = tiktoken.get_encoding(encoding_name)
                except Exception as e:
                    logger.warning(f"Failed to get tiktoken encoder: {e}")
                    self._encoders[model] = None
            
            return self._encoders[model]
    
    def count_tokens(self, text: str, model: str = "gpt-3.5-turbo") -> int:
        """Count tokens in text."""
        encoder = self._get_encoder(model)
        
        if encoder:
            try:
                return len(encoder.encode(text))
            except Exception:
                pass
        
        # Fallback: ~4 chars per token
        return len(text) // 4
    
    def count_messages(self, messages: list, model: str = "gpt-3.5-turbo") -> int:
        """Count tokens in a list of messages."""
        total = 0
        for msg in messages:
            if isinstance(msg, dict):
                content = msg.get("content", "")
            elif hasattr(msg, "content"):
                content = msg.content
            else:
                content = str(msg)
            total += self.count_tokens(content, model)
            total += 4  # Message overhead
        return total


class CostTracker:
    """
    Tracks LLM costs in real-time.
    """
    
    def __init__(self):
        self.token_counter = TokenCounter()
        self._session_metrics: Dict[str, SessionMetrics] = {}
        self._total_cost = 0.0
        self._hourly_cost = 0.0
        self._hourly_reset_time = time.time()
        self._lock = Lock()
    
    def calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Calculate cost for a call."""
        pricing = config.llm_cost.models.get(model, {})
        input_cost = pricing.get("input", 0.001) * input_tokens / 1000
        output_cost = pricing.get("output", 0.002) * output_tokens / 1000
        return input_cost + output_cost
    
    def track_call(
        self,
        session_id: str,
        model: str,
        operation: str,
        input_tokens: int,
        output_tokens: int,
        duration_ms: float,
        cached: bool = False,
        error: str = None,
    ) -> LLMCallMetrics:
        """
        Track an LLM call and return metrics.
        """
        cost = 0.0 if cached else self.calculate_cost(model, input_tokens, output_tokens)
        
        metrics = LLMCallMetrics(
            model=model,
            operation=operation,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            duration_ms=duration_ms,
            cost_usd=cost,
            cached=cached,
            error=error,
        )
        
        with self._lock:
            # Update session metrics
            if session_id not in self._session_metrics:
                self._session_metrics[session_id] = SessionMetrics(session_id=session_id)
            
            session = self._session_metrics[session_id]
            session.total_turns += 1
            session.total_input_tokens += input_tokens
            session.total_output_tokens += output_tokens
            session.total_cost_usd += cost
            session.total_duration_ms += duration_ms
            
            if cached:
                session.cache_hits += 1
            else:
                session.cache_misses += 1
            
            if error:
                session.errors += 1
            
            # Update global metrics
            self._total_cost += cost
            
            # Reset hourly counter if needed
            if time.time() - self._hourly_reset_time > 3600:
                self._hourly_cost = 0.0
                self._hourly_reset_time = time.time()
            self._hourly_cost += cost
            
            # Check budget limits
            self._check_limits(session, cost)
        
        return metrics
    
    def _check_limits(self, session: SessionMetrics, cost: float):
        """Check if any cost limits are exceeded."""
        limits = config.llm_cost
        
        if cost > limits.max_cost_per_turn:
            logger.warning(
                f"Turn cost ${cost:.4f} exceeds limit ${limits.max_cost_per_turn}"
            )
        
        if session.total_cost_usd > limits.max_cost_per_session:
            logger.error(
                f"Session {session.session_id} cost ${session.total_cost_usd:.4f} "
                f"exceeds limit ${limits.max_cost_per_session}"
            )
        
        if self._hourly_cost > limits.max_cost_per_hour:
            logger.critical(
                f"Hourly cost ${self._hourly_cost:.2f} exceeds limit ${limits.max_cost_per_hour}"
            )
    
    def get_session_metrics(self, session_id: str) -> Optional[SessionMetrics]:
        """Get metrics for a session."""
        return self._session_metrics.get(session_id)
    
    def get_global_metrics(self) -> dict:
        """Get global cost metrics."""
        return {
            "total_cost_usd": self._total_cost,
            "hourly_cost_usd": self._hourly_cost,
            "active_sessions": len(self._session_metrics),
        }


class TelemetryManager:
    """
    Central manager for all telemetry: traces, metrics, and costs.
    """
    
    def __init__(self):
        self.enabled = config.telemetry.enabled and OTEL_AVAILABLE
        self.cost_tracker = CostTracker()
        self._tracer = None
        self._meter = None
        
        if self.enabled:
            self._setup_telemetry()
        else:
            logger.info("Telemetry disabled or OpenTelemetry not available")
    
    def _setup_telemetry(self):
        """Initialize OpenTelemetry."""
        try:
            # Create resource
            resource = Resource.create({
                ResourceAttributes.SERVICE_NAME: config.telemetry.service_name,
                ResourceAttributes.SERVICE_VERSION: config.telemetry.service_version,
            })
            
            # Setup tracing
            tracer_provider = TracerProvider(resource=resource)
            
            # Add exporter based on config
            if config.telemetry.trace_exporter == "console":
                tracer_provider.add_span_processor(
                    BatchSpanProcessor(ConsoleSpanExporter())
                )
            # TODO: Add GCP, Jaeger exporters
            
            trace.set_tracer_provider(tracer_provider)
            self._tracer = trace.get_tracer(__name__)
            
            # Setup metrics
            if config.telemetry.metrics_exporter == "console":
                reader = PeriodicExportingMetricReader(
                    ConsoleMetricExporter(),
                    export_interval_millis=60000,  # Every 60 seconds
                )
                meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
                metrics.set_meter_provider(meter_provider)
                self._meter = metrics.get_meter(__name__)
            
            logger.info("OpenTelemetry initialized")
            
        except Exception as e:
            logger.error(f"Failed to setup telemetry: {e}")
            self.enabled = False
    
    @contextmanager
    def trace_span(
        self,
        name: str,
        attributes: Dict[str, Any] = None,
    ):
        """
        Create a trace span.
        
        Usage:
            with telemetry.trace_span("llm_call", {"model": "gpt-3.5"}) as span:
                result = await llm.invoke(...)
                span.set_attribute("tokens", token_count)
        """
        if not self.enabled or not self._tracer:
            yield DummySpan()
            return
        
        with self._tracer.start_as_current_span(name) as span:
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, value)
            try:
                yield span
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise
    
    def track_llm_call(
        self,
        session_id: str,
        model: str,
        operation: str,
        input_messages: list,
        output_text: str,
        duration_ms: float,
        cached: bool = False,
        error: str = None,
    ) -> LLMCallMetrics:
        """
        Track an LLM call with full metrics.
        """
        # Count tokens
        input_tokens = self.cost_tracker.token_counter.count_messages(input_messages, model)
        output_tokens = self.cost_tracker.token_counter.count_tokens(output_text, model)
        
        # Track costs
        metrics = self.cost_tracker.track_call(
            session_id=session_id,
            model=model,
            operation=operation,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            duration_ms=duration_ms,
            cached=cached,
            error=error,
        )
        
        # Log structured
        logger.info(
            f"LLM Call: {operation} | model={model} | "
            f"tokens={input_tokens}+{output_tokens} | "
            f"cost=${metrics.cost_usd:.4f} | "
            f"duration={duration_ms:.0f}ms | "
            f"cached={cached}"
        )
        
        return metrics
    
    def get_health(self) -> dict:
        """Get telemetry health status."""
        return {
            "enabled": self.enabled,
            "otel_available": OTEL_AVAILABLE,
            "tiktoken_available": TIKTOKEN_AVAILABLE,
            "global_metrics": self.cost_tracker.get_global_metrics(),
        }


class DummySpan:
    """No-op span when telemetry is disabled."""
    def set_attribute(self, key, value): pass
    def set_status(self, status): pass
    def record_exception(self, exc): pass
    def add_event(self, name, attributes=None): pass


# Global instance
_telemetry: Optional[TelemetryManager] = None


def get_telemetry() -> TelemetryManager:
    """Get the global telemetry manager."""
    global _telemetry
    if _telemetry is None:
        _telemetry = TelemetryManager()
    return _telemetry


def trace_span(name: str, attributes: Dict[str, Any] = None):
    """Convenience function to create a trace span."""
    return get_telemetry().trace_span(name, attributes)


def track_llm_cost(
    session_id: str,
    model: str,
    operation: str,
    input_messages: list,
    output_text: str,
    duration_ms: float,
    cached: bool = False,
    error: str = None,
) -> LLMCallMetrics:
    """Convenience function to track LLM costs."""
    return get_telemetry().track_llm_call(
        session_id=session_id,
        model=model,
        operation=operation,
        input_messages=input_messages,
        output_text=output_text,
        duration_ms=duration_ms,
        cached=cached,
        error=error,
    )
