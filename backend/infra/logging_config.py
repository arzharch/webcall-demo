"""
Structured logging for production with correlation IDs and JSON output.
Provides context-aware logging for distributed tracing.
"""
import logging
import json
import time
import sys
import threading
from typing import Optional, Dict, Any
from contextlib import contextmanager
from functools import wraps


# Thread-local storage for request context
_context = threading.local()


def get_correlation_id() -> Optional[str]:
    """Get the current correlation ID from context."""
    return getattr(_context, "correlation_id", None)


def set_correlation_id(correlation_id: str):
    """Set the correlation ID for the current thread."""
    _context.correlation_id = correlation_id


def get_session_id() -> Optional[str]:
    """Get the current session ID from context."""
    return getattr(_context, "session_id", None)


def set_session_id(session_id: str):
    """Set the session ID for the current thread."""
    _context.session_id = session_id


@contextmanager
def log_context(correlation_id: str = None, session_id: str = None, **extra):
    """
    Context manager to set logging context for a block.
    
    Usage:
        with log_context(session_id="abc123", intent="booking"):
            logger.info("Processing request")
    """
    old_correlation_id = getattr(_context, "correlation_id", None)
    old_session_id = getattr(_context, "session_id", None)
    old_extra = getattr(_context, "extra", {})
    
    if correlation_id:
        _context.correlation_id = correlation_id
    if session_id:
        _context.session_id = session_id
    _context.extra = {**old_extra, **extra}
    
    try:
        yield
    finally:
        _context.correlation_id = old_correlation_id
        _context.session_id = old_session_id
        _context.extra = old_extra


class StructuredFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.
    Compatible with Google Cloud Logging and other log aggregators.
    """
    
    def __init__(self, service_name: str = "bella-voice-ai"):
        super().__init__()
        self.service_name = service_name
    
    def format(self, record: logging.LogRecord) -> str:
        # Base log entry
        log_entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + f".{int(record.msecs):03d}Z",
            "severity": record.levelname,
            "message": record.getMessage(),
            "service": self.service_name,
            "logger": record.name,
        }
        
        # Add location info
        log_entry["logging.googleapis.com/sourceLocation"] = {
            "file": record.filename,
            "line": record.lineno,
            "function": record.funcName,
        }
        
        # Add correlation ID if present
        correlation_id = get_correlation_id()
        if correlation_id:
            log_entry["logging.googleapis.com/trace"] = correlation_id
        
        # Add session ID if present
        session_id = get_session_id()
        if session_id:
            log_entry["session_id"] = session_id
        
        # Add any extra context
        extra = getattr(_context, "extra", {})
        if extra:
            log_entry["context"] = extra
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add any extra attributes from the log call
        for key, value in record.__dict__.items():
            if key not in (
                "name", "msg", "args", "created", "filename", "funcName",
                "levelname", "levelno", "lineno", "module", "msecs",
                "pathname", "process", "processName", "relativeCreated",
                "stack_info", "exc_info", "exc_text", "thread", "threadName",
                "message", "asctime"
            ):
                if not key.startswith("_"):
                    log_entry[key] = value
        
        return json.dumps(log_entry, default=str)


class HumanReadableFormatter(logging.Formatter):
    """
    Human-readable formatter for local development.
    Includes colors and context info.
    """
    
    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"
    
    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        
        # Timestamp
        timestamp = time.strftime("%H:%M:%S", time.localtime(record.created))
        
        # Build prefix with context
        prefix_parts = [f"{timestamp}"]
        
        session_id = get_session_id()
        if session_id:
            prefix_parts.append(f"[{session_id}]")
        
        prefix = " ".join(prefix_parts)
        
        # Format message
        message = record.getMessage()
        
        # Add level with color
        formatted = f"{prefix} {color}{record.levelname:8}{self.RESET} {message}"
        
        # Add exception if present
        if record.exc_info:
            formatted += "\n" + self.formatException(record.exc_info)
        
        return formatted


def setup_logging(
    level: str = "INFO",
    json_output: bool = False,
    service_name: str = "bella-voice-ai",
):
    """
    Configure logging for the application.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        json_output: Use JSON format (for production)
        service_name: Service name for structured logs
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create handler
    handler = logging.StreamHandler(sys.stdout)
    
    if json_output:
        handler.setFormatter(StructuredFormatter(service_name))
    else:
        handler.setFormatter(HumanReadableFormatter())
    
    root_logger.addHandler(handler)
    
    # Reduce noise from libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("google").setLevel(logging.WARNING)


class PerformanceLogger:
    """
    Helper for logging performance metrics.
    """
    
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
        self._timers: Dict[str, float] = {}
    
    @contextmanager
    def timer(self, operation: str, log_level: int = logging.INFO):
        """
        Context manager to time and log an operation.
        
        Usage:
            with perf_logger.timer("tts_synthesis"):
                audio = synthesize(text)
        """
        start = time.time()
        try:
            yield
        finally:
            duration_ms = (time.time() - start) * 1000
            self.logger.log(
                log_level,
                f"{operation} completed in {duration_ms:.1f}ms",
                extra={"operation": operation, "duration_ms": duration_ms}
            )
    
    def log_latency(
        self,
        operation: str,
        duration_ms: float,
        thresholds: Dict[str, float] = None,
    ):
        """
        Log latency with automatic severity based on thresholds.
        """
        thresholds = thresholds or {"warning": 500, "error": 1000}
        
        if duration_ms >= thresholds.get("error", 1000):
            level = logging.ERROR
            status = "SLOW"
        elif duration_ms >= thresholds.get("warning", 500):
            level = logging.WARNING
            status = "WARN"
        else:
            level = logging.INFO
            status = "OK"
        
        self.logger.log(
            level,
            f"[{status}] {operation}: {duration_ms:.1f}ms",
            extra={"operation": operation, "duration_ms": duration_ms, "status": status}
        )


# Convenience decorator for timing functions
def log_execution_time(logger: logging.Logger = None, operation_name: str = None):
    """
    Decorator to log function execution time.
    
    Usage:
        @log_execution_time()
        async def process_request(text):
            ...
    """
    def decorator(func):
        nonlocal operation_name
        if operation_name is None:
            operation_name = func.__name__
        
        _logger = logger or logging.getLogger(func.__module__)
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.time()
            try:
                return await func(*args, **kwargs)
            finally:
                duration_ms = (time.time() - start) * 1000
                _logger.info(
                    f"{operation_name} completed in {duration_ms:.1f}ms",
                    extra={"operation": operation_name, "duration_ms": duration_ms}
                )
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                duration_ms = (time.time() - start) * 1000
                _logger.info(
                    f"{operation_name} completed in {duration_ms:.1f}ms",
                    extra={"operation": operation_name, "duration_ms": duration_ms}
                )
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator
