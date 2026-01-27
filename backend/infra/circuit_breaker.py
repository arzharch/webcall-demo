"""
Circuit breaker implementation for graceful degradation.
Prevents cascade failures when external services are down.
"""
import time
import logging
import functools
from typing import Callable, Optional, Any, Dict
from enum import Enum
from threading import Lock
from dataclasses import dataclass, field

from infra.config import config

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitStats:
    """Statistics for a circuit breaker."""
    name: str
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    total_calls: int = 0
    total_failures: int = 0


class CircuitBreaker:
    """
    Circuit breaker for a single service/endpoint.
    
    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Service is failing, requests are rejected immediately
    - HALF_OPEN: Testing recovery, allows one request through
    """
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = None,
        recovery_timeout: float = None,
        expected_exceptions: tuple = None,
    ):
        self.name = name
        self.failure_threshold = failure_threshold or config.circuit_breaker.failure_threshold
        self.recovery_timeout = recovery_timeout or config.circuit_breaker.recovery_timeout
        self.expected_exceptions = expected_exceptions or (Exception,)
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._lock = Lock()
        
        # Metrics
        self._total_calls = 0
        self._total_failures = 0
    
    @property
    def state(self) -> CircuitState:
        """Get current state, checking for recovery timeout."""
        with self._lock:
            if self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._state = CircuitState.HALF_OPEN
                    logger.info(f"Circuit '{self.name}' entering HALF_OPEN state")
            return self._state
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to try recovery."""
        if self._last_failure_time is None:
            return True
        return time.time() - self._last_failure_time >= self.recovery_timeout
    
    def record_success(self):
        """Record a successful call."""
        with self._lock:
            self._success_count += 1
            self._total_calls += 1
            
            if self._state == CircuitState.HALF_OPEN:
                # Success in half-open means service recovered
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                logger.info(f"Circuit '{self.name}' CLOSED (service recovered)")
    
    def record_failure(self, error: Exception):
        """Record a failed call."""
        with self._lock:
            self._failure_count += 1
            self._total_calls += 1
            self._total_failures += 1
            self._last_failure_time = time.time()
            
            if self._state == CircuitState.HALF_OPEN:
                # Failure in half-open means service still down
                self._state = CircuitState.OPEN
                logger.warning(f"Circuit '{self.name}' OPEN (recovery failed)")
            
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.failure_threshold:
                    self._state = CircuitState.OPEN
                    logger.warning(
                        f"Circuit '{self.name}' OPEN after {self._failure_count} failures"
                    )
    
    def is_available(self) -> bool:
        """Check if requests can proceed."""
        return self.state != CircuitState.OPEN
    
    def get_stats(self) -> CircuitStats:
        """Get current statistics."""
        return CircuitStats(
            name=self.name,
            state=self._state,
            failure_count=self._failure_count,
            success_count=self._success_count,
            last_failure_time=self._last_failure_time,
            total_calls=self._total_calls,
            total_failures=self._total_failures,
        )


class CircuitBreakerManager:
    """
    Manages multiple circuit breakers for different services.
    """
    
    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = Lock()
    
    def get_breaker(
        self,
        name: str,
        failure_threshold: int = None,
        recovery_timeout: float = None,
    ) -> CircuitBreaker:
        """Get or create a circuit breaker."""
        with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(
                    name=name,
                    failure_threshold=failure_threshold,
                    recovery_timeout=recovery_timeout,
                )
            return self._breakers[name]
    
    def get_all_stats(self) -> Dict[str, CircuitStats]:
        """Get stats for all breakers."""
        return {name: cb.get_stats() for name, cb in self._breakers.items()}
    
    def health_check(self) -> Dict[str, bool]:
        """Check health of all services."""
        return {name: cb.is_available() for name, cb in self._breakers.items()}


# Global manager instance
_manager: Optional[CircuitBreakerManager] = None


def get_circuit_manager() -> CircuitBreakerManager:
    """Get the global circuit breaker manager."""
    global _manager
    if _manager is None:
        _manager = CircuitBreakerManager()
    return _manager


class CircuitOpenError(Exception):
    """Raised when a circuit breaker is open."""
    def __init__(self, breaker_name: str):
        self.breaker_name = breaker_name
        super().__init__(f"Circuit breaker '{breaker_name}' is OPEN")


def circuit_breaker(
    name: str,
    failure_threshold: int = None,
    recovery_timeout: float = None,
    fallback: Callable = None,
):
    """
    Decorator to wrap a function with circuit breaker protection.
    
    Args:
        name: Identifier for this circuit
        failure_threshold: Number of failures before opening
        recovery_timeout: Seconds to wait before attempting recovery
        fallback: Optional fallback function to call when circuit is open
    
    Usage:
        @circuit_breaker("openai", failure_threshold=5, fallback=lambda *a, **k: "Fallback response")
        async def call_openai(prompt):
            ...
    """
    def decorator(func: Callable):
        manager = get_circuit_manager()
        breaker = manager.get_breaker(name, failure_threshold, recovery_timeout)
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            if not breaker.is_available():
                logger.warning(f"Circuit '{name}' is OPEN, using fallback")
                if fallback:
                    return fallback(*args, **kwargs)
                raise CircuitOpenError(name)
            
            try:
                result = await func(*args, **kwargs)
                breaker.record_success()
                return result
            except Exception as e:
                breaker.record_failure(e)
                if fallback and not breaker.is_available():
                    return fallback(*args, **kwargs)
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            if not breaker.is_available():
                logger.warning(f"Circuit '{name}' is OPEN, using fallback")
                if fallback:
                    return fallback(*args, **kwargs)
                raise CircuitOpenError(name)
            
            try:
                result = func(*args, **kwargs)
                breaker.record_success()
                return result
            except Exception as e:
                breaker.record_failure(e)
                if fallback and not breaker.is_available():
                    return fallback(*args, **kwargs)
                raise
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


# Pre-defined fallback responses
FALLBACK_RESPONSES = {
    "llm_failure": "I'm having a brief moment. Could you repeat that?",
    "tts_failure": None,  # Will use cached audio
    "high_load": "We're experiencing high call volume. Please hold.",
    "timeout": "Let me try that again for you.",
}


def get_fallback_response(error_type: str) -> Optional[str]:
    """Get a fallback response for a given error type."""
    return FALLBACK_RESPONSES.get(error_type)
