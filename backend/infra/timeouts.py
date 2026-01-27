"""
Timeout management for external calls.
Provides consistent timeout handling with logging.
"""
import asyncio
import functools
import logging
from typing import Callable, Any, Optional, TypeVar
from dataclasses import dataclass

from infra.config import config

logger = logging.getLogger(__name__)

T = TypeVar('T')


class TimeoutError(Exception):
    """Raised when an operation times out."""
    def __init__(self, operation: str, timeout: float):
        self.operation = operation
        self.timeout = timeout
        super().__init__(f"Operation '{operation}' timed out after {timeout}s")


@dataclass
class TimeoutStats:
    """Statistics for timeout tracking."""
    operation: str
    total_calls: int = 0
    timeout_count: int = 0
    avg_duration_ms: float = 0.0


class TimeoutManager:
    """
    Manages timeouts for various operations with statistics tracking.
    """
    
    def __init__(self):
        self._stats: dict[str, TimeoutStats] = {}
        self._durations: dict[str, list[float]] = {}  # Rolling window
        self._window_size = 100  # Keep last N durations
    
    def get_timeout(self, operation: str) -> float:
        """Get configured timeout for an operation."""
        timeouts = config.timeouts
        return getattr(timeouts, operation, timeouts.total_turn)
    
    def record_success(self, operation: str, duration_ms: float):
        """Record a successful operation."""
        if operation not in self._stats:
            self._stats[operation] = TimeoutStats(operation=operation)
            self._durations[operation] = []
        
        stats = self._stats[operation]
        stats.total_calls += 1
        
        # Update rolling average
        durations = self._durations[operation]
        durations.append(duration_ms)
        if len(durations) > self._window_size:
            durations.pop(0)
        stats.avg_duration_ms = sum(durations) / len(durations)
    
    def record_timeout(self, operation: str):
        """Record a timeout."""
        if operation not in self._stats:
            self._stats[operation] = TimeoutStats(operation=operation)
            self._durations[operation] = []
        
        self._stats[operation].total_calls += 1
        self._stats[operation].timeout_count += 1
        logger.warning(f"Timeout recorded for '{operation}'")
    
    def get_stats(self, operation: str = None) -> dict:
        """Get statistics for one or all operations."""
        if operation:
            return self._stats.get(operation, TimeoutStats(operation=operation)).__dict__
        return {op: stats.__dict__ for op, stats in self._stats.items()}
    
    async def execute_with_timeout(
        self,
        operation: str,
        coro,
        timeout: float = None,
        fallback: Any = None,
    ) -> Any:
        """
        Execute a coroutine with timeout.
        
        Args:
            operation: Name of the operation (for logging/stats)
            coro: The coroutine to execute
            timeout: Override default timeout
            fallback: Value to return on timeout (if None, raises TimeoutError)
        
        Returns:
            Result of the coroutine or fallback value
        """
        timeout = timeout or self.get_timeout(operation)
        start_time = asyncio.get_event_loop().time()
        
        try:
            result = await asyncio.wait_for(coro, timeout=timeout)
            duration_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            self.record_success(operation, duration_ms)
            return result
            
        except asyncio.TimeoutError:
            self.record_timeout(operation)
            if fallback is not None:
                logger.warning(f"'{operation}' timed out, using fallback")
                return fallback
            raise TimeoutError(operation, timeout)


# Global instance
_timeout_manager: Optional[TimeoutManager] = None


def get_timeout_manager() -> TimeoutManager:
    """Get the global timeout manager."""
    global _timeout_manager
    if _timeout_manager is None:
        _timeout_manager = TimeoutManager()
    return _timeout_manager


def with_timeout(
    operation: str,
    timeout: float = None,
    fallback: Any = None,
):
    """
    Decorator to add timeout to an async function.
    
    Args:
        operation: Name of the operation
        timeout: Timeout in seconds (uses config default if None)
        fallback: Value to return on timeout
    
    Usage:
        @with_timeout("llm_call", timeout=3.0, fallback="Sorry, try again.")
        async def call_llm(prompt):
            ...
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            manager = get_timeout_manager()
            coro = func(*args, **kwargs)
            return await manager.execute_with_timeout(
                operation=operation,
                coro=coro,
                timeout=timeout,
                fallback=fallback,
            )
        return wrapper
    return decorator


class OperationTimer:
    """
    Context manager for timing operations.
    
    Usage:
        async with OperationTimer("llm_call") as timer:
            result = await llm.invoke(...)
        print(f"Took {timer.duration_ms}ms")
    """
    
    def __init__(self, operation: str, log_threshold_ms: float = 1000):
        self.operation = operation
        self.log_threshold_ms = log_threshold_ms
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.duration_ms: float = 0
    
    async def __aenter__(self):
        self.start_time = asyncio.get_event_loop().time()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.end_time = asyncio.get_event_loop().time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        
        # Record in manager
        manager = get_timeout_manager()
        if exc_type is None:
            manager.record_success(self.operation, self.duration_ms)
        elif exc_type is asyncio.TimeoutError:
            manager.record_timeout(self.operation)
        
        # Log slow operations
        if self.duration_ms > self.log_threshold_ms:
            logger.warning(
                f"Slow operation '{self.operation}': {self.duration_ms:.0f}ms "
                f"(threshold: {self.log_threshold_ms}ms)"
            )
        
        return False  # Don't suppress exceptions
    
    def __enter__(self):
        import time
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        
        if self.duration_ms > self.log_threshold_ms:
            logger.warning(
                f"Slow operation '{self.operation}': {self.duration_ms:.0f}ms"
            )
        
        return False
