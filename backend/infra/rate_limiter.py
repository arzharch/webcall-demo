"""
Concurrency and rate limiting for burst protection.
Prevents system overload during traffic spikes.
"""
import time
import asyncio
import logging
from typing import Optional
from collections import deque
from threading import Lock
from dataclasses import dataclass

from infra.config import config

logger = logging.getLogger(__name__)


@dataclass
class LimiterStats:
    """Statistics for limiters."""
    name: str
    current_usage: int
    max_capacity: int
    rejected_count: int
    total_requests: int


class ConcurrencyLimiter:
    """
    Limits concurrent executions using asyncio.Semaphore.
    Useful for limiting concurrent LLM or TTS calls.
    """
    
    def __init__(self, name: str, max_concurrent: int = None):
        self.name = name
        self.max_concurrent = max_concurrent or config.concurrency.max_concurrent_llm_calls
        self._semaphore = asyncio.Semaphore(self.max_concurrent)
        self._current = 0
        self._rejected = 0
        self._total = 0
        self._lock = Lock()
    
    async def acquire(self, timeout: float = None) -> bool:
        """
        Acquire a slot for execution.
        
        Args:
            timeout: Max seconds to wait for a slot. None = wait forever.
        
        Returns:
            True if acquired, False if timed out.
        """
        self._total += 1
        
        try:
            if timeout:
                await asyncio.wait_for(self._semaphore.acquire(), timeout=timeout)
            else:
                await self._semaphore.acquire()
            
            with self._lock:
                self._current += 1
            
            return True
            
        except asyncio.TimeoutError:
            with self._lock:
                self._rejected += 1
            logger.warning(f"Concurrency limit reached for '{self.name}' (max: {self.max_concurrent})")
            return False
    
    def release(self):
        """Release a slot after execution."""
        self._semaphore.release()
        with self._lock:
            self._current = max(0, self._current - 1)
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.acquire()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        self.release()
        return False
    
    def get_stats(self) -> LimiterStats:
        """Get current statistics."""
        return LimiterStats(
            name=self.name,
            current_usage=self._current,
            max_capacity=self.max_concurrent,
            rejected_count=self._rejected,
            total_requests=self._total,
        )
    
    @property
    def available_slots(self) -> int:
        """Number of available slots."""
        return self.max_concurrent - self._current


class RateLimiter:
    """
    Token bucket rate limiter for requests per minute.
    Allows bursts up to bucket size, then enforces rate.
    """
    
    def __init__(
        self,
        name: str,
        requests_per_minute: int = None,
        burst_size: int = None,
    ):
        self.name = name
        self.rate = (requests_per_minute or config.concurrency.requests_per_minute) / 60.0  # per second
        self.burst_size = burst_size or config.concurrency.burst_size
        
        self._tokens = float(self.burst_size)  # Start with full bucket
        self._last_update = time.time()
        self._rejected = 0
        self._total = 0
        self._lock = Lock()
    
    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self._last_update
        self._tokens = min(self.burst_size, self._tokens + elapsed * self.rate)
        self._last_update = now
    
    def try_acquire(self, tokens: int = 1) -> bool:
        """
        Try to acquire tokens without blocking.
        
        Args:
            tokens: Number of tokens to acquire (default 1)
        
        Returns:
            True if acquired, False if rate limited
        """
        with self._lock:
            self._total += 1
            self._refill()
            
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            else:
                self._rejected += 1
                logger.warning(f"Rate limit exceeded for '{self.name}' ({self.rate * 60:.0f}/min)")
                return False
    
    async def acquire(self, tokens: int = 1, max_wait: float = 5.0) -> bool:
        """
        Acquire tokens, waiting if necessary.
        
        Args:
            tokens: Number of tokens to acquire
            max_wait: Maximum seconds to wait
        
        Returns:
            True if acquired, False if timed out
        """
        start = time.time()
        
        while time.time() - start < max_wait:
            if self.try_acquire(tokens):
                return True
            
            # Calculate wait time for tokens to refill
            wait_time = (tokens - self._tokens) / self.rate
            wait_time = min(wait_time, 0.1)  # Max 100ms between checks
            await asyncio.sleep(wait_time)
        
        with self._lock:
            self._rejected += 1
        return False
    
    def get_stats(self) -> dict:
        """Get current statistics."""
        with self._lock:
            self._refill()
            return {
                "name": self.name,
                "tokens_available": self._tokens,
                "burst_size": self.burst_size,
                "rate_per_minute": self.rate * 60,
                "rejected_count": self._rejected,
                "total_requests": self._total,
            }


class BackpressureController:
    """
    Monitors system load and applies backpressure when overloaded.
    """
    
    def __init__(
        self,
        llm_limiter: ConcurrencyLimiter,
        tts_limiter: ConcurrencyLimiter,
        rate_limiter: RateLimiter,
    ):
        self.llm_limiter = llm_limiter
        self.tts_limiter = tts_limiter
        self.rate_limiter = rate_limiter
        
        # Thresholds
        self.warning_threshold = 0.7  # 70% capacity
        self.critical_threshold = 0.9  # 90% capacity
    
    @property
    def load_factor(self) -> float:
        """Calculate current load factor (0.0 to 1.0+)."""
        llm_load = 1.0 - (self.llm_limiter.available_slots / self.llm_limiter.max_concurrent)
        tts_load = 1.0 - (self.tts_limiter.available_slots / self.tts_limiter.max_concurrent)
        return max(llm_load, tts_load)
    
    @property
    def status(self) -> str:
        """Get current system status."""
        load = self.load_factor
        if load >= self.critical_threshold:
            return "critical"
        elif load >= self.warning_threshold:
            return "warning"
        return "healthy"
    
    def should_accept_request(self) -> bool:
        """Check if system can accept new requests."""
        # Always accept if healthy
        if self.status == "healthy":
            return True
        
        # In warning state, check rate limiter
        if self.status == "warning":
            return self.rate_limiter.try_acquire()
        
        # In critical state, apply aggressive rejection
        # Only accept 20% of requests
        import random
        return random.random() < 0.2 and self.rate_limiter.try_acquire()
    
    def get_status_message(self) -> Optional[str]:
        """Get message to play if system is overloaded."""
        if self.status == "critical":
            return "We're experiencing high call volume. Please hold briefly."
        return None
    
    def get_metrics(self) -> dict:
        """Get all metrics for monitoring."""
        return {
            "status": self.status,
            "load_factor": self.load_factor,
            "llm": self.llm_limiter.get_stats(),
            "tts": self.tts_limiter.get_stats(),
            "rate": self.rate_limiter.get_stats(),
        }


# Global instances
_llm_limiter: Optional[ConcurrencyLimiter] = None
_tts_limiter: Optional[ConcurrencyLimiter] = None
_rate_limiter: Optional[RateLimiter] = None
_backpressure: Optional[BackpressureController] = None


def get_llm_limiter() -> ConcurrencyLimiter:
    """Get LLM concurrency limiter."""
    global _llm_limiter
    if _llm_limiter is None:
        _llm_limiter = ConcurrencyLimiter("llm", config.concurrency.max_concurrent_llm_calls)
    return _llm_limiter


def get_tts_limiter() -> ConcurrencyLimiter:
    """Get TTS concurrency limiter."""
    global _tts_limiter
    if _tts_limiter is None:
        _tts_limiter = ConcurrencyLimiter("tts", config.concurrency.max_concurrent_tts_calls)
    return _tts_limiter


def get_rate_limiter() -> RateLimiter:
    """Get global rate limiter."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter("global")
    return _rate_limiter


def get_backpressure_controller() -> BackpressureController:
    """Get backpressure controller."""
    global _backpressure
    if _backpressure is None:
        _backpressure = BackpressureController(
            get_llm_limiter(),
            get_tts_limiter(),
            get_rate_limiter(),
        )
    return _backpressure
