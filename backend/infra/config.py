"""
Production configuration for the voice AI system.
Centralized config with environment variable overrides.
"""
import os
from dataclasses import dataclass, field
from typing import Dict, Optional
from dotenv import load_dotenv

load_dotenv()


@dataclass
class RedisConfig:
    """Redis connection configuration."""
    host: str = field(default_factory=lambda: os.getenv("REDIS_HOST", "localhost"))
    port: int = field(default_factory=lambda: int(os.getenv("REDIS_PORT", "6379")))
    password: Optional[str] = field(default_factory=lambda: os.getenv("REDIS_PASSWORD"))
    db: int = field(default_factory=lambda: int(os.getenv("REDIS_DB", "0")))
    
    # Connection pool settings
    max_connections: int = 20
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0
    retry_on_timeout: bool = True
    
    # Cache TTLs (seconds)
    session_ttl: int = 1800  # 30 minutes
    tts_cache_ttl: int = 86400  # 24 hours
    llm_cache_ttl: int = 300  # 5 minutes
    
    @property
    def url(self) -> str:
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"


@dataclass
class TimeoutConfig:
    """Timeout settings for external calls."""
    intent_classification: float = 2.0
    slot_extraction: float = 3.0
    response_generation: float = 5.0
    tts_synthesis: float = 3.0
    total_turn: float = 12.0
    redis_operation: float = 1.0
    

@dataclass 
class CircuitBreakerConfig:
    """Circuit breaker settings."""
    failure_threshold: int = 5  # Open after N failures
    recovery_timeout: float = 30.0  # Seconds before trying again
    expected_exception: tuple = (Exception,)  # Exceptions that trigger breaker


@dataclass
class ConcurrencyConfig:
    """Concurrency and rate limiting settings."""
    max_concurrent_llm_calls: int = 20
    max_concurrent_tts_calls: int = 30
    requests_per_minute: int = 100  # Per instance
    burst_size: int = 20


@dataclass
class LLMCostConfig:
    """LLM pricing for cost tracking (per 1K tokens)."""
    # OpenAI pricing as of 2024
    models: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    })
    
    # Budget limits
    max_cost_per_turn: float = 0.02  # $0.02
    max_cost_per_session: float = 0.50  # $0.50
    max_cost_per_hour: float = 100.0  # $100


@dataclass
class TelemetryConfig:
    """OpenTelemetry configuration."""
    enabled: bool = field(default_factory=lambda: os.getenv("TELEMETRY_ENABLED", "true").lower() == "true")
    service_name: str = "bella-voice-ai"
    service_version: str = "1.0.0"
    
    # Export destinations
    trace_exporter: str = field(default_factory=lambda: os.getenv("TRACE_EXPORTER", "console"))  # console, gcp, jaeger
    metrics_exporter: str = field(default_factory=lambda: os.getenv("METRICS_EXPORTER", "console"))  # console, gcp, prometheus
    
    # GCP specific
    gcp_project_id: Optional[str] = field(default_factory=lambda: os.getenv("GCP_PROJECT_ID"))
    
    # Sampling
    trace_sample_rate: float = 1.0  # 100% for now, reduce in high-volume prod


@dataclass
class ProductionConfig:
    """Master configuration aggregating all sub-configs."""
    redis: RedisConfig = field(default_factory=RedisConfig)
    timeouts: TimeoutConfig = field(default_factory=TimeoutConfig)
    circuit_breaker: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    concurrency: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)
    llm_cost: LLMCostConfig = field(default_factory=LLMCostConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    
    # Feature flags
    enable_tts_cache: bool = True
    enable_llm_cache: bool = True
    enable_session_persistence: bool = True
    
    @classmethod
    def from_env(cls) -> "ProductionConfig":
        """Create config from environment variables."""
        return cls()


# Global config instance
config = ProductionConfig.from_env()
