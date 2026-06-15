"""
Health check and readiness endpoints for production deployment.
Provides Kubernetes/Cloud Run compatible health probes.
"""
import time
import logging
from typing import Dict, Any, List, Callable
from dataclasses import dataclass, field
from enum import Enum

from infra.redis_cache import get_redis_client, REDIS_AVAILABLE
from infra.circuit_breaker import CircuitBreakerManager
from infra.telemetry import get_telemetry

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class ComponentHealth:
    name: str
    status: HealthStatus
    latency_ms: float = 0.0
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


class HealthChecker:
    """
    Comprehensive health checking for all system components.
    """
    
    def __init__(self):
        self._checks: Dict[str, Callable[[], ComponentHealth]] = {}
        self._last_check_time: float = 0
        self._cached_result: Dict[str, Any] = {}
        self._cache_ttl: float = 5.0  # Cache health for 5 seconds
        
        # Register default checks
        self._register_default_checks()
    
    def _register_default_checks(self):
        """Register built-in health checks."""
        self.register_check("redis", self._check_redis)
        self.register_check("database", self._check_database)
        self.register_check("circuit_breakers", self._check_circuit_breakers)
        self.register_check("telemetry", self._check_telemetry)
    
    def register_check(self, name: str, check_fn: Callable[[], ComponentHealth]):
        """Register a custom health check."""
        self._checks[name] = check_fn

    def _check_database(self) -> ComponentHealth:
        """Check SQLite database connectivity."""
        start = time.time()
        try:
            import database as db
            with db.get_db() as conn:
                conn.execute("SELECT 1")
            
            latency = (time.time() - start) * 1000
            return ComponentHealth(
                name="database",
                status=HealthStatus.HEALTHY,
                latency_ms=latency,
                message="Connected",
            )
        except Exception as e:
            return ComponentHealth(
                name="database",
                status=HealthStatus.UNHEALTHY,
                latency_ms=(time.time() - start) * 1000,
                message=str(e),
            )
    
    def _check_redis(self) -> ComponentHealth:
        """Check Redis connectivity."""
        start = time.time()
        
        if not REDIS_AVAILABLE:
            return ComponentHealth(
                name="redis",
                status=HealthStatus.DEGRADED,
                message="Redis library not installed (caching disabled)",
            )
        
        client = get_redis_client()
        if not client:
            return ComponentHealth(
                name="redis",
                status=HealthStatus.DEGRADED,
                message="Redis not connected (caching disabled)",
            )
        
        try:
            client.ping()
            latency = (time.time() - start) * 1000
            
            # Get Redis info
            info = client.info("memory")
            
            return ComponentHealth(
                name="redis",
                status=HealthStatus.HEALTHY,
                latency_ms=latency,
                message="Connected",
                details={
                    "used_memory_human": info.get("used_memory_human", "unknown"),
                    "connected_clients": info.get("connected_clients", 0),
                }
            )
        except Exception as e:
            return ComponentHealth(
                name="redis",
                status=HealthStatus.UNHEALTHY,
                latency_ms=(time.time() - start) * 1000,
                message=str(e),
            )
    
    def _check_circuit_breakers(self) -> ComponentHealth:
        """Check circuit breaker states."""
        try:
            from infra.circuit_breaker import get_circuit_manager
            manager = get_circuit_manager()
            health = manager.health_check()
            stats = manager.get_all_stats()
            
            # health_check returns {name: is_available}
            # stats returns {name: CircuitStats}
            
            open_breakers = [
                name for name, available in health.items()
                if not available
            ]
            
            details = {
                name: {
                    "state": s.state.value,
                    "failure_count": s.failure_count,
                    "total_calls": s.total_calls
                } for name, s in stats.items()
            }
            
            if open_breakers:
                return ComponentHealth(
                    name="circuit_breakers",
                    status=HealthStatus.DEGRADED,
                    message=f"Open circuits: {', '.join(open_breakers)}",
                    details=details,
                )
            
            return ComponentHealth(
                name="circuit_breakers",
                status=HealthStatus.HEALTHY,
                message="All circuits closed",
                details=details,
            )
        except Exception as e:
            return ComponentHealth(
                name="circuit_breakers",
                status=HealthStatus.UNHEALTHY,
                message=str(e),
            )
    
    def _check_telemetry(self) -> ComponentHealth:
        """Check telemetry system."""
        try:
            telemetry = get_telemetry()
            health = telemetry.get_health()
            
            return ComponentHealth(
                name="telemetry",
                status=HealthStatus.HEALTHY if health["enabled"] else HealthStatus.DEGRADED,
                message="Enabled" if health["enabled"] else "Disabled (metrics not collected)",
                details=health,
            )
        except Exception as e:
            return ComponentHealth(
                name="telemetry",
                status=HealthStatus.DEGRADED,
                message=str(e),
            )
    
    def check_health(self, use_cache: bool = True) -> Dict[str, Any]:
        """
        Run all health checks and return aggregated status.
        
        Returns:
            {
                "status": "healthy|degraded|unhealthy",
                "timestamp": 1234567890.123,
                "components": {...}
            }
        """
        # Return cached result if fresh
        if use_cache and (time.time() - self._last_check_time) < self._cache_ttl:
            return self._cached_result
        
        components = {}
        overall_status = HealthStatus.HEALTHY
        
        for name, check_fn in self._checks.items():
            try:
                result = check_fn()
                components[name] = {
                    "status": result.status.value,
                    "latency_ms": result.latency_ms,
                    "message": result.message,
                    "details": result.details,
                }
                
                # Downgrade overall status
                if result.status == HealthStatus.UNHEALTHY:
                    overall_status = HealthStatus.UNHEALTHY
                elif result.status == HealthStatus.DEGRADED and overall_status != HealthStatus.UNHEALTHY:
                    overall_status = HealthStatus.DEGRADED
                    
            except Exception as e:
                components[name] = {
                    "status": "error",
                    "message": str(e),
                }
                overall_status = HealthStatus.UNHEALTHY
        
        result = {
            "status": overall_status.value,
            "timestamp": time.time(),
            "components": components,
        }
        
        self._cached_result = result
        self._last_check_time = time.time()
        
        return result
    
    def check_liveness(self) -> Dict[str, Any]:
        """
        Kubernetes liveness probe - is the process alive?
        Should be fast and not check dependencies.
        """
        return {
            "status": "alive",
            "timestamp": time.time(),
        }
    
    def check_readiness(self) -> Dict[str, Any]:
        """
        Kubernetes readiness probe - can we serve traffic?
        Checks critical dependencies.
        """
        health = self.check_health(use_cache=True)
        
        # Ready if healthy or degraded (still functional)
        is_ready = health["status"] in ["healthy", "degraded"]
        
        return {
            "ready": is_ready,
            "status": health["status"],
            "timestamp": time.time(),
        }


# Global instance
_health_checker: "HealthChecker" = None


def get_health_checker() -> HealthChecker:
    """Get the global health checker."""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
    return _health_checker


# ============ HTTP Endpoint Helpers ============
# These can be used with Flask, FastAPI, or any HTTP framework

def health_endpoint_handler() -> tuple:
    """
    Handler for /health endpoint.
    Returns (response_dict, status_code)
    """
    health = get_health_checker().check_health()
    status_code = 200 if health["status"] == "healthy" else 503
    return health, status_code


def liveness_endpoint_handler() -> tuple:
    """
    Handler for /healthz or /livez endpoint.
    Returns (response_dict, status_code)
    """
    return get_health_checker().check_liveness(), 200


def readiness_endpoint_handler() -> tuple:
    """
    Handler for /readyz endpoint.
    Returns (response_dict, status_code)
    """
    result = get_health_checker().check_readiness()
    status_code = 200 if result["ready"] else 503
    return result, status_code
