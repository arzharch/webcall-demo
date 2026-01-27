"""
Redis caching layer for TTS audio, LLM responses, and session state.
Provides both sync and async interfaces with automatic fallback if Redis unavailable.
"""
import json
import hashlib
import logging
from typing import Optional, Any, Union, TYPE_CHECKING
from contextlib import contextmanager
import pickle

try:
    import redis
    from redis import ConnectionPool
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None
    ConnectionPool = None

from infra.config import config

logger = logging.getLogger(__name__)


# Global connection pool (reused across calls)
_redis_pool = None  # type: Optional[Any]
_redis_client = None  # type: Optional[Any]


def get_redis_client() -> Optional[Any]:
    """
    Get or create a Redis client with connection pooling.
    Returns None if Redis is not available or not configured.
    """
    global _redis_pool, _redis_client
    
    if not REDIS_AVAILABLE:
        logger.warning("Redis library not installed. Caching disabled.")
        return None
    
    if _redis_client is not None:
        try:
            _redis_client.ping()
            return _redis_client
        except Exception:
            logger.warning("Redis connection lost, attempting reconnect...")
            _redis_client = None
            _redis_pool = None
    
    try:
        if _redis_pool is None:
            _redis_pool = ConnectionPool(
                host=config.redis.host,
                port=config.redis.port,
                password=config.redis.password,
                db=config.redis.db,
                max_connections=config.redis.max_connections,
                socket_timeout=config.redis.socket_timeout,
                socket_connect_timeout=config.redis.socket_connect_timeout,
                retry_on_timeout=config.redis.retry_on_timeout,
                decode_responses=False,  # We store bytes for audio
            )
        
        _redis_client = redis.Redis(connection_pool=_redis_pool)
        _redis_client.ping()
        logger.info(f"Redis connected: {config.redis.host}:{config.redis.port}")
        return _redis_client
        
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}. Caching disabled.")
        return None


class RedisCache:
    """
    High-level caching interface with specialized methods for TTS, LLM, and sessions.
    Automatically falls back to no-op if Redis is unavailable.
    """
    
    def __init__(self):
        self._client = get_redis_client()
        self._enabled = self._client is not None
        
        # Metrics
        self.hits = 0
        self.misses = 0
    
    @property
    def enabled(self) -> bool:
        return self._enabled
    
    def _safe_operation(self, operation: str, default: Any = None):
        """Decorator-like context for safe Redis operations."""
        @contextmanager
        def wrapper():
            if not self._enabled:
                yield default
                return
            try:
                yield
            except Exception as e:
                logger.warning(f"Redis {operation} failed: {e}")
                yield default
        return wrapper()
    
    # ==================== TTS CACHE ====================
    
    def get_tts_cache_key(self, text: str, voice_id: str, speaking_rate: float = 1.0) -> str:
        """Generate cache key for TTS audio."""
        content = f"{text}|{voice_id}|{speaking_rate}"
        return f"tts:{hashlib.md5(content.encode()).hexdigest()}"
    
    def get_tts_audio(self, text: str, voice_id: str, speaking_rate: float = 1.0) -> Optional[bytes]:
        """Retrieve cached TTS audio."""
        if not self._enabled:
            return None
        
        try:
            key = self.get_tts_cache_key(text, voice_id, speaking_rate)
            audio = self._client.get(key)
            
            if audio:
                self.hits += 1
                logger.debug(f"TTS cache HIT: {key[:20]}...")
                return audio
            else:
                self.misses += 1
                return None
                
        except Exception as e:
            logger.warning(f"TTS cache get failed: {e}")
            return None
    
    def set_tts_audio(self, text: str, voice_id: str, audio_data: bytes, speaking_rate: float = 1.0) -> bool:
        """Store TTS audio in cache."""
        if not self._enabled:
            return False
        
        try:
            key = self.get_tts_cache_key(text, voice_id, speaking_rate)
            self._client.setex(key, config.redis.tts_cache_ttl, audio_data)
            logger.debug(f"TTS cache SET: {key[:20]}... ({len(audio_data)} bytes)")
            return True
            
        except Exception as e:
            logger.warning(f"TTS cache set failed: {e}")
            return False
    
    # ==================== LLM CACHE ====================
    
    def get_llm_cache_key(self, prompt_hash: str, model: str) -> str:
        """Generate cache key for LLM response."""
        return f"llm:{model}:{prompt_hash}"
    
    def get_llm_response(self, messages: list, model: str) -> Optional[str]:
        """Retrieve cached LLM response."""
        if not self._enabled or not config.enable_llm_cache:
            return None
        
        try:
            # Create deterministic hash of messages
            content = json.dumps(messages, sort_keys=True, default=str)
            prompt_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
            key = self.get_llm_cache_key(prompt_hash, model)
            
            response = self._client.get(key)
            if response:
                self.hits += 1
                logger.debug(f"LLM cache HIT: {key}")
                return response.decode('utf-8')
            else:
                self.misses += 1
                return None
                
        except Exception as e:
            logger.warning(f"LLM cache get failed: {e}")
            return None
    
    def set_llm_response(self, messages: list, model: str, response: str) -> bool:
        """Store LLM response in cache."""
        if not self._enabled or not config.enable_llm_cache:
            return False
        
        try:
            content = json.dumps(messages, sort_keys=True, default=str)
            prompt_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
            key = self.get_llm_cache_key(prompt_hash, model)
            
            self._client.setex(key, config.redis.llm_cache_ttl, response.encode('utf-8'))
            logger.debug(f"LLM cache SET: {key}")
            return True
            
        except Exception as e:
            logger.warning(f"LLM cache set failed: {e}")
            return False
    
    # ==================== SESSION CACHE ====================
    
    def get_session_key(self, session_id: str) -> str:
        """Generate cache key for session state."""
        return f"session:{session_id}"
    
    def get_session(self, session_id: str) -> Optional[dict]:
        """Retrieve session state from cache."""
        if not self._enabled or not config.enable_session_persistence:
            return None
        
        try:
            key = self.get_session_key(session_id)
            data = self._client.get(key)
            
            if data:
                return pickle.loads(data)
            return None
            
        except Exception as e:
            logger.warning(f"Session get failed: {e}")
            return None
    
    def set_session(self, session_id: str, state: dict) -> bool:
        """Store session state in cache."""
        if not self._enabled or not config.enable_session_persistence:
            return False
        
        try:
            key = self.get_session_key(session_id)
            data = pickle.dumps(state)
            self._client.setex(key, config.redis.session_ttl, data)
            return True
            
        except Exception as e:
            logger.warning(f"Session set failed: {e}")
            return False
    
    def delete_session(self, session_id: str) -> bool:
        """Delete session from cache."""
        if not self._enabled:
            return False
        
        try:
            key = self.get_session_key(session_id)
            self._client.delete(key)
            return True
        except Exception as e:
            logger.warning(f"Session delete failed: {e}")
            return False
    
    # ==================== UTILITIES ====================
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0.0
        
        return {
            "enabled": self._enabled,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.1f}%",
        }
    
    def health_check(self) -> bool:
        """Check if Redis is healthy."""
        if not self._enabled:
            return False
        try:
            return self._client.ping()
        except Exception:
            return False
    
    def warm_tts_cache(self, phrases: list, synthesize_fn) -> int:
        """
        Pre-warm TTS cache with common phrases.
        
        Args:
            phrases: List of (text, voice_id, speaking_rate) tuples
            synthesize_fn: Function that takes (text, voice_id, rate) and returns audio bytes
        
        Returns:
            Number of phrases cached
        """
        if not self._enabled:
            logger.warning("Cannot warm TTS cache: Redis not available")
            return 0
        
        cached = 0
        for text, voice_id, rate in phrases:
            # Skip if already cached
            if self.get_tts_audio(text, voice_id, rate):
                continue
            
            try:
                audio = synthesize_fn(text, voice_id, rate)
                if audio:
                    self.set_tts_audio(text, voice_id, audio, rate)
                    cached += 1
            except Exception as e:
                logger.warning(f"Failed to warm cache for '{text[:30]}...': {e}")
        
        logger.info(f"TTS cache warmed with {cached} phrases")
        return cached


# Singleton instance
_cache_instance: Optional[RedisCache] = None


def get_cache() -> RedisCache:
    """Get or create the global cache instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = RedisCache()
    return _cache_instance
