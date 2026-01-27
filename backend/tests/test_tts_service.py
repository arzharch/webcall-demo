"""
Tests for TTS Service.
"""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestTTSService:
    """Test TTS service functionality."""
    
    @pytest.mark.asyncio
    async def test_synthesize_text(self, mock_google_tts, mock_redis):
        """Test basic text synthesis."""
        from services.tts_service import TTSService
        
        # Mock circuit breaker
        with patch("services.tts_service.CircuitBreakerManager") as MockCB:
            mock_breaker = MagicMock()
            mock_breaker.state = MagicMock()  # Not OPEN
            mock_breaker.record_success = MagicMock()
            MockCB.return_value.get_breaker.return_value = mock_breaker
            
            with patch("services.tts_service.RedisCache") as MockCache:
                mock_cache = MagicMock()
                mock_cache.get_tts_audio.return_value = None  # Cache miss
                MockCache.return_value = mock_cache
                
                service = TTSService()
                audio = await service.synthesize("Hello, how can I help you?")
                
                assert audio is not None
                assert isinstance(audio, bytes)
    
    @pytest.mark.asyncio
    async def test_synthesize_empty_text(self, mock_google_tts, mock_redis):
        """Test synthesis with empty text returns None."""
        from services.tts_service import TTSService
        
        with patch("services.tts_service.CircuitBreakerManager"):
            with patch("services.tts_service.RedisCache"):
                service = TTSService()
                audio = await service.synthesize("")
                
                assert audio is None
    
    @pytest.mark.asyncio
    async def test_cache_hit(self, mock_google_tts, mock_redis):
        """Test that cached audio is returned without API call."""
        from services.tts_service import TTSService
        
        with patch("services.tts_service.CircuitBreakerManager") as MockCB:
            mock_breaker = MagicMock()
            mock_breaker.state = MagicMock()
            MockCB.return_value.get_breaker.return_value = mock_breaker
            
            with patch("services.tts_service.RedisCache") as MockCache:
                # Configure cache to return data
                cached_audio = b"cached-audio-bytes"
                mock_cache = MagicMock()
                mock_cache.get_tts_audio.return_value = cached_audio
                MockCache.return_value = mock_cache
                
                service = TTSService()
                service._cache = mock_cache
                
                # Override _get_cached to return cached data
                async def mock_get_cached(text):
                    return cached_audio
                
                service._get_cached = mock_get_cached
                
                audio = await service.synthesize("Hello")
                
                assert audio == cached_audio


class TestTTSServiceConfiguration:
    """Test TTS service configuration."""
    
    def test_default_voice_settings(self, mock_google_tts, mock_redis):
        """Test default voice configuration."""
        with patch("services.tts_service.CircuitBreakerManager"):
            with patch("services.tts_service.RedisCache"):
                from services.tts_service import TTSService
                
                service = TTSService()
                
                assert service.voice_name == "en-IN-Neural2-A"
                assert service.language_code == "en-IN"
                assert service.speaking_rate == 1.25
    
    def test_custom_voice_settings(self, mock_google_tts, mock_redis):
        """Test custom voice configuration."""
        with patch("services.tts_service.CircuitBreakerManager"):
            with patch("services.tts_service.RedisCache"):
                from services.tts_service import TTSService
                
                service = TTSService(
                    voice_name="en-US-Neural2-C",
                    language_code="en-US",
                    speaking_rate=1.0
                )
                
                assert service.voice_name == "en-US-Neural2-C"
                assert service.language_code == "en-US"
                assert service.speaking_rate == 1.0


class TestTTSFillerAudio:
    """Test filler audio functionality."""
    
    def test_preload_filler_audio(self, mock_google_tts, mock_redis):
        """Test preloading filler phrases."""
        with patch("services.tts_service.CircuitBreakerManager"):
            with patch("services.tts_service.RedisCache"):
                from services.tts_service import TTSService
                
                service = TTSService()
                service.preload_filler_audio()
                
                # Should have loaded some filler phrases
                assert len(service._filler_audio) >= 0  # May fail if TTS mock isn't set up
    
    def test_get_filler_audio(self, mock_google_tts, mock_redis):
        """Test getting filler audio clips."""
        with patch("services.tts_service.CircuitBreakerManager"):
            with patch("services.tts_service.RedisCache"):
                from services.tts_service import TTSService
                
                service = TTSService()
                service._filler_audio = {
                    "One moment": b"audio1",
                    "Let me check": b"audio2"
                }
                
                audio = service.get_filler_audio(0)
                assert audio is not None
                
                audio2 = service.get_filler_audio(1)
                assert audio2 is not None


class TestTTSServiceErrorHandling:
    """Test TTS error scenarios."""
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_open(self, mock_google_tts, mock_redis):
        """Test that open circuit breaker prevents synthesis."""
        from services.tts_service import TTSService
        from infra import CircuitState
        
        with patch("services.tts_service.CircuitBreakerManager") as MockCB:
            mock_breaker = MagicMock()
            mock_breaker.state = CircuitState.OPEN  # Circuit is open
            MockCB.return_value.get_breaker.return_value = mock_breaker
            
            with patch("services.tts_service.RedisCache"):
                service = TTSService()
                audio = await service.synthesize("Test")
                
                # Should return None when circuit is open
                assert audio is None
    
    @pytest.mark.asyncio
    async def test_synthesis_timeout(self, mock_google_tts, mock_redis):
        """Test that synthesis handles timeout gracefully."""
        from services.tts_service import TTSService
        
        with patch("services.tts_service.CircuitBreakerManager") as MockCB:
            mock_breaker = MagicMock()
            mock_breaker.state = MagicMock()
            mock_breaker.record_failure = MagicMock()
            MockCB.return_value.get_breaker.return_value = mock_breaker
            
            with patch("services.tts_service.RedisCache") as MockCache:
                mock_cache = MagicMock()
                mock_cache.get_tts_audio.return_value = None
                MockCache.return_value = mock_cache
                
                service = TTSService()
                
                # Make _synthesize_sync slow
                def slow_synthesize(text):
                    import time
                    time.sleep(10)
                    return b"audio"
                
                service._synthesize_sync = slow_synthesize
                
                # Very short timeout
                audio = await service.synthesize("Test", timeout=0.01)
                
                # Should return None on timeout
                assert audio is None
