"""
Text-to-Speech Service using Google Cloud TTS.
Includes Redis caching and circuit breaker protection.
"""
import asyncio
import hashlib
import logging
from typing import Optional
from google.cloud import texttospeech

from config import settings
from infra import RedisCache, CircuitBreakerManager, CircuitState

logger = logging.getLogger(__name__)


class TTSService:
    """
    Google Cloud Text-to-Speech service with production features:
    - Redis caching for repeated phrases
    - Circuit breaker for fault tolerance
    - Async/await support for WebSocket integration
    """
    
    # Pre-defined audio assets
    FILLER_PHRASES = [
        "Just a moment, let me check.",
        "Let me look that up for you.",
        "One moment please.",
    ]
    
    def __init__(
        self,
        voice_name: str = "en-IN-Neural2-A",
        language_code: str = "en-IN",
        speaking_rate: float = 1.25,
        cache: Optional[RedisCache] = None,
        circuit_breakers: Optional[CircuitBreakerManager] = None,
    ):
        """
        Initialize TTS service.
        
        Args:
            voice_name: Google TTS voice name
            language_code: BCP-47 language code
            speaking_rate: Speech rate (1.0 = normal)
            cache: Redis cache instance for caching
            circuit_breakers: Circuit breaker manager
        """
        self.voice_name = voice_name
        self.language_code = language_code
        self.speaking_rate = speaking_rate
        
        # Initialize Google TTS client
        self._client = texttospeech.TextToSpeechClient()
        
        # Production infrastructure
        self._cache = cache or RedisCache()
        self._circuit_breakers = circuit_breakers or CircuitBreakerManager()
        self._breaker = self._circuit_breakers.get_breaker("tts")
        
        # Preloaded audio assets
        self._filler_audio: dict[str, bytes] = {}
        
        # Voice config (reusable)
        self._voice = texttospeech.VoiceSelectionParams(
            language_code=self.language_code,
            name=self.voice_name,
        )
        self._audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=24000,  # High quality
            speaking_rate=self.speaking_rate,
        )
        
    def preload_filler_audio(self):
        """Pre-synthesize filler phrases for latency masking."""
        logger.info("Preloading filler audio...")
        for phrase in self.FILLER_PHRASES:
            try:
                audio = self._synthesize_sync(phrase)
                if audio:
                    self._filler_audio[phrase] = audio
                    logger.debug(f"Preloaded: {phrase}")
            except Exception as e:
                logger.warning(f"Failed to preload filler '{phrase}': {e}")
        logger.info(f"Preloaded {len(self._filler_audio)} filler phrases")
        
    def get_filler_audio(self, index: int = 0) -> Optional[bytes]:
        """Get a preloaded filler audio clip."""
        phrases = list(self._filler_audio.keys())
        if not phrases:
            return None
        phrase = phrases[index % len(phrases)]
        return self._filler_audio.get(phrase)
    
    async def synthesize(self, text: str, timeout: float = 5.0) -> Optional[bytes]:
        """
        Convert text to speech audio.
        
        Args:
            text: Text to synthesize
            timeout: Timeout in seconds
            
        Returns:
            LINEAR16 audio bytes at 24kHz, or None on failure
        """
        # Clean and validate text
        clean_text = self._clean_text(text)
        if not clean_text:
            return None
        
        # Check circuit breaker
        if self._breaker.state == CircuitState.OPEN:
            logger.warning("TTS circuit breaker OPEN, skipping synthesis")
            return None
        
        try:
            # Check cache first
            cached_audio = await self._get_cached(clean_text)
            if cached_audio:
                logger.debug(f"TTS cache HIT: {clean_text[:30]}...")
                return cached_audio
            
            # Synthesize with timeout
            loop = asyncio.get_event_loop()
            audio = await asyncio.wait_for(
                loop.run_in_executor(None, self._synthesize_sync, clean_text),
                timeout=timeout
            )
            
            if audio:
                # Cache the result
                await self._set_cached(clean_text, audio)
                self._breaker.record_success()
                logger.debug(f"TTS synthesized: {len(audio)} bytes for {len(clean_text)} chars")
                
            return audio
            
        except asyncio.TimeoutError as e:
            logger.error("TTS synthesis timed out")
            self._breaker.record_failure(e)
            return None
        except Exception as e:
            logger.error(f"TTS synthesis error: {e}")
            self._breaker.record_failure(e)
            return None
    
    def synthesize_sync(self, text: str) -> Optional[bytes]:
        """Synchronous synthesis (for blocking contexts)."""
        clean_text = self._clean_text(text)
        if not clean_text:
            return None
        
        # Check cache synchronously
        cached = self._cache.get_tts_audio(clean_text, self.voice_name)
        if cached:
            return cached
        
        # Synthesize
        audio = self._synthesize_sync(clean_text)
        if audio:
            self._cache.set_tts_audio(clean_text, self.voice_name, audio)
        return audio
    
    def _synthesize_sync(self, text: str) -> Optional[bytes]:
        """Internal synchronous synthesis."""
        try:
            synthesis_input = texttospeech.SynthesisInput(text=text)
            response = self._client.synthesize_speech(
                input=synthesis_input,
                voice=self._voice,
                audio_config=self._audio_config,
            )
            return response.audio_content
        except Exception as e:
            logger.error(f"Google TTS API error: {e}")
            return None
    
    def _clean_text(self, text: str) -> str:
        """Clean text for synthesis."""
        if not text:
            return ""
        # Remove markdown asterisks and extra whitespace
        clean = text.replace("*", "").strip()
        return clean
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return f"tts:{self.voice_name}:{text_hash}"
    
    async def _get_cached(self, text: str) -> Optional[bytes]:
        """Get cached audio from Redis."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._cache.get_tts_audio,
            text,
            self.voice_name
        )
    
    async def _set_cached(self, text: str, audio: bytes):
        """Cache audio in Redis."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            self._cache.set_tts_audio,
            text,
            self.voice_name,
            audio
        )


class TTSStreamProcessor:
    """
    Processes streaming text into sentence chunks for TTS.
    Enables low-latency sentence-by-sentence synthesis.
    """
    
    SENTENCE_ENDINGS = {'.', '?', '!'}
    
    def __init__(self, tts_service: TTSService):
        self.tts = tts_service
        self._buffer = ""
        self._pending_tasks: list = []
        
    async def process_chunk(self, text_chunk: str) -> Optional[bytes]:
        """
        Process a streaming text chunk.
        Returns audio if a complete sentence is detected.
        
        Args:
            text_chunk: Incoming text chunk from LLM
            
        Returns:
            Audio bytes if sentence complete, None otherwise
        """
        self._buffer += text_chunk
        
        # Check for sentence endings
        for ending in self.SENTENCE_ENDINGS:
            if ending in text_chunk:
                sentence = self._buffer.strip()
                self._buffer = ""
                
                if sentence:
                    return await self.tts.synthesize(sentence)
        
        return None
    
    async def flush(self) -> Optional[bytes]:
        """
        Flush remaining buffer and synthesize.
        Call at end of LLM response.
        """
        if self._buffer.strip():
            sentence = self._buffer.strip()
            self._buffer = ""
            return await self.tts.synthesize(sentence)
        return None
    
    def reset(self):
        """Reset the processor state."""
        self._buffer = ""
        self._pending_tasks.clear()
