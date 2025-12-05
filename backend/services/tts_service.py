import asyncio
import io
import logging
from functools import lru_cache
from typing import AsyncIterator, Optional

import numpy as np
import soundfile as sf
from TTS.api import TTS

from backend.config import get_settings

logger = logging.getLogger(__name__)

class TTSService:
    """Text-to-Speech using Coqui TTS"""
    
    def __init__(self):
        self.settings = get_settings()
        self.model = None
        self.is_initialized = False
    
    async def initialize(self):
        """Initialize TTS model"""
        try:
            self.model = TTS(
                model_name=self.settings.TTS_MODEL,
                in_memory=True,
                gpu=False  # Use CPU for stability
            )
            self.is_initialized = True
            logger.info(f"TTS initialized: {self.settings.TTS_MODEL}")
        except Exception as e:
            logger.error(f"TTS initialization error: {e}")
            raise
    
    async def synthesize(self, text: str) -> bytes:
        """Synthesize text to audio bytes"""
        if not self.is_initialized:
            await self.initialize()
        
        try:
            # Generate speech
            wav_array = await self._generate_speech(text)
            
            # Convert to bytes
            audio_bytes = self._array_to_bytes(wav_array)
            
            logger.info(f"Synthesized: {text[:100]}")
            return audio_bytes
        
        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            return b""
    
    async def _generate_speech(self, text: str) -> np.ndarray:
        """Generate speech array"""
        # Run TTS in thread to avoid blocking
        loop = asyncio.get_event_loop()
        wav = await loop.run_in_executor(
            None,
            lambda: self.model.tts(text)
        )
        
        return np.array(wav) if isinstance(wav, list) else wav
    
    def _array_to_bytes(self, audio_array: np.ndarray) -> bytes:
        """Convert audio array to WAV bytes"""
        buffer = io.BytesIO()
        
        sf.write(
            buffer,
            audio_array,
            self.settings.SAMPLE_RATE,
            format='WAV'
        )
        
        buffer.seek(0)
        return buffer.read()

@lru_cache()
def get_tts_service() -> TTSService:
    """Get cached TTS service"""
    return TTSService()
