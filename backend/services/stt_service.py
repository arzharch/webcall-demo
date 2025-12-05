import asyncio
import io
import logging
import numpy as np
from functools import lru_cache
from typing import Optional

from faster_whisper import WhisperModel

from backend.config import get_settings

logger = logging.getLogger(__name__)

class STTService:
    """
    Speech-to-Text service using Faster-Whisper.
    This class runs the transcription model in a separate thread to avoid
    blocking the main asyncio event loop.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.model: Optional[WhisperModel] = None
        self.is_initialized = False
    
    async def initialize(self):
        """
        Initializes and loads the Whisper model in an executor to prevent blocking.
        """
        if self.is_initialized:
            return
        
        print(f"🔄 Initializing STT Service (Whisper model: {self.settings.STT_MODEL})...")
        loop = asyncio.get_running_loop()
        
        def _load_model():
            """Loads the model in a synchronous function."""
            return WhisperModel(
                self.settings.STT_MODEL,
                device=self.settings.STT_DEVICE,
                compute_type=self.settings.STT_COMPUTE_TYPE,
                language=self.settings.STT_LANGUAGE
            )
            
        try:
            self.model = await loop.run_in_executor(None, _load_model)
            self.is_initialized = True
            logger.info(f"STT initialized: {self.settings.STT_MODEL}")
            print("✅ STT Service initialized.")
        except Exception as e:
            logger.error(f"STT initialization error: {e}")
            raise
    
    async def transcribe(self, audio_data: bytes) -> Optional[str]:
        """
        Transcribes a chunk of raw PCM audio bytes into text.
        The actual transcription is run in an executor thread.
        """
        if not self.is_initialized:
            await self.initialize()
        
        try:
            # Convert bytes to audio file-like object
            audio_file = io.BytesIO(audio_data)
            
            # Transcribe
            segments, info = self.model.transcribe(
                audio_file,
                language=self.settings.STT_LANGUAGE,
                beam_size=5
            )
            
            # Combine segments
            text = " ".join([segment.text for segment in segments])
            
            logger.info(f"Transcribed: {text[:100]}")
            return text.strip() or None
        
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return None

@lru_cache()
def get_stt_service() -> STTService:
    """Get a cached singleton instance of the STTService."""
    return STTService()

