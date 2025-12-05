import asyncio
import numpy as np
from functools import lru_cache
from typing import Optional

from faster_whisper import WhisperModel
from config import get_settings

class STTService:
    """
    Speech-to-Text service using Faster-Whisper.
    This class runs the transcription model in a separate thread to avoid
    blocking the main asyncio event loop.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.model: Optional[WhisperModel] = None
        self._initialized = False
    
    async def initialize(self):
        """
        Initializes and loads the Whisper model in an executor to prevent blocking.
        """
        if self._initialized:
            return
        
        print(f"🔄 Initializing STT Service (Whisper model: {self.settings.STT_MODEL})...")
        loop = asyncio.get_running_loop()
        
        def _load_model():
            """Loads the model in a synchronous function."""
            return WhisperModel(
                self.settings.STT_MODEL,
                device=self.settings.STT_DEVICE,
                compute_type=self.settings.STT_COMPUTE_TYPE
            )
            
        self.model = await loop.run_in_executor(None, _load_model)
        self._initialized = True
        print("✅ STT Service initialized.")
    
    async def transcribe_audio(self, audio_data: bytes) -> str:
        """
        Transcribes a chunk of raw PCM audio bytes into text.
        The actual transcription is run in an executor thread.
        """
        if not self._initialized:
            await self.initialize()
            
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        loop = asyncio.get_running_loop()

        def _transcribe():
            """Transcribes the audio in a synchronous function."""
            segments, info = self.model.transcribe(audio_array, language="en")
            return " ".join([s.text for s in segments]).strip()

        try:
            transcription = await loop.run_in_executor(None, _transcribe)
            if transcription:
                print(f"🎤 STT Transcription: '{transcription}'")
            return transcription
        except Exception as e:
            print(f"❌ STT Error: {e}")
            return ""

@lru_cache()
def get_stt_service() -> STTService:
    """Get a cached singleton instance of the STTService."""
    return STTService()