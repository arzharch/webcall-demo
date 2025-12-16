from deepgram import DeepgramClient, PrerecordedOptions
import asyncio
import logging
from functools import lru_cache

from config_clean import get_settings

logger = logging.getLogger(__name__)

class STTService:
    """Speech-to-Text using Deepgram"""
    
    def __init__(self):
        self.settings = get_settings()
        self.client = DeepgramClient(self.settings.DEEPGRAM_API_KEY)
        logger.info("✅ STT Service initialized (Deepgram)")
    
    async def transcribe(self, audio_bytes: bytes) -> str:
        """
        Transcribe audio bytes to text
        
        Args:
            audio_bytes: Raw audio data (PCM/WAV)
        
        Returns:
            Transcribed text
        """
        try:
            options = PrerecordedOptions(
                model=self.settings.DEEPGRAM_MODEL,
                language=self.settings.DEEPGRAM_LANGUAGE,
                smart_format=True,
                punctuate=True,
            )
            
            # Run blocking API call in executor to avoid blocking event loop
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.listen.prerecorded.v("1").transcribe_file(
                    {"buffer": audio_bytes},
                    options
                )
            )
            
            transcript = response.results.channels[0].alternatives[0].transcript
            
            if transcript:
                logger.info(f"👂 Transcribed: {transcript}")
            
            return transcript
        
        except Exception as e:
            logger.error(f"❌ STT Error: {e}", exc_info=True)
            return ""

@lru_cache()
def get_stt_service() -> STTService:
    """Get singleton STT service"""
    return STTService()
