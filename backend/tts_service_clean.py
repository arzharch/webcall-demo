from google.cloud import texttospeech
import asyncio
import logging
from functools import lru_cache

from config_clean import get_settings

logger = logging.getLogger(__name__)

class TTSService:
    """Text-to-Speech using Google Cloud TTS"""
    
    def __init__(self):
        self.settings = get_settings()
        self.client = texttospeech.TextToSpeechClient()
        
        self.voice = texttospeech.VoiceSelectionParams(
            language_code=self.settings.TTS_LANGUAGE_CODE,
            name=self.settings.TTS_VOICE_NAME,
        )
        
        self.audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            speaking_rate=self.settings.TTS_SPEAKING_RATE,
            pitch=self.settings.TTS_PITCH,
            sample_rate_hertz=self.settings.SAMPLE_RATE,
        )
        
        logger.info("✅ TTS Service initialized (Google Cloud)")
    
    async def synthesize(self, text: str) -> bytes:
        """
        Convert text to speech audio
        
        Args:
            text: Text to synthesize
        
        Returns:
            Audio bytes (LINEAR16 PCM)
        """
        if not text or not text.strip():
            return b""
        
        try:
            synthesis_input = texttospeech.SynthesisInput(text=text)
            
            # Run blocking API call in executor to avoid blocking event loop
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.synthesize_speech(
                    input=synthesis_input,
                    voice=self.voice,
                    audio_config=self.audio_config
                )
            )
            
            logger.info(f"🔊 Synthesized: {text[:50]}...")
            
            return response.audio_content
        
        except Exception as e:
            logger.error(f"❌ TTS Error: {e}", exc_info=True)
            return b""

@lru_cache()
def get_tts_service() -> TTSService:
    """Get singleton TTS service"""
    return TTSService()
