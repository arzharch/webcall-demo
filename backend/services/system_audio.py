"""
System Audio Assets Service.
Provides pre-generated audio for system sounds like ringing and error messages.
"""
import asyncio
import logging
from typing import Optional
from google.cloud import texttospeech

logger = logging.getLogger(__name__)


class SystemAudioService:
    """
    Manages pre-generated system audio assets.
    
    Audio Assets:
    - Ringing sound: Gentle tone to play while connecting
    - Error message: "Unable to connect at the moment" for failures
    - Hold messages: Various hold/wait messages
    """
    
    # System audio phrases
    SYSTEM_PHRASES = {
        "connecting": "Please hold while we connect your call.",
        "connection_error": "We're unable to connect at the moment. Please try again shortly.",
        "goodbye": "Thank you for calling Bella Cucina. Goodbye!",
        "hold": "Please hold for just a moment.",
    }
    
    def __init__(
        self,
        voice_name: str = "en-IN-Neural2-A",
        language_code: str = "en-IN",
        speaking_rate: float = 1.0,
    ):
        self.voice_name = voice_name
        self.language_code = language_code
        self.speaking_rate = speaking_rate
        
        self._client: Optional[texttospeech.TextToSpeechClient] = None
        self._audio_cache: dict[str, bytes] = {}
        self._ringing_audio: Optional[bytes] = None
        
    def initialize(self):
        """Initialize the TTS client."""
        try:
            self._client = texttospeech.TextToSpeechClient()
            logger.info("System audio service initialized")
        except Exception as e:
            logger.error(f"Failed to initialize system audio service: {e}")
    
    def preload_all(self):
        """Pre-generate all system audio assets."""
        if not self._client:
            self.initialize()
        
        logger.info("Preloading system audio assets...")
        
        # Generate spoken phrases
        for key, phrase in self.SYSTEM_PHRASES.items():
            try:
                audio = self._synthesize(phrase)
                if audio:
                    self._audio_cache[key] = audio
                    logger.debug(f"Preloaded system audio: {key}")
            except Exception as e:
                logger.warning(f"Failed to preload '{key}': {e}")
        
        # Generate ringing tone
        self._generate_ringing_tone()
        
        logger.info(f"Preloaded {len(self._audio_cache)} system audio assets")
    
    def _synthesize(self, text: str) -> Optional[bytes]:
        """Synthesize text to audio."""
        if not self._client:
            return None
        
        try:
            input_text = texttospeech.SynthesisInput(text=text)
            
            voice = texttospeech.VoiceSelectionParams(
                language_code=self.language_code,
                name=self.voice_name,
            )
            
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.LINEAR16,
                sample_rate_hertz=24000,
                speaking_rate=self.speaking_rate,
            )
            
            response = self._client.synthesize_speech(
                input=input_text,
                voice=voice,
                audio_config=audio_config
            )
            
            return response.audio_content
            
        except Exception as e:
            logger.error(f"TTS synthesis error: {e}")
            return None
    
    def _generate_ringing_tone(self):
        """
        Generate a gentle ringing/connecting tone.
        Uses a soft "ring ring" phrase synthesized with TTS.
        """
        # For simplicity, we use TTS to create a "ringing" effect
        # In production, you might use an actual audio file
        ringing_phrase = "Connecting your call... ring... ring..."
        
        try:
            if self._client:
                input_text = texttospeech.SynthesisInput(text=ringing_phrase)
                
                voice = texttospeech.VoiceSelectionParams(
                    language_code=self.language_code,
                    name=self.voice_name,
                )
                
                # Slower rate for ringing effect
                audio_config = texttospeech.AudioConfig(
                    audio_encoding=texttospeech.AudioEncoding.LINEAR16,
                    sample_rate_hertz=24000,
                    speaking_rate=0.85,  # Slower for dramatic effect
                )
                
                response = self._client.synthesize_speech(
                    input=input_text,
                    voice=voice,
                    audio_config=audio_config
                )
                
                self._ringing_audio = response.audio_content
                logger.debug("Generated ringing tone")
                
        except Exception as e:
            logger.warning(f"Failed to generate ringing tone: {e}")
    
    def get_audio(self, key: str) -> Optional[bytes]:
        """
        Get preloaded system audio.
        
        Args:
            key: Audio key (connecting, connection_error, goodbye, hold)
            
        Returns:
            LINEAR16 audio bytes at 24kHz, or None
        """
        return self._audio_cache.get(key)
    
    def get_ringing_audio(self) -> Optional[bytes]:
        """Get the ringing/connecting tone."""
        return self._ringing_audio
    
    def get_connection_error_audio(self) -> Optional[bytes]:
        """Get the 'unable to connect' audio."""
        return self._audio_cache.get("connection_error")
    
    def get_connecting_audio(self) -> Optional[bytes]:
        """Get the 'please hold while connecting' audio."""
        return self._audio_cache.get("connecting")


# Singleton instance
_system_audio: Optional[SystemAudioService] = None


def get_system_audio() -> SystemAudioService:
    """Get or create the system audio service singleton."""
    global _system_audio
    if _system_audio is None:
        _system_audio = SystemAudioService()
    return _system_audio


def preload_system_audio():
    """Initialize and preload all system audio assets."""
    service = get_system_audio()
    service.preload_all()
    return service
