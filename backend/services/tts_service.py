import asyncio
from typing import Optional

from google.cloud import texttospeech

from backend.config import Settings, get_settings
from backend.services.cost_tracker import CostTracker


class GoogleTTSService:
    """Produces LINEAR16 audio buffers suitable for WebRTC streaming."""

    def __init__(self, settings: Settings | None = None, cost_tracker: CostTracker | None = None) -> None:
        self.settings = settings or get_settings()
        self._client = texttospeech.TextToSpeechClient()
        self.cost_tracker = cost_tracker

    async def synthesize(self, text: str) -> Optional[bytes]:
        if not text:
            return None

        loop = asyncio.get_event_loop()
        audio_content = await loop.run_in_executor(None, lambda: self._synthesize_sync(text))
        if audio_content and self.cost_tracker:
            self.cost_tracker.add_tts_characters(len(text))
        return audio_content

    def _synthesize_sync(self, text: str) -> bytes:
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code=self.settings.TTS_LANGUAGE_CODE,
            name=self.settings.TTS_VOICE_NAME,
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=self.settings.SAMPLE_RATE,
            speaking_rate=self.settings.TTS_SPEAKING_RATE,
            pitch=self.settings.TTS_PITCH,
        )
        response = self._client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config,
        )
        return response.audio_content