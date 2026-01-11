import asyncio
import io
import wave
from typing import Optional

from deepgram import DeepgramClient, DeepgramClientOptions

from backend.config import get_settings, Settings


class DeepgramSTTService:
    """Thin wrapper around Deepgram's prerecorded endpoint for short chunks."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        options = DeepgramClientOptions(api_key=self.settings.DEEPGRAM_API_KEY)
        self._client = DeepgramClient(options)

    async def transcribe_pcm(self, pcm_bytes: bytes) -> Optional[str]:
        if not pcm_bytes:
            return None

        wav_bytes = self._pcm16_to_wav(pcm_bytes, self.settings.SAMPLE_RATE)
        payload = {"buffer": wav_bytes, "mimetype": "audio/wav"}
        request_options = {
            "model": self.settings.DEEPGRAM_MODEL,
            "language": self.settings.DEEPGRAM_LANGUAGE,
            "smart_format": True,
            "punctuate": True,
        }

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._client.listen.prerecorded.v("1").transcribe(
                payload, request_options
            ),
        )

        try:
            channels = response.get("results", {}).get("channels", [])
            if not channels:
                return None
            alternatives = channels[0].get("alternatives", [])
            if not alternatives:
                return None
            transcript = alternatives[0].get("transcript", "").strip()
            return transcript or None
        except AttributeError:
            return None

    @staticmethod
    def _pcm16_to_wav(pcm_bytes: bytes, sample_rate: int) -> bytes:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm_bytes)
        buffer.seek(0)
        return buffer.read()
*** End of File