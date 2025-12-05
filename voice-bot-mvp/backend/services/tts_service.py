import asyncio
import io
from functools import lru_cache
from typing import Optional, AsyncIterator
import re
import numpy as np

from TTS.api import TTS
import soundfile as sf
from config import get_settings

class TTSService:
    """
    Text-to-Speech service using Coqui TTS with true streaming support.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.tts: Optional[TTS] = None
        self._initialized = False
    
    async def initialize(self):
        """Initializes and loads the Coqui TTS model."""
        if self._initialized:
            return
        
        print(f"🔄 Initializing TTS Service (Coqui model: {self.settings.TTS_MODEL})...")
        loop = asyncio.get_running_loop()
        def _load_model():
            return TTS(model_name=self.settings.TTS_MODEL, progress_bar=False)
        self.tts = await loop.run_in_executor(None, _load_model)
        self._initialized = True
        print("✅ TTS Service initialized.")

    async def synthesize_streaming(self, text_stream: AsyncIterator[str]) -> AsyncIterator[bytes]:
        """
        Synthesizes text into speech audio, streaming the output.
        This implementation processes sentences as they are received from the text stream.
        """
        if not self._initialized:
            await self.initialize()

        loop = asyncio.get_running_loop()
        
        async for text in text_stream:
            print(f"🔊 TTS Streaming Synthesis for: '{text}'")
            try:
                def _synthesize_stream():
                    # This is a generator function from the TTS library
                    return self.tts.tts_stream(
                        text=text,
                        speaker=self.tts.speakers[0],
                        language=self.tts.languages[0],
                        speed=1.0 
                    )
                
                # The TTS library's stream is blocking, so we run it in an executor
                audio_chunks_generator = await loop.run_in_executor(None, _synthesize_stream)

                for chunk in audio_chunks_generator:
                    # The chunk from the library is a list or numpy array, convert to bytes
                    audio_array = np.array(chunk, dtype=np.float32)
                    with io.BytesIO() as wav_io:
                        sf.write(wav_io, audio_array, self.settings.SAMPLE_RATE, format='WAV')
                        yield wav_io.getvalue()

            except Exception as e:
                print(f"❌ TTS Error during streaming for text '{text}': {e}")
                # Continue to the next text chunk
                continue

    async def synthesize(self, text: str) -> bytes:
        """
        Synthesizes a single block of text to audio. This is a non-streaming
        convenience wrapper around the streaming method.
        """
        async def text_generator():
            yield text
        
        audio_chunks = []
        async for chunk in self.synthesize_streaming(text_generator()):
            audio_chunks.append(chunk)
            
        return b"".join(audio_chunks)

@lru_cache()
def get_tts_service() -> TTSService:
    """Get a cached singleton instance of the TTSService."""
    return TTSService()
