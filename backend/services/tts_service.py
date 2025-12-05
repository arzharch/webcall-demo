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
        Synthesizes text into speech audio, streaming the output. This version
        buffers text chunks into sentences before synthesizing, which produces
        more natural-sounding speech.
        """
        if not self._initialized:
            await self.initialize()

        loop = asyncio.get_running_loop()
        sentence_buffer = ""
        sentence_delimiters = re.compile(r'(?<=[.?!])\s*')

        async for text_chunk in text_stream:
            sentence_buffer += text_chunk
            
            # Split the buffer into sentences
            sentences = sentence_delimiters.split(sentence_buffer)
            
            # The last part of the split might be an incomplete sentence,
            # so we keep it in the buffer.
            sentence_buffer = sentences.pop(-1)
            
            for sentence in sentences:
                if not sentence: continue

                print(f"🔊 TTS Synthesizing sentence: '{sentence}'")
                try:
                    def _synthesize_stream():
                        return self.tts.tts_stream(
                            text=sentence,
                            speaker=self.tts.speakers[0],
                            language=self.tts.languages[0],
                            speed=1.0 
                        )
                    
                    audio_chunks_generator = await loop.run_in_executor(None, _synthesize_stream)
                    for chunk in audio_chunks_generator:
                        audio_array = np.frombuffer(np.array(chunk), dtype=np.float32)
                        with io.BytesIO() as wav_io:
                            sf.write(wav_io, audio_array, self.settings.SAMPLE_RATE, format='WAV')
                            yield wav_io.getvalue()
                except Exception as e:
                    print(f"❌ TTS Error during streaming for sentence '{sentence}': {e}")
                    continue
        
        # Synthesize any remaining text in the buffer after the stream ends
        if sentence_buffer.strip():
            print(f"🔊 TTS Synthesizing final buffer: '{sentence_buffer}'")
            try:
                def _synthesize_stream():
                    return self.tts.tts_stream(
                        text=sentence_buffer,
                        speaker=self.tts.speakers[0],
                        language=self.tts.languages[0],
                        speed=1.0
                    )
                audio_chunks_generator = await loop.run_in_executor(None, _synthesize_stream)
                for chunk in audio_chunks_generator:
                    audio_array = np.frombuffer(np.array(chunk), dtype=np.float32)
                    with io.BytesIO() as wav_io:
                        sf.write(wav_io, audio_array, self.settings.SAMPLE_RATE, format='WAV')
                        yield wav_io.getvalue()
            except Exception as e:
                print(f"❌ TTS Error on final buffer: {e}")


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