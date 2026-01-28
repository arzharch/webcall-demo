"""
Speech-to-Text Service using Deepgram.
Handles real-time transcription with VAD events.
"""
import asyncio
import logging
from typing import Callable, Optional
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
)
from config import settings

logger = logging.getLogger(__name__)


class STTService:
    """
    Deepgram Speech-to-Text service for real-time transcription.
    Designed for WebSocket audio streaming from browser.
    """
    
    def __init__(
        self,
        on_speech_start: Callable[[], None],
        on_final_transcript: Callable[[str], None],
        on_interim_transcript: Optional[Callable[[str], None]] = None,
        language: str = "en-IN",
        model: str = "nova-2",
    ):
        """
        Initialize STT service.
        
        Args:
            on_speech_start: Callback when VAD detects speech started
            on_final_transcript: Callback with final transcript text
            on_interim_transcript: Optional callback for interim results
            language: Language code for transcription
            model: Deepgram model to use
        """
        self.on_speech_start = on_speech_start
        self.on_final_transcript = on_final_transcript
        self.on_interim_transcript = on_interim_transcript
        self.language = language
        self.model = model
        
        self._client: Optional[DeepgramClient] = None
        self._connection = None
        self._is_connected = False
        
    async def connect(self, max_retries: int = 3) -> bool:
        """
        Establish connection to Deepgram with retry logic.
        Returns True if successful.
        """
        for attempt in range(max_retries):
            try:
                logger.info(f"Deepgram connection attempt {attempt + 1}/{max_retries}...")
                
                config = DeepgramClientOptions(options={"keepalive": "true"})
                self._client = DeepgramClient(settings.DEEPGRAM_API_KEY, config)
                self._connection = self._client.listen.asynclive.v("1")
                
                # Set up event handlers
                self._connection.on(LiveTranscriptionEvents.Transcript, self._on_message)
                self._connection.on(LiveTranscriptionEvents.SpeechStarted, self._on_speech_started)
                self._connection.on(LiveTranscriptionEvents.Error, self._on_error)
                
                # Configure live options
                options = LiveOptions(
                    model=self.model,
                    language=self.language,
                    smart_format=True,
                    encoding="linear16",
                    channels=1,
                    sample_rate=16000,  # Browser sends 16kHz
                    interim_results=True,
                    utterance_end_ms="1000",
                    vad_events=True,  # Enable VAD events for barge-in
                )
                
                # Use asyncio.wait_for with a shorter timeout for faster fallback
                await asyncio.wait_for(
                    self._connection.start(options),
                    timeout=3.0  # Reduced validation timeout (was 15.0)
                )
                self._is_connected = True
                logger.info("✅ Deepgram STT connected successfully")
                return True
                
            except asyncio.TimeoutError:
                logger.warning(f"Deepgram connection attempt {attempt + 1} timed out")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)  # Brief pause before retry
                continue
                    
            except Exception as e:
                import traceback
                logger.error(f"Failed to connect to Deepgram (attempt {attempt + 1}): {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                logger.error(f"API Key present: {bool(settings.DEEPGRAM_API_KEY)}, Length: {len(settings.DEEPGRAM_API_KEY) if settings.DEEPGRAM_API_KEY else 0}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                continue
        
        logger.error("❌ All Deepgram connection attempts failed. Check your API key and network.")
        self._is_connected = False
        return False
    
    async def send_audio(self, audio_data: bytes):
        """
        Send audio chunk to Deepgram for transcription.
        Audio should be LINEAR16, 16kHz, mono.
        
        Args:
            audio_data: Raw audio bytes
        """
        if not self._is_connected or not self._connection:
            logger.warning("Cannot send audio - not connected")
            return
        
        try:
            await self._connection.send(audio_data)
        except Exception as e:
            logger.error(f"Error sending audio to Deepgram: {e}")
            
    async def disconnect(self):
        """Close the Deepgram connection."""
        if self._connection:
            try:
                await self._connection.finish()
            except Exception as e:
                logger.debug(f"Error finishing Deepgram connection: {e}")
        self._is_connected = False
        logger.info("Deepgram STT disconnected")
        
    @property
    def is_connected(self) -> bool:
        return self._is_connected
    
    # ==================== Event Handlers ====================
    
    async def _on_message(self, *args, **kwargs):
        """Handle transcript messages from Deepgram."""
        try:
            # The result is typically the second argument
            result = args[1] if len(args) > 1 else kwargs.get('result')
            if not result:
                return
            
            transcript = result.channel.alternatives[0].transcript
            is_final = result.speech_final
            
            if transcript:
                if is_final:
                    logger.debug(f"Final transcript: {transcript}")
                    self.on_final_transcript(transcript)
                elif self.on_interim_transcript:
                    self.on_interim_transcript(transcript)
                    
        except Exception as e:
            logger.error(f"Error processing Deepgram message: {e}")
    
    async def _on_speech_started(self, *args, **kwargs):
        """Handle VAD speech started event."""
        logger.debug("VAD: Speech started")
        self.on_speech_start()
        
    async def _on_error(self, *args, **kwargs):
        """Handle Deepgram errors."""
        error = args[1] if len(args) > 1 else kwargs.get('error', 'Unknown error')
        logger.error(f"Deepgram error: {error}")


class STTServiceSync:
    """
    Synchronous wrapper for STT service.
    Runs in a dedicated thread with its own event loop.
    For compatibility with existing sync code patterns.
    """
    
    def __init__(
        self,
        on_speech_start: Callable[[], None],
        on_final_transcript: Callable[[str], None],
    ):
        self._stt = STTService(on_speech_start, on_final_transcript)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._running = False
        
    def start(self) -> bool:
        """Start the STT service."""
        self._running = True
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        
        # Connect synchronously
        connected = self._loop.run_until_complete(self._stt.connect())
        return connected
    
    def send_audio(self, audio_data: bytes):
        """Send audio synchronously."""
        if self._loop and self._running:
            asyncio.run_coroutine_threadsafe(
                self._stt.send_audio(audio_data),
                self._loop
            )
            
    def stop(self):
        """Stop the STT service."""
        self._running = False
        if self._loop:
            self._loop.run_until_complete(self._stt.disconnect())
            self._loop.close()
