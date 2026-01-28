"""
Google Cloud Speech-to-Text Service.
Fallback STT provider when Deepgram is unavailable.
"""
import asyncio
import logging
import queue
import threading
from typing import Callable, Optional
from google.cloud import speech

from config import settings

logger = logging.getLogger(__name__)


class GoogleSTTService:
    """
    Google Cloud Speech-to-Text service for real-time transcription.
    Used as a fallback when Deepgram is unavailable.
    
    Uses streaming recognition with interim results.
    """
    
    def __init__(
        self,
        on_speech_start: Callable[[], None],
        on_final_transcript: Callable[[str], None],
        on_interim_transcript: Optional[Callable[[str], None]] = None,
        language: str = "en-IN",
        sample_rate: int = 16000,
    ):
        """
        Initialize Google STT service.
        
        Args:
            on_speech_start: Callback when speech is detected
            on_final_transcript: Callback with final transcript text
            on_interim_transcript: Optional callback for interim results
            language: Language code for transcription
            sample_rate: Audio sample rate in Hz
        """
        self.on_speech_start = on_speech_start
        self.on_final_transcript = on_final_transcript
        self.on_interim_transcript = on_interim_transcript
        self.language = language
        self.sample_rate = sample_rate
        
        self._client: Optional[speech.SpeechClient] = None
        self._is_connected = False
        self._audio_queue: queue.Queue = queue.Queue()
        self._streaming_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._speech_started = False
        
    async def connect(self, timeout: float = 10.0) -> bool:
        """
        Initialize Google STT client and start streaming.
        Returns True if successful.
        """
        try:
            logger.info("Connecting to Google Cloud STT...")
            
            # Initialize client
            self._client = speech.SpeechClient()
            
            # Clear any existing state
            self._stop_event.clear()
            self._audio_queue = queue.Queue()
            self._speech_started = False
            
            # Start streaming thread
            self._streaming_thread = threading.Thread(
                target=self._run_streaming,
                daemon=True
            )
            self._streaming_thread.start()
            
            # Wait briefly to confirm it started
            await asyncio.sleep(0.1)
            
            if self._streaming_thread.is_alive():
                self._is_connected = True
                logger.info("✅ Google Cloud STT connected successfully")
                return True
            else:
                logger.error("Google STT streaming thread failed to start")
                return False
                
        except Exception as e:
            logger.error(f"Failed to connect to Google Cloud STT: {e}")
            self._is_connected = False
            return False
    
    def _run_streaming(self):
        """Background thread running the streaming recognition."""
        try:
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=self.sample_rate,
                language_code=self.language,
                enable_automatic_punctuation=True,
                model="latest_short",  # Optimized for short utterances
            )
            
            streaming_config = speech.StreamingRecognitionConfig(
                config=config,
                interim_results=True,
                single_utterance=False,
            )
            
            # Generator for audio chunks
            def request_generator():
                # First request must be config only
                yield speech.StreamingRecognizeRequest(
                    streaming_config=streaming_config
                )
                
                # Subsequent requests are audio
                while not self._stop_event.is_set():
                    try:
                        audio_chunk = self._audio_queue.get(timeout=0.1)
                        if audio_chunk is None:  # Poison pill
                            break
                        yield speech.StreamingRecognizeRequest(
                            audio_content=audio_chunk
                        )
                    except queue.Empty:
                        continue
            
            # Start streaming
            responses = self._client.streaming_recognize(
                requests=request_generator()
            )
            
            # Process responses
            for response in responses:
                if self._stop_event.is_set():
                    break
                    
                for result in response.results:
                    if not result.alternatives:
                        continue
                    
                    transcript = result.alternatives[0].transcript
                    
                    # Detect speech start
                    if transcript and not self._speech_started:
                        self._speech_started = True
                        self.on_speech_start()
                    
                    if result.is_final:
                        logger.debug(f"Google STT final: {transcript}")
                        self.on_final_transcript(transcript)
                        self._speech_started = False  # Reset for next utterance
                    elif self.on_interim_transcript and transcript:
                        self.on_interim_transcript(transcript)
                        
        except Exception as e:
            if not self._stop_event.is_set():
                logger.error(f"Google STT streaming error: {e}")
        finally:
            logger.info("Google STT streaming thread ended")
    
    async def send_audio(self, audio_data: bytes):
        """
        Send audio chunk for transcription.
        Audio should be LINEAR16, 16kHz, mono.
        
        Args:
            audio_data: Raw audio bytes
        """
        if not self._is_connected:
            return
        
        try:
            self._audio_queue.put_nowait(audio_data)
        except queue.Full:
            logger.warning("Google STT audio queue full, dropping chunk")
            
    async def disconnect(self):
        """Close the Google STT connection."""
        self._stop_event.set()
        self._audio_queue.put(None)  # Poison pill
        
        if self._streaming_thread and self._streaming_thread.is_alive():
            self._streaming_thread.join(timeout=2.0)
        
        self._is_connected = False
        logger.info("Google Cloud STT disconnected")
        
    @property
    def is_connected(self) -> bool:
        return self._is_connected


class STTFallbackService:
    """
    STT service with automatic fallback.
    Tries Deepgram first, falls back to Google Cloud STT.
    """
    
    def __init__(
        self,
        on_speech_start: Callable[[], None],
        on_final_transcript: Callable[[str], None],
        on_interim_transcript: Optional[Callable[[str], None]] = None,
    ):
        self.on_speech_start = on_speech_start
        self.on_final_transcript = on_final_transcript
        self.on_interim_transcript = on_interim_transcript
        
        self._active_service = None
        self._provider = None  # "deepgram" or "google"
        
    async def connect(self) -> tuple[bool, str]:
        """
        Connect to STT service with fallback logic.
        Returns (success, provider_name).
        """
        from services.stt_service import STTService
        
        # Try Deepgram first
        logger.info("Attempting Deepgram STT connection...")
        deepgram_stt = STTService(
            on_speech_start=self.on_speech_start,
            on_final_transcript=self.on_final_transcript,
            on_interim_transcript=self.on_interim_transcript,
        )
        
        connected = await deepgram_stt.connect(max_retries=2)
        if connected:
            self._active_service = deepgram_stt
            self._provider = "deepgram"
            logger.info("✅ Using Deepgram STT")
            return True, "deepgram"
        
        # Fallback to Google Cloud STT
        logger.warning("Deepgram failed, falling back to Google Cloud STT...")
        google_stt = GoogleSTTService(
            on_speech_start=self.on_speech_start,
            on_final_transcript=self.on_final_transcript,
            on_interim_transcript=self.on_interim_transcript,
        )
        
        connected = await google_stt.connect()
        if connected:
            self._active_service = google_stt
            self._provider = "google"
            logger.info("✅ Using Google Cloud STT (fallback)")
            return True, "google"
        
        logger.error("❌ All STT providers failed")
        return False, "none"
    
    async def send_audio(self, audio_data: bytes):
        """Send audio to active STT service."""
        if self._active_service:
            await self._active_service.send_audio(audio_data)
    
    async def disconnect(self):
        """Disconnect active STT service."""
        if self._active_service:
            await self._active_service.disconnect()
    
    @property
    def is_connected(self) -> bool:
        return self._active_service.is_connected if self._active_service else False
    
    @property
    def provider(self) -> Optional[str]:
        return self._provider
