"""
Voice Session Manager for WebSocket-based voice calls.
Handles the full lifecycle of a voice call session.
"""
import asyncio
import hashlib
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional, Deque, Any

from fastapi import WebSocket

from agent.agent import BellaAgent
from agent.state import SessionState
from services.stt_service import STTService
from services.tts_service import TTSService, TTSStreamProcessor
from infra import (
    CircuitBreakerManager,
    CircuitState,
    ConcurrencyLimiter,
    trace_span,
    track_llm_cost,
)
from infra.config import config
import database as db

logger = logging.getLogger(__name__)


class CallStatus(str, Enum):
    """Voice call status."""
    CONNECTING = "connecting"
    ACTIVE = "active"
    PROCESSING = "processing"  # Agent is thinking
    SPEAKING = "speaking"      # TTS playing
    PAUSED = "paused"
    ENDED = "ended"
    ERROR = "error"


@dataclass
class VoiceSessionMetrics:
    """Metrics tracked during a voice session."""
    turn_count: int = 0
    interruption_count: int = 0
    error_count: int = 0
    total_tts_ms: int = 0
    total_stt_ms: int = 0
    total_llm_ms: int = 0
    response_latencies: list = field(default_factory=list)
    
    def avg_latency_ms(self) -> float:
        if not self.response_latencies:
            return 0.0
        return sum(self.response_latencies) / len(self.response_latencies)


class VoiceSession:
    """
    Manages a single voice call session over WebSocket.
    
    Responsibilities:
    - Receive audio from browser via WebSocket
    - Send audio to Deepgram for STT
    - Process transcripts through agent
    - Stream TTS audio back to browser
    - Handle interruptions (barge-in)
    - Track metrics and persist to database
    """
    
    # Buffer settings
    DEBOUNCE_DELAY = 1.0  # Seconds to wait for continuation
    MAX_BUFFER_SIZE = 10  # Max transcript chunks before force processing
    MAX_ACCUMULATION_TIME = 8.0  # Max seconds to accumulate
    
    def __init__(
        self,
        websocket: WebSocket,
        caller_name: str,
        session_id: Optional[str] = None,
        phone_number: Optional[str] = None,
    ):
        """
        Initialize a voice session.
        
        Args:
            websocket: FastAPI WebSocket connection
            caller_name: Name of the caller
            session_id: Optional session ID (generated if not provided)
            phone_number: Optional phone number for analytics
        """
        self.websocket = websocket
        self.caller_name = caller_name
        self.session_id = session_id or str(uuid.uuid4())
        self.phone_number = phone_number
        
        # Status
        self.status = CallStatus.CONNECTING
        self.started_at = time.time()
        self.ended_at: Optional[float] = None
        
        # Agent
        self.agent_state = SessionState(caller_name=caller_name)
        self.agent = BellaAgent(self.agent_state)
        
        # Services (initialized in start())
        self.stt: Optional[STTService] = None
        self.tts: Optional[TTSService] = None
        self.tts_processor: Optional[TTSStreamProcessor] = None
        
        # Infrastructure
        self.circuit_breakers = CircuitBreakerManager()
        self.concurrency_limiter = ConcurrencyLimiter(
            "voice_calls", 
            config.concurrency.max_concurrent_llm_calls
        )
        
        # Transcript buffering (for debounce)
        self._input_buffer: list[str] = []
        self._buffer_start_time: Optional[float] = None
        self._debounce_task: Optional[asyncio.Task] = None
        self._agent_task: Optional[asyncio.Task] = None
        self._current_processing_text: Optional[str] = None
        
        # Response loop detection
        self._response_history: Deque[str] = deque(maxlen=3)
        
        # Metrics
        self.metrics = VoiceSessionMetrics()
        
        # State flags
        self._is_speaking = False
        self._stop_playback = False
        
        logger.info(f"VoiceSession created: {self.session_id} for {caller_name}")
        
    async def start(self):
        """
        Initialize and start the voice session.
        Call after WebSocket connection is established.
        """
        try:
            # Create database record
            db.create_call(
                call_id=self.session_id,
                caller_name=self.caller_name,
                phone_number=self.phone_number
            )
            
            # Initialize TTS
            self.tts = TTSService(
                circuit_breakers=self.circuit_breakers
            )
            self.tts.preload_filler_audio()
            self.tts_processor = TTSStreamProcessor(self.tts)
            
            # Initialize STT
            self.stt = STTService(
                on_speech_start=self._on_speech_start,
                on_final_transcript=self._on_transcript,
            )
            
            # Connect to Deepgram
            connected = await self.stt.connect()
            if not connected:
                self.status = CallStatus.ERROR
                await self._send_status("error", "Failed to connect to speech service")
                return False
            
            self.status = CallStatus.ACTIVE
            await self._send_status("connected", "Voice session ready")
            
            # Send greeting
            await self._speak("Hello! Welcome to Bella Cucina. How may I help you today?")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start session {self.session_id}: {e}")
            self.status = CallStatus.ERROR
            return False
    
    async def handle_audio(self, audio_data: bytes):
        """
        Handle incoming audio from browser.
        
        Args:
            audio_data: Raw audio bytes (LINEAR16, 16kHz, mono)
        """
        if self.status not in (CallStatus.ACTIVE, CallStatus.PROCESSING, CallStatus.SPEAKING):
            return
        
        if self.stt and self.stt.is_connected:
            await self.stt.send_audio(audio_data)
    
    async def end(self, reason: str = "user_ended"):
        """
        End the voice session gracefully.
        
        Args:
            reason: Reason for ending (user_ended, transferred, error, timeout)
        """
        self.status = CallStatus.ENDED
        self.ended_at = time.time()
        
        # Cancel any pending tasks
        if self._debounce_task and not self._debounce_task.done():
            self._debounce_task.cancel()
        if self._agent_task and not self._agent_task.done():
            self._agent_task.cancel()
        
        # Disconnect STT
        if self.stt:
            await self.stt.disconnect()
        
        # Save call analytics to database
        try:
            result = db.end_call(
                call_id=self.session_id,
                end_reason=reason,
                turn_count=self.metrics.turn_count,
                interruption_count=self.metrics.interruption_count,
                error_count=self.metrics.error_count,
                total_tts_ms=self.metrics.total_tts_ms,
                total_stt_ms=self.metrics.total_stt_ms,
                total_llm_ms=self.metrics.total_llm_ms,
            )
            logger.info(f"Session {self.session_id} ended: {result}")
        except Exception as e:
            logger.error(f"Failed to save call analytics: {e}")
        
        await self._send_status("ended", reason)
    
    # ==================== STT Callbacks ====================
    
    def _on_speech_start(self):
        """Called when VAD detects speech started (barge-in)."""
        if not self._is_speaking:
            return
        
        logger.info(f"[{self.session_id}] Interruption detected")
        self.metrics.interruption_count += 1
        
        # Stop current playback
        self._stop_playback = True
        
        # Cancel agent task if running
        if self._agent_task and not self._agent_task.done():
            self._agent_task.cancel()
            
            # Rescue context - keep current processing text in buffer
            if self._current_processing_text:
                self._input_buffer.insert(0, self._current_processing_text)
                self._current_processing_text = None
        
        # Cancel debounce timer
        if self._debounce_task and not self._debounce_task.done():
            self._debounce_task.cancel()
    
    def _on_transcript(self, text: str):
        """Called when final transcript is received."""
        if not text.strip():
            return
        
        # Schedule processing in the event loop
        asyncio.create_task(self._handle_transcript(text))
    
    async def _handle_transcript(self, text: str):
        """Process a transcript segment."""
        # Track buffer start time
        if not self._input_buffer:
            self._buffer_start_time = time.time()
        
        self._input_buffer.append(text)
        
        # Check if we should force processing
        should_force = (
            len(self._input_buffer) >= self.MAX_BUFFER_SIZE or
            (self._buffer_start_time and 
             time.time() - self._buffer_start_time > self.MAX_ACCUMULATION_TIME)
        )
        
        if should_force:
            logger.debug(f"Force processing buffer (size: {len(self._input_buffer)})")
            if self._debounce_task:
                self._debounce_task.cancel()
            await self._process_buffer()
            return
        
        # Reschedule debounce
        if self._debounce_task and not self._debounce_task.done():
            self._debounce_task.cancel()
        
        self._debounce_task = asyncio.create_task(self._debounce_and_process())
    
    async def _debounce_and_process(self):
        """Wait for debounce delay then process buffer."""
        try:
            await asyncio.sleep(self.DEBOUNCE_DELAY)
            await self._process_buffer()
        except asyncio.CancelledError:
            pass
    
    async def _process_buffer(self):
        """Process accumulated transcript buffer."""
        if not self._input_buffer:
            return
        
        # Stop any current playback
        self._stop_playback = True
        
        full_text = " ".join(self._input_buffer)
        self._input_buffer = []
        self._buffer_start_time = None
        
        # Send user message to frontend
        await self._send_message("user", full_text)
        
        # Save to database
        db.add_transcript(
            call_id=self.session_id,
            turn_number=self.metrics.turn_count,
            role="user",
            content=full_text,
        )
        
        # Store for potential rescue on interruption
        self._current_processing_text = full_text
        
        # Process through agent
        self._agent_task = asyncio.create_task(self._run_agent(full_text))
    
    async def _run_agent(self, text: str):
        """Process user input through the agent."""
        turn_start = time.time()
        self.status = CallStatus.PROCESSING
        self.metrics.turn_count += 1
        
        try:
            async with self.concurrency_limiter:
                # Check LLM circuit breaker
                llm_breaker = self.circuit_breakers.get_breaker("llm")
                if llm_breaker.state == CircuitState.OPEN:
                    logger.error("LLM circuit breaker OPEN")
                    self.metrics.error_count += 1
                    await self._speak("I'm having trouble connecting. Please try again in a moment.")
                    return
                
                full_response = ""
                current_sentence = ""
                
                llm_start = time.time()
                
                with trace_span("llm_agent_call", {"input_length": len(text)}) as span:
                    try:
                        async for chunk in self.agent.orchestrator.process_message(
                            self.agent.state, text
                        ):
                            # Handle error tokens
                            if "Invalid Format" in str(chunk) or "Exception" in str(chunk):
                                logger.warning(f"Suppressed error: {chunk}")
                                continue
                            
                            current_sentence += chunk
                            
                            # Check for sentence completion
                            if any(p in chunk for p in ['.', '?', '!']):
                                sentence = current_sentence.strip()
                                if sentence:
                                    # Loop detection
                                    response_hash = hashlib.md5(sentence.encode()).hexdigest()
                                    if list(self._response_history).count(response_hash) >= 2:
                                        logger.error("Loop detected - transferring")
                                        await self._speak(
                                            "I apologize, I seem to be having trouble. "
                                            "Let me transfer you to someone who can help."
                                        )
                                        await self.end("transferred")
                                        return
                                    
                                    self._response_history.append(response_hash)
                                    
                                    # Speak sentence immediately
                                    await self._speak(sentence)
                                    full_response += sentence + " "
                                    
                                current_sentence = ""
                        
                        # Flush remaining
                        if current_sentence.strip():
                            await self._speak(current_sentence)
                            full_response += current_sentence
                        
                        llm_breaker.record_success()
                        
                    except asyncio.TimeoutError:
                        logger.error("LLM timeout")
                        llm_breaker.record_failure()
                        self.metrics.error_count += 1
                        await self._speak("I'm taking too long. Could you repeat that?")
                        return
                    except Exception as e:
                        logger.error(f"LLM error: {e}")
                        llm_breaker.record_failure()
                        self.metrics.error_count += 1
                        raise
                    
                    llm_ms = int((time.time() - llm_start) * 1000)
                    self.metrics.total_llm_ms += llm_ms
                    span.set_attribute("duration_ms", llm_ms)
                    
                    track_llm_cost(
                        session_id=self.session_id,
                        model="gpt-3.5-turbo",
                        operation="agent_response",
                        input_messages=[{"content": text}],
                        output_text=full_response,
                        duration_ms=llm_ms,
                    )
                
                # Clear processing text on success
                self._current_processing_text = None
                
                final_response = full_response.strip()
                if not final_response:
                    final_response = "I'm sorry, I didn't catch that. Could you repeat?"
                    await self._speak(final_response)
                
                # Save assistant response
                latency_ms = int((time.time() - turn_start) * 1000)
                self.metrics.response_latencies.append(latency_ms)
                
                db.add_transcript(
                    call_id=self.session_id,
                    turn_number=self.metrics.turn_count,
                    role="assistant",
                    content=final_response,
                    latency_ms=latency_ms,
                )
                
                await self._send_message("assistant", final_response)
                
                logger.info(f"Turn {self.metrics.turn_count} complete: {latency_ms}ms")
                
        except asyncio.CancelledError:
            logger.info("Agent task cancelled (interrupted)")
        except Exception as e:
            logger.error(f"Agent error: {e}")
            self.metrics.error_count += 1
        finally:
            self.status = CallStatus.ACTIVE
    
    async def _speak(self, text: str):
        """Synthesize and send TTS audio."""
        if not text.strip():
            return
        
        self._stop_playback = False
        self.status = CallStatus.SPEAKING
        self._is_speaking = True
        
        try:
            tts_start = time.time()
            audio = await self.tts.synthesize(text)
            tts_ms = int((time.time() - tts_start) * 1000)
            self.metrics.total_tts_ms += tts_ms
            
            if audio and not self._stop_playback:
                await self._send_audio(audio)
                
        except Exception as e:
            logger.error(f"TTS error: {e}")
        finally:
            self._is_speaking = False
            if self.status == CallStatus.SPEAKING:
                self.status = CallStatus.ACTIVE
    
    # ==================== WebSocket Helpers ====================
    
    async def _send_audio(self, audio_data: bytes):
        """Send audio data to browser via WebSocket."""
        try:
            await self.websocket.send_bytes(audio_data)
        except Exception as e:
            logger.error(f"Failed to send audio: {e}")
    
    async def _send_message(self, role: str, content: str):
        """Send a transcript message to browser."""
        try:
            await self.websocket.send_json({
                "type": "transcript",
                "role": role,
                "content": content,
                "timestamp": time.time(),
            })
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
    
    async def _send_status(self, status: str, message: str = ""):
        """Send status update to browser."""
        try:
            await self.websocket.send_json({
                "type": "status",
                "status": status,
                "message": message,
                "session_id": self.session_id,
            })
        except Exception as e:
            logger.error(f"Failed to send status: {e}")


class VoiceSessionManager:
    """
    Manages multiple concurrent voice sessions.
    Provides session lookup and cleanup.
    """
    
    def __init__(self, max_sessions: int = 100):
        self.max_sessions = max_sessions
        self._sessions: dict[str, VoiceSession] = {}
        self._lock = asyncio.Lock()
        
    async def create_session(
        self,
        websocket: WebSocket,
        caller_name: str,
        phone_number: Optional[str] = None,
    ) -> VoiceSession:
        """Create and register a new voice session."""
        async with self._lock:
            if len(self._sessions) >= self.max_sessions:
                # Clean up ended sessions
                await self._cleanup_ended()
                
                if len(self._sessions) >= self.max_sessions:
                    raise RuntimeError("Maximum concurrent sessions reached")
            
            session = VoiceSession(
                websocket=websocket,
                caller_name=caller_name,
                phone_number=phone_number,
            )
            self._sessions[session.session_id] = session
            return session
    
    def get_session(self, session_id: str) -> Optional[VoiceSession]:
        """Get a session by ID."""
        return self._sessions.get(session_id)
    
    async def end_session(self, session_id: str, reason: str = "user_ended"):
        """End and remove a session."""
        async with self._lock:
            session = self._sessions.pop(session_id, None)
            if session:
                await session.end(reason)
    
    async def _cleanup_ended(self):
        """Remove ended sessions from the registry."""
        ended_ids = [
            sid for sid, session in self._sessions.items()
            if session.status == CallStatus.ENDED
        ]
        for sid in ended_ids:
            del self._sessions[sid]
    
    @property
    def active_session_count(self) -> int:
        """Number of active sessions."""
        return sum(
            1 for s in self._sessions.values()
            if s.status not in (CallStatus.ENDED, CallStatus.ERROR)
        )
    
    def get_all_sessions(self) -> list[dict]:
        """Get summary of all sessions."""
        return [
            {
                "session_id": s.session_id,
                "caller_name": s.caller_name,
                "status": s.status.value,
                "turn_count": s.metrics.turn_count,
                "duration": time.time() - s.started_at,
            }
            for s in self._sessions.values()
        ]


# Global session manager instance
_session_manager: Optional[VoiceSessionManager] = None


def get_voice_session_manager() -> VoiceSessionManager:
    """Get the global voice session manager."""
    global _session_manager
    if _session_manager is None:
        _session_manager = VoiceSessionManager()
    return _session_manager
