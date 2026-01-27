import tkinter as tk
from tkinter import scrolledtext
import asyncio
import threading
import os
import queue
import time
import sys
import logging
import hashlib
import uuid
from typing import Optional, List, Deque
from collections import deque
import numpy as np
import sounddevice as sd
from google.cloud import texttospeech
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
    Microphone,
)
from dotenv import load_dotenv

# Import Agent
from agent.agent import BellaAgent
from agent.state import SessionState

# Import Production Infrastructure
from infra import (
    RedisCache,
    CircuitBreakerManager,
    CircuitState,
    ConcurrencyLimiter,
    RateLimiter,
    TimeoutManager,
    with_timeout,
    trace_span,
    track_llm_cost,
    # Phase 2 additions
    get_session_manager,
    setup_logging,
    log_context,
    PerformanceLogger,
    get_health_checker,
)
from infra.config import config
from infra.telemetry import get_telemetry

# --- Configuration ---
load_dotenv()
GOOGLE_CREDS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
SAMPLE_RATE = 16000 # Deepgram & Mic
TTS_SAMPLE_RATE = 24000 # Google Neural Voice

# Environment detection
IS_PRODUCTION = os.getenv("ENV", "development").lower() == "production"
JSON_LOGS = os.getenv("JSON_LOGS", "false").lower() == "true" or IS_PRODUCTION

if not GOOGLE_CREDS or not DEEPGRAM_API_KEY:
    raise RuntimeError("Environment variables not configured correctly.")

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_CREDS

# Configure Logging (Phase 2: Structured logging)
setup_logging(
    level=os.getenv("LOG_LEVEL", "INFO"),
    json_output=JSON_LOGS,
    service_name="bella-voice-ai",
)
logger = logging.getLogger(__name__)
perf_logger = PerformanceLogger(logger)

# ===== PRODUCTION INFRASTRUCTURE INITIALIZATION =====
# These are initialized once at module load (cold start)
redis_cache = RedisCache()
circuit_breakers = CircuitBreakerManager()
concurrency_limiter = ConcurrencyLimiter("voice_calls", config.concurrency.max_concurrent_llm_calls)
rate_limiter = RateLimiter("voice_api", config.concurrency.requests_per_minute, config.concurrency.burst_size)
timeout_manager = TimeoutManager()
telemetry = get_telemetry()
session_manager = get_session_manager()  # Phase 2: Session persistence
health_checker = get_health_checker()     # Phase 2: Health checks

# Register circuit breakers for external services
circuit_breakers.get_breaker("tts")
circuit_breakers.get_breaker("llm")
circuit_breakers.get_breaker("deepgram")

class AudioPlayer:
    """
    Handles audio playback with support for immediate cancellation.
    Uses sounddevice OutputStream.
    """
    def __init__(self):
        self._stream = None
        self._queue = queue.Queue()
        self._playback_thread = None
        self._stop_event = threading.Event()
        self.is_playing = False

    def start(self):
        self._stop_event.clear()
        self._playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
        self._playback_thread.start()

    def play_audio(self, audio_data: bytes):
        """Enqueue audio data for playback."""
        # Clear stop flag when intentionally playing new audio
        self._stop_event.clear()
        np_audio = np.frombuffer(audio_data, dtype=np.int16)
        self._queue.put(np_audio)


    def stop_playback(self):
        """Immediately stop current and pending playback."""
        self.is_playing = False
        # Set the stop flag to prevent new audio from being enqueued
        self._stop_event.set()
        
        # Clear the queue first so nothing new follows
        with self._queue.mutex:
            self._queue.queue.clear()
        
        # Stop the stream if it's running
        if self._stream and self._stream.active:
             try:
                # sd.stop() is immediate and aborts the current buffer
                self._stream.stop()
                # We need to close/nullify to force a restart in the loop
                self._stream.close() 
                self._stream = None 
             except Exception as e:
                logger.debug(f"Stream stop error (benign): {e}")

    def _playback_loop(self):
        try:
            # Run forever - stop_event only stops current playback, not the thread
            while True:
                try:
                    data = self._queue.get(timeout=0.1)
                    
                    # Check if we should skip this audio (interrupted)
                    if self._stop_event.is_set():
                        self._queue.task_done()
                        continue
                    
                    # Ensure stream exists and is active
                    if self._stream is None or not self._stream.active:
                        self._stream = sd.OutputStream(
                            samplerate=24000, 
                            channels=1, 
                            dtype='int16'
                        )
                        self._stream.start()
                    
                    self.is_playing = True
                    self._stream.write(data)
                    self.is_playing = False
                    self._queue.task_done()
                except queue.Empty:
                    # Clear stop event when queue is empty so next audio can play
                    if self._stop_event.is_set():
                        self._stop_event.clear()
                    continue
                except Exception as e:
                    # Log but keep running
                    logger.debug(f"Playback error: {e}")
            
            if self._stream:
                self._stream.stop()
                self._stream.close()
            
        except Exception as e:
            logger.error(f"Audio Output Stream Error: {e}")

class ConversationManager:
    def __init__(self, agent: BellaAgent, ui_callback, caller_name: str = "Guest"):
        self.agent = agent
        self.log_ui = ui_callback
        
        # Phase 2: Create persistent session
        self.call_session = session_manager.create_session(
            caller_name=caller_name,
            phone_number=None,  # Could be passed from telephony integration
        )
        self.session_id = self.call_session.session_id
        
        self.tts_client = texttospeech.TextToSpeechClient()
        self.player = AudioPlayer()
        self.player.start()
        
        # Audio Assets
        # Synthesize a "Let me check" filler on startup to have it ready perfectly fast
        self.filler_audio: Optional[bytes] = None
        self._preload_filler()
        
        # State
        self.loop = None
        self.input_buffer: List[str] = []
        self.buffer_start_time: Optional[float] = None  # Track when buffer accumulation started
        self.last_speech_time = 0
        self.debounce_timer: Optional[asyncio.Task] = None
        self.current_agent_task: Optional[asyncio.Task] = None
        
        # Loop Circuit Breaker
        self.response_history: Deque[str] = deque(maxlen=2)
        
        # Debounce settings
        self.DEBOUNCE_DELAY = 1.0 # Seconds to wait for "Make that two"
        self.MAX_BUFFER_SIZE = 10  # Max number of transcript chunks to accumulate
        self.MAX_ACCUMULATION_TIME = 8.0  # Max seconds to accumulate before forcing process
        
        logger.info(f"Session started: {self.session_id} for {caller_name}")
        
    def _preload_filler(self):
        """Pre-synthesize filler audio for latency masking."""
        try:
            input_text = texttospeech.SynthesisInput(text="Just a moment, let me check.")
            voice = texttospeech.VoiceSelectionParams(
                language_code="en-IN", name="en-IN-Neural2-A"
            )
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.LINEAR16,
                sample_rate_hertz=24000
            )
            response = self.tts_client.synthesize_speech(
                input=input_text, voice=voice, audio_config=audio_config
            )
            self.filler_audio = response.audio_content
            logger.info("Filler audio preloaded.")
        except Exception as e:
            logger.error(f"Failed to preload filler: {e}")

    def start_loop(self):
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_forever()
        except KeyboardInterrupt:
            pass

    def on_speech_start(self):
        """Called immediately when VAD detects speech."""
        logger.info("⚡ INTERRUPTION DETECTED")
        # Removed UI log for interruption
        
        # 1. Stop Audio
        self.player.stop_playback()
        
        # 2. Cancel Brain
        if self.current_agent_task and not self.current_agent_task.done():
            self.current_agent_task.cancel()
            self.log_ui("[System] Thinking cancelled.")
            
            # RESCUE CONTEXT: If we were processing text, put it back in buffer
            # This ensures "Book Saturday" + "Sorry Sunday" becomes one query
            if hasattr(self, 'current_processing_text') and self.current_processing_text:
                logger.info(f"Rescuing text: {self.current_processing_text}")
                # We insert it at the start of the buffer
                self.input_buffer.insert(0, self.current_processing_text)
                self.current_processing_text = None # Consumed
            
        # 3. Cancel Debouncer (if user paused briefly then continued)
        if self.debounce_timer and not self.debounce_timer.done():
            self.debounce_timer.cancel()

    def on_transcript(self, text: str):
        """Called when a transcript segment is final."""
        if not text.strip(): return
        
        # Track buffer start time
        if not self.input_buffer:
            self.buffer_start_time = time.time()
        
        # Add to buffer
        self.input_buffer.append(text)
        
        # Force processing if buffer is too large or has been accumulating too long
        should_force_process = (
            len(self.input_buffer) >= self.MAX_BUFFER_SIZE or
            (self.buffer_start_time and time.time() - self.buffer_start_time > self.MAX_ACCUMULATION_TIME)
        )
        
        if should_force_process:
            logger.info(f"Forcing buffer process (size: {len(self.input_buffer)})")
            if self.debounce_timer:
                self.debounce_timer.cancel()
            future = asyncio.run_coroutine_threadsafe(self._process_buffer(), self.loop)
            return
        
        # Reschedule processing
        if self.debounce_timer:
            self.debounce_timer.cancel()
        
        # Schedule next process
        future = asyncio.run_coroutine_threadsafe(self._schedule_debounce(), self.loop)

    async def _schedule_debounce(self):
        try:
            await asyncio.sleep(self.DEBOUNCE_DELAY)
            await self._process_buffer()
        except asyncio.CancelledError:
            pass

    async def _process_buffer(self):
        if not self.input_buffer: return
        
        # Super-Safety: Kill any lingering audio from previous turns before starting new thought
        self.player.stop_playback()
        
        full_text = " ".join(self.input_buffer)
        self.input_buffer = [] # Clear immediately so new speech fills new buffer
        self.buffer_start_time = None  # Reset accumulation timer
        
        self.log_ui(f"User: {full_text}")
        
        # Phase 2: Track user turn in persistent session
        self.call_session.add_turn("user", full_text)
        session_manager.update_session(self.call_session)
        
        # Store for potential rescue
        self.current_processing_text = full_text
        
        self.current_agent_task = asyncio.create_task(self._run_agent(full_text))

    async def _run_agent(self, text: str):
        """
        Process user input through the agent.
        PRODUCTION ENHANCEMENTS:
        - Concurrency control (semaphore-based)
        - Circuit breaker for LLM
        - Timeout protection
        - Cost tracking via telemetry
        - Session persistence (Phase 2)
        """
        turn_start_time = time.time()
        
        # Phase 2: Set logging context for this turn
        with log_context(session_id=self.session_id, turn=self.call_session.turn_count):
            try:
                # PRODUCTION: Acquire concurrency slot
                async with concurrency_limiter:
                    # Check LLM circuit breaker
                    llm_breaker = circuit_breakers.get_breaker("llm")
                    if llm_breaker.state == CircuitState.OPEN:
                        logger.error("LLM circuit breaker OPEN - using fallback")
                        self.call_session.error_count += 1
                        await self.speak_async("I'm having trouble connecting. Please try again in a moment.")
                        return
                message_buffer = ""
                current_sentence = ""
                full_response_accumulator = ""
                
                # FIX #11: TTS Pipelining - track pending synthesis tasks
                pending_tts_tasks = []
                
                # CONDITIONAL FILLER: Only play for complex/long queries
                should_play_filler = (
                    len(text) > 30 and
                    self.filler_audio and
                    not self.player.is_playing and
                    any(word in text.lower() for word in ['booking', 'reservation', 'table', 'party', 'change', 'update'])
                )
                
                if should_play_filler:
                    logger.info("Playing filler audio...")
                    self.log_ui("Bella: (thinking...) Just a moment, let me check.")
                    self.player.play_audio(self.filler_audio)
                
                # PRODUCTION: Wrap LLM call with telemetry span and timeout
                llm_start_time = time.time()
                
                with trace_span("llm_agent_call", {"input_length": len(text)}) as span:
                    try:
                        # Use timeout for total turn
                        async def process_with_timeout():
                            async for chunk in self.agent.orchestrator.process_message(self.agent.state, text):
                                # HANDLE ERROR TOKENS GRACEFULLY
                                if "Invalid Format" in str(chunk) or "Exception" in str(chunk):
                                    logger.error(f"Suppressed Error Output: {chunk}")
                                    continue
                                yield chunk
                        
                        async for chunk in process_with_timeout():
                            message_buffer += chunk
                            current_sentence += chunk
                            
                            # Simple Sentence Streaming Logic:
                            if any(punct in chunk for punct in ['.', '?', '!']):
                                to_speak = current_sentence.strip()
                                if to_speak:
                                    # FIX #15: Check for loop BEFORE speaking
                                    response_hash = hashlib.md5(to_speak.encode()).hexdigest()
                                    if self.response_history.count(response_hash) >= 2:
                                        # Phase 2: Request transfer instead of terminating
                                        self.call_session.loop_count += 1
                                        logger.error("⚠️ LOOP DETECTED. Requesting transfer to human agent.")
                                        self.log_ui("⚠️ LOOP DETECTED. Transferring to human agent.")
                                        
                                        # Create transfer ticket
                                        transfer_ticket = session_manager.request_transfer(
                                            session=self.call_session,
                                            target="human",
                                            reason="Loop detected - AI unable to resolve query"
                                        )
                                        
                                        termination_msg = "I apologize, I seem to be having trouble helping you. Let me transfer you to a human agent who can assist better."
                                        await self.speak_async(termination_msg)
                                        
                                        # Log transfer ticket for monitoring
                                        logger.info(f"Transfer ticket created: {transfer_ticket['ticket_id']}")
                                        
                                        # In production, this would trigger actual transfer
                                        # For now, we end the session gracefully
                                        session_manager.end_session(self.session_id, "transferred")
                                        await asyncio.sleep(3)
                                        sys.exit(0)
                                    
                                    logger.info(f"Streaming Sentence: {to_speak}")
                                    
                                    # FIX #11: Pipeline TTS - start synthesis immediately
                                    tts_task = asyncio.create_task(self.speak_async(to_speak))
                                    pending_tts_tasks.append(tts_task)
                                    
                                    full_response_accumulator += to_speak + " "
                                    self.response_history.append(response_hash)
                                current_sentence = ""
                        
                        # Record LLM success
                        llm_breaker.record_success()
                        
                    except asyncio.TimeoutError:
                        logger.error("LLM agent call timed out")
                        llm_breaker.record_failure()
                        self.call_session.error_count += 1
                        await self.speak_async("I'm taking too long to respond. Could you please repeat that?")
                        return
                    except Exception as e:
                        logger.error(f"LLM agent error: {e}")
                        llm_breaker.record_failure()
                        self.call_session.error_count += 1
                        raise
                    
                    # Track LLM cost
                    llm_duration_ms = (time.time() - llm_start_time) * 1000
                    span.set_attribute("duration_ms", llm_duration_ms)
                    span.set_attribute("output_length", len(full_response_accumulator))
                    
                    # Track cost (approximate - actual tokens counted in telemetry module)
                    track_llm_cost(
                        session_id=self.session_id,
                        model="gpt-3.5-turbo",
                        operation="agent_response",
                        input_messages=[{"content": text}],
                        output_text=full_response_accumulator,
                        duration_ms=llm_duration_ms,
                    )

                # Flush remaining
                if current_sentence.strip():
                    logger.info(f"Streaming Final: {current_sentence}")
                    tts_task = asyncio.create_task(self.speak_async(current_sentence))
                    pending_tts_tasks.append(tts_task)
                    full_response_accumulator += current_sentence
                
                # Wait for all TTS tasks to complete (they run in parallel with playback)
                if pending_tts_tasks:
                    await asyncio.gather(*pending_tts_tasks, return_exceptions=True)

                # Clear processing text as we are done successfully
                self.current_processing_text = None

                final_response = full_response_accumulator.strip()
                
                if not final_response:
                    final_response = "I'm sorry, I didn't quite catch that. Could you please repeat?"
                    await self.speak_async(final_response)
                
                self.log_ui(f"Bella: {final_response}")
                
                # Phase 2: Track assistant turn in persistent session
                self.call_session.add_turn("assistant", final_response)
                session_manager.update_session(self.call_session)
                
                # PRODUCTION: Log turn metrics
                turn_duration_ms = (time.time() - turn_start_time) * 1000
                logger.info(f"📊 Turn complete: {turn_duration_ms:.0f}ms total")

            except asyncio.CancelledError:
                self.call_session.interruption_count += 1
                logger.info("Agent task cancelled (user interrupted)")
            except Exception as e:
                self.call_session.error_count += 1
                logger.error(f"Agent Error: {e}")

    def speak(self, text: str):
        # Fallback synchronous speak method if needed, but we mostly use speak_async
        asyncio.run_coroutine_threadsafe(self.speak_async(text), self.loop)

    async def speak_async(self, text: str):
        """
        Converts text to speech and plays it.
        PRODUCTION ENHANCEMENTS:
        - Redis TTS caching (24hr TTL)
        - Circuit breaker for TTS service
        - Timeout protection
        - Telemetry tracing
        """
        try:
            # Note: Don't check stop_event here - let play_audio handle it
            # The stop_event is only for interrupting current playback

            clean_text = text.replace("*", "")
            if not clean_text.strip(): return
            
            start_time = time.time()
            cached = False
            audio_content = None
            
            # PRODUCTION: Check Redis TTS cache first
            with trace_span("tts_synthesis", {"text_length": len(clean_text)}) as span:
                # Try cache first (sync method, run in executor)
                audio_content = await asyncio.get_event_loop().run_in_executor(
                    None, redis_cache.get_tts_audio, clean_text, "en-IN-Neural2-A"
                )
                
                if audio_content:
                    cached = True
                    logger.info(f"TTS cache HIT: {clean_text[:30]}...")
                else:
                    # Cache miss - synthesize with circuit breaker protection
                    tts_breaker = circuit_breakers.get_breaker("tts")
                    
                    if tts_breaker.state == CircuitState.OPEN:
                        # Circuit is open - use a fallback (silent or prerecorded)
                        logger.warning("TTS circuit breaker OPEN, skipping synthesis")
                        return
                    
                    try:
                        s_input = texttospeech.SynthesisInput(text=clean_text)
                        voice = texttospeech.VoiceSelectionParams(
                            language_code="en-IN", name="en-IN-Neural2-A"
                        )
                        audio_config = texttospeech.AudioConfig(
                            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
                            speaking_rate=1.25
                        )
                        
                        # Wrap in timeout
                        async def tts_call():
                            return await self.loop.run_in_executor(
                                None, 
                                lambda: self.tts_client.synthesize_speech(
                                    input=s_input, voice=voice, audio_config=audio_config
                                )
                            )
                        
                        response = await asyncio.wait_for(
                            tts_call(),
                            timeout=config.timeouts.tts_synthesis
                        )
                        
                        audio_content = response.audio_content
                        tts_breaker.record_success()
                        
                        # Cache the result (sync method, run in executor)
                        await asyncio.get_event_loop().run_in_executor(
                            None, redis_cache.set_tts_audio, clean_text, "en-IN-Neural2-A", audio_content
                        )
                        
                    except asyncio.TimeoutError as e:
                        logger.error("TTS synthesis timed out")
                        tts_breaker.record_failure(e)
                        return
                    except Exception as e:
                        logger.error(f"TTS synthesis error: {e}")
                        tts_breaker.record_failure(e)
                        raise
                
                duration_ms = (time.time() - start_time) * 1000
                span.set_attribute("duration_ms", duration_ms)
                span.set_attribute("cached", cached)
                logger.info(f"TTS {'(cached)' if cached else '(fresh)'}: {duration_ms:.0f}ms for {len(clean_text)} chars")
            
            # Play audio - the player handles interruption internally
            if audio_content:
                self.player.play_audio(audio_content)
            
        except Exception as e:
            logger.error(f"TTS Error: {e}")

class DeepgramService:
    def __init__(self, on_speech_start, on_final_transcript):
        self.on_speech_start = on_speech_start
        self.on_final_transcript = on_final_transcript
        self.running = False
        self._thread = None

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        
    def stop(self):
        self.running = False

    def _run_loop(self):
        # Dedicated loop for Deepgram
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._transcribe())

    async def _transcribe(self):
        try:
            config = DeepgramClientOptions(options={"keepalive": "true"})
            dg_client = DeepgramClient(DEEPGRAM_API_KEY, config)
            dg_connection = dg_client.listen.asynclive.v("1")

            async def on_message(self_inner, result, **kwargs):
                # 1. Detect VAD (Speech Started) via Metadata?
                # Deepgram python SDK maps events. 
                # We can deduce speech started if we get a transient transcript with high confidence
                # or if we get an explicit 'SpeechStarted' event (requires specific config)
                
                # Check for Utterance End
                # For now, we will assume ANY transcript means speech started if it wasn't there before
                
                transcript = result.channel.alternatives[0].transcript
                is_final = result.speech_final
                
                if transcript:
                   pass # You could trigger on_speech_start here for "very fast" barge in
                   
                if is_final and transcript.strip():
                   self.on_final_transcript(transcript)

            async def on_utterance_end(self_inner, *args, **kwargs):
                pass
                
            async def on_speech_started(self_inner, *args, **kwargs):
                # This event requires 'vad_events=True' in options
                self.on_speech_start()
            
            dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)
            dg_connection.on(LiveTranscriptionEvents.SpeechStarted, on_speech_started)

            options = LiveOptions(
                model="nova-2", 
                language="en-IN", 
                smart_format=True,
                encoding="linear16", 
                channels=1, 
                sample_rate=SAMPLE_RATE,
                interim_results=True,
                utterance_end_ms="1000",
                vad_events=True # ENABLE VAD EVENTS
            )

            await dg_connection.start(options)
            mic = Microphone(dg_connection.send)
            mic.start()

            while self.running and mic.is_active():
                await asyncio.sleep(0.1)
            
            mic.finish()
            await dg_connection.finish()

        except Exception as e:
            logger.error(f"Deepgram Error: {e}")

class VoiceAppV2:
    def __init__(self, root):
        self.root = root
        self.root.title("Bella Cucina - Production Prototype (V2)")
        self.root.geometry("800x600")
        
        self.state = SessionState(caller_name="Guest")
        self.agent = BellaAgent(self.state)
        
        # Phase 2: Manager is created when call starts (with caller name)
        self.manager = None
        self.dg_service = None
        
        # Start Manager Loop in generic thread
        self.logic_thread = None
        
        self.setup_ui()
        
        # Phase 2: Handle window close to cleanup session
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
    def setup_ui(self):
        # Name Input
        name_frame = tk.Frame(self.root)
        name_frame.pack(pady=(20, 0))
        tk.Label(name_frame, text="Your Name:", font=("Arial", 12)).pack(side=tk.LEFT, padx=5)
        self.name_entry = tk.Entry(name_frame, font=("Arial", 12))
        self.name_entry.insert(0, "Guest")
        self.name_entry.pack(side=tk.LEFT, padx=5)

        self.status_label = tk.Label(self.root, text="Ready", font=("Arial", 16), fg="blue")
        self.status_label.pack(pady=20)
        
        # Phase 2: Show session stats
        self.stats_label = tk.Label(self.root, text="Sessions: 0 | Health: checking...", font=("Arial", 10), fg="gray")
        self.stats_label.pack()
        self._update_stats()

        self.toggle_button = tk.Button(
            self.root, text="Start Call", command=self.toggle_call,
            font=("Arial", 14, "bold"), bg="#4CAF50", fg="white", width=20
        )
        self.toggle_button.pack(pady=10)

        self.transcript_box = scrolledtext.ScrolledText(self.root, width=80, height=20, font=("Consolas", 10))
        self.transcript_box.pack(padx=20, pady=20)
    
    def _update_stats(self):
        """Update stats display periodically."""
        try:
            stats = session_manager.get_session_stats()
            health = health_checker.check_health(use_cache=True)
            self.stats_label.config(
                text=f"Active Sessions: {stats['active_sessions']} | Health: {health['status']}",
                fg="green" if health["status"] == "healthy" else "orange"
            )
        except Exception:
            pass
        # Update every 5 seconds
        self.root.after(5000, self._update_stats)

    def toggle_call(self):
        if self.dg_service is None or not self.dg_service.running:
            user_name = self.name_entry.get().strip() or "Guest"
            self.state.caller_name = user_name
            
            # Phase 2: Create manager with caller name for session tracking
            self.manager = ConversationManager(self.agent, self.log_to_ui, caller_name=user_name)
            
            # Start manager loop thread
            self.logic_thread = threading.Thread(target=self.manager.start_loop, daemon=True)
            self.logic_thread.start()
            
            # Wait for loop to start
            import time
            time.sleep(0.1)
            
            self.dg_service = DeepgramService(
                on_speech_start=self.manager.on_speech_start,
                on_final_transcript=self.manager.on_transcript
            )
            
            self.dg_service.start()
            self.status_label.config(text=f"Call Active - Session: {self.manager.session_id}", fg="red")
            self.toggle_button.config(text="End Call", bg="#f44336")
            
            # Initial Greeting via Manager
            greeting = f"Hello {self.state.caller_name}! Welcome to Bella Cucina."
            asyncio.run_coroutine_threadsafe(self.manager.speak_async(greeting), self.manager.loop)
            
        else:
            # Phase 2: End session properly
            if self.manager and self.manager.call_session:
                session_manager.end_session(self.manager.session_id, "user_ended")
                logger.info(f"Call ended by user. Session: {self.manager.session_id}")
            
            self.dg_service.stop()
            self.status_label.config(text="Ready", fg="blue")
            self.toggle_button.config(text="Start Call", bg="#4CAF50")
    
    def on_close(self):
        """Handle window close - cleanup session."""
        if self.manager and hasattr(self.manager, 'call_session'):
            session_manager.end_session(self.manager.session_id, "app_closed")
        self.root.destroy()

    def log_to_ui(self, message: str):
        self.root.after(0, lambda: self._append_text(message))

    def _append_text(self, message):
        self.transcript_box.insert(tk.END, message + "\n\n")
        self.transcript_box.see(tk.END)

if __name__ == "__main__":
    # Log startup info
    logger.info("Starting Bella Cucina Voice AI (Phase 2)")
    logger.info(f"Environment: {'PRODUCTION' if IS_PRODUCTION else 'DEVELOPMENT'}")
    logger.info(f"Redis available: {session_manager._redis is not None}")
    
    root = tk.Tk()
    app = VoiceAppV2(root)
    root.mainloop()
