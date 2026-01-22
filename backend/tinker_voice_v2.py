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

# --- Configuration ---
load_dotenv()
GOOGLE_CREDS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
SAMPLE_RATE = 16000 # Deepgram & Mic
TTS_SAMPLE_RATE = 24000 # Google Neural Voice

if not GOOGLE_CREDS or not DEEPGRAM_API_KEY:
    raise RuntimeError("Environment variables not configured correctly.")

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_CREDS

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        np_audio = np.frombuffer(audio_data, dtype=np.int16)
        self._queue.put(np_audio)

    def stop_playback(self):
        """Immediately stop current and pending playback."""
        self.is_playing = False
        # Clear the queue
        with self._queue.mutex:
            self._queue.queue.clear()
        
        # Stop the stream if it's running
        if self._stream and self._stream.active:
            # We DONT call stop() here because restarting it is slow/buggy on Windows
            # Instead we abort the write (not easily possible in sounddevice without C-level)
            # OR we just clear queue and let it play out the tiny chunk leftover.
            
            # Actually, simply aborting the queue is usually enough for "fast" feel.
            # If we call stop(), we MUST restart it in the loop.
             try:
                self._stream.stop()
             except:
                pass

    def _playback_loop(self):
        try:
            # Reconstruct stream on demand or keep open. Keeping open is faster.
            # Using 24000 since verified user intent for Neural2 voice quality
            self._stream = sd.OutputStream(
                samplerate=24000, 
                channels=1, 
                dtype='int16'
            )
            self._stream.start()
            
            while not self._stop_event.is_set():
                try:
                    data = self._queue.get(timeout=0.1)
                    
                    if self._stream.stopped:
                         self._stream.start()
                         
                    self.is_playing = True
                    self._stream.write(data)
                    self.is_playing = False
                    self._queue.task_done()
                except queue.Empty:
                    continue
                except Exception as e:
                    # Ignore "stream stopped" error if we stopped it on purpose
                    if "PaErrorCode -9983" not in str(e):
                        logger.error(f"Playback error: {e}")
            
            self._stream.stop()
            self._stream.close()
            
        except Exception as e:
            logger.error(f"Audio Output Stream Error: {e}")

class ConversationManager:
    def __init__(self, agent: BellaAgent, ui_callback):
        self.agent = agent
        self.log_ui = ui_callback
        
        self.tts_client = texttospeech.TextToSpeechClient()
        self.player = AudioPlayer()
        self.player.start()
        
        # State
        self.loop = None
        self.input_buffer: List[str] = []
        self.last_speech_time = 0
        self.debounce_timer: Optional[asyncio.Task] = None
        self.current_agent_task: Optional[asyncio.Task] = None
        
        # Loop Circuit Breaker
        self.response_history: Deque[str] = deque(maxlen=2)
        
        # Debounce settings
        self.DEBOUNCE_DELAY = 1.0 # Seconds to wait for "Make that two"
        
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
        
        # UI Log removed for buffered text
        
        # Add to buffer
        self.input_buffer.append(text)
        
        # Reschedule processing
        if self.debounce_timer:
            self.debounce_timer.cancel()
        
        # Schedule next process
        # We use call_soon_threadsafe to schedule the async task from this thread
        future = asyncio.run_coroutine_threadsafe(self._schedule_debounce(), self.loop)

    async def _schedule_debounce(self):
        try:
            await asyncio.sleep(self.DEBOUNCE_DELAY)
            await self._process_buffer()
        except asyncio.CancelledError:
            pass

    async def _process_buffer(self):
        if not self.input_buffer: return
        
        full_text = " ".join(self.input_buffer)
        self.input_buffer = [] # Clear immediately so new speech fills new buffer
        
        self.log_ui(f"User: {full_text}")
        
        # Store for potential rescue
        self.current_processing_text = full_text
        
        self.current_agent_task = asyncio.create_task(self._run_agent(full_text))

    async def _run_agent(self, text: str):
        try:
            # If we started running, we are "consuming" the text, but until it's done,
            # it might be interrupted. The rescue logic in on_speech_start handles 
            # putting it back.
            
            message_buffer = ""
            current_sentence = ""
            full_response_accumulator = ""
            
            # We will use the agent directly effectively
            async for chunk in self.agent.orchestrator.process_message(self.agent.state, text):
               # HANDLE ERROR TOKENS GRACEFULLY
               if "Invalid Format" in str(chunk) or "Exception" in str(chunk):
                   logger.error(f"Suppressed Error Output: {chunk}")
                   continue
                   
               message_buffer += chunk
               current_sentence += chunk
               
               # Simple Sentence Streaming Logic:
               if any(punct in chunk for punct in ['.', '?', '!']):
                   # Found a punctuation mark in this chunk, likely end of sentence
                   to_speak = current_sentence.strip()
                   if to_speak:
                       logger.info(f"Streaming Sentence: {to_speak}")
                       await self.speak_async(to_speak)
                       full_response_accumulator += to_speak + " "
                   current_sentence = ""

            # Flush remaining
            if current_sentence.strip():
                logger.info(f"Streaming Final: {current_sentence}")
                await self.speak_async(current_sentence)
                full_response_accumulator += current_sentence

            # Clear processing text as we are done successfully
            self.current_processing_text = None

            final_response = full_response_accumulator.strip()
            
            if not final_response:
                final_response = "I'm sorry, I didn't quite catch that. Could you please repeat?"
                await self.speak_async(final_response)
            
            # 2. Loop Check
            # We check loop against the FULL response, even though we already spoke it.
            # If loop detected, we might have already spoken it once, but we will kill the session.
            response_hash = hashlib.md5(final_response.encode()).hexdigest()
            if self.response_history.count(response_hash) >= 2:
                termination_msg = " I am having technical difficulties. I will terminate the call now. Goodbye." # Space to separate
                self.log_ui("⚠️ LOOP DETECTED. TERMINATING SCRIPT.")
                await self.speak_async(termination_msg)
                await asyncio.sleep(3)
                sys.exit(0) # Hard exit on loop
            
            self.response_history.append(response_hash)
            
            self.log_ui(f"Bella: {final_response}")
            # self.log_ui("Listening to user...") 
            # self.log_ui("Bella is speaking...") # Already speaking via stream
            # await self.speak_async(final_response) # Don't speak again!

        except asyncio.CancelledError:
            logger.info("Agent task cancelled")
        except Exception as e:
            logger.error(f"Agent Error: {e}")
    def speak(self, text: str):
        # Fallback synchronous speak method if needed, but we mostly use speak_async
        asyncio.run_coroutine_threadsafe(self.speak_async(text), self.loop)

    async def speak_async(self, text: str):
        """Converts text to speech and plays it."""
        try:
            clean_text = text.replace("*", "")
            if not clean_text.strip(): return
            
            s_input = texttospeech.SynthesisInput(text=clean_text)
            voice = texttospeech.VoiceSelectionParams(language_code="en-IN", name="en-IN-Neural2-A")
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.LINEAR16,
                speaking_rate=1.25 # User liked faster
            )
            
            # Blocking network call, run in executor
            response = await self.loop.run_in_executor(
                None, 
                lambda: self.tts_client.synthesize_speech(input=s_input, voice=voice, audio_config=audio_config)
            )
            
            self.player.play_audio(response.audio_content)
            
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
                language="en-US", 
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
        
        self.manager = ConversationManager(self.agent, self.log_to_ui)
        self.dg_service = DeepgramService(
            on_speech_start=self.manager.on_speech_start,
            on_final_transcript=self.manager.on_transcript
        )
        
        # Start Manager Loop in generic thread
        self.logic_thread = threading.Thread(target=self.manager.start_loop, daemon=True)
        self.logic_thread.start()
        
        self.setup_ui()
        
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

        self.toggle_button = tk.Button(
            self.root, text="Start Call", command=self.toggle_call,
            font=("Arial", 14, "bold"), bg="#4CAF50", fg="white", width=20
        )
        self.toggle_button.pack(pady=10)

        self.transcript_box = scrolledtext.ScrolledText(self.root, width=80, height=20, font=("Consolas", 10))
        self.transcript_box.pack(padx=20, pady=20)

    def toggle_call(self):
        if not self.dg_service.running:
            user_name = self.name_entry.get().strip()
            if user_name: self.state.caller_name = user_name
            
            self.dg_service.start()
            self.status_label.config(text="Call Active", fg="red")
            self.toggle_button.config(text="End Call", bg="#f44336")
            
            # Initial Greeting via Manager
            # We use speak_async directly in the manager loop
            greeting = f"Hello {self.state.caller_name}! Welcome to Bella Cucina."
            asyncio.run_coroutine_threadsafe(self.manager.speak_async(greeting), self.manager.loop)
            
        else:
            self.dg_service.stop()
            self.status_label.config(text="Ready", fg="blue")
            self.toggle_button.config(text="Start Call", bg="#4CAF50")

    def log_to_ui(self, message: str):
        self.root.after(0, lambda: self._append_text(message))

    def _append_text(self, message):
        self.transcript_box.insert(tk.END, message + "\n\n")
        self.transcript_box.see(tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = VoiceAppV2(root)
    root.mainloop()
