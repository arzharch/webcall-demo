import tkinter as tk
from tkinter import scrolledtext
import asyncio
import threading
import os
import queue
import time
import sys
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
SAMPLE_RATE = 16000

if not GOOGLE_CREDS or not DEEPGRAM_API_KEY:
    raise RuntimeError("Environment variables not configured correctly.")

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_CREDS

class RealtimeTranscriber:
    def __init__(self, on_transcript_callback):
        self.on_transcript_callback = on_transcript_callback
        self._thread = None
        self._loop = None
        self.dg_stream = None
        self.mic = None
        self.running = False

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False
        if self._loop:
            self._loop.call_soon_threadsafe(self._stop_loop)

    def _stop_loop(self):
        if self.mic: self.mic.finish()
        if self.dg_stream: asyncio.create_task(self.dg_stream.finish())

    def _run(self):
        # Fix for Windows Event Loop Issue (Deepgram/Websockets)
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._transcribe())

    async def _transcribe(self):
        try:
            config = DeepgramClientOptions(options={"keepalive": "true"})
            dg_client = DeepgramClient(DEEPGRAM_API_KEY, config)
            self.dg_stream = dg_client.listen.asynclive.v("1")

            async def on_message(self_inner, result, **kwargs):
                transcript = result.channel.alternatives[0].transcript
                if transcript and result.speech_final:
                    self.on_transcript_callback(transcript)

            self.dg_stream.on(LiveTranscriptionEvents.Transcript, on_message)

            options = LiveOptions(
                model="nova-2", language="en-US", punctuate=True,
                encoding="linear16", channels=1, sample_rate=SAMPLE_RATE,
                endpointing=300
            )

            await self.dg_stream.start(options)
            self.mic = Microphone(self.dg_stream.send)
            self.mic.start()

            while self.running and self.mic.is_active():
                await asyncio.sleep(0.1)

        except Exception as e:
            print(f"Deepgram Error: {e}")

class VoiceAgentApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Bella Cucina - Voice Agent")
        self.root.geometry("800x600")

        # --- Services ---
        self.tts_client = texttospeech.TextToSpeechClient()
        self.transcriber = RealtimeTranscriber(self.handle_final_transcript)
        
        # --- Agent ---
        self.state = SessionState(caller_name="Guest")
        self.agent = BellaAgent(self.state)
        
        # --- State ---
        self.is_listening = False
        self.input_queue = queue.Queue()
        
        # --- UI ---
        self.setup_ui()
        
        # --- Worker Thread ---
        self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.worker_thread.start()

    def setup_ui(self):
        # Name Input
        name_frame = tk.Frame(self.root)
        name_frame.pack(pady=(20, 0))
        
        tk.Label(name_frame, text="Your Name:", font=("Arial", 12)).pack(side=tk.LEFT, padx=5)
        self.name_entry = tk.Entry(name_frame, font=("Arial", 12))
        self.name_entry.insert(0, "Guest")
        self.name_entry.pack(side=tk.LEFT, padx=5)

        # Status Label
        self.status_label = tk.Label(self.root, text="Ready", font=("Arial", 16), fg="blue")
        self.status_label.pack(pady=20)

        # Toggle Button
        self.toggle_button = tk.Button(
            self.root, text="Start Conversation", command=self.toggle_listening,
            font=("Arial", 14, "bold"), bg="#4CAF50", fg="white", 
            width=20, height=2
        )
        self.toggle_button.pack(pady=10)

        # Transcript Area
        self.transcript_box = scrolledtext.ScrolledText(self.root, width=80, height=20, font=("Consolas", 10))
        self.transcript_box.pack(padx=20, pady=20)
        self.transcript_box.insert(tk.END, "--- Conversation Started ---\n\n")

    def toggle_listening(self):
        if not self.is_listening:
            # Update Caller Name from UI
            user_name = self.name_entry.get().strip()
            if user_name:
                self.state.caller_name = user_name
            
            self.transcriber.start()
            self.is_listening = True
            self.status_label.config(text="Listening...", fg="red")
            self.toggle_button.config(text="Stop Conversation", bg="#f44336")

            # INITIAL GREETING - NEW LOGIC
            # Fire an initial greeting without waiting for user input
            threading.Thread(target=self._initial_greeting, daemon=True).start()

        else:
            self.transcriber.stop()
            self.is_listening = False
            self.status_label.config(text="Paused", fg="orange")
            self.toggle_button.config(text="Resume Conversation", bg="#4CAF50")

    def _initial_greeting(self):
        """Send a welcome message to the user immediately."""
        try:
            greeting_text = f"Hello {self.state.caller_name}! Welcome to Bella Cucina. How may I assist you today?"
            self.log_to_ui(f"Bella: {greeting_text}")
            
            # Use the loop logic similar to process queue if needed, or just direct speak
            # Be careful about thread safety with TTS client
            self.root.after(0, lambda: self.status_label.config(text="Bella is speaking...", fg="green"))
            self.speak(greeting_text)
            self.root.after(0, lambda: self.status_label.config(text="Listening...", fg="red"))
        except Exception as e:
            print(f"Error in initial greeting: {e}")

    def handle_final_transcript(self, transcript: str):
        """Called by Deepgram thread when speech is detected."""
        if not transcript.strip():
            return
        
        self.log_to_ui(f"User: {transcript}")
        self.input_queue.put(transcript)

    def _process_queue(self):
        """Worker thread to handle LLM processing and TTS sequentially."""
        # Create a persistent event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        while True:
            try:
                transcript = self.input_queue.get()
                
                # Show "Thinking..." state
                self.root.after(0, lambda: self.status_label.config(text="Bella is thinking...", fg="purple"))
                
                # 1. Get Agent Response
                # Use the persistent loop to run the async agent
                response_text = loop.run_until_complete(self.agent.respond(transcript))
                
                self.log_to_ui(f"Bella: {response_text}")
                
                # 2. Text to Speech
                self.root.after(0, lambda: self.status_label.config(text="Bella is speaking...", fg="green"))
                self.speak(response_text)
                
            except Exception as e:
                print(f"Error in processing: {e}")
                self.log_to_ui(f"System Error: {e}")
            finally:
                # Reset Status
                if self.is_listening:
                    self.root.after(0, lambda: self.status_label.config(text="Listening...", fg="red"))
                else:
                    self.root.after(0, lambda: self.status_label.config(text="Paused/Ready", fg="blue"))
                    
                self.input_queue.task_done()

    def speak(self, text: str):
        try:
            # Clean text (remove emojis or markdown if necessary used by TTS)
            clean_text = text.replace("*", "")
            
            s_input = texttospeech.SynthesisInput(text=clean_text)
            
            # Note: Using en-IN-Neural2-A as verified in simple_ui.py
            voice = texttospeech.VoiceSelectionParams(language_code="en-IN", name="en-IN-Neural2-A")
            # Speed up the voice slightly (default is 1.0)
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.LINEAR16,
                speaking_rate=1.35
            )
            
            response = self.tts_client.synthesize_speech(input=s_input, voice=voice, audio_config=audio_config)
            
            # Play Audio using sounddevice
            import numpy as np
            audio_data = np.frombuffer(response.audio_content, dtype=np.int16)
            sd.play(audio_data, SAMPLE_RATE)
            sd.wait() # Wait until finished speaking to avoid listening to self?
            
            # If we want to avoid the bot listening to itself, we might want to pause 
            # listening here, but Deepgram "endpointing" usually handles pauses.
            # However, 'sd.wait()' blocks this thread, which is good because we don't 
            # want to process the next queue item until speaking is done.
            
        except Exception as e:
            print(f"TTS Error: {e}")
            self.log_to_ui(f"TTS Error: {e}")

    def log_to_ui(self, message: str):
        self.root.after(0, lambda: self._append_text(message))

    def _append_text(self, message):
        self.transcript_box.insert(tk.END, message + "\n\n")
        self.transcript_box.see(tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = VoiceAgentApp(root)
    root.mainloop()
