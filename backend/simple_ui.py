import tkinter as tk
from tkinter import scrolledtext
import asyncio
import threading
import os
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

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        if self._loop:
            self._loop.call_soon_threadsafe(self._stop_loop)

    def _stop_loop(self):
        if self.mic: self.mic.finish()
        if self.dg_stream: asyncio.create_task(self.dg_stream.finish())

    def _run(self):
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

            while self.mic.is_active():
                await asyncio.sleep(0.1)

        except Exception as e:
            print(f"Deepgram Error: {e}")

class SimpleRealtimeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Synthion AI Agent")
        self.root.geometry("600x450")

        self.tts_client = texttospeech.TextToSpeechClient()
        self.transcriber = RealtimeTranscriber(self.handle_final_transcript)
        self.is_listening = False

        # UI Components
        self.status_label = tk.Label(root, text="Ready", font=("Arial", 14), fg="blue")
        self.status_label.pack(pady=10)

        self.toggle_button = tk.Button(
            root, text="Start Listening", command=self.toggle_listening,
            font=("Arial", 12), bg="#4CAF50", fg="white", width=20
        )
        self.toggle_button.pack(pady=10)

        self.transcript_box = scrolledtext.ScrolledText(root, width=70, height=15)
        self.transcript_box.pack(padx=10, pady=10)

    def toggle_listening(self):
        if not self.is_listening:
            self.transcriber.start()
            self.is_listening = True
            self.status_label.config(text="Listening...", fg="red")
            self.toggle_button.config(text="Stop Listening", bg="#f44336")
        else:
            self.transcriber.stop()
            self.is_listening = False
            self.status_label.config(text="Ready", fg="blue")
            self.toggle_button.config(text="Start Listening", bg="#4CAF50")

    def handle_final_transcript(self, transcript: str):
        self.log_to_ui(f"You: {transcript}")
        # Add your LLM logic here
        response = f"I heard you say: {transcript}"
        self.log_to_ui(f"Assistant: {response}")
        self.speak(response)

    def speak(self, text: str):
        # Using Google TTS from your previous code
        s_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(language_code="en-IN", name="en-IN-Neural2-A")
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.LINEAR16)
        
        response = self.tts_client.synthesize_speech(input=s_input, voice=voice, audio_config=audio_config)
        
        # Play the audio
        import numpy as np
        audio_data = np.frombuffer(response.audio_content, dtype=np.int16)
        sd.play(audio_data, SAMPLE_RATE)

    def log_to_ui(self, message: str):
        self.root.after(0, lambda: self._append_text(message))

    def _append_text(self, message):
        self.transcript_box.insert(tk.END, message + "\n\n")
        self.transcript_box.see(tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = SimpleRealtimeApp(root)
    root.mainloop()