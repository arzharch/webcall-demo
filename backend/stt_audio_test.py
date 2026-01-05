"""
Live Speech-to-Text Recorder with Tkinter UI
Streams audio from microphone and transcribes in real-time using Deepgram
"""
import tkinter as tk
from tkinter import scrolledtext
import asyncio
import threading
import os

import sounddevice as sd
from deepgram import AsyncDeepgramClient
from deepgram.core.events import EventType
from deepgram.extensions.types.sockets import ListenV1ControlMessage, ListenV1SocketClientResponse
from dotenv import load_dotenv

load_dotenv()


class TranscriptManager:
    """Collects and manages transcript fragments during streaming."""
    
    def __init__(self):
        self.fragments = []
    
    def add_fragment(self, text: str):
        self.fragments.append(text)
    
    def get_combined_transcript(self) -> str:
        return ' '.join(self.fragments).strip()
    
    def reset(self):
        self.fragments.clear()


class LiveSTTRecorderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🎤 Live Speech-to-Text Recorder")
        self.root.geometry("700x600")
        
        # Initialize Deepgram client
        api_key = os.getenv('DEEPGRAM_API_KEY')
        if not api_key:
            raise ValueError("DEEPGRAM_API_KEY not found in .env file")
        
        # SDK initialization
        self.dg_client = AsyncDeepgramClient(api_key=api_key)
        
        # Streaming state
        self.is_streaming = False
        self.transcript_manager = TranscriptManager()
        self.stream_thread = None
        self.loop = None
        self.audio_stream = None
        self.audio_queue = None
        self.committed_text = ""
        
        # Create UI
        self.create_widgets()
        
    def create_widgets(self):
        # Title
        title = tk.Label(
            self.root,
            text="🎤 Live Speech-to-Text Recorder",
            font=("Arial", 20, "bold"),
            pady=15
        )
        title.pack()
        
        # Instructions
        instructions = tk.Label(
            self.root,
            text="Click 'Start Live Transcription' and speak into your microphone.\n"
                 "Transcription appears in real-time below.",
            font=("Arial", 11),
            pady=5
        )
        instructions.pack()
        
        # Status label
        self.status_label = tk.Label(
            self.root,
            text="Ready to start",
            font=("Arial", 13, "bold"),
            fg="blue",
            pady=10
        )
        self.status_label.pack()
        
        # Button frame
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=15)
        
        # Start button
        self.start_button = tk.Button(
            button_frame,
            text="🎙️ Start Live Transcription",
            command=self.start_streaming,
            font=("Arial", 12, "bold"),
            bg="#4CAF50",
            fg="white",
            width=20,
            height=2
        )
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        # Stop button
        self.stop_button = tk.Button(
            button_frame,
            text="⏹️ Stop Transcription",
            command=self.stop_streaming,
            font=("Arial", 12, "bold"),
            bg="#f44336",
            fg="white",
            width=20,
            height=2,
            state=tk.DISABLED
        )
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # Clear button
        self.clear_button = tk.Button(
            button_frame,
            text="🗑️ Clear",
            command=self.clear_transcript,
            font=("Arial", 12),
            bg="#FF9800",
            fg="white",
            width=10,
            height=2
        )
        self.clear_button.pack(side=tk.LEFT, padx=5)
        
        # Transcription label
        transcription_label = tk.Label(
            self.root,
            text="Live Transcription:",
            font=("Arial", 13, "bold"),
            pady=10
        )
        transcription_label.pack()
        
        # Transcription text area
        self.transcription_text = scrolledtext.ScrolledText(
            self.root,
            width=80,
            height=20,
            font=("Arial", 11),
            wrap=tk.WORD,
            bg="#f5f5f5"
        )
        self.transcription_text.pack(padx=15, pady=10, fill=tk.BOTH, expand=True)
        
    def start_streaming(self):
        """Start live transcription streaming"""
        if self.is_streaming:
            return
        
        self.is_streaming = True
        self.committed_text = ""
        
        # Update UI
        self.status_label.config(text="🔴 LIVE - Listening...", fg="red")
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        
        # Start streaming in a separate thread
        self.stream_thread = threading.Thread(target=self.run_stream, daemon=True)
        self.stream_thread.start()
    
    def run_stream(self):
        """Run the async streaming in a separate thread"""
        try:
            # Create new event loop for this thread
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            
            # Run the streaming coroutine
            self.loop.run_until_complete(self.stream_transcription())
            
        except Exception as e:
            print(f"Streaming error: {e}")
            import traceback
            traceback.print_exc()
            self.update_status(f"❌ Error: {str(e)}", "red")
        finally:
            if self.loop:
                self.loop.close()
                self.loop = None
    
    async def stream_transcription(self):
        """Main streaming logic using Deepgram Listen v1 websockets"""
        try:
            async with self.dg_client.listen.v1.connect(
                model="nova-2",
                language="en-US",
                punctuate="true",
                encoding="linear16",
                channels="1",
                sample_rate="16000",
                endpointing="300",
                interim_results="true",
                smart_format="true",
            ) as connection:
                self.audio_queue = asyncio.Queue()

                def handle_message(message: ListenV1SocketClientResponse):
                    try:
                        if getattr(message, "type", "").lower() != "results":
                            return
                        if not message.channel.alternatives:
                            return
                        text = message.channel.alternatives[0].transcript.strip()
                        if not text:
                            return
                        is_final = bool(getattr(message, "is_final", False))
                        speech_final = bool(getattr(message, "speech_final", False))
                        if not is_final and not speech_final:
                            interim_text = text
                            if self.committed_text and interim_text.startswith(self.committed_text):
                                interim_text = interim_text[len(self.committed_text):].strip()
                            if interim_text:
                                self.transcript_manager.reset()
                                self.transcript_manager.add_fragment(interim_text)
                                self.update_interim_transcript(interim_text)
                        else:
                            new_text = text
                            if self.committed_text and new_text.startswith(self.committed_text):
                                new_text = new_text[len(self.committed_text):].strip()
                            if new_text:
                                self.append_final_transcript(new_text)
                            self.committed_text = text
                            self.transcript_manager.reset()
                    except Exception as exc:
                        print(f"Error handling transcript: {exc}")

                def handle_error(error):
                    print(f"[Deepgram Error] {error}")
                    self.update_status(f"⚠️ Stream error: {error}", "orange")

                def handle_close(_):
                    if self.is_streaming:
                        self.update_status("Connection closed", "orange")

                connection.on(EventType.MESSAGE, handle_message)
                connection.on(EventType.ERROR, handle_error)
                connection.on(EventType.CLOSE, handle_close)

                listen_task = asyncio.create_task(connection.start_listening())

                def audio_callback(indata, frames, time_info, status):
                    if status:
                        print(f"[Audio Warning] {status}")
                    if not self.is_streaming:
                        return
                    data = bytes(indata)
                    if not data or not self.loop or not self.audio_queue:
                        return
                    try:
                        self.loop.call_soon_threadsafe(self.audio_queue.put_nowait, data)
                    except RuntimeError:
                        pass

                self.audio_stream = sd.RawInputStream(
                    samplerate=16000,
                    channels=1,
                    dtype="int16",
                    blocksize=0,
                    callback=audio_callback,
                )
                self.audio_stream.start()
                print("✅ Live transcription started")

                try:
                    while self.is_streaming:
                        try:
                            chunk = await asyncio.wait_for(self.audio_queue.get(), timeout=0.5)
                        except asyncio.TimeoutError:
                            continue
                        if chunk:
                            await connection.send_media(chunk)
                finally:
                    self._cleanup_audio_stream()
                    try:
                        await connection.send_control(ListenV1ControlMessage(type="Finalize"))
                    except Exception as finalize_error:
                        print(f"Finalize error: {finalize_error}")
                    try:
                        await asyncio.wait_for(listen_task, timeout=5)
                    except asyncio.TimeoutError:
                        listen_task.cancel()
                        print("Listen task timed out")

                print("✅ Live transcription stopped")

        except Exception as e:
            print(f"Stream error: {e}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            self.audio_queue = None
    
    def stop_streaming(self):
        """Stop live transcription streaming"""
        self.is_streaming = False
        
        # Update UI
        self.update_status("⏹️ Stopping...", "orange")
        self.stop_button.config(state=tk.DISABLED)
        if self.loop and self.audio_queue:
            def _notify_queue():
                if self.audio_queue:
                    self.audio_queue.put_nowait(b"")
            try:
                self.loop.call_soon_threadsafe(_notify_queue)
            except RuntimeError:
                pass
        
        # Wait a moment then reset UI
        self.root.after(1000, self.reset_ui)
    
    def reset_ui(self):
        """Reset UI after stopping"""
        self.status_label.config(text="Ready to start", fg="blue")
        self.start_button.config(state=tk.NORMAL)
    
    def update_interim_transcript(self, text):
        """Update with interim (non-final) transcript in gray"""
        def update():
            # Remove previous interim line if exists
            content = self.transcription_text.get("1.0", tk.END)
            lines = content.split('\n')
            if lines and lines[-2].startswith("  ➜ "):
                self.transcription_text.delete("end-2l", "end-1l")
            
            # Add new interim text
            self.transcription_text.insert(tk.END, f"  ➜ {text}\n", "interim")
            self.transcription_text.tag_config("interim", foreground="gray")
            self.transcription_text.see(tk.END)
        
        self.root.after(0, update)
    
    def append_final_transcript(self, text):
        """Append final transcript"""
        def update():
            # Remove interim line if exists
            content = self.transcription_text.get("1.0", tk.END)
            lines = content.split('\n')
            if lines and lines[-2].startswith("  ➜ "):
                self.transcription_text.delete("end-2l", "end-1l")
            
            # Add final transcript
            self.transcription_text.insert(tk.END, f"🗣️ {text}\n\n", "final")
            self.transcription_text.tag_config("final", foreground="black")
            self.transcription_text.see(tk.END)
        
        self.root.after(0, update)

    def _cleanup_audio_stream(self):
        """Stop and release the audio capture stream."""
        if self.audio_stream:
            try:
                self.audio_stream.stop()
            except Exception:
                pass
            try:
                self.audio_stream.close()
            except Exception:
                pass
            self.audio_stream = None
    
    def clear_transcript(self):
        """Clear the transcript text area"""
        self.transcription_text.delete("1.0", tk.END)
        self.transcript_manager.reset()
        self.committed_text = ""
    
    def update_status(self, message, color):
        """Update status label from any thread"""
        self.root.after(0, lambda: self.status_label.config(text=message, fg=color))
    
    def on_closing(self):
        """Handle window closing"""
        if self.is_streaming:
            self.stop_streaming()
        self.root.destroy()


def main():
    print("🎤 Starting Live Speech-to-Text Recorder...")
    print("✅ Loading UI...")
    
    root = tk.Tk()
    app = LiveSTTRecorderApp(root)
    
    # Handle window close
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    print("✅ UI loaded! Application is ready.")
    print("📝 Make sure your microphone is connected and working.")
    
    root.mainloop()


if __name__ == '__main__':
    main()