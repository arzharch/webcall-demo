import asyncio
import contextlib
import json
import logging
import threading
import tkinter as tk
from dataclasses import dataclass
from fractions import Fraction
from tkinter import messagebox, scrolledtext, ttk
from typing import Callable, Optional

import aiohttp
import numpy as np
import sounddevice as sd
import websockets
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamTrack
from av import AudioFrame

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tinker-ui")


class MicrophoneStreamTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, sample_rate: int = 16000, chunk_ms: int = 20) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.samples = int(sample_rate * (chunk_ms / 1000))
        self.stream = sd.InputStream(
            samplerate=sample_rate, channels=1, dtype="int16", blocksize=self.samples
        )
        self.stream.start()
        self._seq = 0

    async def recv(self) -> AudioFrame:
        loop = asyncio.get_event_loop()
        data, _ = await loop.run_in_executor(None, self.stream.read, self.samples)
        pcm = np.squeeze(data).astype(np.int16)
        frame = AudioFrame(format="s16", layout="mono", samples=len(pcm))
        frame.planes[0].update(pcm.tobytes())
        frame.sample_rate = self.sample_rate
        frame.time_base = Fraction(1, self.sample_rate)
        frame.pts = self._seq * len(pcm)
        self._seq += 1
        return frame

    async def stop(self) -> None:
        await super().stop()
        self.stream.stop()
        self.stream.close()


class SpeakerPlayer:
    def __init__(self, sample_rate: int = 16000) -> None:
        self.sample_rate = sample_rate
        self.stream = sd.OutputStream(
            samplerate=sample_rate, channels=1, dtype="int16"
        )
        self.stream.start()
        self._task: Optional[asyncio.Task] = None

    async def play(self, track: MediaStreamTrack) -> None:
        try:
            while True:
                frame = await track.recv()
                pcm = frame.to_ndarray().astype("int16")
                await asyncio.get_event_loop().run_in_executor(
                    None, self.stream.write, pcm
                )
        except asyncio.CancelledError:
            pass
        finally:
            self.stream.stop()
            self.stream.close()

    def start(self, track: MediaStreamTrack, loop: asyncio.AbstractEventLoop) -> None:
        self._task = loop.create_task(self.play(track))

    async def close(self) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task


@dataclass
class TranscriptEvent:
    speaker: str
    text: str


class VoiceWorker:
    def __init__(
        self,
        api_base: str,
        caller_name: str,
        log_cb: Callable[[str], None],
        transcript_cb: Callable[[TranscriptEvent], None],
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.caller_name = caller_name
        self.log = log_cb
        self.push_transcript = transcript_cb

        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)

        self.session: Optional[aiohttp.ClientSession] = None
        self.pc: Optional[RTCPeerConnection] = None
        self.mic_track: Optional[MicrophoneStreamTrack] = None
        self.speaker: Optional[SpeakerPlayer] = None
        self.ws_task: Optional[asyncio.Task] = None
        self.stop_event: Optional[asyncio.Event] = None
        self.session_id: Optional[str] = None
        self.signaling_token: Optional[str] = None

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        if self.loop.is_running() and self.stop_event:
            self.loop.call_soon_threadsafe(self.stop_event.set)
            self.thread.join(timeout=5)

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._main())
        self.loop.close()

    async def _main(self) -> None:
        self.stop_event = asyncio.Event()
        self.session = aiohttp.ClientSession()
        try:
            await self._start_call()
            self.log("Call active. Speak into your microphone.")
            await self.stop_event.wait()
        except Exception as exc:
            logger.exception("Worker error: %s", exc)
            self.log(f"Error: {exc}")
        finally:
            await self._cleanup()

    async def _start_call(self) -> None:
        async with self.session.post(
            f"{self.api_base}/session/start",
            json={"caller_name": self.caller_name},
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
            self.session_id = data["session_id"]
            self.signaling_token = data["signaling_token"]
            self.log(f"Session {self.session_id} created.")

        await self._connect_webrtc()
        self.ws_task = self.loop.create_task(self._consume_transcripts())

    async def _connect_webrtc(self) -> None:
        if not self.session_id or not self.signaling_token:
            raise RuntimeError("Session not initialized")

        self.pc = RTCPeerConnection()
        self.mic_track = MicrophoneStreamTrack()
        self.pc.addTrack(self.mic_track)

        self.speaker = SpeakerPlayer()

        @self.pc.on("track")
        def on_track(track: MediaStreamTrack) -> None:
            if track.kind == "audio" and self.speaker:
                self.log("Remote audio track received.")
                self.speaker.start(track, self.loop)

        offer = await self.pc.createOffer()
        await self.pc.setLocalDescription(offer)

        payload = {
            "session_id": self.session_id,
            "signaling_token": self.signaling_token,
            "sdp": offer.sdp,
            "type": offer.type,
        }
        async with self.session.post(
            f"{self.api_base}/webrtc/offer", json=payload
        ) as resp:
            resp.raise_for_status()
            answer = await resp.json()

        await self.pc.setRemoteDescription(
            RTCSessionDescription(sdp=answer["sdp"], type=answer["type"])
        )
        self.log("WebRTC negotiation complete.")

    async def _consume_transcripts(self) -> None:
        if not self.session_id:
            return
        ws_url = self.api_base.replace("http", "ws") + f"/ws/text/{self.session_id}"
        try:
            async with websockets.connect(ws_url) as ws:
                async for message in ws:
                    data = json.loads(message)
                    self.push_transcript(
                        TranscriptEvent(
                            speaker=data.get("speaker", "assistant"),
                            text=data.get("text", ""),
                        )
                    )
        except Exception as exc:
            self.log(f"Transcript stream closed: {exc}")

    async def _cleanup(self) -> None:
        tasks = []
        if self.ws_task:
            self.ws_task.cancel()
            tasks.append(self.ws_task)
        if self.pc:
            tasks.append(self.pc.close())
        if self.mic_track:
            tasks.append(self.mic_track.stop())
        if self.speaker:
            tasks.append(self.speaker.close())
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if self.session:
            await self.session.close()
        self.log("Call ended.")


class TkVoiceApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Bella Voice – Tk Client")
        self.api_var = tk.StringVar(value="http://localhost:8000")
        self.name_var = tk.StringVar()

        self.worker: Optional[VoiceWorker] = None

        self._build_ui()

    def _build_ui(self) -> None:
        frame = ttk.Frame(self.root, padding=12)
        frame.grid(row=0, column=0, sticky="nsew")

        ttk.Label(frame, text="API Base URL").grid(row=0, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.api_var, width=40).grid(
            row=0, column=1, sticky="we"
        )

        ttk.Label(frame, text="Caller Name").grid(row=1, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.name_var, width=30).grid(
            row=1, column=1, sticky="we"
        )

        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=2, column=0, columnspan=2, pady=8, sticky="we")
        ttk.Button(btn_frame, text="Start Call", command=self.start_call).grid(
            row=0, column=0, padx=(0, 8)
        )
        ttk.Button(btn_frame, text="Stop Call", command=self.stop_call).grid(
            row=0, column=1
        )

        ttk.Label(frame, text="Logs").grid(row=3, column=0, sticky="w")
        self.log_box = scrolledtext.ScrolledText(frame, height=10, state="disabled")
        self.log_box.grid(row=4, column=0, columnspan=2, sticky="nsew")

        ttk.Label(frame, text="Transcript").grid(row=5, column=0, sticky="w")
        self.transcript_box = scrolledtext.ScrolledText(
            frame, height=12, state="disabled"
        )
        self.transcript_box.grid(row=6, column=0, columnspan=2, sticky="nsew")

        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(6, weight=1)

    def start_call(self) -> None:
        if self.worker:
            messagebox.showinfo("Call running", "A call is already active.")
            return
        name = self.name_var.get().strip().lower()
        if len(name) < 2:
            messagebox.showerror("Invalid name", "Enter at least two characters.")
            return

        api = self.api_var.get().strip()
        self.worker = VoiceWorker(api, name, self._log, self._add_transcript)
        self.worker.start()
        self._log("Connecting…")

    def stop_call(self) -> None:
        if self.worker:
            self.worker.stop()
            self.worker = None

    def _log(self, message: str) -> None:
        def append() -> None:
            self.log_box.configure(state="normal")
            self.log_box.insert(tk.END, f"{message}\n")
            self.log_box.configure(state="disabled")
            self.log_box.see(tk.END)

        self.root.after(0, append)

    def _add_transcript(self, event: TranscriptEvent) -> None:
        def append() -> None:
            self.transcript_box.configure(state="normal")
            self.transcript_box.insert(
                tk.END, f"{event.speaker.title()}: {event.text}\n"
            )
            self.transcript_box.configure(state="disabled")
            self.transcript_box.see(tk.END)

        self.root.after(0, append)

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    app = TkVoiceApp()
    app.run()