import asyncio
import secrets
import time
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Dict, Optional

from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.exceptions import InvalidStateError
from aiortc.mediastreams import MediaStreamTrack
from av import AudioFrame
from backend.agent.agent import BellaAgent
from backend.agent.state import SessionState
from backend.config import Settings, get_settings
from backend.database import Database, get_database
from backend.services.cost_tracker import CostTracker
from backend.services.restaurant import get_restaurant_template
from backend.services.stt_service import DeepgramSTTService
from backend.services.tts_service import GoogleTTSService
from backend.services.vad import SpeechGate


class TTSAudioTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, sample_rate: int) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self._queue: asyncio.Queue[AudioFrame] = asyncio.Queue()
        self._closed = False

    async def recv(self) -> AudioFrame:
        if self._closed:
            raise InvalidStateError("Audio track closed")
        frame = await self._queue.get()
        return frame

    async def enqueue_pcm(self, pcm_bytes: bytes) -> None:
        if not pcm_bytes or self._closed:
            return
        frame_bytes = int(self.sample_rate * 20 / 1000) * 2  # 20 ms
        for chunk in _chunk_bytes(pcm_bytes, frame_bytes):
            samples = max(1, len(chunk) // 2)
            frame = AudioFrame(format="s16", layout="mono", samples=samples)
            frame.planes[0].update(chunk)
            frame.sample_rate = self.sample_rate
            frame.time_base = Fraction(1, self.sample_rate)
            await self._queue.put(frame)

    async def close(self) -> None:
        self._closed = True
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break


@dataclass
class VoiceCallSession:
    session_id: str
    caller_name: str
    settings: Settings
    database: Database
    agent: BellaAgent
    cost_tracker: CostTracker
    stt_service: DeepgramSTTService
    tts_service: GoogleTTSService
    vad: SpeechGate

    peer_connection: Optional[RTCPeerConnection] = None
    tts_track: Optional[TTSAudioTrack] = None
    session_state: SessionState = None
    _tasks: list = None

    def __post_init__(self) -> None:
        self.session_state = SessionState(
            session_id=self.session_id, caller_name=self.caller_name
        )
        self._tasks = []

    async def attach_peer_connection(self, offer: Dict[str, str]) -> Dict[str, str]:
        if self.peer_connection:
            await self.peer_connection.close()

        pc = RTCPeerConnection()
        self.peer_connection = pc
        self.tts_track = TTSAudioTrack(self.settings.SAMPLE_RATE)
        pc.addTrack(self.tts_track)

        @pc.on("track")
        async def on_track(track: MediaStreamTrack) -> None:
            if track.kind == "audio":
                task = asyncio.create_task(self._consume_audio(track))
                self._tasks.append(task)

        @pc.on("connectionstatechange")
        async def on_state_change() -> None:
            if pc.connectionState in {"failed", "closed"}:
                await self.shutdown(status="terminated")

        rtc_offer = RTCSessionDescription(sdp=offer["sdp"], type=offer.get("type", "offer"))
        await pc.setRemoteDescription(rtc_offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

    async def _consume_audio(self, track: MediaStreamTrack) -> None:
        try:
            while True:
                frame = await track.recv()
                pcm = frame.to_ndarray(format="s16", layout="mono")
                pcm_bytes = pcm.tobytes()
                frame_duration = len(pcm_bytes) / (2 * self.settings.SAMPLE_RATE)
                self.cost_tracker.add_stt_seconds(frame_duration)
                for chunk in _chunk_bytes(pcm_bytes, self.vad.frame_bytes):
                    if len(chunk) != self.vad.frame_bytes:
                        continue
                    voiced = self.vad.process(chunk)
                    if voiced:
                        transcript = await self.stt_service.transcribe_pcm(voiced)
                        if transcript:
                            await self._handle_transcript(transcript)
        except asyncio.CancelledError:
            return
        except Exception:
            await self.shutdown(status="error")

    async def _handle_transcript(self, user_text: str) -> None:
        await self.database.record_message(self.session_id, "user", user_text)

        ai_response = ""
        async for chunk in self.agent.process_message(self.session_state, user_text):
            ai_response = chunk

        if not ai_response:
            ai_response = "I did not catch that. Could you repeat?"

        self.session_state.update_history(user_text, ai_response)
        await self.database.record_message(self.session_id, "assistant", ai_response)

        # naive token estimate: 1 token per 4 chars
        est_prompt_tokens = max(1, len(user_text) // 4)
        est_completion_tokens = max(1, len(ai_response) // 4)
        self.cost_tracker.add_llm_usage(est_prompt_tokens, est_completion_tokens)

        audio_bytes = await self.tts_service.synthesize(ai_response)
        if audio_bytes and self.tts_track:
            await self.tts_track.enqueue_pcm(audio_bytes)

    async def shutdown(self, status: str = "completed") -> None:
        for task in self._tasks:
            task.cancel()
        self._tasks.clear()
        if self.peer_connection:
            await self.peer_connection.close()
            self.peer_connection = None
        if self.tts_track:
            await self.tts_track.close()
            self.tts_track = None
        snapshot = self.cost_tracker.snapshot()
        last_ai_message = None
        for entry in reversed(self.session_state.conversation_history):
            if entry.startswith("AI:"):
                last_ai_message = entry.replace("AI:", "").strip()
                break
        await self.database.close_call_session(
            session_id=self.session_id,
            status=status,
            last_agent_message=last_ai_message,
            cost_snapshot=snapshot,
        )


class VoiceCallManager:
    """Orchestrates lifecycle of low-latency voice sessions."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.database = get_database()
        self.template = get_restaurant_template()
        self.sessions: Dict[str, VoiceCallSession] = {}
        self._lock = asyncio.Lock()
        self._ready = False

    async def startup(self) -> None:
        await self.database.init()
        self._ready = True

    async def create_session(self, caller_name: str) -> Dict[str, Any]:
        if not self._ready:
            await self.startup()
        session_id = secrets.token_hex(8)
        signaling_token = secrets.token_urlsafe(24)
        expires_at = int(time.time()) + self.settings.SIGNALING_TOKEN_TTL_SECONDS

        agent = BellaAgent(google_api_key=self.settings.GOOGLE_API_KEY)
        cost_tracker = CostTracker(self.settings)
        stt_service = DeepgramSTTService(self.settings)
        tts_service = GoogleTTSService(self.settings, cost_tracker=cost_tracker)
        vad = SpeechGate(
            sample_rate=self.settings.SAMPLE_RATE,
            aggressiveness=self.settings.VAD_AGGRESSIVENESS,
            frame_ms=self.settings.VAD_FRAME_MS,
            padding_ms=self.settings.VAD_END_WINDOW_MS,
        )

        session = VoiceCallSession(
            session_id=session_id,
            caller_name=caller_name,
            settings=self.settings,
            database=self.database,
            agent=agent,
            cost_tracker=cost_tracker,
            stt_service=stt_service,
            tts_service=tts_service,
            vad=vad,
        )

        async with self._lock:
            self.sessions[session_id] = session

        await self.database.create_call_session(
            session_id=session_id,
            caller_name=caller_name,
            signaling_token=signaling_token,
            token_expires_at=expires_at,
        )

        return {
            "session_id": session_id,
            "signaling_token": signaling_token,
            "token_expires_at": expires_at,
        }

    async def handle_offer(
        self, session_id: str, signaling_token: str, offer: Dict[str, str]
    ) -> Dict[str, str]:
        if not self._ready:
            await self.startup()
        valid = await self.database.verify_signaling_token(session_id, signaling_token)
        if not valid:
            raise PermissionError("Invalid or expired signaling token")

        session = self.sessions.get(session_id)
        if not session:
            raise ValueError("Session not found")

        return await session.attach_peer_connection(offer)

    async def ingest_text(self, session_id: str, text: str) -> str:
        if not self._ready:
            await self.startup()
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError("Session not found")
        await session._handle_transcript(text)
        for entry in reversed(session.session_state.conversation_history):
            if entry.startswith("AI:"):
                return entry.replace("AI:", "").strip()
        return ""

    async def end_session(self, session_id: str, status: str = "completed") -> None:
        session = self.sessions.pop(session_id, None)
        if session:
            await session.shutdown(status=status)


_manager: Optional[VoiceCallManager] = None


def get_voice_manager() -> VoiceCallManager:
    global _manager
    if _manager is None:
        _manager = VoiceCallManager()
    return _manager


def _chunk_bytes(data: bytes, frame_bytes: int):
    for idx in range(0, len(data), frame_bytes):
        yield data[idx : idx + frame_bytes]
*** End of File