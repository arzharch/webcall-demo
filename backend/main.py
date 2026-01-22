from __future__ import annotations

import logging

from dotenv import load_dotenv

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(name)s - %(levelname)s - %(message)s",
)
# ---

from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import get_settings
from services.voice_session import VoiceCallManager, get_voice_manager

# Load environment variables
load_dotenv()

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Bella Cucina Voice Bot",
    description="A streaming, agentic voice bot for a restaurant.",
    version="1.0.0",
)

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class StartSessionPayload(BaseModel):
    caller_name: str = Field(..., min_length=2)


class SessionMeta(BaseModel):
    session_id: str
    signaling_token: str


class OfferPayload(BaseModel):
    session_id: str
    signaling_token: str
    sdp: str
    type: str


@app.post("/session/start", response_model=SessionMeta)
async def start_session(
    payload: StartSessionPayload,
    manager: VoiceCallManager = Depends(get_voice_manager),
):
    name = payload.caller_name.strip().lower()
    meta = await manager.create_session(name)
    return SessionMeta(**meta)


@app.post("/webrtc/offer")
async def handle_offer(
    payload: OfferPayload,
    manager: VoiceCallManager = Depends(get_voice_manager),
):
    try:
        answer = await manager.accept_offer(
            payload.session_id,
            payload.signaling_token,
            {"sdp": payload.sdp, "type": payload.type},
        )
        return answer
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.websocket("/ws/text/{session_id}")
async def transcript_ws(
    websocket: WebSocket,
    session_id: str,
    manager: VoiceCallManager = Depends(get_voice_manager),
):
    await websocket.accept()
    try:
        queue = manager.subscribe_transcripts(session_id)
    except ValueError:
        await websocket.close(code=4404)
        return

    try:
        while True:
            message = await queue.get()
            await websocket.send_json(message)
    except WebSocketDisconnect:
        pass
    finally:
        manager.unsubscribe_transcripts(session_id, queue)
