from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, WebSocket
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.websockets import WebSocketDisconnect

from backend.services.voice_session import get_voice_manager, VoiceCallManager

# Load environment variables
load_dotenv()

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Bella Cucina Voice Bot",
    description="A streaming, agentic voice bot for a restaurant.",
    version="1.0.0",
)

def get_manager() -> VoiceCallManager:
    return get_voice_manager()


@app.on_event("startup")
async def bootstrap_voice_stack() -> None:
    await get_voice_manager().startup()


class SessionStartRequest(BaseModel):
    caller_name: str = Field(min_length=2, description="Name captured before joining the call")


class OfferRequest(BaseModel):
    session_id: str
    signaling_token: str
    sdp: str
    type: str = "offer"

# --- REST Endpoint for Session Initiation ---

@app.post("/session/start")
async def start_session(payload: SessionStartRequest, manager: VoiceCallManager = Depends(get_manager)):
    """Starts a new session linked to a caller name and returns signaling secrets."""
    session_meta = await manager.create_session(payload.caller_name)
    return session_meta


@app.post("/webrtc/offer")
async def negotiate_offer(payload: OfferRequest, manager: VoiceCallManager = Depends(get_manager)):
    """Handles WebRTC offer/answer exchange for browser callers."""
    try:
        answer = await manager.handle_offer(
            session_id=payload.session_id,
            signaling_token=payload.signaling_token,
            offer={"sdp": payload.sdp, "type": payload.type},
        )
        return answer
    except PermissionError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.delete("/session/{session_id}")
async def end_session(session_id: str, manager: VoiceCallManager = Depends(get_manager)):
    await manager.end_session(session_id, status="ended_by_user")
    return JSONResponse({"session_id": session_id, "status": "closed"})

# --- WebSocket Endpoint for Conversation ---

@app.websocket("/ws/text/{session_id}")
async def websocket_text_bridge(websocket: WebSocket, session_id: str):
    """Fallback text channel for debugging without WebRTC audio."""
    await websocket.accept()
    manager = get_voice_manager()

    try:
        while True:
            payload = await websocket.receive_text()
            try:
                ai_reply = await manager.ingest_text(session_id, payload)
            except ValueError as exc:
                await websocket.send_text(f"Session error: {exc}")
                await websocket.close(code=1008)
                return
            await websocket.send_text(ai_reply or "...")
    except WebSocketDisconnect:
        await manager.end_session(session_id, status="disconnected")
    except Exception as exc:
        await manager.end_session(session_id, status="error")
        await websocket.close(code=1011, reason=str(exc))
