"""
FastAPI Backend for Bella Voice AI.
WebSocket-based real-time voice calling.
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Load environment before other imports
load_dotenv()

from config import settings
from services.voice_session import VoiceSession, VoiceSessionManager, get_voice_session_manager
from services.system_audio import preload_system_audio
import database as db
from infra import get_health_checker, setup_logging

# Configure logging
setup_logging(
    level=os.getenv("LOG_LEVEL", "INFO"),
    json_output=os.getenv("ENV", "development").lower() == "production",
    service_name="bella-voice-api",
)
logger = logging.getLogger(__name__)


# ==================== Lifespan Events ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - startup and shutdown."""
    # Startup
    logger.info("🚀 Starting Bella Voice API...")
    db.init_database()
    logger.info("✅ Database initialized")
    
    # Preload system audio assets (ringing, error messages, etc.)
    try:
        logger.info("📢 Preloading system audio assets...")
        preload_system_audio()
        logger.info("✅ System audio preloaded")
    except Exception as e:
        logger.warning(f"⚠️ Failed to preload system audio: {e}")
    
    yield
    
    # Shutdown
    logger.info("👋 Shutting down Bella Voice API...")
    
    # End all active sessions
    manager = get_voice_session_manager()
    for session_info in manager.get_all_sessions():
        await manager.end_session(session_info["session_id"], "server_shutdown")
    
    logger.info("✅ Cleanup complete")


# ==================== FastAPI App ====================

app = FastAPI(
    title="Bella Cucina Voice AI",
    description="Real-time voice AI assistant for restaurant reservations",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== Models ====================

class CallStartRequest(BaseModel):
    """Request to start a new voice call."""
    caller_name: str = Field(..., min_length=1, max_length=100, description="Caller's name")
    phone_number: Optional[str] = Field(None, description="Optional phone number")


class CallStartResponse(BaseModel):
    """Response after call is initiated."""
    session_id: str
    status: str
    websocket_url: str


class CallEndRequest(BaseModel):
    """Request to end a call."""
    reason: str = Field(default="user_ended", description="Reason for ending")


class BookingCreateRequest(BaseModel):
    """Request to create a booking directly (for testing/admin)."""
    name: str
    party_size: int = Field(..., ge=1, le=20)
    booking_date: str  # YYYY-MM-DD
    booking_time: str  # HH:MM
    notes: Optional[str] = None


class BookingResponse(BaseModel):
    """Booking response model."""
    id: int
    name: str
    party_size: int
    booking_date: str
    booking_time: str
    status: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    checks: dict


class StatsResponse(BaseModel):
    """Daily statistics response."""
    date: str
    total_calls: Optional[int] = 0
    completed_calls: Optional[int] = 0
    avg_duration: Optional[float] = None
    total_cost: Optional[float] = None
    total_bookings: Optional[int] = 0


# ==================== REST Endpoints ====================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    Returns status of all dependencies.
    """
    checker = get_health_checker()
    health = checker.check_health()
    
    # Correctly extract status and components from the health dict
    status = health.get("status", "unhealthy")
    return HealthResponse(
        status=status,
        checks=health.get("components", {})
    )


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Bella Cucina Voice AI",
        "version": "2.0.0",
        "docs": "/docs",
        "websocket": "/ws/call?name=<caller_name>",
    }


@app.get("/stats", response_model=StatsResponse)
async def get_stats(date: Optional[str] = None):
    """
    Get daily statistics.
    
    Args:
        date: Date in YYYY-MM-DD format (defaults to today)
    """
    stats = db.get_daily_stats(date)
    return StatsResponse(**stats)


@app.get("/calls")
async def list_calls(
    status: Optional[str] = None,
    limit: int = Query(default=50, le=100),
):
    """
    List recent calls with optional status filter.
    """
    # For now, just return active calls
    # In production, add proper pagination and filtering
    if status == "active":
        calls = db.get_active_calls()
    else:
        # Return last N calls - would need a new db function
        calls = db.get_active_calls()  # Simplified
    return {"calls": calls[:limit]}


@app.get("/calls/{call_id}")
async def get_call(call_id: str):
    """Get details of a specific call."""
    call = db.get_call(call_id)
    if not call:
        raise HTTPException(status_code=404, detail="Call not found")
    return call


@app.get("/calls/{call_id}/transcripts")
async def get_call_transcripts(call_id: str):
    """Get transcripts for a call."""
    transcripts = db.get_transcripts(call_id)
    return {"call_id": call_id, "transcripts": transcripts}


@app.get("/calls/{call_id}/analytics")
async def get_call_analytics(call_id: str):
    """Get analytics for a specific call."""
    analytics = db.get_call_analytics(call_id)
    if not analytics:
        raise HTTPException(status_code=404, detail="Analytics not found")
    return analytics


# ==================== Booking Endpoints ====================

@app.post("/bookings", response_model=BookingResponse)
async def create_booking(request: BookingCreateRequest):
    """Create a new booking (admin/testing endpoint)."""
    booking = db.create_booking(
        name=request.name,
        party_size=request.party_size,
        booking_date=request.booking_date,
        booking_time=request.booking_time,
        notes=request.notes,
    )
    return BookingResponse(**booking)


@app.get("/bookings", response_model=list[BookingResponse])
async def list_bookings(
    name: Optional[str] = None,
    date: Optional[str] = None,
):
    """
    List bookings with optional filters.
    
    Args:
        name: Filter by customer name
        date: Filter by booking date (YYYY-MM-DD)
    """
    bookings = db.find_bookings(name=name, date=date)
    return [BookingResponse(**b) for b in bookings]


@app.get("/bookings/{booking_id}", response_model=BookingResponse)
async def get_booking(booking_id: int):
    """Get a specific booking."""
    booking = db.get_booking(booking_id)
    if not booking:
        raise HTTPException(status_code=404, detail="Booking not found")
    return BookingResponse(**booking)


@app.delete("/bookings/{booking_id}")
async def cancel_booking(booking_id: int):
    """Cancel a booking."""
    success = db.cancel_booking(booking_id)
    if not success:
        raise HTTPException(status_code=404, detail="Booking not found")
    return {"status": "cancelled", "booking_id": booking_id}


# ==================== Active Sessions ====================

@app.get("/sessions")
async def list_active_sessions():
    """List all active voice sessions."""
    manager = get_voice_session_manager()
    return {
        "active_count": manager.active_session_count,
        "sessions": manager.get_all_sessions(),
    }


# ==================== WebSocket Endpoint ====================

@app.websocket("/ws/call")
async def voice_call_websocket(
    websocket: WebSocket,
    name: str = Query(..., min_length=1, description="Caller name"),
    phone: Optional[str] = Query(None, description="Phone number"),
):
    """
    WebSocket endpoint for voice calls.
    
    Protocol:
    1. Client connects with name query param
    2. Server sends status: "connected" with session_id
    3. Client sends binary audio data (LINEAR16, 16kHz, mono)
    4. Server sends:
       - Binary: TTS audio (LINEAR16, 24kHz, mono)
       - JSON: { type: "transcript", role: "user"|"assistant", content: "..." }
       - JSON: { type: "status", status: "...", message: "..." }
    5. Client can send JSON: { type: "end", reason: "user_ended" } to end call
    """
    await websocket.accept()
    
    manager = get_voice_session_manager()
    session: Optional[VoiceSession] = None
    
    try:
        # Create session
        session = await manager.create_session(
            websocket=websocket,
            caller_name=name,
            phone_number=phone,
        )
        
        # Start session (connects STT, sends greeting)
        started = await session.start()
        if not started:
            await websocket.close(code=1011, reason="Failed to start session")
            return
        
        logger.info(f"Voice call started: {session.session_id} for {name}")
        
        # Main message loop
        while True:
            try:
                message = await websocket.receive()
                
                if message["type"] == "websocket.disconnect":
                    break
                
                if "bytes" in message:
                    # Audio data from browser
                    await session.handle_audio(message["bytes"])
                    
                elif "text" in message:
                    # JSON control message
                    import json
                    data = json.loads(message["text"])
                    
                    if data.get("type") == "end":
                        reason = data.get("reason", "user_ended")
                        await session.end(reason)
                        break
                        
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Error in WebSocket loop: {e}")
                break
        
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        
    finally:
        # Cleanup
        if session:
            await manager.end_session(session.session_id, "disconnected")
        logger.info(f"Voice call ended: {session.session_id if session else 'unknown'}")


# ==================== Error Handlers ====================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# ==================== Run with Uvicorn ====================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("ENV", "development").lower() != "production",
        log_level="info",
    )
