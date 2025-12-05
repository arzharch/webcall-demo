import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import uvicorn
from typing import List
import torch
import numpy as np

# Use relative imports from backend root
from config import get_settings
from models import SessionState, MessageRole
from agent.state import get_conversation, end_conversation
from agent.orchestrator import ConversationOrchestrator
from services.stt_service import get_stt_service
from services.tts_service import get_tts_service
from services.crm_service import get_crm_service
from services.rag_service import get_rag_service

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Background Tasks ---
async def session_cleanup_task():
    """Periodically cleans up old sessions to prevent memory leaks."""
    while True:
        await asyncio.sleep(3600) # Run every hour
        cleanup_old_sessions()

# --- App Initialization & Middleware ---
settings = get_settings()
app = FastAPI(
    title="Bella Cucina Voice Bot",
    description="AI-powered restaurant reservation assistant",
    version="2.0.0"
)
app.add_middleware(CORSMiddleware, allow_origins=settings.CORS_ORIGINS, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- Service and Model Loading ---
@app.on_event("startup")
async def startup():
    """Initialize services on startup"""
    global _stt_service, _tts_service, _crm_service, _rag_service
    
    try:
        logger.info("Initializing services...")
        
        # Initialize each service
        _stt_service = get_stt_service()
        await _stt_service.initialize()
        
        _tts_service = get_tts_service()
        await _tts_service.initialize()
        
        _crm_service = get_crm_service()
        await _crm_service.initialize()
        
        _rag_service = get_rag_service()
        await _rag_service.initialize()
        
        logger.info(f"Services ready: STT, TTS, CRM, RAG")
    except Exception as e:
        logger.error(f"Startup error: {e}")
        # Don't raise - allow graceful degradation

    print("🚀 Server starting up...")
    # Using asyncio.gather to initialize services concurrently
    await asyncio.gather(
        get_rag_service().initialize(),
        get_crm_service().initialize(),
        get_stt_service().initialize(),
        get_tts_service().initialize()
    )
    get_orchestrator()
    # Load VAD model
    global vad_model, vad_utils
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
    vad_model = model
    vad_utils = utils
    
    # Start background tasks
    asyncio.create_task(session_cleanup_task())
    
    print("✅ All services and background tasks are running.")

# Pre-warm models on startup to avoid first-call latency
@app.on_event("startup")
async def warmup_models():
    logger.info("🔥 Warming up models...")
    
    # Warm up TTS
    tts = get_tts_service()
    await tts.synthesize("Hello")
    
    # Warm up STT
    stt = get_stt_service()
    # Create dummy audio file for warm-up
    
    # Warm up LLM
    orchestrator = get_orchestrator()
    async for _ in orchestrator.stream_response("warmup", "test"):
        break
    
    logger.info("✅ Models ready!")

# --- REST Endpoints ---
@app.get("/", tags=["Health"])
async def health_check():
    return {"status": "ok", "message": "Welcome to the Voice Bot Streaming API!"}

@app.get("/tickets", response_model=List[Ticket], tags=["CRM"])
async def list_tickets():
    crm_service = get_crm_service()
    return await crm_service.list_tickets()

@app.post("/session/start")
async def start_session():
    """Start a new conversation session"""
    try:
        session = SessionState()
        
        return {
            "call_id": session.call_id,
            "session_id": session.session_id,
            "status": "active",
            "message": "Session started. Please begin speaking."
        }
    except Exception as e:
        logger.error(f"Session start error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Streaming WebSocket Endpoint ---
@app.websocket("/ws/audio/{call_id}")
async def websocket_audio_endpoint(websocket: WebSocket, call_id: str):
    """Complete voice conversation flow"""
    await websocket.accept()
    logger.info(f"🎙️ WebSocket connected: {call_id}")
    
    # Get services
    stt = get_stt_service()
    tts = get_tts_service()
    orchestrator = get_orchestrator()
    
    # Initialize session
    session_id = f"session_{call_id}"
    session = get_conversation(call_id)
    
    # Send welcome message
    welcome_text = "Hello! I'm Maria from Bella Cucina. How can I help you today?"
    welcome_audio = await tts.synthesize(welcome_text)
    await websocket.send_bytes(welcome_audio)
    
    try:
        while True:
            # Receive audio from client
            audio_data = await websocket.receive_bytes()
            
            # Transcribe
            transcript = await stt.transcribe(audio_data)
            
            if not transcript or len(transcript.strip()) < 3:
                continue  # Ignore very short/empty transcripts
            
            logger.info(f"User said: {transcript}")
            
            # Send transcript back to frontend
            await websocket.send_json({
                "type": "transcript",
                "role": "user",
                "content": transcript
            })
            
            # Stream response from agent
            full_response = ""
            async for sentence in orchestrator.stream_response(session_id, transcript):
                full_response += sentence + " "
                
                # Synthesize sentence
                audio_chunk = await tts.synthesize(sentence)
                
                # Send audio to client
                await websocket.send_bytes(audio_chunk)
                
                # Also send text transcript
                await websocket.send_json({
                    "type": "transcript",
                    "role": "assistant",
                    "content": sentence
                })
            
            logger.info(f"Assistant said: {full_response}")
    
    except WebSocketDisconnect:
        logger.info(f"Client disconnected: {call_id}")
        end_conversation(call_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        try:
            await websocket.close(code=1011, reason=str(e))
        except:
            pass

# --- Uvicorn Runner ---
if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
