import asyncio
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import List

from config import get_settings
from models import Ticket
from agent.state import get_session_state, end_session, cleanup_old_sessions
from agent.orchestrator import get_orchestrator
from services.rag_service import get_rag_service
from services.crm_service import get_crm_service
from services.stt_service import get_stt_service
from services.tts_service import get_tts_service

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- App Initialization & Middleware ---
settings = get_settings()
app = FastAPI(title="Bella Cucina - Voice Bot API")
app.add_middleware(
    CORSMiddleware, 
    allow_origins=settings.CORS_ORIGINS, 
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"]
)

# --- Background Tasks ---
async def session_cleanup_task():
    """Periodically cleans up old sessions to prevent memory leaks."""
    while True:
        await asyncio.sleep(3600)
        cleanup_old_sessions()
        print("🧹 Cleaned up old sessions")

# --- Service and Model Loading ---
@app.on_event("startup")
async def startup_event():
    print("🚀 Server starting up...")
    await asyncio.gather(
        get_rag_service().initialize(),
        get_crm_service().initialize(),
        get_stt_service().initialize(),
        get_tts_service().initialize()
    )
    get_orchestrator()
    asyncio.create_task(session_cleanup_task())
    print("✅ All services initialized and ready!")

# --- REST Endpoints ---
@app.get("/", tags=["Health"])
async def health_check():
    return {"status": "ok", "message": "Bella Cucina Voice Bot API"}

@app.get("/tickets", response_model=List[Ticket], tags=["CRM"])
async def list_tickets():
    crm_service = get_crm_service()
    return await crm_service.list_tickets()

@app.post("/session/start")
async def start_session():
    """Create new session"""
    call_id = f"call_{int(asyncio.get_event_loop().time() * 1000)}"
    return {"call_id": call_id, "status": "ready"}

# --- WebSocket Endpoint ---
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
    
    # Send welcome message
    welcome_text = "Hello! I'm Maria from Bella Cucina. How can I help you today?"
    try:
        welcome_audio = await tts.synthesize(welcome_text)
        await websocket.send_bytes(welcome_audio)
        await websocket.send_json({
            "type": "transcript",
            "role": "assistant",
            "content": welcome_text
        })
        logger.info(f"🤖 Maria: {welcome_text}")
    except Exception as e:
        logger.error(f"Failed to send welcome: {e}")
    
    try:
        while True:
            # Receive audio from client
            audio_data = await websocket.receive_bytes()
            
            if len(audio_data) < 1000:
                continue
            
            # Transcribe
            transcript = await stt.transcribe(audio_data)
            
            if not transcript or len(transcript.strip()) < 3:
                continue
            
            logger.info(f"👤 User: {transcript}")
            
            # Send transcript to frontend
            await websocket.send_json({
                "type": "transcript",
                "role": "user",
                "content": transcript
            })
            
            # Get AI response
            try:
                response = await orchestrator.stream_response(session_id, transcript)
                
                # Handle response - could be string or async generator
                if hasattr(response, '__aiter__'):
                    # Async generator - stream responses
                    full_response = ""
                    async for sentence in response:
                        if sentence and isinstance(sentence, str) and sentence.strip():
                            full_response += sentence + " "
                            
                            # Synthesize and send
                            audio_chunk = await tts.synthesize(sentence)
                            await websocket.send_bytes(audio_chunk)
                            
                            # Send transcript
                            await websocket.send_json({
                                "type": "transcript",
                                "role": "assistant",
                                "content": sentence
                            })
                    
                    logger.info(f"🤖 Maria: {full_response.strip()}")
                
                elif isinstance(response, str):
                    # Single string response
                    audio_chunk = await tts.synthesize(response)
                    await websocket.send_bytes(audio_chunk)
                    
                    await websocket.send_json({
                        "type": "transcript",
                        "role": "assistant",
                        "content": response
                    })
                    
                    logger.info(f"🤖 Maria: {response}")
                
                else:
                    logger.error(f"Unexpected response type: {type(response)}")
            
            except Exception as e:
                logger.error(f"Error processing response: {e}", exc_info=True)
                error_msg = "I apologize, I'm having trouble right now. Could you repeat that?"
                error_audio = await tts.synthesize(error_msg)
                await websocket.send_bytes(error_audio)
                await websocket.send_json({
                    "type": "transcript",
                    "role": "assistant",
                    "content": error_msg
                })
    
    except WebSocketDisconnect:
        logger.info(f"Client disconnected: {call_id}")
        end_session(session_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        try:
            await websocket.close(code=1011, reason=str(e)[:100])
        except:
            pass

# --- Uvicorn Runner ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)