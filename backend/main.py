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
    """WebSocket endpoint for audio streaming"""
    await websocket.accept()
    
    session = get_conversation(call_id)
    orchestrator = ConversationOrchestrator(session)
    
    logger.info(f"WebSocket connected: {call_id}")
    
    # Queues and Events for managing concurrent tasks
    user_audio_queue = asyncio.Queue()
    bot_response_queue = asyncio.Queue()
    interruption_event = asyncio.Event()

    async def audio_receiver():
        """Receives audio from the client, performs VAD, and queues speech for transcription."""
        audio_buffer = []
        try:
            while True:
                data = await websocket.receive_bytes()
                audio_buffer.append(data)

                # Simple interruption: if the bot is talking, signal an interruption
                if not bot_response_queue.empty():
                    interruption_event.set()
                    print("🎤 User interruption detected.")

                # VAD processing
                # This is a simplified VAD logic. A more robust implementation
                # would use precise speech start/end timestamps from `get_speech_ts`.
                audio_int16 = np.frombuffer(b"".join(audio_buffer), dtype=np.int16)
                audio_float32 = audio_int16.astype(np.float32) / 32768.0
                speech_prob = vad_model(torch.from_numpy(audio_float32), settings.SAMPLE_RATE).item()

                if speech_prob < 0.5: # Threshold for silence
                    if len(audio_buffer) > 1: # Avoid transcribing tiny silent chunks
                        full_audio = b"".join(audio_buffer)
                        await user_audio_queue.put(full_audio)
                    audio_buffer.clear()

        except WebSocketDisconnect:
            print("Receiver: Client disconnected.")
            await user_audio_queue.put(None) # Signal end to other tasks
        except Exception as e:
            print(f"Receiver Error: {e}")
            await user_audio_queue.put(None)

    async def response_handler():
        """Processes user speech, gets a response from the agent, and queues it for sending."""
        try:
            while True:
                audio_to_transcribe = await user_audio_queue.get()
                if audio_to_transcribe is None:
                    break # End signal

                user_text = await stt_service.transcribe_audio(audio_to_transcribe)
                if user_text:
                    state.add_message(role="user", content=user_text)
                    
                    # Get a streaming response from the orchestrator
                    text_stream = orchestrator.stream_response(state, user_text)
                    
                    # We put the entire stream generator into the queue
                    await bot_response_queue.put(text_stream)

                user_audio_queue.task_done()
        except Exception as e:
            print(f"Handler Error: {e}")

    async def audio_sender():
        """Streams the bot's audio response to the client, handling interruptions."""
        try:
            # Initial Greeting
            initial_greeting = "Welcome to Bella Cucina. How can I help?"
            state.add_message(role="assistant", content=initial_greeting)
            async def initial_stream(): yield initial_greeting
            audio_stream = tts_service.synthesize_streaming(initial_stream())
            async for audio_chunk in audio_stream:
                if interruption_event.is_set():
                    print("Sender: Interruption detected, stopping greeting.")
                    break
                await websocket.send_bytes(audio_chunk)

            # Main response loop
            while True:
                interruption_event.clear()
                text_stream_generator = await bot_response_queue.get()
                
                full_response = ""
                audio_stream = tts_service.synthesize_streaming(text_stream_generator)
                async for audio_chunk in audio_stream:
                    if interruption_event.is_set():
                        print("Sender: Interruption detected, stopping playback.")
                        # Clear the rest of the current response queue
                        while not bot_response_queue.empty():
                            bot_response_queue.get_nowait()
                        break
                    await websocket.send_bytes(audio_chunk)
                
                # The full text is not easily available here, this would require another refactor
                # For now, we save an incomplete message.
                if not interruption_event.is_set():
                     state.add_message(role="assistant", content="<streaming_response>")
                
                bot_response_queue.task_done()
                if user_audio_queue.empty() and websocket.client_state != 1:
                    break

        except Exception as e:
            print(f"Sender Error: {e}")

    # Run the concurrent tasks
    receiver_task = asyncio.create_task(audio_receiver())
    handler_task = asyncio.create_task(response_handler())
    sender_task = asyncio.create_task(audio_sender())

    await asyncio.gather(receiver_task, handler_task, sender_task)
    
    print(f"🛑 WebSocket connection closing for call: {call_id}")
    end_conversation(call_id)

# --- Uvicorn Runner ---
if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
