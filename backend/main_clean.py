from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict
from contextlib import asynccontextmanager
import logging
from datetime import datetime
import uuid

from config_clean import get_settings, ensure_directories
from models_clean import CallTranscript, Ticket, Message, MessageRole, TicketStatus
from stt_service_clean import get_stt_service
from tts_service_clean import get_tts_service
from llm_service_clean import get_llm_service
from storage_service_clean import get_storage_service

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Active calls tracking
active_calls: Dict[str, CallTranscript] = {}

# Settings
settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown"""
    # Startup
    logger.info("🚀 Bella Cucina Voice Bot starting...")
    ensure_directories()
    
    # Initialize services
    try:
        stt_service = get_stt_service()
        tts_service = get_tts_service()
        llm_service = get_llm_service()
        storage_service = get_storage_service()
        
        app.state.stt_service = stt_service
        app.state.tts_service = tts_service
        app.state.llm_service = llm_service
        app.state.storage_service = storage_service
        
        logger.info(f"📡 Server: http://{settings.HOST}:{settings.PORT}")
        logger.info("✅ All services ready")
    except Exception as e:
        logger.error(f"❌ Failed to initialize services: {e}", exc_info=True)
        raise
    
    yield
    
    # Shutdown
    logger.info("👋 Shutting down...")

# Initialize app
app = FastAPI(
    title="Bella Cucina Voice Bot",
    description="AI Restaurant Assistant",
    version="2.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "service": "Bella Cucina Voice Bot",
        "version": "2.0.0",
        "status": "running"
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "active_calls": len(active_calls),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/session/start")
async def start_session():
    """Create new call session"""
    call_id = f"call_{uuid.uuid4().hex[:12]}"
    
    transcript = CallTranscript(
        call_id=call_id,
        start_time=datetime.now()
    )
    
    active_calls[call_id] = transcript
    
    logger.info(f"📞 New call started: {call_id}")
    
    return {
        "call_id": call_id,
        "status": "ready",
        "timestamp": transcript.start_time.isoformat()
    }

@app.get("/transcripts")
async def list_transcripts():
    """List all call transcripts"""
    call_ids = await app.state.storage_service.list_transcripts()
    return {"transcripts": call_ids, "count": len(call_ids)}

@app.get("/transcripts/{call_id}")
async def get_transcript(call_id: str):
    """Get specific call transcript"""
    transcript = await app.state.storage_service.load_transcript(call_id)
    if not transcript:
        raise HTTPException(status_code=404, detail="Transcript not found")
    return transcript

@app.get("/tickets")
async def list_tickets():
    """List all booking tickets"""
    tickets = await app.state.storage_service.list_tickets()
    return {"tickets": tickets, "count": len(tickets)}

@app.get("/tickets/{ticket_id}")
async def get_ticket(ticket_id: str):
    """Get specific ticket"""
    ticket = await app.state.storage_service.load_ticket(ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return ticket

@app.websocket("/ws/audio/{call_id}")
async def websocket_endpoint(websocket: WebSocket, call_id: str):
    """WebSocket endpoint for real-time voice conversation"""
    await websocket.accept()
    logger.info(f"🔌 WebSocket connected: {call_id}")
    
    # Get services from app state
    stt_service = app.state.stt_service
    tts_service = app.state.tts_service
    llm_service = app.state.llm_service
    storage_service = app.state.storage_service
    
    # Get or create transcript
    if call_id not in active_calls:
        transcript = CallTranscript(
            call_id=call_id,
            start_time=datetime.now()
        )
        active_calls[call_id] = transcript
    else:
        transcript = active_calls[call_id]
    
    # Audio buffer
    audio_buffer = bytearray()
    buffer_threshold = settings.SAMPLE_RATE * settings.AUDIO_BUFFER_SECONDS
    
    # Send greeting
    greeting = "Hello! I'm Maria from Bella Cucina. How can I help you today?"
    try:
        greeting_audio = await tts_service.synthesize(greeting)
        if greeting_audio:
            await websocket.send_bytes(greeting_audio)
            await websocket.send_json({
                "type": "transcript",
                "role": "assistant",
                "content": greeting
            })
            
            transcript.messages.append(Message(
                role=MessageRole.ASSISTANT,
                content=greeting
            ))
            
            logger.info(f"🤖 Maria: {greeting}")
    
    except Exception as e:
        logger.error(f"❌ Greeting error: {e}", exc_info=True)
    
    try:
        while True:
            data = await websocket.receive()
            
            if "bytes" in data:
                audio_chunk = data["bytes"]
                audio_buffer.extend(audio_chunk)
                
                # Process when buffer reaches threshold
                if len(audio_buffer) >= buffer_threshold:
                    logger.info(f"🎤 Processing audio buffer ({len(audio_buffer)} bytes)")
                    
                    # Transcribe
                    user_text = await stt_service.transcribe(bytes(audio_buffer))
                    audio_buffer.clear()
                    
                    if not user_text or len(user_text.strip()) < settings.MIN_TRANSCRIPTION_LENGTH:
                        continue
                    
                    logger.info(f"👤 User: {user_text}")
                    
                    # Add user message to transcript
                    transcript.messages.append(Message(
                        role=MessageRole.USER,
                        content=user_text
                    ))
                    
                    # Send transcript to client
                    await websocket.send_json({
                        "type": "transcript",
                        "role": "user",
                        "content": user_text
                    })
                    
                    # Generate response (pass history without current user message)
                    conversation_history = transcript.messages[:-1]
                    response_text = await llm_service.generate_response(
                        conversation_history,
                        user_text
                    )
                    
                    logger.info(f"🤖 Maria: {response_text}")
                    
                    # Add assistant message
                    transcript.messages.append(Message(
                        role=MessageRole.ASSISTANT,
                        content=response_text
                    ))
                    
                    # Synthesize and send
                    response_audio = await tts_service.synthesize(response_text)
                    if response_audio:
                        await websocket.send_bytes(response_audio)
                        await websocket.send_json({
                            "type": "transcript",
                            "role": "assistant",
                            "content": response_text
                        })
            
            elif "text" in data:
                message = data["text"]
                if message == "end_call":
                    logger.info(f"📴 Call ending: {call_id}")
                    break
    
    except WebSocketDisconnect:
        logger.info(f"🔌 Client disconnected: {call_id}")
    
    except Exception as e:
        logger.error(f"❌ WebSocket error: {e}", exc_info=True)
    
    finally:
        # Finalize call
        await finalize_call(call_id, transcript, llm_service, storage_service)

async def finalize_call(
    call_id: str,
    transcript: CallTranscript,
    llm_service,
    storage_service
):
    """Finalize call: generate summary, detect booking, rate, and save"""
    transcript.end_time = datetime.now()
    transcript.duration_seconds = (
        transcript.end_time - transcript.start_time
    ).total_seconds()
    
    logger.info(f"📊 Processing call summary for {call_id}...")
    
    # Minimum messages: greeting + user + response = 3
    if len(transcript.messages) > 2:
        transcript.summary = await llm_service.generate_summary(transcript.messages)
        
        # Detect booking
        booking_intent = await llm_service.detect_booking_intent(transcript.messages)
        
        # Create ticket if booking detected with complete info
        if booking_intent.has_booking_intent and all([
            booking_intent.customer_name,
            booking_intent.date,
            booking_intent.time,
            booking_intent.party_size
        ]):
            ticket = Ticket(
                ticket_id=f"ticket_{uuid.uuid4().hex[:8]}",
                call_id=call_id,
                customer_name=booking_intent.customer_name,
                phone=booking_intent.phone,
                date=booking_intent.date,
                time=booking_intent.time,
                party_size=booking_intent.party_size,
                special_requests=booking_intent.special_requests,
                status=TicketStatus.PENDING
            )
            
            await storage_service.save_ticket(ticket)
            transcript.booking_created = True
            logger.info(f"🎫 Ticket created: {ticket.ticket_id}")
        
        # Rate call
        transcript.rating = await llm_service.rate_call(
            transcript.messages,
            transcript.booking_created
        )
    
    # Save transcript
    await storage_service.save_transcript(transcript)
    
    # Remove from active calls
    if call_id in active_calls:
        del active_calls[call_id]
    
    logger.info(f"✅ Call completed: {call_id} (⭐{transcript.rating}/5)")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.HOST,
        port=settings.PORT,
        log_level="info"
    )
