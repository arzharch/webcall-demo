import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import torch
import numpy as np

from config import get_settings
from models import Ticket
from agent.state import get_session_state, end_session, cleanup_old_sessions
from agent.orchestrator import get_orchestrator
from services.rag_service import get_rag_service
from services.crm_service import get_crm_service
from services.stt_service import get_stt_service
from services.tts_service import get_tts_service

# --- App Initialization & Middleware ---
settings = get_settings()
app = FastAPI(title="Bella Cucina - Voice Bot API v2 (Streaming)")
app.add_middleware(CORSMiddleware, allow_origins=settings.CORS_ORIGINS, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- Background Tasks ---
async def session_cleanup_task():
    """Periodically cleans up old sessions to prevent memory leaks."""
    while True:
        await asyncio.sleep(3600) # Run every hour
        cleanup_old_sessions()

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

# --- Streaming WebSocket Endpoint ---
@app.websocket("/ws/audio/{call_id}")
async def websocket_audio_endpoint(websocket: WebSocket, call_id: str):
    """
    Handles the main voice conversation flow with real-time streaming,
    interruption, and stateful VAD.
    """
    await websocket.accept()
    print(f"🎙️ WebSocket connection established for call: {call_id}")

    session_id = f"session_{call_id}"
    stt_service = get_stt_service()
    tts_service = get_tts_service()
    orchestrator = get_orchestrator()

    # Queues and Events for managing concurrent tasks
    user_speech_queue = asyncio.Queue()
    bot_response_queue = asyncio.Queue()
    interruption_event = asyncio.Event()

    async def audio_receiver():
        """
        Receives audio from the client, performs stateful VAD, and queues speech for transcription.
        """
        vad_iterator = vad_utils.VADIterator(vad_model)
        is_speaking = False
        audio_buffer = []
        # VAD Parameters
        MIN_SILENCE_FRAMES = 15  # Corresponds to ~480ms of silence
        SPEECH_THRESHOLD = 0.5
        silence_frames = 0
        
        try:
            while True:
                data = await websocket.receive_bytes()
                # The VAD model expects 16kHz audio chunks. 
                # The size of `data` depends on the client's MediaRecorder `timeslice`.
                # Assuming client sends chunks of 320-1600 bytes for 20-100ms latency.
                audio_float32 = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                
                speech_prob = vad_iterator(torch.from_numpy(audio_float32), return_seconds=False)
                
                if speech_prob['speech'] > SPEECH_THRESHOLD:
                    # User is speaking
                    silence_frames = 0
                    if not is_speaking:
                        is_speaking = True
                        print("🎤 User speech started.")
                        if not bot_response_queue.empty():
                             interruption_event.set()
                             print("   - Interruption detected!")
                    audio_buffer.append(data)
                else:
                    # User is silent
                    if is_speaking:
                        silence_frames += 1
                        if silence_frames > MIN_SILENCE_FRAMES:
                            is_speaking = False
                            print("🎤 User speech ended.")
                            full_audio_chunk = b"".join(audio_buffer)
                            await user_speech_queue.put(full_audio_chunk)
                            audio_buffer.clear()
                            silence_frames = 0
        except WebSocketDisconnect:
            print("Receiver: Client disconnected.")
            await user_speech_queue.put(None)
        except Exception as e:
            print(f"Receiver Error: {e}")
            await user_speech_queue.put(None)

    async def response_handler():
        """
        Processes transcribed user speech, gets a streaming response from the
        LangGraph agent, and puts the response stream and full text into a queue.
        """
        try:
            while True:
                audio_to_transcribe = await user_speech_queue.get()
                if audio_to_transcribe is None:
                    await bot_response_queue.put(None) # Propagate end signal
                    break

                user_text = await stt_service.transcribe_audio(audio_to_transcribe)
                if user_text:
                    # Update state with user message
                    update_session_state(session_id, {"messages": [HumanMessage(content=user_text)]})
                    
                    # Get a streaming response from the orchestrator
                    text_stream_generator = orchestrator.stream_response(session_id, user_text)
                    
                    # Queue the response generator for the sender task
                    await bot_response_queue.put(text_stream_generator)

                user_speech_queue.task_done()
        except Exception as e:
            print(f"Handler Error: {e}")

    async def audio_sender():
        """
        Streams the bot's audio response to the client, handling interruptions and state updates.
        """
        try:
            # Initial Greeting
            initial_greeting = "Welcome to Bella Cucina. How can I help?"
            update_session_state(session_id, {"messages": [AIMessage(content=initial_greeting)]})
            async def initial_stream_gen(): yield initial_greeting
            audio_stream = tts_service.synthesize_streaming(initial_stream_gen())
            async for audio_chunk in audio_stream:
                await websocket.send_bytes(audio_chunk)

            # Main response loop
            while True:
                response_stream = await bot_response_queue.get()
                if response_stream is None:
                    break # End signal
                
                interruption_event.clear()
                
                # We need to consume the text stream to get the full response for history,
                # while also passing it to the TTS service.
                async def text_tee():
                    full_response = ""
                    async for chunk in response_stream:
                        full_response += chunk
                        yield chunk
                    # After streaming is done, update the state with the full response
                    update_session_state(session_id, {"messages": [AIMessage(content=full_response)]})
                
                text_generator = text_tee()
                audio_stream = tts_service.synthesize_streaming(text_generator)

                async for audio_chunk in audio_stream:
                    if interruption_event.is_set():
                        print("Sender: Interruption detected, stopping playback.")
                        # Consume the rest of the audio stream to allow cleanup
                        async for _ in audio_stream: pass
                        break
                    await websocket.send_bytes(audio_chunk)
                
                bot_response_queue.task_done()

        except Exception as e:
            print(f"Sender Error: {e}")

    # Run the concurrent tasks
    tasks = [audio_receiver(), response_handler(), audio_sender()]
    try:
        await asyncio.gather(*tasks)
    except Exception as e:
        print(f"Core WebSocket task gathering error: {e}")
    finally:
        print(f"🛑 WebSocket connection closing for call: {call_id}")
        end_session(session_id)

# --- Uvicorn Runner ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)
