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
    await websocket.accept()
    print(f"🎙️ WebSocket connection established for call: {call_id}")

    session_id = f"session_{call_id}"
    state = get_session_state(call_id, session_id)
    
    stt_service = get_stt_service()
    tts_service = get_tts_service()
    orchestrator = get_orchestrator()

    # Queues and Events for managing concurrent tasks
    user_audio_queue = asyncio.Queue()
    bot_response_queue = asyncio.Queue()
    interruption_event = asyncio.Event()

    async def audio_receiver():
        """Receives audio from the client, performs VAD, and queues speech for transcription."""
        audio_buffer = b""
        # VAD specific components
        # (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = vad_utils
        vad_iterator = vad_utils.VADIterator(vad_model)
        
        # Adjust constants for VAD based on sample rate and chunk size (default 16kHz VAD model)
        # Assuming `data` chunks are around 1 second or less.
        MIN_SILENCE_DURATION_MS = 200 # milliseconds of silence to consider speech end
        SPEECH_THRESHOLD = 0.5 # probability threshold for speech detection

        try:
            while True:
                data = await websocket.receive_bytes()
                
                # Convert bytes to float32 numpy array for VAD
                audio_float32 = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Process audio chunk with VADIterator
                speech_prob = vad_iterator(audio_float32, return_seconds=True)
                
                if speech_prob is not None:
                    # speech_prob is a float indicating speech probability
                    # A robust VAD should use `VADIterator` output, not just probability on a single chunk.
                    # This is still a simplified VAD. For precise speech timestamps:
                    # - Accumulate audio in `audio_buffer` until `vad_iterator.reset_states()` indicates end of speech
                    # - Then put the full segment into `user_audio_queue`

                    # Simplified VAD for now: If any speech activity, consider user speaking
                    if speech_prob > SPEECH_THRESHOLD:
                        audio_buffer += data
                        # If bot is talking, interrupt it
                        if not bot_response_queue.empty():
                            interruption_event.set()
                            print("🎤 User interruption detected.")
                    elif len(audio_buffer) > 0 and speech_prob < (1 - SPEECH_THRESHOLD):
                        # User is silent and we have accumulated audio, process it
                        full_audio_chunk = audio_buffer
                        await user_audio_queue.put(full_audio_chunk)
                        audio_buffer = b"" # Reset buffer
                else: # VADIterator can return None if it's still buffering or no decision yet
                    audio_buffer += data

        except WebSocketDisconnect:
            print("Receiver: Client disconnected.")
            await user_audio_queue.put(None) # Signal end to other tasks
        except Exception as e:
            print(f"Receiver Error: {e}")
            await user_audio_queue.put(None)

    async def response_handler():
        """Processes user speech, gets a streaming response from the agent, and queues it for sending."""
        try:
            while True:
                audio_to_transcribe = await user_audio_queue.get()
                if audio_to_transcribe is None:
                    break # End signal

                user_text = await stt_service.transcribe_audio(audio_to_transcribe)
                if user_text:
                    state.add_message(role="user", content=user_text)
                    
                    # Get a streaming response from the orchestrator
                    text_stream_generator = orchestrator.stream_response(state, user_text)
                    
                    # We put the entire stream generator into the queue
                    await bot_response_queue.put(text_stream_generator)

                user_audio_queue.task_done()
        except Exception as e:
            print(f"Handler Error: {e}")

    async def audio_sender():
        """Streams the bot's audio response to the client, handling interruptions."""
        try:
            # Initial Greeting
            initial_greeting = "Welcome to Bella Cucina. How can I help?"
            state.add_message(role="assistant", content=initial_greeting)
            async def initial_stream_gen(): yield initial_greeting
            audio_stream = tts_service.synthesize_streaming(initial_stream_gen())
            async for audio_chunk in audio_stream:
                if interruption_event.is_set():
                    print("Sender: Interruption detected, stopping greeting.")
                    break
                await websocket.send_bytes(audio_chunk)

            # Main response loop
            while True:
                interruption_event.clear() # Clear event for the new response
                text_stream_generator = await bot_response_queue.get()
                
                collected_text = "" # To store the full bot response
                audio_stream = tts_service.synthesize_streaming(text_stream_generator)
                async for audio_chunk in audio_stream:
                    if interruption_event.is_set():
                        print("Sender: Interruption detected, stopping playback.")
                        # Clear the rest of the current response queue
                        while not bot_response_queue.empty():
                            bot_response_queue.get_nowait()
                        break
                    await websocket.send_bytes(audio_chunk)
                
                # After sending audio, collect the full text that was sent
                # This requires a way to collect the text from text_stream_generator.
                # A more robust solution might involve sending (text, audio_chunk) pairs
                # from the orchestrator to the bot_response_queue.
                # For now, we save an incomplete message.
                if not interruption_event.is_set():
                    state.add_message(role="assistant", content=collected_text if collected_text else "<streaming_response_completed>")
                
                bot_response_queue.task_done()
                if user_audio_queue.empty() and websocket.client_state != 1:
                    break # Consider breaking only if session truly ended.

        except Exception as e:
            print(f"Sender Error: {e}")

    # Run the concurrent tasks
    receiver_task = asyncio.create_task(audio_receiver())
    handler_task = asyncio.create_task(response_handler())
    sender_task = asyncio.create_task(audio_sender())

    try:
        await asyncio.gather(receiver_task, handler_task, sender_task)
    except Exception as e:
        print(f"Tasks gathering error: {e}")
    finally:
        print(f"🛑 WebSocket connection closing for call: {call_id}")
        end_session(session_id)

# --- Uvicorn Runner ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)
