import os
import uuid
from fastapi import FastAPI, WebSocket, Depends
from starlette.websockets import WebSocketDisconnect
from dotenv import load_dotenv

from agent.agent import BellaAgent
from agent.state import SessionState

# Load environment variables
load_dotenv()

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Bella Cucina Voice Bot",
    description="A streaming, agentic voice bot for a restaurant.",
    version="1.0.0",
)

# --- In-Memory State Management ---
# (In a production scenario, use Redis or a database)
active_sessions = {}

# --- Dependencies ---

def get_agent() -> BellaAgent:
    """Dependency to get a singleton instance of the BellaAgent."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")
    # This could be enhanced to use a singleton pattern
    return BellaAgent(google_api_key=api_key)

# --- REST Endpoint for Session Initiation ---

@app.post("/session/start")
async def start_session():
    """Starts a new conversation session and returns a unique session ID."""
    session_id = str(uuid.uuid4())
    initial_state = SessionState(session_id=session_id)
    active_sessions[session_id] = initial_state
    return {"session_id": session_id}

# --- WebSocket Endpoint for Conversation ---

@app.websocket("/ws/audio/{session_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    session_id: str,
    agent: BellaAgent = Depends(get_agent),
):
    """Handles the main audio conversation loop."""
    await websocket.accept()
    
    if session_id not in active_sessions:
        await websocket.close(code=1008, reason="Invalid session_id")
        return

    state = active_sessions[session_id]
    
    try:
        while True:
            # For this phase, we'll receive text instead of audio bytes
            user_message = await websocket.receive_text()

            # Process message with the agent and stream response
            async for response_chunk in agent.process_message(state, user_message):
                await websocket.send_text(response_chunk)
            
            # Update history after the full response is sent
            # Note: A more sophisticated approach would be needed to get the full AI response
            # when streaming token-by-token. For now, we assume the last chunk is the full message.
            state.update_history(user_message, "AI response will be logged here.") # Placeholder

    except WebSocketDisconnect:
        print(f"Client disconnected from session {session_id}")
        # Clean up the session
        if session_id in active_sessions:
            del active_sessions[session_id]
    except Exception as e:
        print(f"An error occurred in session {session_id}: {e}")
        # Clean up the session
        if session_id in active_sessions:
            del active_sessions[session_id]
        await websocket.close(code=1011, reason="Internal Server Error")
