from typing import List, Dict, Any, Optional, TypedDict
from datetime import datetime, timedelta
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langgraph.graph.message import add_messages

# LangGraph operates on a state dictionary. We define its structure here.
class SessionState(TypedDict):
    call_id: str
    session_id: str
    messages: List[BaseMessage]
    # Any other custom state can be added here
    metadata: Dict[str, Any]

# In-memory storage for active conversation states.
_active_sessions: Dict[str, SessionState] = {}
_session_last_updated: Dict[str, datetime] = {}

def get_session_state(call_id: str, session_id: str) -> SessionState:
    """Retrieves or creates a session state."""
    if session_id not in _active_sessions:
        _active_sessions[session_id] = {
            "call_id": call_id,
            "session_id": session_id,
            "messages": [],
            "metadata": {}
        }
    _session_last_updated[session_id] = datetime.utcnow()
    return _active_sessions[session_id]

def update_session_state(session_id: str, new_state: dict):
    """Updates the state for a given session using LangGraph's 'add_messages'."""
    if session_id in _active_sessions:
        # Use add_messages to correctly append new messages to the history
        current_state = _active_sessions[session_id]
        updated_state = add_messages(current_state, new_state)
        _active_sessions[session_id] = updated_state
        _session_last_updated[session_id] = datetime.utcnow()

def get_full_state(session_id: str) -> Optional[SessionState]:
    """Gets the full state dictionary for a session."""
    return _active_sessions.get(session_id)
    
def end_session(session_id: str) -> Optional[SessionState]:
    """Removes a session from the active store."""
    if session_id in _active_sessions:
        _session_last_updated.pop(session_id, None)
        return _active_sessions.pop(session_id, None)
    return None

def cleanup_old_sessions(max_age_hours: int = 24):
    """Removes sessions that have not been updated for a certain time."""
    cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
    sessions_to_remove = [
        sid for sid, last_updated in _session_last_updated.items() 
        if last_updated < cutoff
    ]
    
    if sessions_to_remove:
        print(f"🧹 Cleaning up {len(sessions_to_remove)} old sessions...")
        for sid in sessions_to_remove:
            end_session(sid)
    else:
        print("🧹 No old sessions to clean up.")
