from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from models import Message, BookingIntent

class SessionState(BaseModel):
    """Represents the complete state of a single conversation session."""
    call_id: str
    session_id: str
    messages: List[Message] = Field(default_factory=list)
    booking_intent: BookingIntent = Field(default_factory=BookingIntent)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    last_updated: datetime = Field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """LangChain compatible dictionary representation."""
        return {"messages": self.messages, "booking_intent": self.booking_intent, **self.metadata}

# In-memory storage for active conversation states.
_active_sessions: Dict[str, SessionState] = {}

def get_session_state(call_id: str, session_id: str) -> SessionState:
    """Retrieves or creates a session state."""
    if session_id not in _active_sessions:
        _active_sessions[session_id] = SessionState(call_id=call_id, session_id=session_id)
    session = _active_sessions[session_id]
    session.last_updated = datetime.utcnow()
    return session

def update_session_state(session_id: str, state: SessionState):
    """Updates the state for a given session."""
    state.last_updated = datetime.utcnow()
    _active_sessions[session_id] = state

def end_session(session_id: str) -> Optional[SessionState]:
    """Removes a session from the active store."""
    if session_id in _active_sessions:
        return _active_sessions.pop(session_id)
    return None

def cleanup_old_sessions(max_age_hours: int = 24):
    """Removes sessions that have not been updated for a certain time."""
    cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
    sessions_to_remove = [
        sid for sid, state in _active_sessions.items() 
        if state.last_updated < cutoff
    ]
    
    if sessions_to_remove:
        print(f"🧹 Cleaning up {len(sessions_to_remove)} old sessions...")
        for sid in sessions_to_remove:
            del _active_sessions[sid]
    else:
        print("🧹 No old sessions to clean up.")
