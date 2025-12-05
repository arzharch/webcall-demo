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

from typing import Dict, List  # ADD List HERE
from backend.models import SessionState

# In-memory conversation storage
_active_conversations: Dict[str, SessionState] = {}

def get_conversation(call_id: str) -> SessionState:
    """Get or create conversation context"""
    if call_id not in _active_conversations:
        _active_conversations[call_id] = SessionState(call_id=call_id)
    return _active_conversations[call_id]

def end_conversation(call_id: str) -> SessionState:
    """End and return conversation context"""
    context = _active_conversations.get(call_id)
    if context:
        # Keep in memory for now (can add cleanup later)
        pass
    return context

def list_active_conversations() -> List[str]:
    """List all active conversation IDs"""
    return list(_active_conversations.keys())

def cleanup_old_conversations(max_age_hours: int = 24) -> int:
    """Clean up old conversations"""
    from datetime import datetime, timedelta
    
    cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
    to_remove = [
        call_id for call_id, ctx in _active_conversations.items()
        if ctx.started_at < cutoff
    ]
    
    for call_id in to_remove:
        del _active_conversations[call_id]
    
    return len(to_remove)

