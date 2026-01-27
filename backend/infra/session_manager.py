"""
Session persistence for call transfers and multi-instance deployments.
Enables warm handoffs between agents and session recovery after failures.
"""
import json
import time
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field, asdict
from datetime import datetime
import uuid

from infra.redis_cache import get_redis_client, REDIS_AVAILABLE
from infra.config import config

logger = logging.getLogger(__name__)


@dataclass
class CallSession:
    """
    Persistent call session state for transfers and recovery.
    """
    session_id: str
    caller_name: str
    phone_number: Optional[str] = None
    
    # Conversation state
    current_intent: Optional[str] = None
    booking_slots: Dict[str, Any] = field(default_factory=dict)
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    
    # Call metadata
    started_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    turn_count: int = 0
    
    # Transfer state
    transfer_requested: bool = False
    transfer_reason: Optional[str] = None
    transfer_target: Optional[str] = None  # "human", "specialist", "callback"
    
    # Quality metrics
    interruption_count: int = 0
    loop_count: int = 0
    error_count: int = 0
    
    def to_dict(self) -> dict:
        """Serialize session to dict for Redis storage."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "CallSession":
        """Deserialize session from dict."""
        return cls(**data)
    
    def add_turn(self, role: str, content: str):
        """Add a conversation turn and update metrics."""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": time.time()
        })
        self.turn_count += 1
        self.last_activity = time.time()
        
        # Keep only last 20 turns to limit memory
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
    
    def get_context_summary(self) -> str:
        """Generate a summary for transfer handoff."""
        summary_parts = [f"Caller: {self.caller_name}"]
        
        if self.current_intent:
            summary_parts.append(f"Intent: {self.current_intent}")
        
        if self.booking_slots:
            slots_str = ", ".join(f"{k}: {v}" for k, v in self.booking_slots.items() if v)
            if slots_str:
                summary_parts.append(f"Booking Details: {slots_str}")
        
        if self.transfer_reason:
            summary_parts.append(f"Transfer Reason: {self.transfer_reason}")
        
        # Last few exchanges
        if self.conversation_history:
            recent = self.conversation_history[-4:]
            summary_parts.append("Recent Conversation:")
            for turn in recent:
                role = turn["role"].capitalize()
                content = turn["content"][:100] + "..." if len(turn["content"]) > 100 else turn["content"]
                summary_parts.append(f"  {role}: {content}")
        
        return "\n".join(summary_parts)


class SessionManager:
    """
    Manages persistent sessions across instances and transfers.
    """
    
    SESSION_PREFIX = "session:"
    ACTIVE_CALLS_KEY = "active_calls"
    
    def __init__(self):
        self._local_sessions: Dict[str, CallSession] = {}
        self._redis = get_redis_client()
    
    def create_session(
        self,
        caller_name: str,
        phone_number: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> CallSession:
        """
        Create a new call session.
        """
        session_id = session_id or str(uuid.uuid4())[:12]
        
        session = CallSession(
            session_id=session_id,
            caller_name=caller_name,
            phone_number=phone_number,
        )
        
        # Store locally
        self._local_sessions[session_id] = session
        
        # Persist to Redis
        self._persist_session(session)
        
        # Track active call
        self._add_active_call(session_id)
        
        logger.info(f"Created session {session_id} for {caller_name}")
        return session
    
    def get_session(self, session_id: str) -> Optional[CallSession]:
        """
        Get session from local cache or Redis.
        """
        # Check local first
        if session_id in self._local_sessions:
            return self._local_sessions[session_id]
        
        # Try Redis
        session = self._load_session(session_id)
        if session:
            self._local_sessions[session_id] = session
            return session
        
        return None
    
    def update_session(self, session: CallSession):
        """
        Update session state in Redis.
        """
        session.last_activity = time.time()
        self._local_sessions[session.session_id] = session
        self._persist_session(session)
    
    def end_session(self, session_id: str, reason: str = "completed"):
        """
        End a call session and archive it.
        """
        session = self.get_session(session_id)
        if session:
            # Archive before deletion
            self._archive_session(session, reason)
            
            # Remove from active
            self._remove_active_call(session_id)
            
            # Clear from local cache
            if session_id in self._local_sessions:
                del self._local_sessions[session_id]
            
            # Delete from Redis (or let TTL expire)
            if self._redis:
                try:
                    self._redis.delete(f"{self.SESSION_PREFIX}{session_id}")
                except Exception as e:
                    logger.warning(f"Failed to delete session from Redis: {e}")
        
        logger.info(f"Ended session {session_id}: {reason}")
    
    def request_transfer(
        self,
        session: CallSession,
        target: str,
        reason: str,
    ) -> dict:
        """
        Request a call transfer to human agent or specialist.
        Returns transfer ticket for handoff.
        """
        session.transfer_requested = True
        session.transfer_reason = reason
        session.transfer_target = target
        
        self.update_session(session)
        
        # Generate transfer ticket
        ticket = {
            "ticket_id": f"TRF-{session.session_id}-{int(time.time())}",
            "session_id": session.session_id,
            "caller_name": session.caller_name,
            "phone_number": session.phone_number,
            "target": target,
            "reason": reason,
            "context_summary": session.get_context_summary(),
            "created_at": datetime.now().isoformat(),
            "priority": self._calculate_priority(session),
        }
        
        # Store transfer ticket in Redis for queue
        if self._redis:
            try:
                self._redis.lpush("transfer_queue", json.dumps(ticket))
                self._redis.expire("transfer_queue", 3600)  # 1 hour TTL
            except Exception as e:
                logger.error(f"Failed to queue transfer: {e}")
        
        logger.info(f"Transfer requested: {ticket['ticket_id']} -> {target}")
        return ticket
    
    def get_active_sessions(self) -> List[str]:
        """
        Get list of active session IDs.
        """
        if not self._redis:
            return list(self._local_sessions.keys())
        
        try:
            members = self._redis.smembers(self.ACTIVE_CALLS_KEY)
            return [m.decode() if isinstance(m, bytes) else m for m in members]
        except Exception:
            return list(self._local_sessions.keys())
    
    def get_session_stats(self) -> dict:
        """
        Get statistics about sessions.
        """
        active_ids = self.get_active_sessions()
        
        total_turns = 0
        total_errors = 0
        total_interruptions = 0
        
        for session_id in active_ids:
            session = self.get_session(session_id)
            if session:
                total_turns += session.turn_count
                total_errors += session.error_count
                total_interruptions += session.interruption_count
        
        return {
            "active_sessions": len(active_ids),
            "total_turns": total_turns,
            "total_errors": total_errors,
            "total_interruptions": total_interruptions,
        }
    
    # =========== Private Methods ===========
    
    def _persist_session(self, session: CallSession):
        """Persist session to Redis."""
        if not self._redis:
            return
        
        try:
            key = f"{self.SESSION_PREFIX}{session.session_id}"
            data = json.dumps(session.to_dict())
            self._redis.setex(key, config.redis.session_ttl, data)
        except Exception as e:
            logger.warning(f"Failed to persist session: {e}")
    
    def _load_session(self, session_id: str) -> Optional[CallSession]:
        """Load session from Redis."""
        if not self._redis:
            return None
        
        try:
            key = f"{self.SESSION_PREFIX}{session_id}"
            data = self._redis.get(key)
            if data:
                if isinstance(data, bytes):
                    data = data.decode()
                return CallSession.from_dict(json.loads(data))
        except Exception as e:
            logger.warning(f"Failed to load session: {e}")
        
        return None
    
    def _add_active_call(self, session_id: str):
        """Add to active calls set."""
        if not self._redis:
            return
        
        try:
            self._redis.sadd(self.ACTIVE_CALLS_KEY, session_id)
        except Exception:
            pass
    
    def _remove_active_call(self, session_id: str):
        """Remove from active calls set."""
        if not self._redis:
            return
        
        try:
            self._redis.srem(self.ACTIVE_CALLS_KEY, session_id)
        except Exception:
            pass
    
    def _archive_session(self, session: CallSession, end_reason: str):
        """Archive completed session for analytics."""
        if not self._redis:
            return
        
        try:
            archive_data = {
                **session.to_dict(),
                "ended_at": time.time(),
                "end_reason": end_reason,
                "duration_seconds": time.time() - session.started_at,
            }
            
            # Store in daily archive list
            date_key = datetime.now().strftime("archive:%Y-%m-%d")
            self._redis.rpush(date_key, json.dumps(archive_data))
            self._redis.expire(date_key, 86400 * 7)  # Keep 7 days
            
        except Exception as e:
            logger.warning(f"Failed to archive session: {e}")
    
    def _calculate_priority(self, session: CallSession) -> str:
        """Calculate transfer priority based on session state."""
        # High priority: errors, long wait, loops
        if session.error_count >= 2 or session.loop_count >= 1:
            return "high"
        if session.turn_count >= 10:
            return "medium"
        return "normal"


# Global instance
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """Get the global session manager."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager
