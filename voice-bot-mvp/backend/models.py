from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime
from enum import Enum
import uuid

class ConversationState(str, Enum):
    """Conversation flow states"""
    GREETING = "greeting"
    INQUIRY = "inquiry"
    BOOKING = "booking"
    CONFIRMATION = "confirmation"
    CLOSING = "closing"
    ENDED = "ended"

class MessageRole(str, Enum):
    """Message roles"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool" # Changed from FUNCTION for LangChain compatibility

class Message(BaseModel):
    """Individual conversation message"""
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    tool_calls: Optional[List[Dict[str, Any]]] = None # For LangChain agent
    tool_call_id: Optional[str] = None # For LangChain tool response

class BookingIntent(BaseModel):
    """Extracted booking information"""
    date: Optional[str] = None
    time: Optional[str] = None
    party_size: Optional[int] = None
    customer_name: Optional[str] = None
    phone: Optional[str] = None
    special_requests: Optional[str] = None
    
    def is_complete(self) -> bool:
        """Check if all required fields are present"""
        return all([
            self.date,
            self.time,
            self.party_size,
            self.customer_name
        ])

class ConversationContext(BaseModel):
    """Complete conversation state"""
    call_id: str = Field(default_factory=lambda: f"call_{uuid.uuid4().hex[:8]}")
    session_id: str = Field(default_factory=lambda: f"session_{uuid.uuid4().hex[:12]}")
    messages: List[Message] = []
    state: ConversationState = ConversationState.GREETING
    booking_intent: BookingIntent = Field(default_factory=BookingIntent)
    customer_info: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None
    
    def add_message(self, role: MessageRole, content: str, **kwargs):
        """Add a message to conversation history"""
        msg = Message(role=role, content=content, **kwargs)
        self.messages.append(msg)
        return msg
    
    def get_recent_messages(self, n: int = 10) -> List[Message]:
        """Get last N messages"""
        return self.messages[-n:]

class TicketStatus(str, Enum):
    """Ticket statuses"""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    CANCELLED = "cancelled"
    COMPLETED = "completed"

class Ticket(BaseModel):
    """CRM Ticket for reservations"""
    id: str = Field(default_factory=lambda: f"ticket_{uuid.uuid4().hex[:8]}")
    call_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    customer_name: Optional[str] = None
    customer_phone: Optional[str] = None
    intent: str = "reservation"
    details: Dict[str, Any] = Field(default_factory=dict)
    status: TicketStatus = TicketStatus.PENDING
    transcript: List[Dict[str, str]] = Field(default_factory=list)
    summary: Optional[str] = None
    
    class Config:
        use_enum_values = True

class AudioChunk(BaseModel):
    """Audio data chunk"""
    data: bytes
    format: str = "pcm16"
    sample_rate: int = 16000
    channels: int = 1
    
class RAGDocument(BaseModel):
    """Document for RAG system"""
    id: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None

class SearchResult(BaseModel):
    """RAG search result"""
    content: str
    score: float
    metadata: Dict[str, Any]
