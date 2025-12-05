from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import uuid

class MessageRole(str, Enum):
    """Message roles for conversation"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"

class Message(BaseModel):
    """Single conversation message"""
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None

class BookingIntent(BaseModel):
    """Extracted booking details"""
    date: Optional[str] = None
    time: Optional[str] = None
    party_size: Optional[int] = None
    customer_name: Optional[str] = None
    phone: Optional[str] = None
    special_requests: Optional[str] = None
    
    def is_complete(self) -> bool:
        """Check if booking has all required fields"""
        return all([self.date, self.time, self.party_size, self.customer_name])

class SessionState(BaseModel):
    """User session state for LangChain"""
    call_id: str = Field(default_factory=lambda: f"call_{uuid.uuid4().hex[:8]}")
    session_id: str = Field(default_factory=lambda: f"session_{uuid.uuid4().hex[:12]}")
    messages: List[Message] = Field(default_factory=list)
    current_intent: str = "unknown"
    booking_slot: BookingIntent = Field(default_factory=BookingIntent)
    context_data: Dict[str, Any] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=datetime.utcnow)
    
    def add_message(self, role: MessageRole, content: str):
        """Add message to conversation"""
        self.messages.append(Message(role=role, content=content))
    
    def get_recent_messages(self, n: int = 10) -> List[Message]:
        """Get last n messages"""
        return self.messages[-n:]
    
    def to_langchain_messages(self) -> List[Dict[str, str]]:
        """Convert to LangChain message format"""
        return [
            {"role": msg.role.value, "content": msg.content}
            for msg in self.get_recent_messages()
        ]

class TicketStatus(str, Enum):
    """Reservation ticket status"""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    CANCELLED = "cancelled"
    COMPLETED = "completed"

class Ticket(BaseModel):
    """CRM ticket for reservations"""
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

class RAGDocument(BaseModel):
    """Document in knowledge base"""
    id: str
    content: str
    source: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class SearchResult(BaseModel):
    """RAG search result"""
    document: RAGDocument
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
