from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from enum import Enum

class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"

class Message(BaseModel):
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)

class CallTranscript(BaseModel):
    call_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    messages: List[Message] = Field(default_factory=list)
    summary: Optional[str] = None
    rating: Optional[int] = Field(default=None, ge=1, le=5)
    duration_seconds: Optional[float] = None
    booking_created: bool = False

class TicketStatus(str, Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    CANCELLED = "cancelled"
    COMPLETED = "completed"

class Ticket(BaseModel):
    ticket_id: str
    call_id: str
    customer_name: str
    phone: Optional[str] = None
    date: str  # YYYY-MM-DD
    time: str  # HH:MM
    party_size: int = Field(ge=1, le=12)
    special_requests: Optional[str] = None
    status: TicketStatus = TicketStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.now)

class BookingIntent(BaseModel):
    """Detected booking intent from conversation"""
    has_booking_intent: bool
    customer_name: Optional[str] = None
    date: Optional[str] = None
    time: Optional[str] = None
    party_size: Optional[int] = None
    phone: Optional[str] = None
    special_requests: Optional[str] = None
    
class BookingIntent(BaseModel):
    """Detected booking intent from conversation"""
    has_booking_intent: bool
    customer_name: Optional[str] = None
    date: Optional[str] = None
    time: Optional[str] = None
    party_size: Optional[int] = None
    phone: Optional[str] = None
    special_requests: Optional[str] = None
