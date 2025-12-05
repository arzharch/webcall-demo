from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import uuid

# --- Data models used across services ---

class BookingIntent(BaseModel):
    """
    Represents the information extracted by the agent for a potential reservation.
    This model is used by the CRM service and the 'create_reservation' tool.
    """
    date: Optional[str] = None
    time: Optional[str] = None
    party_size: Optional[int] = None
    customer_name: Optional[str] = None
    phone: Optional[str] = None
    special_requests: Optional[str] = None
    
    def is_complete(self) -> bool:
        """Checks if the core information for a booking has been gathered."""
        return all([
            self.date,
            self.time,
            self.party_size,
            self.customer_name
        ])

class TicketStatus(str, Enum):
    """Represents the status of a CRM ticket."""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    CANCELLED = "cancelled"
    COMPLETED = "completed"

class Ticket(BaseModel):
    """
    Represents a CRM ticket, typically for a reservation.
    This is the primary data model for the CRM service.
    """
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

# --- RAG Service Data Models ---

class RAGDocument(BaseModel):
    """A document processed and stored by the RAG service."""
    id: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None

class SearchResult(BaseModel):
    """A single search result returned by the RAG service."""
    content: str
    score: float
    metadata: Dict[str, Any]