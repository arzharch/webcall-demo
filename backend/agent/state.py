from typing import List, Optional, Literal
from datetime import datetime

from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel, Field

Intent = Literal[
    "make_booking",
    "update_booking",
    "cancel_booking",
    "find_booking",
    "check_availability",
    "search_menu",
    "general_query",
    "off_topic",
    "unknown",
]


class BookingSlot(BaseModel):
    """Tracks information for a booking in progress."""
    name: Optional[str] = None
    party_size: Optional[int] = None
    date: Optional[str] = None  # YYYY-MM-DD
    time: Optional[str] = None  # HH:MM
    notes: Optional[str] = None
    booking_id: Optional[int] = None
    
    def is_complete_for_new_booking(self) -> bool:
        """Check if we have all required info for a new booking."""
        # Name is optional - can use caller_name instead
        return all([self.party_size, self.date, self.time])
    
    def get_missing_fields(self) -> List[str]:
        """Get list of missing required fields."""
        missing = []
        if not self.party_size:
            missing.append("party size")
        if not self.date:
            missing.append("date")
        if not self.time:
            missing.append("time")
        if not self.name:
            missing.append("name")
        return missing


class SessionState(BaseModel):
    """
    Manages the state of a conversation session, including history, intent, and booking details.
    """
    caller_name: str
    conversation_history: List[BaseMessage] = Field(
        default_factory=lambda: [HumanMessage(content="Hello, how can I help you?")]
    )
    current_intent: Optional[Intent] = "unknown"
    booking_slot: BookingSlot = Field(default_factory=BookingSlot)
    awaiting_confirmation: bool = False  # True when we've asked user to confirm booking
    confusion_count: int = 0  # Track consecutive gibberish/confusion messages
