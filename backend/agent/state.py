from pydantic import BaseModel, Field
from typing import List, Optional

class BookingDetails(BaseModel):
    """Represents the details of a potential booking."""
    date: Optional[str] = None
    time: Optional[str] = None
    party_size: Optional[int] = None
    
class SessionState(BaseModel):
    """Tracks the full state of a conversation session."""
    session_id: str
    caller_name: Optional[str] = None
    current_intent: str = "GENERAL_CONVERSATION"
    booking_details: BookingDetails = Field(default_factory=BookingDetails)
    conversation_history: List[str] = Field(default_factory=list)

    def update_history(self, user_message: str, ai_message: str):
        """Adds a user/AI interaction to the conversation history."""
        self.conversation_history.append(f"User: {user_message}")
        self.conversation_history.append(f"AI: {ai_message}")

    def get_formatted_history(self) -> str:
        """Returns the conversation history as a single formatted string."""
        return "\n".join(self.conversation_history)
