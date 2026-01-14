"""Manages booking conversation flow and information extraction."""
import re
from datetime import datetime, timedelta
from typing import Optional, Tuple

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from loguru import logger

from agent.state import SessionState, BookingSlot
from agent.tools import check_availability, make_booking


class BookingUpdate(BaseModel):
    """Structured extraction of booking information."""
    party_size: Optional[int] = Field(None, description="Number of people for the reservation")
    date: Optional[str] = Field(None, description="Date of reservation in YYYY-MM-DD format")
    time: Optional[str] = Field(None, description="Time of reservation in HH:MM (24-hour) format")
    name: Optional[str] = Field(None, description="Name of the contact person")
    notes: Optional[str] = Field(None, description="Any special requests or dietary restrictions")


class BookingManager:
    """Handles booking flow with slot filling and natural conversation."""
    
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
    
    def _get_date_context(self) -> dict:
        """Get current date context for extraction."""
        today = datetime.now()
        return {
            "today": today,
            "tomorrow": today + timedelta(days=1),
            "saturday": today + timedelta(days=(5 - today.weekday()) % 7 or 7),
            "friday": today + timedelta(days=(4 - today.weekday()) % 7 or 7),
            "sunday": today + timedelta(days=(6 - today.weekday()) % 7 or 7),
            "monday": today + timedelta(days=(0 - today.weekday()) % 7 or 7),
            "tuesday": today + timedelta(days=(1 - today.weekday()) % 7 or 7),
            "wednesday": today + timedelta(days=(2 - today.weekday()) % 7 or 7),
            "thursday": today + timedelta(days=(3 - today.weekday()) % 7 or 7),
        }
    
    def _build_extraction_prompt(self) -> str:
        """Build the extraction prompt with current date context."""
        dates = self._get_date_context()
        today = dates["today"]
        
        return (
            "Extract booking information from the conversational context.\n"
            f"Today is {today.strftime('%A, %B %d, %Y')}.\n\n"
            "CRITICAL RULES:\n"
            "- Look at the CONVERSATION HISTORY to understand context.\n"
            "- If the user says '2' after being asked 'How many people?', then party_size is 2.\n"
            "- Phrases like 'me and my friend' count as party_size=2.\n"
            "- Phrases like 'just me' count as party_size=1.\n"
            "- Phrases like 'couple' count as party_size=2.\n"
            "- If information was provided in previous messages, INCLUDE IT in the output.\n\n"
            "DATE PARSING:\n"
            f'- "tonight"/"today" -> {dates["today"].strftime("%Y-%m-%d")}\n'
            f'- "tomorrow" -> {dates["tomorrow"].strftime("%Y-%m-%d")}\n'
            f'- "saturday" -> {dates["saturday"].strftime("%Y-%m-%d")}\n'
            f'- "friday" -> {dates["friday"].strftime("%Y-%m-%d")}\n'
            f'- "sunday" -> {dates["sunday"].strftime("%Y-%m-%d")}\n'
            f'- "monday" -> {dates["monday"].strftime("%Y-%m-%d")}\n'
            '- Any other day name -> find next occurrence\n\n'
            "TIME PARSING:\n"
            '- "6:30 pm" -> "18:30"\n'
            '- "7" -> "19:00" (if dinner context implication)\n\n'
            "{format_instructions}"
        )
    
    async def extract_info(self, user_message: str, current_slot: BookingSlot, chat_history: list[BaseMessage] = []) -> BookingSlot:
        """Extract booking info from user message and update the slot."""
        try:
            # Setup Pydantic parser
            parser = PydanticOutputParser(pydantic_object=BookingUpdate)
            
            # Build prompt dynamically with current date context
            prompt = ChatPromptTemplate.from_messages([
                ("system", self._build_extraction_prompt()),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}")
            ])
            
            # Inject format instructions
            prompt = prompt.partial(format_instructions=parser.get_format_instructions())
            
            chain = prompt | self.llm | parser
            
            # Use recent history for context (last 5 messages), excluding the current user message
            history_context = chat_history[:-1][-5:] if chat_history else []
            
            result: BookingUpdate = await chain.ainvoke({
                "input": user_message,
                "history": history_context
            })
            
            # Update slot with extracted information
            if result.party_size is not None:
                current_slot.party_size = result.party_size
            if result.date:
                current_slot.date = result.date
            if result.time:
                current_slot.time = result.time
            if result.name:
                current_slot.name = result.name
            if result.notes:
                current_slot.notes = result.notes
            
            logger.info(f"Extracted info: {result}")
            logger.info(f"Updated booking slot: {current_slot}")
            
        except Exception as e:
            logger.warning(f"Failed to extract info: {e}")
            # Fallback for simple cases if JSON parsing fails heavily
            pass
        

        return current_slot
    
    async def handle_booking_conversation(self, state: SessionState, user_message: str) -> str:
        """
        Manage the booking conversation flow with slot filling.
        Returns a natural response to the user.
        """
        # Extract information from user's message
        state.booking_slot = await self.extract_info(user_message, state.booking_slot, state.conversation_history)
        
        # Check if user is confirming a booking
        if state.awaiting_confirmation:
            if self._is_confirmation(user_message):
                # User confirmed, actually make the booking
                result = make_booking.invoke({
                    "name": state.booking_slot.name or state.caller_name,
                    "party_size": state.booking_slot.party_size,
                    "date_str": state.booking_slot.date,
                    "time_str": state.booking_slot.time,
                    "notes": state.booking_slot.notes
                })
                state.awaiting_confirmation = False
                state.booking_slot = BookingSlot()  # Reset
                return result
            elif self._is_rejection(user_message):
                # User wants to change something
                state.awaiting_confirmation = False
                return "No problem! What would you like to change?"
            else:
                # User might be providing additional info, try to extract again
                state.booking_slot = await self.extract_info(user_message, state.booking_slot)
                # Continue with normal flow below
        
        # Check if we have all required information
        if state.booking_slot.is_complete_for_new_booking():
            # We have everything, check availability and confirm
            try:
                availability = check_availability.invoke({
                    "party_size": state.booking_slot.party_size,
                    "date_str": state.booking_slot.date,
                    "time_str": state.booking_slot.time
                })
                
                if "available" in availability.lower():
                    # Confirm with user
                    state.awaiting_confirmation = True
                    return (
                        f"Great! Table for {state.booking_slot.party_size} "
                        f"on {self._format_date_friendly(state.booking_slot.date)} at "
                        f"{self._format_time_friendly(state.booking_slot.time)} "
                        f"under {state.booking_slot.name or state.caller_name}. Shall I confirm that?"
                    )
                else:
                    # Not available, inform user
                    state.booking_slot.time = None  # Clear time so they can pick another
                    return availability + " What other time would work for you?"
            except Exception as e:
                logger.error(f"Error checking availability: {e}")
                return "Let me check that for you. What time would work best?"
        
        # We're missing information, ask for it naturally
        missing = state.booking_slot.get_missing_fields()
        return self._ask_for_missing_info(missing, state.booking_slot)
    
    def _is_confirmation(self, message: str) -> bool:
        """Check if message is a confirmation."""
        confirmations = ["yes", "yeah", "yep", "correct", "right", "sure", "confirm", "ok", "okay", "perfect", "sounds good"]
        return any(word in message.lower() for word in confirmations)
    
    def _is_rejection(self, message: str) -> bool:
        """Check if message is a rejection/change request."""
        rejections = ["no", "nope", "not", "change", "different", "actually", "wait"]
        return any(word in message.lower() for word in rejections)
    
    def _ask_for_missing_info(self, missing: list[str], slot: BookingSlot) -> str:
        """Generate a natural question for missing information."""
        # Build context of what we have
        context_parts = []
        if slot.party_size:
            context_parts.append(f"for {slot.party_size} people")
        if slot.date:
            context_parts.append(f"on {self._format_date_friendly(slot.date)}")
        if slot.time:
            context_parts.append(f"at {self._format_time_friendly(slot.time)}")
        
        context = " ".join(context_parts) if context_parts else ""
        
        if "party size" in missing:
            if context:
                return f"Perfect! How many people {context.replace('for', 'will be joining you')}?"
            return "How many people will be dining with us?"
        elif "date" in missing:
            if context:
                return f"Great! What day would work best {context}?"
            return "What day would you like to come in?"
        elif "time" in missing:
            if slot.date:
                return f"What time works for you on {self._format_date_friendly(slot.date)}?"
            return "What time would you prefer?"
        elif "name" in missing:
            if context:
                return f"Excellent! May I have a name for the reservation {context}?"
            return "May I have a name for the reservation?"
        
        return "Could you provide a few more details?"
    
    def _format_date_friendly(self, date_str: str) -> str:
        """Convert YYYY-MM-DD to friendly format."""
        try:
            date = datetime.strptime(date_str, "%Y-%m-%d")
            today = datetime.now().date()
            date_obj = date.date()
            
            if date_obj == today:
                return "tonight"
            elif date_obj == today + timedelta(days=1):
                return "tomorrow"
            else:
                return date.strftime("%A, %B %d")
        except:
            return date_str
    
    def _format_time_friendly(self, time_str: str) -> str:
        """Convert HH:MM to friendly format."""
        try:
            time = datetime.strptime(time_str, "%H:%M")
            return time.strftime("%I:%M %p").lstrip("0")
        except:
            return time_str
