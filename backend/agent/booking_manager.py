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
    reasoning: Optional[str] = Field(None, description="Verify specific date and time from the user. Explain your extraction logic briefly. Example: 'User said next tuesday, which is 2024-01-30'")
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
            "- FIRST, populate the 'reasoning' field to explain your thought process.\n"
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
            "CRITICAL - STATE PRESERVATION:\n"
            "- Only extract NEW or CHANGED information.\n"
            "- If the user just provides a time (e.g., '11am'), DO NOT output 'date' unless they explicitly said 'today' or a day name.\n"
            "- If valid info exists in history (e.g. date=2024-01-24), DO NOT OVERWRITE IT with 'today's' date unless the user explicitly changed it.\n"
            "- Return null/None for fields that are not mentioned or implied in the current turn.\n"
            "TIME PARSING:\n"
            '- "6:30 pm" -> "18:30"\n'
            '- "2 pm" -> "14:00"\n'
            '- "7" -> "19:00" (if dinner context implication)\n'
            '- DO NOT AUTOCORRECT times to match opening hours. If user says "11am", output "11:00". If user says "2pm", output "14:00". The system will handle validation.\n\n'
            "{format_instructions}"
        )
    
    async def extract_info(self, user_message: str, current_slot: BookingSlot, chat_history: list[BaseMessage] = []) -> BookingSlot:
        """Extract booking info from user message and update the slot with retry logic."""
        max_retries = 1
        for attempt in range(max_retries + 1):
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
                
                # Add retry context if this is a correction attempt
                input_text = user_message
                if attempt > 0:
                    input_text += f"\n\nSYSTEM: Your previous output was invalid. Please ensure valid JSON format matching schema."

                result: BookingUpdate = await chain.ainvoke({
                    "input": input_text,
                    "history": history_context
                })
                
                # Update slot with extracted information
                if result.party_size is not None:
                    current_slot.party_size = result.party_size
                
                # Logic to prevent date overwriting with defaults
                if result.date:
                    # Only update if the result.date is different from current or if we had none
                    current_slot.date = result.date
                
                if result.time:
                    # Validate common format issues from LLM
                    try:
                        # sometimes LLM extracts '14:00' correctly, sometimes converts to '2:00 PM'
                        # enforce 24h format if it slipped through
                        t = result.time.lower().strip()
                        if "pm" in t or "am" in t:
                            if ":" in t:
                                t_dt = datetime.strptime(t, "%I:%M %p")
                            else:
                                # Handle "2 pm" without minutes
                                t_dt = datetime.strptime(t, "%I %p")
                            current_slot.time = t_dt.strftime("%H:%M")
                        elif ":" not in t and t.isdigit():
                             # Handle "14" -> "14:00"
                             current_slot.time = f"{int(t):02d}:00"
                        else:
                            current_slot.time = result.time
                    except Exception as e:
                        logger.warning(f"Time parsing fallback failed for '{result.time}': {e}")
                        current_slot.time = result.time
                if result.name:
                    current_slot.name = result.name
                if result.notes:
                    current_slot.notes = result.notes
                
                logger.info(f"Extracted info: {result}")
                logger.info(f"Reasoning: {result.reasoning}")
                logger.info(f"Updated booking slot: {current_slot}")
                
                # If successful, break loop
                break
                
            except Exception as e:
                logger.warning(f"Failed to extract info (attempt {attempt+1}): {e}")
                
                if attempt == max_retries:
                    logger.error("All extraction attempts failed.")
                    # Fallback for simple cases if JSON parsing fails heavily
                    pass
        

        return current_slot
    
    async def _generate_natural_refusal(self, instruction: str) -> str:
        """Uses LLM to generate a polite, natural refusal based on business logic."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "You are Bella, a friendly restaurant hostess. The system has rejected a user's request. "
             "Explain the rejection naturally and politely ask for an alternative. "
             "Keep it short (1-2 sentences). Do NOT apologize excessively."
            ),
            ("human", f"System Instruction: {instruction}")
        ])
        chain = prompt | self.llm
        response = await chain.ainvoke({})
        return response.content

    async def handle_booking_conversation(self, state: SessionState, user_message: str) -> str:
        """
        Manage the booking conversation flow with slot filling.
        Returns a natural response to the user.
        """
        # ... (rest of the method logic) ...
        # 1. Check if user is answering a confirmation request
        # We do this BEFORE extraction to avoid confusing "yes" with data
        
        # PRE-CHECK: Apply caller name if available and not "Guest" (Case insensitive)
        # This ensures we don't ask for name if we already know who called.
        if not state.booking_slot.name and state.caller_name:
             normalized_name = state.caller_name.strip()
             if normalized_name.lower() != "guest":
                 state.booking_slot.name = normalized_name
                 logger.info(f"Applied caller name to booking slot: {normalized_name}")

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
            else:
                # User rejected or is providing new info (e.g., "No, actually 5 people")
                # Clear the flag and proceed to extraction
                state.awaiting_confirmation = False

        # 2. Extract information from user's message
        # (Name logic moved to Pre-Check)

        state.booking_slot = await self.extract_info(user_message, state.booking_slot, state.conversation_history)
        
        # Re-apply caller name if extraction returned None/cleared it and we have a valid caller name
        if not state.booking_slot.name and state.caller_name:
             normalized_name = state.caller_name.strip()
             if normalized_name.lower() != "guest":
                state.booking_slot.name = normalized_name
        

        # 3. Validation Logic (Business Hours)
        # If time is present, we must ensure it's valid BEFORE checking completeness
        if state.booking_slot.time and state.booking_slot.date:
            try:
                # Basic check: Weekends 12-11, Weekdays 5-11
                dt_date = datetime.strptime(state.booking_slot.date, "%Y-%m-%d")
                dt_time = datetime.strptime(state.booking_slot.time, "%H:%M")
                
                is_weekend = dt_date.weekday() >= 5
                hour = dt_time.hour
                
                valid = False
                if is_weekend:
                    if 12 <= hour < 23: valid = True # 12pm to 11pm
                    open_str = "12 PM to 11 PM"
                else:
                    if 17 <= hour < 23: valid = True # 5pm to 11pm
                    open_str = "5 PM to 11 PM"
                

                if not valid:
                    requested_time = state.booking_slot.time
                    state.booking_slot.time = None # Clear invalid time
                    
                    # USE LLM GENERATION:
                    instruction = f"The user requested {requested_time}. We are only open from {open_str} on {dt_date.strftime('%A')}. Tell them accurately."
                    return await self._generate_natural_refusal(instruction)
                    
            except ValueError:
                pass # Date/time format error, ignore and let standard flow handle
        
        # 4. Check if we have all required information
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
        return await self._ask_for_missing_info(missing, state.booking_slot)
    
    def _is_confirmation(self, message: str) -> bool:
        """Check if message is a confirmation."""
        confirmations = ["yes", "yeah", "yep", "correct", "right", "sure", "confirm", "ok", "okay", "perfect", "sounds good"]
        return any(word in message.lower() for word in confirmations)
    
    def _is_rejection(self, message: str) -> bool:
        """Check if message is a rejection/change request."""
        rejections = ["no", "nope", "not", "change", "different", "actually", "wait"]
        return any(word in message.lower() for word in rejections)
    
    
    async def _ask_for_missing_info(self, missing: list[str], slot: BookingSlot) -> str:
        """Generate a natural question for missing information using LLM."""
        
        # Build a context string describing what we already know
        known_info = []
        if slot.party_size: known_info.append(f"Party Size: {slot.party_size}")
        if slot.date: known_info.append(f"Date: {slot.date}")
        if slot.time: known_info.append(f"Time: {slot.time}")
        if slot.name: known_info.append(f"Name: {slot.name}")
        
        known_str = ", ".join(known_info) if known_info else "Nothing yet"
        missing_str = ", ".join(missing)
        
        system_prompt = (
            "You are Bella, a professional restaurant hostess.\n"
            "Your goal is to collect missing reservation details from the user.\n"
            f"Current Known Info: {known_str}\n"
            f"Missing Info: {missing_str}\n\n"
            "TASK: Ask the user for the MISSING information naturally.\n"
            "RULES:\n"
            "- If 'party_size' is missing, ask for number of guests.\n"
            "- If 'date' is missing, ask when they would like to come.\n"
            "- If 'time' is missing, ask for the preferred time.\n"
            "- If 'name' is missing, ask for the booking name.\n"
            "- DO NOT ask for information we already have.\n"
            "- Keep it short (1 sentence ideally).\n"
            "- Be conversational, not robotic.\n"
            "- If asking for time on a Weekend (Sat/Sun), mention we are open 12 PM - 11 PM.\n"
            "- If asking for time on a Weekday, mention we are open 5 PM - 11 PM.\n"
            "- Example: 'Great! And what time works best for you on Saturday?'"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Generate the next question.")
        ])
        
        chain = prompt | self.llm
        response = await chain.ainvoke({})
        return response.content

    
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
