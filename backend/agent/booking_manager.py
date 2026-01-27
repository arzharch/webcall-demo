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
        
        def get_next_weekday(target_day: int) -> datetime:
            """Get next occurrence of target_day (0=Mon, 6=Sun). If today, return 7 days later."""
            days_ahead = (target_day - today.weekday() + 7) % 7
            if days_ahead == 0:  # If today is the target day, user likely means next week
                days_ahead = 7
            return today + timedelta(days=days_ahead)
        
        return {
            "today": today,
            "tomorrow": today + timedelta(days=1),
            "monday": get_next_weekday(0),
            "tuesday": get_next_weekday(1),
            "wednesday": get_next_weekday(2),
            "thursday": get_next_weekday(3),
            "friday": get_next_weekday(4),
            "saturday": get_next_weekday(5),
            "sunday": get_next_weekday(6),
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
            "CRITICAL - EXTRACTION RULES:\n"
            "- Extract ONLY fields that are explicitly mentioned or strongly implied in the current turn.\n"
            "- If the user just provides a time (e.g., '11am'), extract only time, NOT date.\n"
            "- For fields NOT mentioned: return None/null. The system will preserve existing values.\n"
            "- EXCEPTION: If user says 'change to X' or 'actually Y', extract the new value even if field was previously set.\n"
            "CONFLICT RESOLUTION (CRITICAL):\n"
            "- If the input contains self-corrections (e.g. 'Table for 5, actually 6', 'Saturday... no wait, Sunday', 'I meant today'), ALWAYS extract the FINAL intention.\n"
            "- Use the 'reasoning' field to explain the correction (e.g. 'User initially said 5 but corrected to 6').\n"
            "- IF YOU DETECT A CORRECTION, YOU MUST OUTPUT THE NEW VALUE. Do not return null for that field.\n"
            f"- Example: User says 'I meant today'. Reasoning: 'User corrected date to today'. Output: {{{{ 'date': '{dates['today'].strftime('%Y-%m-%d')}' }}}}\n"
            "TIME PARSING:\n"
            '- "6:30 pm" -> "18:30"\n'
            '- "2 pm" -> "14:00"\n'
            '- "9:30" or "nine thirty" WITHOUT am/pm -> Context matters:\n'
            '  * If user is booking dinner/evening -> assume PM (21:30)\n'
            '  * Restaurants typically don\'t open at 9:30 AM, so default to PM\n'
            '  * Use the reasoning field to explain your interpretation\n'
            '- "7" alone -> "19:00" (dinner context - 7 PM)\n'
            '- DO NOT AUTOCORRECT times to match opening hours. If user says "11am", output "11:00". The system will validate.\n\n'
            "{format_instructions}")
        
    
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
                    # Intelligence Check:
                    # If party_size changes abruptly (e.g., from 7 to 1) without explicit correction,
                    # it might be a misunderstanding of "Me" (could mean 'add me' or 'just me').
                    # For now, we trust the LLM reasoning, but we log it.
                    if current_slot.party_size and current_slot.party_size != result.party_size:
                        logger.info(f"Party size update detected: {current_slot.party_size} -> {result.party_size}")
                    
                    current_slot.party_size = result.party_size
                
                # Logic to prevent date overwriting with defaults
                if result.date:
                    if current_slot.date and current_slot.date != result.date:
                         logger.info(f"Date update detected: {current_slot.date} -> {result.date}")
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
                            final_time_str = result.time
                            
                        # Compare logic
                        if current_slot.time and current_slot.time != final_time_str:
                             logger.info(f"Time update detected: {current_slot.time} -> {final_time_str}")
                             
                        current_slot.time = final_time_str
                    except Exception as e:
                        logger.warning(f"Time parsing fallback failed for '{result.time}': {e}")
                        current_slot.time = result.time
                
                # NAME VALIDATION: Only update name if it seems intentional
                # Avoid overwriting with companion names (e.g., "me and Echo" shouldn't set name to "Echo")
                if result.name:
                    # Check if this looks like a booking name vs. a mentioned companion
                    # Heuristic: Don't overwrite if message contains "and" + the extracted name (likely a companion)
                    user_msg_lower = user_message.lower()
                    name_lower = result.name.lower()
                    
                    is_companion_mention = (
                        (" and " + name_lower in user_msg_lower) or 
                        (name_lower + " and " in user_msg_lower)
                    )
                    
                    if not is_companion_mention:
                        current_slot.name = result.name
                        logger.info(f"Updated booking name to: {result.name}")
                    else:
                        logger.info(f"Skipping name update - '{result.name}' appears to be a companion mention")
                
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
        """Uses LLM to generate a natural response based on instruction."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "You are Bella, a friendly restaurant hostess. "
             "Task: Communicate the system's response to the user.\n"
             "Style: Polite, natural, warm, and concise (1-2 sentences).\n"
             "Rules: Do NOT apologize excessively. Focus on the solution."
            ),
            ("human", f"System Instruction: {instruction}")
        ])
        chain = prompt | self.llm
        response = await chain.ainvoke({})
        return response.content


    async def _generate_success_response(self, slot: BookingSlot, user_message: str) -> str:
        """Generate a natural success/confirmation message, addressing any side questions."""
        system_prompt = (
            "You are Bella, a professional restaurant hostess.\n"
            "We have all the necessary booking details and found a table.\n"
            f"Booking: Table for {slot.party_size} on {slot.date} at {slot.time} under {slot.name}.\n"
            f"User's Last Message: \"{user_message}\"\n\n"
            "TASK: Ask the user to CONFIRM the booking details.\n"
            "CRITICAL RULES:\n"
            "1. SIDE QUESTIONS: If the user asked a question in their last message (e.g. 'parking?', 'vegan?'), ANSWER IT FIRST.\n"
            "2. CONFIRMATION: Then, state the booking details clearly and ask 'Shall I confirm?'\n"
            "3. STYLE: Conversational, warm, short."
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Generate the confirmation response.")
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
            # FIX #16: Handle corrections during confirmation (e.g., "Yes, but make it 8pm")
            # Check for confirmation with modifications
            has_confirmation_word = self._is_confirmation(user_message)
            has_modification_word = any(word in user_message.lower() for word in [
                'but', 'except', 'change', 'actually', 'make it', 'switch', 'different'
            ])
            
            if has_confirmation_word and has_modification_word:
                # User is confirming but with a change: "Yes but 8pm" or "Confirm but change name to Bob"
                logger.info("Detected confirmation with modification - extracting changes")
                # Extract the modification
                state.booking_slot = await self.extract_info(user_message, state.booking_slot, state.conversation_history)
                # Stay in confirmation mode - show updated details and ask again
                return await self._generate_success_response(state.booking_slot, user_message)
                
            elif has_confirmation_word:
                # Pure confirmation with no modifications
                result = make_booking.invoke({
                    "name": state.booking_slot.name or state.caller_name,
                    "party_size": state.booking_slot.party_size,
                    "date_str": state.booking_slot.date,
                    "time_str": state.booking_slot.time,
                    "notes": state.booking_slot.notes
                })
                
                # Extract booking ID from result for future updates/cancellations
                import re
                booking_id_match = re.search(r'booking ID is (\d+)', result)
                if booking_id_match:
                    state.last_booking_id = int(booking_id_match.group(1))
                    logger.info(f"Saved last_booking_id: {state.last_booking_id}")
                
                state.awaiting_confirmation = False
                state.booking_slot = BookingSlot()  # Reset
                return result
            else:
                # User rejected or is providing completely new info (e.g., "No, actually 5 people")
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
                    return await self._generate_success_response(state.booking_slot, user_message)
                else:
                    # Not available, inform user
                    state.booking_slot.time = None  # Clear time so they can pick another
                    return availability + " What other time would work for you?"
            except Exception as e:
                logger.error(f"Error checking availability: {e}")
                return "Let me check that for you. What time would work best?"
        
        # We're missing information, ask for it naturally
        missing = state.booking_slot.get_missing_fields()
        return await self._ask_for_missing_info(missing, state.booking_slot, user_message)
    
    def _is_confirmation(self, message: str) -> bool:
        """Check if message is a confirmation."""
        confirmations = ["yes", "yeah", "yep", "correct", "right", "sure", "confirm", "ok", "okay", "perfect", "sounds good"]
        return any(word in message.lower() for word in confirmations)
    
    def _is_rejection(self, message: str) -> bool:
        """Check if message is a rejection/change request."""
        rejections = ["no", "nope", "not", "change", "different", "actually", "wait"]
        return any(word in message.lower() for word in rejections)
    
    
    async def _ask_for_missing_info(self, missing: list[str], slot: BookingSlot, user_input_context: str = "") -> str:
        """Generate a natural question for missing information using LLM, handling side questions."""
        
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
            f"Missing Info: {missing_str}\n"
            f"User's Last Message: \"{user_input_context}\"\n\n"
            "TASK: Generate the response to the user.\n"
            "CRITICAL RULES:\n"
            "1. SIDE QUESTIONS: CHECK if the user explicitly asked a question in their last message (e.g. 'Do you have parking?', 'Is it vegan?').\n"
            "   - IF AND ONLY IF they asked, answer it first briefly.\n"
            "   - EXAMPLES of answers (only use if asked):\n"
            "     - Parking: 'Yes, we have valet.'\n"
            "     - Vegan: 'Yes, we have a separate vegan menu.'\n"
            "   - IF NO QUESTION detected, DO NOT invent an answer.\n"
            "2. THEN ASK FOR MISSING INFO: After answering (if needed), ask for the missing details naturally.\n"
            "3. SPECIFIC MISSING INFO RULES:\n"
            "   - 'party_size': Ask for number of guests.\n"
            "   - 'date': Ask when they would like to come.\n"
            "   - 'time': Ask for preferred time. (Weekends: Open 12-11 PM, Weekdays: 5-11 PM).\n"
            "   - 'name': Ask for booking name.\n"
            "4. STYLE: Conversational, warm, short (1-2 sentences). Don't repeat info we have.\n"
            "5. Example: 'Now, how many guests will be joining you?'"
            "6. Answer only as much as asked oir needed to be asked do not extend queries beyond that."
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Generate the response.")
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
