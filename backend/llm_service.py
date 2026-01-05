import google.generativeai as genai
from typing import List
import json
import logging
import re
import asyncio
from functools import lru_cache

from backend.config import get_settings
from backend.models import Message, MessageRole, BookingIntent
from backend.prompts import get_system_prompt

logger = logging.getLogger(__name__)

class LLMService:
    """LLM service using Gemini for conversation and booking detection"""
    
    def __init__(self):
        self.settings = get_settings()
        
        genai.configure(api_key=self.settings.GEMINI_API_KEY)
        
        self.model = genai.GenerativeModel(
            model_name=self.settings.GEMINI_MODEL,
            generation_config={
                "max_output_tokens": self.settings.MAX_TOKENS,
                "temperature": self.settings.TEMPERATURE,
            }
        )
        
        self.system_prompt = get_system_prompt()
        
        logger.info(f"✅ LLM Service initialized (Gemini {self.settings.GEMINI_MODEL})")
    
    async def generate_response(self, conversation_history: List[Message], user_message: str) -> str:
        """
        Generate response to user message
        
        Args:
            conversation_history: Previous messages (excluding current user message)
            user_message: Current user message
        
        Returns:
            Assistant response text
        """
        try:
            # Build chat history: system prompt + conversation history
            history = [{"role": "user", "parts": [self.system_prompt]}]
            
            for msg in conversation_history:
                role = "user" if msg.role == MessageRole.USER else "model"
                history.append({"role": role, "parts": [msg.content]})
            
            # Run blocking API call in executor
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self._generate_sync(history, user_message)
            )
            
            assistant_message = response.text.strip()
            
            logger.info(f"🤖 Generated response: {assistant_message[:100]}...")
            
            return assistant_message
        
        except Exception as e:
            logger.error(f"❌ LLM Error: {e}", exc_info=True)
            return "I apologize, I'm having trouble right now. Could you please repeat that?"
    
    def _generate_sync(self, history: List[dict], user_message: str):
        """Synchronous helper for Gemini API call"""
        chat = self.model.start_chat(history=history)
        return chat.send_message(user_message)
    
    async def detect_booking_intent(self, conversation_history: List[Message]) -> BookingIntent:
        """
        Analyze conversation to detect booking intent and extract details
        
        Args:
            conversation_history: All messages in conversation
        
        Returns:
            BookingIntent with extracted information
        """
        try:
            conversation_text = "\n".join([
                f"{'Customer' if msg.role == MessageRole.USER else 'Maria'}: {msg.content}"
                for msg in conversation_history
            ])
            
            analysis_prompt = f"""Analyze this restaurant conversation and extract booking information.

Conversation:
{conversation_text}

Return a JSON object with:
{{
  "has_booking_intent": true/false,
  "customer_name": "name or null",
  "date": "YYYY-MM-DD or null",
  "time": "HH:MM or null",
  "party_size": number or null,
  "phone": "phone or null",
  "special_requests": "requests or null"
}}

Only return the JSON object, nothing else."""
            
            # Run blocking API call in executor
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.model.generate_content(analysis_prompt)
            )
            
            result_text = response.text.strip()
            
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                booking_data = json.loads(json_match.group())
                intent = BookingIntent(**booking_data)
                
                if intent.has_booking_intent:
                    logger.info(f"📅 Booking detected: {intent.customer_name}, {intent.date} {intent.time}, party of {intent.party_size}")
                
                return intent
            else:
                return BookingIntent(has_booking_intent=False)
        
        except Exception as e:
            logger.error(f"❌ Booking detection error: {e}", exc_info=True)
            return BookingIntent(has_booking_intent=False)
    
    async def generate_summary(self, conversation_history: List[Message]) -> str:
        """
        Generate summary of the conversation
        
        Args:
            conversation_history: All messages
        
        Returns:
            Summary text
        """
        try:
            conversation_text = "\n".join([
                f"{'Customer' if msg.role == MessageRole.USER else 'Maria'}: {msg.content}"
                for msg in conversation_history
            ])
            
            summary_prompt = f"""Summarize this restaurant phone call in 2-3 sentences.

Conversation:
{conversation_text}

Include:
- Main purpose (inquiry, booking, etc.)
- Key details (what they asked about, booking details if any)
- Outcome (resolved, booked, declined, etc.)

Summary:"""
            
            # Run blocking API call in executor
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.model.generate_content(summary_prompt)
            )
            
            summary = response.text.strip()
            
            logger.info(f"📝 Generated summary")
            
            return summary
        
        except Exception as e:
            logger.error(f"❌ Summary error: {e}", exc_info=True)
            return "Call summary unavailable"
    
    async def rate_call(self, conversation_history: List[Message], booking_created: bool) -> int:
        """
        Rate the call quality (1-5)
        
        Args:
            conversation_history: All messages
            booking_created: Whether a booking was made
        
        Returns:
            Rating from 1-5
        """
        try:
            score = 3
            
            if booking_created:
                score += 1
            
            # Minimum 6 messages = 3 exchanges (greeting + 2 user interactions)
            if len(conversation_history) >= 6:
                score += 1
            
            score = min(5, score)
            
            logger.info(f"⭐ Call rated: {score}/5")
            
            return score
        
        except Exception as e:
            logger.error(f"❌ Rating error: {e}", exc_info=True)
            return 3

@lru_cache()
def get_llm_service() -> LLMService:
    """Get singleton LLM service"""
    return LLMService()
