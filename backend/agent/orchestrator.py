from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from typing import Any, Dict, List, AsyncIterator
import asyncio
import json
import logging

from backend.config import get_settings
from backend.models import SessionState, MessageRole
from backend.agent.prompts import get_system_prompt
from backend.services.rag_service import get_rag_service
from backend.services.crm_service import get_crm_service

logger = logging.getLogger(__name__)
settings = get_settings()

# ==================== TOOLS ====================

@tool
async def search_menu(query: str) -> str:
    """Search menu for items matching query.
    
    Args:
        query: What menu items to search for (e.g., 'vegetarian', 'pasta', 'desserts')
    
    Returns:
        JSON string with matching menu items and prices
    """
    try:
        rag = get_rag_service()
        results = await rag.search_menu(query)
        
        items = []
        for result in results:
            items.append({
                "name": result.document.metadata.get("name", "Unknown"),
                "price": result.document.metadata.get("price", "N/A"),
                "description": result.document.content[:200],
                "score": round(result.score, 3)
            })
        
        return json.dumps({"results": items, "count": len(items)}, indent=2)
    except Exception as e:
        logger.error(f"Menu search error: {e}")
        return json.dumps({"error": str(e), "results": []})

@tool
async def check_availability(date: str, time: str, party_size: int) -> str:
    """Check if restaurant has tables available.
    
    Args:
        date: Reservation date (YYYY-MM-DD format)
        time: Reservation time (HH:MM 24h format)
        party_size: Number of people
    
    Returns:
        JSON with availability status
    """
    try:
        # Mock availability logic - always available for demo
        available = True
        
        if available:
            return json.dumps({
                "available": True,
                "date": date,
                "time": time,
                "party_size": party_size,
                "message": f"Great! We have availability for {party_size} on {date} at {time}"
            })
        else:
            return json.dumps({
                "available": False,
                "date": date,
                "time": time,
                "party_size": party_size,
                "message": f"Sorry, we're fully booked for {party_size} on {date} at {time}"
            })
    except Exception as e:
        logger.error(f"Availability check error: {e}")
        return json.dumps({"error": str(e), "available": False})

@tool
async def create_reservation(
    date: str,
    time: str,
    party_size: int,
    customer_name: str,
    phone: str = None,
    special_requests: str = None
) -> str:
    """Create a confirmed reservation.
    
    Args:
        date: Reservation date (YYYY-MM-DD)
        time: Reservation time (HH:MM 24h)
        party_size: Number of people
        customer_name: Customer's name
        phone: Customer's phone number
        special_requests: Any special requests or dietary restrictions
    
    Returns:
        Reservation confirmation with ID
    """
    try:
        from datetime import datetime
        
        reservation_id = f"RES_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        confirmation = {
            "reservation_id": reservation_id,
            "date": date,
            "time": time,
            "party_size": party_size,
            "customer_name": customer_name,
            "phone": phone,
            "special_requests": special_requests,
            "status": "confirmed",
            "message": f"Your reservation is confirmed! Confirmation ID: {reservation_id}"
        }
        
        # Save to CRM
        crm = get_crm_service()
        await crm.create_ticket(
            call_id="temp",
            customer_name=customer_name,
            customer_phone=phone,
            details=confirmation
        )
        
        return json.dumps(confirmation, indent=2)
    except Exception as e:
        logger.error(f"Reservation creation error: {e}")
        return json.dumps({"error": str(e)})

@tool
async def get_restaurant_info(topic: str = None) -> str:
    """Get general restaurant information.
    
    Args:
        topic: Specific info to retrieve (hours, location, policies, etc.)
    
    Returns:
        Restaurant information as string
    """
    try:
        rag = get_rag_service()
        info = await rag.get_restaurant_info(topic)
        return str(info)
    except Exception as e:
        logger.error(f"Restaurant info error: {e}")
        return f"Error retrieving info: {str(e)}"

# ==================== ORCHESTRATOR ====================

class ConversationOrchestrator:
    """LangChain-based conversation orchestrator"""
    
    def __init__(self, session: SessionState):
        self.session = session
        self.settings = get_settings()
        self.tools = [search_menu, check_availability, create_reservation, get_restaurant_info]
        
        # Initialize Gemini LLM
        self.llm = ChatGoogleGenerativeAI(
            model=self.settings.GEMINI_MODEL,
            google_api_key=self.settings.GEMINI_API_KEY,
            temperature=self.settings.TEMPERATURE,
            max_output_tokens=self.settings.MAX_TOKENS
        )
        
        # Bind tools to LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # Memory for context
        self.memory = ConversationBufferWindowMemory(
            k=self.settings.CONTEXT_WINDOW,
            return_messages=True
        )
    
    async def process_message(self, user_input: str) -> AsyncIterator[str]:
        """Process user message and stream response chunks.
        
        Args:
            user_input: User's text input from STT
        
        Yields:
            Response chunks for TTS streaming
        """
        try:
            # Add user message to session
            self.session.add_message(MessageRole.USER, user_input)
            
            # Add to memory
            self.memory.chat_memory.add_message(
                HumanMessage(content=user_input)
            )
            
            # Build context for LLM
            system_prompt = get_system_prompt(
                intent=self.session.current_intent,
                context_data=self.session.context_data
            )
            
            messages = self.memory.chat_memory.messages
            
            # Call LLM with tools
            response = await asyncio.to_thread(
                self.llm_with_tools.invoke,
                messages
            )
            
            # Handle tool calls if present
            if hasattr(response, 'tool_calls') and response.tool_calls:
                await self._handle_tool_calls(response.tool_calls, messages)
                response = response.content
            else:
                response = response.content if hasattr(response, 'content') else str(response)
            
            # Add assistant response to session
            self.session.add_message(MessageRole.ASSISTANT, response)
            self.memory.chat_memory.add_message(AIMessage(content=response))
            
            # Stream response in chunks (sentence by sentence)
            sentences = self._split_sentences(response)
            for sentence in sentences:
                if sentence.strip():
                    yield sentence + " "
                    await asyncio.sleep(0.1)  # Small delay for streaming effect
        
        except Exception as e:
            logger.error(f"Orchestrator error: {e}")
            error_response = "I apologize for the confusion. Could you please repeat that?"
            self.session.add_message(MessageRole.ASSISTANT, error_response)
            yield error_response
    
    async def _handle_tool_calls(self, tool_calls: List[Dict], messages: List) -> None:
        """Handle tool calls from LLM"""
        for tool_call in tool_calls:
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})
            
            logger.info(f"Executing tool: {tool_name} with args: {tool_args}")
            
            # Execute the tool
            if tool_name == "search_menu":
                await search_menu.invoke(tool_args)
            elif tool_name == "check_availability":
                await check_availability.invoke(tool_args)
            elif tool_name == "create_reservation":
                await create_reservation.invoke(tool_args)
            elif tool_name == "get_restaurant_info":
                await get_restaurant_info.invoke(tool_args)
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences for streaming"""
        import re
        
        # Split on punctuation but keep it
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s for s in sentences if s.strip()]
