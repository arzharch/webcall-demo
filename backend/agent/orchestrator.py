import json
import logging
import random
import re
from typing import AsyncIterator
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from agent.state import SessionState, get_session_state, update_session_state
from agent.prompts import SYSTEM_PROMPT
from config import get_settings
from services.rag_service import get_rag_service
from services.crm_service import get_crm_service
from models import BookingIntent, MessageRole, Message

logger = logging.getLogger(__name__)
settings = get_settings()

# --- Tools Definition ---
@tool
async def search_menu(query: str) -> str:
    """Searches the restaurant's menu for specific dishes, ingredients, or dietary options."""
    try:
        rag_service = get_rag_service()
        results = await rag_service.search_menu(query, top_k=3)
        if results:
            items = []
            for r in results:
                items.append(f"{r.metadata.get('name', 'Item')}: {r.content} - {r.metadata.get('price', 'N/A')}")
            return "\n".join(items)
        return "No matching menu items found."
    except Exception as e:
        logger.error(f"search_menu error: {e}")
        return "Unable to search menu at this time."

@tool
async def get_restaurant_info(topic: str) -> str:
    """Finds general information about the restaurant (e.g., hours, location, policies)."""
    try:
        rag_service = get_rag_service()
        info = await rag_service.get_restaurant_info(topic)
        return info if info else "I don't have that information right now."
    except Exception as e:
        logger.error(f"get_restaurant_info error: {e}")
        return "Unable to retrieve restaurant info at this time."

@tool
async def check_availability(date: str, time: str, party_size: int) -> str:
    """Checks for table availability."""
    try:
        if party_size > 12:
            return json.dumps({"available": False, "reason": "Party size exceeds maximum of 12"})
        
        # Simulate availability check (80% chance of availability)
        available = random.random() < 0.8
        return json.dumps({
            "available": available,
            "date": date,
            "time": time,
            "party_size": party_size
        })
    except Exception as e:
        logger.error(f"check_availability error: {e}")
        return json.dumps({"available": False, "reason": "Error checking availability"})

@tool
async def create_reservation(date: str, time: str, party_size: int, customer_name: str, call_id: str) -> str:
    """Creates a confirmed reservation and a CRM ticket."""
    try:
        crm_service = get_crm_service()
        
        # Create booking intent
        booking = BookingIntent(
            date=date,
            time=time,
            party_size=party_size,
            customer_name=customer_name
        )
        
        # Create ticket
        ticket = await crm_service.create_ticket_from_intent(
            booking_intent=booking,
            call_id=call_id,
            summary=f"Reservation for {customer_name}, party of {party_size}"
        )
        
        return json.dumps({
            "success": True,
            "confirmation_number": ticket.id,
            "details": {
                "name": customer_name,
                "date": date,
                "time": time,
                "party_size": party_size
            }
        })
    except Exception as e:
        logger.error(f"create_reservation error: {e}")
        return json.dumps({"success": False, "error": "Failed to create reservation"})

# --- Orchestrator ---
class ConversationOrchestrator:
    """Manages the conversation flow using LangGraph state machine."""

    def __init__(self):
        self.settings = get_settings()
        self.llm = ChatGoogleGenerativeAI(
            model=self.settings.GEMINI_MODEL,
            temperature=self.settings.TEMPERATURE,
            google_api_key=self.settings.GEMINI_API_KEY,
            max_output_tokens=self.settings.MAX_TOKENS
        )
        
        # Define tools
        self.tools = [search_menu, get_restaurant_info, check_availability, create_reservation]
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # Build graph
        self.memory = MemorySaver()
        self.graph = self._build_graph()
        
        logger.info("✅ ConversationOrchestrator initialized")

    def _build_graph(self):
        """Build LangGraph state machine"""
        builder = StateGraph(dict)
        
        def agent_node(state: dict):
            """Agent node that calls LLM"""
            messages = state.get("messages", [])
            
            # Add system prompt if not present
            if not messages or not isinstance(messages[0], SystemMessage):
                messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
            
            # Call LLM
            response = self.llm_with_tools.invoke(messages)
            
            # Add response to messages
            return {"messages": messages + [response]}
        
        # Add nodes
        builder.add_node("agent", agent_node)
        builder.add_node("tools", ToolNode(self.tools))
        
        # Add edges
        builder.add_edge(START, "agent")
        builder.add_conditional_edges("agent", tools_condition)
        builder.add_edge("tools", "agent")
        
        return builder.compile(checkpointer=self.memory)

    async def stream_response(self, session_id: str, user_message: str) -> AsyncIterator[str]:
        """Stream response from the agent"""
        try:
            # Get or create session state
            state = get_session_state(call_id=session_id, session_id=session_id)
            
            # Add user message to state
            state.messages.append(Message(
                role=MessageRole.USER,
                content=user_message
            ))
            
            # Convert to LangChain messages
            lc_messages = [
                HumanMessage(content=m.content) if m.role == MessageRole.USER 
                else AIMessage(content=m.content)
                for m in state.messages
            ]
            
            config = {"configurable": {"thread_id": session_id}}
            
            # Stream from graph
            buffer = ""
            async for event in self.graph.astream({"messages": lc_messages}, config, stream_mode="values"):
                messages = event.get("messages", [])
                if messages:
                    latest = messages[-1]
                    
                    # Extract content from AIMessage
                    if hasattr(latest, "content") and isinstance(latest.content, str):
                        content = latest.content
                        
                        # Add to buffer
                        buffer += content
                        
                        # Split into sentences
                        sentences = re.split(r'([.!?]+\s+)', buffer)
                        
                        # Yield complete sentences
                        for i in range(0, len(sentences) - 1, 2):
                            if i + 1 < len(sentences):
                                sentence = sentences[i] + sentences[i + 1]
                                if sentence.strip():
                                    yield sentence.strip()
                        
                        # Keep incomplete sentence in buffer
                        buffer = sentences[-1] if len(sentences) % 2 == 1 else ""
            
            # Yield remaining buffer
            if buffer.strip():
                yield buffer.strip()
            
            # Update session state
            if lc_messages:
                latest_ai = lc_messages[-1]
                if hasattr(latest_ai, "content"):
                    state.messages.append(Message(
                        role=MessageRole.ASSISTANT,
                        content=latest_ai.content
                    ))
            
            update_session_state(session_id, state)
            
        except Exception as e:
            logger.error(f"stream_response error: {e}", exc_info=True)
            yield "I apologize, I'm having trouble processing that. Could you repeat your request?"

_orchestrator_instance = None

def get_orchestrator() -> ConversationOrchestrator:
    """Returns a singleton instance of the ConversationOrchestrator."""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = ConversationOrchestrator()
    return _orchestrator_instance