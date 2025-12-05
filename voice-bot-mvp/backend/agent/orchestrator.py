from typing import AsyncIterator
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from agent.state import SessionState, update_session_state
from agent.prompts import SYSTEM_PROMPT
from config import get_settings
from services.rag_service import get_rag_service
from services.crm_service import get_crm_service
from models import BookingIntent

# --- Tools Definition ---
# Tools remain the same, but they will be invoked by a ToolNode in the graph.

@tool
async def search_menu(query: str) -> str:
    """Searches the restaurant's menu for specific dishes, ingredients, or dietary options."""
    rag_service = get_rag_service()
    results = await rag_service.search(query=query, top_k=3)
    return json.dumps([r.dict() for r in results]) if results else "No relevant menu items found."

@tool
async def get_restaurant_info(topic: str) -> str:
    """Finds general information about the restaurant (e.g., hours, location, policies)."""
    rag_service = get_rag_service()
    results = await rag_service.search(query=topic, top_k=2)
    return "\n".join([r.content for r in results]) if results else "No information found on that topic."

@tool
async def check_availability(date: str, time: str, party_size: int) -> str:
    """Checks for table availability."""
    if party_size > 12:
        return json.dumps({"available": False, "reason": "For parties larger than 12, please call the restaurant directly."})
    import random
    available = random.random() < 0.8
    return json.dumps({"available": available})

@tool
async def create_reservation(date: str, time: str, party_size: int, customer_name: str, call_id: str) -> str:
    """Creates a confirmed reservation and a CRM ticket."""
    crm_service = get_crm_service()
    booking_intent = BookingIntent(date=date, time=time, party_size=party_size, customer_name=customer_name)
    ticket = await crm_service.create_ticket_from_intent(booking_intent, call_id=call_id)
    return json.dumps({"status": "success", "confirmation_id": ticket.id})

# --- Graph Definition ---

class ConversationOrchestrator:
    """Manages the conversation flow using a LangGraph state machine."""

    def __init__(self):
        settings = get_settings()
        self.llm = ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            temperature=settings.TEMPERATURE,
            google_api_key=settings.GEMINI_API_KEY
        )
        
        # Bind tools
        self.tools = [search_menu, check_availability, create_reservation, get_restaurant_info]
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # Build graph
        self.graph = self._build_graph()
        self.memory = MemorySaver()
    
    def _build_graph(self):
        """Builds the LangGraph state machine."""
        builder = StateGraph(SessionState)

        def agent_node(state: SessionState):
            # Convert to LangChain messages
            messages = [
                HumanMessage(content=m.content) if m.role == MessageRole.USER 
                else AIMessage(content=m.content)
                for m in state.messages
            ]
            
            # Add system prompt
            system_msg = HumanMessage(content=SYSTEM_PROMPT)
            messages.insert(0, system_msg)
            
            # Call LLM
            response = self.llm_with_tools.invoke(messages)
            
            # Add to state
            state.messages.append(Message(
                role=MessageRole.ASSISTANT,
                content=response.content or "",
                tool_calls=response.tool_calls if hasattr(response, 'tool_calls') else None
            ))
            
            return state
        
        # Add nodes
        builder.add_node("agent", agent_node)
        builder.add_node("tools", ToolNode(self.tools))
        
        # Add edges
        builder.add_edge(START, "agent")
        builder.add_conditional_edges("agent", tools_condition)
        builder.add_edge("tools", "agent")
        
        return builder.compile(checkpointer=self.memory)
    
    async def stream_response(self, session_id: str, user_message: str) -> AsyncIterator[str]:
        """Stream response with proper sentence splitting for TTS"""
        
        # Get session state
        state = get_session_state(session_id, session_id)
        
        # Add user message
        state.messages.append(Message(
            role=MessageRole.USER,
            content=user_message
        ))
        
        config = {"configurable": {"thread_id": session_id}}
        
        # Stream from graph
        buffer = ""
        async for event in self.graph.astream(state, config, stream_mode="updates"):
            for node_name, node_output in event.items():
                if node_name == "agent":
                    # Extract message content
                    if hasattr(node_output, 'messages'):
                        latest = node_output.messages[-1]
                        content = latest.content if hasattr(latest, 'content') else str(latest)
                        
                        # Stream sentence by sentence for TTS
                        buffer += content
                        
                        # Split on sentence boundaries
                        sentences = re.split(r'([.!?]+\s+)', buffer)
                        
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
        update_session_state(session_id, state)