from typing import AsyncIterator
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from agent.state import SessionState
from agent.prompts import SYSTEM_PROMPT
from config import get_settings
from services.rag_service import get_rag_service
from services.crm_service import get_crm_service
from models import BookingIntent

# --- Tools Definition ---
# These tools are the interface between the agent and the outside world.

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
        return json.dumps({"available": False, "reason": "For parties larger than 12, please call the restaurant directly."}))
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
        self.tools = [search_menu, get_restaurant_info, check_availability, create_reservation]
        llm = ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            temperature=settings.TEMPERATURE,
            google_api_key=settings.GEMINI_API_KEY,
            convert_system_message_to_human=True
        )
        self.model_with_tools = llm.bind_tools(self.tools)
        self.graph_app = self._build_graph()

    def _build_graph(self):
        """Builds the LangGraph state machine."""
        workflow = StateGraph(SessionState)

        def tools_condition(state: SessionState) -> str:
            """A function to decide whether to call tools or end the conversation."""
            last_message = state['messages'][-1]
            if last_message.tool_calls:
                return "tools"
            return END

        async def call_model(state: SessionState):
            """The primary node that calls the LLM. This is the 'brain' of the agent."""
            messages = state['messages']
            # The system prompt is now part of the history, which is more robust
            augmented_messages: list[BaseMessage] = [HumanMessage(content=SYSTEM_PROMPT)] + messages
            response = await self.model_with_tools.ainvoke(augmented_messages)
            return {"messages": [response]}

        tool_node = ToolNode(self.tools)

        workflow.add_node("agent", call_model)
        workflow.add_node("tools", tool_node)
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges("agent", tools_condition)
        workflow.add_edge("tools", "agent")

        # The checkpointer allows the graph to be stateful
        return workflow.compile(checkpointer=MemorySaver())

    async def stream_response(self, session_id: str, user_message: str) -> AsyncIterator[str]:
        """Processes a user message using the graph and streams the text response."""
        config = {"configurable": {"thread_id": session_id}}
        inputs = {"messages": [HumanMessage(content=user_message)]}

        # Stream events from the graph. We'll yield the content of AIMessage chunks.
        async for event in self.graph_app.astream(inputs, config=config, stream_mode="updates"):
            for value in event.values():
                new_messages = value.get('messages', [])
                if new_messages:
                    last_message = new_messages[-1]
                    if isinstance(last_message, AIMessage) and not last_message.tool_calls:
                        yield last_message.content

_orchestrator_instance = None

def get_orchestrator() -> "ConversationOrchestrator":
    """Returns a singleton instance of the ConversationOrchestrator."""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = ConversationOrchestrator()
    return _orchestrator_instance