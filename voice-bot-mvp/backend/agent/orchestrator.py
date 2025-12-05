import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from typing import AsyncIterator

from agent.state import SessionState
from agent.prompts import SYSTEM_PROMPT
from config import get_settings
from services.rag_service import get_rag_service
from services.crm_service import get_crm_service
from models import BookingIntent, MessageRole

# --- Tools Definition ---
# These functions are decorated with @tool, turning them into LangChain tools
# that the agent can decide to call.

@tool
async def search_menu(query: str) -> str:
    """
    Use this tool to answer any user questions about menu items, ingredients,
    prices, or dietary options (e.g., vegetarian, gluten-free).
    """
    print(f"🛠️ Tool call: search_menu(query='{query}')")
    rag_service = get_rag_service()
    results = await rag_service.search(query=query, top_k=3)
    if not results:
        return "No relevant menu items found."
    return json.dumps([r.dict() for r in results])

@tool
async def get_restaurant_info(topic: str) -> str:
    """
    Use this tool to find general information about the restaurant, such as
    hours, location, address, phone number, parking, dress code, or policies.
    """
    print(f"🛠️ Tool call: get_restaurant_info(topic='{topic}')")
    rag_service = get_rag_service()
    results = await rag_service.search(query=topic, top_k=2)
    if not results:
        return "No information found on that topic."
    return "\n".join([r.content for r in results])

@tool
async def check_availability(date: str, time: str, party_size: int) -> str:
    """
    Checks if a table is available for a given date, time, and party size.
    This is a preliminary check and does not guarantee a reservation.
    """
    print(f"🛠️ Tool call: check_availability(date='{date}', time='{time}', party_size={party_size})")
    # In a real system, this would query a booking database. Here, we simulate it.
    if party_size > 12:
        return json.dumps({"available": False, "reason": "For parties larger than 12, please call the restaurant directly."}))
    # Mocking 80% availability
    import random
    available = random.random() < 0.8
    if available:
        return json.dumps({"available": True, "message": "A table is available at that time."})
    else:
        return json.dumps({"available": False, "reason": "Sorry, we are fully booked at that time. Would you like to try another time?"})

@tool
async def create_reservation(date: str, time: str, party_size: int, customer_name: str, call_id: str) -> str:
    """
    Creates a confirmed reservation and a CRM ticket once all details have been
    gathered and confirmed by the user.
    """
    print(f"🛠️ Tool call: create_reservation(customer_name='{customer_name}', call_id='{call_id}')")
    crm_service = get_crm_service()
    booking_intent = BookingIntent(date=date, time=time, party_size=party_size, customer_name=customer_name)
    ticket = await crm_service.create_ticket_from_intent(booking_intent, call_id=call_id, summary=f"Reservation for {customer_name}")
    return json.dumps({
        "status": "success",
        "confirmation_id": ticket.id,
        "message": f"Reservation confirmed for {customer_name}. The confirmation ID is {ticket.id}."
    })

class ConversationOrchestrator:
    """Manages the conversation flow using a LangChain agent."""

    def __init__(self):
        self.settings = get_settings()
        self.llm = ChatGoogleGenerativeAI(
            model=self.settings.GEMINI_MODEL,
            temperature=self.settings.TEMPERATURE,
            google_api_key=self.settings.GEMINI_API_KEY,
            convert_system_message_to_human=True # Important for Gemini
        )
        self.tools = [search_menu, get_restaurant_info, check_availability, create_reservation]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        
        agent = create_tool_calling_agent(self.llm, self.tools, prompt)
        self.agent_executor = AgentExecutor(
            agent=agent, 
            tools=self.tools, 
            verbose=self.settings.AGENT_VERBOSE
        )

    async def stream_response(self, state: SessionState, user_message: str) -> AsyncIterator[str]:
        """
        Processes a user message using the agent and streams the response chunks.
        """
        history = []
        for msg in state.messages:
            if msg.role == MessageRole.USER:
                history.append(HumanMessage(content=msg.content))
            elif msg.role == MessageRole.ASSISTANT:
                history.append(AIMessage(content=msg.content))

        # Use astream_events to get a stream of events from the agent
        event_stream = self.agent_executor.astream_events(
            {
                "input": user_message,
                "chat_history": history,
                "call_id": state.call_id # Pass call_id to tools
            },
            version="v1"
        )
        
        # Process the stream and yield final output chunks
        async for event in event_stream:
            kind = event["event"]
            if kind == "on_llm_end":
                pass # This event contains the full response, we are streaming tokens
            elif kind == "on_chain_stream":
                # Event containing a chunk of the final output
                chunk = event["data"].get("chunk")
                if chunk and isinstance(chunk, AIMessage):
                    # Yield the content of the AIMessage chunk
                    yield chunk.content

_orchestrator_instance = None

def get_orchestrator() -> "ConversationOrchestrator":
    """Returns a singleton instance of the ConversationOrchestrator."""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = ConversationOrchestrator()
    return _orchestrator_instance
