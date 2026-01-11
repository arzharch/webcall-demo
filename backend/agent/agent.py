import sqlite3
from contextvars import ContextVar
from datetime import datetime
from typing import AsyncGenerator, Optional

from langchain.agents import AgentExecutor, tool
from langchain.pydantic_v1 import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from agent.state import SessionState
from backend.config import get_settings
from backend.services.restaurant import get_restaurant_template


settings = get_settings()
restaurant_template = get_restaurant_template()
SESSION_CONTEXT: ContextVar[dict] = ContextVar("SESSION_CONTEXT", default={})


class MenuSearchInput(BaseModel):
    query: str = Field(description="The guest's question about the menu.")


class AvailabilityCheckInput(BaseModel):
    date: str = Field(description="Desired date in YYYY-MM-DD format.")
    time: str = Field(description="Desired time, e.g. '19:30' or '7 PM'.")
    party_size: int = Field(description="Party size between 1 and 12.")
    special_requests: Optional[str] = Field(
        default=None, description="Optional notes such as allergies or celebrations."
    )
    caller_name: Optional[str] = Field(
        default=None, description="Caller name if already captured in the call setup."
    )
    session_id: Optional[str] = Field(
        default=None, description="Internal session identifier for logging."
    )


def _menu_lookup(query: str) -> str:
    query_lower = query.lower()
    matches = []
    for item in restaurant_template.menu:
        haystack = f"{item['name']} {item['description']} {' '.join(item.get('dietary', []))}".lower()
        if query_lower in haystack:
            matches.append(item)
    if not matches:
        return "I did not find an exact match, but we offer handcrafted pasta, wood-fired pizzas, and seasonal chef's specials."

    top = matches[:3]
    lines = [
        f"{entry['name']} ({entry['price']}): {entry['description']}"
        for entry in top
    ]
    return "\n".join(lines)


def _normalize_time(raw_time: str) -> Optional[str]:
    raw = raw_time.strip().upper()
    patterns = ["%H:%M", "%I:%M %p", "%I %p"]
    for pattern in patterns:
        try:
            parsed = datetime.strptime(raw, pattern)
            return parsed.strftime("%H:%M")
        except ValueError:
            continue
    return None


@tool(args_schema=MenuSearchInput)
def search_menu(query: str) -> str:
    """Answer detailed menu questions using the restaurant knowledge base."""
    return _menu_lookup(query)


@tool(args_schema=AvailabilityCheckInput)
def check_availability(
    date: str,
    time: str,
    party_size: int,
    special_requests: Optional[str] = None,
    caller_name: Optional[str] = None,
    session_id: Optional[str] = None,
) -> str:
    """Look up a table slot and tentatively book it in SQLite."""

    ctx = SESSION_CONTEXT.get({})
    active_session = session_id or ctx.get("session_id")
    effective_name = caller_name or ctx.get("caller_name") or "Guest"

    if not active_session:
        return "I could not verify the session. Please restart the call."

    try:
        target_date = datetime.fromisoformat(date)
    except ValueError:
        return "Could you share the date in YYYY-MM-DD format?"

    slot = _normalize_time(time)
    if slot is None:
        return "I can book every 30 minutes. Please use a format like 19:30."

    if not (1 <= party_size <= settings.MAX_PARTY_SIZE):
        return f"We can host parties up to {settings.MAX_PARTY_SIZE} guests."

    slots_for_day = restaurant_template.generate_slots_for_date(target_date)
    if slot not in slots_for_day:
        return (
            f"We seat between {settings.RESERVATION_SERVICE_START} and {settings.RESERVATION_SERVICE_END}."
        )

    event = restaurant_template.get_event_for_day(target_date.strftime("%A"))

    with sqlite3.connect(settings.SQLITE_DB_PATH) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        capacity = conn.execute(
            """
            SELECT COUNT(*) FROM reservations
            WHERE reservation_date = ? AND reservation_time = ? AND status != 'cancelled'
            """,
            (date, slot),
        ).fetchone()[0]

        if capacity >= settings.TABLES_PER_SLOT:
            return f"We are fully booked at {slot} on {date}. Could we try another time?"

        conn.execute(
            """
            INSERT INTO reservations (
                session_id, caller_name, reservation_date, reservation_time,
                party_size, status, special_requests, event_tag,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, 'pending', ?, ?, datetime('now'), datetime('now'))
            """,
            (
                active_session,
                effective_name,
                date,
                slot,
                party_size,
                special_requests,
                event["title"] if event else None,
            ),
        )

        conn.commit()

    confirmation = (
        f"I penciled you in for {party_size} guests on {date} at {slot}. "
        "I'll send a confirmation shortly."
    )
    if event:
        confirmation += f" That evening features our {event['title'].lower()}!"
    if special_requests:
        confirmation += f" Noted your request: {special_requests}."
    return confirmation


class BellaAgent:
    """The 'brain' of the voice bot, powered by LangChain."""

    def __init__(self, google_api_key: str):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=google_api_key,
            temperature=0.3,
        )
        self.tools = [search_menu, check_availability]
        self.agent_executor = self._create_agent_executor()

    def _create_agent_executor(self):
        """Creates the LangChain agent and executor."""
        system_prompt = (
            "You are Maria, Bella Cucina's concierge. Keep replies under 3 sentences, "
            "sound natural for a phone call, and collect booking details methodically. "
            f"The dining room has {settings.TOTAL_TABLES} tables with up to {settings.TABLES_PER_SLOT} "
            "bookable per 30-minute slot. Business hours run from "
            f"{settings.RESERVATION_SERVICE_START} to {settings.RESERVATION_SERVICE_END}. "
            "Use check_availability once you know date, time, and party size, and include the caller's name."
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        return AgentExecutor(agent=self.llm | prompt, tools=self.tools, verbose=True)

    async def process_message(
        self, state: SessionState, user_message: str
    ) -> AsyncGenerator[str, None]:
        """Processes a user message, invokes tools, and streams the reply."""
        context_token = SESSION_CONTEXT.set(
            {"session_id": state.session_id, "caller_name": state.caller_name}
        )

        response_stream = self.agent_executor.astream(
            {
                "input": user_message,
                "chat_history": state.get_formatted_history(),
            }
        )

        try:
            async for chunk in response_stream:
                if "output" in chunk:
                    yield chunk["output"]
        finally:
            SESSION_CONTEXT.reset(context_token)
