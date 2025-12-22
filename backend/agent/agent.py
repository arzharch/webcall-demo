import os
from typing import AsyncGenerator

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableBranch, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, tool
from langchain.pydantic_v1 import BaseModel, Field

from agent.state import SessionState


# --- LangChain Tools ---

@tool
def search_menu(query: str) -> str:
    """
    Searches the restaurant's menu items and answers questions about them.
    Use this for any questions related to food, drinks, ingredients, or prices.
    Example: 'What's in the Carbonara?' or 'Do you have vegan options?'
    """
    # In a real implementation, this would query a vector database (FAISS).
    # For this MVP, we'll use a placeholder response.
    return "The Carbonara is a classic Roman dish with spaghetti, eggs, Pecorino Romano cheese, and guanciale. We also have a delicious vegetarian lasagna."

@tool
def check_availability(date: str, time: str, party_size: int) -> str:
    """
    Checks for table availability and makes a booking.
    Use this when the user wants to reserve a table.
    Captures date, time, and party_size. If any are missing, the LLM will ask for them.
    """
    # In a real implementation, this would check a booking system and create a ticket.
    return f"I've booked a table for {party_size} people on {date} at {time}. We look forward to seeing you!"

# --- Pydantic Models for Tool Input ---

class MenuSearchInput(BaseModel):
    query: str = Field(description="The user's question about the menu.")

class AvailabilityCheckInput(BaseModel):
    date: str = Field(description="The desired date for the booking, e.g., 'tomorrow' or '2024-08-15'.")
    time: str = Field(description="The desired time for the booking, e.g., '7 PM'.")
    party_size: int = Field(description="The number of people in the party.")


# --- The Orchestrator ---

class BellaAgent:
    """The 'brain' of the voice bot, powered by LangChain."""

    def __init__(self, google_api_key: str):
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=google_api_key, temperature=0)
        self.tools = [search_menu, check_availability]
        self.agent_executor = self._create_agent_executor()

    def _create_agent_executor(self):
        """Creates the LangChain agent and executor."""
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful and friendly restaurant receptionist named Bella. Your goal is to assist users with their questions and bookings. Be conversational and natural."),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        agent = AgentExecutor(agent=self.llm | prompt, tools=self.tools, verbose=True)
        return agent

    async def process_message(self, state: SessionState, user_message: str) -> AsyncGenerator[str, None]:
        """
        Processes a user message, invokes the appropriate tools, and streams the response.
        """
        # This is a simplified placeholder. A real implementation would use LCEL
        # with RunnableBranch for intent detection and routing.
        # For now, we'll pass the message directly to the agent.

        response_stream = self.agent_executor.astream(
            {
                "input": user_message,
                "chat_history": state.get_formatted_history(),
            }
        )

        async for chunk in response_stream:
            # This simplified streaming only yields the final output.
            # A more advanced version would stream tokens as they are generated.
            if "output" in chunk:
                yield chunk["output"]
