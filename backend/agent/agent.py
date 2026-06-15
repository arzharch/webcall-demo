"""Main agent orchestrator."""

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    from langchain_community.chat_models import ChatOpenAI

try:
    from langchain_ollama import ChatOllama
except ImportError:  # Fallback until langchain-ollama is installed everywhere
    from langchain_community.chat_models import ChatOllama

from agent.state import SessionState
from agent.orchestrator import ConversationOrchestrator


class BellaAgent:
    """Agent that orchestrates the conversation."""

    def __init__(self, state: SessionState):
        self.state = state
        # Use OpenAI GPT-3.5 Turbo
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        self.orchestrator = ConversationOrchestrator(llm=llm)

    async def respond(self, user_text: str) -> str:
        """
        Respond to user input by invoking the orchestrator.
        """
        # The orchestrator's process_message will update the conversation history
        # We need to collect the streamed response here.
        full_reply = ""
        async for chunk in self.orchestrator.process_message(self.state, user_text):
            full_reply += chunk
        
        return full_reply