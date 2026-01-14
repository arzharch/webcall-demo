import json
from typing import AsyncGenerator, Callable, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableBranch, RunnableLambda, RunnablePassthrough

try:
    from langchain.agents import AgentExecutor, create_react_agent
except ImportError:  # Fallback for older LangChain installs
    from langchain_classic.agents import AgentExecutor, create_react_agent
from loguru import logger

from agent.state import SessionState, Intent
from agent.booking_manager import BookingManager
from agent.tools import (
    check_availability,
    make_booking,
    cancel_booking,
    find_booking,
    update_booking,
    search_menu,
)


class ConversationOrchestrator:
    """
    Manages the conversation flow, intent routing, and tool execution for the Bella bot.
    """

    def __init__(self, llm: BaseChatModel):
        self.llm = llm  # Use the provided LLM instance for flexibility

        # Booking manager for slot-filling conversation flow
        self.booking_manager = BookingManager(llm)

        self.booking_tools = [check_availability, make_booking, cancel_booking, find_booking, update_booking]
        self.menu_tools = [search_menu]
        self.all_tools = self.booking_tools + self.menu_tools # For potential future use or debugging

        # For complex queries (cancel, find, update), use agent-based approach
        self.booking_query_executor = self._build_agent_executor(
            self.llm, [cancel_booking, find_booking, update_booking], "booking assistant"
        )
        self.menu_agent_executor = self._build_agent_executor(
            self.llm, self.menu_tools, "menu assistant"
        )
        self.general_conversation_chain = self._build_general_chain(self.llm)

        # Intent Classifier Chain
        self.intent_classifier_chain = self._build_intent_classifier_chain(self.llm)

    def _build_agent_executor(
        self, llm: BaseChatModel, tools: List[Callable], system_message_prefix: str
    ) -> AgentExecutor:
        """
        Builds a LangChain AgentExecutor for handling tool calls and conversational turns,
        specialized for a given set of tools.
        """
        tool_descriptions = "\n".join(
            [
                f"- {tool.name}: {getattr(tool, 'description', '') or 'No description provided.'}"
                for tool in tools
            ]
        )
        tool_names = ", ".join([tool.name for tool in tools])

        base_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are Maria, the friendly maître d' at Bella Cucina restaurant.\n\n"
                    "CRITICAL: Keep ALL responses SHORT - 1-3 sentences maximum. Be conversational.\n\n"
                    "You are a {prefix}. You have these tools available:\n{tools}\n\n"
                    "RESPONSE FORMAT:\n"
                    "When you need to use a tool:\n"
                    "Thought: [your brief reasoning]\n"
                    "Action: [tool name from: {tool_names}]\n"
                    "Action Input: [JSON with tool arguments]\n"
                    "Observation: [tool result]\n"
                    "... (repeat if needed)\n"
                    "Thought: I now have enough information\n"
                    "Final Answer: [brief, friendly response to customer]\n\n"
                    "When you have the Final Answer, respond naturally and briefly."
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}\n{agent_scratchpad}"),
            ]
        ).partial(prefix=system_message_prefix, tools=tool_descriptions, tool_names=tool_names)
        prompt = base_prompt
        agent = create_react_agent(llm, tools, prompt)
        return AgentExecutor(agent=agent, tools=tools, verbose=True)

    def _build_intent_classifier_chain(self, llm: BaseChatModel) -> Runnable:
        """
        Builds a chain to classify the user's intent using zero-shot classification with conversation context.
        """
        router_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert intent classifier for a restaurant reservation system. "
                    "Analyze the user's message in the context of the conversation and classify it into ONE of these categories:\n\n"
                    "1. 'make_booking' - User wants to create a NEW reservation (e.g., 'I want to book a table', 'Can I reserve for 4 people?')\n"
                    "2. 'update_booking' - User wants to CHANGE an existing reservation (e.g., 'Change my booking to 8pm', 'Can I modify my reservation?')\n"
                    "3. 'cancel_booking' - User wants to CANCEL a reservation (e.g., 'Cancel my booking', 'I need to cancel')\n"
                    "4. 'find_booking' - User wants to CHECK or LOOK UP an existing reservation (e.g., 'Do I have a reservation?', 'Check my booking')\n"
                    "5. 'check_availability' - User wants to CHECK if a time/date is available (e.g., 'Are you open Friday?', 'Do you have tables tonight?')\n"
                    "6. 'search_menu' - User asks about food, menu items, or restaurant offerings (e.g., 'What do you serve?', 'Do you have vegan options?')\n"
                    "7. 'general_query' - Greetings, small talk, questions about the restaurant, or anything else (e.g., 'Hi', 'How are you?', 'Where are you located?')\n\n"
                    "IMPORTANT RULES:\n"
                    "- Greetings like 'hi', 'hello', 'hey' are ALWAYS 'general_query'\n"
                    "- Small talk like 'how are you?' is ALWAYS 'general_query'\n"
                    "- Random text or gibberish is 'general_query'\n"
                    "- If uncertain, default to 'general_query'\n"
                    "- Consider the conversation context - if they're in the middle of booking, related questions are likely booking-related\n\n"
                    "Respond with ONLY the intent category name (e.g., make_booking). Do NOT use Markdown formatting, backticks, or escaping."
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ]
        )
        # The output of this chain will be the intent category string
        return router_prompt | llm | RunnableLambda(lambda x: x.content.strip().lower().replace('\\', '').replace('`', '').replace("'", "").replace('"', ""))

    def _build_general_chain(self, llm: BaseChatModel) -> Runnable:
        """
        Builds a simple chain for general conversation using the LLM directly.
        """
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are Maria, the friendly maître d' at Bella Cucina, a cozy Italian restaurant.\n\n"
                    "CONVERSATION STYLE:\n"
                    "- Keep responses SHORT and natural (1-3 sentences max)\n"
                    "- Be warm, friendly, and professional\n"
                    "- Speak like a real person having a conversation, not writing an essay\n"
                    "- You HELP customers, you are NOT the customer\n"
                    "- Don't repeat greetings or welcoming phrases multiple times\n"
                    "- If asked multiple questions, answer them briefly\n\n"
                    "RESTAURANT INFO:\n"
                    "- Name: Bella Cucina\n"
                    "- Cuisine: Italian\n"
                    "- Hours: 5:00 PM - 11:00 PM daily\n"
                    "- Services: Reservations, dine-in, takeout\n\n"
                    "RESPONSES:\n"
                    "- Greetings → Respond warmly and offer help\n"
                    "- Questions about restaurant → Answer briefly\n"
                    "- Booking requests → Acknowledge and help them get started\n"
                    "- Menu questions → Provide brief info or offer to check details\n"
                    "- Small talk → Be friendly but guide toward how you can help\n\n"
                    "Remember: Keep it SHORT and conversational!"
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ]
        )
        return prompt | llm

    async def process_message(
        self, state: SessionState, user_message: str
    ) -> AsyncGenerator[str, None]:
        """
        Processes a user message, classifies intent, routes to the appropriate chain,
        and yields response chunks.
        """
        # Add the current user message to the conversation history
        state.conversation_history.append(HumanMessage(content=user_message))

        # Classify the intent of the user's message WITH conversation history for context
        intent_category = await self.intent_classifier_chain.ainvoke({
            "input": user_message,
            "chat_history": state.conversation_history[:-1]  # Exclude the just-added user message
        })
        state.current_intent = intent_category  # Update state with detected intent

        logger.info(f"Detected intent: {state.current_intent}")

        # Handle make_booking with slot-filling conversation manager
        if state.current_intent == "make_booking" or state.awaiting_confirmation:
            response = await self.booking_manager.handle_booking_conversation(state, user_message)
            state.conversation_history.append(AIMessage(content=response))
            logger.info(f"Updated session state: {state}")
            yield response
            return

        # For other intents, use the routing logic
        full_orchestrator_chain = RunnableBranch(
            (
                lambda x: x["current_intent"]
                in [
                    "update_booking",
                    "cancel_booking",
                    "find_booking",
                    "check_availability",
                ],
                self.booking_query_executor
            ),
            (
                lambda x: x["current_intent"] == "search_menu",
                self.menu_agent_executor
            ),
            self.general_conversation_chain,
        )

        # Prepare input for the full orchestrator chain
        chain_input = {
            "input": user_message,
            "chat_history": state.conversation_history[:-1],  # Pass history excluding current user message (it's 'input')
            "current_intent": state.current_intent,  # Pass current intent for the branch condition
        }

        full_response = ""
        # Invoke the chosen chain and stream its output
        async for chunk in full_orchestrator_chain.astream(chain_input):
            # Process chunks based on whether they come from an AgentExecutor or a simple LLM chain
            if isinstance(chunk, dict):
                # AgentExecutor chunks (e.g., {'output': '...', 'actions': [...], 'steps': [...]})
                if "output" in chunk:
                    full_response += chunk["output"]
                    yield chunk["output"]
                elif "messages" in chunk:
                    # Sometimes agent returns messages directly, particularly if it's the final output
                    for msg in chunk["messages"]:
                        if isinstance(msg, AIMessage):
                            full_response += msg.content
                            yield msg.content
                # Log agent actions and tool observations for debugging
                if "actions" in chunk:
                    for action in chunk["actions"]:
                        logger.info(f"Agent Action: {action.tool} - {action.tool_input}")
                if "steps" in chunk:
                    for step in chunk["steps"]:
                        logger.info(f"Tool Observation: {step.observation}")
            elif isinstance(chunk, AIMessage):
                # Simple LLM chain chunk (e.g., AIMessage(content='...'))
                full_response += chunk.content
                yield chunk.content
            else:
                # Fallback for any other unexpected chunk types (e.g., direct string output)
                full_response += str(chunk)
                yield str(chunk)

        # Append the AI's full response to the conversation history
        state.conversation_history.append(AIMessage(content=full_response))
        logger.info(f"Updated session state: {state}")