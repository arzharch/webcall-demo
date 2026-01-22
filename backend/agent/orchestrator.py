import json
from datetime import datetime
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
        # Added check_availability here so 'check_availability' intent has a tool to use
        self.booking_query_executor = self._build_agent_executor(
            self.llm, [cancel_booking, find_booking, update_booking, check_availability, make_booking], "booking assistant"
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
                    "You are Maria, the friendly maître d' at Bella Cucina restaurant.\n"
                    "Today is {current_date}.\n\n"
                    "CRITICAL: Keep ALL responses SHORT - 1-3 sentences maximum. Be conversational.\n\n"
                    "You are a {prefix}. You have these tools available:\n{tools}\n\n"
                    "RESPONSE FORMAT (Strictly Follow This):\n"
                    "When you need to use a tool:\n"
                    "Thought: [your brief reasoning]\n"
                    "Action: [EXACT tool name from [{tool_names}] ONLY. Do NOT add arguments/parentheses here.]\n"
                    "Action Input: [JSON object with arguments. Ensure keys match tool definition.]\n"
                    "Observation: [tool result]\n"
                    "... (repeat if needed)\n"
                    "Thought: I now have enough information\n"
                    "Final Answer: [brief, friendly response to customer]\n\n"
                    "When you have the Final Answer, respond naturally and briefly."
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}\n{agent_scratchpad}"),
            ]
        ).partial(
            prefix=system_message_prefix, 
            tools=tool_descriptions, 
            tool_names=tool_names,
            current_date=datetime.now().strftime("%A, %B %d, %Y")
        )
        prompt = base_prompt
        agent = create_react_agent(llm, tools, prompt)
        return AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

    def _validate_intent(self, intent_output: str) -> str:
        """
        Validates and sanitizes the intent output.
        If it's not a valid category, defaults to 'general_query'.
        """
        # Clean the intent output more aggressively
        intent_output = intent_output.strip().lower().replace("'", "").replace('"', "")
        
        valid_intents = [
            "make_booking", "update_booking", "cancel_booking",
            "find_booking", "check_availability", "search_menu",  
            "general_query", "off_topic"
        ]
        
        # Clean the output
        cleaned = intent_output.strip().lower().replace('\\', '').replace('`', '').replace("'", "").replace('"', "")
        
        # Strict match only.
        # Fallbacks like "substring match" are dangerous if the LLM hallucinated a paragraph.
        # If the LLM output "I think the user wants to make_booking", strict match fails.
        # But we previously had "Respond with ONLY the intent".
        if cleaned in valid_intents:
            return cleaned
        
        # Super safe fallback for "off_topic" if the LLM output gibberish or refused
        if "off_topic" in cleaned or "irrelevant" in cleaned:
             return "off_topic"
             
        # If truly unrecognizable, general_query is the safest default
        # because the general chain has better conversational abilities to recovery.
        return "general_query"

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
                    "7. 'general_query' - Greetings, small talk, ending conversation, polite closing, CONFIRMATIONS (e.g. 'Yes', 'Okay'), CLARIFICATIONS, or Short Answers to questions (e.g. 'Bob', '7pm').\n"
                    "8. 'off_topic' - Inputs completely unrelated to the restaurant (e.g. math questions, general knowledge, 'how tall is eiffel tower', code requests).\n\n"
                    "IMPORTANT RULES:\n"
                    "- If the input is SHORT (1-2 words) appearing in the middle of a flow, prefer 'general_query' or 'make_booking' over 'off_topic'.\n"
                    "- AMBIGUOUS INPUT: If user says something vague like 'Kadin' or 'Maybe', classify as 'general_query' so the agent can ask to clarify, NOT 'off_topic'.\n"
                    "- Greetings like 'hi', 'hello', 'hey' are ALWAYS 'general_query'\n"
                    "- Closing remarks like 'thank you', 'thanks', 'bye', 'goodbye', 'ok thanks' are ALWAYS 'general_query'\n"
                    "- Small talk like 'how are you?' is ALWAYS 'general_query'\n"
                    "- Random text or gibberish -> 'off_topic' (ONLY if truly nonsensical)\n"
                    "- Consider the conversation context - if they're in the middle of booking, related questions are likely booking-related\n"
                    "- Only switch context if the user EXPLICITLY asks for a different task.\n\n"
                    "Respond with ONLY the intent category name (e.g., make_booking). Do NOT explain yourself."
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ]
        )
        # The output of this chain will be the intent category string
        return router_prompt | llm | RunnableLambda(lambda x: self._validate_intent(x.content))

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
                    "GUARDRAILS & BOUNDARIES:\n"
                    "- If the user input is gibberish, unintelligible, or completely random characters (e.g. 'sdfhjksd'), respond with ONLY: ERROR_GIBBERISH\n"
                    "- If the user asks about off-topic subjects (general knowledge, math, history, other places, 'how tall is eiffel tower'), respond with: 'Please do not disturb, I only handle restaurant queries.'\n\n"
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

    async def _handle_availability_check(self, state: SessionState, user_message: str):
        """Wrapper to call booking manager since RunnableLambda doesn't await properly in branch"""
        return await self.booking_manager.handle_booking_conversation(state, user_message)

    async def process_message(
        self, state: SessionState, user_message: str
    ) -> AsyncGenerator[str, None]:
        """
        Processes a user message, classifies intent, routes to the appropriate chain,
        and yields response chunks.
        """
        # ... existing guardrails ...
        
        # Explicit handling for check_availability routing to avoid RunnableBranch async issues
        # that result in "<coroutine object ...>" being returned as string.
        
        # Safety Truncation: Prevent prompt injection via massive inputs
        safe_message = user_message[:1000] if len(user_message) > 1000 else user_message
        
        # Add the current user message to the conversation history
        # (We store the full message for record, but use safe_message for processing if distinct)
        state.conversation_history.append(HumanMessage(content=user_message))

        # Classify the intent of the user's message WITH conversation history for context
        intent_category = await self.intent_classifier_chain.ainvoke({
            "input": safe_message,
            "chat_history": state.conversation_history[:-1]  # Exclude the just-added user message
        })
        state.current_intent = intent_category  # Update state with detected intent

        logger.info(f"Detected intent: {state.current_intent}")
        
        # Immediate short-circuit for off-topic/garbage to save tokens
        if state.current_intent == "off_topic":
            response = "I'm sorry, but I only handle restaurant bookings and queries. Please let me know if you need help with a reservation."
            state.conversation_history.append(AIMessage(content=response))
            yield response
            return

        # Handle make_booking AND check_availability with slot-filling conversation manager
        # We group them because checking availability usually requires the same slots (date/time/party)
        if state.current_intent in ["make_booking", "check_availability"] or state.awaiting_confirmation:
            response = await self.booking_manager.handle_booking_conversation(state, user_message)
            state.conversation_history.append(AIMessage(content=response))
            logger.info(f"Updated session state: {state}")
            yield response
            return

        # For other intents, use the routing logic
        # ... logic for update/cancel/find/menu etc ...
        
        full_orchestrator_chain = RunnableBranch(
            (
                lambda x: x["current_intent"]
                in [
                    "update_booking",
                    "cancel_booking",
                    "find_booking",
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
            content_to_yield = ""
            
            if isinstance(chunk, dict):
                if "output" in chunk:
                    content_to_yield = chunk["output"]
                elif "messages" in chunk:
                    for msg in chunk["messages"]:
                        if isinstance(msg, AIMessage):
                            content_to_yield += msg.content
                if "actions" in chunk:
                    for action in chunk["actions"]:
                        logger.info(f"Agent Action: {action.tool} - {action.tool_input}")
                if "steps" in chunk:
                    for step in chunk["steps"]:
                        logger.info(f"Tool Observation: {step.observation}")
            elif isinstance(chunk, AIMessage):
                content_to_yield = chunk.content
            else:
                content_to_yield = str(chunk)

            # Check for specific error tokens from General Chain
            if "ERROR_GIBBERISH" in content_to_yield:
                state.confusion_count += 1
                if state.confusion_count >= 2:
                    content_to_yield = "I'm having trouble understanding. A human agent will call you back soon to assist you."
                    # Reset or end session logic could go here
                else:
                    content_to_yield = "I didn't quite get that. Could you please rephrase?"
            else:
                # Reset confusion count if we got a normal response (implying normal input)
                # But only if it's not a streaming chunk in the middle of a sentence
                # Ideally check at the end, but for simplicity:
                if len(content_to_yield.strip()) > 5:
                     state.confusion_count = 0

            full_response += content_to_yield
            yield content_to_yield

        # Append the AI's full response to the conversation history
        state.conversation_history.append(AIMessage(content=full_response))
        logger.info(f"Updated session state: {state}")