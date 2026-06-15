import json
from datetime import datetime
from typing import AsyncGenerator, Callable, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableBranch, RunnableLambda, RunnablePassthrough

# FIX #17: Switch from ReAct to OpenAI function calling for better tool reliability
try:
    from langchain.agents import AgentExecutor, create_openai_tools_agent
    USE_OPENAI_TOOLS = True
except ImportError:
    try:
        from langchain.agents import AgentExecutor, create_react_agent
        USE_OPENAI_TOOLS = False
    except ImportError:  # Fallback for older LangChain installs
        from langchain_classic.agents import AgentExecutor, create_react_agent
        USE_OPENAI_TOOLS = False

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
        Builds a LangChain AgentExecutor for handling tool calls and conversational turns.
        Uses OpenAI function calling if available (more reliable than ReAct).
        """
        tool_descriptions = "\n".join(
            [
                f"- {tool.name}: {getattr(tool, 'description', '') or 'No description provided.'}"
                for tool in tools
            ]
        )
        
        if USE_OPENAI_TOOLS and hasattr(llm, 'bind_tools'):
            # Use OpenAI function calling for structured, reliable tool invocation
            logger.info(f"Building agent with OpenAI function calling for {system_message_prefix}")
            
            prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    "You are Bella, the warm and charming hostess at Bella Cucina, an authentic Italian restaurant.\n"
                    "Today is {current_date}.\n\n"
                    "PERSONA: Use a warm, welcoming tone. Feel free to use occasional Italian flair (e.g., 'Prego', 'Benvenuto') but keep it subtle. You are professional but not robotic.\n\n"
                    "CRITICAL: Keep ALL responses SHORT - 1-2 sentences maximum. Be conversational.\n\n"
                    "You are a {prefix}. Use the available tools to help customers.\n"
                    "Available tools:\n{tools}\n\n"
                    "RESPONSE RULES:\n"
                    "- After completing a booking, give a SHORT confirmation with booking ID\n"
                    "- If user says 'thank you' or 'bye', respond warmly and DO NOT repeat booking details\n"
                    "- Never say 'we have booked' before user confirms - say 'Shall I confirm?'\n"
                    "- AVOID REPETITION: If you just said something, assume the user heard it. Don't repeat full booking details unless they changed.\n"
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]).partial(
                prefix=system_message_prefix,
                tools=tool_descriptions,
                current_date=datetime.now().strftime("%A, %B %d, %Y")
            )
            
            agent = create_openai_tools_agent(llm, tools, prompt)
        else:
            # Fallback to ReAct agent
            logger.warning(f"OpenAI tools not available, using ReAct agent for {system_message_prefix}")
            tool_names = ", ".join([tool.name for tool in tools])
            
            base_prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    "You are Bella, the warm and charming hostess at Bella Cucina, an authentic Italian restaurant.\n"
                    "Today is {current_date}.\n\n"
                    "PERSONA: Use a warm, welcoming tone. Occasional Italian words are welcome.\n"
                    "CRITICAL: Keep ALL responses SHORT - 1-2 sentences maximum. Be conversational.\n\n"
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
            ]).partial(
                prefix=system_message_prefix, 
                tools=tool_descriptions, 
                tool_names=tool_names,
                current_date=datetime.now().strftime("%A, %B %d, %Y")
            )
            agent = create_react_agent(llm, tools, base_prompt)
        
        return AgentExecutor(
            agent=agent, 
            tools=tools, 
            verbose=True, 
            handle_parsing_errors=self._handle_tool_parsing_error
        )
    
    def _handle_tool_parsing_error(self, error: Exception) -> str:
        """
        Custom error handler for tool parsing failures.
        Logs the error and provides a clear fallback message.
        """
        logger.error(f"Tool parsing failed: {error}")
        # Return a clear instruction for the agent to rephrase or clarify
        return (
            "The tool call format was invalid. Please try again with the correct format:\n"
            "Action: [tool_name]\n"
            "Action Input: {\"parameter\": \"value\"}\n"
            "If you cannot determine the correct parameters, ask the user for clarification instead."
        )

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
                    "You are Bella, the friendly hostess at Bella Cucina, a cozy Italian restaurant.\n\n"
                    "CONVERSATION STYLE:\n"
                    "- Keep responses SHORT and natural (1-2 sentences max)\n"
                    "- Be warm, friendly, and professional\n"
                    "- Speak like a real person having a conversation\n"
                    "- You HELP customers, you are NOT the customer\n\n"
                    "CONTEXT AWARENESS:\n"
                    "- You have access to the full conversation history\n"
                    "- If user asks about their booking details, party size, time, etc. - check the chat history\n"
                    "- Answer questions about what was discussed in the conversation\n"
                    "- If user asks 'how many people' or 'what time' - look at previous messages for context\n\n"
                    "CRITICAL RESPONSE RULES:\n"
                    "- THANK YOU / THANKS / BYE → Respond warmly: 'You're welcome! Have a great day!' or 'Goodbye! We look forward to seeing you!'\n"
                    "- Don't repeat confirmation details after booking is complete\n"
                    "- Don't ask follow-up questions after user says thank you/bye\n\n"
                    "RESTAURANT INFO:\n"
                    "- Name: Bella Cucina\n"
                    "- Cuisine: Italian\n"
                    "- Hours: Weekdays 5-11 PM, Weekends 12-11 PM\n"
                    "- Services: Reservations, dine-in, takeout\n\n"
                    "GUARDRAILS:\n"
                    "- If input is gibberish/unintelligible, respond: ERROR_GIBBERISH\n"
                    "- If off-topic (general knowledge, math, etc.), respond: 'I can only help with restaurant bookings and questions.'\n"
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ]
        )
        return prompt | llm

    async def _handle_availability_check(self, state: SessionState, user_message: str):
        """Wrapper to call booking manager since RunnableLambda doesn't await properly in branch"""
        return await self.booking_manager.handle_booking_conversation(state, user_message)

    # Define keywords for fast intent switching detection
    INTENT_SWITCH_KEYWORDS = {
        "cancel_booking": {"cancel", "delete", "remove booking"},
        "update_booking": {"update", "change", "modify", "postpone", "reschedule", "move"},
        "find_booking": {"find", "look up", "check my", "my booking", "my reservation", "how many reservations"},
    }

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

        # Check for intent-switching keywords that override current flow
        lower_msg = user_message.lower()
        force_reclassify = False
        
        # Quick keyword check
        for intent, keywords in self.INTENT_SWITCH_KEYWORDS.items():
            if any(kw in lower_msg for kw in keywords):
                force_reclassify = True
                break

        # OPTIMIZATION: Skip intent classification if already in active booking flow
        # BUT allow re-classification if user mentions cancel/update/find keywords
        skip_classification = (
            not force_reclassify and
            (state.awaiting_confirmation or 
             (state.current_intent in ["make_booking", "check_availability"] and 
              not state.booking_slot.is_complete_for_new_booking()))
        )
        
        if not skip_classification:
            # Classify the intent of the user's message WITH conversation history for context
            intent_category = await self.intent_classifier_chain.ainvoke({
                "input": safe_message,
                "chat_history": state.conversation_history[:-1]  # Exclude the just-added user message
            })
            state.current_intent = intent_category  # Update state with detected intent
        else:
            logger.info(f"Skipping intent classification - continuing {state.current_intent} flow")

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

        # For other intents, use manual routing instead of RunnableBranch for better control
        # over buffering and error handling (specifically for ERROR_GIBBERISH)
        
        executor = None
        is_general = False
        
        if state.current_intent in ["update_booking", "cancel_booking", "find_booking"]:
            executor = self.booking_query_executor
        elif state.current_intent == "search_menu":
            executor = self.menu_agent_executor
        else:
            is_general = True

        # Prepare input for the chain/executor
        # Include last_booking_id context for update/cancel operations
        enhanced_input = user_message
        if state.current_intent in ["update_booking", "cancel_booking"] and state.last_booking_id:
            enhanced_input = f"{user_message} (Note: The user's most recent booking ID is {state.last_booking_id})"
        elif state.current_intent == "find_booking" and state.caller_name:
            enhanced_input = f"{user_message} (Note: The user's name is {state.caller_name})"
        
        chain_input = {
            "input": enhanced_input,
            "chat_history": state.conversation_history[:-1],
            "current_intent": state.current_intent,
        }

        full_response = ""

        if is_general:
            # Buffer general responses to safely check for ERROR_GIBBERISH without token splitting issues
            response_msg = await self.general_conversation_chain.ainvoke(chain_input)
            content_to_yield = response_msg.content
            
            if "ERROR_GIBBERISH" in content_to_yield:
                state.confusion_count += 1
                if state.confusion_count >= 2:
                    content_to_yield = "I'm having trouble understanding. A human agent will call you back soon to assist you."
                else:
                    content_to_yield = "I didn't quite get that. Could you please rephrase?"
            else:
                # Only reset confusion after 2 consecutive valid responses to avoid hair-trigger resets
                if len(content_to_yield.strip()) > 5 and state.confusion_count > 0:
                     state.confusion_count = max(0, state.confusion_count - 1)
            
            full_response += content_to_yield
            yield content_to_yield
            
        else:
            # Stream from AgentExecutor
            async for chunk in executor.astream(chain_input):
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

                full_response += content_to_yield
                yield content_to_yield

        # Append the AI's full response to the conversation history
        state.conversation_history.append(AIMessage(content=full_response))
        logger.info(f"Updated session state: {state}")