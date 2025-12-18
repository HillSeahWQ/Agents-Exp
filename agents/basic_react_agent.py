from dotenv import load_dotenv
import logging
from typing import Annotated, Literal, TypedDict, List, Dict, Any, Optional
import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    BaseMessage, 
    HumanMessage, 
    AIMessage, 
    SystemMessage,
    ToolMessage,
    trim_messages
)
from langchain_core.tools import tool, StructuredTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages

from memory_strategies.base import MemoryStrategy
from memory_strategies.full_history import FullHistoryMemory
from utils.logging.logger import get_logger

logger = get_logger(__name__)

# ============================================================================
# AGENT STATE DEFINITION
# ============================================================================
def replace_scratchpad(left: List[BaseMessage], right: List[BaseMessage]) -> List[BaseMessage]:
    """Always replace scratchpad (don't accumulate)"""
    return right


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages] # Conversation History - Persist through multiple agent.invokes of the same threadid
    agent_scratchpad: Annotated[List[BaseMessage], replace_scratchpad] # One agent.invoke specific intermediate reasoning steps
    llm_calls_count: int # One agent.invoke specific count of number of llm calls
    
# ============================================================================
# AGENT DEFINITION
# ============================================================================
class BasicReactAgent:
    """
    An asynchronous agentic system that orchestrates LLM reasoning with tool execution.
    
    This agent uses a graph-based architecture to alternate between LLM reasoning (agent_node)
    and tool execution (tool_node), maintaining conversation history and state across turns.
    It supports both async and sync execution patterns.
    
    Attributes:
        tools (List[StructuredTool]): Available tools for the agent to use
        max_llm_call_count (int): Maximum reasoning iterations before stopping
        llm (ChatOpenAI): Language model for agent reasoning
        system_message (str): System prompt defining agent behavior
        memory_strategy (MemoryStrategy): Strategy for processing conversation history
        tool_dict (Dict[str, StructuredTool]): Mapping of tool names to tool objects
        graph (CompiledStateGraph): Compiled LangGraph workflow
    """
    def __init__(
        self,
        tools: List[StructuredTool],
        llm: Optional[ChatOpenAI] = None,
        system_message: Optional[SystemMessage] = None,
        max_llm_calls_count = 10,
        memory_strategy: Optional[MemoryStrategy] = None
    ):
        """
        Initialize the BasicAgent with tools, LLM, and configuration.
        
        Args:
            tools: List of StructuredTool objects the agent can use
            llm: Language model instance (defaults to GPT-4o with temp=0)
            system_message: Custom system prompt (defaults to helpful assistant)
            max_llm_calls_count: Maximum LLM calls per user query (default: 10)
            memory_strategy: Strategy for processing conversation history (default: FullHistoryMemory)
        """
        self.tools = tools
        self.max_llm_call_count = max_llm_calls_count
        self.llm = llm or ChatOpenAI(
            model="gpt-4o", 
            temperature=0.0
        )
        self.system_message = system_message or """You are a helpful AI assistant with access to tools.
        IMPORTANT INSTRUCTIONS:
        1. Use tools when you need information you don't have (current data, calculations, external info)
        2. Respond directly when you can answer from your knowledge
        3. If a tool returns an error, acknowledge it and try a different approach
        4. Be concise in your responses
        5. When you have gathered all necessary information from tools, provide your final answer directly

        Guidelines on when to use tools:
        - Need current information? → Use appropriate tool
        - Can answer from knowledge? → Respond directly
        - Simple questions? → Respond directly

        You can respond with or without tools - use your judgment!
        """
        self.memory_strategy = memory_strategy or FullHistoryMemory()
        
        self.llm_with_tools = self.llm.bind_tools(tools=tools, tool_choice="auto")
        self.tool_dict = {tool.name: tool for tool in self.tools}
        self.graph = self._build_graph()
        
    async def _agent_node(
        self,
        state: AgentState
    )-> AgentState:
        """
        LLM reasoning node: Processes conversation history and decides next action.
        
        This node invokes the LLM with the full context (system message + conversation + scratchpad)
        to either respond to the user or call tools for additional information.
        
        Args:
            state: Current agent state containing messages, scratchpad, and counters
            
        Returns:
            Updated state with LLM response appended to scratchpad and incremented call count
        """
        conversation_history = state["messages"]
        agent_scratchpad = state["agent_scratchpad"]
        llm_calls_count = state["llm_calls_count"]
        
        processed_conversation_history = self.memory_strategy.process_messages(conversation_history)
        response = await self.llm_with_tools.ainvoke([self.system_message] + processed_conversation_history + agent_scratchpad)
        
        return {
            "agent_scratchpad": agent_scratchpad + [response],
            "llm_calls_count": llm_calls_count + 1
        }
    
    async def _tool_node(
        self,
        state: AgentState
    ) -> AgentState:
        """
        Tool execution node: Executes all tool calls from the last LLM response in parallel.
        
        Extracts tool calls from the most recent AIMessage, executes them concurrently using
        asyncio.gather, and appends results as ToolMessages to the scratchpad.
        
        Args:
            state: Current agent state with tool calls in the last scratchpad message
            
        Returns:
            Updated state with tool results appended to scratchpad
        """

        agent_scratchpad = state["agent_scratchpad"]
        tool_calls = agent_scratchpad[-1].tool_calls
        
        tool_calls_coroutines = [
            self._execute_single_tool(
                tool_name=tool_call["name"],
                tool_kwargs_dict=tool_call["args"],
                tool_call_id=tool_call["id"]
            ) for tool_call in tool_calls
        ]
        
        tool_messages = await asyncio.gather(*tool_calls_coroutines)
        
        return {"agent_scratchpad": agent_scratchpad + tool_messages}
    
    async def _execute_single_tool(
        self,
        tool_name: str,
        tool_kwargs_dict: Dict,
        tool_call_id: str
    ) -> ToolMessage:
        """
        Execute a single tool call with error handling.
        
        Looks up the tool by name, executes it with the provided arguments (using async
        if available, otherwise in a thread), and wraps the result in a ToolMessage.
        
        Args:
            tool_name: Name of the tool to execute
            tool_kwargs_dict: Arguments to pass to the tool
            tool_call_id: Unique identifier for this tool call
            
        Returns:
            ToolMessage containing the tool result or error message
            
        Note:
            - Automatically handles both async and sync tools
            - Errors are caught and returned as descriptive error messages
            - All operations are logged for debugging
        """
        tool_obj = self.tool_dict.get(tool_name)
        
        if tool_obj:
            try:
                if tool_obj.coroutine:
                    result = await tool_obj.ainvoke(tool_kwargs_dict)
                else:
                    result = await asyncio.to_thread(tool_obj.invoke, tool_kwargs_dict)
                logger.debug(f"Tool result: {result} - Executed tool: '{tool_name}' with tool kwargs: '{tool_kwargs_dict}'")
            except Exception as e:
                result = f"Tool '{tool_name}' execution error with tool kwargs: '{tool_kwargs_dict}'"
                logger.error(result)
        else:
            result = f"Tool '{tool_name}' not found"
            logger.error(result)
        
        tool_message = ToolMessage(
            content=str(result),
            tool_call_id=tool_call_id
        )
        
        return tool_message
    
    def _should_continue_from_agent_node(
        self,
        state: AgentState
    ) -> str:
        """
        Routing logic: Decides whether to continue to tools, or end the workflow.
        
        Checks three conditions:
        1. Max LLM calls reached → end
        2. Empty scratchpad → end
        3. Last message has tool calls → route to tool_node
        4. Otherwise → end (agent provided final answer)
        
        Args:
            state: Current agent state
            
        Returns:
            "tool_node" to execute tools, or "end" to terminate the workflow
        """
        agent_scratchpad = state["agent_scratchpad"]
        llm_calls_count = state["llm_calls_count"]
        
        if llm_calls_count >= self.max_llm_call_count:
            return "end"
        
        if not agent_scratchpad:
            return "end"
        
        last_ai_message = agent_scratchpad[-1]
        
        if hasattr(last_ai_message, "tool_calls") and len(last_ai_message.tool_calls) > 0:
            return "tool_node"
        
        return "end"

    def _build_graph(self):
        """
        Construct the LangGraph workflow with agent and tool nodes.
        
        Creates a state graph with:
        - agent_node: LLM reasoning
        - tool_node: Parallel tool execution
        - Conditional routing between nodes based on tool calls
        - Memory checkpointing for conversation persistence
        
        Returns:
            Compiled StateGraph with memory persistence enabled
        """
        graph = StateGraph(AgentState)
        
        graph.add_node(node="agent_node", action=self._agent_node)
        graph.add_node(node="tool_node", action=self._tool_node)
        
        graph.add_edge(start_key=START, end_key="agent_node")
        graph.add_conditional_edges(
            source="agent_node",
            path=self._should_continue_from_agent_node,
            path_map={
                "tool_node": "tool_node",
                "end": END
            }
        )
        graph.add_edge(start_key="tool_node", end_key="agent_node")
        
        return graph.compile(checkpointer=MemorySaver())
        
    async def ainvoke(
        self,
        human_query: str,
        thread_id: str
    ) -> AgentState:
        """
        Main async entry point: Process a user query and return the agent's response.
        
        This method orchestrates the complete agentic workflow:
        1. Wraps user query as HumanMessage
        2. Initializes agent state (scratchpad, counters)
        3. Executes the graph workflow (reasoning + tool use)
        4. Extracts final response and stop reason
        5. Updates conversation history in persistent memory
        6. Returns complete result with metrics
        
        Args:
            human_query: The user's question or request
            thread_id: Unique identifier for this conversation thread (enables multi-turn)
            
        Returns:
            Dictionary containing:
                - response (str): Final answer to the user
                - conversation_history (List[BaseMessage]): Full chat history
                - scratchpad (List[BaseMessage]): Internal reasoning trace
                - stop_reason (str): Why the agent stopped (completed, max_iterations, etc.)
                - thread_id (str): Thread identifier
                - metrics (Dict): Tool calls, LLM calls, and state sizes
        """
        # Format frontend inputs - human query + threadid corresponding to the particular ongoing chat
        human_message = HumanMessage(content=human_query)
        config={"configurable": {"thread_id": thread_id}}
        
        # Create the initial state - append human message to conversation history + reset agent scratchpad for inner intermediate workings and llm call counts for this particular agent.invoke
        initial_state = {
            "messages": [human_message],
            "agent_scratchpad": [],
            "llm_calls_count": 0
        }
        
        # Run the agentic flow graph to get the final state
        final_state = await self.graph.ainvoke(
            input=initial_state,
            config=config
        )
        
        # Extract final ai response to human query / stop reason (eg: successfully replied user, llm calls count exceed max threshold, ...)
        conversation_history = final_state["messages"]
        scratchpad = final_state["agent_scratchpad"]
        llm_calls_count = final_state["llm_calls_count"]
        
        response_content, stop_reason = self._extract_response(
            scratchpad=scratchpad,
            llm_calls_count=llm_calls_count
        )
        
        # Update the conversation history - ensures conversation history only has a list of alternating (human query i, final ai message to human query i)
        final_answer_message = AIMessage(content=response_content)
        updated_conversation = conversation_history + [final_answer_message]

        final_checkpoint_state = {
            "messages": updated_conversation,
            "agent_scratchpad": [],
            "llm_calls_count": 0
        }
        
        # NOTE: update_state is sync in LangGraph (it's just state mutation)
        self.graph.update_state(config, final_checkpoint_state)
        
        # Some metrics to log
        metrics = {
            "tool_calls": sum(1 for m in scratchpad if isinstance(m, ToolMessage)),
            "llm_calls": llm_calls_count,
            "scratchpad_size": len(scratchpad),
            "conversation_size": len(updated_conversation)
        }
        
        return {
            "response": response_content,
            "conversation_history": updated_conversation,
            "scratchpad": scratchpad,
            "stop_reason": stop_reason,
            "thread_id": thread_id,
            "metrics": metrics
        }
    
    
    def invoke(self, human_query: str, thread_id: str) -> Dict[str, Any]:
        """
        Synchronous wrapper for ainvoke: Allows usage without async/await.
        
        This method provides backward compatibility for simple scripts that don't
        use async/await. It internally creates an event loop and runs ainvoke().
        
        Args:
            human_query: The user's question or request
            thread_id: Unique identifier for this conversation thread
            
        Returns:
            Same dictionary as ainvoke() with response, history, and metrics
            
        Example:
            >>> agent = BasicAgent(tools=my_tools)
            >>> result = agent.invoke("What's 2+2?", "thread-123")  # No await needed
            >>> print(result["response"])
        """
        return asyncio.run(self.ainvoke(human_query, thread_id))
    
    
    def _extract_response(
        self, 
        scratchpad: List[BaseMessage],
        llm_calls_count: int
    ) -> tuple[str, str]:
        """
        Extract the final response and stop reason from the agent's scratchpad.
        
        Analyzes the scratchpad to determine:
        1. What response to return to the user
        2. Why the agent stopped (successful completion vs. error/limit)
        
        Handles multiple termination scenarios:
        - Normal completion: Agent provided final answer without pending tool calls
        - Max iterations: Hit LLM call limit
        - Incomplete tool calls: Agent requested tools but workflow ended
        - No synthesis: Agent got tool results but didn't formulate answer
        - Unknown: Unexpected termination state
        
        Args:
            scratchpad: List of messages from internal reasoning trace
            llm_calls_count: Total number of LLM calls made
            
        Returns:
            Tuple of (response_content, stop_reason):
                - response_content: Message to show the user
                - stop_reason: One of: "completed", "max_iterations", 
                  "incomplete_tool_calls", "no_synthesis", "unknown", "empty_scratchpad"
        """
        if not scratchpad:
            return "No response generated", "empty_scratchpad"
        
        last_message = scratchpad[-1]
        
        if isinstance(last_message, AIMessage) and not (hasattr(last_message, 'tool_calls') and last_message.tool_calls):
            return last_message.content, "completed"
        
        elif llm_calls_count >= self.max_llm_calls_count:
            if isinstance(last_message, AIMessage) and last_message.content:
                return last_message.content, "max_iterations"
            else:
                return (
                    f"I apologize, but I've reached the maximum number of reasoning steps ({self.max_llm_calls_count}) "
                    f"while trying to answer your question. Please try rephrasing or breaking it into smaller parts."
                ), "max_iterations"
        
        elif isinstance(last_message, AIMessage) and hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return (
                "I apologize, but I was unable to complete processing your request. "
                "Please try rephrasing your question."
            ), "incomplete_tool_calls"
        
        elif isinstance(last_message, ToolMessage):
            return (
                f"I gathered some information but wasn't able to formulate a complete answer. "
                f"Please try asking your question differently."
            ), "no_synthesis"
        
        else:
            return (
                "I apologize, but I encountered an unexpected issue. "
                "Please try asking your question again."
            ), "unknown"