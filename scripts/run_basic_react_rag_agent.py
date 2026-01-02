"""
Example script: Basic React Agent with RAG tools
Agent - agents/basic_react_agent
Tools - tools/rag

USAGE:
    # Interactive mode (default)
    uv run python scripts/run_basic_react_rag_agent.py
    uv run python scripts/run_basic_react_rag_agent.py --mode interactive
    
    # Run example queries
    uv run python scripts/run_basic_react_rag_agent.py --mode examples
    
    # Run examples without showing detailed scratchpad
    uv run python scripts/run_basic_react_rag_agent.py --mode examples --no-scratchpad
    
    # Single query mode
    uv run python scripts/run_basic_react_rag_agent.py --query "What does X cover for surgeries?"
"""
import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from agents.basic_react_agent import BasicReactAgent
from memory_strategies.full_history import FullHistoryMemory
from tools.rag.company_x_rag_tool import search_company_x_documents, search_company_x_documents_detailed
from utils.logging.logger import get_logger
from utils.running_agent.interactive import run_interactive_chat, run_example_queries
from utils.running_agent.pretty_print import pretty_print_agent_response

logger = get_logger(__name__)
load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

SYSTEM_MESSAGE = SystemMessage(content="""You are a helpful AI assistant with access to Company X's internal documentation.

IMPORTANT INSTRUCTIONS:
1. When users ask about Company X policies, benefits, procedures, or documentation:
   - Use the Company X documents tool to find relevant information
   - Always cite your sources (mention the document name and page number)
   - If information is not found in the documents, clearly state that

2. For general questions that don't require Company X-specific information:
   - Answer directly from your knowledge
   - Be clear about what is general knowledge vs. Company X-specific

3. Best practices:
   - Search with specific, focused queries
   - If the first search doesn't yield good results, try rephrasing
   - Summarize findings clearly and concisely
   - Always provide source citations for Company X-specific information

Example interactions:
- User: "What does Company X cover for surgeries?"
  → Use tool: search_company_x_documents("Company X surgery coverage benefits")
  
- User: "What is Python?"
  → Answer directly without tools (general knowledge)
"""
)

LLM_MODEL_NAME = "gpt-4o" # TODO: EDIT desired OpenAI LLM model name
TEMPERATURE = 0.0 # TODO: EDIT desired LLM's temperature
MAX_LLM_CALLS_COUNT = 10 # TODO: EDIT desired maximum LLM calls threshold per query
MEMORY_STRATEGY = FullHistoryMemory() # TODO: EDIT desired memory strategy to trim conversation history
TOOLS = [search_company_x_documents] # TODO: EDIT desired list of tools accessible by agent

# Example queries for testing
EXAMPLE_QUERIES = [
    {
        "thread_id": "example-1",
        "query": "How much does Company X cover for surgeries?"
    },
    {
        "thread_id": "example-2", 
        "query": "What are the hospitals covered by Company X's health insurance?"
    },
    {
        "thread_id": "example-3",
        "query": "What is the process for filing a medical claim?"
    }
]

# ============================================================================
# CREATE AGENT
# ============================================================================

def create_agent():
    """Create and return a configured RAG agent."""
    return BasicReactAgent(
        tools=TOOLS,
        memory_strategy=MEMORY_STRATEGY,
        llm=ChatOpenAI(model=LLM_MODEL_NAME, temperature=TEMPERATURE),
        max_llm_calls_count=MAX_LLM_CALLS_COUNT,
        system_message=SYSTEM_MESSAGE
    )


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="X RAG Agent - Search internal documentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=""" Examples:
            # Interactive chat mode
            uv run python scripts/run_basic_react_rag_agent.py
            
            # Run predefined examples
            uv run python scripts/run_basic_react_rag_agent.py --mode examples
            
            # Single query
            uv run python scripts/run_basic_react_rag_agent.py --query "What does X cover?"
            
            # Examples without detailed scratchpad
            uv run python scripts/run_basic_react_rag_agent.py --mode examples --no-scratchpad
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["interactive", "examples"],
        default="interactive",
        help="Run mode: interactive chat or example queries (default: interactive)"
    )
    
    parser.add_argument(
        "--query",
        type=str,
        help="Single query to run (overrides --mode)"
    )
    
    parser.add_argument(
        "--thread-id",
        type=str,
        default="default-session",
        help="Thread ID for conversation history (default: default-session)"
    )
    
    parser.add_argument(
        "--no-scratchpad",
        action="store_true",
        help="Don't show detailed tool execution traces"
    )
    
    args = parser.parse_args()
    
    try:
        # Create agent
        logger.info("Initializing RAG agent...")
        agent = create_agent()
        logger.info("Agent initialized successfully")
        
        # Single query mode
        if args.query:
            logger.info(f"Running single query: {args.query}")
            result = agent.invoke(
                human_query=args.query,
                thread_id=args.thread_id
            )
            pretty_print_agent_response(result, show_scratchpad=not args.no_scratchpad)
        
        # Interactive mode
        elif args.mode == "interactive":
            run_interactive_chat(
                agent=agent,
                thread_id=args.thread_id,
                title="X RAG AGENT - Interactive Mode"
            )
        
        # Examples mode
        else:
            run_example_queries(
                agent=agent,
                examples=EXAMPLE_QUERIES,
                show_scratchpad=not args.no_scratchpad
            )
        
        return 0
    
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"\n  Fatal Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())