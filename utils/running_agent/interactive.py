"""
Interactive mode utilities for running agents in chat mode.
"""
from typing import Any
from utils.running_agent.pretty_print import print_header, print_compact_response
from utils.logging.logger import get_logger

logger = get_logger(__name__)


def run_interactive_chat(agent: Any, thread_id: str = "interactive-session", title: str = "INTERACTIVE AGENT"):
    """
    Run agent in interactive chat mode.
    
    Args:
        agent: Agent instance with invoke() method
        thread_id: Conversation thread identifier
        title: Title to display in the header
    """
    print_header(title)
    print("\nAsk me anything! Type 'quit', 'exit', or 'q' to stop.\n")
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            # Check for exit commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            # Skip empty inputs
            if not user_input:
                continue
            
            # Process query
            print("\n🤖 Agent: (thinking...)")
            result = agent.invoke(
                human_query=user_input,
                thread_id=thread_id
            )
            
            # Display response
            print_compact_response(result)
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            logger.error(f"Error during interaction: {e}", exc_info=True)
            print(f"\n❌ Error: {e}")


def run_example_queries(agent: Any, examples: list, show_scratchpad: bool = True):
    """
    Run a list of predefined example queries.
    
    Args:
        agent: Agent instance with invoke() method
        examples: List of dicts with 'thread_id' and 'query' keys
        show_scratchpad: Whether to show detailed execution traces
    """
    from utils.running_agent.pretty_print import pretty_print_agent_response
    
    print_header("RUNNING EXAMPLE QUERIES")
    
    for i, example in enumerate(examples, 1):
        print(f"\n\n{'='*80}")
        print(f"EXAMPLE {i}: {example['query']}")
        print('='*80)
        
        try:
            result = agent.invoke(
                human_query=example['query'],
                thread_id=example['thread_id']
            )
            
            pretty_print_agent_response(result, show_scratchpad=show_scratchpad)
        
        except Exception as e:
            logger.error(f"Error in example {i}: {e}", exc_info=True)
            print(f"\n❌ Error: {e}")