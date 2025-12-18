"""
Pretty printing utilities for agent responses and outputs.
"""
from typing import Dict, Any


def pretty_print_agent_response(response: Dict[str, Any], show_scratchpad: bool = True):
    """
    Pretty print agent response with formatting.
    
    Args:
        response: Agent response dictionary containing:
            - response: Final answer text
            - metrics: Dict with tool_calls, llm_calls, etc.
            - scratchpad: List of intermediate messages
            - stop_reason: Why the agent stopped
        show_scratchpad: Whether to show detailed tool execution steps
    """
    print("\n" + "="*80)
    print("AGENT RESPONSE")
    print("="*80)
    
    print("\n📝 Response:")
    print("-" * 80)
    print(response.get("response", ""))
    
    print("\n📊 Metrics:")
    print("-" * 80)
    metrics = response.get("metrics", {})
    print(f"  Tool Calls: {metrics.get('tool_calls', 0)}")
    print(f"  LLM Calls: {metrics.get('llm_calls', 0)}")
    print(f"  Stop Reason: {response.get('stop_reason', 'unknown')}")
    
    # Show scratchpad details if requested and there were tool calls
    if show_scratchpad and metrics.get('tool_calls', 0) > 0:
        print("\n🔧 Tool Execution Details:")
        print("-" * 80)
        scratchpad = response.get("scratchpad", [])
        for i, msg in enumerate(scratchpad, 1):
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                print(f"\n  Step {i}: Tool Calls")
                for tc in msg.tool_calls:
                    print(f"    - {tc['name']}({tc['args']})")
            elif hasattr(msg, "tool_call_id"):
                content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
                print(f"\n  Step {i}: Tool Result")
                print(f"    {content}")
    
    print("\n" + "="*80)


def print_header(title: str, width: int = 80):
    """
    Print a formatted header.
    
    Args:
        title: Header text
        width: Total width of the header
    """
    print("\n" + "="*width)
    print(title)
    print("="*width)


def print_section(title: str, width: int = 80):
    """
    Print a formatted section divider.
    
    Args:
        title: Section title
        width: Total width of the divider
    """
    print(f"\n{title}")
    print("-" * width)


def print_compact_response(response: Dict[str, Any]):
    """
    Print a compact version of the agent response (for interactive mode).
    
    Args:
        response: Agent response dictionary
    """
    print(f"\n🤖 Agent:\n{response['response']}")
    
    # Show minimal metrics
    metrics = response.get('metrics', {})
    if metrics.get('tool_calls', 0) > 0:
        print(f"\n   [Used {metrics['tool_calls']} tool call(s), {metrics['llm_calls']} LLM call(s)]")