"""
Tools module - centralized tool imports for agents
"""

# Test tools
from .examples.test_tools import sync_tool, async_tool

# RAG tools
from .rag.X_rag_tool import (
    search_kyndryl_documents,
    search_kyndryl_documents_detailed
)

# Tool collections for easy importing
TEST_TOOLS = [sync_tool, async_tool]
RAG_TOOLS = [search_kyndryl_documents]
RAG_TOOLS_DETAILED = [search_kyndryl_documents_detailed]
ALL_TOOLS = TEST_TOOLS + RAG_TOOLS

__all__ = [
    # Individual tools
    "sync_tool",
    "async_tool",
    "search_kyndryl_documents",
    "search_kyndryl_documents_detailed",
    
    # Tool collections
    "TEST_TOOLS",
    "RAG_TOOLS",
    "RAG_TOOLS_DETAILED",
    "ALL_TOOLS"
]