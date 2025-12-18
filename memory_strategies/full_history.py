from typing import List
from langchain_core.messages import BaseMessage
from .base import MemoryStrategy

class FullHistoryMemory(MemoryStrategy):
    """Keep all messages - use for short conversations"""
    
    def process_messages(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        return messages