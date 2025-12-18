from typing import List
from langchain_core.messages import BaseMessage, SystemMessage
from .base import MemoryStrategy

class RecentKMemory(MemoryStrategy):
    """Keep only the k most recent messages"""
    
    def __init__(self, k: int = 10):
        self.k = k
    
    def process_messages(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
        other_msgs = [m for m in messages if not isinstance(m, SystemMessage)]
        recent = other_msgs[-self.k:] if len(other_msgs) > self.k else other_msgs
        return system_msgs + recent