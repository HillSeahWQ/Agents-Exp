from typing import List
from langchain_core.messages import BaseMessage

class MemoryStrategy:
    """Base class for memory management strategies"""
    
    def process_messages(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """Process messages according to strategy"""
        raise NotImplementedError