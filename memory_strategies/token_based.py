from typing import List, Literal
from langchain_core.messages import BaseMessage, trim_messages
from .base import MemoryStrategy

class TokenBasedMemory(MemoryStrategy):
    """Keep messages within a token budget using LangChain's trim_messages"""
    
    def __init__(self, max_tokens: int = 4000, strategy: Literal["first", "last"] = "last"):
        self.max_tokens = max_tokens
        self.strategy = strategy
    
    def process_messages(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        return trim_messages(
            messages,
            max_tokens=self.max_tokens,
            strategy=self.strategy,
            token_counter=len,
            allow_partial=False,
            start_on="human"
        )