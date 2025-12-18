from typing import List, Optional
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from .base import MemoryStrategy

class SlidingWindowWithSummaryMemory(MemoryStrategy):
    """Keep recent k messages + summary of older messages"""
    
    def __init__(self, k: int = 10, llm: Optional[ChatOpenAI] = None):
        self.k = k
        self.llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self._summary_cache = {}
    
    def _create_summary(self, messages: List[BaseMessage]) -> str:
        """Create a summary of older messages"""
        cache_key = hash(str([m.content for m in messages]))
        
        if cache_key in self._summary_cache:
            return self._summary_cache[cache_key]
        
        summary_prompt = ChatPromptTemplate.from_messages([
            ("system", "Summarize the following conversation history concisely, "
                      "capturing key information, decisions, and context. "
                      "Keep it under 200 words."),
            MessagesPlaceholder(variable_name="messages")
        ])
        
        response = self.llm.invoke(
            summary_prompt.format_messages(messages=messages)
        )
        
        summary = response.content
        self._summary_cache[cache_key] = summary
        return summary
    
    def process_messages(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
        other_msgs = [m for m in messages if not isinstance(m, SystemMessage)]
        
        if len(other_msgs) <= self.k:
            return messages
        
        old_msgs = other_msgs[:-self.k]
        recent_msgs = other_msgs[-self.k:]
        
        summary = self._create_summary(old_msgs)
        summary_msg = SystemMessage(
            content=f"Previous conversation summary: {summary}"
        )
        
        return system_msgs + [summary_msg] + recent_msgs