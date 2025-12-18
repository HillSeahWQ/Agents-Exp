# `basic_react_agent.py` 

- Production-Ready Agentic System with LangGraph
- implements a robust asynchronous agent that combines LLM reasoning with tool execution,
providing fine-grained control over conversation history, memory persistence, and execution flow.

---

##  KEY FEATURES

### 1. **Hybrid Tool Execution (Async + Sync)**

* Natively supports both async and sync tools
* **Async tools:** executed directly via `await tool.ainvoke()`
* **Sync tools:** wrapped in `asyncio.to_thread()` for non-blocking execution
* Parallel tool execution using `asyncio.gather()` for high throughput

---

### 2. **Automatic Tool Choice (Not Forced)**

* LLM autonomously decides when to use tools vs. respond directly
* Tool choice mode: `"auto"`
* Prevents wasteful tool calls for simple conversational queries

---

### 3. **Flexible Memory Management**

Pluggable memory abstractions control how chat history is preserved and fed to the LLM.

Built-in strategies include:

* `FullHistoryMemory`
* `LastKMessages`
* `SlidingWindowMemory`
* others…

State separation:

* `state["messages"]`: Persistent user-facing conversation history
* `state["agent_scratchpad"]`: Ephemeral tool reasoning trace (reset every query)

---

### 4. **Thread-Based Conversation Persistence**

* MemorySaver stores state per `thread_id`
* Same `thread_id` → automatic state restoration
* Ephemeral in-process memory (non-durable)
* Ideal for multi-turn agents in an application server

---

### 5. **Max LLM Call Limiting (Anti-Loop Protection)**

* Configurable per-query cap on LLM calls
* Default: **10**
* Prevents infinite loops, runaway costs
* Counter stored in `AgentState` and resets on every invocation

---

### 6. **Intelligent Final Answer Extraction**

Robust handling of problematic generation scenarios:

* Max-iteration cutoff mid-deliberation
* Orphaned/unfinished tool calls
* Missing summarization after tool execution

Returns:

* Clean user-facing response
* Debug metadata (stop reason, tool usage count)

---

### 7. **Clean Conversation History (Scratchpad Pattern)**

* `messages` = only user messages + final AI answers
* `agent_scratchpad` = reasoning chain + tool results
* Benefits:

  * Reduced token usage
  * No tool chatter polluting conversation
  * Better LLM behavior in long conversations

---

## ARCHITECTURE

```
User Query → HumanMessage
     ↓
┌────────────────────────────────────┐
│   LangGraph Workflow (per query)   │
│                                    │
│  ┌──────────────┐                 │
│  │ agent_node   │  (LLM thinking) │
│  └──────┬───────┘                 │
│         │                          │
│    Tool calls?                     │
│    /        \                      │
│  Yes        No                     │
│  │          └─→ END                │
│  ↓                                 │
│  ┌──────────────┐                 │
│  │  tool_node   │ (run tools)     │
│  └──────┬───────┘                 │
│         │                          │
│         └─→ Loop back to agent    │
└────────────────────────────────────┘
     ↓
 Final Answer Extraction
     ↓
 Update Persistent Memory
     ↓
Return Response + Metrics
```
