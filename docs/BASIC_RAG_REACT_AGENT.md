# BASIC RAG React Agent — Usage Guide

## 0. Pre-Requisites
- Install uv
- Install Python >= 3.12

---

# 1. Environment Setup

## Create `.env` (based on `.env.example`)
Ensure the following variable exists:

OPENAI_API_KEY=your-key-here

---

# 2. Ingest Documents into FAISS

## 2a. Place your input documents
Put all files into:

Agent-H/vector_db/data/[INPUT_FOLDER_NAME]

## 2b. Run the ingestion pipeline (in order)

### 1. Chunking
Edit INPUT_FOLDER_NAME inside:

Agent-H/vector_db/scripts/run_chunking.py

Run:

uv run vector_db/scripts/run_chunking.py

### 2. FAISS ingestion
Edit the following inside `run_ingestion_faiss.py`:

- INPUT_FOLDER_NAME = "your_folder"
- FAISS_INDEX_NAME = "your_index_name"

Then run:

uv run vector_db/scripts/run_ingestion_faiss.py

---

# 3. Create Your Own RAG Tool

Path:

Agent-H/tools/rag/YOUR_OWN_RAG_TOOL.py

Steps:

1. Copy:
   Agent-H/tools/rag/example_rag_tool.py

2. Update the index name:
   FAISS_INDEX_NAME = "YOUR_FAISS_INDEX_NAME"
   (must match the value used in Step 2b)

3. Register your tool:
   - Add import to Agent-H/tools/rag/__init__.py
   - Add import to Agent-H/tools/__init__.py
   - Add your tool to the RAG_TOOLS config in tools/__init__.py

---

# 4. Interactive Mode (Default)

Start the agent:

uv run python scripts/run_basic_react_rag_agent.py

Or explicitly:

uv run python scripts/run_basic_react_rag_agent.py --mode interactive

Example interaction:

👤 You: What does X cover for surgeries?
🤖 Agent: Based on the documentation...

---

# 5. Run Example Queries

uv run python scripts/run_basic_react_rag_agent.py --mode examples

With minimal output:

uv run python scripts/run_basic_react_rag_agent.py --mode examples --no-scratchpad

---

# 6. Single Query Mode

uv run python scripts/run_basic_react_rag_agent.py --query "What does X cover for surgeries?"

Custom thread ID:

uv run python scripts/run_basic_react_rag_agent.py --query "What are the covered hospitals?" --thread-id "my-session-123"

No scratchpad:

uv run python scripts/run_basic_react_rag_agent.py --query "What is the claims process?" --no-scratchpad

---

# Command Reference

## Commands

| Command | Description |
|--------|-------------|
| uv run python scripts/run_basic_react_rag_agent.py | Interactive mode |
| uv run python scripts/run_basic_react_rag_agent.py --mode examples | Run example queries |
| uv run python scripts/run_basic_react_rag_agent.py --query "..." | Single query mode |

## Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| --mode | choice | interactive | Run mode: interactive or examples |
| --query | string | None | Runs a single query |
| --thread-id | string | default-session | Conversation thread ID |
| --no-scratchpad | flag | False | Hide tool execution traces |

---

# Examples

## Example 1: Quick Test

uv run python scripts/run_basic_react_rag_agent.py --query "What are X's surgical benefits?"

## Example 2: Multi-turn Conversation

uv run python scripts/run_basic_react_rag_agent.py --mode interactive --thread-id "user-john-123"

Follow-up questions will use conversation context.

## Example 3: Testing with Examples

uv run python scripts/run_basic_react_rag_agent.py --mode examples

No traces:

uv run python scripts/run_basic_react_rag_agent.py --mode examples --no-scratchpad

## Example 4: Debugging with Full Traces

uv run python scripts/run_basic_react_rag_agent.py --query "What hospitals are covered?" --thread-id "debug-session"

---

# Output Formats

## Compact Output (interactive mode)

🤖 Agent:  
Based on the documentation...  
[Used X tool calls, Y LLM calls]

## Detailed Output (examples or single query)

Shows:
- Final answer
- Tool calls
- LLM calls
- Step-by-step tool execution

---

# Customizing Queries

## Predefined Example Queries

In scripts/run_basic_react_rag_agent.py:

EXAMPLE_QUERIES = [
    {"thread_id": "example-1", "query": "Your custom query here"},
]

## Custom Scripts

Example:

from agents.basic_react_agent import BasicReactAgent
from tools import RAG_TOOLS
from utils import run_interactive_chat

agent = BasicReactAgent(tools=RAG_TOOLS)
run_interactive_chat(agent, title="MY CUSTOM AGENT")

---

# Troubleshooting

## FAISS Index Not Found

Ensure these files exist:

vector_db/faiss_indices/YOUR_INDEX_NAME.faiss.index  
vector_db/faiss_indices/YOUR_INDEX_NAME_metadata.pkl  
vector_db/faiss_indices/YOUR_INDEX_NAME_contents.pkl

## No Results Found

- Rephrase your query  
- Use more specific keywords  
- Verify the document exists in your input folder  

## API Key Error

Ensure `.env` contains:

OPENAI_API_KEY=sk-xxxx

---

# Advanced Usage

## Using Different Agents

from agents.basic_react_agent import BasicReactAgent  
from tools import RAG_TOOLS, TEST_TOOLS  

rag_agent = BasicReactAgent(tools=RAG_TOOLS)  
full_agent = BasicReactAgent(tools=RAG_TOOLS + TEST_TOOLS)

## Custom Memory Strategies

from memory_strategies.sliding_window import SlidingWindowMemory  

agent = BasicReactAgent(  
    tools=RAG_TOOLS,  
    memory_strategy=SlidingWindowMemory(window_size=10)  
)

## Programmatic Usage

from scripts.run_rag_agent import create_agent  

agent = create_agent()  
result = agent.invoke(  
    human_query="What are the benefits?",  
    thread_id="my-thread"  
)

print(result["response"])  
print(result["metrics"]["tool_calls"])

---

# Tips

1. Use specific queries for better results  
2. In interactive mode, leverage conversation history  
3. For debugging, set LOG_LEVEL=DEBUG in `.env`  
4. Adjust top_k (1–20) for more context  
5. Use thread IDs to maintain separate conversation histories