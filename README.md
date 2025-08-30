# memory-agent  

[![View on GitHub](https://img.shields.io/badge/View%20on-GitHub-181717?style=for-the-badge&logo=github)](https://github.com/gzileni/memory-agent)  
[![GitHub stars](https://img.shields.io/github/stars/gzileni/memory-agent?style=social)](https://github.com/gzileni/memory-agent/stargazers)  
[![GitHub forks](https://img.shields.io/github/forks/gzileni/memory-agent?style=social)](https://github.com/gzileni/memory-agent/network)  

The library allows you to manage both [persistence](https://langchain-ai.github.io/langgraph/how-tos/persistence/) and [**memory**](https://langchain-ai.github.io/langgraph/concepts/memory/#what-is-memory) for a LangGraph agent.  

**memory-agent** uses [Redis](https://redis.io/) as the short-term memory backend and [QDrant](https://qdrant.tech/) for long-term persistence and semantic search.  

---

## 🔑 Key Features

- **Dual-layer memory system**
  - **Short-term memory with Redis** → ultra-fast, volatile, TTL-based storage for ongoing sessions.
  - **Long-term persistence with Qdrant** → semantic search, embeddings, and retrieval across sessions.
- **LangGraph integration** → build fully stateful LLM-powered agents with checkpoints and memory tools.  
- **Multi-LLM support**
  - [OpenAI](https://platform.openai.com/) (via `AgentOpenAI`)  
  - [Ollama](https://ollama.com/) (via `AgentOllama`) for local inference.  
- **Embeddings flexibility**
  - OpenAI embeddings (default).
  - Hugging Face local embeddings for **air-gapped environments**.
  - Ollama embeddings (`nomic-embed-text`, etc.).  
- **Automatic memory management**
  - Background summarization & reflection to condense context.
  - Checkpoint pruning (`filter_minutes`) for resource control.
- **Observability**
  - Structured logging, **Grafana/Loki compatible**.
- **Easy installation & deployment**
  - Simple `pip install`.
  - Ready-to-use with **Docker** ([docs here](./docker/README.md)).  

---

## Memory vs Persistence  

When developing agents with LangGraph (or LLM-based systems in general), it’s crucial to distinguish between **memory** and **persistence**.  

### Persistence (Qdrant)
- **Permanent storage** across sessions.  
- **Examples:** embeddings, vector databases, long-term conversation history.  
- **Why Qdrant?**
  - Disk persistence.
  - Vector similarity search at scale.
  - Advanced queries with metadata & filters.  

### Memory (Redis)
- **Temporary state** during a conversation or workflow.  
- **Examples:** current conversation state, volatile variables.  
- **Why Redis?**
  - High-performance in-RAM operations.
  - TTL & automatic cleanup.
  - Multi-worker support.  

| Function        | Database | Main Reason                                   |
|-----------------|----------|-----------------------------------------------|
| **Memory**      | Redis    | Performance, TTL, fast session context        |
| **Persistence** | Qdrant   | Vector search, scalable long-term storage     |  

---

## Installation  

```bash
pip install memory-agent
```

For development with Ollama or local embeddings:  

```bash
# Install Ollama
https://ollama.ai  

# Install Hugging Face tools
pip install --upgrade huggingface_hub
```

---

## Usage Examples  

### OpenAI  

```python
import asyncio
from agent_openai import AgentOpenAI
import os

async def main():
    agent = AgentOpenAI(
        key_search="agent_openai_demo",
        model_name="gpt-4.1-mini",
        temperature=0.2,
        model_embedding_name="text-embedding-3-small",
        llm_api_key=os.getenv("OPENAI_API_KEY"),
        thread_id="thread-a"
    )

    result = await agent.ainvoke("Summarize Python 3.12 new features in 3 points.")
    print(result)

    async for chunk in agent.stream("Write a 2-sentence abstract about prompt engineering."):
        print(chunk)

asyncio.run(main())
```

#### Example — Memory per Conversation Thread

```python
import asyncio
from agent_openai import AgentOpenAI
import os

async def main():
    agent = AgentOpenAI(
        key_search="agent_openai_demo",
        model_name="gpt-4.1-mini",
        temperature=0.2,
        model_embedding_name="text-embedding-3-small",
        llm_api_key=os.getenv("OPENAI_API_KEY")
    )

    thread_id: str = "thread-a"

    response = await agent.ainvoke(
        "Know which display mode I prefer?",
        thread_id=thread_id
    )
    print(response["messages"][-1].content)

    # Save a preference in memory
    await agent.ainvoke(
        "dark. Remember that.",
        thread_id=thread_id
    )

    # Continuing in the same thread (thread-a), the agent recalls the preference
    response = await agent.ainvoke(
        "Do you remember my display mode preference?",
        thread_id=thread_id
    )
    print(response["messages"][-1].content)

    thread_id: str = "thread-b"
    response = await agent.ainvoke(
        "Hey there. Do you remember me? What are my preferences?",
        thread_id=thread_id
    )
    print(response["messages"][-1].content)

asyncio.run(main())
```

### Ollama  

```python
import asyncio
from agent_ollama import AgentOllama

async def main():
    agent = AgentOllama(
        key_search="agent_ollama_demo",
        model_name="llama3.1",
        base_url="http://localhost:11434",
        temperature=0.2,
        model_embedding_name="nomic-embed-text",
        model_embedding_url="http://localhost:11434",
        thread_id="thread-a"
    )

    result = await agent.ainvoke("Explain the ReAct pattern in 5 lines.")
    print(result)

    async for chunk in agent.stream("Summarize the last user message in 3 bullet points."):
        print(chunk)

asyncio.run(main())
```

---

## Memory Agents AI  

**MemoryAgent** is also an agent based on LangGraph with long-term memory able to save, search, and synthesize conversation memories.  

- **Types of message storage**
  - `hotpath`: explicit saves performed by the agent via tools (notes or checkpoints created consciously).  
  - `background`: memories automatically extracted from conversations in a “subconscious” way without direct intervention.  

- **Memory management**
  - Obsolete checkpoints can be removed automatically after a configurable interval.  
  - Support for common backends: Redis (storage/checkpointer) and Qdrant (vector indexing and semantic search).  

- **Key features**
  - Orchestration of tools, prompts, and agent logic in synchronous (`ainvoke`) or streaming (`stream`) executions.  
  - Configurable via parameters such as `thread_id`, `key_search`, `host_persistence_config`, `model_name`, etc.  
  - Designed to easily integrate tools and memory/synthesis pipelines from the langmem/langchain ecosystem.  

### Quick best practices
- Keep a stable `thread_id` to grow memory per session/user.  
- Use different `key_search` values for distinct roles or domains.  
- Enable periodic checkpoint cleaning (`refresh_checkpointer=True`) to contain resource usage.  
- Choose Redis for checkpoint persistence and Qdrant for semantic search when vector search is required.  

### Configuration  

Relevant parameters supported by the constructors (via `**kwargs`):  

- `thread_id: str` → Unique conversation ID (default: generated UUID).  
- `key_search: str` → Namespace/key for indexing the agent’s memories.  
- `model_name: str` → Model name (e.g. `gpt-4.1-mini` for OpenAI, `llama3.1` for Ollama).  
- `model_provider: Literal["openai","ollama"]` → Automatically set in wrappers.  
- `base_url: Optional[str]` → Provider endpoint (required for self-hosted instances; default for Ollama: `http://localhost:11434`).  
- `temperature: float` → Model creativity (if supported).  
- `tools: list` → Additional tools to expose to the agent besides memory tools.  
- `max_recursion_limit: int` → Maximum depth of agent steps (approximate default: `25`).  
- `filter_minutes: int` → Time window for checkpoint cleaning (approximate default: `15`).  
- `refresh_checkpointer: bool` → If true, cleans old checkpoints for the current `thread_id`.  
- `host_persistence_config: dict` → Checkpointer backend configuration, e.g.:  
```python
host_persistence_config={
        "host": "localhost",
        "port": 6379,
        "db": 0,
}
```

For OpenAI, ensure you have the ENV:  
```bash
export OPENAI_API_KEY="sk-..."
```

---

## Docker  

See [Docker README](./docker/README.md) for instructions on running Redis, Qdrant, and memory-agent in containers.  

---

## Troubleshooting  

- **Network / 401 error**: verify `OPENAI_API_KEY` (for OpenAI) or the reachability of `base_url` (for Ollama).  
- **Model not found**: in Ollama run `ollama pull llama3.1` (or the model you want).  
- **Token limit exceeded**: reduce `max_recursion_limit`, use lower temperature, or enable/refine short-term synthesis.  
- **Store/Checkpointer**: if using Redis or another backend, verify `host_persistence_config`.  

---
