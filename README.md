[![View on GitHub](https://img.shields.io/badge/View%20on-GitHub-181717?style=for-the-badge&logo=github)](https://github.com/gzileni/memory-agent)
[![GitHub stars](https://img.shields.io/github/stars/gzileni/memory-agent?style=social)](https://github.com/gzileni/memory-agent/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/gzileni/memory-agent?style=social)](https://github.com/gzileni/memory-agent/network)

The library allows you to manage both [persistence](https://langchain-ai.github.io/langgraph/how-tos/persistence/) and [**memory**](https://langchain-ai.github.io/langgraph/concepts/memory/#what-is-memory) for a LangGraph agent.

**memory-agent** uses [Redis](https://redis.io/) as the short-term memory database and [QDrant](https://qdrant.tech/) for long-term persistence.

## Table of Contents

* [Key Features](#key-features)
* [Memory vs Persistence](#memory-vs-persistence)
  * [Persistence](#persistence)
  * [Memory](#memory)
* [Recommended Architecture](#recommended-architecture)
* [Installation](#installation)
* [Usage Example (Redis + LangGraph)](#usage-example-redis--langgraph)
* [Ollama — MemoryOllama](#ollama--memoryollama)
* [Vector Database (QDrant)](#vector-database-qdrant)
* [Custom Text Embedding Model](#custom-text-embedding-model)
* [Docker Compose (Redis + Qdrant)](#docker-compose-redis--qdrant)
* [Grafana Logging](#grafana-logging)

---

## Key Features

* **Clear separation** between short-term memory (Redis) and long-term persistence (Qdrant).
* **LangGraph integration** for building LLM-based agents.
* **Redis for memory**:

  * Super-fast in-memory performance.
  * Multi-process support and distributed scalability.
  * TTL and automatic data expiration.
* **Qdrant for persistence**:

  * Vector similarity search (text, images, embeddings).
  * Advanced queries with filters, payloads, and metadata.
  * Scales to millions of vectors.
* **LLM support**: OpenAI and **Ollama**.
* **Easy installation** via `pip`.
* **Grafana/Loki compatible logging** for observability.
* **Local Hugging Face embeddings support** for air-gapped environments.

---

## Memory vs Persistence

When developing agents with LangGraph (or LLM-based systems in general), it’s crucial to distinguish between **memory** and **persistence**.

### Persistence

**Definition:** Permanent/long-term storage of information that can be retrieved across sessions.

**Examples:**

* Conversation history
* Vector embeddings and knowledge bases
* Agent logs and audits

**Characteristics:**

* Non-volatile (survives crashes and restarts)
* Searchable and queryable across history
* Scalable for long-term growth

**Why Qdrant for persistence?**

* Specialized vector engine (similarity search)
* Reliable disk persistence
* Powerful API with filtering, metadata, and payloads
* Handles millions of vectors efficiently

---

### Memory

**Definition:** Temporary, session-specific information kept only during the task lifecycle.

**Examples:**

* Current conversation state
* Temporary variables
* Volatile graph step context

**Characteristics:**

* Volatile (lost on restart)
* Extremely fast (RAM-based)
* Can be shared across multiple processes/instances

**Why Redis for memory?**

* High-performance in-RAM operations
* Multi-worker scalability
* Simple API
* TTL with auto-cleaning

---

## Recommended Architecture

| Function        | Recommended Database | Main Reason                                      |
| --------------- | -------------------- | ------------------------------------------------ |
| **Memory**      | Redis                | Performance, simplicity, TTL, multi-process      |
| **Persistence** | Qdrant               | Vector search, reliable persistence, scalability |

---

## Installation

```bash
pip install memory-agent
```

---

## Usage Example (Redis + LangGraph)

```python
from memory_agent import MemoryCheckPointer
from memory_agent.openai import MemoryOpenAI
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
import os

os.environ["OPENAI_API_KEY"] = "sk-..."
llm = init_chat_model("openai:gpt-4.1")

class State(TypedDict):
    messages: Annotated[list, add_messages]

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")

async def main(user_input, thread_id):
    mem = MemoryOpenAI(
        model_embedding_name="text-embedding-3-small",
        qdrant_url="http://localhost:6333",
        key_search="memory_agent"
    )

    # Redis as checkpoint manager (temporary memory)
    async with MemoryCheckpointer.from_conn_info(
        host="localhost", port=6379, db=0
    ) as checkpointer:

        # Delete checkpoints older than 15 minutes
        await checkpointer.adelete_by_thread_id(
            thread_id=thread_id,
            filter_minutes=15
        )

        graph = graph_builder.compile(
            checkpointer=checkpointer,               # Persistence of checkpoints
            store=mem.get_in_memory_store(),         # Long-term memory store
        )
        graph.name = "ChatBot"

        result = await graph.ainvoke(
            {"messages": [{"role": "human", "content": user_input}]},
            config={"configurable": {"thread_id": thread_id}, "recursion_limit": 25}
        )
        print(result)
```

The `key_search` parameter specifies a unique search key that identifies an agent’s message context. When the library is used by multiple agents, provide distinct `key_search` values to keep each agent’s memory separate.

---

## Ollama — MemoryOllama

If you use [Ollama](https://ollama.com/) (or want a local embeddings server), you can initialize **MemoryOllama** like this:

```python
from memory_agent.ollama import MemoryOllama

memory_store = MemoryOllama(
    model_embedding_name="nomic-embed-text",
    model_embedding_url="http://localhost:11434",
    qdrant_url="http://localhost:6333",
    key_search="memory_agent"
)
```

> Use `model_embedding_name` to select the embedding model available on Ollama (e.g., `nomic-embed-text`) and `model_embedding_url` to point to your local instance.

---

## Vector Database (QDrant)

You can use QDrant directly as a vector store (synchronous or asynchronous), even without Redis:

```python
from memory_agent import MemoryPersistence

qdrant = MemoryPersistence(
    model_embedding_vs_name="BAAI/bge-large-en-v1.5",
    qdrant_url="http://localhost:6333"
)
client = qdrant.get_client()
client_async = qdrant.get_client_async()
```

`model_embedding_vs_name` specifies the embedding model used by QDrant (default: `BAAI/bge-large-en-v1.5`).
For available embedding models, see QDrant’s embeddings documentation.

---

## Custom Text Embedding Model

For better performance or air-gapped environments, you can download Hugging Face embedding models locally and configure QDrant to use them.

**1) Install Hugging Face client**

```bash
pip install --upgrade huggingface_hub
```

**2) Create model directories**

```bash
mkdir -p /models/multilingual-e5-large
mkdir -p /models/bge-small-en-v1.5
mkdir -p /models/bge-large-en-v1.5
```

**3) Download models**

```bash
huggingface-cli download intfloat/multilingual-e5-large --local-dir /models/multilingual-e5-large
huggingface-cli download BAAI/bge-small-en-v1.5 --local-dir /models/bge-small-en-v1.5
huggingface-cli download BAAI/bge-large-en-v1.5 --local-dir /models/bge-large-en-v1.5
```

**4) Configure QDrant to use local models**

```python
from memory_agent import MemoryPersistence

qdrant = MemoryPersistence(
    model_embedding_vs_name="BAAI/bge-large-en-v1.5",
    model_embedding_vs_path="/models/bge-large-en-v1.5",
    model_embedding_vs_type="local",
    qdrant_url="http://localhost:6333"
)
client = qdrant.get_client()
client_async = qdrant.get_client_async()
```

---

## [Docker](./docker/README.md)
