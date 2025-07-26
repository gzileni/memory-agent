# [memory-agent](https://github.com/gzileni/memory-agent)

The library allows you to manage both [persistence](https://langchain-ai.github.io/langgraph/how-tos/persistence/) and [**memory**](https://langchain-ai.github.io/langgraph/concepts/memory/#what-is-memory) for a LangGraph agent.

**memory-agent** uses [Redis](https://redis.io/) as the long-term memory database and [QDrant](https://qdrant.tech/) for persistence.

## Memory vs Persistence

When developing agents with LangGraph (or LLM-based systems in general), it's crucial to distinguish between **memory** and **persistence** of state and data.  
This distinction affects both the architecture and the choice of databases used in the project.

### Persistence

**Persistence** refers to the permanent (or long-term) storage of information that needs to be retrieved across different sessions or after a long period of time.

**Examples of persistence:**  

- Conversation history  
- Vector embeddings and knowledge bases  
- Agent logs and audits

**Characteristics of persistence:**  

- **Non-volatile data**: survives restarts, crashes, and scales over time  
- **Historical access**: you can search, filter, and retrieve data even after a long time  
- **Optimized for complex queries**

#### Why use Qdrant for persistence?

- **Vectorization & Similarity**: Qdrant is a specialized engine for similarity search between embeddings (LLM, NLP, images, etc.), ideal for agents that need to retrieve information, conversation history, knowledge bases, etc.
- **Reliable persistence**: Qdrant securely and efficiently saves all data to disk.
- **Scalability**: Handles millions of vectors and high-performance similarity queries, even at large scale.
- **Powerful API**: Supports filters, payloads, metadata, and advanced queries, perfect for integrating complex data into LangGraph agents.

### Memory

**Memory** represents all the temporary information that the agent keeps only during a session or the lifecycle of a specific task.

**Examples of memory:**  

- Current conversation state  
- Temporary variables  
- Volatile context between graph steps

**Characteristics of memory:**  

- **Volatile**: lost when the process ends or is restarted  
- **Very fast**: only used for short-lived data  
- **Scalable**: can be shared across multiple processes/instances if needed

#### Why use Redis for memory?

- **Performance**: Redis operates entirely in RAM, ensuring extremely fast reads and writes—ideal for temporary data and frequent access.
- **Multi-process & Scalability**: Redis allows multiple agent instances to access/share the same temporary state, which is essential in distributed environments or with multiple workers.
- **Ease of use**: Redis provides simple primitives (hashes, lists, sets) and an API that is easy to integrate with Python.
- **Expiration support (TTL)**: You can set automatic expiration on data, so temporary memory “self-cleans”.

#### Architectural Choice

| Function             | Recommended Database | Reasoning                                              |
|----------------------|---------------------|--------------------------------------------------------|
| Memory               | Redis               | Performance, multi-process, data expiration, simplicity|
| Persistence          | Qdrant              | Vectorization, semantic similarity, scalability        |

### Installation

To install **memory-agent** via pip, run:

```bash
pip install memory-agent
```

### Usage Example

Below is a practical example of how to use the library to manage long-term memory with Redis in a LangGraph agent.

```python
import os
from memory_agent import MemoryCheckPointer, MemoryPersistence

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model

os.environ["OPENAI_API_KEY"] = "sk-..."
llm = init_chat_model("openai:gpt-4.1")

class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

graph_builder = StateGraph(State)
# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")

async def main(user_input, thread_id):
    # Configurazione della connessione a Redis
    memory_checkpointer_config = {
        "host": "localhost",
        "port": 6379,
        "db": 0
    }

    # Creazione della configurazione per il thread
    config = {
        "configurable": {
            "thread_id": thread_id
        },
        "recursion_limit": 25
    }

    memory_store = MemoryPersistence(model_embeggind_type="openai", 
                                     model_embedding_name="text-embedding-3-small",
                                     qdrant_url="http://localhost:6333")
        
    # Utilizzo del context manager per la memoria Redis
    async with MemoryCheckpointer.from_conn_info(
        host=memory_checkpointer_config["host"],
        port=memory_checkpointer_config["port"],
        db=memory_checkpointer_config["db"]
    ) as checkpointer:

        # Delete checkpoints older than 15 minutes for the current thread
        await checkpointer.adelete_by_thread_id(thread_id=thread_id, filter_minutes=15)

        # Compiling the graph with the checkpointer and in-memory store
        graph_sql = graph_builder.compile(
          checkpointer=checkpointer,  # Persistence
          store=memory_store.get_in_memory_store(),  # Long-term memory
        )
        graph_sql.name = "ChatBot"

        # Run the graph with user input
        input_data = {
          "messages": [{
            "role": "human",
            "content": user_input,
          }]
        }
        result = await graph_sql.ainvoke(input_data, config=config)
        print(result)
```

#### Ollama or VLLM

If you use [Ollama](https://ollama.com/) or a custom LLM server such as [VLLM](https://docs.vllm.ai/en/latest/), you need to initialize the `MemoryPersistence` object as follows:

```python
memory_store = MemoryPersistence(model_embeggind_type="ollama", 
                                 model_embedding_name="nomic-embed-text",
                                 model_embedding_url="http://localhost:11434/api/embeddings",
                                 qdrant_url="http://localhost:6333")
```

```python
memory_store = MemoryPersistence(model_embeggind_type="vllm", 
                                 model_embedding_name="....",
                                 model_embedding_url="....",
                                 qdrant_url="http://localhost:6333")
```

### Vector Database

Two QDrant instances are available for use as a vector database: one synchronous and one asynchronous. You can use QDrant directly as a vector store without the Redis component, for example:

```python
import os
from memory_agent import MemoryPersistence

# Istanza sincrona di QDrant
qdrant = MemoryPersistence(model_embedding_vs_name="BAAI/bge-large-en-v1.5", 
                           qdrant_url="http://localhost:6333")
client = qdrant.get_client()  
client_async = qdrant.get_client_async()
```

These instances allow you to use only the QDrant database for vector memory management, either synchronously or asynchronously, depending on your application's needs.

### Custom Text Embedding Model

By default, QDrant automatically downloads text embedding models from Hugging Face. However, to improve performance or work in environments without Internet access, you can download the models locally and configure QDrant (or your application) to use these local paths.

#### Downloading and Using Local Embedding Models

1 - **Install the Hugging Face client:**

  ```bash
  pip install --upgrade huggingface_hub
  ```

2 - **Create directories for the models:**

  ```bash
  mkdir -p /models/multilingual-e5-large
  mkdir -p /models/bge-small-en-v1.5
  mkdir -p /models/bge-large-en-v1.5
  ```

3 - **Download the desired models:**

  ```bash
  huggingface-cli download intfloat/multilingual-e5-large --local-dir /models/multilingual-e5-large
  huggingface-cli download BAAI/bge-small-en-v1.5 --local-dir /models/bge-small-en-v1.5
  huggingface-cli download BAAI/bge-large-en-v1.5 --local-dir /models/bge-large-en-v1.5
  ```

4 - **Configure your application or QDrant** to use the local paths of the downloaded models instead of downloading them from Hugging Face each time.

```python
import os
from memory_agent import MemoryPersistence

# Istanza sincrona di QDrant
qdrant = MemoryPersistence(model_embedding_vs_name="BAAI/bge-large-en-v1.5", 
                           model_embedding_vs_path="/models/bge-large-en-v1.5"
                           model_embedding_vs_type="local",
                           qdrant_url="http://localhost:6333")
client = qdrant.get_client()  
client_async = qdrant.get_client_async()
```

## Docker

To easily start the required services (Redis, QDrant), you can use the following `docker-compose.yml` file:

```yaml

services:

  memory-redis:
    container_name: memory-redis
    restart: always
    image: redis:7
    ports:
      - "6379:6379"
    volumes:
      - memory-redis-data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 2s
      timeout: 2s
      retries: 30
    networks:
      - memory-network

  memory-qdrant:
    container_name: memory-qdrant
    platform: linux/amd64
    image: qdrant/qdrant:v1.13.4
    restart: always
    ports:
      - "6333:6333"
      - "6334:6334"
    expose:
      - 6333
      - 6334
      - 6335
    volumes:
      - memory-qdrant-data:/qdrant/storage:z
      - ./qdrant/config.yml:/qdrant/config.yml:ro
    networks:
      - memory-network

volumes:
  memory-qdrant-data:
  memory-redis-data:

networks:
  memory-network:
    name: memory-network
    driver: bridge
    ipam:
      driver: default
      config:
        - subnet: 172.16.110.0/24

```

This **Docker Compose** stack integrates the main services for Retrieval-Augmented Generation (RAG) projects, knowledge graphs, and log monitoring:

- **Redis** (in-memory DB/cache)
- **Qdrant** (vector database)

### Included Services

| Service   | Port  | Purpose                            |
| --------- | ----- | ---------------------------------- |
| Redis     | 6379  | Cache, message broker              |
| Qdrant    | 6333  | Vector search DB (API)             |
| Qdrant    | 6334  | gRPC API                           |

#### Requirements

- Docker ≥ 20.10
- Docker Compose (plugin or standalone)
- At least 4GB RAM available (≥ 8GB recommended for Neo4j + Qdrant)

#### Quick Start

1. **Start the stack:**

    ```bash
    docker compose up -d
    ```

2. **Check status:**

    ```bash
    docker compose ps
    ```

#### Service Details

##### 1. Redis (`memory-redis`)

- **Port:** 6379
- **Persistent data:** `memory-redis-data`
- **Usage:** cache/session store for microservices, AI RAG, or NLP pipelines.
- **Integrated healthcheck.**

##### 2. Qdrant (`memory-qdrant`)

- **Platform:** `linux/amd64` (universal compatibility)
- **Ports:** 6333 (REST), 6334 (gRPC)
- **Persistent data:** `memory-qdrant-data`
- **Custom config:** mounts `./qdrant/config.yml`
- **Usage:** vector DB for semantic search (e.g., with LangChain, Haystack...)

#### Networks, Volumes, and Security

- All services are on the **private Docker network** `memory-network` (`172.16.110.0/24`)
- **Docker volumes:** all data is persistent and will not be lost between restarts.
- **Security tip:** Always change default passwords!

#### Service Access

- **Qdrant API:** [http://localhost:6333](http://localhost:6333)
- **Redis:** `redis://localhost:6379`

#### FAQ & Troubleshooting

- **Q: Where is persistent data stored?**  
    A: In Docker volumes. Check with `docker volume ls`.

- **Q: Qdrant doesn't start on Apple Silicon?**  
    A: Specify `platform: linux/amd64` as already set in the file.

#### Extra: Cleanup Example

To remove **all** the stack and associated data:

```bash
docker compose down -v
```

## Grafana Logging

To enable logging compatible with Grafana and Loki, simply set the following environment variables:

- `APP_NAME`: The name of your application (default: `"logger"`)
- `LOKI_URL`: The URL of your Loki instance (for example: `"http://localhost:3100/loki/api/v1/push"`)
- `LOG_LEVEL`: The desired log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, or `CRITICAL`)
- `APP_SERVICE`: The name of the service (default: `"logger_service"`)
- `APP_VERSION`: The version of your application (default: `"1.0.0"`)

Once these variables are set, your logs will be compatible with Grafana dashboards and Loki log aggregation.
