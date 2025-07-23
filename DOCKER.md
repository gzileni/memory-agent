# Docker

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

* **Redis** (in-memory DB/cache)
* **Qdrant** (vector database)

## Included Services

| Service   | Port  | Purpose                            |
| --------- | ----- | ---------------------------------- |
| Redis     | 6379  | Cache, message broker              |
| Qdrant    | 6333  | Vector search DB (API)             |
| Qdrant    | 6334  | gRPC API                           |

### Requirements

* Docker ≥ 20.10
* Docker Compose (plugin or standalone)
* At least 4GB RAM available (≥ 8GB recommended for Neo4j + Qdrant)

### Quick Start

1. **Start the stack:**

    ```bash
    docker compose up -d
    ```

2. **Check status:**

    ```bash
    docker compose ps
    ```

### Service Details

#### 1. Redis (`memory-redis`)

* **Port:** 6379
* **Persistent data:** `memory-redis-data`
* **Usage:** cache/session store for microservices, AI RAG, or NLP pipelines.
* **Integrated healthcheck.**

#### 2. Qdrant (`memory-qdrant`)

* **Platform:** `linux/amd64` (universal compatibility)
* **Ports:** 6333 (REST), 6334 (gRPC)
* **Persistent data:** `memory-qdrant-data`
* **Custom config:** mounts `./qdrant/config.yml`
* **Usage:** vector DB for semantic search (e.g., with LangChain, Haystack...)

### Networks, Volumes, and Security

* All services are on the **private Docker network** `memory-network` (`172.16.110.0/24`)
* **Docker volumes:** all data is persistent and will not be lost between restarts.
* **Security tip:** Always change default passwords!

### Service Access

* **Qdrant API:** [http://localhost:6333](http://localhost:6333)
* **Redis:** `redis://localhost:6379`

### FAQ & Troubleshooting

* **Q: Where is persistent data stored?**  
    A: In Docker volumes. Check with `docker volume ls`.

* **Q: Qdrant doesn't start on Apple Silicon?**  
    A: Specify `platform: linux/amd64` as already set in the file.

### Extra: Cleanup Example

To remove **all** the stack and associated data:

```bash
docker compose down -v
```
