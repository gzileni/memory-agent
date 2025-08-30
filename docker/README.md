# memory-agent — Docker Compose

This repository folder contains the Docker Compose configuration for the memory-agent service stack (Redis, Qdrant, Loki, Promtail and Grafana).

## Overview

Services exposed by the compose file
- Grafana: http://localhost:3000 (anonymous access enabled)
- Loki: http://localhost:3100
- Redis: port 6379
- Qdrant: ports 6333 (http) and 6334 (internal)

Persistent volumes are defined for Qdrant, Redis, Grafana and Promtail logs.

## Prerequisites

- Docker Engine
- Docker Compose (v2 recommended) or docker-compose
- A valid OpenAI API key (if memory-agent uses OpenAI)

## Files

- docker-compose.yml — service definitions
- README.md — this file
- .env (not committed) — runtime environment values
- .env.example — example environment file (provided below)

## Quick start

1. Copy the example env file:
    `cp .env.example .env`

2. Edit `.env` and set your real values (OpenAI API key, model, etc).

3. Start the stack:
    `docker compose up -d`
    or, if you use the standalone binary:
    `docker-compose up -d`

4. Check logs:
    `docker compose logs -f memory-agent`
    `docker compose logs -f memory-agent-grafana`

5. Stop and remove containers (optionally remove volumes):
    `docker compose down`
    `docker compose down -v  # also remove named volumes`

or 

`make run`

## Environment variables

Important variables used by the memory-agent service (define them in `.env`):

- `OPENAI_API_KEY` — API key for OpenAI (if applicable)
- `COLLECTION_NAME` — name of the vector collection in Qdrant
- `MODEL_EMBEDDING` — embedding model identifier (e.g. text-embedding-3-small)
- `LLM_URL` — URL for an LLM API (if using a self-hosted/remote LLM)
- `LOKI_URL` — Loki push endpoint (default in compose: http://kgrag-mcp-loki:3100/loki/api/v1/push)

The compose file also maps services to ports on localhost so you can connect to them directly for development.

## Ports summary

- 3000 — Grafana UI
- 3100 — Loki API
- 6379 — Redis
- 6333 — Qdrant HTTP
- 6334 — Qdrant internal/replication

## Persistence

Named volumes (defined in docker-compose.yml):
- qdrant_data
- redis_data
- grafana_data
- loki_log

To reset data, stop the stack and remove volumes:
`docker compose down -v`

or 

`make stop`

## Troubleshooting

- If a port is already in use, change the host port mapping in docker-compose.yml.
- If Grafana is unreachable, confirm the container is running and port 3000 is free.
- Check container logs with: docker compose logs -f <service-name>

## Example .env.example

Create a file named `.env` from the following template and fill values:

```env
# Required: OpenAI API key (if used)
OPENAI_API_KEY=your_openai_api_key_here

# Qdrant collection name used by memory-agent
COLLECTION_NAME=memory_agent_collection

# Embedding model, e.g. text-embedding-3-small
MODEL_EMBEDDING=text-embedding-3-small

# Optional LLM endpoint (if using a remote LLM)
LLM_URL=http://localhost:8080

# Loki push URL (optional override)
LOKI_URL=http://localhost:3100/loki/api/v1/push
```

That's all — after populating `.env` run `docker compose up -d` to start the system.
