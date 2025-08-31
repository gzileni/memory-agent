ARG BASE_IMAGE_CPU=qdrant/qdrant:v1.15.0
ARG BASE_IMAGE_GPU=qdrant/qdrant:gpu-nvidia-latest
ARG PLATFORM=cpu

# === Stage per CPU ===
FROM ${BASE_IMAGE_CPU} AS base_cpu

# === Stage per GPU ===
FROM ${BASE_IMAGE_GPU} AS base_gpu

# === Seleziona lo stage finale ===
ARG PLATFORM
FROM base_${PLATFORM} AS final

# QDrant port
EXPOSE 6333 6334

ENV \
  VECTORDB_SENTENCE_TYPE="hf" \
  VECTORDB_SENTENCE_MODEL="BAAI/bge-small-en-v1.5" \
  QDRANT_URL="http://localhost:6333" \
  REDIS_URL="redis://localhost:6379" \
  REDIS_HOST="localhost" \
  REDIS_PORT=6379 \
  REDIS_DB=10

# Installazione di Redis
RUN apt-get update && \
    apt-get install -y lsb-release curl gpg wget && \
    curl -fsSL https://packages.redis.io/gpg | gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg && \
    chmod 644 /usr/share/keyrings/redis-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb $(lsb_release -cs) main" > /etc/apt/sources.list.d/redis.list && \
    apt-get install -y supervisor redis redis-server && \
    mkdir -p /var/log/supervisor

COPY qdrant.yml /etc/qdrant/config.yml

# Redis port
EXPOSE 6379

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        lsb-release \
        build-essential \
        python3 \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf
CMD ["supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]