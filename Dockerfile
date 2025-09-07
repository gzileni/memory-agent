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
  NEO4J_PLUGINS='["apoc"]' \
  NEO4J_apoc_export_file_enabled=true \
  NEO4J_apoc_import_file_enabled=true \
  NEO4J_apoc_import_file_use__neo4j__config=true \
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

# Neo4j Installation
RUN wget -O - https://debian.neo4j.com/neotechnology.gpg.key | apt-key add -
RUN echo 'deb https://debian.neo4j.com stable 5' | tee /etc/apt/sources.list.d/neo4j.list && \
    apt-get update
RUN apt-get install neo4j -y
COPY neo4j.conf /etc/neo4j/neo4j.conf

# Neo4J ports
EXPOSE 7474 7687

# Aggiungi libssl1.1 da Debian Bullseye con pinning e keyring corretti
RUN set -eux; \
  apt-get update; \
  apt-get install -y --no-install-recommends ca-certificates curl gnupg debian-archive-keyring; \
  echo 'deb [arch=amd64 signed-by=/usr/share/keyrings/debian-archive-keyring.gpg] https://deb.debian.org/debian bullseye main' \
    > /etc/apt/sources.list.d/debian-bullseye.list; \
  echo 'deb [arch=amd64 signed-by=/usr/share/keyrings/debian-archive-keyring.gpg] https://deb.debian.org/debian-security bullseye-security main' \
    > /etc/apt/sources.list.d/debian-bullseye-security.list; \
  printf 'Package: libssl1.1\nPin: release n=bullseye\nPin-Priority: 990\n' \
    > /etc/apt/preferences.d/libssl1.1.pref; \
  apt-get update; \
  apt-get install -y --no-install-recommends libssl1.1; \
  rm -rf /var/lib/apt/lists/*


# Installazione Redis Stack (APT ufficiale Redis)
# --- Redis Stack (APT ufficiale, con fallback codename) ---
# --- Redis CLI dagli apt Debian (bookworm) ---
# Lo installiamo PRIMA di aggiungere il repo Redis, così prende la versione nativa (libssl3)
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends redis-tools; \
    rm -rf /var/lib/apt/lists/*

# --- Redis Stack Server dal repo ufficiale Redis (bullseye) ---
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends ca-certificates curl gnupg lsb-release wget; \
    install -d -m 0755 /usr/share/keyrings; \
    curl -fsSL https://packages.redis.io/gpg \
      | gpg --dearmor --batch --yes --no-tty -o /usr/share/keyrings/redis-archive-keyring.gpg; \
    echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb bullseye main" \
      > /etc/apt/sources.list.d/redis.list; \
    apt-get update; \
    apt-get install -y --no-install-recommends redis-stack-server; \
    rm -rf /var/lib/apt/lists/*
# --- Fine Redis Stack ---

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