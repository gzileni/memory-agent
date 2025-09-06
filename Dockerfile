# ------------------------------------------------------------
# Base
# ------------------------------------------------------------
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 LC_ALL=C.UTF-8

# ------------------------------------------------------------
# System deps + supervisord
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl gpg lsb-release \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# Install Redis Stack (includes RediSearch + RedisJSON)
# Docs: add official Redis APT repo, then install redis-stack-server
# ------------------------------------------------------------
RUN set -eux; \
    curl -fsSL https://packages.redis.io/gpg \
      | gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg; \
    chmod 644 /usr/share/keyrings/redis-archive-keyring.gpg; \
    echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb $(lsb_release -cs) main" \
      > /etc/apt/sources.list.d/redis.list; \
    apt-get update; \
    apt-get install -y --no-install-recommends redis-stack-server redis-tools; \
    rm -rf /var/lib/apt/lists/*

# Dati Redis
RUN mkdir -p /data && chown -R root:root /data
VOLUME ["/data"]

# ------------------------------------------------------------
# Supervisord
# Nota: il tuo supervisord.conf deve avere un programma 'redis' tipo:
#   [program:redis]
#   command=/usr/bin/redis-stack-server --protected-mode no --daemonize no --dir /data
#   autostart=true
#   autorestart=true
#   priority=5
# ------------------------------------------------------------
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# ------------------------------------------------------------
# Porte
# 6379 = Redis; 8001 = RedisInsight (se in futuro vorrai usarla)
# ------------------------------------------------------------
EXPOSE 6379 8001

# ------------------------------------------------------------
# Healthcheck: verifica che Redis risponda a PING
# ------------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=5 \
  CMD redis-cli -h 127.0.0.1 -p 6379 ping | grep -q PONG || exit 1

# ------------------------------------------------------------
# Avvio
# ------------------------------------------------------------
CMD ["supervisord", "-n", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
