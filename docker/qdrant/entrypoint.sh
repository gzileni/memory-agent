#!/bin/bash

# Abilita il controllo degli errori
set -e

# Messaggio di avvio
echo "Avvio del container Qdrant..."

# Configura variabili di ambiente (se necessario)
export QDRANT__LOG_LEVEL="${QDRANT__LOG_LEVEL:-INFO}"
export QDRANT__SERVICE_GRPC_PORT="${QDRANT__SERVICE_GRPC_PORT:-6334}"
export QDRANT__SERVICE_HTTP_PORT="${QDRANT__SERVICE_HTTP_PORT:-6333}"

# Assicurati che il volume sia montato correttamente
if [ ! -d "/qdrant/storage" ]; then
  echo "Creazione della directory di storage..."
  mkdir -p /qdrant/storage
fi

# Avvia Qdrant
echo "Avvio di Qdrant..."
exec qdrant
