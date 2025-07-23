# Usa l'immagine base di qdrant con la versione desiderata
FROM qdrant/qdrant:v1.13.0

# Crea la directory di lavoro
RUN mkdir -p /qdrant

# Copia del file di configurazione (opzionale: se vuoi includerlo nell'immagine)
# Assicurati che "qdrant/config.yml" sia presente nella directory di build
COPY ./config.yml /qdrant/config.yml

# Espone le porte necessarie
EXPOSE 6333 6334 6335

# Imposta l'entrypoint per l'avvio del container
ENTRYPOINT ["/qdrant/entrypoint.sh"]
