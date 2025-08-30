
# CHANGELOG

Tutte le modifiche significative al progetto sono documentate qui.

## [2.0.0] - 2025-08-30

### Aggiunte
- Supporto per build Qdrant sia CPU che GPU tramite Dockerfile.
- Nuovo servizio `memory-agent` in docker-compose per sostituire Redis e Qdrant.
- Classi dedicate per gli embedding: `MemoryOllama` e `MemoryOpenAI`.
- Workflow GitHub Actions per pubblicazione automatica delle immagini Docker.
- README dettagliato per setup e utilizzo Docker.
  
### Modifiche
- Refactor dell'architettura del memory agent; riorganizzazione del flusso e delle responsabilità.
- Refactoring di `memory.py` e `memory_persistence.py` per semplificare la gestione dei modelli di embedding e la configurazione.
- `docker-compose.yml` aggiornato per usare `memory-agent` al posto di Redis/Qdrant.
- Makefile esteso con supporto a file di ambiente per comandi Docker.
- Gestione delle variabili d'ambiente aggiornata; aggiunte configurazioni per Grafana e Loki.
- Versione aggiornata a 2.0.0 in `pyproject.toml`.

### Rotture (Breaking changes)
- Architettura di memoria riprogettata: i servizi Redis e Qdrant sono rimossi dal compose. È necessario migrare i dati/configurazioni verso il nuovo `memory-agent` e aggiornare le istanze di deployment di conseguenza.

### Note
- Consultare il nuovo README per istruzioni complete su migrazione, build (CPU/GPU) e deployment.
- Controllare il workflow GitHub Actions per dettagli sulla pubblicazione delle immagini Docker.
- Verificare le nuove variabili d'ambiente per Grafana/Loki e aggiornare le configurazioni di monitoraggio.
