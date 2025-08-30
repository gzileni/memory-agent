# CHANGELOG

All significant changes to the project are documented here.

## [2.0.0] - 2025-08-20

### Added
- Support for both CPU and GPU Qdrant builds via Dockerfile.
- New `memory-agent` service in docker-compose to replace Redis and Qdrant.
- Dedicated classes for embeddings: `MemoryOllama` and `MemoryOpenAI`.
- GitHub Actions workflow for automatic publishing of Docker images.
- Detailed README for Docker setup and usage.

### Changed
- Refactored memory agent architecture; reorganized flow and responsibilities.
- Refactored `memory.py` and `memory_persistence.py` to simplify management of embedding models and configuration.
- Updated `docker-compose.yml` to use `memory-agent` instead of Redis/Qdrant.
- Extended Makefile with support for environment files in Docker commands.
- Updated management of environment variables; added configurations for Grafana and Loki.
- Version updated to 2.0.0 in `pyproject.toml`.

### Breaking changes
- Memory architecture redesigned: Redis and Qdrant services are removed from the compose. Data/configurations must be migrated to the new `memory-agent` and deployment instances updated accordingly.

### Notes
- Refer to the new README for complete instructions on migration, build (CPU/GPU), and deployment.
- Check the GitHub Actions workflow for details on Docker image publishing.
- Review the new environment variables for Grafana/Loki and update monitoring configurations.