# CHANGELOG

All significant changes to the project are documented here.

## [2.1.2] - 2025-08-30

### Added
- Added functionality to extract the model from the Ollama server.
- New packages added to the project for improved compatibility and features.

### Changed
- Updated imports to reflect the new `memory_agent` package structure.
- Updated README title to include link to docker-compose file.

### Fixed
- Updated CHANGELOG to include new classes and significant changes for v2.1.0.
- Translated CHANGELOG to English and updated sections for consistency.

### Chore
- Removed Release Drafter configuration file.

### Notes
- Version bumped to 2.1.2 to reflect compatibility and feature improvements.
- See README for latest setup instructions.

## [2.1.0] - 2025-08-30

### Added
- Introduced `AgentOllama` and `AgentOpenAI` classes for better integration with respective models.
- Implemented a new `MemoryAgent` class to manage memory operations and interactions.
- Added a new `State` class to track summaries and improve state management.
- Updated dependencies in `pyproject.toml` and `requirements.txt` for compatibility and new features.

### Changed
- Refactored memory agent architecture and enhanced functionality.
- Removed the entrypoint script for Qdrant as it is no longer needed.
- Updated `MemoryStore` class to improve thread handling and renamed methods for clarity.
- Refactored `MemoryPersistence` class to align with new naming conventions.

### Breaking changes
- Significant refactor: previous Qdrant entrypoint and legacy memory flow removed. Ensure you migrate custom scripts and integrations to use the new agent classes and state management flow.

### Notes
- Version bumped to 2.1.0 to reflect substantial improvements.
- See README and migration docs for update instructions and compatibility notes.


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
