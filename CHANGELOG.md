# CHANGELOG

All significant changes to the project are documented here.

## [2.1.0] - 2025-01-28

### Added
- New `AgentOllama` class for enhanced integration with Ollama models
- New `AgentOpenAI` class for enhanced integration with OpenAI models  
- Enhanced `MemoryAgent` class for comprehensive memory operations management
- New `State` class for improved state management and summary tracking
- Dedicated `MemoryOllama` and `MemoryOpenAI` classes for model-specific memory handling
- GitHub Actions workflow for automatic Jekyll GitHub Pages deployment
- Enhanced logging system with `MemoryLog` utilities and Loki integration support
- Improved memory checkpointer functionality with better thread handling
- New memory Redis integration capabilities

### Changed
- Refactored memory agent architecture with reorganized flow and responsibilities
- Updated `MemoryStore` class with improved thread handling and clearer method naming
- Refactored `MemoryPersistence` class to align with new naming conventions
- Enhanced dependency management in `pyproject.toml` for better compatibility
- Improved documentation structure and Docker setup instructions
- Updated memory persistence layer with better error handling and logging
- Enhanced state management for better memory operation tracking

### Removed
- Removed entrypoint script for Qdrant as it is no longer needed
- Simplified deployment configuration by removing redundant components

### Notes
- Version 2.1.0 introduces significant architectural improvements for better memory management
- Enhanced integration with both Ollama and OpenAI models through dedicated agent classes
- Improved error handling and logging throughout the memory management pipeline
- Better separation of concerns between different memory operation types

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