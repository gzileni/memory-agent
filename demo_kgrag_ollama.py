from memory_agent.kgrag.ollama import KGragOllama
from demo_config import (
    model_ollama,
    neo4j_auth,
    redis_config,
    model_embedding_vs_config,
    collection_config,
    model_embedding_config,
    aws_config,
    qdrant_config
)

kgrag_ollama = KGragOllama(
    path_type="fs",
    path_download=aws_config.get("path_download"),
    format_file="pdf",
    neo4j_auth=neo4j_auth,
    qdrant_config=qdrant_config,
    host_persistence_config=redis_config,
    aws_config=aws_config,
    model_embedding_config=model_embedding_config,
    model_embedding_vs_config=model_embedding_vs_config,
    collection_config=collection_config,
    llm_config=model_ollama
)
