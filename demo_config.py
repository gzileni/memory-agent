import os
from qdrant_client.http.models import Distance
from typing import Any

thread_id = "thread_demo"
user_id = "user_demo"
session_id = "session_demo"
collection_name = "agent_ollama_demo"

neo4j_auth: dict[str, Any] = {
    "url": "neo4j://localhost:7687",
    "username": None,
    "password": None,
    "database": None,
}

aws_config = {
    "access_key_id": os.environ.get("AWS_ACCESS_KEY_ID"),
    "secret_access_key": os.environ.get("AWS_SECRET_ACCESS_KEY"),
    "bucket": os.environ.get("AWS_BUCKET_NAME"),
    "region": os.environ.get("AWS_REGION")
}

model_ollama = {
    "model": "llama3.1",
    "model_provider": "ollama",
    "api_key": None,
    "base_url": "http://localhost:11434",
    "temperature": 0.5,
}

redis_config = {
    "host": "localhost",
    "port": 6379,
    "db": 0,
}

model_embedding_vs_config = {
    "path": None,
    "type": "hf",
    "name": "BAAI/bge-large-en-v1.5"
}

collection_config = {
    "collection_name": collection_name,
    "vectors_config": {
        "size": 768,
        # COSINE = "Cosine"
        # EUCLID = "Euclid"
        # DOT = "Dot"
        # MANHATTAN = "Manhattan"
        "distance": Distance.COSINE
    }
}

qdrant_config = {
    "url": "http://localhost:6333",
}

model_embedding_config = {
    "name": "nomic-embed-text",
    "url": "http://localhost:11434"
}
