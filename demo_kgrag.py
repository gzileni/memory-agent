import asyncio
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


async def ingestion(path: str) -> str:
    async for d in kgrag_ollama.process_documents(
        path=path,
        force=True
    ):
        if d == "ERROR":
            return f"Error processing document {path}."
    return f"Document {path} ingested successfully."


async def query(prompt: str) -> str:
    response = await kgrag_ollama.query(prompt)
    return response


async def main():
    path = "/Users/giuseppezileni/arxiv/2508.20435v1.pdf"
    result = await ingestion(path)
    print(result)
    result = await query(
        "Come misurare empiricamente il “consumo effettivo” "
        "e stimare la CAWF?"
    )
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
