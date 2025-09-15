import asyncio
from demo_kgrag_ollama import kgrag_ollama


async def ingestion(path: str):
    async for d in kgrag_ollama.process_documents(
        path=path,
        force=True
    ):
        if d == "ERROR":
            return f"Error processing document {path}."
    return f"Document {path} ingested successfully."


async def main():
    path = "/Users/giuseppezileni/arxiv/2508.20435v1.pdf"
    result = await ingestion(path)
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
