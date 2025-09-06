import uuid
import asyncio
from memory_agent.agent.ollama import AgentOllama
from qdrant_client.http.models import Distance

thread_id = "demo_agent_ollama"
user_id = "user_demo"
session_id = str(uuid.uuid4())
collection_name = "agent_ollama_demo"

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
        "size": 1536,
        # COSINE = "Cosine"
        # EUCLID = "Euclid"
        # DOT = "Dot"
        # MANHATTAN = "Manhattan"
        "distance": Distance.COSINE
    }
}

model_embedding_config = {
    "name": "nomic-embed-text",
    "url": "http://localhost:11434"
}

# semantic manage the memory and Ollama as LLM
agent = AgentOllama(
    thread_id=thread_id,
    user_id=user_id,
    session_id=session_id,
    refresh_checkpointer=False,
    # https://python.langchain.com/api_reference/langchain/chat_models/langchain.chat_models.base.init_chat_model.html
    llm_config=model_ollama,
    host_persistence_config=redis_config,
    model_embedding_vs_config=model_embedding_vs_config,
    model_embedding_config=model_embedding_config,
    qdrant_config={
        "url": "http://localhost:6333",
    },
    collection_config=collection_config,
    store_type="semantic"
)


async def run_agent(msg: str):
    response = await agent.ainvoke(msg)
    if response.get("error") is not None:
        err = response.get("error")
        if err is not None:
            print("Error: ", err.get("message"))
            return
    result = response.get("result")
    if result is not None:
        print(result.get("content"))


async def run_agent_stream(msg: str):
    async for token in agent.astream(msg):
        if token.get("error") is not None:
            err = token.get("error")
            if err is not None:
                print("Error: ", err.get("message"))
                return
        print("---> ", end="", flush=True)
        result = token.get("result")
        if result is not None:
            print(result.get("content"))


async def main():
    msg = "My name is Giuseppe. Remember that."
    await run_agent(msg)
    # msg = "What is the capital of France?"
    # await run_agent_stream(msg)
    msg = "What is my name?"
    await run_agent_stream(msg)


if __name__ == "__main__":
    asyncio.run(main())
