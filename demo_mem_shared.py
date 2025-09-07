import asyncio
from memory_agent.agent.ollama import AgentOllama
from demo_config import (
    thread_id,
    user_id,
    session_id,
    model_ollama,
    redis_config,
    model_embedding_vs_config,
    model_embedding_config,
    qdrant_config,
    collection_config
)
# semantic manage the memory and Ollama as LLM
agent_1 = AgentOllama(
    thread_id=thread_id,
    user_id=user_id,
    session_id=session_id,
    refresh_checkpointer=False,
    # https://python.langchain.com/api_reference/langchain/chat_models/langchain.chat_models.base.init_chat_model.html
    llm_config=model_ollama,
    host_persistence_config=redis_config,
    model_embedding_vs_config=model_embedding_vs_config,
    model_embedding_config=model_embedding_config,
    qdrant_config=qdrant_config,
    collection_config=collection_config,
    store_type="semantic"
)

agent_2 = AgentOllama(
    thread_id=thread_id,
    user_id=user_id,
    session_id=session_id,
    # https://python.langchain.com/api_reference/langchain/chat_models/langchain.chat_models.base.init_chat_model.html
    llm_config=model_ollama,
    collection_config=collection_config,
    store_type="semantic"
)


async def run_agent_1(msg: str):
    response = agent_1.invoke(msg)
    print(f"Agent 1 response: {response}")


async def run_agent_2(msg: str):
    response = agent_2.invoke(msg)
    print(f"Agent 2 response: {response}")


async def main():
    print("----")
    print("Running Agent 1\n")
    msg = "My name is Giuseppe. Remember that."
    await run_agent_1(msg)
    print("----")
    print("Running Agent 2\n")
    msg = "What is my name?"
    await run_agent_2(msg)


if __name__ == "__main__":
    asyncio.run(main())
