import os
from memory_agent.agent.memory_agent import MemoryAgent
from memory_agent.kgrag.openai import MemoryOpenAI
from typing import Any


class AgentOpenAI(MemoryAgent):
    """
    An agent for managing and utilizing memory with the OpenAI model.
    Args:
        **kwargs: Arbitrary keyword arguments for configuration.
        key_search (str): The key to use for searching memories.
        mem (MemoryOpenAI): The memory instance to use for the agent.
    """

    mem: MemoryOpenAI

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mem = MemoryOpenAI(**kwargs)
        self.llm_config = {
            "model": "gpt-4.1-mini",
            "model_provider": "openai",
            "api_key": os.getenv("OPENAI_API_KEY"),
            "base_url": None,
            "temperature": self.TEMPERATURE_DEFAULT,
        }

    def index_store(self) -> Any:
        return {
            "embed": self.mem.model_embedding,
            "dims": self.mem._get_collection_dim(),
            "fields": ["$"]
        }

    def embed_query(self, text: str) -> list[float]:
        return self.mem.model_embedding.embed_query(text)
