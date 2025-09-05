import os
from memory_agent.memory_agent import MemoryAgent
from memory_agent.openai import MemoryOpenAI
from langgraph.store.memory import InMemoryStore


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

        self.mem = MemoryOpenAI(
            **kwargs
        )

        self.llm_config = {
            "model": "gpt-4.1-mini",
            "model_provider": "openai",
            "api_key": os.getenv("OPENAI_API_KEY"),
            "base_url": None,
            "temperature": self.TEMPERATURE_DEFAULT,
        }

    def store(self) -> InMemoryStore:
        return self.mem.in_memory_store()
