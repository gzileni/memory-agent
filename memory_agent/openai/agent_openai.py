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
    key_search: str = "agent_openai"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.llm_api_key is None:
            raise ValueError("LLM_API_KEY must be set")

        self.key_search = kwargs.get(
            "key_search",
            self.key_search
        )
        self.mem = MemoryOpenAI(
            key_search=self.key_search,
            **kwargs
        )
        self.model_name = kwargs.get("model_name", "gpt-4.1-mini")
        self.model_provider = "openai"
        self.base_url = kwargs.get("base_url", None)

    def store(self) -> InMemoryStore:
        return self.mem.in_memory_store()
