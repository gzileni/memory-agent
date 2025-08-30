from memory_agent.memory_agent import MemoryAgent
from memory_agent.ollama import MemoryOllama
from langgraph.store.memory import InMemoryStore


class AgentOllama(MemoryAgent):
    """
    An agent for managing and utilizing memory with the Ollama model.
    """
    mem: MemoryOllama
    key_search: str = "agent_ollama"

    def __init__(self, **kwargs):
        """
        Initialize the AgentOllama with the given parameters.
        Args:
            **kwargs: Arbitrary keyword arguments for configuration.
            key_search (str): The key to use for searching memories.
            mem (MemoryOllama): The memory instance to use for the agent.
        """
        super().__init__(**kwargs)
        self.key_search = kwargs.get("key_search", self.key_search)
        self.mem = MemoryOllama(
            key_search=self.key_search,
            **kwargs
        )
        self.model_name = kwargs.get("model_name", "llama3.1")
        self.model_provider = "ollama"
        self.base_url = kwargs.get("base_url", "https://localhost:11434")

    def store(self) -> InMemoryStore:
        return self.mem.in_memory_store()
