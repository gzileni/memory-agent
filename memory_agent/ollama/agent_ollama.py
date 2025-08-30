import json
import requests
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
        self.ollama_pull()

    def store(self) -> InMemoryStore:
        return self.mem.in_memory_store()

    def ollama_pull(self) -> tuple[bool, str]:
        """
        Pulls a model from the Ollama server.

        Args:
            ollama_url (str): The base URL of the Ollama server.
            model_name (str): The name of the model to pull.

        Returns:
            dict: The response from the Ollama server.
        """
        payload = {"name": self.model_name}
        ollama_api = f"{self.base_url}/api/pull"
        error: bool = False
        response: str = ""

        with requests.post(ollama_api, json=payload, stream=True) as r:
            for line in r.iter_lines():
                if line:
                    data = json.loads(line.decode('utf-8'))

                    if data is None:
                        response = (
                            f"Model {self.model_name} not found on Ollama "
                            "server."
                        )
                        error = True
                        break

                    if "error" in data:
                        response = (
                            f"Error pulling model {self.model_name}: "
                            f"{data['error']}"
                        )
                        error = True
                        break

                    if "status" not in data:
                        response = (
                            "Unexpected response format for model "
                            f"{self.model_name}: {data}"
                        )
                        error = True
                        break

                    if data.get("status") == "success":
                        response = "Modello scaricato con successo!"
                        error = False
                        break

                    if data.get("status") == "error":
                        response = f"Errore durante il download: {data}"
                        error = True
                        break

                    if data.get("status") == "stream":
                        self.logger.debug("Streaming output:", data)

        return error, response
