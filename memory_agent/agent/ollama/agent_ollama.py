import json
import requests
from memory_agent.agent.memory_agent import MemoryAgent
from memory_agent.kgrag.ollama import MemoryOllama
from typing import Any


class AgentOllama(MemoryAgent):
    """
    An agent for managing and utilizing memory with the Ollama model.
    """

    mem: MemoryOllama

    def __init__(self, **kwargs):
        """
        Initialize the AgentOllama with the given parameters.
        Args:
            **kwargs: Arbitrary keyword arguments for configuration.
            key_search (str): The key to use for searching memories.
            mem (MemoryOllama): The memory instance to use for the agent.
        """
        super().__init__(**kwargs)
        self.mem = MemoryOllama(**kwargs)
        self.llm_config = {
            "model": "llama3.1",
            "model_provider": "ollama",
            "api_key": None,
            "base_url": "http://localhost:11434",
            "temperature": self.TEMPERATURE_DEFAULT,
        }
        self.ollama_pull()

    def ollama_pull(self) -> tuple[bool, str]:
        """
        Pulls a model from the Ollama server.

        Args:
            ollama_url (str): The base URL of the Ollama server.
            model_name (str): The name of the model to pull.

        Returns:
            dict: The response from the Ollama server.
        """
        payload = {"name": self.llm_config["model"]}
        ollama_api = f"{self.llm_config['base_url']}/api/pull"
        error: bool = False
        response: str = ""

        with requests.post(ollama_api, json=payload, stream=True) as r:
            for line in r.iter_lines():
                if line:
                    data = json.loads(line.decode('utf-8'))

                    if data is None:
                        response = (
                            f"Model {self.llm_config['model']} "
                            "not found on Ollama "
                            "server."
                        )
                        error = True
                        break

                    if "error" in data:
                        response = (
                            f"Error pulling model {self.llm_config['model']}: "
                            f"{data['error']}"
                        )
                        error = True
                        break

                    if "status" not in data:
                        response = (
                            "Unexpected response format for model "
                            f"{self.llm_config['model']}: {data}"
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

    def embed_query(self, text: str) -> list[float]:
        return self.mem.model_embedding.embed_query(text)

    def index_store(self) -> Any:
        return {
            "embed": self.mem.model_embedding,
            "dims": self.mem._get_collection_dim(),
            "fields": ["$"]
        }
