from memory_agent.kgrag.memory_graph import MemoryGraph
from typing import Any
from langchain_ollama import OllamaEmbeddings
from langgraph.store.base import IndexConfig


class MemoryOllama(MemoryGraph):
    """
    Memory agent for Ollama embeddings.
    Args:
        **kwargs: Arbitrary keyword arguments for configuration.
        model_embedding_name (str): The name of the model to use
            for embeddings.
        model_embedding_url (str): The base URL for the Ollama server.
    Methods:
        get_embedding_model: Initializes the embedding model.
        memory_config: Returns the memory configuration.
    """

    model_embedding: OllamaEmbeddings
    model_embedding_name: str | None = None
    model_embedding_url: str | None = None

    def __init__(self, **kwargs: Any) -> None:
        """
        Initializes an instance of MemoryAgent with the provided parameters.
        Args:
            **kwargs: Arbitrary keyword arguments for configuration.
            model_embedding_name (str): The name of the model to use
                for embeddings.
            model_embedding_url (str): The base URL for the Ollama server.
        Raises:
            ValueError: If the model_embedding_url is not set.
        """
        super().__init__(**kwargs)

        self.model_embedding_config["name"] = "nomic-embed-text"
        self.model_embedding_config["url"] = kwargs.get(
            "model_embedding_url",
            "http://localhost:11434"
        )

        if self.model_embedding_config["name"] is None:
            msg = "model_embedding_name not set"
            self.logger.error(msg)
            raise ValueError(msg)

        if self.model_embedding_config["url"] is None:
            msg = (
                (
                    "model_embedding_url not set, "
                    "using default Ollama base URL"
                )
            )
            self.logger.error(msg)
            raise ValueError(msg)

        self.get_embedding_model()

    def get_embedding_model(self) -> None:
        """
        Get the language model_embedding_name to use for generating text.

        Returns:
            None: sets self.model_embedding with the chosen embedding model.
        Raises:
            ValueError: If the model_embedding_type or
                model_embedding_name is not set.
            Exception: If there is an error during the loading
                of the embedding model.
        """
        try:
            self.logger.info("Using Ollama embeddings")
            # strip trailing slash and append path
            self.model_embedding = OllamaEmbeddings(
                model=str(self.model_embedding_config["name"]),
                base_url=self.model_embedding_config["url"]
            )
        except Exception as e:
            msg = (
                f"Errore durante il caricamento del modello di embedding: {e}"
            )
            self.logger.error(msg)
            raise e

    def memory_config(self) -> IndexConfig:
        """
        Get the memory configuration for the agent.

        Returns:
            IndexConfig: The memory configuration.
        """
        collection_dim = self._get_collection_dim()
        return {
            "embed": self.model_embedding,
            "dims": collection_dim,
        }
