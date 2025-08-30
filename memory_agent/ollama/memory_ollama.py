from memory_agent.memory_persistence import MemoryPersistence
from typing import Any
from langchain_ollama import OllamaEmbeddings
from langgraph.store.base import IndexConfig


class MemoryOllama(MemoryPersistence):

    model_embedding: OllamaEmbeddings
    model_embedding_name: str | None = None
    model_embedding_url: str | None = None

    def __init__(self, **kwargs: Any) -> None:
        """
        Initializes an instance of MemoryAgent with the provided parameters.
        Args:
            model_embedding_name (str): The name of the model to use
                for embeddings.
            model_embedding_url (str): The URL of the model to use
                for embeddings.
        """
        super().__init__(**kwargs)

        self.model_embedding_name = kwargs.get(
            "model_embedding_name",
            "nomic-embed-text"
        )
        self.model_embedding_url = kwargs.get(
            "model_embedding_url",
            None
        )
        if self.model_embedding_url is None:
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
            base_url = str(self.model_embedding_url).rstrip("/")
            self.model_embedding_url = f"{base_url}/api/embeddings"
            self.model_embedding = OllamaEmbeddings(
                model=str(self.model_embedding_name),
                base_url=self.model_embedding_url
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
        return {
            "embed": self.model_embedding,
            "dims": self.collection_dim,
        }
