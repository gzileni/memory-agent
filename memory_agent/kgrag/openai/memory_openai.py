import os
from memory_agent.kgrag.memory_graph import MemoryGraph
from typing import Any
from langchain_openai import OpenAIEmbeddings
from langgraph.store.base import IndexConfig
from pydantic import SecretStr


class MemoryOpenAI(MemoryGraph):
    """
    Memory agent for OpenAI embeddings.
    Args:
        **kwargs: Arbitrary keyword arguments for configuration.
        model_embedding_name (str): The name of the model to use
            for embeddings.
        llm_api_key (str): The API key for the language model.
    Methods:
        get_embedding_model: Initializes the embedding model.
        memory_config: Returns the memory configuration.
    """
    model_embedding: OpenAIEmbeddings
    llm_api_key: SecretStr | None = None

    def __init__(self, **kwargs: Any) -> None:
        """
        Initializes an instance of MemoryOpenAI with the provided parameters.
        Args:
            model_embedding_name (str): The name of the model to use
                for embeddings.
            llm_api_key (str): The API key for the language model.
        Raises:
            ValueError: If the llm_api_key is not set.
        """
        super().__init__(**kwargs)

        self.model_embedding_config["name"] = "text-embedding-3-small"

        api_key = kwargs.get(
            "llm_api_key",
            os.getenv("OPENAI_API_KEY")
        )

        self.llm_api_key = (
            SecretStr(api_key)
            if api_key is not None
            else None
        )

        if self.llm_api_key is None:
            raise ValueError("OPENAI_API_KEY must be set")

        self.model_embedding_name = kwargs.get(
            "model_embedding_name",
            self.model_embedding_name
        )
        self.get_embedding_model()

    def get_embedding_model(self):
        """
        Get the language model_embedding_name to use for generating text.

        Returns:
            Any: The language model_embedding_name to use.
        Raises:
            ValueError: If the model_embedding_type or
                model_embedding_name is not set.
            Exception: If there is an error during the loading
                of the embedding model.
        """
        try:

            if self.model_embedding_config["name"] is None:
                raise ValueError("model_embedding_name must be set")

            collection_dim = self._get_collection_dim()

            self.model_embedding = OpenAIEmbeddings(
                model=self.model_embedding_config["name"],
                dimensions=collection_dim,
                api_key=self.llm_api_key,
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
