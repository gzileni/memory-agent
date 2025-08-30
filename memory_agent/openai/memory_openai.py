import os
from memory_agent.memory_persistence import MemoryPersistence
from typing import Any
from langchain_openai import OpenAIEmbeddings
from langgraph.store.base import IndexConfig
from pydantic import SecretStr


class MemoryOpenAI(MemoryPersistence):

    model_embedding: OpenAIEmbeddings
    model_embedding_name: str = "text-embedding-3-small"
    openai_api_key: str | None = None

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.openai_api_key = kwargs.get(
            "openai_api_key",
            os.getenv("OPENAI_API_KEY")
        )
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

            if self.model_embedding_name is None:
                raise ValueError("model_embedding_name must be set")

            api_key = (
                SecretStr(self.openai_api_key)
                if self.openai_api_key is not None
                else None
            )
            self.model_embedding = OpenAIEmbeddings(
                model=self.model_embedding_name,
                dimensions=self.collection_dim,
                api_key=api_key,
            )
        except Exception as e:
            msg = (
                f"Errore durante il caricamento del modello di embedding: {e}"
            )
            self.logger.error(msg)
            raise e

    def get_config(self) -> IndexConfig:
        return {
            "embed": self.model_embedding,
            "dims": self.collection_dim,
        }
