import uuid
import os
from abc import abstractmethod
from typing import Any
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import IndexConfig
from memory_agent import get_logger


class MemoryStore:
    """
    Class representing an agent that uses a memory store to manage
    input and output.
    """

    thread_id: str | None = str(uuid.uuid4())
    logger = get_logger(
        name="memory_store",
        loki_url=os.getenv("LOKI_URL")
    )

    def __init__(self, **kwargs: Any) -> None:
        """
        Initializes an instance of MemoryAgent with the provided parameters.

        Args:
            *args (Any): Positional arguments. If provided, the first
                argument represents the LLM model and the second represents
                the embedding model_embedding_name.
            **kwargs (Any): Optional arguments such as:
                - model_embedding_type (str): The type of language model
                    to use (e.g., 'openai', 'ollama', 'vllm', 'custom').
                    Default value: 'openai'.
                - model_embedding_name (str): The embedding model to use.
                    If not specified, the 'MODEL_EMBEDDING' environment
                    variable is used. Default value: 'text-embedding-3-small'.
                - model_embedding_url (str, optional):
                    The base URL for the model.
                - collection_name (str, optional): The name of the collection.
                    Default value: 'memory_store'.
                - collection_dim (int, optional):
                    The dimension of the collection. Default value: 1536.
                - model_embedding_path (str, optional):
                    The path to the model embedding file.
        """
        self.thread_id = kwargs.get("thread_id", self.thread_id)

    @abstractmethod
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
        pass

    @abstractmethod
    def memory_config(self) -> IndexConfig:
        """
        Get the configuration for the in-memory store.

        Returns:
            IndexConfig: The configuration for the in-memory store.
        """
        pass

    def in_memory_store(self) -> InMemoryStore:
        """
        Get the in-memory store.

        Returns:
            InMemoryStore: The in-memory store.
        """
        return InMemoryStore(index=self.memory_config())
