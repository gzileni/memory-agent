import uuid
import os
from abc import abstractmethod
from typing import Any, Optional
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import IndexConfig
from memory_agent import get_logger
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableConfig


class MemoryStore:
    """
    Class representing an agent that uses a memory store to manage
    input and output.
    """

    thread_id: str = str(uuid.uuid4())
    user_id: str = "*"
    session_id: str = "*"
    logger = get_logger(
        name="memory_agent",
        loki_url=os.getenv("LOKI_URL")
    )
    llm_config: dict[str, Any] = {
        "model": None,
        "model_provider": None,
        "api_key": None,
        "base_url": None,
        "temperature": None,
    }

    llm_model: BaseChatModel
    max_recursion_limit: int = 25
    TEMPERATURE_DEFAULT: float = 0.5

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
        self.user_id = kwargs.get("user_id", self.user_id)
        self.session_id = kwargs.get("session_id", self.session_id)
        self.max_recursion_limit = kwargs.get(
            "max_recursion_limit",
            self.max_recursion_limit
        )
        self.llm_config = kwargs.get("llm_config", self.llm_config)
        self.llm_model = self._create_model(**self.llm_config)
        os.environ["NEO4J_AUTH"] = "none"

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

    def _create_model(
        self,
        **model_config
    ) -> BaseChatModel:
        """
        Get the chat model for the agent.
        Args:
            **model_config: The configuration for the model.
        Returns:
            BaseChatModel: The chat model for the agent.
        """
        return init_chat_model(**model_config)

    def _params(
        self,
        thread_id: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        **kwargs
    ):
        """
        Prepares the configuration and input data for the agent
        based on the provided prompt and thread ID.
        Args:
            prompt (str): The user input prompt to be processed by the agent.
            thread_id (str): A unique identifier for the thread,
            used for tracking and logging.
        Returns:
            tuple: A tuple containing the configuration for the agent
            and the input data structured for processing.
        """

        max_recursion_limit = kwargs.get(
            "max_recursion_limit",
            self.max_recursion_limit
        )

        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "recursion_limit": max_recursion_limit,
            }
        }

        if user_id:
            config["configurable"]["user_id"] = user_id

        if session_id:
            config["configurable"]["session_id"] = session_id

        return config
