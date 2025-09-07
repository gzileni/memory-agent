import uuid
import os
from langmem import (
    create_memory_store_manager
)
from .memory_schemas import Episode, UserProfile, Triple
from typing import Literal
from memory_agent import (
    get_logger
)
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from typing import Any
from langchain_core.runnables import RunnableConfig
from abc import abstractmethod
from langgraph.store.base import BaseStore

MemoryStoreType = Literal["episodic", "user", "semantic"]


class MemoryManager:
    """
    A manager for handling memory operations within the MemoryAgent.
    Args:
        thread_id (str): The ID of the thread.
        user_id (str): The ID of the user.
            Default:
                "*"
        session_id (str): The ID of the session.
            Default:
                "*"
        action_type (MemoryActionType): The type of action to perform.
            Default:
                "hotpath"
            Values:
                "hotpath"
                "background"
        store_type (MemoryStoreType): The type of memory store to use.
            Default:
                "semantic"
            Values:
                "episodic"
                "user"
                "semantic"
        host_persistence_config (dict[str, Any]): The configuration
            for host persistence.
            Default:
                {
                    "host": "localhost",
                    "port": 6379,
                    "db": 0
                }
        llm_config (dict[str, Any]): The configuration for the LLM.
            Default:
                {
                    "model": "llama3.1",
                    "model_provider": "ollama",
                    "api_key": None,
                    "base_url": "http://localhost:11434",
                    "temperature": 0.7,
                }
    Methods:
        store(): Get the in-memory store for the agent.
        update_memory(): Update memory with new conversation data.
    Attributes:
        namespace (tuple): The namespace for the memory store.
        thread_id (str): The ID of the thread.
        user_id (str): The ID of the user.
        session_id (str): The ID of the session.
        logger: The logger for the memory manager.
        action_type (MemoryActionType): The type of action to perform.
        store_type (MemoryStoreType): The type of memory store to use.
        llm_config (dict[str, Any]): The configuration for the LLM.
        llm_model: The chat model for the agent.
        host_persistence_config (dict[str, Any]): The configuration
            for host persistence.
    """

    TEMPERATURE_DEFAULT = 0.7
    namespace: tuple
    thread_id: str = str(uuid.uuid4())
    user_id: str = "*"
    session_id: str = "*"
    logger = get_logger(
        name="memory_store",
        loki_url=os.getenv("LOKI_URL")
    )
    store_type: MemoryStoreType = "semantic"

    llm_config: dict[str, Any] = {
        "model": None,
        "model_provider": None,
        "api_key": None,
        "base_url": None,
        "temperature": None,
    }

    llm_model: BaseChatModel
    host_persistence_config: dict[str, Any] = {
        "host": "localhost",
        "port": 6379,
        "db": 0,
        "decode_responses": True
    }
    vector_store: BaseStore

    def __init__(self, **kwargs):
        """
        Initialize the MemoryManager with the given parameters.
        Args:
            thread_id (str): The ID of the thread.
            user_id (str): The ID of the user.
            session_id (str): The ID of the session.
            action_type (MemoryActionType): The type of action to perform.
                Default:
                    "hotpath"
                Values:
                    "hotpath"
                    "background"
            store_type (MemoryStoreType): The type of memory store to use.
                Default:
                    "semantic"
                Values:
                    "episodic"
                    "user"
                    "semantic"
            host_persistence_config (dict[str, Any]): The configuration
                for host persistence.
                Default:
                    {
                        "host": "localhost",
                        "port": 6379,
                        "db": 0
                    }
            llm_config (dict[str, Any]): The configuration for the LLM.
                Default:
                    {
                        "model": "llama3.1",
                        "model_provider": "ollama",
                        "api_key": None,
                        "base_url": "http://localhost:11434",
                        "temperature": 0.7,
                    }
        """
        self.thread_id = kwargs.get("thread_id", self.thread_id)
        self.user_id = kwargs.get("user_id", self.user_id)
        self.session_id = kwargs.get("session_id", self.session_id)
        self.store_type = kwargs.get("store_type", self.store_type)
        self.llm_config = kwargs.get("llm_config", self.llm_config)
        self.llm_model = self._create_model(**self.llm_config)

        self.host_persistence_config = kwargs.get(
            "host_persistence_config",
            self.host_persistence_config
        )
        msg = "Redis config initialized: %s"
        self.logger.info(msg, self.host_persistence_config)

        msg = (
            "Initializing MemoryAgent with thread_id: %s, "
            "user_id: %s, "
            "session_id: %s"
        )
        self.logger.info(
            msg,
            self.thread_id,
            self.user_id,
            self.session_id
        )
        self.namespace = (
            "memories",
            self.thread_id,
            self.user_id,
            self.session_id,
        )

    @abstractmethod
    def index_store(self) -> Any:
        """
        Get the index store configuration.
        Returns:
            Any: The index store configuration.
        """
        pass

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """
        Embed a query text into a vector.
        Args:
            text (str): The text to embed.
        Returns:
            list[float]: The embedded vector.
        """
        pass

    def _convert_namespace(self) -> str:
        """
        Convert the namespace tuple to a string.
        Returns:
            str: The converted namespace string.
        """
        return ":".join(self.namespace)

    def _redis_uri_store(self) -> str:
        """
        Create a Redis URI from the host persistence configuration.
        Returns:
            str: The Redis URI.
        """
        host = self.host_persistence_config["host"]
        port = self.host_persistence_config["port"]
        # db = self.host_persistence_config["db"]
        return f"redis://{host}:{port}"

    def _prompt(self, state):
        """
        Prepare the messages for the LLM.
        Args:
            state (dict): The current state of the agent.
        Returns:
            list: The prepared messages for the LLM.
        """
        # Get store from configured contextvar;
        # Same as that provided to `create_react_agent`

        memories = self._get_similar(state)
        system_message = "You are a helpful assistant."

        if memories:
            if self.store_type == "episodic":
                system_message += "\n\n### EPISODIC MEMORY:"
                for i, item in enumerate(memories, start=1):
                    if item is not None:
                        episode = item.value["content"]
                        system_message += f"""
                        Episode {i}:
                        When: {episode['observation']}
                        Thought: {episode['thoughts']}
                        Did: {episode['action']}
                        Result: {episode['result']}
                        """
            elif self.store_type == "user":
                system_message += f"""<User Profile>:
                {memories[0].value}
                </User Profile>
                """
            else:
                system_message += f"""

                ## Memories
                <memories>
                {memories}
                </memories>
                """

        return [
            {"role": "system", "content": system_message},
            *state["messages"]
        ]

    def create_memory(
        self,
        model,
        instructions: str,
        schemas: list,
        namespace: tuple,
        store: Any,
        **kwargs
    ):
        """
        Create episodic memory.
        Args:
            **kwargs_model: Additional keyword arguments for the memory model.
        Return:
            A MemoryManager instance for episodic memory.
        """

        return create_memory_store_manager(
            model,
            schemas=schemas,
            instructions=instructions,
            namespace=namespace,
            store=store,
            **kwargs,
        )

    def _get_similar(
        self,
        state
    ):
        """
        Find similar past episodes in the store.
        Args:
            store: The memory store to search.
            messages: The list of messages to find similarities.
            namespace: The namespace to use for the search.
        """
        try:
            query: str = state["messages"][-1].content
            if self.vector_store is None:
                raise ValueError("Vector store is not initialized.")
            similar = self.vector_store.search(
                self.namespace,
                query=query
            )
            return similar
        except Exception as e:
            self.logger.error("Error searching for similar memories: %s", e)
            raise e

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

    def update_memory(
        self,
        messages,
        config: RunnableConfig,
        **kwargs
    ):
        """
        Update memory with new conversation data.
        Args:
            messages: A list of messages to update memory with.
            config: The configuration for the update.
            store_type: The type of memory store to use.
            delay: The delay in seconds before updating memory.
            **kwargs: Additional keyword arguments.
        Return:
            A list of updated Episode objects.
        """

        # validate store_type against allowed values
        try:
            allowed_store_types = ("episodic", "user", "semantic")
            if self.store_type not in allowed_store_types:
                raise ValueError(
                    f"store_type must be one of {allowed_store_types}"
                )

            # Determine namespace based on store_type
            if self.store_type == "episodic":
                instructions = (
                    "Extract examples of successful explanations, "
                    "capturing the full chain of reasoning. "
                    "Be concise in your explanations and precise in the "
                    "logic of your reasoning."
                )
                schemas = [Episode]
            elif self.store_type == "user":
                instructions = (
                    "Extract user profile information"
                )
                schemas = [UserProfile]
            else:  # semantic
                instructions = (
                    "Extract user preferences and any other useful information"
                )
                schemas = [Triple]

            mem = self.create_memory(
                self.llm_model,
                instructions=instructions,
                schemas=schemas,
                namespace=self.namespace,
                store=self.vector_store,
                **kwargs
            )

            if messages is None or len(messages) == 0:
                raise ValueError("Messages cannot be None or empty.")

            input: Any = {"messages": messages}

            return mem.invoke(
                input,
                config=config
            )
        except Exception as e:
            self.logger.error("Error updating memory: %s", e)
            raise e
