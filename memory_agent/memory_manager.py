import uuid
import os
from langmem import (
    create_memory_store_manager,
    ReflectionExecutor
)
from memory_schemas import Episode, UserProfile, Triple
from typing import Literal
from .memory_log import get_logger
from abc import abstractmethod
from langgraph.store.memory import InMemoryStore
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.config import get_store
from typing import Any
from langchain_core.runnables import RunnableConfig

MemoryStoreType = Literal["episodic", "user", "semantic"]
MemoryActionType = Literal["hotpath", "background"]


class MemoryManager:
    """
    A manager for handling memory operations within the MemoryAgent.
    """

    namespace: tuple
    thread_id: str = str(uuid.uuid4())
    user_id: str = "*"
    session_id: str = "*"
    logger = get_logger(
        name="memory_store",
        loki_url=os.getenv("LOKI_URL")
    )
    refresh_checkpointer: bool = True
    filter_minutes: int = 60
    action_type: MemoryActionType = "hotpath"
    store_type: MemoryStoreType = "semantic"

    llm_config: dict[str, Any] = {
        "model": "llama3.1",
        "model_provider": "ollama",
        "api_key": None,
        "base_url": "http://localhost:11434",
        "temperature": 0.7,
    }

    llm_model: BaseChatModel
    host_persistence_config: dict[str, Any] = {
        "host": "localhost",
        "port": 6379,
        "db": 0,
    }

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
            filter_minutes (int): The number of minutes to filter memories.
                Default:
                    60
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
        self.action_type = kwargs.get("action_type", self.action_type)
        self.store_type = kwargs.get("store_type", self.store_type)
        self.filter_minutes = kwargs.get("filter_minutes", self.filter_minutes)
        self.llm_config = kwargs.get("llm_config", self.llm_config)
        self.llm_model = self._create_model(**self.llm_config)

        self.host_persistence_config = kwargs.get(
            "host_persistence_config",
            self.host_persistence_config
        )

        self.host_persistence_config = {
            "host": os.getenv(
                "REDIS_HOST",
                self.host_persistence_config.get("host", "localhost"),
            ),
            "port": os.getenv(
                "REDIS_PORT",
                self.host_persistence_config.get("port", 6379),
            ),
            "db": os.getenv(
                "REDIS_DB",
                self.host_persistence_config.get("db", 0),
            ),
        }
        msg = "Host persistence config initialized: %s"
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
            *state["messages"],
        ]

    def create_memory(
        self,
        model,
        instructions: str,
        schemas: list,
        namespace: tuple,
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
            store=self.store(),
            **kwargs,
        )

    @abstractmethod
    def store(self) -> InMemoryStore:
        """
        Get the in-memory store for the agent.
        Returns:
            InMemoryStore: The in-memory store for the agent.
        """
        pass

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
        store = get_store()
        return store.search(
            # Search within the same namespace as the one
            # we've configured for the agent
            self.namespace,
            query=state["messages"][-1].content,
        )

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

    async def update_memory(
        self,
        messages,
        config: RunnableConfig,
        store_type: MemoryStoreType = "semantic",
        delay: int = 10,
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
        allowed_store_types = ("episodic", "user", "semantic")
        if store_type not in allowed_store_types:
            raise ValueError(
                f"store_type must be one of {allowed_store_types}"
            )

        # Determine namespace based on store_type
        if store_type == "episodic":
            instructions = (
                "Extract examples of successful explanations, "
                "capturing the full chain of reasoning. "
                "Be concise in your explanations and precise in the "
                "logic of your reasoning."
            )
            schemas = [Episode]
        elif store_type == "user":
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
            **kwargs
        )

        input = {"messages": messages}

        if self.action_type == "background":
            executor = ReflectionExecutor(
                mem,
                store=self.store()
            )

            return executor.submit(
                input,
                after_seconds=delay,
                config=config
            )
        else:
            return await mem.ainvoke(
                input,  # type: ignore
                config=config
            )
