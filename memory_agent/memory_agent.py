import uuid
import os
from typing import AsyncIterable, Any, Optional, Literal
from langgraph.prebuilt import create_react_agent
from langchain_core.runnables import RunnableConfig
from langmem.short_term import SummarizationNode
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.graph.state import CompiledStateGraph
from .memory_checkpointer import MemoryCheckpointer
from langgraph.config import get_store
from .memory_log import get_logger, get_metadata
from langmem import (
    create_manage_memory_tool,
    create_search_memory_tool,
    create_memory_store_manager,
    ReflectionExecutor
)
from abc import abstractmethod
from langgraph.store.memory import InMemoryStore
from .state import State
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import SecretStr

MemoryAgentType = Literal["hotpath", "background"]


class MemoryAgent:
    """
    A memory agent for managing and utilizing memory in AI applications.
    Args:
        **kwargs: Arbitrary keyword arguments for configuration.
            - thread_id (str): Unique identifier for the thread.
            - host_persistence_config (dict): Configuration for
                host persistence.
            - max_recursion_limit (int): Maximum recursion limit for the agent.
            - filter_minutes (int): Minutes to filter old checkpoints.
            - model_name (str): Name of the language model to use.
            - model_provider (str): Provider of the language model.
            - base_url (str, optional): Base URL for the model API.
            - temperature (float): Temperature setting for the model.
            - tools (list): List of tools available for the agent.
            - refresh_checkpointer (bool): Whether to refresh the checkpointer.
            - agent (CompiledStateGraph): The agent instance.
    """
    thread_id: str = str(uuid.uuid4())
    host_persistence_config: dict[str, str | int] = {}
    logger = get_logger(
        name="memory_store",
        loki_url=os.getenv("LOKI_URL")
    )
    max_recursion_limit: int = 25
    summarize_node: SummarizationNode
    model_name: str
    model_provider: str
    base_url: Optional[str] = None
    temperature: float = 0.7
    tools: list = []
    agent: Optional[CompiledStateGraph] = None
    refresh_checkpointer: bool = True
    filter_minutes: int = 15
    memory_agent_type: MemoryAgentType = "hotpath"
    llm_api_key: SecretStr | None = None

    def __init__(self, **kwargs):
        """
        Initialize the MemoryAgent with the given parameters.
        Args:
            **kwargs: Arbitrary keyword arguments for configuration.
                - thread_id (str): Unique identifier for the thread.
                - host_persistence_config (dict): Configuration
                    for host persistence.
                - max_recursion_limit (int): Maximum recursion limit
                    for the agent.
                - filter_minutes (int): Minutes to filter old checkpoints.
                - model_name (str): Name of the language model to use.
                - model_provider (str): Provider of the language model.
                - base_url (str, optional): Base URL for the model API.
                - temperature (float): Temperature setting for the model.
                - tools (list): List of tools available for the agent.
                - refresh_checkpointer (bool): Whether to refresh
                    the checkpointer.
                - agent (CompiledStateGraph): The agent instance.
                - llm_api_key (SecretStr | None): The API key for the
                    language model.
        """
        self.thread_id = kwargs.get("thread_id", self.thread_id)
        msg = "Initializing MemoryAgent with thread_id: %s"
        self.logger.info(msg, self.thread_id)
        self.memory_agent_type = kwargs.get(
            "memory_agent_type",
            self.memory_agent_type
        )
        if self.memory_agent_type not in ("hotpath", "background"):
            msg: str = (
                f"Invalid memory_agent_type: {self.memory_agent_type}"
                f" (must be one of: {['hotpath', 'background']})"
            )
            raise ValueError(msg)
        else:
            self.logger.info(
                "Memory Type: %s",
                self.memory_agent_type
            )

        self.host_persistence_config = kwargs.get(
            "host_persistence_config",
            self.host_persistence_config
        )

        if self.host_persistence_config == {}:
            self.host_persistence_config = {
                "host": os.getenv("REDIS_HOST", "localhost"),
                "port": os.getenv("REDIS_PORT", 6379),
                "db": os.getenv("REDIS_DB", 0),
            }
            msg = "Host persistence config initialized: %s"
            self.logger.info(msg, self.host_persistence_config)

        self.max_recursion_limit = kwargs.get(
            "max_recursion_limit",
            self.max_recursion_limit
        )

        self.summarize_node = SummarizationNode(
            token_counter=count_tokens_approximately,
            model=self.model(),
            max_tokens=384,
            max_summary_tokens=128,
            output_messages_key="llm_input_messages",
        )

        self.filter_minutes = kwargs.get(
            "filter_minutes", self.filter_minutes
        )

        self.agent = kwargs.get("agent", self.agent)
        self.refresh_checkpointer = kwargs.get(
            "refresh_checkpointer",
            self.refresh_checkpointer
        )

        api_key: str | None = kwargs.get(
            "llm_api_key",
            None
        )

        self.llm_api_key = (
            SecretStr(api_key)
            if api_key is not None
            else None
        )

    def _memory_manager(self):
        """
        Create a memory manager for the agent.
        """
        return create_memory_store_manager(
            self.model(),
            # Store memories in the "memories" namespace (aka directory)
            namespace=("memories",),
            store=self.store()
        )

    def _executor(self):
        """
        Create an executor for the agent.
        """
        return ReflectionExecutor(
            self._memory_manager(),
            store=self.store()
        )

    def _save_background_messages(
        self,
        config: RunnableConfig,
        prompt: str,
        response,
        delay: int = 10
    ):
        """
        Save background messages for later processing.
        Args:
            config (RunnableConfig): The configuration for the runnable.
            prompt (str): The user prompt to save.
            response: The response from the agent.
            delay (int): Delay before processing the messages.
        """
        # build message list separately to keep line length within limits
        # use HumanMessage so the list is List[BaseMessage]
        messages = [{"role": "user", "content": prompt}, response]
        to_process = {"messages": messages}
        # depending on app context.
        self._executor().submit(
            to_process,
            after_seconds=delay,
            config=config
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
        store = get_store()
        memories = store.search(
            # Search within the same namespace as the one
            # we've configured for the agent
            ("memories",),
            query=state["messages"][-1].content,
        )
        system_msg = f"""You are a helpful assistant.

        ## Memories
        <memories>
        {memories}
        </memories>
        """
        return [{"role": "system", "content": system_msg}, *state["messages"]]

    @abstractmethod
    def store(self) -> InMemoryStore:
        """
        Get the in-memory store for the agent.
        Returns:
            InMemoryStore: The in-memory store for the agent.
        """
        pass

    def model(
        self,
        **kwargs
    ) -> BaseChatModel:
        """
        Get the chat model for the agent.
        Returns:
            BaseChatModel: The chat model for the agent.
        """
        return init_chat_model(
            self.model_name,
            model_provider=self.model_provider,
            temperature=self.temperature,
            base_url=self.base_url,
            api_key=self.llm_api_key,
            **kwargs
        )

    async def create_agent(
        self,
        checkpointer,
        **kwargs_model
    ) -> CompiledStateGraph:
        """
        Create the agent's state graph.
        Args:
            checkpointer: The checkpointer instance to use for managing state.
        Returns:
            CompiledStateGraph: The compiled state graph for the agent.
        """
        return create_react_agent(
            model=self.model(**kwargs_model),
            prompt=self._prompt,
            tools=await self._get_tools(),
            store=self.store(),
            state_schema=State,
            pre_model_hook=self.summarize_node,
            checkpointer=checkpointer
        )

    async def _get_tools(self):
        """
        Get the tools available for the agent.
        Returns:
            list: A list of tools available for the agent.
        """

        self.tools.extend([
            create_manage_memory_tool(namespace=("memories",))
        ])

        if self.memory_agent_type == "background":
            self.tools.extend([
                create_search_memory_tool(namespace=("memories",))
            ])

        return self.tools

    def _params(self, prompt, thread_id):
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
        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "recursion_limit": self.max_recursion_limit,
            }
        }

        input_data = {"messages": [{"role": "user", "content": prompt}]}
        return config, input_data

    def _response(
        self,
        thread_id: str,
        result: dict | None,
        error: dict | None
    ) -> dict:
        """
        Get the agent's response based on the current state.
        Args:
            thread_id (str): The ID of the thread.
            result (dict | None): The result of the agent's processing.
            error (dict | None): Any error that occurred during processing.
        Returns:
            dict: The response to be sent back to the client.
        """

        response: dict = {
            "jsonrpc": "2.0",
            "id": thread_id
        }

        if result is None and error is None:
            raise ValueError("Both result and error are None")

        if result is not None:
            response["result"] = result

        if error is not None:
            response["error"] = error

        return response

    async def _refresh(self, checkpointer):
        # Delete checkpoints older than 15 minutes
        # for the current thread
        if self.refresh_checkpointer:
            await checkpointer.adelete_by_thread_id(
                thread_id=self.thread_id,
                filter_minutes=self.filter_minutes
            )

    async def ainvoke(
        self,
        prompt: str,
        thread_id: Optional[str] = None,
        **kwargs_model
    ):
        """
        Asynchronously runs the agent with the given prompt.

        Args:
            prompt (str): The user input prompt to be processed by the agent.
            tools (list): A list of tools available for the agent to use.
            thread_id (str): A unique identifier for the thread,
                used for tracking and logging.
        """
        try:

            if thread_id is not None:
                self.thread_id = thread_id

            config, input_data = self._params(
                prompt,
                self.thread_id
            )

            result: dict = {
                'is_task_complete': False,
                'require_user_input': True,
                'content': (
                    'We are unable to process your request at the moment. '
                    'Please try again.'
                )
            }

            async with MemoryCheckpointer.from_conn_info(
                host=str(self.host_persistence_config["host"]),
                port=int(self.host_persistence_config["port"]),
                db=int(self.host_persistence_config["db"])
            ) as checkpointer:

                await self._refresh(checkpointer)

                if self.agent is None:
                    self.logger.info("Creating new default agent")
                    self.agent = await self.create_agent(
                        checkpointer,
                        **kwargs_model
                    )
                else:
                    self.logger.info("Using existing agent")
                    self.agent.checkpointer = checkpointer

                response_agent = await self.agent.ainvoke(
                    input=input_data,
                    config=config,
                    stream_mode="updates"
                )

                if (
                    "messages" in response_agent
                    and len(response_agent["messages"]) > 0
                ):
                    event_messages = response_agent["messages"]
                    event_response = event_messages[-1].content
                    # If there are messages from the agent,
                    # return the last message
                    self.logger.info(
                        (
                            f">>> Response event from agent: "
                            f"{event_response}"
                        ),
                        extra=get_metadata(thread_id=self.thread_id)
                    )

                    result["content"] = event_response

                if self.memory_agent_type == "background":
                    self._save_background_messages(
                        config,
                        prompt,
                        response_agent
                    )

                return self._response(
                    thread_id=self.thread_id,
                    result=result,
                    error=None
                )
        except Exception as e:
            self.logger.error(
                f"Error occurred while invoking agent: {e}",
                extra=get_metadata(thread_id=self.thread_id)
            )
            return self._response(
                thread_id=self.thread_id,
                result=None,
                error={"message": str(e)}
            )

    async def stream(
        self,
        prompt: str,
        thread_id: Optional[str] = None,
        **kwargs_model
    ) -> AsyncIterable[dict[str, Any]]:
        """
        Asynchronously streams response chunks from the agent based
        on the provided prompt.

        Args:
            prompt (str): The user input prompt to be processed by the agent.
            thread_id (str, optional): A unique identifier for the thread,
                used for tracking and logging. If not provided, a new
                thread ID will be generated.
            **kwargs_model: Additional keyword arguments for the model.
        """

        if thread_id is not None:
            self.thread_id = thread_id

        result: dict = {
            'is_task_complete': False,
            'require_user_input': True,
            'content': (
                'We are unable to process your request at the moment. '
                'Please try again.'
            )
        }

        try:

            config, input_data = self._params(
                prompt,
                self.thread_id
            )

            async with MemoryCheckpointer.from_conn_info(
                    host=str(self.host_persistence_config["host"]),
                    port=int(self.host_persistence_config["port"]),
                    db=int(self.host_persistence_config["db"])
            ) as checkpointer:

                # Delete checkpoints older than 15 minutes
                # for the current thread
                await self._refresh(checkpointer)

                if self.agent is None:
                    self.logger.info("Creating new default agent")
                    self.agent = await self.create_agent(
                        checkpointer,
                        **kwargs_model
                    )
                else:
                    self.logger.info("Using existing agent")
                    self.agent.checkpointer = checkpointer

                index: int = 1
                async for event in self.agent.astream(
                    input=input_data,
                    config=config,
                    stream_mode="updates"
                ):
                    event_index: str = f"Event {index}"
                    self.logger.debug(
                        f">>> {event_index} received: {event}",
                        extra=get_metadata(thread_id=self.thread_id)
                    )
                    event_item = None

                    if "agent" in event:
                        event_item = event["agent"]
                        agent_process: str = (
                            f'{event_index} - Looking up the knowledge base...'
                        )
                        self.logger.debug(
                            agent_process,
                            extra=get_metadata(thread_id=self.thread_id)
                        )
                        result = {
                            'is_task_complete': True,
                            'require_user_input': False,
                            'content': agent_process,
                        }
                        yield self._response(
                            thread_id=self.thread_id,
                            result=result,
                            error=None
                        )

                    elif "tools" in event:
                        event_item = event["tools"]
                        tool_process: str = (
                            f'{event_index} - Processing the knowledge base...'
                        )
                        self.logger.debug(
                            tool_process,
                            extra=get_metadata(thread_id=self.thread_id)
                        )
                        result["content"] = {
                            'is_task_complete': False,
                            'require_user_input': False,
                            'content': tool_process,
                        }
                        yield self._response(
                            thread_id=self.thread_id,
                            result=result,
                            error=None
                        )

                    if event_item is not None:
                        if (
                            "messages" in event_item
                            and len(event_item["messages"]) > 0
                        ):
                            event_messages = event_item["messages"]
                            event_response = event_messages[-1].content
                            # If there are messages from the agent, return
                            # the last message
                            self.logger.info(
                                (
                                    f">>> Response event from agent: "
                                    f"{event_response}"
                                ),
                                extra=get_metadata(thread_id=self.thread_id)
                            )
                            if (
                                event_response
                                or (len(event_response) > 0)
                            ):
                                result["content"] = {
                                    'is_task_complete': True,
                                    'require_user_input': False,
                                    'content': event_response,
                                }

                                yield self._response(
                                    thread_id=self.thread_id,
                                    result=result,
                                    error=None
                                )

                                if self.memory_agent_type == "background":
                                    self._save_background_messages(
                                        config,
                                        prompt,
                                        event_response
                                    )
                    index += 1

        except Exception as e:
            # In caso di errore, restituisce un messaggio di errore
            self.logger.error(
                f"Error occurred while processing event: {str(e)}",
                extra=get_metadata(thread_id=self.thread_id)
            )
            yield self._response(
                thread_id=self.thread_id,
                result=result,
                error={
                    "code": 500,
                    "message": str(e)
                }
            )
            raise e
