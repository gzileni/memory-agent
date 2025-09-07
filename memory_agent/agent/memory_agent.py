import os
from typing import AsyncIterable, Any, Optional
from langgraph.prebuilt import create_react_agent
from langchain_core.runnables import RunnableConfig
from langmem.short_term import SummarizationNode
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.graph.state import CompiledStateGraph
from ..memory_log import get_metadata
from langmem import (
    create_manage_memory_tool,
    create_search_memory_tool,
)
from .memory_manager import MemoryManager
from .state import State
from langgraph.store.redis import RedisStore
from langgraph.checkpoint.redis import RedisSaver


class MemoryAgent(MemoryManager):
    """
    A memory agent for managing and utilizing memory in AI applications.
    Args:
        max_recursion_limit (int): Maximum recursion depth for the agent.
        summarize_node (SummarizationNode): Node for summarizing conversations.
        tools (list): A list of tools available for the agent.
        agent (CompiledStateGraph | None): Predefined agent state graph.
        max_tokens (int): Maximum tokens for the agent's input.
        max_summary_tokens (int): Maximum tokens for summarization.
    Methods:
        create_agent(checkpointer, **kwargs): Create the agent's state graph.
        ainvoke(prompt, thread_id=None, **kwargs_model): Asynchronously run
            the agent.
        stream(prompt, thread_id=None, **kwargs_model): Asynchronously stream
            response chunks.
    """
    max_recursion_limit: int = 25
    summarize_node: SummarizationNode
    tools: list = []
    agent: Optional[CompiledStateGraph] = None
    max_tokens: int = 384
    max_summary_tokens: int = 128

    def __init__(self, **kwargs):
        """
        Initialize the MemoryAgent with the given parameters.
        Args:
            max_tokens (int): Maximum tokens for the agent's input.
            max_summary_tokens (int): Maximum tokens for summarization.
            max_recursion_limit (int): Maximum recursion depth for the agent.
            agent (CompiledStateGraph | None): Predefined agent state graph.
            refresh_checkpointer (bool): Whether to refresh the checkpointer.
            **kwargs: Arbitrary keyword arguments for configuration.
        """
        super().__init__(**kwargs)
        self.max_tokens = kwargs.get("max_tokens", self.max_tokens)
        self.max_summary_tokens = kwargs.get(
            "max_summary_tokens",
            self.max_summary_tokens
        )

        self.max_recursion_limit = kwargs.get(
            "max_recursion_limit",
            self.max_recursion_limit
        )

        self.summarize_node = SummarizationNode(
            token_counter=count_tokens_approximately,
            model=self.llm_model,
            max_tokens=384,
            max_summary_tokens=128,
            output_messages_key="llm_input_messages",
        )

        self.agent = kwargs.get("agent", self.agent)
        self.refresh_checkpointer = kwargs.get(
            "refresh_checkpointer",
            self.refresh_checkpointer
        )

    def create_agent(
        self,
        checkpointer,
        **kwargs
    ) -> CompiledStateGraph:
        """
        Create the agent's state graph.
        Args:
            checkpointer: The checkpointer instance to use for managing state.
            **kwargs: Arbitrary keyword arguments for configuration.
        Returns:
            CompiledStateGraph: The compiled state graph for the agent.
        """
        return create_react_agent(
            model=self.llm_model,
            tools=self._get_tools(),
            state_schema=State,
            pre_model_hook=self.summarize_node,
            checkpointer=checkpointer,
            prompt=self._prompt,
            store=self.vector_store,
            **kwargs
        )

    def _get_tools(self):
        """
        Get the tools available for the agent.
        Returns:
            list: A list of tools available for the agent.
        """

        self.tools.extend([
            create_manage_memory_tool(
                namespace=self.namespace,
                store=self.vector_store
            ),
            create_search_memory_tool(
                namespace=self.namespace,
                store=self.vector_store
            )
        ])

        return self.tools

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

    def _process_event(
        self,
        config,
        event_item: dict | None
    ) -> str:
        """
        Process the event item and update memory if necessary.
        Args:
            config: The configuration for the agent.
            event_item (dict | None): The event item to process.
        Returns:
            tuple: A tuple containing the result and a boolean indicating
            if the processing was successful.
        """
        event_response: str = ""
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
                        f">>> Response event: "
                        f"{event_response}"
                    ),
                    extra=get_metadata(thread_id=self.thread_id)
                )
                if (
                    event_response
                    or (len(event_response) > 0)
                ):
                    self.update_memory(
                        event_messages,
                        config=config
                    )

        return event_response

    def invoke(
        self,
        prompt: str,
        thread_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        config_model: dict[str, Any] = {}
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

            if user_id is not None:
                self.user_id = user_id

            if session_id is not None:
                self.session_id = session_id

            config = self._params(
                self.thread_id,
                self.user_id,
                self.session_id
            )

            conn_string = self._redis_uri_store()
            with (
                RedisStore.from_conn_string(
                    conn_string,
                    index=self.index_store()
                ) as store,
                RedisSaver.from_conn_string(conn_string) as checkpointer,
            ):
                store.setup()
                self.vector_store = store
                checkpointer.setup()

                if self.agent is None:
                    self.logger.info("Creating new default agent")
                    self.agent = self.create_agent(
                        checkpointer,
                        **config_model
                    )
                else:
                    self.logger.info("Using existing agent")
                    self.agent.checkpointer = checkpointer

                input_data = {"messages": [
                    {"role": "user", "content": prompt}
                ]}

                response_agent = self.agent.invoke(
                    input=input_data,
                    config=config
                )

                return self._process_event(
                    config=config,
                    event_item=response_agent
                )
        except Exception as e:
            self.logger.error(
                f"Error occurred while invoking agent: {e}",
                extra=get_metadata(thread_id=self.thread_id)
            )
            raise e

    async def stream(
        self,
        prompt: str,
        thread_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        **kwargs_model
    ) -> AsyncIterable[str]:
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

        if user_id is not None:
            self.user_id = user_id

        if session_id is not None:
            self.session_id = session_id

        try:

            config = self._params(
                self.thread_id,
                self.user_id,
                self.session_id
            )

            conn_string = self._redis_uri_store()
            with (
                RedisStore.from_conn_string(conn_string) as store,
                RedisSaver.from_conn_string(conn_string) as checkpointer,
            ):
                store.setup()
                self.vector_store = store
                checkpointer.setup()

                if self.agent is None:
                    self.logger.info("Creating new default agent")
                    self.agent = self.create_agent(
                        checkpointer,
                        **kwargs_model
                    )
                else:
                    self.logger.info("Using existing agent")
                    self.agent.checkpointer = checkpointer

                input_data = {"messages": [
                    {"role": "user", "content": prompt}
                ]}

                index: int = 1
                events = self.agent.stream(
                    input=input_data,
                    config=config,
                    stream_mode="updates",
                    debug=os.getenv("APP_ENV") == "development"
                )

                for event in events:
                    event_index: str = f"Event {index}"
                    self.logger.debug(
                        f">>> {event_index} received: {event}",
                        extra=get_metadata(thread_id=self.thread_id)
                    )
                    event_item = None

                    if "agent" in event:
                        event_item = event["agent"]
                        agent_process: str = (
                            f'{event_index} - Looking up the response agent...'
                        )
                        self.logger.debug(
                            agent_process,
                            extra=get_metadata(thread_id=self.thread_id)
                        )

                    elif "tools" in event:
                        event_item = event["tools"]
                        tool_process: str = (
                            f'{event_index} - Processing the tools...'
                        )
                        self.logger.debug(
                            tool_process,
                            extra=get_metadata(thread_id=self.thread_id)
                        )

                    if event_item is not None:
                        yield self._process_event(
                            config=config,
                            event_item=event_item
                        )
                    index += 1

        except Exception as e:
            # In caso di errore, restituisce un messaggio di errore
            self.logger.error(
                f"Error occurred while processing event: {str(e)}",
                extra=get_metadata(thread_id=self.thread_id)
            )
            raise e
