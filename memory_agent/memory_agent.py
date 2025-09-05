from typing import AsyncIterable, Any, Optional
from langgraph.prebuilt import create_react_agent
from langchain_core.runnables import RunnableConfig
from langmem.short_term import SummarizationNode
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.graph.state import CompiledStateGraph
from .memory_checkpointer import MemoryCheckpointer
from .memory_log import get_metadata
from langmem import (
    create_manage_memory_tool,
    create_search_memory_tool,
)
from .memory_manager import MemoryManager
from .state import State


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

    async def create_agent(
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
            prompt=self._prompt,
            tools=await self._get_tools(),
            store=self.store(),
            state_schema=State,
            pre_model_hook=self.summarize_node,
            checkpointer=checkpointer,
            **kwargs
        )

    async def _get_tools(self):
        """
        Get the tools available for the agent.
        Returns:
            list: A list of tools available for the agent.
        """

        self.tools.extend([
            create_manage_memory_tool(
                namespace=self.namespace,
                store=self.store()
            ),
            create_search_memory_tool(
                namespace=self.namespace,
                store=self.store()
            )
        ])

        return self.tools

    def _params(
        self,
        prompt,
        thread_id
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
        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "recursion_limit": self.max_recursion_limit,
            }
        }

        if self.user_id:
            config["configurable"]["user_id"] = self.user_id

        if self.session_id:
            config["configurable"]["session_id"] = self.session_id

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
                    config=config
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

                    await self.update_memory(
                        event_response,
                        config=config
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

    async def astream(
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

                                await self.update_memory(
                                    event_response,
                                    config=config
                                )

                                yield self._response(
                                    thread_id=self.thread_id,
                                    result=result,
                                    error=None
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
