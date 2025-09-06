import os
import uuid
import json
import datetime
from typing import (
    LiteralString,
    AsyncGenerator,
    Any,
    Optional,
    Literal
)
from neo4j import GraphDatabase, Driver
from langchain.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_community.document_loaders import (CSVLoader,
                                                  PyPDFLoader,
                                                  JSONLoader)
from memory_agent.memory_log import get_metadata
from .components import GraphComponents
from .utils import print_progress_bar
from .prompts import AGENT_PROMPT, parser_prompt
from .cache import MemoryRedisCacheRetriever
from abc import abstractmethod
from langchain_core.runnables import RunnableSerializable
from pyaws_s3 import S3Client
from qdrant_client.models import PointStruct
from langchain_ollama import OllamaEmbeddings
from neo4j_graphrag.retrievers import QdrantNeo4jRetriever
from langchain_openai import OpenAIEmbeddings
from .memory_persistence import MemoryPersistence

PathType = Literal["fs", "s3"]
FormatFile = Literal["pdf", "csv", "json"]


class MemoryGraph(MemoryPersistence):
    """
    MemoryGraph is a class that manages a memory graph using
    Qdrant as the vector store.
    It provides methods to add messages, delete collections,
    and retrieve data.
    """

    neo4j_auth: dict[str, Any] = {
        "url": "neo4j://localhost:7687",
        "username": "neo4j",
        "password": "neo4j",
        "database": None
    }
    neo4j_config: dict[str, Any] = {
        "max_connection_lifetime": 1000,
        "max_connection_pool_size": 50,
        "connection_acquisition_timeout": 30,
        "encrypted": False,
        "trust": "TRUST_ALL_CERTIFICATES"
    }
    neo4j_driver: Driver | None = None
    aws_config: dict[str, Any] | None = {
        "access_key_id": None,
        "secret_access_key": None,
        "bucket": None,
        "region": None
    }
    path_download: str | None
    path_type: PathType = "fs"
    format_file: FormatFile = "pdf"
    fieldnames: list[str] = [
        "file_name",
        "updated_at",
        "ingested",
        "timestamp"
    ]
    memory_redis: MemoryRedisCacheRetriever
    host_persistence_config: dict[str, Any] = {
        "host": "localhost",
        "port": 6379,
        "db": 0,
        "decode_responses": True
    }
    model_embedding_config: dict[str, Any] = {
        "name": None,
        "url": None
    }

    def __init__(self, **kwargs):
        """
        Initialize the MemoryStoreGraph with the provided parameters.

        """
        super().__init__(**kwargs)
        self.neo4j_config = kwargs.get(
            "neo4j_config", self.neo4j_config
        )
        self.neo4j_auth = kwargs.get(
            "neo4j_auth", self.neo4j_auth
        )
        self.host_persistence_config = kwargs.get(
            'host_persistence_config',
            self.host_persistence_config
        )
        self.format_file = kwargs.get('format_file', 'pdf')
        self.path_type = kwargs.get('path_type', 'fs')
        self.aws_config = kwargs.get('aws_config', self.aws_config)
        self.path_download = kwargs.get('path_download', None)

        self.model_embedding_config = kwargs.get(
            "model_embedding_config",
            self.model_embedding_config
        )

        if self.path_type == "s3":
            if self.aws_config is None:
                raise ValueError(
                    "AWS configuration is required for S3 path type."
                )

            if self.path_download is None:
                msg: str = (
                    "Path for downloading files from S3 is not set. "
                    "Please provide a valid path."
                )
                self.logger.error(
                    msg,
                    extra=get_metadata(
                        thread_id=str(self.thread_id)
                    )
                )
                raise ValueError(msg)

            self._create_download_dir(self.path_download)

        self._init_neo4j()
        self._init_redis()

    def _init_redis(self):
        """
        Initialize the Redis connection with the provided parameters.
        """
        try:
            if self.host_persistence_config is None:
                self.host_persistence_config = {
                    "host": "localhost",
                    "port": 6379,
                    "db": 0,
                }

            self.memory_redis = MemoryRedisCacheRetriever(
                **self.host_persistence_config,
                key_search=self.key_search
            )
        except Exception as e:
            msg: str = f"Error connecting to Redis: {str(e)}"
            self.logger.error(
                msg,
                extra=get_metadata(thread_id=str(self.thread_id))
            )
            raise ConnectionError(msg)

    def _init_neo4j(self):
        """
        Initialize the Neo4j driver with the provided authentication details.
        """
        try:
            if self.neo4j_auth is not None:
                self.neo4j_driver = GraphDatabase.driver(
                    self.neo4j_auth["url"],
                    auth=(
                        self.neo4j_auth["username"],
                        self.neo4j_auth["password"]
                    ),
                    **self.neo4j_config
                )

                if self.neo4j_auth['database'] is not None:
                    self.logger.debug(
                        f"Using Neo4j database: {self.neo4j_auth['database']}",
                        extra=get_metadata(thread_id=str(self.thread_id))
                    )
                    self.create_database_if_not_exists(
                        self.neo4j_auth['database']
                    )
        except Exception as e:
            msg: str = f"Error connecting to Neo4j: {str(e)}"
            self.logger.error(
                msg,
                extra=get_metadata(thread_id=str(self.thread_id))
            )
            raise ConnectionError(msg)

    @abstractmethod
    def chain(self, prompt: ChatPromptTemplate) -> RunnableSerializable:
        """
        Get the client LLM instance.
        """
        pass

    def delete_all_relationships(self):
        """
        Delete all relationships in the Neo4j database.
        This method is useful for clearing the graph before a new ingestion.
        Raises:
            ValueError: If the Neo4j driver is not initialized.
        """

        if not self.neo4j_driver:
            msg: str = "Neo4j driver is not initialized."
            self.logger.error(
                msg,
                extra=get_metadata(thread_id=str(self.thread_id))
            )
            raise ValueError(msg)

        with self.neo4j_driver.session() as session:
            session.run("MATCH ()-[r]-() DELETE r")

    def create_database_if_not_exists(self, db_name):
        """
        Create a Neo4j database if it does not already exist.
        Args:
            db_name (str): The name of the database to create.
        Returns:
            None
        Raises:
            ValueError: If the Neo4j driver is not initialized.
        """
        if not self.neo4j_driver:
            msg: str = "Neo4j driver is not initialized."
            self.logger.error(
                msg,
                extra=get_metadata(thread_id=str(self.thread_id))
            )
            raise ValueError(msg)

        with self.neo4j_driver.session(database="system") as session:
            # Controlla se il database esiste già
            result = session.run("SHOW DATABASES")
            dbs = [record["name"] for record in result]
            if db_name in dbs:
                self.logger.debug(
                    f"Il database '{db_name}' esiste già. "
                    "Nessuna azione eseguita."
                )
            else:
                query: LiteralString = "CREATE DATABASE {db_name}"
                session.run(query)
                self.logger.debug(f"Database '{db_name}' creato.")

    def _ensure_str(self, val) -> str:
        """
        Ensure that the value is a string. If it is a list,
        join the elements into a string.
        Args:
            val: The value to ensure is a string.
        Returns:
            str: The value as a string.
        """
        if isinstance(val, list):
            return ", ".join(str(x) for x in val)
        elif not isinstance(val, str):
            return str(val)
        return val

    async def llm_parser(
        self,
        prompt_text: str,
        prompt_user: Optional[str] = None
    ) -> GraphComponents:
        """
        Uses OpenAI's LLM to parse the prompt and extract graph components.
        Args:
            prompt (str): The input text containing nodes and relationships.
        Returns:
            GraphComponents: A Pydantic model containing
            the extracted graph components.
        Raises:
            ValueError: If the OpenAI response content is None.
        """
        try:
            prompt_parser: str = parser_prompt(prompt_user)

            prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    prompt_parser
                ),
                ("human", "{input_text}")
            ])

            response = await self.chain(prompt).ainvoke(
                {"input_text": prompt_text}
            )

            if response is None:
                self.logger.error(
                    "OpenAI response content is None",
                    extra=get_metadata(thread_id=str(self.thread_id))
                )
                raise ValueError("OpenAI response content is None")

            raw_content = response.content

            self.logger.debug(f"Raw content from LLM: {raw_content}")

            # Ensure raw_content is a JSON string
            if not isinstance(raw_content, (str, bytes, bytearray)):
                raw_content = json.dumps(raw_content)

            return GraphComponents.model_validate_json(raw_content)
        except Exception as e:
            self.logger.error(
                f"Error during LLM parsing: {str(e)}",
                extra=get_metadata(thread_id=str(self.thread_id))
            )
            raise e

    async def extract_graph_components(
        self,
        raw_data: str,
        **kwargs
    ) -> tuple[dict[str, str], list[dict[str, str]]]:
        """
        Extract nodes and relationships from the provided
        raw data using LLM.
        Args:
            raw_data (str): The input text containing nodes and relationships.
            prompt_user (str | None): The user prompt to guide the extraction.
        Returns:
            tuple: A tuple containing a dictionary of nodes and a
                list of relationships.
        """

        prompt_user: str | None = kwargs.get("prompt_user", None)

        prompt: str = (
            f"Extract nodes and relationships from the following text:\n"
            f"{raw_data}"
        )

        self.logger.debug(
            f"Extracting graph components from raw data: {prompt}",
            extra=get_metadata(thread_id=str(self.thread_id))
        )
        # Assuming this returns a list of dictionaries
        parsed_response = await self.llm_parser(
            prompt,
            prompt_user=prompt_user
        )
        if not parsed_response:
            msg: str = (
                "Parsed response is empty or does not contain "
                "'graph' attribute"
            )
            self.logger.error(
                msg,
                extra=get_metadata(thread_id=str(self.thread_id))
            )
            raise ValueError(msg)

        # Assuming the 'graph' structure is a key in the parsed response
        parsed_response = parsed_response.graph
        nodes = {}
        relationships = []

        for entry in parsed_response:
            target_node = entry.target_node  # Get target_node from the entry
            node = entry.node
            relationship = entry.relationship  # Get relationship if available

            # Add nodes to the dictionary with a unique ID
            if node not in nodes:
                nodes[node] = str(uuid.uuid4())

            if target_node and target_node not in nodes:
                nodes[target_node] = str(uuid.uuid4())

            # Add relationship to the relationships list with node IDs
            if target_node and relationship:
                relationships.append({
                    "source": nodes[node],
                    "target": nodes[target_node],
                    "type": relationship
                })

        msg: str = (
            f"Extracted {len(nodes)} nodes and "
            f"{len(relationships)} relationships from the raw data."
        )
        self.logger.debug(
            msg,
            extra=get_metadata(
                thread_id=str(self.thread_id)
            )
        )
        return nodes, relationships

    def ingest_to_neo4j(
        self,
        nodes: dict[str, str],
        relationships: list[dict[str, str]]
    ):
        """
        Ingest nodes and relationships into Neo4j.
        Args:
            nodes (dict): A dictionary of nodes with their
                names and unique IDs.
            relationships (list): A list of relationships, each represented
                as a dictionary with source, target, and type.
        Returns:
            dict: A dictionary of nodes with their names and unique IDs.
        Raises:
            ValueError: If Neo4j driver is not initialized or if nodes
                or relationships are empty.
        """
        try:
            if not self.neo4j_driver:
                msg: str = "Neo4j driver is not initialized."
                self.logger.error(
                    msg,
                    extra=get_metadata(thread_id=str(self.thread_id))
                )
                raise ValueError(msg)

            with self.neo4j_driver.session() as session:
                # Create nodes in Neo4j
                for name, node_id in nodes.items():
                    session.run(
                        "CREATE (n:Entity {id: $id, name: $name})",
                        id=node_id,
                        name=name
                    )

                # Create relationships in Neo4j
                for relationship in relationships:
                    session.run(
                        (
                            "MATCH (a:Entity {id: $source_id}), "
                            "(b:Entity {id: $target_id}) "
                            "CREATE (a)-[:RELATIONSHIP {type: $type}]->(b)"
                        ),
                        source_id=relationship["source"],
                        target_id=relationship["target"],
                        type=relationship["type"]
                    )

            return nodes
        except Exception as e:
            self.logger.error(
                f"Error during Neo4j ingestion: {str(e)}",
                extra=get_metadata(thread_id=str(self.thread_id))
            )
            raise e

    def _fetch_related_graph(self, entity_ids):
        """
        Fetch a subgraph related to the given entity IDs from Neo4j.
        Args:
            entity_ids (list): A list of entity IDs to fetch related
            nodes and relationships.
        Returns:
            list: A list of dictionaries representing the subgraph,
                where each dictionary contains:
                - "entity": The entity node.
                - "relationship": The relationship to the related node.
                - "related_node": The related node.
        Raises:
            ValueError: If the Neo4j client is not initialized
                or if entity_ids is empty.
        """

        if not self.neo4j_driver:
            msg: str = "Neo4j driver is not initialized."
            self.logger.error(
                msg,
                extra=get_metadata(thread_id=str(self.thread_id))
            )
            raise ValueError(msg)

        query = """
        MATCH (e:Entity)-[r1]-(n1)-[r2]-(n2)
        WHERE e.id IN $entity_ids
        RETURN e, r1 as r, n1 as related, r2, n2
        UNION
        MATCH (e:Entity)-[r]-(related)
        WHERE e.id IN $entity_ids
        RETURN e, r, related, null as r2, null as n2
        """
        with self.neo4j_driver.session() as session:
            result = session.run(query, entity_ids=entity_ids)
            subgraph = []
            for record in result:
                subgraph.append({
                    "entity": record["e"],
                    "relationship": record["r"],
                    "related_node": record["related"]
                })
                if record["r2"] and record["n2"]:
                    subgraph.append({
                        "entity": record["related"],
                        "relationship": record["r2"],
                        "related_node": record["n2"]
                    })
        return subgraph

    def _format_graph_context(self, subgraph):
        """
        Format the subgraph into a context suitable for LLM processing.
        Args:
            subgraph (list): A list of dictionaries representing
                the subgraph, where each dictionary contains:
                - "entity": The entity node.
                - "relationship": The relationship to the related node.
                - "related_node": The related node.
        Returns:
            dict: A dictionary containing:
                - "nodes": A list of unique node names.
                - "edges": A list of edges in the format "entity
                    relationship related_node".
        Raises:
            ValueError: If the subgraph is empty or if
                the Neo4j driver is not initialized.
        """

        nodes = set()
        edges = []

        for entry in subgraph:
            entity = entry["entity"]
            related = entry["related_node"]
            relationship = entry["relationship"]

            nodes.add(entity["name"])
            nodes.add(related["name"])

            edges.append(
                f"{entity['name']} {relationship['type']} {related['name']}"
            )

        return {"nodes": list(nodes), "edges": edges}

    async def _stream(self, graph_context, user_query):
        """
        Run the GraphRAG process using the provided
            graph context and user query.
        Args:
            graph_context (dict): A dictionary containing the graph
                context with nodes and edges.
            user_query (str): The user's query to be answered
                using the graph context.
            **kwargs: Additional keyword arguments for LLM configuration.
        Returns:
            str: The response from the LLM based on the graph
                context and user query.
        Raises:
            ValueError: If the graph context is not provided or
                if the user query is empty.
            Exception: If there is an error querying the LLM.
        """
        try:
            input, chain = self._get_chain_graph(user_query, graph_context)

            async for response in chain.astream(input):
                if response is None:
                    self.logger.error(
                        "OpenAI response content is None",
                        extra=get_metadata(thread_id=str(self.thread_id))
                    )
                    raise ValueError("OpenAI response content is None")

                if isinstance(response.content, list):
                    answer_text = "\n".join(
                        str(item) for item in response.content
                    )
                else:
                    answer_text = str(response.content)

                yield answer_text.strip()

        except Exception as e:
            self.logger.error(
                f"Error during LLM processing: {str(e)}",
                extra=get_metadata(thread_id=str(self.thread_id))
            )
            raise e

    async def _run(self, graph_context, user_query):
        """
        Run the GraphRAG process using the provided
            graph context and user query.
        Args:
            graph_context (dict): A dictionary containing the graph
                context with nodes and edges.
            user_query (str): The user's query to be answered
                using the graph context.
            **kwargs: Additional keyword arguments for LLM configuration.
        Returns:
            str: The response from the LLM based on the graph
                context and user query.
        Raises:
            ValueError: If the graph context is not provided or
                if the user query is empty.
            Exception: If there is an error querying the LLM.
        """
        try:
            input, chain = self._get_chain_graph(user_query, graph_context)

            response = await chain.ainvoke(input)

            if response is None:
                self.logger.error(
                    "OpenAI response content is None",
                    extra=get_metadata(thread_id=str(self.thread_id))
                )
                raise ValueError("OpenAI response content is None")

            if isinstance(response.content, list):
                answer_text = "\n".join(str(item) for item in response.content)
            else:
                answer_text = str(response.content)
            return answer_text.strip()
        except Exception as e:
            self.logger.error(
                f"Error during LLM processing: {str(e)}",
                extra=get_metadata(thread_id=str(self.thread_id))
            )
            raise e

    def _get_chain_graph(self, user_query, graph_context):
        """
        Get the chain for processing the graph context and user query.
        Args:
            user_query (str): The user's query to be answered
                using the graph context.
            graph_context (dict): A dictionary containing the graph
                context with nodes and edges.
        Returns:
            tuple: A tuple containing:
                - nodes_str (str): A string representation of the nodes.
                - edges_str (str): A string representation of the edges.
                - chain (ChatPromptTemplate): The chain for processing
                    the graph context and user query.
        Raises:
            ValueError: If the graph context is not provided or
                if the user query is empty.
        """

        nodes_str: str = ", ".join(graph_context["nodes"])
        edges_str: str = "; ".join(graph_context["edges"])

        prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    "Provide the answer for the following question:"
                ),
                (
                    "human",
                    AGENT_PROMPT
                )
            ])

        return {
            "nodes_str": nodes_str,
            "edges_str": edges_str,
            "user_query": user_query
        }, self.chain(prompt=prompt)

    async def _ingestion_batch(
        self,
        documents: list[Document],
        **kwargs
    ) -> AsyncGenerator[Any, Any]:
        """
        Ingest a batch of documents into the memory graph.
        Args:
            documents (list[Document]): A list of Document
            objects to be ingested.
            is_delete_relationships (bool, optional): Whether to delete
                all existing relationships before ingestion.
                Defaults to True.
            limit (int, optional): The maximum number of documents to ingest
                in a single batch.
                Defaults to 0. (No limit, ingest all documents)
            **kwargs: Additional keyword arguments for ingestion configuration.
        Returns:
            None
        Raises:
            ValueError: If the collection name is not provided or
            if the documents list is empty.
        """
        if not documents:
            raise ValueError("No documents provided for ingestion")
        limit = kwargs.get("limit", 0)

        if limit > 0:
            documents = documents[:limit]
            self.logger.debug(
                f"Limiting ingestion to the first {limit} documents.",
                extra=get_metadata(thread_id=str(self.thread_id))
            )

        index: int = 1
        for document in documents:
            title = document.metadata.get("title", "Untitled")
            msg = f"Ingesting document {index}/{len(documents)}: {title}"
            self.logger.debug(
                f"{msg}: {title}",
                extra=get_metadata(thread_id=str(self.thread_id))
            )
            raw_data = document.page_content
            try:
                async for step in self._ingestion(
                    raw_data=raw_data,
                    metadata=document.metadata
                ):
                    yield f"{step} ({index}/{len(documents)} - {title})"
            except Exception as e:
                self.logger.error(
                    f"Error during ingestion of document {index}: {str(e)}",
                    extra=get_metadata(thread_id=str(self.thread_id))
                )
                index += 1
                yield "ERROR"
                continue

            index += 1

    async def _ingestion(self, raw_data, metadata: dict | None = None):
        """
        Ingest data into the memory graph.
        This method should be implemented to handle the ingestion process.
        Args:
            raw_data (str): The raw data to be ingested.
            collection_name (str, optional): The name of the Qdrant
                collection to use for ingestion.
            **kwargs: Additional keyword arguments for ingestion configuration.
        Raises:
            NotImplementedError: If the method is not implemented.
        """
        try:
            collection_name = self._get_collection_name()
            yield "Analyzing raw data for graph components."
            nodes, relationships = await self.extract_graph_components(
                raw_data
            )
            yield "Extracted graph components from raw data."

            self.logger.debug(
                f"Extracted {len(nodes)} nodes and "
                f"{len(relationships)} relationships from the raw data."
            )
            yield "Saving nodes and relationships"
            node_id_mapping = self.ingest_to_neo4j(nodes, relationships)
            self.logger.debug(
                f"Ingested {len(node_id_mapping)} nodes into Neo4j."
            )
            yield "Vectorizing raw data and ingesting data"

            await self.ingest_to_qdrant(
                raw_data=raw_data,
                node_id_mapping=node_id_mapping,
                metadata=metadata
            )
            yield "Vectorized raw data and ingested data"
            self.logger.debug(
                f"Ingested data into Qdrant collection {collection_name}."
            )
        except Exception as e:
            self.logger.error(
                f"Error during ingestion process: {str(e)}",
                extra=get_metadata(thread_id=str(self.thread_id))
            )
            raise e

    async def query_stream(
        self,
        query: str,
        entity_ids: Optional[list[Any]] = None
    ):
        """
        Query the memory graph using the provided query.
        Args:
            query (str): The query to be executed on the memory graph.
            collection_name (str, optional): The name of the Qdrant
                collection to use for the query.
            entity_ids (Optional[list[Any]]): A list of entity IDs
                to fetch related nodes and relationships.
        Returns:
            list: A list of search results from the memory graph.
        Raises:
            ValueError: If the collection name is not provided
                or if the query is empty.
        """
        try:
            if not query:
                raise ValueError("Query must not be empty")

            graph_context = self._get_graph_context(
                query,
                entity_ids=entity_ids
            )
            async for s in self._stream(
                graph_context=graph_context,
                user_query=query
            ):
                self.logger.debug(f"Generated answer from LLM: {s}")
                yield s
        except Exception as e:
            self.logger.error(
                f"Error during query process: {str(e)}",
                extra=get_metadata(thread_id=str(self.thread_id))
            )
            raise e

    async def query(
        self,
        query: str,
        entity_ids: Optional[list[Any]] = None
    ):
        """
        Query the memory graph using the provided query.
        Args:
            query (str): The query to be executed on the memory graph.
            collection_name (str, optional): The name of the Qdrant
                collection to use for the query.
            entity_ids (Optional[list[Any]]): A list of entity IDs
                to fetch related nodes and relationships.
        Returns:
            list: A list of search results from the memory graph.
        Raises:
            ValueError: If the collection name is not provided
                or if the query is empty.
        """
        try:
            if not query:
                raise ValueError("Query must not be empty")

            graph_context = self._get_graph_context(
                query,
                entity_ids=entity_ids
            )

            # Run the LLM to get the answer
            answer = await self._run(
                graph_context=graph_context,
                user_query=query
            )
            self.logger.debug(f"Generated answer from LLM: {answer}")
            return answer
        except Exception as e:
            self.logger.error(
                f"Error during query process: {str(e)}",
                extra=get_metadata(thread_id=str(self.thread_id))
            )
            raise e

    def retrieve_ids(self, query: str):
        """
        Retrieve entity IDs from the memory graph based on the provided query.
        Args:
            query (str): The query to be executed on the memory graph.
        Returns:
            list: A list of entity IDs retrieved from the memory graph.
        Raises:
            ValueError: If the Neo4j driver is not initialized
                or if the query is empty.
        """
        try:
            if not query:
                raise ValueError("Query must not be empty")

            if not self.neo4j_driver:
                msg: str = "Neo4j driver is not initialized."
                self.logger.error(
                    msg,
                    extra=get_metadata(thread_id=str(self.thread_id))
                )
                raise ValueError(msg)

            retriever_result = self.retriever_search(
                query=query,
                neo4j_driver=self.neo4j_driver
            )
            return [
                item.content.split("'id': '")[1].split("'")[0]
                for item in retriever_result.items
            ]
        except Exception as e:
            self.logger.error(
                f"Error during retrieval process: {str(e)}",
                extra=get_metadata(thread_id=str(self.thread_id))
            )
            raise e

    def _get_graph_context(
        self,
        query,
        entity_ids: Optional[list[Any]] = None
    ):
        """
        Get the graph context for the provided query.
        Args:
            query (str): The query to be executed on the memory graph.
            entity_ids (Optional[list[Any]]): A list of entity IDs
                to fetch related nodes and relationships.
        Returns:
            dict: A dictionary containing the graph context with
            nodes and edges.
        Raises:
            ValueError: If the Neo4j driver is not initialized
                or if the query is empty.
        """
        if entity_ids is None:
            entity_ids = self.retrieve_ids(query=query)

        self.logger.debug(
                f"Extracted {len(entity_ids)} entity IDs from the "
                "retriever results."
            )
        subgraph = self._fetch_related_graph(entity_ids=entity_ids)
        self.logger.debug(
                f"Fetched subgraph with {len(subgraph)} related nodes "
                "and relationships from Neo4j."
            )
        graph_context = self._format_graph_context(subgraph)
        self.logger.debug(
                f"Formatted graph context with {len(graph_context['nodes'])} "
                f"nodes and {len(graph_context['edges'])} edges."
            )

        return graph_context

    async def load_cache(self) -> list[dict[str, Any]]:
        """
        Loads the cache from a CSV file and returns a list of dictionaries.

        Returns:
            list[dict]: List of dictionaries containing the cache data.
        """
        return await self.memory_redis.get_cache()

    async def _update_cache(
        self,
        file_name: str,
        error: bool = False
    ) -> None:
        """
        Updates or appends a row in a CSV file with file name, timestamp,
        and optional extra metadata.

        Args:
            file_name (str): Name of the file.
            error (bool): Whether an error occurred during processing.
            Defaults to False.
        Returns:
            None
        Raises:
            ValueError: If the file name is not provided.
        Raises:
            Exception: If there is an error while updating the cache.
        """

        # Read existing rows

        try:
            item = await self.memory_redis.get_cache_by(file_name=file_name)
            if item:
                msg: str = (
                    f"File {file_name} already exists in cache. "
                    "Updating existing entry."
                )
                self.logger.info(
                    msg,
                    extra=get_metadata(thread_id=str(self.thread_id))
                )
                # Update existing entry
                await self.memory_redis.update_cache_by(
                    file_name=file_name,
                    updates={
                        "ingested": 1,
                        "error": 1 if error else 0,
                    })
                return

            self.memory_redis.add_cache(data={
                "file_name": file_name,
                "ingested": 1,
                "error": 1 if error else 0,
                "update_at": (
                    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                )
            })
        except Exception as e:
            self.logger.error(
                f"Error updating cache for file {file_name}: {e}",
                extra=get_metadata(thread_id=str(self.thread_id))
            )
            raise e

    def _create_download_dir(
        self,
        path: str,
        delete: bool = False
    ) -> str:
        """
        Creates a temporary local directory to store downloaded files.
        Args:
            path (str): The local path where files will be downloaded.
            delete (bool): Whether to delete existing files in the directory.
        """

        # create the directory if it doesn't exist
        if not os.path.exists(path):
            self.logger.info(f"Creating directory: {path}",
                             extra=get_metadata(
                                thread_id=str(self.thread_id)
                             ))
            os.makedirs(path, exist_ok=True)

        # Se esistono file nella cartella local_dir, cancellali tutti
        if delete:
            for filename in os.listdir(path):
                file_path = os.path.join(path, filename)
                try:
                    if os.path.isfile(file_path):
                        self.logger.info(
                            f"Deleting file: {file_path}",
                            extra=get_metadata(thread_id=str(self.thread_id))
                        )
                        os.remove(file_path)
                except Exception as e:
                    self.logger.error(
                        f"Errore durante la cancellazione di {file_path}: {e}",
                        extra=get_metadata(thread_id=str(self.thread_id))
                    )
                    raise e
        return path

    def _get_documents_from_path(
        self,
        path: str,
        format_file: FormatFile = "pdf",
        **kwargs_loaders: Any
    ) -> list[Document]:
        """
        Loads documents from a specified path based on the file format.
        Args:
        - path (str): The path to the document file.
        """

        if path is None:
            # Log the error message and raise a ValueError
            # to indicate that the path is not set.
            # This will help in debugging and ensure that the user
            # is aware of the issue.
            # Raise a ValueError to indicate that the path is not set.
            # Log the error message
            msg: str = (
                "Path is not set. Please provide a valid path."
            )
            self.logger.error(msg,
                              extra=get_metadata(
                                  thread_id=str(self.thread)
                              ))
            raise ValueError(msg)

        loader: Any = None
        if format_file == "pdf":
            loader = PyPDFLoader(path, **kwargs_loaders)
        elif format_file == "csv":
            loader = CSVLoader(path, **kwargs_loaders)
        elif format_file == "json":
            loader = JSONLoader(path, jq_schema='.', **kwargs_loaders)
        else:
            msg: str = (
                "Unsupported format"
                "Please provide either 'pdf' or 'csv' or 'json'."
            )
            # Log the error message and raise a ValueError
            # to indicate unsupported format
            # This will help in debugging and ensure that
            # the user is aware of the issue.
            # Raise a ValueError to indicate unsupported format
            # This will help in debugging and ensure that
            # the user is aware of the issue.
            # Log the error message
            self.logger.error(
                msg,
                extra=get_metadata(thread_id=str(self.thread))
            )
            raise ValueError(msg)

        if loader is None:
            msg: str = (
                "Loader is not set. "
                "Please provide a valid loader for the specified format."
            )
            self.logger.error(
                msg,
                extra=get_metadata(thread_id=str(self.thread_id))
            )
            raise ValueError(msg)

        return loader.load()

    async def _filter_files(
        self,
        files: list[str],
        refresh: bool = False
    ) -> list[str]:
        """
        Filters files based on a prefix.

        Args:
        - files (list[str]): List of file names.
        - prefix (str): The prefix to filter files.

        Returns:
        - list[str]: A list of filtered file names.
        """
        # Check if files already exist in the cache
        # Only keep files that are in the cache and have ingested == False
        files_to_process = []
        extra = get_metadata(thread_id=str(self.thread_id))

        if not refresh:
            index = 1
            for file in files:
                cached_file = await self.memory_redis.get_cache_by(
                    file_name=file
                )
                # stampa la stessa linea di log
                # con la percentuale nella stessa riga
                print_progress_bar(index, len(files),
                                   prefix=f"Checking file {file}",
                                   length=50)
                if cached_file is None:
                    files_to_process.append(file)
                    index += 1
                    continue

                if cached_file.get("ingested") == 0:
                    files_to_process.append(file)

                index += 1
        else:
            self.logger.info("Refreshing cache. All files will be processed.",
                             extra=extra
                             )
            files_to_process = files

        self.logger.info(
            f"Filtered files: {len(files_to_process)} files to process.",
            extra=extra
        )
        return files_to_process

    async def process_documents(self, **kwargs: Any):
        """
        Processes documents based on the specified path type and format.
        Args:
        - documents (list[Document]): List of Document objects to process.
        - path (str): The local path where files are located.
            If documents are provided, this is ignored.
        - force (bool): Whether to force the processing of files even
            if they already exist in the path.
        - bucket_name (str): The name of the S3 bucket.
            If documents are provided, this is ignored.
        - aws_region (str): The AWS region
            If documents are provided, this is ignored.
        """

        documents: list[Document] = kwargs.get("documents", [])
        extra: dict = get_metadata(thread_id=str(self.thread_id))
        force: bool = kwargs.get("force", False)

        collection_name = self._get_collection_name()

        if len(documents) == 0:
            path = kwargs.get("path", None)
            if path is None:
                msg: str = (
                    "Path is not set. "
                    "Please provide a valid path."
                )
                self.logger.error(msg,
                                  extra=extra)
                raise ValueError(msg)

            if os.path.exists(path) and not force:
                msg: str = (
                    f"Path '{path}' exist. "
                    "Ingestion not required."
                )
                self.logger.warning(
                    msg,
                    extra=extra
                )
                return

            file = os.path.basename(path)
            metadata = {
                "object_name": file,
                "local_path": path
            }

            bucket_name = kwargs.get("bucket_name", None)
            if bucket_name is not None:
                metadata["bucket_name"] = bucket_name

            aws_region = kwargs.get("aws_region", None)
            if aws_region is not None:
                metadata["aws_region"] = aws_region

            ext = os.path.splitext(path)[1].lower()
            if ext == ".pdf":
                self.format_file = "pdf"
            elif ext == ".csv":
                self.format_file = "csv"
            elif ext == ".json":
                self.format_file = "json"
            else:
                msg = f"Unsupported file extension: {ext}"
                self.logger.error(
                    msg,
                    extra=extra
                )
                raise ValueError(msg)

            docs = self._get_documents_from_path(
                path,
                headers=metadata,
                format_file=self.format_file
            )
            if not docs:
                self.logger.warning(
                    f"No documents found in {path}. Skipping.",
                    extra=extra
                )
                return

            for doc in docs:
                doc.metadata.update(metadata)
        else:
            docs = documents

        # Call the ingestion method from the parent class
        async for d in self._ingestion_batch(
            documents=docs,
            collection_name=collection_name,
            thread=self.thread_id
        ):
            yield d

    async def _process_documents_s3(self, **kwargs: Any):
        """
        Processes documents from S3 by downloading them to a
        local path and loading them.

        Args:
        - prefix (str): The S3 prefix to filter files.
        - limit (int): The maximum number of files to download.
        - path_download (str): The local path where files will be downloaded.
        - start (int): The index to start downloading files from.
        - refresh (bool): Whether to refresh the cache.
        - force (bool): Whether to force the download of files even
            if they already exist locally

        Returns:
        - list[Document]: A list of Document objects loaded
            from the downloaded files.
        """

        prefix = kwargs.get("prefix", None)
        limit = kwargs.get("limit", 0)
        start = kwargs.get("start", 0)
        path_download = kwargs.get("path_download", None)
        refresh = kwargs.get("refresh", False)

        if self.aws_config is None:
            msg: str = (
                "AWS configuration is not set. "
                "Please provide valid AWS configuration."
            )
            # Log the error message and raise a ValueError
            # to indicate that the AWS configuration is not set.
            # This will help in debugging and ensure that the user is
            # aware of the issue.
            # Raise a ValueError to indicate that the AWS configuration
            # is not set.
            # Log the error message
            self.logger.error(
                msg,
                extra=get_metadata(thread_id=str(self.thread_id))
            )
            raise ValueError(msg)

        aws_access_key_id = self.aws_config.get(
            "access_key_id",
            None
        )
        aws_secret_access_key = self.aws_config.get(
            "secret_access_key",
            None
        )
        s3_bucket = self.aws_config.get(
            "bucket",
            None
        )
        aws_region = self.aws_config.get(
            "region",
            None
        )

        self.logger.info("Processing documents from S3.")
        if not all([
            aws_access_key_id,
            aws_secret_access_key,
            s3_bucket,
            aws_region
        ]):
            msg: str = (
                "Missing AWS credentials or bucket information for S3 access. "
                "Please provide valid aws_access_key_id, "
                "aws_secret_access_key, "
                "s3_bucket, and aws_region."
            )
            # Log the error message and raise a ValueError
            # to indicate that the AWS credentials or bucket
            # information is missing.
            # This will help in debugging and ensure that the user is
            # aware of the issue.
            # Raise a ValueError to indicate that the AWS credentials or
            # bucket information is missing.
            # Log the error message
            self.logger.error(
                msg,
                extra=get_metadata(thread_id=str(self.thread_id))
            )
            raise ValueError(msg)

        if prefix is None:
            msg: str = (
                "Prefix is not set. "
                "Please provide a valid prefix for S3 files."
            )
            # Log the error message and raise a ValueError
            # to indicate that the prefix is not set.
            # This will help in debugging and ensure that the user is
            # aware of the issue.
            # Raise a ValueError to indicate that the prefix is not set.
            # Log the error message
            self.logger.error(
                msg,
                extra=get_metadata(thread_id=str(self.thread_id))
            )
            # Raise a ValueError to indicate that the prefix is not set.
            # This will help in debugging and ensure that the user is
            # aware of the issue.
            raise ValueError(msg)

        if path_download is None:
            msg: str = (
                "Path for downloading files from S3 is not set. "
                "Please provide a valid path."
            )
            self.logger.error(
                msg,
                extra=get_metadata(thread_id=str(self.thread_id))
            )
            raise ValueError(msg)

        s3_client = S3Client(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            bucket_name=s3_bucket,
            region_name=aws_region
        )
        list_files: list[str] = s3_client.list_files(prefix)
        list_files.sort()

        # filter files based on the cache
        list_files = await self._filter_files(list_files, refresh=refresh)

        if start > 0:
            self.logger.warning(
                f"Starting from index: {start}",
                extra=get_metadata(thread_id=str(self.thread_id))
            )
            list_files = list_files[start:]

        # limit the number of files to download
        if limit > 0:
            self.logger.warning(
                f"Limiting the number of files to download to: {limit}",
                extra=get_metadata(thread_id=str(self.thread_id))
            )
            list_files = list_files[:limit]

        # create a temporary local directory to store downloaded files
        index: int = 1
        size = len(list_files)
        for file in list_files:
            self.logger.info(
                f"Downloading file: {file}",
                extra=get_metadata(thread_id=str(self.thread_id))
            )
            # download the file into the temporary local folder
            local_file = f"{path_download}/{os.path.basename(file)}"
            if not os.path.exists(local_file):
                s3_client.download(file, local_path=local_file)
                self.logger.info(
                    f"File {file} downloaded to: {path_download}",
                    extra=get_metadata(thread_id=str(self.thread_id))
                )
            else:
                self.logger.info(
                    f"File {file} already exists locally in {path_download}",
                    extra=get_metadata(thread_id=str(self.thread_id))
                )

            async for d in self.process_documents(
                path=local_file,
                bucket_name=s3_bucket,
                aws_region=aws_region
            ):
                if d == "ERROR":
                    await self._update_cache(file_name=file, error=True)
                    index += 1
                    continue

            await self._update_cache(file_name=file)
            self.logger.info(
                f"Updating cache for file: {file}",
                extra=get_metadata(thread_id=str(self.thread_id))
            )
            yield index, size, file
            print_progress_bar(
                index,
                size,
                prefix=f"Ingesting documents {file}",
                length=50
            )
            index += 1

    async def _get_files_from_path(
        self,
        path: str,
        limit: int = 0,
        start: int = 0,
        refresh: bool = False
    ) -> list[str]:
        """
        Retrieves a list of files from a specified path.

        Args:
        - path (str): The local path where files are located.

        Returns:
        - list[str]: A list of file names in the specified path.
        """

        self.logger.info(
            "Processing documents from the local filesystem.",
            extra=get_metadata(thread_id=str(self.thread_id))
        )

        if path is None:
            msg: str = (
                "Path is not set. "
                "Please provide a valid path."
            )
            self.logger.error(msg,
                              extra=get_metadata(thread_id=str(self.thread)))
            raise ValueError(msg)

        pdf_files = [f for f in os.listdir(path) if f.lower().endswith(".pdf")]

        if start > 0:
            # Start from a specific index
            self.logger.warning(
                f"Starting from index: {start}",
                extra=get_metadata(thread_id=str(self.thread_id))
            )
            pdf_files = pdf_files[start:]

        if limit > 0:
            # Limit the number of files to process
            self.logger.warning(
                f"Limiting the number of files to process to: {limit}",
                extra=get_metadata(thread_id=str(self.thread_id))
            )
            pdf_files = pdf_files[:limit]

        return await self._filter_files(pdf_files, refresh=refresh)

    def _get_docs(self, path: str) -> list[Document]:
        """
        Loads documents from a specified path based on the file format.

        Args:
        - path (str): The path to the document file.

        Returns:
        - list[Document]: A list of Document objects
            loaded from the specified path.
        """

        metadata = {
            "object_name": path,
            "storage_type": "fs"
        }

        docs = self._get_documents_from_path(path)
        for doc in docs:
            doc.metadata.update(metadata)
        return docs

    async def _process_documents_fs(self, **kwargs: Any):
        """
        Processes documents from the local filesystem by loading
        them from a specified path.

        Args:
        - path (str): The local path where files are located.
        - limit (int): The maximum number of files to process.
        - start (int): The index to start processing files from.
        - refresh (bool): Whether to refresh the cache.
            if force is True, it will ignore the refresh
            flag and process all files.
        - force (bool): Whether to force the processing of files even
            if they already exist in the cache.

        Returns:
        - list[Document]: A list of Document objects loaded from
            the specified path.
        """

        try:
            path = kwargs.get("path", None)
            limit = kwargs.get("limit", 0)
            start = kwargs.get("start", 0)
            refresh = kwargs.get("refresh", False)

            self.logger.info(
                "Processing documents from the local filesystem.",
                extra=get_metadata(thread_id=str(self.thread_id))
            )

            if path is None:
                msg: str = (
                    "Path is not set. "
                    "Please provide a valid path."
                )
                # Log the error message and raise a ValueError
                # to indicate that the path is not set.
                # This will help in debugging and ensure that the user
                # is aware of the issue.
                self.logger.error(
                    msg,
                    extra=get_metadata(thread_id=str(self.thread_id))
                )
                raise ValueError(msg)

            pdf_files = await self._get_files_from_path(
                path=path,
                limit=limit,
                start=start,
                refresh=refresh
            )

            index: int = 1
            size = len(pdf_files)
            for pdf_file in pdf_files:
                file_path = os.path.join(path, pdf_file)
                # Call the ingestion method from the parent class
                async for d in self.process_documents(
                    path=file_path,
                ):
                    if d == "ERROR":
                        await self._update_cache(file_name=pdf_file,
                                                 error=True)
                        index += 1
                        continue

                    yield index, size, pdf_file
                    prefix: str = f"Ingesting documents {pdf_file}"
                    print_progress_bar(index,
                                       size,
                                       prefix=prefix,
                                       length=50)

                    self.logger.info(
                        f"Updating cache for file: {pdf_file}",
                        extra=get_metadata(thread_id=str(self.thread_id))
                    )
                    await self._update_cache(file_name=pdf_file)
                    yield index, size, pdf_file
                    index += 1

        except Exception as e:
            self.logger.error(
                f"Error processing documents from filesystem: {e}",
                extra=get_metadata(thread_id=str(self.thread_id))
            )
            raise e

    async def process_path(self, **kwargs: Any):
        """
        Processes a PDF document by loading it, splitting it into chunks,
        embedding them, and building a knowledge graph.

        Args:
            - pdf_path (str): The path to the PDF file to be processed.
            - path (str): The local path where files are located
                (if path_type is "fs").
            - path_type (str): The type of path, either "fs" for
                local filesystem or "s3" for AWS S3.
            - prefix (str): The S3 prefix to filter files
                (if path_type is "s3").
            - limit (int): The maximum number of files to download
                (if path_type is "s3").
            - path_download (str): The local path where files will be
                downloaded (if path_type is "s3").

        Returns:
        - None
        """
        try:
            path_type: PathType = kwargs.get("path_type", self.path_type)
            limit = kwargs.get("limit", 0)
            start = kwargs.get("start", 0)
            refresh = kwargs.get("refresh", False)

            if refresh:
                await self._refresh_graph()

            if path_type == "s3":
                prefix = kwargs.get("prefix", None)
                path_download = kwargs.get("path_download", self.path_download)
                async for index, size, file in self._process_documents_s3(
                        path_download=path_download,
                        prefix=prefix,
                        limit=limit,
                        start=start,
                        refresh=refresh
                ):
                    yield index, size, file
            elif path_type == "fs":
                path: str | None = kwargs.get("path", None)
                if path is None:
                    msg: str = (
                        "Path is not set. "
                        "Please provide a valid path."
                    )
                    self.logger.error(
                        msg,
                        extra=get_metadata(thread_id=str(self.thread_id))
                    )
                    raise ValueError(msg)
                async for index, size, file in self._process_documents_fs(
                        path=path,
                        limit=limit,
                        start=start,
                        refresh=refresh
                ):
                    yield index, size, file

        except Exception as e:
            self.logger.error(
                f"Error processing documents: {e}",
                extra=get_metadata(thread_id=str(self.thread_id))
            )
            raise e

    async def _refresh_graph(self):
        """
        Refreshes the graph by deleting all relationships and collections,
        then recreating the collection.
        """

        collection_name = self._get_collection_name()
        collection_dim = self._get_collection_dim()

        self.logger.debug(
            "Deleting cache",
            extra=get_metadata(thread_id=str(self.thread_id))
        )
        await self.memory_redis.delete_cache()
        # Delete all relationships in the graph
        self.logger.info(
            "Deleting all relationships in the graph.",
            extra=get_metadata(thread_id=str(self.thread_id))
        )
        self.delete_all_relationships()
        # Delete all collections in the graph
        self.logger.info(
            f"Deleting collection {collection_name} in the graph.",
            extra=get_metadata(thread_id=str(self.thread_id))
        )
        await self.delete_collection_async(collection_name)
        self.logger.info(
            f"Creating new collection {collection_name} in the graph.",
            extra=get_metadata(thread_id=str(self.thread_id))
        )
        await self.create_collection_async(
            collection_name,
            collection_dim
        )
        self.logger.info(
            "Graph refreshed successfully.",
            extra=get_metadata(thread_id=str(self.thread_id))
        )

    @abstractmethod
    def embeddings(
        self,
        raw_data
    ) -> list:
        """
        Get the embedding model to be used for text embedding.
        Returns:
            The embedding model instance.
        """
        pass

    async def ingest_to_qdrant(
        self,
        raw_data,
        node_id_mapping,
        metadata: dict | None = None
    ):
        """
        Ingest raw data into Qdrant as embeddings.
        Args:
            raw_data (str): The raw data to be ingested.
            node_id_mapping (dict): A mapping of node names to unique IDs.
            collection_name (str, optional): The name of the Qdrant collection.
                If not provided, the default collection name will be used.
            metadata (dict, optional): Additional metadata to be added
                to the payload of each point.
                Defaults to None.
        Returns:
            None
        Raises:
            ValueError: If the collection name is not provided
                or if the raw data is empty.
        """
        try:
            collection_name = self._get_collection_name()
            collection_dim = self._get_collection_dim()

            if await self.create_collection_async(
                collection_name,
                collection_dim
            ):
                self.logger.debug(
                    f"Collection '{collection_name}' created successfully"
                )

            e = self.embeddings(raw_data)

            points = [
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=e,
                    payload={"id": node_id}
                )
                for node_id, e in zip(
                    node_id_mapping.values(),
                    e
                )
            ]

            # Add metadata to each point's payload if provided
            if metadata:
                for point in points:
                    if isinstance(point.payload, dict):
                        point.payload.update(metadata)
                    else:
                        point.payload = metadata

            await self.qdrant_client_async.upsert(
                collection_name=collection_name,
                points=points
            )
        except Exception as e:
            self.logger.error(
                f"Error during Qdrant ingestion: {str(e)}",
                extra=get_metadata(thread_id=str(self.thread_id))
            )
            raise e

    def retriever_search(self, query, neo4j_driver):
        """
        Perform a search using the QdrantNeo4jRetriever.
        Args:
            neo4j_driver (Driver): The Neo4j driver instance.
            qdrant_client (QdrantClient): The Qdrant client instance.
            collection_name (str): The name of the Qdrant collection.
            query (str): The search query.
        Returns:
            list: A list of search results.
        Raises:
            ValueError: If the Neo4j driver or Qdrant client
                is not initialized.
        """
        try:

            collection_name = self._get_collection_name()

            retriever = QdrantNeo4jRetriever(
                driver=neo4j_driver,
                client=self.qdrant_client,
                collection_name=collection_name,
                id_property_external="id",
                id_property_neo4j="id",
            )

            openai_embeddings = self.get_embedding_model()

            if not (
                isinstance(openai_embeddings, OpenAIEmbeddings) or
                (
                    OllamaEmbeddings and
                    isinstance(openai_embeddings, OllamaEmbeddings)
                )
            ):
                msg: str = (
                    "Embedding model must be an instance of "
                    "OpenAIEmbeddings or OllamaEmbeddings"
                )
                self.logger.error(
                    msg,
                    extra=get_metadata(thread_id=str(self.thread_id))
                )
                raise ValueError(msg)

            query_vector = openai_embeddings.embed_query(query)
            results = retriever.search(
                query_vector=query_vector,
                top_k=5
            )

            return results
        except Exception as e:
            self.logger.error(
                f"Error during retriever search: {str(e)}",
                extra=get_metadata(thread_id=str(self.thread_id))
            )
            raise e
