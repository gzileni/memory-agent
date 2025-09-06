import uuid
from typing import Any, Literal
from langchain_qdrant import QdrantVectorStore
from qdrant_client.http.models import Distance
from langchain_core.documents import Document
from qdrant_client import models, AsyncQdrantClient, QdrantClient
from .memory import MemoryStore
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from fastembed.common.model_description import PoolingType, ModelSource
from fastembed import TextEmbedding


TypeEmbeddingModelVs = Literal["local", "hf"]


class MemoryPersistence(MemoryStore):
    """
    MemoryPersistence is a memory management class that integrates with Qdrant,
    a vector database, to persist and retrieve vectorized documents for
    AI agents. It provides both synchronous and asynchronous interfaces for
    managing collections, adding and searching documents, and handling
    vector stores.

    Usage:
        This class is intended to be used as a backend for AI agents requiring
        persistent, searchable memory using Qdrant as the vector database.
    """

    COLLECTION_NAME_DEFAULT = "memory_agent"
    COLLECTION_DIM_DEFAULT = 1536
    model_embedding_vs: TextEmbedding | None = None
    model_embedding_vs_config: dict[str, Any] = {
        "path": None,
        "type": "hf",
        "name": "BAAI/bge-large-en-v1.5"
    }
    qdrant_config: dict[str, Any] = {
        "url": "http://localhost:6333",
    }
    collection_config: dict[str, Any] = {
        "collection_name": COLLECTION_NAME_DEFAULT,
        "vectors_config": {
            "size": COLLECTION_DIM_DEFAULT,
            "distance": Distance.COSINE
        }
    }
    key_search: str | None = None
    qdrant_client_async: AsyncQdrantClient
    qdrant_client: QdrantClient

    def __init__(self, **kwargs: Any) -> None:
        """
        Initializes an instance of MemoryAgentQdrant with the
        provided parameters.

        Args:
            *args (Any): Positional arguments passed to the
                parent class initializer.
            **kwargs (Any): Optional arguments including:
                - qdrant_url (str, optional): The URL of the Qdrant server.
        """
        super().__init__(**kwargs)

        self.key_search = kwargs.get("key_search", "thread_agent")
        self.qdrant_config = kwargs.get(
            "qdrant_config",
            self.qdrant_config
        )

        if self.qdrant_config is None:
            raise ValueError("qdrant_config must be set")

        self.model_embedding_vs_config = kwargs.get(
            "model_embedding_vs_config",
            self.model_embedding_vs_config
        )
        if self.model_embedding_vs_config is None:
            raise ValueError("model_embedding_vs_config must be set")

        self._init_qdrant()

    def _get_collection_name(self) -> str:
        """
        Get the collection name from the Qdrant configuration.

        Returns:
            str: The collection name.
        Raises:
            ValueError: If the collection name is not set in the
                Qdrant configuration.
        """
        collection_name: str | None = self.collection_config.get(
            "collection_name",
            self.COLLECTION_NAME_DEFAULT
        )
        if collection_name is None:
            raise ValueError(
                "collection_name must be set in collection_config"
            )
        return collection_name

    def _get_collection_dim(self) -> int:
        """
        Get the collection dimension from the Qdrant configuration.
        Returns:
            int: The collection dimension.
            Raises:
            ValueError: If the vectors configuration or size
                is not set in the Qdrant configuration.
        """
        vectors_config = self.collection_config.get(
            "vectors_config",
            None
        )
        if vectors_config is None:
            raise ValueError("Vectors configuration must be provided")
        collection_dim = vectors_config.get(
            "size",
            self.COLLECTION_DIM_DEFAULT
        )
        return collection_dim

    def get_embedding_model_vs(self) -> Any:
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

            model_name: str = self.model_embedding_vs_config.get(
                "name",
                "BAAI/bge-large-en-v1.5"
            )
            model_type: str = self.model_embedding_vs_config.get(
                "type",
                "hf"
            )
            model_path: str | None = self.model_embedding_vs_config.get(
                "path",
                None
            )

            if model_name is None:
                raise ValueError("model_embedding_vs_name must be set")

            if model_type.lower() == 'local':
                if model_path is None:
                    msg = (
                        "model_embedding_path not set, "
                        "using default local model path"
                    )
                    self.logger.error(msg)
                    raise ValueError("model_embedding_path must be set")
                TextEmbedding.add_custom_model(
                    model=model_name,
                    pooling=PoolingType.MEAN,
                    normalization=True,
                    sources=ModelSource(hf=model_name),
                    dim=384,
                    model_file=model_path,
                )
                return TextEmbedding(model=model_name)
            elif model_type.lower() == 'hf':
                return TextEmbedding(model=model_name)
        except Exception as e:
            msg = (
                f"Errore durante il caricamento del modello di embedding "
                f"per il database vettoriale: {e}"
            )
            self.logger.error(msg)
            raise e

    def _init_qdrant(
        self
    ):
        """
        Set the Qdrant URL for the client.

        Args:
            qdrant_url (str): The URL of the Qdrant server.
        """
        url = self.qdrant_config.get("url", None)
        if url is None:
            raise ValueError("qdrant_url must be set")
        self.qdrant_client_async = AsyncQdrantClient(url=url)
        self.qdrant_client = QdrantClient(url=url)

        model_name: str = self.model_embedding_vs_config.get(
            "name",
            "BAAI/bge-large-en-v1.5"
        )

        self.qdrant_client_async.set_model(model_name)
        self.qdrant_client.set_model(model_name)

    async def get_vector_store(
        self
    ) -> QdrantVectorStore:
        """
        Get or create a Qdrant vector store for the specified collection.
        Args:
            collection (str | None): The name of the collection to use.
                If None, uses the default collection name.

        Returns:
            QdrantVectorStore:
                The Qdrant vector store for the specified collection.
        """
        qdrant_url = self.qdrant_config.get("url", None)
        if qdrant_url is None:
            raise ValueError("qdrant_url must be set")

        collection_name = self._get_collection_name()
        collections_list = await self.qdrant_client_async.get_collections()
        existing_collections = [
            col.name for col in collections_list.collections
        ]

        # Check if the collection exists, if not, create it
        if collection_name not in existing_collections:
            await self.qdrant_client_async.create_collection(
                **self.collection_config
            )
            self.logger.info(
                f"Collection '{collection_name}' created successfully!"
            )
        else:
            self.logger.info(f"Collection '{collection_name}' already exists.")

        # Initialize Qdrant vector store from the existing collection
        return QdrantVectorStore.from_existing_collection(
            embedding=self.get_embedding_model_vs(),
            collection_name=collection_name,
            url=qdrant_url
        )

    def client_async(self) -> AsyncQdrantClient:
        """
        Get the Qdrant client.

        Returns:
            AsyncQdrantClient: The Qdrant client.
        """
        return self.qdrant_client_async

    def client(self) -> QdrantClient:
        """
        Get the Qdrant client.

        Returns:
            QdrantClient: The Qdrant client.
        """
        return self.qdrant_client

    async def search_filter_async(
        self,
        query: str,
        metadata_value: str,
        collection: str | None = None
    ) -> list[Document]:
        """
        Get the filter conditions for the Qdrant search.

        Args:
            query:  The search query.
            metadata_value:  The value to match in the metadata field.
            collection (str | None): The name of the collection to use.
                If None, uses the default collection name.

        Returns:
            list: A list of filter conditions for the Qdrant search.
        """
        metadata_query = [
            models.FieldCondition(
                key=str(self.key_search),
                match=models.MatchValue(value=metadata_value)
            )
        ]

        vs = await self.get_vector_store()

        return await vs.asimilarity_search(
            query=query,
            k=1,
            filter=metadata_query
        )

    async def save_async(
        self,
        last_message: str,
        thread: str | None = None,
        custom_metadata: dict[str, Any] | None = None
    ):
        """
        Save the last message to the Qdrant vector store.

        Args:
            last_message (str): The last message content.
            thread_id (str, optional): The thread ID to associate
                with the message.

        Raises:
            Exception: If there is an error saving the message.
            :param thread_id:  Thread ID
            :param last_message:  Last message content
        """
        if not last_message.strip():
            return

        metadata: dict = {
            "thread": thread if thread else self.thread_id,
        }
        if custom_metadata is not None:
            metadata["custom"] = custom_metadata

        # Save the response to the database
        vs = await self.get_vector_store()
        if last_message is not None and last_message != "":
            doc_id = str(uuid.uuid4())
            message_doc = Document(
                page_content=last_message,
                metadata=metadata,
                id=doc_id
            )
            await vs.aadd_documents([message_doc], ids=[doc_id])

    async def delete_collection_async(self, collection: str):
        """
        Deletes a collection from Qdrant based on the
        specified collection name.

        Args:
            collection (str): The name of the collection to delete.
                Defaults to os.getenv("COLLECTION_NAME").
        """
        try:
            await self.qdrant_client_async.delete_collection(
                collection_name=collection
            )
            self.logger.info(
                f"Collection '{collection}' deleted "
                "successfully."
            )
        except Exception as e:
            self.logger.error(f"Error deleting collection '{collection}': {e}")
            raise e

    def delete_collection(self, collection: str):
        """
        Deletes a collection from Qdrant based on the specified
        collection name.

        Args:
            collection (str): The name of the collection to delete.
                Defaults to os.getenv("COLLECTION_NAME").
        """
        try:
            self.qdrant_client.delete_collection(collection_name=collection)
            self.logger.info(
                f"Collection '{collection}' deleted successfully."
            )
        except Exception as e:
            self.logger.error(f"Error deleting collection '{collection}': {e}")
            raise e

    async def retriever(self, urls: list[str], **kwargs) -> Any:
        """
        Asynchronously retrieves a Qdrant retriever for the specified
        collection.

        Args:
            urls (list[str]): A list of URLs to load documents from.
            headers (dict[str, Any], optional): Headers to use when loading
                documents. Defaults to None.

        Returns:
            Any: The Qdrant retriever for the specified collection.

        Info:
            - https://langchain-ai.github.io/langgraph/how-tos/
              multi-agent-network/#using-a-custom-agent-implementation
            - https://langchain-ai.github.io/langgraph/tutorials/rag/
              langgraph_agentic_rag/#retriever
        """

        headers = kwargs.get("headers", None)

        # Load documents from the provided URLs using WebBaseLoader
        docs = [
            WebBaseLoader(url, header_template=headers).load()
            for url in urls
        ]
        docs_list = [item for sublist in docs for item in sublist]

        # Split the loaded documents into chunks using
        # RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=100,
            chunk_overlap=50
        )
        doc_splits = text_splitter.split_documents(docs_list)

        # Initialize the vector store and add the document splits
        vs = await self.get_vector_store()
        await vs.aadd_documents(doc_splits)

        # Return the Qdrant retriever
        return vs.as_retriever()

    async def create_collection_async(
        self, collection_name, vector_dimension=1536
    ) -> bool:
        """
        Create a collection in Qdrant with the specified name and
        vector dimension.

        Args:
            client (QdrantClient): The Qdrant client instance.
            collection_name (str): The name of the collection to create.
            vector_dimension (int): The dimension of the vectors in
                the collection.

        Returns:
            None
        """
        # Try to fetch the collection status
        try:
            await self.qdrant_client_async.get_collection(collection_name)
            self.logger.info(
                f"Skipping creating collection; '{collection_name}' "
                "already exists."
            )
            return True
        except Exception as e:
            # If collection does not exist, an error will be thrown,
            # so we create the collection
            if 'Not found: Collection' in str(e):
                self.logger.info(
                    f"Collection '{collection_name}' not found. "
                    "Creating it now..."
                )

                await self.qdrant_client_async.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_dimension,
                        distance=models.Distance.COSINE
                    )
                )
                self.logger.info(
                    f"Collection '{collection_name}' created "
                    "successfully."
                )
                return True
            else:
                self.logger.error(
                    f"Error while checking collection: {e}"
                )
                return False

    async def add_documents_async(
        self,
        documents: list[Document]
    ):
        """
        Adds documents to the vector store, checking if they already exist.

        Args:
            documents (List[Document]): A list of documents to add.
        """
        new_documents = []
        vs = await self.get_vector_store()

        for doc in documents:
            # Check if the document already exists in the vector store
            existing_docs = await vs.asimilarity_search(
                doc.page_content,
                k=1
            )
            if (
                existing_docs and
                existing_docs[0].page_content == doc.page_content
            ):
                # Skip adding the document if it already exists
                continue

            # Create a new document with a unique ID
            d = Document(
                page_content=doc.page_content,
                id=str(uuid.uuid4())
            )
            new_documents.append(d)

        # Add only new documents to the vector store
        if new_documents:
            self.logger.info(
                f"Adding {len(new_documents)} new documents "
                "to the vector store"
            )
            await vs.aadd_documents(new_documents)

    def create_collection(
        self,
        collection_name,
        vector_dimension=1536
    ) -> bool:
        """
        Create a collection in Qdrant with the specified name and
        vector dimension.

        Args:
            client (QdrantClient): The Qdrant client instance.
            collection_name (str): The name of the collection to create.
            vector_dimension (int): The dimension of the vectors in
                the collection.

        Returns:
            None
        """
        # Try to fetch the collection status
        try:
            result = self.qdrant_client.get_collection(collection_name)
            self.logger.info(
                f"created collection; '{result}' already exists."
            )
            return True
        except Exception as e:
            # If collection does not exist, an error will be thrown,
            # so we create the collection
            if 'Not found: Collection' in str(e):
                self.logger.info(
                    f"Collection '{collection_name}' not found. "
                    "Creating it now..."
                )

                self.qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_dimension,
                        distance=models.Distance.COSINE
                    )
                )

                self.logger.info(
                    f"Collection '{collection_name}' created successfully."
                )
                return True
            else:
                self.logger.error(
                    f"Error while checking collection: {e}"
                )
                return False
