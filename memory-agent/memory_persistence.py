import os
import uuid
from typing import Any
from langchain_qdrant import QdrantVectorStore
from qdrant_client.http.models import Distance, VectorParams
from langchain_core.documents import Document
from qdrant_client import models, AsyncQdrantClient, QdrantClient
from .memory import MemoryStore
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class MemoryPersistence(MemoryStore):
    """
    MemoryPersistence is a memory management class that integrates with Qdrant, a vector database, to persist and retrieve vectorized documents for AI agents. It provides both synchronous and asynchronous interfaces for managing collections, adding and searching documents, and handling vector stores.
    Attributes:
        qdrant_url (str): The URL of the Qdrant server.
        qdrant_client_async (AsyncQdrantClient): Asynchronous Qdrant client instance.
        qdrant_client (QdrantClient): Synchronous Qdrant client instance.
        model_embedding_vs_type (str): Type of embedding model to use for vector storage.
        model_embedding_vs_path (str | None): Optional path to a custom embedding model.
        model_embedding_vs_name (str): Name of the embedding model to use.
    Methods:
        __init__(**kwargs): Initializes the MemoryPersistence instance with Qdrant connection and embedding model configuration.
        async get_vector_store(collection: str | None = None) -> QdrantVectorStore:
            Retrieves or creates a Qdrant vector store for a specified collection.
        get_client_async() -> AsyncQdrantClient:
            Returns the asynchronous Qdrant client.
        get_client() -> QdrantClient:
            Returns the synchronous Qdrant client.
        async search_filter_async(query: str, metadata_value: str, collection: str | None = None):
            Performs a similarity search in Qdrant with metadata filtering.
        async save_async(last_message: str, thread: str | None = None, custom_metadata: dict[str, Any] | None = None, collection: str | None = None):
            Saves a message as a document in the Qdrant vector store.
        async delete_collection_async(collection: str):
            Asynchronously deletes a collection from Qdrant.
        delete_collection(collection: str):
            Synchronously deletes a collection from Qdrant.
        async retriever(urls: list[str], **kwargs) -> Any:
            Loads documents from URLs, splits them, adds them to Qdrant, and returns a retriever.
        async create_collection_async(collection_name, vector_dimension=1536) -> bool:
            Asynchronously creates a collection in Qdrant if it does not exist.
        async add_documents_async(documents: list[Document], collection: str | None = None):
            Adds documents to the vector store, avoiding duplicates.
        create_collection(collection_name, vector_dimension=1536) -> bool:
            Synchronously creates a collection in Qdrant if it does not exist.
        ValueError: If required configuration parameters are missing.
        Exception: For errors during collection or document operations.
    Usage:
        This class is intended to be used as a backend for AI agents requiring persistent, searchable memory using Qdrant as the vector database.
    """

    qdrant_url : str | None = None
    qdrant_client_async: AsyncQdrantClient
    qdrant_client: QdrantClient

    def __init__(self, **kwargs: Any) -> None:
        """
        Initializes an instance of MemoryAgentQdrant with the provided parameters.

        Args:
            *args (Any): Positional arguments passed to the parent class initializer.
            **kwargs (Any): Optional arguments including:
                - qdrant_url (str, optional): The URL of the Qdrant server.
        """
        super().__init__(**kwargs)
        self.qdrant_url = kwargs.get("qdrant_url", None)
        
        if self.qdrant_url is None:
            raise ValueError("Qdrant URL not provided. Please set the 'qdrant_url' parameter.")
        
        self.model_embedding_vs_type = kwargs.get("model_embedding_vs_type", "hf")
        self.model_embedding_vs_path = kwargs.get("model_embedding_vs_path", None)
        self.model_embedding_vs_name = kwargs.get("model_embedding_vs_name", "BAAI/bge-large-en-v1.5") 
        
        if self.model_embedding_vs_type is None and self.model_embedding_vs_name is None:
            raise ValueError("Either 'model_embedding_vs_type' or 'model_embedding_vs_name' must be provided.")
        
        self.qdrant_client_async = AsyncQdrantClient(url=self.qdrant_url)
        self.qdrant_client_async.set_model(self.model_embedding_vs_name)
        
        self.qdrant_client = QdrantClient(url=self.qdrant_url)
        self.qdrant_client.set_model(self.model_embedding_vs_name)

    async def get_vector_store(self, collection : str | None = None) -> QdrantVectorStore:
        """
        Get or create a Qdrant vector store for the specified collection.
        Args:
            collection (str | None): The name of the collection to use. If None, uses the default collection name.

        Returns:
            QdrantVectorStore: The Qdrant vector store for the specified collection.
        """
        collection_name = collection if collection else self.collection_name
        collections_list = await self.qdrant_client_async.get_collections()
        existing_collections = [col.name for col in collections_list.collections]

        # Check if the collection exists, if not, create it
        if self.collection_name not in existing_collections:
            await self.qdrant_client_async.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=self.collection_dim, distance=Distance.COSINE)
            )
            print(f"Collection '{collection_name}' created successfully!")
        else:
            print(f"Collection '{collection_name}' already exists.")

        # Initialize Qdrant vector store from the existing collection
        return QdrantVectorStore.from_existing_collection(
            embedding=self.get_embedding_model_vs(),
            collection_name=collection_name,
            url=self.qdrant_url
        )
        
    def get_client_async(self) -> AsyncQdrantClient:
        """
        Get the Qdrant client.

        Returns:
            AsyncQdrantClient: The Qdrant client.
        """
        return self.qdrant_client_async
    
    def get_client(self) -> QdrantClient:
        """
        Get the Qdrant client.

        Returns:
            QdrantClient: The Qdrant client.
        """
        return self.qdrant_client

    async def search_filter_async(self, query : str, metadata_value: str, collection : str | None = None) -> list[Document]:
        """
        Get the filter conditions for the Qdrant search.

        Args:
            query:  The search query.
            metadata_value:  The value to match in the metadata field.
            collection (str | None): The name of the collection to use. If None, uses the default collection name.

        Returns:
            list: A list of filter conditions for the Qdrant search.
        """
        metadata_query = [models.FieldCondition(key=self.key_search, match=models.MatchValue(value=metadata_value))]

        vs = await self.get_vector_store(collection=collection)

        return await vs.asimilarity_search(
            query=query,
            k=1,
            filter=metadata_query
        )

    async def save_async(self, last_message: str, thread: str | None = None, custom_metadata : dict[str, Any] | None = None, collection: str | None = None):
        """
        Save the last message to the Qdrant vector store.

        Args:
            last_message (str): The last message content.
            thread_id (str, optional): The thread ID to associate with the message.

        Raises:
            Exception: If there is an error saving the message.
            :param thread_id:  Thread ID
            :param last_message:  Last message content
        """
        if not last_message.strip():
            return

        metadata : dict = {
            "thread": thread if thread else self.thread,
        }
        
        if custom_metadata is not None:
            metadata["custom"] = custom_metadata

        # Save the response to the database
        vs = await self.get_vector_store(collection=collection)
        if last_message is not None and last_message != "":
            doc_id = str(uuid.uuid4())
            message_doc = Document(
                page_content=last_message,
                metadata=metadata,
                id=doc_id
            )
            await vs.aadd_documents([message_doc], ids=[doc_id])
            
    async def delete_collection_async(self, collection : str):
        """
        Deletes a collection from Qdrant based on the specified collection name.

        Args:
            collection (str): The name of the collection to delete. Defaults to os.getenv("COLLECTION_NAME").
        """
        try:
            await self.qdrant_client_async.delete_collection(collection_name=collection)
            print(f"Collection '{collection}' deleted successfully.")
        except Exception as e:
            print(f"Error deleting collection '{collection}': {e}")
            raise e

    def delete_collection(self, collection: str):
        """
        Deletes a collection from Qdrant based on the specified collection name.

        Args:
            collection (str): The name of the collection to delete. Defaults to os.getenv("COLLECTION_NAME").
        """
        try:
            self.qdrant_client.delete_collection(collection_name=collection)
            print(f"Collection '{collection}' deleted successfully.")
        except Exception as e:
            print(f"Error deleting collection '{collection}': {e}")
            raise e

    async def retriever(self, urls: list[str], **kwargs) -> Any:
        """
        Asynchronously retrieves a Qdrant retriever for the specified collection.

        Args:
            urls (list[str]): A list of URLs to load documents from.
            headers (dict[str, Any], optional): Headers to use when loading documents. Defaults to None.

        Returns:
            Any: The Qdrant retriever for the specified collection.

        Info:
            - https://langchain-ai.github.io/langgraph/how-tos/multi-agent-network/#using-a-custom-agent-implementation
            - https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_agentic_rag/#retriever
        """

        headers = kwargs.get("headers", None)
        collection = kwargs.get("collection", "retriever")

        # Load documents from the provided URLs using WebBaseLoader
        docs = [WebBaseLoader(url, header_template=headers).load() for url in urls]
        docs_list = [item for sublist in docs for item in sublist]

        # Split the loaded documents into chunks using RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=100,
            chunk_overlap=50
        )
        doc_splits = text_splitter.split_documents(docs_list)

        # Initialize the vector store and add the document splits
        vs = await self.get_vector_store(collection=collection)
        await vs.aadd_documents(doc_splits)

        # Return the Qdrant retriever
        return vs.as_retriever()
    
    async def create_collection_async(self, collection_name, vector_dimension = 1536) -> bool:
        """
        Create a collection in Qdrant with the specified name and vector dimension.

        Args:
            client (QdrantClient): The Qdrant client instance.
            collection_name (str): The name of the collection to create.
            vector_dimension (int): The dimension of the vectors in the collection.

        Returns:
            None
        """
        # Try to fetch the collection status
        try:
            await self.qdrant_client_async.get_collection(collection_name)
            print(f"Skipping creating collection; '{collection_name}' already exists.")
            return True
        except Exception as e:
            # If collection does not exist, an error will be thrown, so we create the collection
            if 'Not found: Collection' in str(e):
                print(f"Collection '{collection_name}' not found. Creating it now...")

                await self.qdrant_client_async.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(size=vector_dimension, distance=models.Distance.COSINE)
                )
                print(f"Collection '{collection_name}' created successfully.")
                return True
            else:
                print(f"Error while checking collection: {e}")
                return False
    
    async def add_documents_async(self, documents: list[Document], collection: str | None = None):
        """
        Adds documents to the vector store, checking if they already exist.

        Args:
            documents (List[Document]): A list of documents to add.
        """
        new_documents = []
        vs = await self.get_vector_store(collection=collection)
        
        for doc in documents:
            # Check if the document already exists in the vector store
            existing_docs = await vs.asimilarity_search(doc.page_content, k=1)
            if existing_docs and existing_docs[0].page_content == doc.page_content:
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
            print(f"Adding {len(new_documents)} new documents to the vector store")
            await vs.aadd_documents(new_documents)
    
    def create_collection(self, collection_name, vector_dimension = 1536) -> bool:
        """
        Create a collection in Qdrant with the specified name and vector dimension.

        Args:
            client (QdrantClient): The Qdrant client instance.
            collection_name (str): The name of the collection to create.
            vector_dimension (int): The dimension of the vectors in the collection.

        Returns:
            None
        """
        # Try to fetch the collection status
        try:
            result = self.qdrant_client.get_collection(collection_name)
            print(f"created collection; '{result}' already exists.")
            return True
        except Exception as e:
            # If collection does not exist, an error will be thrown, so we create the collection
            if 'Not found: Collection' in str(e):
                print(f"Collection '{collection_name}' not found. Creating it now...")

                self.qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(size=vector_dimension, distance=models.Distance.COSINE)
                )

                print(f"Collection '{collection_name}' created successfully.")
                return True
            else:
                print(f"Error while checking collection: {e}")
                return False