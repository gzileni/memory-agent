import uuid
from abc import abstractmethod
from typing import Any, Literal
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import IndexConfig
from langchain_core.documents import Document
from fastembed import TextEmbedding
from fastembed.common.model_description import PoolingType, ModelSource

TypeEmbeddingModel = Literal["openai", "ollama", "vllm"]
TypeEmbeddingModelVs = Literal["local", "hf"]

class MemoryStore:
    """
    Class representing an agent that uses a memory store to manage input and output.
    """

    model_embedding_type: TypeEmbeddingModel = "openai"
    model_embedding : OpenAIEmbeddings | OllamaEmbeddings | None = None
    model_embedding_name : str = "text-embedding-3-small"
    model_embedding_url: str | None = None
    model_embedding_vs_path: str | None = None
    model_embedding_vs_type : TypeEmbeddingModelVs = "hf"
    model_embedding_vs_name : str = "BAAI/bge-large-en-v1.5"
    model_embedding_vs : TextEmbedding | None = None
    collection_name: str
    collection_dim: int = 1536
    key_search : str = "metadata.thread"
    thread: str | None = str(uuid.uuid4()) 

    def __init__(self, **kwargs: Any) -> None:
        """
        Initializes an instance of MemoryAgent with the provided parameters.

        Args:
            *args (Any): Positional arguments. If provided, the first argument represents the LLM model
                         and the second represents the embedding model_embedding_name.
            **kwargs (Any): Optional arguments such as:
                - model_embedding_type (str): The type of language model to use (e.g., 'openai', 'ollama', 'vllm', 'custom'). Default value: 'openai'.
                - model_embedding_name (str): The embedding model to use. If not specified, the 'MODEL_EMBEDDING' environment variable is used. Default value: 'text-embedding-3-small'.
                - model_embedding_url (str, optional): The base URL for the model.
                - collection_name (str, optional): The name of the collection. Default value: 'memory_store'.
                - collection_dim (int, optional): The dimension of the collection. Default value: 1536.
                - model_embedding_path (str, optional): The path to the model embedding file.
        """
        self.thread = kwargs.get("thread", "memory_store")
        
        self.collection_name = kwargs.get("collection_name", "memory_store")
        if self.collection_name is None:
            raise ValueError("collection_name must be set")
        
        self.collection_dim = kwargs.get("collection_dim", 1536)
        
        self.model_embedding_type = kwargs.get("model_embedding_type", "openai")
        self.model_embedding_name = kwargs.get("model_embedding_name", "text-embedding-3-small")
        self.model_embedding_url = kwargs.get("model_embedding_url", None)
        
        if self.model_embedding_type is not None and self.model_embedding_name is not None:
            self.model_embedding = self.get_embedding_model()

    def get_embedding_model_vs(self) -> Any:
        """
        Get the language model_embedding_name to use for generating text.

        Returns:
            Any: The language model_embedding_name to use.
        Raises:
            ValueError: If the model_embedding_type or model_embedding_name is not set.
            Exception: If there is an error during the loading of the embedding model.
        """
        try:

            if self.model_embedding_vs_type is None:
                raise ValueError("model_embedding_vs_type must be set")

            if self.model_embedding_vs_name is None:
                raise ValueError("model_embedding_vs_name must be set")

            if self.model_embedding_type.lower() == 'local':
                
                if self.model_embedding_vs_path is None:
                    print("model_embedding_path not set, using default local model path")
                    raise ValueError("model_embedding_path must be set")
                
                TextEmbedding.add_custom_model(
                    model=self.model_embedding_name,
                    pooling=PoolingType.MEAN,
                    normalization=True,
                    sources=ModelSource(hf=self.model_embedding_name),
                    dim=384,
                    model_file=self.model_embedding_vs_path,
                )
                
                return TextEmbedding(model=self.model_embedding_name)
            
            elif self.model_embedding_vs_type.lower() == 'hf':
                return TextEmbedding(model=self.model_embedding_vs_name)
        except Exception as e:
            print(f"Errore durante il caricamento del modello di embedding per il database vettoriale: {e}")
            raise e

    def get_embedding_model(self):
        """
        Get the language model_embedding_name to use for generating text.

        Returns:
            Any: The language model_embedding_name to use.
        Raises:
            ValueError: If the model_embedding_type or model_embedding_name is not set.
            Exception: If there is an error during the loading of the embedding model.
        """
        try:

            if self.model_embedding_type is None:
                print("model_embedding_type must be set")
                raise ValueError("model_embedding_type must be set")

            if self.model_embedding_name is None:
                print("model_embedding_name must be set")
                raise ValueError("model_embedding_name must be set")

            if self.model_embedding_type.lower() == 'openai':
                print("Using OpenAI embeddings")
                return OpenAIEmbeddings(model=self.model_embedding_name)
            elif self.model_embedding_type.lower() == 'ollama':
                print("Using Ollama embeddings")
                if self.model_embedding_url is None:
                    print("model_embedding_url not set, using default Ollama base URL")
                    raise ValueError("model_embedding_url must be set")
                return OllamaEmbeddings(model=self.model_embedding_name, 
                                        base_url=self.model_embedding_url)
            elif self.model_embedding_type.lower() == 'vllm':
                print("Using VLLM embeddings")
                if self.model_embedding_url is None:
                    print("model_embedding_url not set, using default VLLM base URL")
                    raise ValueError("model_embedding_url must be set")
                return OpenAIEmbeddings(model=self.model_embedding_name, base_url=self.model_embedding_url, tiktoken_enabled=False)
        except Exception as e:
            print(f"Errore durante il caricamento del modello di embedding: {e}")
            raise e

    def get_in_memory_store(self):
        """
        Get the in-memory store.

        Returns:
            InMemoryStore: The in-memory store.
        """
        
        openai_embeddings = self.get_embedding_model()
        
        if not (
            isinstance(openai_embeddings, OpenAIEmbeddings) or
            (OllamaEmbeddings and isinstance(openai_embeddings, OllamaEmbeddings))
        ):
            raise ValueError("Embedding model must be an instance of OpenAIEmbeddings or OllamaEmbeddings")
        
        config : IndexConfig | None = {
            "embed": openai_embeddings,
            "dims": self.collection_dim,
        } 
        
        return InMemoryStore(index=config)

    @abstractmethod
    async def get_vector_store(self, collection : str | None = None) -> QdrantVectorStore:
        """ Get or create a vector store for the specified collection.
        Returns:
            Any: The vector store for the specified collection.
        """
        pass

    @abstractmethod
    async def search_filter_async(self, query : str, metadata_value: str, collection : str | None = None) -> list[Document]:
        """
        Get the filter conditions for the search.
        Args:
            query (str): The search query.
            metadata_value (str): The value to match in the metadata field.
        Returns:
            list: A list of filter conditions for the search.
        """
        pass

    @abstractmethod
    async def save_async(self, last_message: str, thread: str | None = None, custom_metadata : dict[str, Any] | None = None):
        """
        Save the last message to the vector store.
        Args:
            last_message (str): The last message content.
            thread_id (str, optional): The thread ID to associate with the message.
            custom_metadata (dict[str, Any], optional): Custom metadata to associate with the message.
        """
        pass