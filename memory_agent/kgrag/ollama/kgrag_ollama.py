from .memory_ollama import MemoryOllama


class KGragOllama(MemoryOllama):
    """
    KGragGraphOllama is a subclass of KGragGraph that uses the Ollama API
    for natural language processing tasks.
    """
    def __init__(self, **kwargs):
        """
        Initialize the KGragGraphOllama with the provided parameters.
        """
        super().__init__(**kwargs)
        llm_config_default = {
            "model": "llama3.1",
            "model_provider": "ollama",
            "api_key": None,
            "base_url": "http://localhost:11434",
            "temperature": 0.5,
        }
        self.llm_config = kwargs.get("llm_config", llm_config_default)

    def embed_query(self, query: str) -> list[float]:
        """
        Get the embedding for a given query.
        Args:
            query (str): The query to be embedded.
        """
        return self.model_embedding.embed_query(query)

    def embeddings(
        self,
        raw_data
    ) -> list:
        """
        Get embeddings for the provided raw data using the Ollama model.
        """
        embeddings = [
                self.embed_query(paragraph)
                for paragraph in raw_data.split("\n")
            ]
        return embeddings
