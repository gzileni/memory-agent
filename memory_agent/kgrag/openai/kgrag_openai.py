import os
from .memory_openai import MemoryOpenAI


class KGragOpenAI(MemoryOpenAI):
    """
    KGragGraphOpenAI is a subclass of KGragGraph that uses the OpenAI API
    for natural language processing tasks.
    """

    def __init__(self, **kwargs):
        """
        Initialize the KGragGraphOpenAI with the provided parameters.
        """
        super().__init__(**kwargs)
        openai_config_default = {
            "model": "gpt-4.1-mini",
            "model_provider": "openai",
            "api_key": os.getenv("OPENAI_API_KEY"),
            "base_url": None,
            "temperature": 0.5,
        }
        self.llm_config = kwargs.get("llm_config", openai_config_default)

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
        Get embeddings for the provided raw data using the OpenAI model.
        """
        embeddings = [
                self.embed_query(paragraph)
                for paragraph in raw_data.split("\n")
            ]
        return embeddings
