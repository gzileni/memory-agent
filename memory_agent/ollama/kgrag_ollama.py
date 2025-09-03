from .memory_ollama import MemoryOllama
from .agent_ollama import AgentOllama


class KGragOllama(MemoryOllama):
    """
    KGragGraphOllama is a subclass of KGragGraph that uses the Ollama API
    for natural language processing tasks.
    """
    ollama_agent: AgentOllama

    def __init__(self, **kwargs):
        """
        Initialize the KGragGraphOllama with the provided parameters.
        """
        super().__init__(**kwargs)
        self.ollama_agent = AgentOllama(**kwargs)

    def embeddings(
        self,
        raw_data
    ) -> list:
        """
        Get embeddings for the provided raw data using the Ollama model.
        """
        embeddings = [
                self.model_embedding.embed_query(paragraph)
                for paragraph in raw_data.split("\n")
            ]
        return embeddings
