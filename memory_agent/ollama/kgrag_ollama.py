from .memory_ollama import MemoryOllama
from .agent_ollama import AgentOllama
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSerializable


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

    def chain(self, prompt: ChatPromptTemplate) -> RunnableSerializable:
        """
        Get the chain for the Ollama agent.
        Args:
            prompt (ChatPromptTemplate): The prompt to use for the chain.
        """
        return self.ollama_agent.chain(prompt)

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
