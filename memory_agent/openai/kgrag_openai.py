from .agent_openai import AgentOpenAI
from .memory_openai import MemoryOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSerializable


class KGragOpenAI(MemoryOpenAI):
    """
    KGragGraphOpenAI is a subclass of KGragGraph that uses the OpenAI API
    for natural language processing tasks.
    """

    openai_agent: AgentOpenAI

    def __init__(self, **kwargs):
        """
        Initialize the KGragGraphOpenAI with the provided parameters.
        """
        super().__init__(**kwargs)
        self.openai_agent = AgentOpenAI(**kwargs)

    def chain(self, prompt: ChatPromptTemplate) -> RunnableSerializable:
        """
        Get the chain for the OpenAI agent.
        Args:
            prompt (ChatPromptTemplate): The prompt to use for the chain.
        """
        return self.openai_agent.chain(prompt)

    def embeddings(
        self,
        raw_data
    ) -> list:
        """
        Get embeddings for the provided raw data using the OpenAI model.
        """
        embeddings = [
                self.model_embedding.embed_query(paragraph)
                for paragraph in raw_data.split("\n")
            ]
        return embeddings
