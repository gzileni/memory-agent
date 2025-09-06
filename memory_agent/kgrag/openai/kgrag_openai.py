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
