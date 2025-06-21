import os
from typing import List
import google.generativeai as genai

class EmbeddingClient:
    """
    A client for generating text embeddings using Google's Gemini models.

    This client requires the `google-generativeai` package to be installed
    (`pip install google-generativeai`) and the `GOOGLE_API_KEY` environment
    variable to be set with a valid API key.
    """

    def __init__(self, api_key: str = None, model: str = "models/embedding-001"):
        """
        Initializes the EmbeddingClient.

        Args:
            api_key (str, optional): The Google API key. If not provided, it will
                                     be read from the GOOGLE_API_KEY environment variable.
            model (str, optional): The name of the embedding model to use.
                                   Defaults to "models/embedding-001". For applications
                                   that are highly sensitive to latency, you might explore
                                   older, smaller models.
        """
        if api_key is None:
            api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Google API key not found. Please set the GOOGLE_API_KEY environment variable.")

        genai.configure(api_key=api_key)
        self.model = model

    def get_embedding(self, text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> List[float]:
        """
        Generates an embedding for a single piece of text.

        Args:
            text (str): The text to embed.
            task_type (str): The intended task for the embedding.
                             Options: "RETRIEVAL_QUERY", "RETRIEVAL_DOCUMENT",
                             "SEMANTIC_SIMILARITY", "CLASSIFICATION", "CLUSTERING".
                             Defaults to "RETRIEVAL_DOCUMENT" for storing memories.

        Returns:
            List[float]: The embedding vector for the text.

        Raises:
            Exception: If there is an error calling the API.
        """
        try:
            result = genai.embed_content(
                model=self.model,
                content=text,
                task_type=task_type
            )
            return result['embedding']
        except Exception as e:
            print(f"An error occurred while generating the embedding: {e}")
            raise

    def get_embeddings(self, texts: List[str], task_type: str = "RETRIEVAL_DOCUMENT") -> List[List[float]]:
        """
        Generates embeddings for a batch of texts.

        Args:
            texts (List[str]): A list of texts to embed.
            task_type (str): The intended task for the embeddings. Defaults to "RETRIEVAL_DOCUMENT".

        Returns:
            List[List[float]]: A list of embedding vectors, one for each text.
        """
        try:
            result = genai.embed_content(
                model=self.model,
                content=texts,
                task_type=task_type
            )
            return result['embedding']
        except Exception as e:
            print(f"An error occurred while generating the embeddings: {e}")
            raise 