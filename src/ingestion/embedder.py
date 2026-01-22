"""
Embedding generation using Ollama's nomic-embed-text model.
"""
from typing import List
import ollama
from ..config import config


class Embedder:
    """
    Generates embeddings using Ollama.
    """

    def __init__(self, model: str = None):
        """
        Initialize embedder.

        Args:
            model: Ollama embedding model name
        """
        self.model = model or config.embedding_model
        self._client = ollama.Client(host=config.ollama_base_url)

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        response = self._client.embeddings(
            model=self.model,
            prompt=text
        )
        return response["embedding"]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        embeddings = []
        for text in texts:
            embedding = self.embed_text(text)
            embeddings.append(embedding)
        return embeddings

    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a search query.
        Same as embed_text but semantically named for clarity.

        Args:
            query: Search query

        Returns:
            Embedding vector
        """
        return self.embed_text(query)
