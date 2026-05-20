"""
Embedding generation using LM Studio's OpenAI-compatible embeddings API.
"""
from typing import List
from openai import OpenAI
from ..config import config

# mxbai-embed-large-v1 was trained with this prefix on queries only (not documents).
# Prepending it at query time meaningfully improves retrieval accuracy.
# Reference: https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1
_MXBAI_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def _apply_query_prefix(model: str, query: str) -> str:
    if "mxbai" in model.lower():
        return _MXBAI_QUERY_PREFIX + query
    return query


class Embedder:
    """
    Generates embeddings via LM Studio's local server.
    """

    def __init__(self, model: str = None):
        self.model = model or config.embedding_model
        self._client = OpenAI(
            base_url=config.lmstudio_base_url,
            api_key="lm-studio",
        )

    def embed_text(self, text: str) -> List[float]:
        response = self._client.embeddings.create(
            model=self.model,
            input=text,
        )
        return response.data[0].embedding

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            embeddings.append(self.embed_text(text))
        return embeddings

    def embed_query(self, query: str) -> List[float]:
        return self.embed_text(_apply_query_prefix(self.model, query))
