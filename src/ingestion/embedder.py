"""
Embedding generation using sentence-transformers (in-process, no server needed).
"""
from typing import List
from sentence_transformers import SentenceTransformer
from ..config import config

# Config label → HuggingFace repo id
_MODEL_MAP = {
    "mxbai-embed-large-v1": "mixedbread-ai/mxbai-embed-large-v1",
}

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
    Generates embeddings in-process via sentence-transformers.
    The model is loaded once on first instantiation and reused.
    Runs on CPU (no GPU/XPU acceleration on this machine).
    """

    def __init__(self, model: str = None):
        self.model = model or config.embedding_model
        repo = _MODEL_MAP.get(self.model, self.model)
        self._st = SentenceTransformer(repo)

    def embed_text(self, text: str) -> List[float]:
        return self._st.encode(text, normalize_embeddings=True).tolist()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return self._st.encode(
            texts, normalize_embeddings=True, batch_size=16
        ).tolist()

    def embed_query(self, query: str) -> List[float]:
        # Prefix applied here — do NOT also pass prompt/prompt_name to encode().
        # sentence-transformers does not auto-apply the mxbai prompt; doing both
        # would double-prefix and degrade retrieval quality.
        return self.embed_text(_apply_query_prefix(self.model, query))
