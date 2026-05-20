"""
Cross-encoder reranking using FlashRank (CPU-optimised, sub-20ms).
"""
from typing import List, Dict, Any


class ChunkReranker:
    """
    Reranks retrieved chunks using a cross-encoder model.
    Scores (query, chunk) pairs jointly — far more accurate than
    bi-encoder similarity alone.

    First call downloads ms-marco-MiniLM-L-12-v2 (~23 MB) and caches it.
    Subsequent calls: sub-20ms for up to 50 candidates.
    """

    def __init__(self, model_name: str = "ms-marco-MiniLM-L-12-v2"):
        from flashrank import Ranker
        self._ranker = Ranker(model_name=model_name)

    def rerank(self, query: str, chunks: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Rerank chunks by relevance to query and return the top_k best.

        Args:
            query: The user query string.
            chunks: List of chunk dicts, each must have an "id" and "text" key.
            top_k: Number of chunks to return after reranking.

        Returns:
            Top-k chunks ordered by cross-encoder score (highest first).
        """
        if not chunks:
            return chunks

        from flashrank import RerankRequest

        passages = [{"id": c["id"], "text": c.get("text", "")} for c in chunks]
        result = self._ranker.rerank(RerankRequest(query=query, passages=passages))

        id_to_chunk = {c["id"]: c for c in chunks}
        reranked = [id_to_chunk[r["id"]] for r in result if r["id"] in id_to_chunk]
        return reranked[:top_k]
