"""
Retriever for finding relevant chunks from the vector store.
"""
import time
from typing import List, Dict, Any, Optional
from .vector_store import VectorStore
from ..ingestion.embedder import Embedder
from ..config import config

# Expected output dimension for each known embedding model.
_KNOWN_DIMENSIONS: dict = {
    "nomic-embed-text-v1.5": 768,
    "text-embedding-mxbai-embed-large-v1": 1024,
}


class Retriever:
    """
    Retrieves relevant chunks for a query.
    """

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        embedder: Optional[Embedder] = None
    ):
        self.vector_store = vector_store or VectorStore()
        self.embedder = embedder or Embedder()
        self.top_k = config.top_k
        self.similarity_threshold = config.similarity_threshold
        self._check_embedding_dimension()
        self._check_parent_child_consistency()

    def _check_parent_child_consistency(self) -> None:
        """
        Warn if use_parent_child_chunking is enabled but the store contains no
        chunks with a parent_id field — meaning the data was ingested without the
        flag and parent-child promotion will silently fall back to child text.
        """
        if not config.use_parent_child_chunking:
            return
        results = self.vector_store._collection.get(limit=5, include=["metadatas"])
        metadatas = results.get("metadatas") or []
        if not metadatas:
            return  # empty store — nothing to validate
        # Use all() so a partially re-ingested store (mixed old/new chunks) is
        # correctly flagged instead of silently passing because one chunk has parent_id.
        has_parent_id = all(
            (m or {}).get("parent_id") is not None for m in metadatas
        )
        if not has_parent_id:
            print(
                "\n[WARNING] use_parent_child_chunking is enabled but stored chunks "
                "have no parent_id.\n"
                "  ACTION REQUIRED: Run 'Re-ingest All' to rebuild the index "
                "with parent-child chunking.\n"
            )

    def _check_embedding_dimension(self) -> None:
        """
        Warn if stored embeddings were created with a different model dimension.

        This detects the case where the embedding_model config was changed but
        the vector store still holds vectors from the old model. Mixing
        dimensions causes ChromaDB errors; mismatched-but-same-dimension models
        produce silently wrong results. The user must run Re-ingest All to fix.
        """
        stored_dim = self.vector_store.get_stored_embedding_dimension()
        if stored_dim is None:
            return  # empty store — fine

        expected_dim = _KNOWN_DIMENSIONS.get(self.embedder.model)
        if expected_dim is not None and stored_dim != expected_dim:
            print(
                f"\n[WARNING] Embedding dimension mismatch detected!\n"
                f"  Stored embeddings: {stored_dim}-dim "
                f"(from a different model)\n"
                f"  Current model '{self.embedder.model}': {expected_dim}-dim\n"
                f"  ACTION REQUIRED: Run 'Re-ingest All' to rebuild the index "
                f"with the new embedding model before querying.\n"
            )

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_categories: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: User's question
            top_k: Number of chunks to retrieve
            filter_categories: Optional list of categories to filter by

        Returns:
            List of relevant chunk dicts with text, metadata, and distance
        """
        top_k = top_k or self.top_k

        # Generate query embedding
        t0 = time.time()
        query_embedding = self.embedder.embed_query(query)
        embed_time = round(time.time() - t0, 2)
        print(f"  [Retriever] Embedding query took {embed_time}s")

        # Build filter if categories specified
        where_filter = None
        if filter_categories:
            # ChromaDB uses $contains for substring matching
            where_filter = {
                "$or": [
                    {"categories": {"$contains": cat}}
                    for cat in filter_categories
                ]
            }

        # Search vector store
        t0 = time.time()
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            where=where_filter
        )
        search_time = round(time.time() - t0, 2)
        print(f"  [Retriever] Vector search took {search_time}s - got {len(results)} results")

        # Return all results - let the LLM decide relevance
        # ChromaDB already returns sorted by distance (lower = more similar)
        return results

    def retrieve_with_scores(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve chunks with similarity scores.

        Args:
            query: User's question
            top_k: Number of chunks to retrieve

        Returns:
            List of chunks with added 'score' field (0-1, higher is better)
        """
        results = self.retrieve(query, top_k)

        # Convert distance to similarity score
        for r in results:
            # L2 distance to similarity: 1 / (1 + distance)
            distance = r.get("distance", 0)
            r["score"] = 1 / (1 + distance)

        return results

    def get_sources(self, results: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Extract unique source information from results.

        Args:
            results: Retrieved chunk results

        Returns:
            List of unique sources with title, author, url
        """
        seen_urls = set()
        sources = []

        for r in results:
            metadata = r.get("metadata", {})
            url = metadata.get("url", "")

            if url and url not in seen_urls:
                seen_urls.add(url)
                sources.append({
                    "title": metadata.get("title", "Unknown"),
                    "author": metadata.get("author", "Unknown"),
                    "url": url,
                    "published_date": metadata.get("published_date", "")
                })

        return sources
