"""
Retriever for finding relevant chunks from the vector store.
"""
import time
from typing import List, Dict, Any, Optional
from .vector_store import VectorStore
from ..ingestion.embedder import Embedder
from ..config import config


class Retriever:
    """
    Retrieves relevant chunks for a query.
    """

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        embedder: Optional[Embedder] = None
    ):
        """
        Initialize the retriever.

        Args:
            vector_store: VectorStore instance
            embedder: Embedder instance
        """
        self.vector_store = vector_store or VectorStore()
        self.embedder = embedder or Embedder()
        self.top_k = config.top_k
        self.similarity_threshold = config.similarity_threshold

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
