"""
Hybrid retriever combining semantic search with BM25 keyword search.
"""
import re
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from .vector_store import VectorStore
from ..ingestion.embedder import Embedder
from ..config import config


class HybridRetriever:
    """
    Combines semantic (embedding) search with BM25 keyword search.

    This catches both:
    - Conceptually similar content (semantic)
    - Exact keyword matches (BM25)
    """

    def __init__(
        self,
        vector_store: VectorStore = None,
        embedder: Embedder = None,
        semantic_weight: float = 0.7,
        bm25_weight: float = 0.3
    ):
        """
        Initialize hybrid retriever.

        Args:
            vector_store: Vector store instance
            embedder: Embedder instance
            semantic_weight: Weight for semantic search scores (0-1)
            bm25_weight: Weight for BM25 scores (0-1)
        """
        self.vector_store = vector_store or VectorStore()
        self.embedder = embedder or Embedder()
        self.semantic_weight = semantic_weight
        self.bm25_weight = bm25_weight

        # BM25 index (built lazily)
        self._bm25 = None
        self._corpus_docs = None
        self._corpus_metadata = None

    def _build_bm25_index(self):
        """Build BM25 index from all documents in vector store."""
        print("[HybridRetriever] Building BM25 index...")

        # Get all documents from ChromaDB
        collection = self.vector_store._collection
        all_data = collection.get(include=["documents", "metadatas"])

        if not all_data["documents"]:
            print("[HybridRetriever] No documents found")
            return

        self._corpus_docs = all_data["documents"]
        self._corpus_metadata = all_data["metadatas"]
        self._corpus_ids = all_data["ids"]

        # Tokenize documents for BM25
        tokenized_corpus = [self._tokenize(doc) for doc in self._corpus_docs]
        self._bm25 = BM25Okapi(tokenized_corpus)

        print(f"[HybridRetriever] BM25 index built with {len(self._corpus_docs)} documents")

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25."""
        # Lowercase and split on non-alphanumeric
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to 0-1 range."""
        if not scores:
            return scores

        min_s = min(scores)
        max_s = max(scores)

        if max_s == min_s:
            return [0.5] * len(scores)

        return [(s - min_s) / (max_s - min_s) for s in scores]

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        use_expansion: bool = False,
        expanded_query: str = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents using hybrid search.

        Args:
            query: User query
            top_k: Number of results to return
            use_expansion: Whether query expansion was used
            expanded_query: The expanded query (if used)

        Returns:
            List of results with combined scores
        """
        top_k = top_k or config.top_k

        # Build BM25 index if needed
        if self._bm25 is None:
            self._build_bm25_index()

        if self._bm25 is None:
            # Fallback to semantic only
            return self._semantic_search(query, top_k)

        # Get semantic results (fetch more for merging)
        semantic_results = self._semantic_search(query, top_k * 2)

        # Get BM25 results
        bm25_query = expanded_query if expanded_query else query
        bm25_results = self._bm25_search(bm25_query, top_k * 2)

        # Merge results using Reciprocal Rank Fusion (RRF)
        merged = self._reciprocal_rank_fusion(
            semantic_results,
            bm25_results,
            top_k
        )

        return merged

    def _semantic_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Perform semantic search."""
        # Embed query
        query_embedding = self.embedder.embed_query(query)

        # Search vector store
        results = self.vector_store.search(query_embedding, top_k=top_k)

        # Convert distance to similarity score (lower distance = higher similarity)
        for r in results:
            # ChromaDB uses L2 distance, convert to similarity
            r["semantic_score"] = 1 / (1 + r.get("distance", 0))

        return results

    def _bm25_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Perform BM25 keyword search."""
        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        # Get top results
        scored_docs = list(zip(range(len(scores)), scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        top_docs = scored_docs[:top_k]

        results = []
        for idx, score in top_docs:
            if score > 0:  # Only include matches
                results.append({
                    "text": self._corpus_docs[idx],
                    "metadata": self._corpus_metadata[idx] if self._corpus_metadata else {},
                    "id": self._corpus_ids[idx] if self._corpus_ids else str(idx),
                    "bm25_score": score
                })

        return results

    def _reciprocal_rank_fusion(
        self,
        semantic_results: List[Dict],
        bm25_results: List[Dict],
        top_k: int,
        k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Merge results using Reciprocal Rank Fusion (RRF).

        RRF is a simple but effective method to combine ranked lists.
        Score = sum(1 / (k + rank)) for each list where the doc appears.
        """
        doc_scores = {}

        # Score from semantic results
        for rank, result in enumerate(semantic_results):
            doc_id = result.get("id", result.get("text", "")[:50])
            rrf_score = self.semantic_weight / (k + rank + 1)

            if doc_id not in doc_scores:
                doc_scores[doc_id] = {
                    "result": result,
                    "rrf_score": 0,
                    "semantic_rank": None,
                    "bm25_rank": None
                }

            doc_scores[doc_id]["rrf_score"] += rrf_score
            doc_scores[doc_id]["semantic_rank"] = rank + 1

        # Score from BM25 results
        for rank, result in enumerate(bm25_results):
            doc_id = result.get("id", result.get("text", "")[:50])
            rrf_score = self.bm25_weight / (k + rank + 1)

            if doc_id not in doc_scores:
                doc_scores[doc_id] = {
                    "result": result,
                    "rrf_score": 0,
                    "semantic_rank": None,
                    "bm25_rank": None
                }

            doc_scores[doc_id]["rrf_score"] += rrf_score
            doc_scores[doc_id]["bm25_rank"] = rank + 1

        # Sort by RRF score and return top_k
        sorted_docs = sorted(
            doc_scores.values(),
            key=lambda x: x["rrf_score"],
            reverse=True
        )[:top_k]

        # Format results
        final_results = []
        for doc in sorted_docs:
            result = doc["result"].copy()
            result["combined_score"] = doc["rrf_score"]
            result["semantic_rank"] = doc["semantic_rank"]
            result["bm25_rank"] = doc["bm25_rank"]
            final_results.append(result)

        return final_results

    def get_retrieval_scores(self, results: List[Dict]) -> List[float]:
        """Extract scores from results for logging."""
        return [r.get("combined_score", r.get("semantic_score", 0)) for r in results]
