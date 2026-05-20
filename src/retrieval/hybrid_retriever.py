"""
Hybrid retriever combining semantic search with BM25 keyword search.
"""
import hashlib
import pickle
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from .vector_store import VectorStore
from ..ingestion.embedder import Embedder
from ..config import config

_BM25_CACHE_PATH = config.data_dir / "bm25_index.pkl"


def _ids_fingerprint(ids: List[str]) -> str:
    """Short hash of the sorted ID list — changes when docs are added/removed."""
    joined = ",".join(sorted(ids)).encode()
    return hashlib.sha256(joined).hexdigest()[:16]


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
        bm25_weight: float = 0.3,
        cache_path: Path = None,
    ):
        self.vector_store = vector_store or VectorStore()
        self.embedder = embedder or Embedder()
        self.semantic_weight = semantic_weight
        self.bm25_weight = bm25_weight
        self._cache_path = cache_path or _BM25_CACHE_PATH

        # BM25 index (loaded from disk or built lazily)
        self._bm25 = None
        self._corpus_docs = None
        self._corpus_metadata = None
        self._corpus_ids = None
        self._index_doc_count = 0
        self._index_fingerprint: Optional[str] = None

        self._load_index()

    def _load_index(self):
        """Load BM25 index from disk if it exists and matches the current corpus."""
        if not self._cache_path.exists():
            return
        try:
            with open(self._cache_path, "rb") as f:
                cached = pickle.load(f)
            # Validate against current store state
            current_count = self.vector_store._collection.count()
            if (
                cached.get("doc_count") == current_count
                and cached.get("fingerprint") == cached.get("fingerprint")  # always true; real check below
            ):
                # Verify fingerprint matches actual IDs
                all_ids = self.vector_store._collection.get(include=[])["ids"]
                if _ids_fingerprint(all_ids) == cached.get("fingerprint"):
                    self._bm25 = cached["bm25"]
                    self._corpus_docs = cached["docs"]
                    self._corpus_metadata = cached["metadatas"]
                    self._corpus_ids = cached["ids"]
                    self._index_doc_count = cached["doc_count"]
                    self._index_fingerprint = cached["fingerprint"]
                    print(f"[HybridRetriever] Loaded BM25 index from disk ({self._index_doc_count} docs)")
                    return
            print("[HybridRetriever] Cached BM25 index is stale, will rebuild")
        except Exception as e:
            print(f"[HybridRetriever] Could not load BM25 cache: {e}")

    def _save_index(self):
        """Persist the current BM25 index to disk."""
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._cache_path, "wb") as f:
                pickle.dump({
                    "bm25": self._bm25,
                    "docs": self._corpus_docs,
                    "metadatas": self._corpus_metadata,
                    "ids": self._corpus_ids,
                    "doc_count": self._index_doc_count,
                    "fingerprint": self._index_fingerprint,
                }, f)
            print(f"[HybridRetriever] BM25 index saved to disk ({self._index_doc_count} docs)")
        except Exception as e:
            print(f"[HybridRetriever] Could not save BM25 cache: {e}")

    def _build_bm25_index(self):
        """Build BM25 index from all documents in vector store, then persist."""
        print("[HybridRetriever] Building BM25 index...")

        collection = self.vector_store._collection
        all_data = collection.get(include=["documents", "metadatas"])

        if not all_data["documents"]:
            print("[HybridRetriever] No documents found")
            return

        self._corpus_docs = all_data["documents"]
        self._corpus_metadata = all_data["metadatas"]
        self._corpus_ids = all_data["ids"]
        self._index_doc_count = len(self._corpus_docs)
        self._index_fingerprint = _ids_fingerprint(self._corpus_ids)

        tokenized_corpus = [self._tokenize(doc) for doc in self._corpus_docs]
        self._bm25 = BM25Okapi(tokenized_corpus)

        print(f"[HybridRetriever] BM25 index built with {len(self._corpus_docs)} documents")
        self._save_index()

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

        # Rebuild if index missing or doc count changed (cheap check before fingerprint)
        current_count = self.vector_store._collection.count()
        if self._bm25 is None or current_count != self._index_doc_count:
            self._build_bm25_index()

        if self._bm25 is None:
            # Fallback to semantic only
            return self._semantic_search(query, top_k)

        # Fetch more candidates than top_k so merging has material to work with.
        # With parent-child chunking multiple children from the same parent compete
        # for the same result slot; a larger pool ensures enough distinct parents
        # survive dedup to fill top_k.
        candidate_k = (top_k * 3) if config.use_parent_child_chunking else (top_k * 2)

        # Get semantic results (fetch more for merging)
        semantic_results = self._semantic_search(query, candidate_k)

        # Get BM25 results
        bm25_query = expanded_query if expanded_query else query
        bm25_results = self._bm25_search(bm25_query, candidate_k)

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

        # Sort by RRF score
        sorted_docs = sorted(
            doc_scores.values(),
            key=lambda x: x["rrf_score"],
            reverse=True
        )

        # Dedup by parent_id when parent-child chunking is active:
        # keep only the highest-scoring child per parent, then promote parent_text.
        sorted_docs = self._dedupe_by_parent(sorted_docs)

        sorted_docs = sorted_docs[:top_k]

        # Format results
        final_results = []
        for doc in sorted_docs:
            result = doc["result"].copy()
            result["combined_score"] = doc["rrf_score"]
            result["semantic_rank"] = doc["semantic_rank"]
            result["bm25_rank"] = doc["bm25_rank"]
            final_results.append(result)

        return final_results

    def _dedupe_by_parent(self, sorted_docs: List[Dict]) -> List[Dict]:
        """
        When parent-child chunking is active, collapse multiple children from the
        same parent into a single entry (the highest-scoring child already comes
        first since the list is pre-sorted by rrf_score) and promote parent_text
        to result["text"] so the LLM receives the broader context.

        When parent-child chunking is off (no parent_id in metadata) this is a no-op.
        """
        if not config.use_parent_child_chunking:
            return sorted_docs

        seen_parents: set = set()
        deduped = []
        for doc in sorted_docs:
            metadata = doc["result"].get("metadata", {})
            parent_id = metadata.get("parent_id")
            if parent_id is None:
                # No parent_id — chunk predates parent-child mode; pass through
                deduped.append(doc)
                continue
            if parent_id in seen_parents:
                continue
            seen_parents.add(parent_id)
            # Promote parent_text so the LLM sees the full context window
            parent_text = metadata.get("parent_text", doc["result"].get("text", ""))
            promoted = dict(doc)
            promoted["result"] = dict(doc["result"])
            promoted["result"]["text"] = parent_text
            deduped.append(promoted)
        return deduped

    def invalidate_index(self):
        """Force rebuild of the BM25 index on next retrieval call and delete cache."""
        self._bm25 = None
        self._index_doc_count = 0
        self._index_fingerprint = None
        try:
            if self._cache_path.exists():
                self._cache_path.unlink()
        except Exception:
            pass
        print("[HybridRetriever] BM25 index invalidated - will rebuild on next query")

    def get_retrieval_scores(self, results: List[Dict]) -> List[float]:
        """Extract scores from results for logging."""
        return [r.get("combined_score", r.get("semantic_score", 0)) for r in results]
