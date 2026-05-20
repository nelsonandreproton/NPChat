"""
ChromaDB vector store operations.
"""
from typing import List, Dict, Any, Optional, TYPE_CHECKING
import chromadb
from chromadb.config import Settings
from ..config import config

if TYPE_CHECKING:
    from ..ingestion.chunker import Chunk


class VectorStore:
    """
    ChromaDB vector store for blog post chunks.
    """

    def __init__(self, collection_name: str = None, persist_directory: str = None):
        """
        Initialize vector store.

        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Path to persist the database
        """
        self.collection_name = collection_name or config.collection_name
        self.persist_directory = persist_directory or str(config.chroma_db_path)

        # Initialize ChromaDB client with persistence
        self._client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )

        # Get or create collection
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Near Partner blog post chunks"}
        )

    def add_chunks(
        self,
        chunks: List["Chunk"],
        embeddings: List[List[float]]
    ) -> int:
        """
        Upsert chunks with their embeddings into the store.

        Uses upsert so re-ingesting the same content is idempotent and edited
        content overwrites stale entries (same content_hash → same ID).

        ID scheme: ``<url_safe>_<content_hash>`` — stable across re-ingests as
        long as the text is unchanged; changes when content changes.

        Returns:
            Number of chunks upserted
        """
        if not chunks:
            return 0

        ids = []
        documents = []
        metadatas = []

        for chunk in chunks:
            url_safe = chunk.metadata.get("url", "unknown").replace("/", "_").replace(":", "_")
            content_hash = chunk.metadata.get("content_hash", str(chunk.chunk_index))
            chunk_id = f"{url_safe}_{content_hash}"

            ids.append(chunk_id)
            documents.append(chunk.text)

            metadata = chunk.metadata.copy()
            if isinstance(metadata.get("categories"), list):
                metadata["categories"] = "|".join(metadata["categories"])
            metadatas.append(metadata)

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

        return len(chunks)

    def get_existing_hashes_for_url(self, url: str) -> set:
        """
        Return the set of content_hash values already stored for a given URL.

        Used by the ingest pipeline to skip embedding chunks that haven't changed.
        """
        results = self._collection.get(
            where={"url": url},
            include=["metadatas"]
        )
        hashes = set()
        for meta in (results.get("metadatas") or []):
            h = (meta or {}).get("content_hash")
            if h:
                hashes.add(h)
        return hashes

    def get_existing_source_hashes_for_url(self, url: str) -> set:
        """
        Return the set of source_hash values already stored for a given URL.

        source_hash is the plain-text chunk hash — stable regardless of whether
        contextual retrieval is on or off. Used as the dedup key so that toggling
        contextual retrieval does not force a re-embed of unchanged content.
        Falls back to content_hash for chunks ingested before source_hash was introduced.
        """
        results = self._collection.get(
            where={"url": url},
            include=["metadatas"]
        )
        hashes = set()
        for meta in (results.get("metadatas") or []):
            meta = meta or {}
            h = meta.get("source_hash") or meta.get("content_hash")
            if h:
                hashes.add(h)
        return hashes

    def search(
        self,
        query_embedding: List[float],
        top_k: int = None,
        where: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            where: Optional metadata filter

        Returns:
            List of results with text, metadata, and distance
        """
        top_k = top_k or config.top_k

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"]
        )

        # Format results
        formatted = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                formatted.append({
                    "text": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else 0,
                    "id": results["ids"][0][i] if results["ids"] else ""
                })

        return formatted

    def get_all_urls(self) -> List[str]:
        """
        Get all unique URLs in the store.

        Returns:
            List of unique blog post URLs
        """
        # Get all items with just metadata
        all_items = self._collection.get(include=["metadatas"])

        urls = set()
        if all_items["metadatas"]:
            for metadata in all_items["metadatas"]:
                if metadata and "url" in metadata:
                    urls.add(metadata["url"])

        return list(urls)

    def delete_by_url(self, url: str) -> int:
        """
        Delete all chunks from a specific URL.

        Args:
            url: Blog post URL

        Returns:
            Number of chunks deleted
        """
        # Get IDs matching this URL
        results = self._collection.get(
            where={"url": url},
            include=[]
        )

        if results["ids"]:
            self._collection.delete(ids=results["ids"])
            return len(results["ids"])

        return 0

    def count(self) -> int:
        """Get total number of chunks in the store."""
        return self._collection.count()

    def get_stored_embedding_dimension(self) -> Optional[int]:
        """
        Return the dimension of stored embeddings by sampling one entry.
        Returns None if the store is empty.
        """
        results = self._collection.get(limit=1, include=["embeddings"])
        embeddings = results.get("embeddings")
        if embeddings is not None and len(embeddings) > 0 and embeddings[0] is not None:
            return len(embeddings[0])
        return None

    def clear(self):
        """Delete all chunks from the store."""
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Near Partner blog post chunks"}
        )
