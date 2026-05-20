"""Tests for content-hash deduplication in chunker and ingest pipeline."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import hashlib
import pytest
from unittest.mock import MagicMock, patch, call


# ---------------------------------------------------------------------------
# Chunker: content_hash stamped on metadata
# ---------------------------------------------------------------------------

class TestChunkerContentHash:
    def _make_post(self, content: str) -> dict:
        return {"url": "http://example.com/post", "title": "T", "content": content}

    def test_chunk_has_content_hash(self):
        from src.ingestion.chunker import TextChunker
        chunker = TextChunker(chunk_size=500, chunk_overlap=50)
        chunks = chunker.chunk_blog_post(self._make_post("Hello world\n\nSecond paragraph here."))
        assert all("content_hash" in c.metadata for c in chunks)

    def test_same_text_same_hash(self):
        from src.ingestion.chunker import TextChunker
        chunker = TextChunker(chunk_size=500, chunk_overlap=50)
        p = self._make_post("Hello world\n\nSecond paragraph here.")
        chunks_a = chunker.chunk_blog_post(p)
        chunks_b = chunker.chunk_blog_post(p)
        hashes_a = [c.metadata["content_hash"] for c in chunks_a]
        hashes_b = [c.metadata["content_hash"] for c in chunks_b]
        assert hashes_a == hashes_b

    def test_different_text_different_hash(self):
        from src.ingestion.chunker import TextChunker
        chunker = TextChunker(chunk_size=500, chunk_overlap=50)
        c1 = chunker.chunk_blog_post(self._make_post("Content A\n\nMore A."))
        c2 = chunker.chunk_blog_post(self._make_post("Content B\n\nMore B."))
        hashes_a = {c.metadata["content_hash"] for c in c1}
        hashes_b = {c.metadata["content_hash"] for c in c2}
        assert hashes_a.isdisjoint(hashes_b)

    def test_hash_is_16_hex_chars(self):
        from src.ingestion.chunker import TextChunker
        chunker = TextChunker(chunk_size=500, chunk_overlap=50)
        chunks = chunker.chunk_blog_post(self._make_post("Some content."))
        for c in chunks:
            h = c.metadata["content_hash"]
            assert len(h) == 16
            assert all(ch in "0123456789abcdef" for ch in h)


# ---------------------------------------------------------------------------
# VectorStore: upsert + get_existing_hashes_for_url
# ---------------------------------------------------------------------------

class TestVectorStoreHashMethods:
    def _make_chunk(self, url: str, idx: int, text: str):
        from src.ingestion.chunker import Chunk, _content_hash
        return Chunk(
            text=text,
            metadata={"url": url, "chunk_index": idx, "content_hash": _content_hash(text)},
            chunk_index=idx,
        )

    def _make_store(self):
        with patch("src.retrieval.vector_store.chromadb.PersistentClient") as mock_client:
            mock_collection = MagicMock()
            mock_client.return_value.get_or_create_collection.return_value = mock_collection
            from src.retrieval.vector_store import VectorStore
            store = VectorStore.__new__(VectorStore)
            store._collection = mock_collection
            return store, mock_collection

    def test_add_chunks_calls_upsert_not_add(self):
        store, coll = self._make_store()
        chunk = self._make_chunk("http://x.com/1", 0, "hello world")
        store.add_chunks([chunk], [[0.1, 0.2]])
        coll.upsert.assert_called_once()
        coll.add.assert_not_called()

    def test_chunk_id_uses_hash(self):
        store, coll = self._make_store()
        chunk = self._make_chunk("http://x.com/1", 0, "hello world")
        store.add_chunks([chunk], [[0.1, 0.2]])
        ids = coll.upsert.call_args[1]["ids"]
        assert chunk.metadata["content_hash"] in ids[0]

    def test_get_existing_hashes_for_url(self):
        store, coll = self._make_store()
        coll.get.return_value = {
            "metadatas": [
                {"url": "http://x.com/1", "content_hash": "abc123"},
                {"url": "http://x.com/1", "content_hash": "def456"},
            ]
        }
        hashes = store.get_existing_hashes_for_url("http://x.com/1")
        assert hashes == {"abc123", "def456"}

    def test_get_existing_hashes_empty(self):
        store, coll = self._make_store()
        coll.get.return_value = {"metadatas": []}
        assert store.get_existing_hashes_for_url("http://unknown.com") == set()


# ---------------------------------------------------------------------------
# IngestPipeline: stale deletion + skip unchanged
# ---------------------------------------------------------------------------

class TestIngestPipelineDedup:
    def _make_pipeline(self):
        mock_chunker = MagicMock()
        mock_embedder = MagicMock()
        mock_store = MagicMock()
        mock_store.add_chunks.return_value = 0

        with patch("src.ingestion.ingest.TextChunker", return_value=mock_chunker), \
             patch("src.ingestion.ingest.Embedder", return_value=mock_embedder), \
             patch("src.ingestion.ingest.VectorStore", return_value=mock_store):
            from src.ingestion.ingest import IngestPipeline
            pipeline = IngestPipeline(
                chunker=mock_chunker,
                embedder=mock_embedder,
                vector_store=mock_store,
            )
        return pipeline, mock_chunker, mock_embedder, mock_store

    def _chunk(self, url: str, idx: int, text: str):
        from src.ingestion.chunker import Chunk
        source_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        return Chunk(
            text=text,
            metadata={"url": url, "chunk_index": idx, "source_hash": source_hash, "content_hash": source_hash},
            chunk_index=idx,
        )

    def test_unchanged_chunks_skipped(self):
        pipeline, chunker, embedder, store = self._make_pipeline()
        chunk = self._chunk("http://x.com/1", 0, "unchanged text")
        chunker.chunk_all_posts.return_value = [chunk]
        store.get_existing_source_hashes_for_url.return_value = {chunk.metadata["source_hash"]}

        result = pipeline.ingest_posts([{"url": "http://x.com/1"}], show_progress=False)

        embedder.embed_texts.assert_not_called()
        assert result["chunks"] == 0

    def test_new_chunk_gets_embedded(self):
        pipeline, chunker, embedder, store = self._make_pipeline()
        chunk = self._chunk("http://x.com/1", 0, "brand new content")
        chunker.chunk_all_posts.return_value = [chunk]
        store.get_existing_source_hashes_for_url.return_value = set()  # nothing stored yet
        embedder.embed_texts.return_value = [[0.1, 0.2]]
        store.add_chunks.return_value = 1

        result = pipeline.ingest_posts([{"url": "http://x.com/1"}], show_progress=False)

        embedder.embed_texts.assert_called_once()
        assert result["chunks"] == 1

    def test_stale_chunks_deleted_when_content_shrinks(self):
        pipeline, chunker, embedder, store = self._make_pipeline()
        # Post now produces 1 chunk; store has 2 (one was deleted from source)
        chunk_new = self._chunk("http://x.com/1", 0, "surviving paragraph")
        chunker.chunk_all_posts.return_value = [chunk_new]

        old_hash = "deadbeef12345678"  # stale hash still in store
        store.get_existing_source_hashes_for_url.return_value = {old_hash, chunk_new.metadata["source_hash"]}
        embedder.embed_texts.return_value = [[0.1]]
        store.add_chunks.return_value = 1
        store.delete_by_url.return_value = 2

        pipeline.ingest_posts([{"url": "http://x.com/1"}], show_progress=False)

        store.delete_by_url.assert_called_once_with("http://x.com/1")

    def test_no_deletion_when_no_stale_chunks(self):
        pipeline, chunker, embedder, store = self._make_pipeline()
        chunk = self._chunk("http://x.com/1", 0, "text")
        chunker.chunk_all_posts.return_value = [chunk]
        store.get_existing_source_hashes_for_url.return_value = {chunk.metadata["source_hash"]}

        pipeline.ingest_posts([{"url": "http://x.com/1"}], show_progress=False)

        store.delete_by_url.assert_not_called()

    def test_get_new_posts_skips_truly_unchanged(self):
        pipeline, chunker, embedder, store = self._make_pipeline()
        post = {"url": "http://x.com/1", "title": "T", "content": "C"}
        store.get_all_urls.return_value = ["http://x.com/1"]

        chunk = self._chunk("http://x.com/1", 0, "C")
        chunker.chunk_blog_post.return_value = [chunk]
        store.get_existing_source_hashes_for_url.return_value = {chunk.metadata["source_hash"]}

        result = pipeline.get_new_posts([post])
        assert result == []

    def test_get_new_posts_includes_changed_post(self):
        pipeline, chunker, embedder, store = self._make_pipeline()
        post = {"url": "http://x.com/1", "title": "T", "content": "new content"}
        store.get_all_urls.return_value = ["http://x.com/1"]

        chunk = self._chunk("http://x.com/1", 0, "new content")
        chunker.chunk_blog_post.return_value = [chunk]
        store.get_existing_source_hashes_for_url.return_value = {"oldhash000000000"}  # different

        result = pipeline.get_new_posts([post])
        assert result == [post]
