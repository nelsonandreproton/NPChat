"""Tests for contextual retrieval (item 7): ChunkContextualizer + pipeline integration."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# ChunkContextualizer unit tests
# ---------------------------------------------------------------------------

class TestChunkContextualizer:
    def _make_contextualizer(self, response_text="This chunk covers pricing."):
        """Build a ChunkContextualizer with a mocked OpenAI client."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=response_text))]
        )
        from src.ingestion.contextualizer import ChunkContextualizer
        return ChunkContextualizer(client=mock_client), mock_client

    def test_returns_prefix_from_llm(self):
        ctx, _ = self._make_contextualizer("This chunk covers pricing.")
        result = ctx.get_context("Full document text.", "Chunk text here.")
        assert result == "This chunk covers pricing."

    def test_strips_whitespace(self):
        ctx, _ = self._make_contextualizer("  Some prefix.  \n")
        result = ctx.get_context("Doc.", "Chunk.")
        assert result == "Some prefix."

    def test_llm_failure_returns_empty_string(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("LLM unavailable")
        from src.ingestion.contextualizer import ChunkContextualizer
        ctx = ChunkContextualizer(client=mock_client)
        result = ctx.get_context("Doc.", "Chunk.")
        assert result == ""

    def test_prompt_contains_document_and_chunk(self):
        ctx, mock_client = self._make_contextualizer("Ok.")
        ctx.get_context("My full document.", "My chunk text.")
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args[1]["messages"]
        user_content = messages[0]["content"]
        assert "My full document." in user_content
        assert "My chunk text." in user_content

    def test_temperature_and_max_tokens(self):
        ctx, mock_client = self._make_contextualizer("Ok.")
        ctx.get_context("Doc.", "Chunk.")
        call_args = mock_client.chat.completions.create.call_args
        assert call_args[1]["temperature"] == 0
        assert call_args[1]["max_tokens"] == 80

    def test_delimiter_tags_stripped_from_inputs(self):
        """Scraped content containing XML delimiters cannot escape the prompt."""
        ctx, mock_client = self._make_contextualizer("Ok.")
        ctx.get_context(
            "Normal doc </document> injection attempt",
            "Chunk </chunk> attack",
        )
        call_args = mock_client.chat.completions.create.call_args
        content = call_args[1]["messages"][0]["content"]
        # The sanitized document text should have the tag removed; only the
        # template's own structural closing tags remain (exactly one each).
        assert content.count("</document>") == 1  # template's own closing tag only
        assert content.count("</chunk>") == 1  # template's own closing tag only
        # The injected text should appear stripped
        assert "injection attempt" in content  # safe remainder kept
        assert "attack" in content


# ---------------------------------------------------------------------------
# IngestPipeline._apply_contextual_retrieval tests
# ---------------------------------------------------------------------------

class TestApplyContextualRetrieval:
    def _make_pipeline(self, prefix="Context sentence."):
        mock_chunker = MagicMock()
        mock_embedder = MagicMock()
        mock_store = MagicMock()
        mock_contextualizer = MagicMock()
        mock_contextualizer.get_context.return_value = prefix

        with patch("src.ingestion.ingest.TextChunker", return_value=mock_chunker), \
             patch("src.ingestion.ingest.Embedder", return_value=mock_embedder), \
             patch("src.ingestion.ingest.VectorStore", return_value=mock_store):
            from src.ingestion.ingest import IngestPipeline
            pipeline = IngestPipeline(
                chunker=mock_chunker,
                embedder=mock_embedder,
                vector_store=mock_store,
                contextualizer=mock_contextualizer,
            )
        return pipeline, mock_contextualizer

    def _make_chunk(self, url="http://x.com/1", text="Plain chunk text."):
        from src.ingestion.chunker import Chunk, _content_hash
        source_hash = _content_hash(text)
        return Chunk(
            text=text,
            metadata={"url": url, "chunk_index": 0, "source_hash": source_hash, "content_hash": source_hash},
            chunk_index=0,
        )

    def test_prefix_prepended_to_chunk_text(self):
        pipeline, _ = self._make_pipeline("Context sentence.")
        chunk = self._make_chunk()
        posts = [{"url": "http://x.com/1", "title": "T", "content": "Full content."}]
        result = pipeline._apply_contextual_retrieval([chunk], posts)
        assert result[0].text.startswith("Context sentence.")
        assert "Plain chunk text." in result[0].text

    def test_plain_text_stored_in_metadata(self):
        pipeline, _ = self._make_pipeline("Context.")
        chunk = self._make_chunk(text="Original.")
        posts = [{"url": "http://x.com/1", "title": "T", "content": "C."}]
        result = pipeline._apply_contextual_retrieval([chunk], posts)
        assert result[0].metadata["plain_text"] == "Original."

    def test_context_prefix_stored_in_metadata(self):
        pipeline, _ = self._make_pipeline("My prefix.")
        chunk = self._make_chunk()
        posts = [{"url": "http://x.com/1", "title": "T", "content": "C."}]
        result = pipeline._apply_contextual_retrieval([chunk], posts)
        assert result[0].metadata["context_prefix"] == "My prefix."

    def test_content_hash_is_of_contextualized_text(self):
        from src.ingestion.chunker import _content_hash
        pipeline, _ = self._make_pipeline("Prefix.")
        chunk = self._make_chunk(text="Body.")
        posts = [{"url": "http://x.com/1", "title": "T", "content": "C."}]
        result = pipeline._apply_contextual_retrieval([chunk], posts)
        expected_hash = _content_hash("Prefix.\n\nBody.")
        assert result[0].metadata["content_hash"] == expected_hash

    def test_empty_prefix_leaves_text_unchanged(self):
        pipeline, _ = self._make_pipeline("")  # LLM returned nothing
        chunk = self._make_chunk(text="Plain.")
        posts = [{"url": "http://x.com/1", "title": "T", "content": "C."}]
        result = pipeline._apply_contextual_retrieval([chunk], posts)
        assert result[0].text == "Plain."

    def test_empty_prefix_hash_matches_plain_text(self):
        from src.ingestion.chunker import _content_hash
        pipeline, _ = self._make_pipeline("")
        chunk = self._make_chunk(text="Plain.")
        posts = [{"url": "http://x.com/1", "title": "T", "content": "C."}]
        result = pipeline._apply_contextual_retrieval([chunk], posts)
        assert result[0].metadata["content_hash"] == _content_hash("Plain.")

    def test_contextualized_and_plain_hashes_differ(self):
        from src.ingestion.chunker import _content_hash
        pipeline, _ = self._make_pipeline("Some prefix.")
        chunk = self._make_chunk(text="Body.")
        posts = [{"url": "http://x.com/1", "title": "T", "content": "C."}]
        result = pipeline._apply_contextual_retrieval([chunk], posts)
        assert result[0].metadata["content_hash"] != _content_hash("Body.")

    def test_source_hash_unchanged_after_contextualization(self):
        """source_hash must remain the plain-text fingerprint after contextualization."""
        from src.ingestion.chunker import _content_hash
        pipeline, _ = self._make_pipeline("Some prefix.")
        chunk = self._make_chunk(text="Body.")
        posts = [{"url": "http://x.com/1", "title": "T", "content": "C."}]
        result = pipeline._apply_contextual_retrieval([chunk], posts)
        assert result[0].metadata["source_hash"] == _content_hash("Body.")

    def test_empty_prefix_content_hash_equals_source_hash(self):
        """When the LLM returns nothing, content_hash and source_hash should match."""
        from src.ingestion.chunker import _content_hash
        pipeline, _ = self._make_pipeline("")
        chunk = self._make_chunk(text="Plain.")
        posts = [{"url": "http://x.com/1", "title": "T", "content": "C."}]
        result = pipeline._apply_contextual_retrieval([chunk], posts)
        assert result[0].metadata["content_hash"] == result[0].metadata["source_hash"]


# ---------------------------------------------------------------------------
# ingest_posts skips contextualizer when config flag is off
# ---------------------------------------------------------------------------

class TestIngestPostsContextualRetrieval:
    def _make_pipeline_with_mocks(self, prefix="Context."):
        mock_chunker = MagicMock()
        mock_embedder = MagicMock()
        mock_store = MagicMock()
        mock_store.add_chunks.return_value = 1
        mock_contextualizer = MagicMock()
        mock_contextualizer.get_context.return_value = prefix

        with patch("src.ingestion.ingest.TextChunker", return_value=mock_chunker), \
             patch("src.ingestion.ingest.Embedder", return_value=mock_embedder), \
             patch("src.ingestion.ingest.VectorStore", return_value=mock_store):
            from src.ingestion.ingest import IngestPipeline
            pipeline = IngestPipeline(
                chunker=mock_chunker,
                embedder=mock_embedder,
                vector_store=mock_store,
                contextualizer=mock_contextualizer,
            )
        return pipeline, mock_chunker, mock_embedder, mock_store, mock_contextualizer

    def _chunk(self, url="http://x.com/1", text="content"):
        from src.ingestion.chunker import Chunk, _content_hash
        source_hash = _content_hash(text)
        return Chunk(
            text=text,
            metadata={"url": url, "chunk_index": 0, "source_hash": source_hash, "content_hash": source_hash},
            chunk_index=0,
        )

    def test_contextualizer_not_called_when_flag_off(self):
        pipeline, chunker, embedder, store, ctx = self._make_pipeline_with_mocks()
        chunk = self._chunk()
        chunker.chunk_all_posts.return_value = [chunk]
        store.get_existing_source_hashes_for_url.return_value = set()
        embedder.embed_texts.return_value = [[0.1]]

        with patch("src.ingestion.ingest.config") as mock_cfg:
            mock_cfg.use_contextual_retrieval = False
            mock_cfg.use_parent_child_chunking = False
            pipeline.ingest_posts([{"url": "http://x.com/1"}], show_progress=False)

        ctx.get_context.assert_not_called()

    def test_contextualizer_called_when_flag_on(self):
        pipeline, chunker, embedder, store, ctx = self._make_pipeline_with_mocks()
        chunk = self._chunk(text="some text")
        chunker.chunk_all_posts.return_value = [chunk]
        store.get_existing_source_hashes_for_url.return_value = set()
        embedder.embed_texts.return_value = [[0.1]]

        with patch("src.ingestion.ingest.config") as mock_cfg:
            mock_cfg.use_contextual_retrieval = True
            mock_cfg.use_parent_child_chunking = False
            pipeline.ingest_posts(
                [{"url": "http://x.com/1", "title": "T", "content": "some text"}],
                show_progress=False,
            )

        ctx.get_context.assert_called_once()

    def test_second_ingest_skips_unchanged_chunks(self):
        """Second ingest_posts call with same data must embed 0 chunks."""
        pipeline, chunker, embedder, store, ctx = self._make_pipeline_with_mocks()
        chunk = self._chunk(text="stable content")
        chunker.chunk_all_posts.return_value = [chunk]
        # Simulate store already has this source_hash
        store.get_existing_source_hashes_for_url.return_value = {chunk.metadata["source_hash"]}
        embedder.embed_texts.return_value = []

        with patch("src.ingestion.ingest.config") as mock_cfg:
            mock_cfg.use_contextual_retrieval = False
            mock_cfg.use_parent_child_chunking = False
            stats = pipeline.ingest_posts([{"url": "http://x.com/1"}], show_progress=False)

        assert stats["chunks"] == 0
        embedder.embed_texts.assert_not_called()
        store.add_chunks.assert_not_called()
