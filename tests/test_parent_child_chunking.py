"""Tests for parent-child chunk architecture (Item 9)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from unittest.mock import MagicMock, patch
import pytest

from src.ingestion.chunker import TextChunker, _content_hash


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _post(content: str, url: str = "https://example.com/post") -> dict:
    return {
        "url": url,
        "title": "Test Post",
        "author": "Author",
        "published_date": "2024-01-01",
        "categories": ["tech"],
        "content": content,
    }


def _long_content(n_paragraphs: int = 8) -> str:
    """Generate content that produces multiple parent chunks."""
    return "\n\n".join(
        f"Paragraph {i} about some technical topic with enough words to be meaningful content in the blog post."
        for i in range(n_paragraphs)
    )


# ---------------------------------------------------------------------------
# chunk_blog_post_with_children structure
# ---------------------------------------------------------------------------

class TestChunkBlogPostWithChildren:
    def test_children_have_parent_id(self):
        chunker = TextChunker()
        children = chunker.chunk_blog_post_with_children(_post(_long_content()))
        assert all("parent_id" in c.metadata for c in children)

    def test_children_have_parent_text(self):
        chunker = TextChunker()
        children = chunker.chunk_blog_post_with_children(_post(_long_content()))
        assert all("parent_text" in c.metadata for c in children)

    def test_child_text_shorter_than_parent_text(self):
        chunker = TextChunker()
        children = chunker.chunk_blog_post_with_children(_post(_long_content(8)))
        for child in children:
            assert len(child.text) <= len(child.metadata["parent_text"])

    def test_children_have_source_hash(self):
        chunker = TextChunker()
        children = chunker.chunk_blog_post_with_children(_post(_long_content()))
        for child in children:
            assert child.metadata["source_hash"] == _content_hash(child.text)

    def test_content_hash_equals_source_hash_before_contextualization(self):
        chunker = TextChunker()
        children = chunker.chunk_blog_post_with_children(_post(_long_content()))
        for child in children:
            assert child.metadata["content_hash"] == child.metadata["source_hash"]

    def test_parent_id_is_stable(self):
        """Same content → same parent_id across two calls."""
        chunker = TextChunker()
        content = _long_content(6)
        run1 = chunker.chunk_blog_post_with_children(_post(content))
        run2 = chunker.chunk_blog_post_with_children(_post(content))
        ids1 = [c.metadata["parent_id"] for c in run1]
        ids2 = [c.metadata["parent_id"] for c in run2]
        assert ids1 == ids2

    def test_children_from_same_parent_share_parent_id(self):
        chunker = TextChunker(chunk_size=400)  # small chunks → multiple children per parent
        children = chunker.chunk_blog_post_with_children(_post(_long_content(10)))
        parent_ids = [c.metadata["parent_id"] for c in children]
        # With small chunk_size, multiple children should share the same parent_id
        assert len(set(parent_ids)) < len(parent_ids)

    def test_empty_content_returns_no_children(self):
        chunker = TextChunker()
        assert chunker.chunk_blog_post_with_children(_post("")) == []

    def test_base_metadata_propagated(self):
        chunker = TextChunker()
        children = chunker.chunk_blog_post_with_children(_post(_long_content()))
        for child in children:
            assert child.metadata["url"] == "https://example.com/post"
            assert child.metadata["title"] == "Test Post"

    def test_parent_text_not_prefixed_by_contextualizer(self):
        """parent_text must stay plain — only child text gets the context prefix."""
        chunker = TextChunker()
        children = chunker.chunk_blog_post_with_children(_post(_long_content()))
        for child in children:
            # parent_text should not start with contextual prefix patterns
            assert not child.metadata["parent_text"].startswith("This chunk")

    def test_chunk_all_posts_with_children(self):
        chunker = TextChunker()
        posts = [_post(_long_content(), url=f"https://example.com/{i}") for i in range(3)]
        children = chunker.chunk_all_posts_with_children(posts)
        assert len(children) > 0
        urls = {c.metadata["url"] for c in children}
        assert len(urls) == 3


# ---------------------------------------------------------------------------
# HybridRetriever._dedupe_by_parent
# ---------------------------------------------------------------------------

def _make_hybrid_retriever():
    mock_store = MagicMock()
    mock_store._collection.count.return_value = 0
    mock_store._collection.get.return_value = {"ids": [], "documents": [], "metadatas": []}
    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = [0.1] * 10

    with patch("src.retrieval.hybrid_retriever.VectorStore", return_value=mock_store), \
         patch("src.retrieval.hybrid_retriever.Embedder", return_value=mock_embedder):
        from src.retrieval.hybrid_retriever import HybridRetriever
        hr = HybridRetriever(vector_store=mock_store, embedder=mock_embedder)
    return hr


def _doc(doc_id: str, parent_id: str | None, parent_text: str = "parent", rrf_score: float = 1.0) -> dict:
    return {
        "result": {
            "id": doc_id,
            "text": f"child text for {doc_id}",
            "metadata": {
                "parent_id": parent_id,
                "parent_text": parent_text,
            },
        },
        "rrf_score": rrf_score,
        "semantic_rank": 1,
        "bm25_rank": None,
    }


class TestDedupeByParent:
    def test_passthrough_when_flag_off(self):
        hr = _make_hybrid_retriever()
        docs = [_doc("a", "p1"), _doc("b", "p1"), _doc("c", "p2")]
        with patch("src.retrieval.hybrid_retriever.config") as mock_cfg:
            mock_cfg.use_parent_child_chunking = False
            result = hr._dedupe_by_parent(docs)
        assert len(result) == 3

    def test_dedupes_to_one_per_parent(self):
        hr = _make_hybrid_retriever()
        docs = [_doc("a", "p1", rrf_score=0.9), _doc("b", "p1", rrf_score=0.5), _doc("c", "p2", rrf_score=0.3)]
        with patch("src.retrieval.hybrid_retriever.config") as mock_cfg:
            mock_cfg.use_parent_child_chunking = True
            result = hr._dedupe_by_parent(docs)
        assert len(result) == 2
        parent_ids = [r["result"]["metadata"]["parent_id"] for r in result]
        assert parent_ids == ["p1", "p2"]

    def test_promotes_parent_text(self):
        hr = _make_hybrid_retriever()
        docs = [_doc("a", "p1", parent_text="THE PARENT TEXT")]
        with patch("src.retrieval.hybrid_retriever.config") as mock_cfg:
            mock_cfg.use_parent_child_chunking = True
            result = hr._dedupe_by_parent(docs)
        assert result[0]["result"]["text"] == "THE PARENT TEXT"

    def test_passthrough_when_no_parent_id(self):
        """Chunks ingested without parent_id (old data) pass through unchanged."""
        hr = _make_hybrid_retriever()
        docs = [
            {"result": {"id": "x", "text": "old chunk", "metadata": {}}, "rrf_score": 1.0, "semantic_rank": 1, "bm25_rank": None},
        ]
        with patch("src.retrieval.hybrid_retriever.config") as mock_cfg:
            mock_cfg.use_parent_child_chunking = True
            result = hr._dedupe_by_parent(docs)
        assert len(result) == 1
        assert result[0]["result"]["text"] == "old chunk"

    def test_highest_scoring_child_wins(self):
        """First child in pre-sorted list is kept (highest rrf_score)."""
        hr = _make_hybrid_retriever()
        # Pre-sorted: a (0.9) before b (0.3), both same parent
        docs = [_doc("a", "p1", parent_text="PARENT A", rrf_score=0.9),
                _doc("b", "p1", parent_text="PARENT A", rrf_score=0.3)]
        with patch("src.retrieval.hybrid_retriever.config") as mock_cfg:
            mock_cfg.use_parent_child_chunking = True
            result = hr._dedupe_by_parent(docs)
        assert len(result) == 1
        # The kept entry should be the one from "a" (first in sorted list)
        assert result[0]["rrf_score"] == 0.9


# ---------------------------------------------------------------------------
# Retriever._check_parent_child_consistency warning
# ---------------------------------------------------------------------------

class TestRetrieverParentChildConsistency:
    def _make_retriever(self, use_flag: bool, has_parent_id: bool):
        mock_store = MagicMock()
        mock_store.get_stored_embedding_dimension.return_value = None
        meta = {"parent_id": "abc"} if has_parent_id else {}
        mock_store._collection.get.return_value = {"metadatas": [meta]}
        mock_embedder = MagicMock()
        mock_embedder.model = "text-embedding-mxbai-embed-large-v1"

        with patch("src.retrieval.retriever.VectorStore", return_value=mock_store), \
             patch("src.retrieval.retriever.Embedder", return_value=mock_embedder), \
             patch("src.retrieval.retriever.config") as mock_cfg:
            mock_cfg.use_parent_child_chunking = use_flag
            mock_cfg.top_k = 3
            mock_cfg.similarity_threshold = 0.0
            from src.retrieval.retriever import Retriever
            retriever = Retriever(vector_store=mock_store, embedder=mock_embedder)
        return retriever

    def test_no_warning_when_flag_off(self, capsys):
        self._make_retriever(use_flag=False, has_parent_id=False)
        out = capsys.readouterr().out
        assert "WARNING" not in out or "parent" not in out.lower()

    def test_warning_when_flag_on_but_no_parent_id(self, capsys):
        self._make_retriever(use_flag=True, has_parent_id=False)
        out = capsys.readouterr().out
        assert "WARNING" in out
        assert "parent_id" in out
        assert "Re-ingest All" in out

    def test_no_warning_when_flag_on_and_has_parent_id(self, capsys):
        self._make_retriever(use_flag=True, has_parent_id=True)
        out = capsys.readouterr().out
        assert "parent_id" not in out or "WARNING" not in out
