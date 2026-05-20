"""Tests for ChunkReranker."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import MagicMock, patch


def _make_chunks(n: int) -> list:
    return [
        {"id": f"chunk_{i}", "text": f"Text about topic {i}", "metadata": {}}
        for i in range(n)
    ]


class TestChunkReranker:
    def test_rerank_returns_top_k(self):
        """Reranker trims results to top_k."""
        chunks = _make_chunks(10)

        mock_ranker = MagicMock()
        mock_ranker.rerank.return_value = [
            {"id": f"chunk_{i}", "score": 10 - i} for i in range(10)
        ]

        with patch("src.retrieval.reranker.ChunkReranker.__init__", lambda self, **kw: None):
            from src.retrieval.reranker import ChunkReranker
            reranker = ChunkReranker.__new__(ChunkReranker)
            reranker._ranker = mock_ranker

            result = reranker.rerank("test query", chunks, top_k=5)

        assert len(result) == 5

    def test_rerank_preserves_chunk_data(self):
        """Reranked chunks keep all original fields."""
        chunks = [
            {"id": "c1", "text": "Near Partner AI services", "metadata": {"title": "AI Post"}},
            {"id": "c2", "text": "Near Partner Salesforce", "metadata": {"title": "SF Post"}},
        ]

        mock_ranker = MagicMock()
        mock_ranker.rerank.return_value = [
            {"id": "c2", "score": 0.9},
            {"id": "c1", "score": 0.7},
        ]

        with patch("src.retrieval.reranker.ChunkReranker.__init__", lambda self, **kw: None):
            from src.retrieval.reranker import ChunkReranker
            reranker = ChunkReranker.__new__(ChunkReranker)
            reranker._ranker = mock_ranker

            result = reranker.rerank("Salesforce", chunks, top_k=2)

        assert result[0]["id"] == "c2"
        assert result[0]["metadata"]["title"] == "SF Post"
        assert result[1]["id"] == "c1"

    def test_rerank_empty_input_returns_empty(self):
        """Empty chunk list returns empty list without calling ranker."""
        mock_ranker = MagicMock()

        with patch("src.retrieval.reranker.ChunkReranker.__init__", lambda self, **kw: None):
            from src.retrieval.reranker import ChunkReranker
            reranker = ChunkReranker.__new__(ChunkReranker)
            reranker._ranker = mock_ranker

            result = reranker.rerank("query", [], top_k=5)

        assert result == []
        mock_ranker.rerank.assert_not_called()

    def test_rerank_fewer_chunks_than_top_k(self):
        """When fewer chunks than top_k, return all of them."""
        chunks = _make_chunks(3)

        mock_ranker = MagicMock()
        mock_ranker.rerank.return_value = [
            {"id": f"chunk_{i}", "score": 3 - i} for i in range(3)
        ]

        with patch("src.retrieval.reranker.ChunkReranker.__init__", lambda self, **kw: None):
            from src.retrieval.reranker import ChunkReranker
            reranker = ChunkReranker.__new__(ChunkReranker)
            reranker._ranker = mock_ranker

            result = reranker.rerank("query", chunks, top_k=10)

        assert len(result) == 3
