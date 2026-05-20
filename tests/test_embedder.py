"""Tests for Embedder: query prefix behavior and model routing."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from unittest.mock import MagicMock, patch
import pytest

from src.ingestion.embedder import _apply_query_prefix, _MXBAI_QUERY_PREFIX


# ---------------------------------------------------------------------------
# _apply_query_prefix unit tests
# ---------------------------------------------------------------------------

class TestApplyQueryPrefix:
    def test_mxbai_model_gets_prefix(self):
        result = _apply_query_prefix("text-embedding-mxbai-embed-large-v1", "hello")
        assert result == _MXBAI_QUERY_PREFIX + "hello"

    def test_mxbai_prefix_case_insensitive(self):
        result = _apply_query_prefix("MXBAI-embed-large", "hello")
        assert result.startswith(_MXBAI_QUERY_PREFIX)

    def test_nomic_model_no_prefix(self):
        result = _apply_query_prefix("nomic-embed-text-v1.5", "hello")
        assert result == "hello"

    def test_unknown_model_no_prefix(self):
        result = _apply_query_prefix("some-other-model", "what is Python?")
        assert result == "what is Python?"

    def test_empty_query_still_prefixed_for_mxbai(self):
        result = _apply_query_prefix("text-embedding-mxbai-embed-large-v1", "")
        assert result == _MXBAI_QUERY_PREFIX


# ---------------------------------------------------------------------------
# Embedder.embed_query vs embed_text prefix behavior
# ---------------------------------------------------------------------------

class TestEmbedder:
    def _make_embedder(self, model="text-embedding-mxbai-embed-large-v1"):
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=[0.1, 0.2, 0.3])]
        )
        with patch("src.ingestion.embedder.OpenAI", return_value=mock_client):
            from src.ingestion.embedder import Embedder
            embedder = Embedder(model=model)
        embedder._client = mock_client
        return embedder, mock_client

    def test_embed_query_prepends_prefix_for_mxbai(self):
        embedder, client = self._make_embedder()
        embedder.embed_query("What is the pricing?")
        call_input = client.embeddings.create.call_args[1]["input"]
        assert call_input.startswith(_MXBAI_QUERY_PREFIX)
        assert "What is the pricing?" in call_input

    def test_embed_text_no_prefix_for_mxbai(self):
        """Documents must NOT get the query prefix."""
        embedder, client = self._make_embedder()
        embedder.embed_text("This is a document chunk.")
        call_input = client.embeddings.create.call_args[1]["input"]
        assert not call_input.startswith(_MXBAI_QUERY_PREFIX)
        assert call_input == "This is a document chunk."

    def test_embed_query_no_prefix_for_nomic(self):
        embedder, client = self._make_embedder(model="nomic-embed-text-v1.5")
        embedder.embed_query("What is the pricing?")
        call_input = client.embeddings.create.call_args[1]["input"]
        assert call_input == "What is the pricing?"

    def test_embed_texts_no_prefix(self):
        """Batch document embedding must not add any prefix."""
        embedder, client = self._make_embedder()
        embedder.embed_texts(["chunk one", "chunk two"])
        calls = client.embeddings.create.call_args_list
        for call in calls:
            assert not call[1]["input"].startswith(_MXBAI_QUERY_PREFIX)

    def test_embed_query_returns_embedding(self):
        embedder, _ = self._make_embedder()
        result = embedder.embed_query("test")
        assert result == [0.1, 0.2, 0.3]


# ---------------------------------------------------------------------------
# Retriever._check_embedding_dimension
# ---------------------------------------------------------------------------

class TestRetrieverDimensionCheck:
    def _make_retriever(self, stored_dim, model="text-embedding-mxbai-embed-large-v1"):
        mock_store = MagicMock()
        mock_store.get_stored_embedding_dimension.return_value = stored_dim
        mock_embedder = MagicMock()
        mock_embedder.model = model

        with patch("src.retrieval.retriever.VectorStore", return_value=mock_store), \
             patch("src.retrieval.retriever.Embedder", return_value=mock_embedder):
            from src.retrieval.retriever import Retriever
            retriever = Retriever(
                vector_store=mock_store,
                embedder=mock_embedder,
            )
        return retriever, mock_store

    def test_no_warning_when_store_empty(self, capsys):
        self._make_retriever(stored_dim=None)
        out = capsys.readouterr().out
        assert "WARNING" not in out

    def test_no_warning_when_dimensions_match(self, capsys):
        self._make_retriever(stored_dim=1024, model="text-embedding-mxbai-embed-large-v1")
        out = capsys.readouterr().out
        assert "WARNING" not in out

    def test_warning_on_dimension_mismatch(self, capsys):
        # Store has 768-dim (nomic); config says mxbai (1024-dim)
        self._make_retriever(stored_dim=768, model="text-embedding-mxbai-embed-large-v1")
        out = capsys.readouterr().out
        assert "WARNING" in out
        assert "Re-ingest All" in out

    def test_no_warning_for_unknown_model_even_if_dim_differs(self, capsys):
        # Can't validate unknown models — stay silent
        self._make_retriever(stored_dim=512, model="some-unknown-model")
        out = capsys.readouterr().out
        assert "WARNING" not in out
