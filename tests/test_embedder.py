"""Tests for Embedder: query prefix behavior and model routing."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from src.ingestion.embedder import _apply_query_prefix, _MXBAI_QUERY_PREFIX


# ---------------------------------------------------------------------------
# _apply_query_prefix unit tests (pure function — no mock needed)
# ---------------------------------------------------------------------------

class TestApplyQueryPrefix:
    def test_mxbai_model_gets_prefix(self):
        result = _apply_query_prefix("mxbai-embed-large-v1", "hello")
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
        result = _apply_query_prefix("mxbai-embed-large-v1", "")
        assert result == _MXBAI_QUERY_PREFIX


# ---------------------------------------------------------------------------
# Embedder.embed_query vs embed_text prefix behavior
# Patch SentenceTransformer so no model is downloaded during tests.
# ---------------------------------------------------------------------------

def _make_embedder(model="mxbai-embed-large-v1"):
    """Return (Embedder, mock_st) with SentenceTransformer patched out."""
    mock_st = MagicMock()
    # encode() returns a numpy array of floats (1-D for single text,
    # 2-D for a list). We match that behaviour so .tolist() works.
    mock_st.encode.side_effect = lambda texts, **kw: (
        np.array([0.1, 0.2, 0.3]) if isinstance(texts, str)
        else np.array([[0.1, 0.2, 0.3]] * len(texts))
    )
    with patch("src.ingestion.embedder.SentenceTransformer", return_value=mock_st):
        from src.ingestion.embedder import Embedder
        embedder = Embedder(model=model)
    return embedder, mock_st


class TestEmbedder:
    def test_embed_query_prepends_prefix_for_mxbai(self):
        embedder, mock_st = _make_embedder()
        embedder.embed_query("What is the pricing?")
        call_text = mock_st.encode.call_args[0][0]
        assert call_text.startswith(_MXBAI_QUERY_PREFIX)
        assert "What is the pricing?" in call_text

    def test_embed_text_no_prefix_for_mxbai(self):
        """Documents must NOT get the query prefix."""
        embedder, mock_st = _make_embedder()
        embedder.embed_text("This is a document chunk.")
        call_text = mock_st.encode.call_args[0][0]
        assert not call_text.startswith(_MXBAI_QUERY_PREFIX)
        assert call_text == "This is a document chunk."

    def test_embed_query_no_prefix_for_nomic(self):
        embedder, mock_st = _make_embedder(model="nomic-embed-text-v1.5")
        embedder.embed_query("What is the pricing?")
        call_text = mock_st.encode.call_args[0][0]
        assert call_text == "What is the pricing?"

    def test_embed_texts_no_prefix(self):
        """Batch document embedding must not add any prefix."""
        embedder, mock_st = _make_embedder()
        embedder.embed_texts(["chunk one", "chunk two"])
        call_texts = mock_st.encode.call_args[0][0]
        for text in call_texts:
            assert not text.startswith(_MXBAI_QUERY_PREFIX)

    def test_embed_query_returns_list_of_floats(self):
        embedder, _ = _make_embedder()
        result = embedder.embed_query("test")
        assert result == [0.1, 0.2, 0.3]


# ---------------------------------------------------------------------------
# Retriever._check_embedding_dimension
# Model strings use the new config labels (keys in _KNOWN_DIMENSIONS).
# ---------------------------------------------------------------------------

class TestRetrieverDimensionCheck:
    def _make_retriever(self, stored_dim, model="mxbai-embed-large-v1"):
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
        self._make_retriever(stored_dim=1024, model="mxbai-embed-large-v1")
        out = capsys.readouterr().out
        assert "WARNING" not in out

    def test_warning_on_dimension_mismatch(self, capsys):
        # Store has 768-dim (nomic); config says mxbai (1024-dim)
        self._make_retriever(stored_dim=768, model="mxbai-embed-large-v1")
        out = capsys.readouterr().out
        assert "WARNING" in out
        assert "Re-ingest All" in out

    def test_no_warning_for_unknown_model_even_if_dim_differs(self, capsys):
        # Can't validate unknown models — stay silent
        self._make_retriever(stored_dim=512, model="some-unknown-model")
        out = capsys.readouterr().out
        assert "WARNING" not in out
