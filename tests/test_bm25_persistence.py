"""Tests for BM25 index disk persistence in HybridRetriever."""
import sys
import pickle
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from unittest.mock import MagicMock, patch


def _make_retriever(cache_path: Path, doc_count: int = 3, ids: list = None):
    """Build a HybridRetriever with a mocked vector store and embedder."""
    ids = ids or [f"id_{i}" for i in range(doc_count)]
    docs = [f"document text {i}" for i in range(doc_count)]
    metadatas = [{"url": f"http://example.com/{i}"} for i in range(doc_count)]

    mock_collection = MagicMock()
    mock_collection.count.return_value = doc_count
    mock_collection.get.return_value = {"documents": docs, "metadatas": metadatas, "ids": ids}

    mock_vs = MagicMock()
    mock_vs._collection = mock_collection

    mock_embedder = MagicMock()

    with patch("src.retrieval.hybrid_retriever.VectorStore", return_value=mock_vs), \
         patch("src.retrieval.hybrid_retriever.Embedder", return_value=mock_embedder):
        from src.retrieval.hybrid_retriever import HybridRetriever
        retriever = HybridRetriever(
            vector_store=mock_vs,
            embedder=mock_embedder,
            cache_path=cache_path,
        )

    return retriever, mock_vs, mock_collection


class TestBM25Persistence:
    def test_index_saved_after_build(self, tmp_path):
        cache = tmp_path / "bm25_index.pkl"
        retriever, _, _ = _make_retriever(cache)
        # No cache yet — trigger a build
        retriever._build_bm25_index()
        assert cache.exists()

    def test_saved_cache_is_valid_pickle(self, tmp_path):
        cache = tmp_path / "bm25_index.pkl"
        retriever, _, _ = _make_retriever(cache)
        retriever._build_bm25_index()
        with open(cache, "rb") as f:
            data = pickle.load(f)
        assert "bm25" in data
        assert "docs" in data
        assert "fingerprint" in data
        assert data["doc_count"] == 3

    def test_index_loaded_from_disk_on_init(self, tmp_path):
        cache = tmp_path / "bm25_index.pkl"
        # First retriever builds and saves
        r1, vs1, coll1 = _make_retriever(cache, doc_count=2, ids=["a", "b"])
        r1._build_bm25_index()
        assert cache.exists()

        # Second retriever with same store should load from cache, not rebuild
        r2, vs2, coll2 = _make_retriever(cache, doc_count=2, ids=["a", "b"])
        # If loaded from disk, _build_bm25_index was NOT called during __init__
        # so collection.get was called only once (for fingerprint check in _load_index)
        # The index should be populated
        assert r2._bm25 is not None
        assert r2._index_doc_count == 2

    def test_stale_cache_triggers_rebuild(self, tmp_path):
        cache = tmp_path / "bm25_index.pkl"
        # Save a cache for 2 docs
        r1, _, _ = _make_retriever(cache, doc_count=2, ids=["a", "b"])
        r1._build_bm25_index()

        # New retriever with 3 docs — count mismatch → should NOT load stale cache
        r2, vs2, coll2 = _make_retriever(cache, doc_count=3, ids=["a", "b", "c"])
        # _load_index sees count=3 != cached 2 → skips load
        assert r2._index_doc_count == 0 or r2._bm25 is None

    def test_invalidate_deletes_cache_file(self, tmp_path):
        cache = tmp_path / "bm25_index.pkl"
        retriever, _, _ = _make_retriever(cache)
        retriever._build_bm25_index()
        assert cache.exists()

        retriever.invalidate_index()

        assert not cache.exists()
        assert retriever._bm25 is None
        assert retriever._index_fingerprint is None

    def test_invalidate_no_cache_file_does_not_raise(self, tmp_path):
        cache = tmp_path / "bm25_index.pkl"
        retriever, _, _ = _make_retriever(cache)
        # No cache file created — invalidate should be safe
        retriever.invalidate_index()  # should not raise

    def test_corrupted_cache_falls_back_to_rebuild(self, tmp_path):
        cache = tmp_path / "bm25_index.pkl"
        cache.write_bytes(b"not valid pickle data")

        retriever, _, _ = _make_retriever(cache)
        # _load_index should catch the exception and leave index empty
        assert retriever._bm25 is None

    def test_fingerprint_stored_and_matches_ids(self, tmp_path):
        from src.retrieval.hybrid_retriever import _ids_fingerprint
        cache = tmp_path / "bm25_index.pkl"
        ids = ["doc_0", "doc_1", "doc_2"]
        retriever, _, _ = _make_retriever(cache, doc_count=3, ids=ids)
        retriever._build_bm25_index()

        with open(cache, "rb") as f:
            data = pickle.load(f)

        assert data["fingerprint"] == _ids_fingerprint(ids)
