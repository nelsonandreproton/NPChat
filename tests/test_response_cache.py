"""Tests for the response cache."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.analytics.response_cache import ResponseCache


@pytest.fixture
def cache(tmp_path):
    return ResponseCache(db_path=str(tmp_path / "test_cache.db"), ttl_hours=1)


SETTINGS = {"top_k": 5, "use_expansion": True, "use_hybrid": True, "use_hyde": False}


class TestResponseCache:
    def test_miss_returns_none(self, cache):
        assert cache.get("unknown query", SETTINGS) is None

    def test_set_and_get_returns_cached_response(self, cache):
        cache.set("what is near partner", SETTINGS, "Near Partner is a tech company.", [])
        result = cache.get("what is near partner", SETTINGS)
        assert result is not None
        assert result["answer"] == "Near Partner is a tech company."
        assert result["cached"] is True

    def test_case_insensitive_key(self, cache):
        cache.set("What Is Near Partner", SETTINGS, "Answer", [])
        result = cache.get("what is near partner", SETTINGS)
        assert result is not None

    def test_different_settings_miss(self, cache):
        cache.set("query", SETTINGS, "Answer", [])
        other_settings = {**SETTINGS, "top_k": 10}
        assert cache.get("query", other_settings) is None

    def test_hit_count_increments(self, cache):
        cache.set("counter query", SETTINGS, "Answer", [])
        cache.get("counter query", SETTINGS)
        cache.get("counter query", SETTINGS)
        stats = cache.get_stats()
        assert stats["total_hits"] >= 2

    def test_clear_removes_all(self, cache):
        cache.set("q1", SETTINGS, "A1", [])
        cache.set("q2", SETTINGS, "A2", [])
        cache.clear()
        assert cache.get("q1", SETTINGS) is None
        assert cache.get_stats()["total_entries"] == 0

    def test_sources_preserved(self, cache):
        sources = [{"title": "Test", "author": "A", "url": "http://test.com"}]
        cache.set("query", SETTINGS, "answer", sources)
        result = cache.get("query", SETTINGS)
        assert result["sources"] == sources

    def test_get_stats_structure(self, cache):
        stats = cache.get_stats()
        assert "total_entries" in stats
        assert "total_hits" in stats
        assert "avg_hits_per_entry" in stats

    def test_get_recent_returns_list(self, cache):
        cache.set("recent query", SETTINGS, "answer", [])
        recent = cache.get_recent(limit=5)
        assert len(recent) >= 1
        assert "query" in recent[0]
