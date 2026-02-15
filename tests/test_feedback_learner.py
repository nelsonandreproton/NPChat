"""Tests for the feedback learning system."""
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.feedback.feedback_learner import FeedbackLearner


@pytest.fixture
def learner(tmp_path):
    """Create a learner with a temporary database."""
    db_path = str(tmp_path / "test_feedback_learning.db")
    return FeedbackLearner(db_path=db_path)


class TestChunkAdjustment:
    def test_positive_feedback_boosts_chunk(self, learner):
        learner.adjust_chunk_score("chunk_1", is_positive=True)
        score = learner.get_chunk_adjustment("chunk_1")
        assert score > 0

    def test_negative_feedback_penalizes_chunk(self, learner):
        learner.adjust_chunk_score("chunk_1", is_positive=False)
        score = learner.get_chunk_adjustment("chunk_1")
        assert score < 0

    def test_multiple_positives_accumulate(self, learner):
        learner.adjust_chunk_score("chunk_1", is_positive=True)
        learner.adjust_chunk_score("chunk_1", is_positive=True)
        score = learner.get_chunk_adjustment("chunk_1")
        assert score > 0.15  # Should be > single boost

    def test_penalty_larger_than_boost(self, learner):
        learner.adjust_chunk_score("chunk_1", is_positive=True)
        single_boost = learner.get_chunk_adjustment("chunk_1")
        learner.adjust_chunk_score("chunk_1", is_positive=False)
        after_penalty = learner.get_chunk_adjustment("chunk_1")
        # Net should be negative (penalty > boost)
        assert after_penalty < 0

    def test_unknown_chunk_returns_zero(self, learner):
        assert learner.get_chunk_adjustment("nonexistent") == 0.0

    def test_get_all_adjustments(self, learner):
        learner.adjust_chunk_score("chunk_a", is_positive=True)
        learner.adjust_chunk_score("chunk_b", is_positive=False)
        adjustments = learner.get_all_chunk_adjustments()
        assert "chunk_a" in adjustments
        assert "chunk_b" in adjustments

    def test_apply_adjustments_reorders_results(self, learner):
        # Boost chunk_b so it should rank higher than chunk_a
        learner.adjust_chunk_score("chunk_b", is_positive=True)
        learner.adjust_chunk_score("chunk_a", is_positive=False)

        results = [
            {"id": "chunk_a", "combined_score": 0.8},
            {"id": "chunk_b", "combined_score": 0.7},
        ]
        adjusted = learner.apply_adjustments_to_results(results)
        # chunk_b should now rank first due to boost
        assert adjusted[0]["id"] == "chunk_b"

    def test_apply_adjustments_clamps_to_01(self, learner):
        # Add many boosts
        for _ in range(20):
            learner.adjust_chunk_score("chunk_1", is_positive=True)

        results = [{"id": "chunk_1", "combined_score": 0.9}]
        adjusted = learner.apply_adjustments_to_results(results)
        assert adjusted[0]["adjusted_score"] <= 1.0
        assert adjusted[0]["adjusted_score"] >= 0.0


class TestQueryFlagging:
    def test_first_negative_not_flagged(self, learner):
        reason = learner.flag_query_if_needed("test query", threshold=2)
        assert reason is None

    def test_threshold_reached_returns_reason(self, learner):
        learner.flag_query_if_needed("test query", threshold=2)
        reason = learner.flag_query_if_needed("test query", threshold=2)
        assert reason is not None
        assert "2" in reason or "negative" in reason.lower()

    def test_get_flagged_queries(self, learner):
        # Trigger flagging
        learner.flag_query_if_needed("bad query", threshold=2)
        learner.flag_query_if_needed("bad query", threshold=2)
        # Update status to pending
        learner.update_flag_status("bad query", negative_count=2, threshold=2)
        flagged = learner.get_flagged_queries(status='pending')
        assert len(flagged) >= 1

    def test_resolve_flag(self, learner):
        learner.flag_query_if_needed("to resolve", threshold=2)
        learner.flag_query_if_needed("to resolve", threshold=2)
        learner.update_flag_status("to resolve", 2, 2)
        learner.resolve_flag("to resolve", "resolved")
        flagged = learner.get_flagged_queries(status='pending')
        queries = [f["query"] for f in flagged]
        assert "to resolve" not in queries


class TestQueryMapping:
    def test_learn_successful_query(self, learner):
        learner.learn_successful_query("how to use salesforce", ["chunk_1", "chunk_2"])
        similar = learner.get_similar_successful_queries("salesforce help")
        assert len(similar) > 0

    def test_multiple_positives_increase_count(self, learner):
        learner.learn_successful_query("salesforce query", ["chunk_1"])
        learner.learn_successful_query("salesforce query", ["chunk_1"])
        similar = learner.get_similar_successful_queries("salesforce")
        assert similar[0]["positive_count"] >= 2

    def test_learned_boosts_returned(self, learner):
        learner.learn_successful_query("low code platform", ["chunk_x"])
        boosts = learner.get_learned_chunk_boosts("low code development")
        # Should find "chunk_x" as it matches "low" and "code"
        assert "chunk_x" in boosts


class TestProcessFeedback:
    def test_positive_feedback_learns_and_boosts(self, learner):
        actions = learner.process_feedback(
            query="What is near partner?",
            is_positive=True,
            chunk_ids=["c1", "c2"]
        )
        assert actions["query_learned"] is True
        assert "c1" in actions["chunks_adjusted"]
        assert "c2" in actions["chunks_adjusted"]

    def test_negative_feedback_penalizes(self, learner):
        actions = learner.process_feedback(
            query="Bad query",
            is_positive=False,
            chunk_ids=["c1"]
        )
        assert "c1" in actions["chunks_adjusted"]
        score = learner.get_chunk_adjustment("c1")
        assert score < 0

    def test_get_stats_returns_dict(self, learner):
        learner.adjust_chunk_score("c1", True)
        learner.adjust_chunk_score("c2", False)
        stats = learner.get_stats()
        assert "chunks" in stats
        assert stats["chunks"]["total_adjusted"] == 2
        assert stats["chunks"]["boosted"] == 1
        assert stats["chunks"]["penalized"] == 1
