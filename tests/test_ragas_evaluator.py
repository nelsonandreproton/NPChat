"""Tests for the RAGAS evaluation harness."""
import sys
import json
import asyncio
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.ragas_evaluator import (
    EvalSample,
    SampleResult,
    EvalReport,
    RAGASEvaluator,
)


# ---------------------------------------------------------------------------
# EvalSample / SampleResult / EvalReport
# ---------------------------------------------------------------------------

class TestSampleResult:
    def test_mean_score_averages_four_metrics(self):
        r = SampleResult("q", faithfulness=0.8, answer_relevancy=0.6,
                         context_precision=0.7, context_relevance=0.5)
        assert r.mean_score() == round((0.8 + 0.6 + 0.7 + 0.5) / 4, 4)

    def test_mean_score_skips_none(self):
        r = SampleResult("q", faithfulness=1.0, answer_relevancy=None,
                         context_precision=None, context_relevance=None)
        assert r.mean_score() == 1.0

    def test_mean_score_all_none_returns_none(self):
        r = SampleResult("q", faithfulness=None, answer_relevancy=None,
                         context_precision=None, context_relevance=None)
        assert r.mean_score() is None

    def test_error_result_has_none_scores(self):
        r = SampleResult("q", faithfulness=None, answer_relevancy=None,
                         context_precision=None, context_relevance=None,
                         error="boom")
        assert r.error == "boom"
        assert r.mean_score() is None


class TestEvalReport:
    def _report(self):
        return EvalReport(
            timestamp="2026-01-01T00:00:00+00:00",
            model_llm="qwen",
            model_embedding="mxbai",
            n_samples=2,
            results=[
                SampleResult("q1", 0.9, 0.8, 0.7, 0.6),
                SampleResult("q2", 0.5, 0.4, 0.3, 0.2),
            ],
        )

    def test_summary_averages_all_metrics(self):
        s = self._report().summary()
        assert s["faithfulness"] == round((0.9 + 0.5) / 2, 4)
        assert s["answer_relevancy"] == round((0.8 + 0.4) / 2, 4)

    def test_mean_overall(self):
        r = self._report()
        # Each sample mean: q1=(0.9+0.8+0.7+0.6)/4=0.75, q2=(0.5+0.4+0.3+0.2)/4=0.35
        assert r.mean_overall() == round((0.75 + 0.35) / 2, 4)

    def test_to_dict_has_expected_keys(self):
        d = self._report().to_dict()
        assert "timestamp" in d
        assert "summary" in d
        assert "results" in d
        assert "mean_overall" in d["summary"]

    def test_save_writes_valid_json(self, tmp_path):
        report = self._report()
        p = tmp_path / "out.json"
        report.save(p)
        data = json.loads(p.read_text())
        assert data["n_samples"] == 2
        assert len(data["results"]) == 2

    def test_save_creates_parent_dirs(self, tmp_path):
        report = self._report()
        p = tmp_path / "nested" / "dir" / "out.json"
        report.save(p)
        assert p.exists()


# ---------------------------------------------------------------------------
# RAGASEvaluator
# ---------------------------------------------------------------------------

def _metric_result(score: float):
    m = MagicMock()
    m.score = score
    return m


def _make_evaluator_with_mocked_metrics():
    evaluator = RAGASEvaluator.__new__(RAGASEvaluator)
    evaluator._metrics_ready = True
    evaluator._faithfulness = MagicMock()
    evaluator._answer_relevancy = MagicMock()
    evaluator._context_precision = MagicMock()
    evaluator._context_relevance = MagicMock()

    evaluator._faithfulness.ascore = AsyncMock(return_value=_metric_result(0.9))
    evaluator._answer_relevancy.ascore = AsyncMock(return_value=_metric_result(0.8))
    evaluator._context_precision.ascore = AsyncMock(return_value=_metric_result(0.7))
    evaluator._context_relevance.ascore = AsyncMock(return_value=_metric_result(0.6))
    return evaluator


class TestRAGASEvaluator:
    def test_evaluate_returns_report_with_correct_n_samples(self):
        ev = _make_evaluator_with_mocked_metrics()
        samples = [
            EvalSample("q1", "a1", ["ctx1"]),
            EvalSample("q2", "a2", ["ctx2"]),
        ]
        report = ev.evaluate(samples)
        assert report.n_samples == 2
        assert len(report.results) == 2

    def test_evaluate_scores_populated(self):
        ev = _make_evaluator_with_mocked_metrics()
        samples = [EvalSample("q", "a", ["ctx"])]
        report = ev.evaluate(samples)
        r = report.results[0]
        assert r.faithfulness == 0.9
        assert r.answer_relevancy == 0.8
        assert r.context_precision == 0.7
        assert r.context_relevance == 0.6

    def test_evaluate_records_error_on_exception(self):
        ev = _make_evaluator_with_mocked_metrics()
        ev._faithfulness.ascore = AsyncMock(side_effect=RuntimeError("LM Studio offline"))
        samples = [EvalSample("q", "a", ["ctx"])]
        report = ev.evaluate(samples)
        r = report.results[0]
        assert r.error is not None
        assert r.faithfulness is None

    def test_evaluate_empty_samples_returns_empty_report(self):
        ev = _make_evaluator_with_mocked_metrics()
        report = ev.evaluate([])
        assert report.n_samples == 0
        assert report.results == []
        assert report.mean_overall() is None

    def test_score_sample_passes_correct_args(self):
        ev = _make_evaluator_with_mocked_metrics()
        sample = EvalSample("my question", "my answer", ["ctx A", "ctx B"])
        asyncio.run(ev._score_sample_async(sample))

        ev._faithfulness.ascore.assert_awaited_once_with(
            user_input="my question",
            response="my answer",
            retrieved_contexts=["ctx A", "ctx B"],
        )
        ev._context_relevance.ascore.assert_awaited_once_with(
            user_input="my question",
            retrieved_contexts=["ctx A", "ctx B"],
        )

    def test_report_timestamp_set(self):
        ev = _make_evaluator_with_mocked_metrics()
        report = ev.evaluate([EvalSample("q", "a", ["c"])])
        assert report.timestamp != ""

    def test_report_model_names_from_config(self):
        ev = _make_evaluator_with_mocked_metrics()
        report = ev.evaluate([EvalSample("q", "a", ["c"])])
        from src.config import config
        assert report.model_llm == config.llm_model
        assert report.model_embedding == config.embedding_model
