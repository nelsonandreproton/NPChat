"""Tests for multi-query retrieval in EnhancedRAGChain."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from unittest.mock import MagicMock, patch


def _make_chunk(cid: str, text: str = "some text") -> dict:
    return {"id": cid, "text": text, "metadata": {"url": f"http://example.com/{cid}"}}


class TestMultiQueryRetrieval:
    def _build_chain(self, use_multi_query: bool):
        """Build an EnhancedRAGChain with all heavy deps mocked."""
        with (
            patch("src.generation.enhanced_rag_chain.HybridRetriever"),
            patch("src.generation.enhanced_rag_chain.QueryExpander"),
            patch("src.generation.enhanced_rag_chain.QueryLogger"),
            patch("src.generation.enhanced_rag_chain.FeedbackLearner"),
            patch("src.generation.enhanced_rag_chain.OllamaLLM"),
            patch("src.generation.enhanced_rag_chain.PromptTemplates"),
        ):
            from src.generation.enhanced_rag_chain import EnhancedRAGChain
            chain = EnhancedRAGChain(
                use_query_expansion=True,
                use_hybrid_search=True,
                use_logging=False,
                use_reranking=False,
                use_multi_query=use_multi_query,
            )
        return chain

    def test_multi_query_unions_chunks_across_variants(self):
        """Chunks from different query variants are merged without duplicates."""
        chain = self._build_chain(use_multi_query=True)

        # query_expander returns 3 variants
        chain.query_expander = MagicMock()
        chain.query_expander.multi_query.return_value = ["q1", "q2", "q3"]

        # Each variant returns overlapping chunks
        chunk_a = _make_chunk("a")
        chunk_b = _make_chunk("b")
        chunk_c = _make_chunk("c")
        chain.hybrid_retriever = MagicMock()
        chain.hybrid_retriever.retrieve.side_effect = [
            [chunk_a, chunk_b],   # variant q1
            [chunk_b, chunk_c],   # variant q2 — chunk_b is duplicate
            [chunk_c],            # variant q3 — chunk_c is duplicate
        ]
        chain.hybrid_retriever.get_retrieval_scores.return_value = [0.9, 0.8, 0.7]

        chain.feedback_learner = MagicMock()
        chain.feedback_learner.apply_adjustments_to_results.side_effect = lambda x: x

        chain.llm = MagicMock()
        chain.llm.generate.return_value = "answer"

        chain.prompts = MagicMock()
        chain.prompts.system_prompt = ""
        chain.prompts.build_rag_prompt.return_value = "prompt"

        result = chain.query("test question", top_k=5)

        # Should have 3 unique chunks (a, b, c) — no duplicates
        assert len(result.retrieved_chunks) == 3
        ids = {c["id"] for c in result.retrieved_chunks}
        assert ids == {"a", "b", "c"}

    def test_multi_query_calls_expander(self):
        """multi_query is called on the query_expander when enabled."""
        chain = self._build_chain(use_multi_query=True)

        chain.query_expander = MagicMock()
        chain.query_expander.multi_query.return_value = ["q1"]

        chunk = _make_chunk("x")
        chain.hybrid_retriever = MagicMock()
        chain.hybrid_retriever.retrieve.return_value = [chunk]
        chain.hybrid_retriever.get_retrieval_scores.return_value = [0.9]

        chain.feedback_learner = MagicMock()
        chain.feedback_learner.apply_adjustments_to_results.side_effect = lambda x: x

        chain.llm = MagicMock()
        chain.llm.generate.return_value = "answer"
        chain.prompts = MagicMock()
        chain.prompts.system_prompt = ""
        chain.prompts.build_rag_prompt.return_value = "prompt"

        chain.query("test question")

        chain.query_expander.multi_query.assert_called_once_with("test question")

    def test_multi_query_disabled_skips_expander(self):
        """multi_query is NOT called when use_multi_query=False."""
        chain = self._build_chain(use_multi_query=False)

        chain.query_expander = MagicMock()

        chunk = _make_chunk("x")
        chain.hybrid_retriever = MagicMock()
        chain.hybrid_retriever.retrieve.return_value = [chunk]
        chain.hybrid_retriever.get_retrieval_scores.return_value = [0.9]

        chain.feedback_learner = MagicMock()
        chain.feedback_learner.apply_adjustments_to_results.side_effect = lambda x: x

        chain.llm = MagicMock()
        chain.llm.generate.return_value = "answer"
        chain.prompts = MagicMock()
        chain.prompts.system_prompt = ""
        chain.prompts.build_rag_prompt.return_value = "prompt"

        chain.query("test question")

        chain.query_expander.multi_query.assert_not_called()

    def test_multi_query_no_expander_falls_back(self):
        """When query_expander is None, multi_query gracefully falls back to single retrieval."""
        chain = self._build_chain(use_multi_query=True)
        chain.query_expander = None  # simulate expansion disabled

        chunk = _make_chunk("x")
        chain.hybrid_retriever = MagicMock()
        chain.hybrid_retriever.retrieve.return_value = [chunk]
        chain.hybrid_retriever.get_retrieval_scores.return_value = [0.9]

        chain.feedback_learner = MagicMock()
        chain.feedback_learner.apply_adjustments_to_results.side_effect = lambda x: x

        chain.llm = MagicMock()
        chain.llm.generate.return_value = "answer"
        chain.prompts = MagicMock()
        chain.prompts.system_prompt = ""
        chain.prompts.build_rag_prompt.return_value = "prompt"

        result = chain.query("test question")

        assert len(result.retrieved_chunks) == 1
        assert result.retrieved_chunks[0]["id"] == "x"
