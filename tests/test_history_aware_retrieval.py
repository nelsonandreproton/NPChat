"""Tests for history-aware retrieval (query condensation) in EnhancedRAGChain."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from unittest.mock import MagicMock, patch

from src.retrieval.query_expansion import QueryExpander


def _make_chunk(cid: str, text: str = "some text") -> dict:
    return {"id": cid, "text": text, "metadata": {"url": f"http://example.com/{cid}"}}


HISTORY = [
    {"role": "user", "content": "Quais são os serviços da Near Partner?"},
    {"role": "assistant", "content": "Salesforce, Low-Code e IA."},
]


class TestQueryExpanderCondenseQuery:
    def _expander_with_mock_client(self) -> QueryExpander:
        qe = QueryExpander.__new__(QueryExpander)
        qe.model = "test-model"
        qe._client = MagicMock()
        return qe

    def test_no_history_returns_query_unchanged_without_llm_call(self):
        qe = self._expander_with_mock_client()
        result = qe.condense_query("E quanto custa isso?", conversation_history=None)
        assert result == "E quanto custa isso?"
        qe._client.chat.completions.create.assert_not_called()

    def test_empty_history_returns_query_unchanged_without_llm_call(self):
        qe = self._expander_with_mock_client()
        result = qe.condense_query("E quanto custa isso?", conversation_history=[])
        assert result == "E quanto custa isso?"
        qe._client.chat.completions.create.assert_not_called()

    def test_with_history_rewrites_query_via_llm(self):
        qe = self._expander_with_mock_client()
        qe._client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                content="Quanto custam os serviços da Near Partner?"
            ))]
        )
        result = qe.condense_query("E quanto custa isso?", conversation_history=HISTORY)
        assert result == "Quanto custam os serviços da Near Partner?"
        qe._client.chat.completions.create.assert_called_once()
        sent_prompt = qe._client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        assert "Quais são os serviços da Near Partner?" in sent_prompt
        assert "E quanto custa isso?" in sent_prompt

    def test_llm_failure_falls_back_to_original_query(self):
        qe = self._expander_with_mock_client()
        qe._client.chat.completions.create.side_effect = RuntimeError("boom")
        result = qe.condense_query("E quanto custa isso?", conversation_history=HISTORY)
        assert result == "E quanto custa isso?"


class TestHistoryAwareRetrievalInChain:
    def _build_chain(self, use_history_aware_retrieval: bool = True):
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
                use_query_expansion=False,
                use_hybrid_search=True,
                use_logging=False,
                use_reranking=False,
                use_multi_query=False,
                use_history_aware_retrieval=use_history_aware_retrieval,
            )

        chunk = _make_chunk("a")
        chain.query_expander = MagicMock()
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
        return chain

    def test_retrieval_uses_condensed_query_when_history_present(self):
        chain = self._build_chain(use_history_aware_retrieval=True)
        chain.query_expander.condense_query.return_value = "Quanto custam os serviços da Near Partner?"

        result = chain.query("E quanto custa isso?", conversation_history=HISTORY)

        chain.query_expander.condense_query.assert_called_once_with("E quanto custa isso?", HISTORY)
        retrieve_kwargs = chain.hybrid_retriever.retrieve.call_args.kwargs
        assert retrieve_kwargs["query"] == "Quanto custam os serviços da Near Partner?"
        assert result.condensed_query == "Quanto custam os serviços da Near Partner?"
        assert result.query == "E quanto custa isso?"  # original question preserved

    def test_no_condensation_without_history(self):
        chain = self._build_chain(use_history_aware_retrieval=True)

        result = chain.query("Quais são os serviços da Near Partner?", conversation_history=None)

        chain.query_expander.condense_query.assert_not_called()
        retrieve_kwargs = chain.hybrid_retriever.retrieve.call_args.kwargs
        assert retrieve_kwargs["query"] == "Quais são os serviços da Near Partner?"
        assert result.condensed_query is None

    def test_disabled_flag_skips_condensation_even_with_history(self):
        chain = self._build_chain(use_history_aware_retrieval=False)

        result = chain.query("E quanto custa isso?", conversation_history=HISTORY)

        chain.query_expander.condense_query.assert_not_called()
        retrieve_kwargs = chain.hybrid_retriever.retrieve.call_args.kwargs
        assert retrieve_kwargs["query"] == "E quanto custa isso?"
        assert result.condensed_query is None

    def test_unchanged_condensation_result_not_reported_as_condensed(self):
        """If the rewrite is identical to the input, condensed_query stays None."""
        chain = self._build_chain(use_history_aware_retrieval=True)
        chain.query_expander.condense_query.return_value = "Quais são os serviços da Near Partner?"

        result = chain.query("Quais são os serviços da Near Partner?", conversation_history=HISTORY)

        assert result.condensed_query is None
