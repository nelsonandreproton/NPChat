"""
Enhanced RAG chain with query expansion, hybrid search, analytics, feedback learning,
conversation history (multi-turn), and auto-quality evaluation.
"""
import time
from typing import Dict, Any, List, Generator, Optional
from dataclasses import dataclass, field
from .llm import OllamaLLM
from .prompts import PromptTemplates, sanitize_user_input
from ..retrieval.hybrid_retriever import HybridRetriever
from ..retrieval.query_expansion import QueryExpander
from ..retrieval.reranker import ChunkReranker
from ..analytics.query_logger import QueryLogger
from ..feedback.feedback_learner import FeedbackLearner
from ..config import config


@dataclass
class EnhancedRAGResponse:
    """Response from the enhanced RAG chain."""
    answer: str
    sources: List[Dict[str, str]]
    retrieved_chunks: List[Dict[str, Any]]
    chunk_ids: List[str]  # For feedback learning
    query: str
    expanded_query: Optional[str] = None
    condensed_query: Optional[str] = None  # Standalone query used for retrieval, if rewritten from history
    timings: Dict[str, float] = field(default_factory=dict)
    log_id: Optional[int] = None
    confidence_score: Optional[float] = None  # Auto-quality score 0-1
    low_confidence: bool = False  # True if system is not confident in answer


class EnhancedRAGChain:
    """
    Enhanced RAG pipeline with:
    - Query expansion (optional)
    - Hybrid search (semantic + BM25)
    - Query logging for analytics
    """

    def __init__(
        self,
        llm: Optional[OllamaLLM] = None,
        use_query_expansion: bool = True,
        use_hybrid_search: bool = True,
        use_logging: bool = True,
        use_reranking: Optional[bool] = None,
        use_multi_query: Optional[bool] = None,
        use_history_aware_retrieval: Optional[bool] = None,
    ):
        self.llm = llm or OllamaLLM()
        self.prompts = PromptTemplates()

        # Feature flags — fall back to config defaults when not explicitly set
        self.use_query_expansion = use_query_expansion
        self.use_hybrid_search = use_hybrid_search
        self.use_logging = use_logging
        self.use_reranking = use_reranking if use_reranking is not None else config.use_reranking
        self.use_multi_query = use_multi_query if use_multi_query is not None else config.use_multi_query
        self.use_history_aware_retrieval = (
            use_history_aware_retrieval if use_history_aware_retrieval is not None
            else config.use_history_aware_retrieval
        )

        # Initialize components
        self.hybrid_retriever = HybridRetriever() if use_hybrid_search else None
        # Always available: needed for history-aware query condensation even
        # when query expansion (HyDE/multi-query) is disabled.
        self.query_expander = QueryExpander()
        self.query_logger = QueryLogger() if use_logging else None
        self.feedback_learner = FeedbackLearner()

        # Reranker — lazy-loaded on first use so startup doesn't block
        self._reranker: Optional[ChunkReranker] = None

        # Fallback to basic retriever if hybrid not used
        if not use_hybrid_search:
            from ..retrieval.retriever import Retriever
            self.basic_retriever = Retriever()

    @property
    def reranker(self) -> ChunkReranker:
        if self._reranker is None:
            self._reranker = ChunkReranker()
        return self._reranker

    def query(
        self,
        question: str,
        top_k: int = 5,
        temperature: float = 0.7,
        use_hyde: bool = False,
        conversation_history: Optional[List[Dict]] = None,
        evaluate_confidence: bool = False
    ) -> EnhancedRAGResponse:
        """
        Process a question through the enhanced RAG pipeline.

        Args:
            question: User's question
            top_k: Number of chunks to retrieve
            temperature: LLM temperature
            use_hyde: Use Hypothetical Document Embedding
            conversation_history: Previous conversation messages for multi-turn context.
                When self.use_history_aware_retrieval is on, this is also used to
                condense follow-up questions into a standalone query before retrieval.
            evaluate_confidence: Run auto-quality evaluation on the response

        Returns:
            EnhancedRAGResponse with answer, sources, and metadata
        """
        timings = {}
        total_start = time.time()
        expanded_query = None
        condensed_query = None

        # Sanitize input
        question = sanitize_user_input(question)

        # Step 0: Condense follow-up questions into a standalone query, so
        # retrieval isn't blind to context from earlier turns (e.g. a
        # question like "e quanto custa isso?" needs the prior topic to
        # retrieve the right chunks). Only runs when there's history to
        # condense from, so single-turn queries pay no extra LLM call.
        retrieval_query = question
        if self.use_history_aware_retrieval and conversation_history:
            t0 = time.time()
            retrieval_query = self.query_expander.condense_query(question, conversation_history)
            timings["query_condensation"] = round(time.time() - t0, 2)
            if retrieval_query != question:
                condensed_query = retrieval_query
                print(f"[EnhancedRAG] Condensed query: '{question}' -> '{retrieval_query}'")

        # Step 1: Query Expansion (optional)
        search_query = retrieval_query
        if self.use_query_expansion and self.query_expander:
            print("[EnhancedRAG] Expanding query...")
            t0 = time.time()

            if use_hyde:
                expanded_query = self.query_expander.generate_hyde(retrieval_query)
                search_query = expanded_query
            else:
                expanded_query = self.query_expander.expand_query(retrieval_query)
                search_query = expanded_query

            timings["query_expansion"] = round(time.time() - t0, 2)
            print(f"[EnhancedRAG] Query expansion took {timings['query_expansion']}s")

        # Step 2: Retrieve relevant chunks
        print("[EnhancedRAG] Starting retrieval...")
        t0 = time.time()

        # Fetch more candidates when reranking so the cross-encoder has room to work
        fetch_k = config.rerank_top_k_candidates if self.use_reranking else top_k

        if self.use_multi_query and self.query_expander:
            # Generate query variants and union results by chunk ID
            variants = self.query_expander.multi_query(retrieval_query)
            print(f"[EnhancedRAG] Multi-query: {len(variants)} variants")
            seen_ids: set = set()
            chunks = []
            for variant in variants:
                if self.use_hybrid_search and self.hybrid_retriever:
                    variant_chunks = self.hybrid_retriever.retrieve(
                        query=variant,
                        top_k=fetch_k,
                        expanded_query=expanded_query,
                    )
                else:
                    variant_chunks = self.basic_retriever.retrieve_with_scores(variant, top_k=fetch_k)
                for c in variant_chunks:
                    cid = c.get("id", "")
                    if cid and cid not in seen_ids:
                        seen_ids.add(cid)
                        chunks.append(c)
            retrieval_scores = (
                self.hybrid_retriever.get_retrieval_scores(chunks)
                if self.use_hybrid_search and self.hybrid_retriever
                else [c.get("distance", 0) for c in chunks]
            )
        elif self.use_hybrid_search and self.hybrid_retriever:
            chunks = self.hybrid_retriever.retrieve(
                query=retrieval_query,  # Standalone/condensed query for embedding
                top_k=fetch_k,
                expanded_query=expanded_query  # Expanded for BM25
            )
            retrieval_scores = self.hybrid_retriever.get_retrieval_scores(chunks)
        else:
            chunks = self.basic_retriever.retrieve_with_scores(search_query, top_k=fetch_k)
            retrieval_scores = [c.get("distance", 0) for c in chunks]

        timings["retrieval"] = round(time.time() - t0, 2)
        print(f"[EnhancedRAG] Retrieval took {timings['retrieval']}s - found {len(chunks)} chunks")

        # Step 2b: Cross-encoder reranking (optional)
        if self.use_reranking and chunks:
            t0 = time.time()
            chunks = self.reranker.rerank(retrieval_query, chunks, top_k=top_k)
            timings["reranking"] = round(time.time() - t0, 2)
            print(f"[EnhancedRAG] Reranking took {timings['reranking']}s → {len(chunks)} chunks kept")

        # Step 2c: Apply feedback-based adjustments
        chunks = self.feedback_learner.apply_adjustments_to_results(chunks)

        # Extract chunk IDs for feedback tracking
        chunk_ids = [c.get("id", "") for c in chunks if c.get("id")]

        # Step 3: Build prompt with context and conversation history
        t0 = time.time()
        prompt = self.prompts.build_rag_prompt(question, chunks, conversation_history)
        timings["prompt_build"] = round(time.time() - t0, 2)

        # Step 4: Generate response with LLM
        print(f"[EnhancedRAG] Starting LLM generation...")
        t0 = time.time()
        answer = self.llm.generate(
            prompt=prompt,
            system_prompt=self.prompts.system_prompt,
            temperature=temperature
        )
        timings["llm_generation"] = round(time.time() - t0, 2)
        print(f"[EnhancedRAG] LLM generation took {timings['llm_generation']}s")

        # Extract sources
        sources = self._extract_sources(chunks)

        # Step 4b: Auto-quality evaluation (optional, adds ~1-2s)
        confidence_score = None
        low_confidence = False
        if evaluate_confidence:
            confidence_score = self._evaluate_confidence(question, chunks, answer)
            low_confidence = confidence_score is not None and confidence_score < 0.5
            print(f"[EnhancedRAG] Confidence score: {confidence_score}")

        timings["total"] = round(time.time() - total_start, 2)
        print(f"[EnhancedRAG] Total time: {timings['total']}s")

        # Step 5: Log query for analytics
        log_id = None
        if self.use_logging and self.query_logger:
            log_id = self.query_logger.log(
                query=question,
                retrieval_scores=retrieval_scores,
                response_time_ms=int(timings["total"] * 1000),
                expanded_query=expanded_query,
                model_used=config.llm_model
            )

        return EnhancedRAGResponse(
            answer=answer,
            sources=sources,
            retrieved_chunks=chunks,
            chunk_ids=chunk_ids,
            query=question,
            expanded_query=expanded_query,
            condensed_query=condensed_query,
            timings=timings,
            log_id=log_id,
            confidence_score=confidence_score,
            low_confidence=low_confidence
        )

    def _evaluate_confidence(
        self,
        question: str,
        chunks: List[Dict],
        answer: str
    ) -> Optional[float]:
        """
        Auto-evaluate response quality using the LLM.

        Returns a confidence score between 0 and 1, or None on failure.
        """
        try:
            context_summary = " | ".join(
                c.get("metadata", {}).get("title", "") for c in chunks[:3]
            )
            eval_prompt = self.prompts.confidence_eval_prompt.format(
                question=question,
                context_summary=context_summary,
                answer=answer[:500]
            )
            from openai import OpenAI
            client = OpenAI(base_url=config.llm_base_url, api_key="not-needed")
            response = client.chat.completions.create(
                model=config.llm_model,
                messages=[{"role": "user", "content": eval_prompt}],
                temperature=0.1,
                max_tokens=10,
            )
            score_str = response.choices[0].message.content.strip()
            score = float(score_str)
            return max(0.0, min(1.0, score))
        except Exception:
            return None

    def update_feedback(self, log_id: int, feedback: str):
        """Update feedback for a logged query."""
        if self.query_logger:
            self.query_logger.update_feedback(log_id, feedback)

    def _extract_sources(self, chunks: List[Dict]) -> List[Dict[str, str]]:
        """Extract unique sources from chunks."""
        seen_urls = set()
        sources = []

        for chunk in chunks:
            meta = chunk.get("metadata", {})
            url = meta.get("url", "")

            if url and url not in seen_urls:
                seen_urls.add(url)
                sources.append({
                    "title": meta.get("title", "Unknown"),
                    "author": meta.get("author", "Unknown"),
                    "url": url
                })

        return sources

    def get_analytics_stats(self) -> Dict[str, Any]:
        """Get analytics statistics."""
        if self.query_logger:
            return self.query_logger.get_stats()
        return {}
