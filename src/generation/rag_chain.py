"""
Complete RAG chain combining retrieval and generation.
"""
import time
from typing import Dict, Any, List, Generator, Optional
from dataclasses import dataclass, field
from .llm import OllamaLLM
from .prompts import PromptTemplates
from ..retrieval.retriever import Retriever


@dataclass
class RAGResponse:
    """Response from the RAG chain."""
    answer: str
    sources: List[Dict[str, str]]
    retrieved_chunks: List[Dict[str, Any]]
    query: str
    timings: Dict[str, float] = field(default_factory=dict)


class RAGChain:
    """
    Complete RAG pipeline: retrieve relevant chunks and generate response.
    """

    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        llm: Optional[OllamaLLM] = None
    ):
        """
        Initialize the RAG chain.

        Args:
            retriever: Retriever instance
            llm: OllamaLLM instance
        """
        self.retriever = retriever or Retriever()
        self.llm = llm or OllamaLLM()
        self.prompts = PromptTemplates()

    def query(
        self,
        question: str,
        top_k: int = 5,
        temperature: float = 0.7
    ) -> RAGResponse:
        """
        Process a question through the RAG pipeline.

        Args:
            question: User's question
            top_k: Number of chunks to retrieve
            temperature: LLM temperature

        Returns:
            RAGResponse with answer, sources, and metadata
        """
        timings = {}
        total_start = time.time()

        # Step 1: Retrieve relevant chunks (includes embedding query)
        print("[RAG] Starting retrieval...")
        t0 = time.time()
        chunks = self.retriever.retrieve_with_scores(question, top_k=top_k)
        timings["retrieval"] = round(time.time() - t0, 2)
        print(f"[RAG] Retrieval took {timings['retrieval']}s - found {len(chunks)} chunks")

        # Step 2: Build prompt with context
        t0 = time.time()
        prompt = self.prompts.build_rag_prompt(question, chunks)
        timings["prompt_build"] = round(time.time() - t0, 2)
        print(f"[RAG] Prompt build took {timings['prompt_build']}s - {len(prompt)} chars")

        # Step 3: Generate response with LLM
        print(f"[RAG] Starting LLM generation...")
        t0 = time.time()
        answer = self.llm.generate(
            prompt=prompt,
            system_prompt=self.prompts.system_prompt,
            temperature=temperature
        )
        timings["llm_generation"] = round(time.time() - t0, 2)
        print(f"[RAG] LLM generation took {timings['llm_generation']}s")

        # Extract sources
        sources = self.retriever.get_sources(chunks)

        timings["total"] = round(time.time() - total_start, 2)
        print(f"[RAG] Total time: {timings['total']}s")

        return RAGResponse(
            answer=answer,
            sources=sources,
            retrieved_chunks=chunks,
            query=question,
            timings=timings
        )

    def query_stream(
        self,
        question: str,
        top_k: int = 5,
        temperature: float = 0.7
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Process a question with streaming response.

        Args:
            question: User's question
            top_k: Number of chunks to retrieve
            temperature: LLM temperature

        Yields:
            Dicts with either 'chunk' (text) or 'done' (final metadata)
        """
        # Retrieve relevant chunks
        chunks = self.retriever.retrieve_with_scores(question, top_k=top_k)

        # Build prompt with context
        prompt = self.prompts.build_rag_prompt(question, chunks)

        # Extract sources early for streaming
        sources = self.retriever.get_sources(chunks)

        # Yield sources first so UI can display them
        yield {"type": "sources", "data": sources}

        # Stream the response
        full_response = ""
        for text_chunk in self.llm.generate_stream(
            prompt=prompt,
            system_prompt=self.prompts.system_prompt,
            temperature=temperature
        ):
            full_response += text_chunk
            yield {"type": "chunk", "data": text_chunk}

        # Yield completion signal with full response
        yield {
            "type": "done",
            "data": {
                "answer": full_response,
                "sources": sources,
                "query": question
            }
        }

    def get_available_categories(self) -> List[str]:
        """
        Get all categories available in the knowledge base.

        Returns:
            List of unique category names
        """
        # This would need to query the vector store for all unique categories
        # For now, return common ones
        return [
            "Business & Strategy",
            "Technology & Innovation",
            "AI & Machine Learning",
            "Software Development"
        ]
