"""
API routes for the chatbot.
"""
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from datetime import datetime, timezone
import json
import ollama

from .schemas import (
    ChatRequest, ChatResponse, Source,
    FeedbackRequest, FeedbackResponse,
    SourceListResponse, HealthResponse
)
from ..generation.enhanced_rag_chain import EnhancedRAGChain
from ..retrieval.vector_store import VectorStore
from ..analytics.response_cache import ResponseCache
from ..feedback.feedback_learner import FeedbackLearner
from ..config import config

router = APIRouter()

# Initialize components (lazy loading)
_rag_chain = None
_vector_store = None
_response_cache = None
_feedback_learner = None


def get_rag_chain() -> EnhancedRAGChain:
    """Get or create enhanced RAG chain instance."""
    global _rag_chain
    if _rag_chain is None:
        _rag_chain = EnhancedRAGChain(
            use_query_expansion=True,
            use_hybrid_search=True,
            use_logging=True
        )
    return _rag_chain


def get_vector_store() -> VectorStore:
    """Get or create vector store instance."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


def get_response_cache() -> ResponseCache:
    """Get or create response cache instance."""
    global _response_cache
    if _response_cache is None:
        _response_cache = ResponseCache()
    return _response_cache


def get_feedback_learner() -> FeedbackLearner:
    """Get or create feedback learner instance."""
    global _feedback_learner
    if _feedback_learner is None:
        _feedback_learner = FeedbackLearner()
    return _feedback_learner


def _sanitize_input(text: str, max_length: int = 1000) -> str:
    """Sanitize user input to prevent prompt injection."""
    # Truncate to max length
    text = text[:max_length]
    # Strip leading/trailing whitespace
    text = text.strip()
    # Remove null bytes
    text = text.replace('\x00', '')
    return text


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send a message and get a response from the chatbot.
    """
    try:
        sanitized = _sanitize_input(request.message)
        if not sanitized:
            raise HTTPException(status_code=400, detail="Message cannot be empty")

        rag_chain = get_rag_chain()
        cache = get_response_cache()

        # Check cache first
        settings = {"top_k": request.top_k, "use_expansion": True, "use_hybrid": True, "use_hyde": False}
        cached = cache.get(sanitized, settings)
        if cached:
            return ChatResponse(
                answer=cached["answer"],
                sources=[
                    Source(title=s.get("title", ""), author=s.get("author", ""), url=s.get("url", ""))
                    for s in cached.get("sources", [])
                ],
                query=sanitized,
                cached=True,
                log_id=None,
                chunk_ids=[]
            )

        result = rag_chain.query(
            question=sanitized,
            top_k=request.top_k,
            temperature=request.temperature,
            conversation_history=request.conversation_history
        )

        # Cache the result
        cache.set(sanitized, settings, result.answer, result.sources)

        return ChatResponse(
            answer=result.answer,
            sources=[
                Source(
                    title=s["title"],
                    author=s["author"],
                    url=s["url"],
                    published_date=s.get("published_date")
                )
                for s in result.sources
            ],
            query=result.query,
            cached=False,
            log_id=result.log_id,
            chunk_ids=result.chunk_ids
        )
    except HTTPException:
        raise
    except Exception as e:
        # Don't expose internal details
        raise HTTPException(status_code=500, detail="An error occurred processing your request")


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Send a message and get a streaming response from the chatbot.
    """
    sanitized = _sanitize_input(request.message)
    if not sanitized:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    def generate():
        try:
            from ..generation.rag_chain import RAGChain
            rag_chain = RAGChain()
            for event in rag_chain.query_stream(
                question=sanitized,
                top_k=request.top_k,
                temperature=request.temperature
            ):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'data': 'An error occurred'})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):
    """
    Submit feedback on a chatbot response and trigger automatic learning.
    """
    from ..feedback.store import FeedbackStore
    from ..feedback.models import Feedback, FeedbackType as FBType

    try:
        store = FeedbackStore()

        feedback = Feedback(
            id="",
            query=request.query,
            response=request.response,
            feedback_type=FBType(request.feedback_type.value),
            correction=request.correction,
            comment=request.comment
        )

        feedback_id = store.add(feedback)

        # Apply automatic learning
        learner = get_feedback_learner()
        cache = get_response_cache()
        is_positive = request.feedback_type.value == "thumbs_up"

        learner.process_feedback(
            query=request.query,
            is_positive=is_positive,
            chunk_ids=request.chunk_ids or [],
            cache=cache
        )

        return FeedbackResponse(
            success=True,
            message="Feedback received. Thank you!",
            feedback_id=feedback_id
        )
    except Exception as e:
        return FeedbackResponse(
            success=False,
            message="Failed to save feedback",
            feedback_id=None
        )


@router.get("/sources", response_model=SourceListResponse)
async def list_sources():
    """
    List all sources in the knowledge base.
    """
    try:
        vector_store = get_vector_store()
        # Get all documents with metadata
        collection = vector_store._collection
        all_data = collection.get(include=["metadatas"])

        seen_urls = set()
        sources = []
        for meta in (all_data.get("metadatas") or []):
            url = meta.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                sources.append(Source(
                    title=meta.get("title", ""),
                    author=meta.get("author", ""),
                    url=url,
                    published_date=meta.get("published_date")
                ))

        return SourceListResponse(sources=sources, total=len(sources))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to retrieve sources")


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Check the health of the chatbot service.
    """
    ollama_connected = False
    try:
        client = ollama.Client(host=config.ollama_base_url)
        client.list()
        ollama_connected = True
    except Exception:
        pass

    vector_count = 0
    try:
        vector_store = get_vector_store()
        vector_count = vector_store.count()
    except Exception:
        pass

    status = "healthy" if ollama_connected and vector_count > 0 else "degraded"

    return HealthResponse(
        status=status,
        ollama_connected=ollama_connected,
        vector_store_count=vector_count,
        timestamp=datetime.now(timezone.utc).isoformat()
    )
