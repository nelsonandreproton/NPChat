"""
API routes for the chatbot.
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from datetime import datetime
import json
import ollama

from .schemas import (
    ChatRequest, ChatResponse, Source,
    FeedbackRequest, FeedbackResponse,
    SourceListResponse, HealthResponse
)
from ..generation.rag_chain import RAGChain
from ..retrieval.vector_store import VectorStore
from ..config import config

router = APIRouter()

# Initialize components (lazy loading)
_rag_chain = None
_vector_store = None


def get_rag_chain() -> RAGChain:
    """Get or create RAG chain instance."""
    global _rag_chain
    if _rag_chain is None:
        _rag_chain = RAGChain()
    return _rag_chain


def get_vector_store() -> VectorStore:
    """Get or create vector store instance."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send a message and get a response from the chatbot.
    """
    try:
        rag_chain = get_rag_chain()
        result = rag_chain.query(
            question=request.message,
            top_k=request.top_k,
            temperature=request.temperature
        )

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
            query=result.query
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Send a message and get a streaming response from the chatbot.
    """
    def generate():
        try:
            rag_chain = get_rag_chain()
            for event in rag_chain.query_stream(
                question=request.message,
                top_k=request.top_k,
                temperature=request.temperature
            ):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):
    """
    Submit feedback on a chatbot response.
    """
    from ..feedback.store import FeedbackStore
    from ..feedback.models import Feedback, FeedbackType as FBType

    try:
        store = FeedbackStore()

        feedback = Feedback(
            id="",  # Will be auto-generated
            query=request.query,
            response=request.response,
            feedback_type=FBType(request.feedback_type.value),
            correction=request.correction,
            comment=request.comment
        )

        feedback_id = store.add(feedback)

        return FeedbackResponse(
            success=True,
            message="Feedback received. Thank you!",
            feedback_id=feedback_id
        )
    except Exception as e:
        return FeedbackResponse(
            success=False,
            message=f"Failed to save feedback: {str(e)}",
            feedback_id=None
        )


@router.get("/sources", response_model=SourceListResponse)
async def list_sources():
    """
    List all blog post sources in the knowledge base.
    """
    try:
        vector_store = get_vector_store()
        urls = vector_store.get_all_urls()

        # Get metadata for each URL (simplified - just returns URLs for now)
        sources = [
            Source(
                title="",  # Would need to query for full metadata
                author="",
                url=url,
                published_date=""
            )
            for url in urls
        ]

        return SourceListResponse(
            sources=sources,
            total=len(sources)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Check the health of the chatbot service.
    """
    # Check Ollama connection
    ollama_connected = False
    try:
        client = ollama.Client(host=config.ollama_base_url)
        client.list()
        ollama_connected = True
    except Exception:
        pass

    # Check vector store
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
        timestamp=datetime.utcnow().isoformat()
    )
