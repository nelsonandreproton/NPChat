"""
Pydantic models for API request/response schemas.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from enum import Enum


class ConversationMessage(BaseModel):
    """A single message in conversation history."""
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str = Field(..., min_length=1, max_length=1000, description="User's question")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")
    temperature: float = Field(default=0.7, ge=0.0, le=1.0, description="LLM temperature")
    conversation_history: List[ConversationMessage] = Field(
        default_factory=list,
        max_length=20,
        description="Previous messages for multi-turn context"
    )

    @field_validator("message")
    @classmethod
    def message_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Message cannot be empty or whitespace only")
        return v.strip()


class Source(BaseModel):
    """Source reference from a blog post."""
    title: str
    author: str
    url: str
    published_date: Optional[str] = None


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    answer: str
    sources: List[Source]
    query: str
    cached: bool = False
    log_id: Optional[int] = None
    chunk_ids: List[str] = Field(default_factory=list)


class FeedbackType(str, Enum):
    """Types of feedback."""
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    CORRECTION = "correction"


class FeedbackRequest(BaseModel):
    """Request model for feedback endpoint."""
    query: str = Field(..., description="Original user query")
    response: str = Field(..., description="Chatbot's response")
    feedback_type: FeedbackType = Field(..., description="Type of feedback")
    correction: Optional[str] = Field(None, description="User's correction if applicable")
    comment: Optional[str] = Field(None, description="Additional comment")
    chunk_ids: List[str] = Field(default_factory=list, description="IDs of chunks used in response")
    log_id: Optional[int] = Field(None, description="Log entry ID for analytics update")


class FeedbackResponse(BaseModel):
    """Response model for feedback endpoint."""
    success: bool
    message: str
    feedback_id: Optional[str] = None


class SourceListResponse(BaseModel):
    """Response model for sources list endpoint."""
    sources: List[Source]
    total: int


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    ollama_connected: bool
    vector_store_count: int
    timestamp: str
