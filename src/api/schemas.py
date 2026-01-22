"""
Pydantic models for API request/response schemas.
"""
from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str = Field(..., min_length=1, description="User's question")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")
    temperature: float = Field(default=0.7, ge=0, le=1, description="LLM temperature")


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
