"""
Data models for the feedback system.
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, List
import json


class FeedbackType(str, Enum):
    """Types of user feedback."""
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    CORRECTION = "correction"
    MISSING_INFO = "missing_info"


@dataclass
class Feedback:
    """User feedback on a chatbot response."""
    id: str
    query: str
    response: str
    feedback_type: FeedbackType
    correction: Optional[str] = None
    comment: Optional[str] = None
    chunk_ids: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "query": self.query,
            "response": self.response,
            "feedback_type": self.feedback_type.value,
            "correction": self.correction,
            "comment": self.comment,
            "chunk_ids": json.dumps(self.chunk_ids),
            "created_at": self.created_at.isoformat()
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Feedback":
        """Create from dictionary."""
        chunk_ids = data.get("chunk_ids", "[]")
        if isinstance(chunk_ids, str):
            chunk_ids = json.loads(chunk_ids)

        return cls(
            id=data["id"],
            query=data["query"],
            response=data["response"],
            feedback_type=FeedbackType(data["feedback_type"]),
            correction=data.get("correction"),
            comment=data.get("comment"),
            chunk_ids=chunk_ids,
            created_at=datetime.fromisoformat(data["created_at"]) if isinstance(data.get("created_at"), str) else data.get("created_at", datetime.utcnow())
        )


@dataclass
class FeedbackStats:
    """Aggregated feedback statistics."""
    total_feedback: int = 0
    thumbs_up: int = 0
    thumbs_down: int = 0
    corrections: int = 0
    missing_info: int = 0

    @property
    def satisfaction_rate(self) -> float:
        """Calculate satisfaction rate (thumbs_up / total)."""
        if self.total_feedback == 0:
            return 0.0
        return self.thumbs_up / self.total_feedback


@dataclass
class QueryPattern:
    """Pattern identified from feedback for improvement."""
    pattern: str
    feedback_type: FeedbackType
    count: int
    example_queries: List[str] = field(default_factory=list)
    suggested_action: Optional[str] = None
