"""
SQLite storage for user feedback.
"""
import sqlite3
from typing import List, Optional
from datetime import datetime
import uuid
from pathlib import Path
from .models import Feedback, FeedbackType, FeedbackStats
from ..config import config


class FeedbackStore:
    """
    SQLite-based storage for user feedback.
    """

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize feedback store.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path or config.feedback_db_path
        self._init_db()

    def _init_db(self):
        """Initialize the database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    response TEXT NOT NULL,
                    feedback_type TEXT NOT NULL,
                    correction TEXT,
                    comment TEXT,
                    chunk_ids TEXT,
                    created_at TEXT NOT NULL
                )
            """)

            # Index for querying by type
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_feedback_type
                ON feedback(feedback_type)
            """)

            # Index for date range queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_feedback_date
                ON feedback(created_at)
            """)

            conn.commit()

    def add(self, feedback: Feedback) -> str:
        """
        Add feedback to the store.

        Args:
            feedback: Feedback object

        Returns:
            Feedback ID
        """
        if not feedback.id:
            feedback.id = f"fb_{uuid.uuid4().hex[:12]}"

        with sqlite3.connect(self.db_path) as conn:
            data = feedback.to_dict()
            conn.execute("""
                INSERT INTO feedback (id, query, response, feedback_type, correction, comment, chunk_ids, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data["id"],
                data["query"],
                data["response"],
                data["feedback_type"],
                data["correction"],
                data["comment"],
                data["chunk_ids"],
                data["created_at"]
            ))
            conn.commit()

        return feedback.id

    def get(self, feedback_id: str) -> Optional[Feedback]:
        """
        Get feedback by ID.

        Args:
            feedback_id: Feedback ID

        Returns:
            Feedback object or None
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM feedback WHERE id = ?",
                (feedback_id,)
            )
            row = cursor.fetchone()

            if row:
                return Feedback.from_dict(dict(row))
            return None

    def get_by_type(
        self,
        feedback_type: FeedbackType,
        limit: int = 100
    ) -> List[Feedback]:
        """
        Get feedback by type.

        Args:
            feedback_type: Type of feedback
            limit: Maximum number of results

        Returns:
            List of Feedback objects
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM feedback
                WHERE feedback_type = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (feedback_type.value, limit))

            return [Feedback.from_dict(dict(row)) for row in cursor.fetchall()]

    def get_recent(self, limit: int = 100) -> List[Feedback]:
        """
        Get recent feedback.

        Args:
            limit: Maximum number of results

        Returns:
            List of Feedback objects
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM feedback
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,))

            return [Feedback.from_dict(dict(row)) for row in cursor.fetchall()]

    def get_stats(self) -> FeedbackStats:
        """
        Get aggregated feedback statistics.

        Returns:
            FeedbackStats object
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT feedback_type, COUNT(*) as count
                FROM feedback
                GROUP BY feedback_type
            """)

            stats = FeedbackStats()
            for row in cursor.fetchall():
                feedback_type, count = row
                stats.total_feedback += count

                if feedback_type == FeedbackType.THUMBS_UP.value:
                    stats.thumbs_up = count
                elif feedback_type == FeedbackType.THUMBS_DOWN.value:
                    stats.thumbs_down = count
                elif feedback_type == FeedbackType.CORRECTION.value:
                    stats.corrections = count
                elif feedback_type == FeedbackType.MISSING_INFO.value:
                    stats.missing_info = count

            return stats

    def get_negative_feedback_chunks(self) -> List[str]:
        """
        Get chunk IDs associated with negative feedback.

        Returns:
            List of chunk IDs
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT chunk_ids FROM feedback
                WHERE feedback_type = ?
            """, (FeedbackType.THUMBS_DOWN.value,))

            chunk_ids = []
            import json
            for row in cursor.fetchall():
                if row[0]:
                    ids = json.loads(row[0])
                    chunk_ids.extend(ids)

            return chunk_ids

    def search_queries(self, search_term: str, limit: int = 50) -> List[Feedback]:
        """
        Search feedback by query text.

        Args:
            search_term: Term to search for
            limit: Maximum results

        Returns:
            List of matching Feedback objects
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM feedback
                WHERE query LIKE ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (f"%{search_term}%", limit))

            return [Feedback.from_dict(dict(row)) for row in cursor.fetchall()]

    def count(self) -> int:
        """Get total feedback count."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM feedback")
            return cursor.fetchone()[0]
