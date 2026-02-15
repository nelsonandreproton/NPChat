"""
Query logging for analytics and continuous improvement.
"""
import sqlite3
import json
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
from ..config import config


@dataclass
class QueryLog:
    """A logged query with metadata."""
    id: Optional[int]
    timestamp: str
    query: str
    expanded_query: Optional[str]
    retrieval_scores: List[float]
    avg_retrieval_score: float
    num_chunks_retrieved: int
    response_time_ms: int
    feedback: Optional[str]  # 'positive', 'negative', None
    model_used: str


class QueryLogger:
    """
    SQLite-based query logger for tracking user queries and retrieval performance.
    """

    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(config.data_dir / "query_logs.db")
        self._init_db()

    def _init_db(self):
        """Initialize the database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS query_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    query TEXT NOT NULL,
                    expanded_query TEXT,
                    retrieval_scores TEXT,
                    avg_retrieval_score REAL,
                    num_chunks_retrieved INTEGER,
                    response_time_ms INTEGER,
                    feedback TEXT,
                    model_used TEXT
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON query_logs(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_avg_score ON query_logs(avg_retrieval_score)")
            conn.commit()

    def log(
        self,
        query: str,
        retrieval_scores: List[float],
        response_time_ms: int,
        expanded_query: str = None,
        model_used: str = None
    ) -> int:
        """Log a query and its retrieval performance. Returns the log ID."""
        avg_score = sum(retrieval_scores) / len(retrieval_scores) if retrieval_scores else 0

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO query_logs
                (timestamp, query, expanded_query, retrieval_scores, avg_retrieval_score,
                 num_chunks_retrieved, response_time_ms, feedback, model_used)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now(timezone.utc).isoformat(),
                query,
                expanded_query,
                json.dumps(retrieval_scores),
                avg_score,
                len(retrieval_scores),
                response_time_ms,
                None,
                model_used or config.llm_model
            ))
            conn.commit()
            return cursor.lastrowid

    def update_feedback(self, log_id: int, feedback: str):
        """Update feedback for a logged query."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE query_logs SET feedback = ? WHERE id = ?", (feedback, log_id))
            conn.commit()

    def get_recent(self, limit: int = 100) -> List[QueryLog]:
        """Get recent query logs."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM query_logs ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._row_to_log(row) for row in rows]

    def get_low_score_queries(self, threshold: float = 0.5, limit: int = 50) -> List[QueryLog]:
        """Get queries with low retrieval scores (potential knowledge gaps)."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM query_logs WHERE avg_retrieval_score < ? ORDER BY avg_retrieval_score ASC LIMIT ?",
                (threshold, limit)
            ).fetchall()
        return [self._row_to_log(row) for row in rows]

    def get_negative_feedback_queries(self, limit: int = 50) -> List[QueryLog]:
        """Get queries that received negative feedback."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM query_logs WHERE feedback = 'negative' ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            ).fetchall()
        return [self._row_to_log(row) for row in rows]

    def get_common_queries(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get most common queries."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT query, COUNT(*) as count, AVG(avg_retrieval_score) as avg_score
                FROM query_logs GROUP BY query ORDER BY count DESC LIMIT ?
            """, (limit,)).fetchall()
        return [{"query": row[0], "count": row[1], "avg_score": row[2]} for row in rows]

    def get_stats(self) -> Dict[str, Any]:
        """Get overall statistics."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("""
                SELECT
                    COUNT(*) as total_queries,
                    AVG(avg_retrieval_score) as overall_avg_score,
                    AVG(response_time_ms) as avg_response_time,
                    SUM(CASE WHEN feedback = 'positive' THEN 1 ELSE 0 END) as positive_feedback,
                    SUM(CASE WHEN feedback = 'negative' THEN 1 ELSE 0 END) as negative_feedback
                FROM query_logs
            """).fetchone()
        return {
            "total_queries": row[0] or 0,
            "avg_retrieval_score": round(row[1] or 0, 3),
            "avg_response_time_ms": round(row[2] or 0, 0),
            "positive_feedback": row[3] or 0,
            "negative_feedback": row[4] or 0
        }

    def _row_to_log(self, row) -> QueryLog:
        """Convert a database row to a QueryLog object."""
        return QueryLog(
            id=row[0],
            timestamp=row[1],
            query=row[2],
            expanded_query=row[3],
            retrieval_scores=json.loads(row[4]) if row[4] else [],
            avg_retrieval_score=row[5] or 0,
            num_chunks_retrieved=row[6] or 0,
            response_time_ms=row[7] or 0,
            feedback=row[8],
            model_used=row[9]
        )
