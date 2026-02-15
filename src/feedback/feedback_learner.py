"""
Automatic feedback learning system.

Implements:
1. Cache invalidation on negative feedback
2. Chunk score boosting/penalizing
3. Auto-flagging repeated issues
4. Query mapping learning from positive feedback
"""
import sqlite3
import json
import hashlib
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
from ..config import config


class FeedbackLearner:
    """
    Learns from user feedback to improve the RAG system automatically.
    """

    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(config.data_dir / "feedback_learning.db")
        self._init_db()

    def _init_db(self):
        """Initialize the database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Chunk adjustments - boost or penalize specific chunks
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunk_adjustments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id TEXT UNIQUE NOT NULL,
                score_adjustment REAL DEFAULT 0,
                positive_count INTEGER DEFAULT 0,
                negative_count INTEGER DEFAULT 0,
                last_updated TEXT NOT NULL
            )
        """)

        # Flagged queries - queries needing review
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS flagged_queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                query_hash TEXT UNIQUE NOT NULL,
                negative_count INTEGER DEFAULT 1,
                flag_reason TEXT,
                status TEXT DEFAULT 'pending',
                created_at TEXT NOT NULL,
                resolved_at TEXT
            )
        """)

        # Query mappings - learned good query expansions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS query_mappings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_query TEXT NOT NULL,
                query_hash TEXT NOT NULL,
                successful_chunks TEXT,
                positive_count INTEGER DEFAULT 1,
                created_at TEXT NOT NULL,
                last_used TEXT NOT NULL
            )
        """)

        # Indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunk_id ON chunk_adjustments(chunk_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_query_hash ON flagged_queries(query_hash)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_mapping_hash ON query_mappings(query_hash)")

        conn.commit()
        conn.close()

    def _hash_query(self, query: str) -> str:
        """Generate hash for a query."""
        normalized = query.lower().strip()
        return hashlib.sha256(normalized.encode()).hexdigest()[:32]

    # =========================================================================
    # 1. CACHE INVALIDATION
    # =========================================================================

    def invalidate_cache_for_query(self, query: str, cache) -> bool:
        """
        Invalidate cached response for a query that received negative feedback.

        Args:
            query: The query to invalidate
            cache: ResponseCache instance

        Returns:
            True if cache was invalidated
        """
        try:
            # Get all possible cache entries for this query (any settings)
            conn = sqlite3.connect(cache.db_path)
            cursor = conn.cursor()

            # Find and delete matching cache entries
            query_lower = query.lower().strip()
            cursor.execute("""
                DELETE FROM response_cache
                WHERE LOWER(query) = ?
            """, (query_lower,))

            deleted = cursor.rowcount
            conn.commit()
            conn.close()

            return deleted > 0
        except Exception as e:
            print(f"[FeedbackLearner] Cache invalidation failed: {e}")
            return False

    # =========================================================================
    # 2. CHUNK BOOSTING/PENALIZING
    # =========================================================================

    def adjust_chunk_score(self, chunk_id: str, is_positive: bool):
        """
        Adjust the score for a chunk based on feedback.

        Positive feedback boosts the chunk, negative penalizes it.

        Args:
            chunk_id: The chunk identifier
            is_positive: True for positive feedback, False for negative
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check if chunk exists
        cursor.execute("SELECT * FROM chunk_adjustments WHERE chunk_id = ?", (chunk_id,))
        row = cursor.fetchone()

        adjustment = 0.1 if is_positive else -0.15  # Penalize more than boost

        if row:
            # Update existing
            if is_positive:
                cursor.execute("""
                    UPDATE chunk_adjustments
                    SET score_adjustment = score_adjustment + ?,
                        positive_count = positive_count + 1,
                        last_updated = ?
                    WHERE chunk_id = ?
                """, (adjustment, datetime.now(timezone.utc).isoformat(), chunk_id))
            else:
                cursor.execute("""
                    UPDATE chunk_adjustments
                    SET score_adjustment = score_adjustment + ?,
                        negative_count = negative_count + 1,
                        last_updated = ?
                    WHERE chunk_id = ?
                """, (adjustment, datetime.now(timezone.utc).isoformat(), chunk_id))
        else:
            # Insert new
            cursor.execute("""
                INSERT INTO chunk_adjustments
                (chunk_id, score_adjustment, positive_count, negative_count, last_updated)
                VALUES (?, ?, ?, ?, ?)
            """, (
                chunk_id,
                adjustment,
                1 if is_positive else 0,
                0 if is_positive else 1,
                datetime.now(timezone.utc).isoformat()
            ))

        conn.commit()
        conn.close()

    def get_chunk_adjustment(self, chunk_id: str) -> float:
        """Get the score adjustment for a chunk."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT score_adjustment FROM chunk_adjustments WHERE chunk_id = ?",
            (chunk_id,)
        )
        row = cursor.fetchone()
        conn.close()

        return row[0] if row else 0.0

    def get_all_chunk_adjustments(self) -> Dict[str, float]:
        """Get all chunk adjustments as a dictionary."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT chunk_id, score_adjustment FROM chunk_adjustments")
        rows = cursor.fetchall()
        conn.close()

        return {row[0]: row[1] for row in rows}

    def apply_adjustments_to_results(
        self,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Apply learned adjustments to retrieval results.

        Args:
            results: List of retrieval results

        Returns:
            Results with adjusted scores, re-sorted
        """
        adjustments = self.get_all_chunk_adjustments()

        if not adjustments:
            return results

        for result in results:
            chunk_id = result.get("id", "")
            if chunk_id in adjustments:
                # Apply adjustment to the score
                original_score = result.get("combined_score", result.get("semantic_score", 0.5))
                adjustment = adjustments[chunk_id]
                result["adjusted_score"] = max(0, min(1, original_score + adjustment))
                result["had_adjustment"] = True
            else:
                result["adjusted_score"] = result.get("combined_score", result.get("semantic_score", 0.5))
                result["had_adjustment"] = False

        # Re-sort by adjusted score
        results.sort(key=lambda x: x.get("adjusted_score", 0), reverse=True)

        return results

    # =========================================================================
    # 3. AUTO-FLAGGING REPEATED ISSUES
    # =========================================================================

    def flag_query_if_needed(self, query: str, threshold: int = 2) -> Optional[str]:
        """
        Flag a query for review if it has received multiple negative feedbacks.

        Args:
            query: The query to check
            threshold: Number of negatives before flagging

        Returns:
            Flag reason if flagged, None otherwise
        """
        query_hash = self._hash_query(query)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check existing flags
        cursor.execute(
            "SELECT negative_count, status FROM flagged_queries WHERE query_hash = ?",
            (query_hash,)
        )
        row = cursor.fetchone()

        if row:
            negative_count, status = row
            new_count = negative_count + 1

            cursor.execute("""
                UPDATE flagged_queries
                SET negative_count = ?
                WHERE query_hash = ?
            """, (new_count, query_hash))

            conn.commit()
            conn.close()

            if new_count >= threshold and status == 'pending':
                return f"Query has {new_count} negative feedbacks"
            return None
        else:
            # First negative - insert but don't flag yet
            cursor.execute("""
                INSERT INTO flagged_queries
                (query, query_hash, negative_count, flag_reason, status, created_at)
                VALUES (?, ?, 1, NULL, 'monitoring', ?)
            """, (query, query_hash, datetime.now(timezone.utc).isoformat()))

            conn.commit()
            conn.close()
            return None

    def update_flag_status(
        self,
        query: str,
        negative_count: int,
        threshold: int = 2
    ):
        """Update flag status based on negative count."""
        query_hash = self._hash_query(query)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if negative_count >= threshold:
            cursor.execute("""
                UPDATE flagged_queries
                SET status = 'pending',
                    flag_reason = ?
                WHERE query_hash = ?
            """, (f"Received {negative_count} negative feedbacks", query_hash))

        conn.commit()
        conn.close()

    def get_flagged_queries(self, status: str = 'pending') -> List[Dict[str, Any]]:
        """Get all flagged queries with given status."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT query, negative_count, flag_reason, created_at, status
            FROM flagged_queries
            WHERE status = ?
            ORDER BY negative_count DESC
        """, (status,))

        rows = cursor.fetchall()
        conn.close()

        return [{
            "query": row[0],
            "negative_count": row[1],
            "flag_reason": row[2],
            "created_at": row[3],
            "status": row[4]
        } for row in rows]

    def resolve_flag(self, query: str, resolution: str = 'resolved'):
        """Mark a flagged query as resolved."""
        query_hash = self._hash_query(query)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE flagged_queries
            SET status = ?, resolved_at = ?
            WHERE query_hash = ?
        """, (resolution, datetime.now(timezone.utc).isoformat(), query_hash))

        conn.commit()
        conn.close()

    # =========================================================================
    # 4. QUERY MAPPING LEARNING
    # =========================================================================

    def learn_successful_query(
        self,
        query: str,
        successful_chunk_ids: List[str]
    ):
        """
        Learn from a query that received positive feedback.

        Stores the query and which chunks were successful for future reference.

        Args:
            query: The original query
            successful_chunk_ids: IDs of chunks that led to good response
        """
        query_hash = self._hash_query(query)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check if mapping exists
        cursor.execute(
            "SELECT id, positive_count FROM query_mappings WHERE query_hash = ?",
            (query_hash,)
        )
        row = cursor.fetchone()

        chunks_json = json.dumps(successful_chunk_ids)
        now = datetime.now(timezone.utc).isoformat()

        if row:
            # Update existing
            cursor.execute("""
                UPDATE query_mappings
                SET positive_count = positive_count + 1,
                    successful_chunks = ?,
                    last_used = ?
                WHERE query_hash = ?
            """, (chunks_json, now, query_hash))
        else:
            # Insert new
            cursor.execute("""
                INSERT INTO query_mappings
                (original_query, query_hash, successful_chunks, positive_count, created_at, last_used)
                VALUES (?, ?, ?, 1, ?, ?)
            """, (query, query_hash, chunks_json, now, now))

        conn.commit()
        conn.close()

    def get_similar_successful_queries(self, query: str) -> List[Dict[str, Any]]:
        """
        Find similar queries that were successful.

        This can be used to boost chunks that worked for similar queries.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # For now, simple keyword matching
        # Could be enhanced with embedding similarity
        words = query.lower().split()

        results = []
        for word in words:
            if len(word) > 3:  # Skip short words
                cursor.execute("""
                    SELECT original_query, successful_chunks, positive_count
                    FROM query_mappings
                    WHERE LOWER(original_query) LIKE ?
                    ORDER BY positive_count DESC
                    LIMIT 5
                """, (f"%{word}%",))

                for row in cursor.fetchall():
                    results.append({
                        "query": row[0],
                        "chunk_ids": json.loads(row[1]) if row[1] else [],
                        "positive_count": row[2]
                    })

        conn.close()

        # Deduplicate
        seen = set()
        unique_results = []
        for r in results:
            if r["query"] not in seen:
                seen.add(r["query"])
                unique_results.append(r)

        return unique_results

    def get_learned_chunk_boosts(self, query: str) -> Dict[str, float]:
        """
        Get chunk boosts based on similar successful queries.

        Returns dict of chunk_id -> boost amount
        """
        similar = self.get_similar_successful_queries(query)

        boosts = {}
        for item in similar:
            boost_amount = 0.05 * item["positive_count"]  # More positives = more boost
            for chunk_id in item["chunk_ids"]:
                if chunk_id in boosts:
                    boosts[chunk_id] += boost_amount
                else:
                    boosts[chunk_id] = boost_amount

        return boosts

    # =========================================================================
    # COMBINED FEEDBACK PROCESSING
    # =========================================================================

    def process_feedback(
        self,
        query: str,
        is_positive: bool,
        chunk_ids: List[str],
        cache=None
    ) -> Dict[str, Any]:
        """
        Process feedback and apply all automatic actions.

        Args:
            query: The user query
            is_positive: True for positive feedback
            chunk_ids: IDs of chunks used in the response
            cache: ResponseCache instance (optional)

        Returns:
            Dict with actions taken
        """
        actions = {
            "cache_invalidated": False,
            "chunks_adjusted": [],
            "query_flagged": False,
            "flag_reason": None,
            "query_learned": False
        }

        if is_positive:
            # Positive feedback actions
            # 1. Boost the chunks that were used
            for chunk_id in chunk_ids:
                self.adjust_chunk_score(chunk_id, is_positive=True)
                actions["chunks_adjusted"].append(chunk_id)

            # 2. Learn the successful query-chunk mapping
            self.learn_successful_query(query, chunk_ids)
            actions["query_learned"] = True

        else:
            # Negative feedback actions
            # 1. Invalidate cache
            if cache:
                actions["cache_invalidated"] = self.invalidate_cache_for_query(query, cache)

            # 2. Penalize the chunks that were used
            for chunk_id in chunk_ids:
                self.adjust_chunk_score(chunk_id, is_positive=False)
                actions["chunks_adjusted"].append(chunk_id)

            # 3. Check if query should be flagged
            flag_reason = self.flag_query_if_needed(query)
            if flag_reason:
                actions["query_flagged"] = True
                actions["flag_reason"] = flag_reason

        return actions

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get learning statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Chunk adjustments
        cursor.execute("""
            SELECT
                COUNT(*) as total_chunks,
                SUM(CASE WHEN score_adjustment > 0 THEN 1 ELSE 0 END) as boosted_chunks,
                SUM(CASE WHEN score_adjustment < 0 THEN 1 ELSE 0 END) as penalized_chunks
            FROM chunk_adjustments
        """)
        chunk_stats = cursor.fetchone()

        # Flagged queries
        cursor.execute("""
            SELECT status, COUNT(*) FROM flagged_queries GROUP BY status
        """)
        flag_stats = {row[0]: row[1] for row in cursor.fetchall()}

        # Query mappings
        cursor.execute("SELECT COUNT(*), SUM(positive_count) FROM query_mappings")
        mapping_stats = cursor.fetchone()

        conn.close()

        return {
            "chunks": {
                "total_adjusted": chunk_stats[0] or 0,
                "boosted": chunk_stats[1] or 0,
                "penalized": chunk_stats[2] or 0
            },
            "flags": flag_stats,
            "mappings": {
                "total_queries": mapping_stats[0] or 0,
                "total_positive_signals": mapping_stats[1] or 0
            }
        }
