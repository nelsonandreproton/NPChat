"""
Response caching to reduce LLM calls.
"""
import sqlite3
import json
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
from ..config import config


class ResponseCache:
    """
    SQLite-based response cache to avoid redundant LLM calls.

    Caches responses based on query similarity and settings.
    """

    def __init__(self, db_path: str = None, ttl_hours: int = 24):
        """
        Initialize response cache.

        Args:
            db_path: Path to SQLite database
            ttl_hours: Time-to-live for cached responses in hours
        """
        self.db_path = db_path or str(config.data_dir / "response_cache.db")
        self.ttl_hours = ttl_hours
        self._init_db()

    def _init_db(self):
        """Initialize the database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS response_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_hash TEXT UNIQUE NOT NULL,
                query TEXT NOT NULL,
                settings_hash TEXT NOT NULL,
                response TEXT NOT NULL,
                sources TEXT,
                created_at TEXT NOT NULL,
                hit_count INTEGER DEFAULT 1
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_query_hash ON response_cache(query_hash)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_created_at ON response_cache(created_at)
        """)

        conn.commit()
        conn.close()

    def _hash_query(self, query: str, settings: Dict[str, Any]) -> str:
        """Generate a hash for the query + settings combination."""
        # Normalize query
        normalized = query.lower().strip()

        # Include relevant settings in hash
        settings_str = json.dumps({
            "top_k": settings.get("top_k", 5),
            "use_expansion": settings.get("use_expansion", True),
            "use_hybrid": settings.get("use_hybrid", True),
            "use_hyde": settings.get("use_hyde", False)
        }, sort_keys=True)

        combined = f"{normalized}|{settings_str}"
        return hashlib.sha256(combined.encode()).hexdigest()[:32]

    def _hash_settings(self, settings: Dict[str, Any]) -> str:
        """Generate a hash for settings only."""
        settings_str = json.dumps(settings, sort_keys=True)
        return hashlib.sha256(settings_str.encode()).hexdigest()[:16]

    def get(self, query: str, settings: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Get cached response if available and not expired.

        Args:
            query: User query
            settings: Current settings

        Returns:
            Cached response dict or None
        """
        query_hash = self._hash_query(query, settings)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check for cached response
        cursor.execute("""
            SELECT response, sources, created_at, id FROM response_cache
            WHERE query_hash = ?
        """, (query_hash,))

        row = cursor.fetchone()

        if row:
            response, sources, created_at, cache_id = row

            # Check if expired
            created = datetime.fromisoformat(created_at)
            if datetime.now(timezone.utc) - created < timedelta(hours=self.ttl_hours):
                # Update hit count
                cursor.execute("""
                    UPDATE response_cache SET hit_count = hit_count + 1 WHERE id = ?
                """, (cache_id,))
                conn.commit()
                conn.close()

                return {
                    "answer": response,
                    "sources": json.loads(sources) if sources else [],
                    "cached": True
                }
            else:
                # Expired, delete it
                cursor.execute("DELETE FROM response_cache WHERE id = ?", (cache_id,))
                conn.commit()

        conn.close()
        return None

    def set(
        self,
        query: str,
        settings: Dict[str, Any],
        response: str,
        sources: list
    ):
        """
        Cache a response.

        Args:
            query: User query
            settings: Current settings
            response: LLM response
            sources: List of sources
        """
        query_hash = self._hash_query(query, settings)
        settings_hash = self._hash_settings(settings)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO response_cache
            (query_hash, query, settings_hash, response, sources, created_at, hit_count)
            VALUES (?, ?, ?, ?, ?, ?, 1)
        """, (
            query_hash,
            query,
            settings_hash,
            response,
            json.dumps(sources),
            datetime.now(timezone.utc).isoformat()
        ))

        conn.commit()
        conn.close()

    def clear(self):
        """Clear all cached responses."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM response_cache")
        conn.commit()
        conn.close()

    def clear_expired(self):
        """Clear only expired responses."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cutoff = (datetime.now(timezone.utc) - timedelta(hours=self.ttl_hours)).isoformat()
        cursor.execute("DELETE FROM response_cache WHERE created_at < ?", (cutoff,))

        deleted = cursor.rowcount
        conn.commit()
        conn.close()

        return deleted

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                COUNT(*) as total_entries,
                SUM(hit_count) as total_hits,
                AVG(hit_count) as avg_hits_per_entry
            FROM response_cache
        """)

        row = cursor.fetchone()
        conn.close()

        return {
            "total_entries": row[0] or 0,
            "total_hits": row[1] or 0,
            "avg_hits_per_entry": round(row[2] or 0, 2)
        }

    def get_recent(self, limit: int = 20) -> list:
        """Get recent cached queries."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT query, hit_count, created_at FROM response_cache
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,))

        rows = cursor.fetchall()
        conn.close()

        return [{"query": r[0], "hits": r[1], "created": r[2]} for r in rows]
