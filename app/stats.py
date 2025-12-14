"""
Stats tracking module for tracking bot usage statistics.
Uses SQLite for thread-safe, atomic operations.
Tracks messages received/sent, tools used, describe operations, and media groups.
"""
import sqlite3
import os
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
from contextlib import contextmanager

STATS_DB = "data/stats.db"


class StatsTracker:
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if StatsTracker._initialized:
            return
        StatsTracker._initialized = True
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize the SQLite database and create tables if needed"""
        os.makedirs(os.path.dirname(STATS_DB), exist_ok=True)
        
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS stats_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    event_subtype TEXT,
                    extra_data TEXT,
                    timestamp TEXT NOT NULL
                )
            """)
            
            # Create indexes for efficient queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_user_timestamp 
                ON stats_events(user_id, timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_event_type_timestamp 
                ON stats_events(event_type, timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON stats_events(timestamp)
            """)
            conn.commit()
    
    @contextmanager
    def _get_connection(self):
        """Get a database connection with WAL mode for better concurrency"""
        conn = sqlite3.connect(STATS_DB, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging for concurrency
        conn.execute("PRAGMA busy_timeout=30000")  # 30 second timeout
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def _get_timestamp(self) -> str:
        """Get current UTC timestamp in ISO format"""
        return datetime.now(timezone.utc).isoformat()
    
    def _get_cutoff_timestamp(self, days: int) -> str:
        """Get timestamp for N days ago"""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        return cutoff.isoformat()
    
    def track_message_received(self, user_id: str, msg_type: str) -> None:
        """
        Track a received message.
        
        Args:
            user_id: The user's ID
            msg_type: Type of message - "text", "voice", "photo", or "document"
        """
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO stats_events (user_id, event_type, event_subtype, timestamp) VALUES (?, ?, ?, ?)",
                (user_id, "message_received", msg_type, self._get_timestamp())
            )
            conn.commit()
    
    def track_message_sent(self, user_id: str) -> None:
        """Track a sent message"""
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO stats_events (user_id, event_type, timestamp) VALUES (?, ?, ?)",
                (user_id, "message_sent", self._get_timestamp())
            )
            conn.commit()
    
    def track_tool_used(self, user_id: str, tool_name: str) -> None:
        """Track a tool usage"""
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO stats_events (user_id, event_type, event_subtype, timestamp) VALUES (?, ?, ?, ?)",
                (user_id, "tool_used", tool_name, self._get_timestamp())
            )
            conn.commit()
    
    def track_describe_used(self, user_id: str, describe_type: str) -> None:
        """
        Track a describe operation.
        
        Args:
            user_id: The user's ID
            describe_type: Type of describe - "image_anthropic", "image_openai", "pdf", "txt", "json", "docx", "xlsx"
        """
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO stats_events (user_id, event_type, event_subtype, timestamp) VALUES (?, ?, ?, ?)",
                (user_id, "describe_used", describe_type, self._get_timestamp())
            )
            conn.commit()
    
    def track_media_group_processed(self, user_id: str, photo_count: int) -> None:
        """Track a media group processing"""
        extra_data = json.dumps({"photo_count": photo_count})
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO stats_events (user_id, event_type, extra_data, timestamp) VALUES (?, ?, ?, ?)",
                (user_id, "media_group", extra_data, self._get_timestamp())
            )
            conn.commit()
    
    def get_user_stats(self, user_id: str, days: Optional[int] = None) -> Dict[str, Any]:
        """
        Get stats for a single user.
        
        Args:
            user_id: The user's ID
            days: If provided, only count events within this many days
            
        Returns:
            Dict with user's stats
        """
        with self._get_connection() as conn:
            # Build WHERE clause
            where_clause = "WHERE user_id = ?"
            params: List[Any] = [user_id]
            
            if days is not None:
                cutoff = self._get_cutoff_timestamp(days)
                where_clause += " AND timestamp >= ?"
                params.append(cutoff)
            
            # Messages received by type
            cursor = conn.execute(f"""
                SELECT event_subtype, COUNT(*) as count 
                FROM stats_events 
                {where_clause} AND event_type = 'message_received'
                GROUP BY event_subtype
            """, params)
            
            messages_received = {"text": 0, "voice": 0, "photo": 0, "document": 0, "total": 0}
            for row in cursor:
                if row["event_subtype"] in messages_received:
                    messages_received[row["event_subtype"]] = row["count"]
                    messages_received["total"] += row["count"]
            
            # Messages sent
            cursor = conn.execute(f"""
                SELECT COUNT(*) as count 
                FROM stats_events 
                {where_clause} AND event_type = 'message_sent'
            """, params)
            messages_sent = cursor.fetchone()["count"]
            
            # Tools used
            cursor = conn.execute(f"""
                SELECT event_subtype, COUNT(*) as count 
                FROM stats_events 
                {where_clause} AND event_type = 'tool_used'
                GROUP BY event_subtype
            """, params)
            
            tools_used = {}
            tools_total = 0
            for row in cursor:
                tools_used[row["event_subtype"]] = row["count"]
                tools_total += row["count"]
            
            # Describe used
            cursor = conn.execute(f"""
                SELECT event_subtype, COUNT(*) as count 
                FROM stats_events 
                {where_clause} AND event_type = 'describe_used'
                GROUP BY event_subtype
            """, params)
            
            describe_used = {}
            describe_total = 0
            for row in cursor:
                describe_used[row["event_subtype"]] = row["count"]
                describe_total += row["count"]
            
            # Media groups
            cursor = conn.execute(f"""
                SELECT COUNT(*) as count 
                FROM stats_events 
                {where_clause} AND event_type = 'media_group'
            """, params)
            media_groups = cursor.fetchone()["count"]
            
            return {
                "messages_received": messages_received,
                "messages_sent": messages_sent,
                "tools_used": tools_used,
                "tools_total": tools_total,
                "describe_used": describe_used,
                "describe_total": describe_total,
                "media_groups_processed": media_groups
            }
    
    def get_aggregated_stats(self, days: Optional[int] = None) -> Dict[str, Any]:
        """
        Get aggregated stats across all users.
        
        Args:
            days: If provided, only count events within this many days
            
        Returns:
            Dict with aggregated stats
        """
        with self._get_connection() as conn:
            # Build WHERE clause for time filtering
            where_clause = ""
            params: List[Any] = []
            
            if days is not None:
                cutoff = self._get_cutoff_timestamp(days)
                where_clause = "WHERE timestamp >= ?"
                params = [cutoff]
            
            # Total unique users
            cursor = conn.execute(f"""
                SELECT COUNT(DISTINCT user_id) as count 
                FROM stats_events
                {where_clause}
            """, params)
            total_users = cursor.fetchone()["count"]
            
            # Messages received by type
            time_filter = f"AND timestamp >= '{self._get_cutoff_timestamp(days)}'" if days else ""
            cursor = conn.execute(f"""
                SELECT event_subtype, COUNT(*) as count 
                FROM stats_events 
                WHERE event_type = 'message_received' {time_filter}
                GROUP BY event_subtype
            """)
            
            messages_received = {"text": 0, "voice": 0, "photo": 0, "document": 0, "total": 0}
            for row in cursor:
                if row["event_subtype"] in messages_received:
                    messages_received[row["event_subtype"]] = row["count"]
                    messages_received["total"] += row["count"]
            
            # Messages sent
            cursor = conn.execute(f"""
                SELECT COUNT(*) as count 
                FROM stats_events 
                WHERE event_type = 'message_sent' {time_filter}
            """)
            messages_sent = cursor.fetchone()["count"]
            
            # Tools used
            cursor = conn.execute(f"""
                SELECT event_subtype, COUNT(*) as count 
                FROM stats_events 
                WHERE event_type = 'tool_used' {time_filter}
                GROUP BY event_subtype
            """)
            
            tools_used = {}
            tools_total = 0
            for row in cursor:
                tools_used[row["event_subtype"]] = row["count"]
                tools_total += row["count"]
            
            # Describe used
            cursor = conn.execute(f"""
                SELECT event_subtype, COUNT(*) as count 
                FROM stats_events 
                WHERE event_type = 'describe_used' {time_filter}
                GROUP BY event_subtype
            """)
            
            describe_used = {}
            describe_total = 0
            for row in cursor:
                describe_used[row["event_subtype"]] = row["count"]
                describe_total += row["count"]
            
            # Media groups
            cursor = conn.execute(f"""
                SELECT COUNT(*) as count 
                FROM stats_events 
                WHERE event_type = 'media_group' {time_filter}
            """)
            media_groups = cursor.fetchone()["count"]
            
            return {
                "messages_received": messages_received,
                "messages_sent": messages_sent,
                "tools_used": tools_used,
                "tools_total": tools_total,
                "describe_used": describe_used,
                "describe_total": describe_total,
                "media_groups_processed": media_groups,
                "total_users": total_users
            }
    
    def get_users_ranked_by_activity(self, days: int = 30) -> List[Tuple[str, int]]:
        """
        Get users ranked by total activity in the specified time period.
        
        Args:
            days: Number of days to consider (default 30)
            
        Returns:
            List of (user_id, activity_count) tuples, sorted by activity desc
        """
        with self._get_connection() as conn:
            cutoff = self._get_cutoff_timestamp(days)
            
            cursor = conn.execute("""
                SELECT user_id, COUNT(*) as activity_count 
                FROM stats_events 
                WHERE timestamp >= ?
                GROUP BY user_id
                ORDER BY activity_count DESC
            """, (cutoff,))
            
            return [(row["user_id"], row["activity_count"]) for row in cursor]


# Global singleton instance
stats_tracker = StatsTracker()
