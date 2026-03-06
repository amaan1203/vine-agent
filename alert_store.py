"""
VINE-Agent v3: Alert Store (SQLite + Redis)
Handles durable persistence of triggered alerts, alert lifecycle management,
and sub-millisecond deduplication (cooldowns) via Redis.
"""

import json
import logging
import sqlite3
from datetime import datetime
from typing import List, Optional

import redis

from alert_engine import Alert

logger = logging.getLogger(__name__)


class AlertStore:
    def __init__(self, sqlite_path: str = "vine_alerts.db", redis_url: str = "redis://localhost:6379/0"):
        self.sqlite_path = sqlite_path
        self._init_sqlite()
        
        try:
            self.redis = redis.from_url(redis_url, decode_responses=True)
            self.redis.ping()
            self.redis_available = True
            logger.info("Connected to Redis for alert deduplication.")
        except redis.ConnectionError:
            self.redis_available = False
            logger.warning("Redis not available. Deduplication will fall back to SQLite (slower).")

    def _init_sqlite(self):
        """Create the alerts table if it doesn't exist."""
        query = """
        CREATE TABLE IF NOT EXISTS alerts (
            alert_id TEXT PRIMARY KEY,
            rule_id TEXT,
            block TEXT,
            variety TEXT,
            severity TEXT,
            alert_type TEXT,
            metric_name TEXT,
            metric_value REAL,
            threshold_value REAL,
            triggered_at TIMESTAMP,
            status TEXT,
            recommendation TEXT,
            reasoning TEXT
        )
        """
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.execute(query)

    def is_in_cooldown(self, alert: Alert) -> bool:
        """Check if an identical alert has fired recently (prevent spam)."""
        key = f"ALERT:{alert.alert_type}:{alert.block}"
        
        if self.redis_available:
            return bool(self.redis.exists(key))
        else:
            # Slower SQLite fallback
            query = """
            SELECT triggered_at FROM alerts 
            WHERE alert_type = ? AND block = ? 
            ORDER BY triggered_at DESC LIMIT 1
            """
            with sqlite3.connect(self.sqlite_path) as conn:
                cur = conn.execute(query, (alert.alert_type, alert.block))
                row = cur.fetchone()
                if row:
                    last_fired = datetime.fromisoformat(row[0])
                    hours_since = (datetime.utcnow() - last_fired).total_seconds() / 3600
                    return hours_since < alert.cooldown_hours
            return False

    def save_alert(self, alert: Alert):
        """Save a new alert to SQLite and set the Redis cooldown key."""
        # 1. Deduplication Cache
        key = f"ALERT:{alert.alert_type}:{alert.block}"
        if self.redis_available:
            # Set key with TTL = cooldown_hours * 3600 seconds
            self.redis.setex(key, int(alert.cooldown_hours * 3600), alert.triggered_at.isoformat())

        # 2. Durable Storage
        query = """
        INSERT INTO alerts 
        (alert_id, rule_id, block, variety, severity, alert_type, metric_name, 
         metric_value, threshold_value, triggered_at, status, recommendation, reasoning)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            alert.alert_id,
            alert.rule_id,
            alert.block,
            alert.variety,
            alert.severity,
            alert.alert_type,
            alert.metric_name,
            alert.metric_value,
            alert.threshold_value,
            alert.triggered_at.isoformat(),
            alert.status,
            alert.recommendation,
            alert.reasoning
        )
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.execute(query, params)
            
        logger.info(f"Saved {alert.severity} alert {alert.alert_id} to store.")

    def get_active_alerts(self, limit: int = 50) -> List[dict]:
        """Fetch ongoing unresolved alerts for the dashboard."""
        query = "SELECT * FROM alerts WHERE status IN ('TRIGGERED', 'ACTIVE') ORDER BY triggered_at DESC LIMIT ?"
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute(query, (limit,))
            return [dict(row) for row in cur.fetchall()]

    def update_status(self, alert_id: str, new_status: str):
        """Lifecycle transition: TRIGGERED -> ACKNOWLEDGED -> RESOLVED"""
        query = "UPDATE alerts SET status = ? WHERE alert_id = ?"
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.execute(query, (new_status, alert_id))
