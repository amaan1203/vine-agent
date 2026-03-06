"""
VINE-Agent v3: Event Scheduler
Runs background jobs (APScheduler) for continuous real-time execution.

Key design decisions for correctness:
1. The Rule Engine fires and IMMEDIATELY saves a raw alert (status=TRIGGERED).
   The alert count in /api/v1/alerts reflects this instantly.
2. The ProactivePipeline (LLM enrichment) runs in a separate Thread so
   it NEVER blocks the scheduler's main execution thread.
3. max_instances=3 prevents scheduler skips when a job takes > 1 minute.
"""

import logging
import threading
from datetime import datetime
from typing import Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

from alert_engine import RuleEngine, Alert
from alert_store import AlertStore

logger = logging.getLogger(__name__)


class VINEScheduler:
    """
    Manages all automated, recurring tasks for the autonomous precision agriculture agent.
    """

    def __init__(self, proactive_pipeline=None):
        self.scheduler = BackgroundScheduler()
        self.rule_engine = RuleEngine()
        self.alert_store = AlertStore()
        self.pipeline = proactive_pipeline
        self._register_jobs()

    def _register_jobs(self):
        # Core poll job — fires every 1 minute (reduced from 5min for demo)
        # max_instances=3 ensures subsequent ticks fire even if an LLM call is ongoing
        self.scheduler.add_job(
            self.evaluate_sensor_windows,
            trigger=IntervalTrigger(minutes=1),
            id="evaluate_sensor_windows",
            name="Poll 48h sliding windows and evaluate threshold rules",
            replace_existing=True,
            max_instances=3,            # ← key fix: allow up to 3 concurrent
            misfire_grace_time=30,      # ← tolerate 30s delay before skipping
        )
        self.scheduler.add_job(
            self._log_daily_job,
            trigger="cron",
            hour=0, minute=0,
            id="daily_raptor_rebuild",
            name="Daily RAPTOR Knowledge Base Rebuild",
        )
        logger.info("Scheduler jobs registered.")

    def evaluate_sensor_windows(self):
        """
        The core autonomous loop:
        1. Fetch latest rolling 48h window for all blocks.
        2. Evaluate against alert_rules.yaml  → O(ms), no LLM
        3. Persist the raw alert immediately  → /api/v1/alerts updates at once
        4. Kick off LLM enrichment in a background thread (non-blocking)
        """
        logger.info(f"[{datetime.utcnow().isoformat()}] Executing Layer 1 Rule Evaluation...")
        import random
        simulated_blocks = [
            {"block": "A", "variety": "Chardonnay",
             "vwc_min": random.uniform(18.0, 26.0), "temp_max": 85.0},
            {"block": "B", "variety": "Pinot Noir",
             "vwc_mean": random.uniform(19.0, 25.0), "temp_max": random.uniform(88.0, 99.0)},
            {"block": "C", "variety": "Cabernet Sauvignon",
             "vwc_min": 25.0, "et0_deficit_48h": random.uniform(0.3, 0.7)},
        ]

        for block_state in simulated_blocks:
            triggered_alerts = self.rule_engine.evaluate(block_state)
            for alert in triggered_alerts:
                if self.alert_store.is_in_cooldown(alert):
                    logger.debug(f"Alert {alert.alert_type} on {alert.block} in cooldown — skipping.")
                    continue

                logger.warning(f"🚨 RULE BREACH: {alert.severity} | {alert.alert_type} | Block {alert.block}")

                # ── Step A: Persist raw alert IMMEDIATELY (status=TRIGGERED) ──
                # This ensures /api/v1/alerts shows the count update right now,
                # even before the expensive LLM enrichment finishes.
                self.alert_store.save_alert(alert)

                # ── Step B: Enrich asynchronously in a background thread ──
                if self.pipeline:
                    enrichment_thread = threading.Thread(
                        target=self._enrich_alert_async,
                        args=(alert,),
                        daemon=True,
                    )
                    enrichment_thread.start()
                    logger.info(f"Enrichment thread started for {alert.alert_id}")

    def _enrich_alert_async(self, alert: Alert):
        """
        Runs the LLM-powered Proactive Pipeline in a background thread.
        Updates the alert's recommendation in SQLite when it completes.
        """
        try:
            logger.info(f"[Thread] Enriching alert {alert.alert_id}...")
            enriched = self.pipeline.generate_recommendation_for_alert(alert)

            # Patch the recommendation back into the stored row
            import sqlite3
            with sqlite3.connect(self.alert_store.sqlite_path) as conn:
                conn.execute(
                    "UPDATE alerts SET recommendation=?, reasoning=?, status=? WHERE alert_id=?",
                    (enriched.recommendation, enriched.reasoning, "ACTIVE", alert.alert_id)
                )
            logger.info(f"[Thread] Enrichment complete for {alert.alert_id}. Recommendation saved.")
        except Exception as e:
            logger.error(f"[Thread] Enrichment failed for {alert.alert_id}: {e}")

    def _log_daily_job(self):
        logger.info("Running daily RAPTOR rebuild... (Stub — wire Raptor.save() here)")

    def start(self):
        self.scheduler.start()
        logger.info("VINE-Agent Scheduler started. Autonomous execution is LIVE.")
        # Re-enrich any alerts that were left as TRIGGERED=null from a previous server session
        if self.pipeline:
            t = threading.Thread(target=self._reenrich_orphaned_alerts, daemon=True)
            t.start()

    def _reenrich_orphaned_alerts(self):
        """
        On startup: find alerts with status=TRIGGERED and recommendation=NULL.
        These are alerts whose enrichment threads died when the server was killed.
        Re-queues one enrichment thread per orphaned alert (serially, to avoid lock contention).
        """
        import sqlite3
        from alert_engine import Alert as AlertCls

        try:
            with sqlite3.connect(self.alert_store.sqlite_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT * FROM alerts WHERE recommendation IS NULL AND status = 'TRIGGERED' ORDER BY triggered_at ASC"
                ).fetchall()

            if not rows:
                logger.info("[Startup] No orphaned alerts found. All recommendations up to date.")
                return

            logger.info(f"[Startup] Found {len(rows)} orphaned alert(s) — re-enriching serially to avoid lock contention.")
            for row in rows:
                d = dict(row)
                alert = AlertCls(
                    alert_id=d["alert_id"], rule_id=d["rule_id"], block=d["block"],
                    variety=d["variety"], severity=d["severity"], alert_type=d["alert_type"],
                    description=f"Re-enriching: {d['alert_type']} on Block {d['block']}",
                    metric_name=d["metric_name"], metric_value=d["metric_value"],
                    threshold_value=d["threshold_value"], operator="lt", cooldown_hours=4,
                    triggered_at=datetime.fromisoformat(d["triggered_at"]),
                )
                logger.info(f"[Startup] Re-enriching {alert.alert_id}...")
                self._enrich_alert_async(alert)   # SERIAL — one at a time, no race on _EMBED_LOCK
                logger.info(f"[Startup] Done re-enriching {alert.alert_id}")
        except Exception as e:
            logger.error(f"[Startup] Failed during orphan re-enrichment: {e}", exc_info=True)

    def stop(self):
        self.scheduler.shutdown()
        logger.info("VINE-Agent Scheduler stopped.")
