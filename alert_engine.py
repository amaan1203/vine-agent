"""
VINE-Agent v3: Alert Trigger Engine (Layer 1)
Evaluates real-time precision agriculture sensor blocks against configurable YAML rules.
Produces Trigger Events (Alerts) to be handled by the Planner Agent.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class Alert:
    """Represents a triggered agrarian alert."""
    alert_id: str                   # Unique ID for the trigger instance
    rule_id: str                    # ID of the rule from alert_rules.yaml
    block: str                      # Vineyard block ID (e.g., 'A')
    variety: str                    # e.g., 'Pinot Noir'
    severity: str                   # CRITICAL, HIGH, MEDIUM, LOW
    alert_type: str                 # e.g., 'IRRIGATION_EMERGENCY'
    description: str                # Human-readable rule text
    metric_name: str                # e.g., 'vwc_min'
    metric_value: float             # e.g., 18.2
    threshold_value: float          # e.g., 20.0
    operator: str                   # 'lt' or 'gt'
    cooldown_hours: int             # Time before re-firing the same alert
    triggered_at: datetime = field(default_factory=datetime.utcnow)
    
    # Optional fields populated later by ProactiveRecommender
    recommendation: Optional[str] = None
    reasoning: Optional[str] = None
    status: str = "TRIGGERED"       # TRIGGERED -> ACTIVE -> ACKNOWLEDGED -> RESOLVED


class RuleEngine:
    """
    Layer 1 Alert Engine: Fast, deterministic, zero-LLM-cost thresholding.
    Loads YAML rules and evaluates SensorContextBlock dictionaries instantly.
    """

    def __init__(self, rules_path: str = "alert_rules.yaml"):
        self.rules_path = rules_path
        self.rules = self._load_rules()
        logger.info(f"Loaded {len(self.rules)} rules from {rules_path}")

    def _load_rules(self) -> List[dict]:
        try:
            with open(self.rules_path, 'r') as f:
                config = yaml.safe_load(f)
                return config.get("rules", [])
        except Exception as e:
            logger.error(f"Failed to load rules from {self.rules_path}: {e}")
            return []

    def evaluate(self, sensor_block: dict) -> List[Alert]:
        """
        Evaluate a single block's real-time state against all applicable rules.
        Expected sensor_block dict mirrors SensorContextBlock fields:
        {
            "block": "A",
            "variety": "Pinot Noir",
            "vwc_min": 19.5,
            ...
        }
        """
        triggered_alerts = []
        block_id = sensor_block.get("block", "Unknown")
        variety = sensor_block.get("variety", "unknown").lower()

        for r in self.rules:
            # Check variety filter
            rule_vars = [v.lower() for v in r.get("varieties", ["all"])]
            if "all" not in rule_vars and variety not in rule_vars:
                continue

            # Check metric existence
            metric = r.get("metric")
            if metric not in sensor_block:
                continue

            val = sensor_block[metric]
            threshold = r.get("threshold", 0.0)
            op = r.get("operator", "gt")

            # Evaluate
            is_triggered = False
            if op == "gt" and val > threshold:
                is_triggered = True
            elif op == "lt" and val < threshold:
                is_triggered = True

            if is_triggered:
                # Generate a unique stable ID based on timestamp window
                # In production, use a hash, here timestamp to the minute
                tstamp = datetime.utcnow().strftime("%Y%m%d%H%M")
                alert_id = f"alert-{block_id}-{r['rule_id']}-{tstamp}"
                
                alert = Alert(
                    alert_id=alert_id,
                    rule_id=r['rule_id'],
                    block=block_id,
                    variety=sensor_block.get("variety", "unknown"),
                    severity=r.get("severity", "MEDIUM"),
                    alert_type=r.get("alert_type", "GENERAL"),
                    description=r.get("description", "Threshold breached."),
                    metric_name=metric,
                    metric_value=float(val),
                    threshold_value=float(threshold),
                    operator=op,
                    cooldown_hours=r.get("cooldown_hours", 1),
                )
                triggered_alerts.append(alert)

        return triggered_alerts


if __name__ == "__main__":
    # Simple self-test
    logging.basicConfig(level=logging.INFO)
    engine = RuleEngine()
    test_block = {
        "block": "A",
        "variety": "Chardonnay",
        "vwc_min": 18.5,       # Should trigger critical_moisture_deficit (<20)
        "temp_max": 97.0,      # Should trigger high_heat_warning (>93) 
        "et0_deficit_48h": 0.6 # Should trigger high_et0_deficit (>0.5)
    }
    alerts = engine.evaluate(test_block)
    for a in alerts:
        print(f"[🔥 TRIGGER] {a.severity} | {a.alert_type} | {a.metric_name}={a.metric_value} ({a.operator} {a.threshold_value})")
