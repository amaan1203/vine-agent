from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Valid priority tiers
PRIORITY_TIERS = ("primary", "secondary", "tertiary")

# Valid source keys
SOURCE_KEYS = ("sensor_data", "raptor_kb", "drone_imagery")

class ContextPriority:

    def __init__(
        self,
        primary: str = "raptor_kb",
        secondary: str = "sensor_data",
        tertiary: str = "drone_imagery",
        reasoning: str = "",
    ):
        self.primary   = primary.strip().lower()
        self.secondary = secondary.strip().lower()
        self.tertiary  = tertiary.strip().lower()
        self.reasoning = reasoning.strip()

    @classmethod
    def from_plan_string(cls, plan_string: str) -> "ContextPriority":
        try:
            primary   = cls._extract(plan_string, "PRIMARY",   "raptor_kb")
            secondary = cls._extract(plan_string, "SECONDARY", "sensor_data")
            tertiary  = cls._extract(plan_string, "TERTIARY",  "drone_imagery")
            reasoning = cls._extract_reasoning(plan_string)
            return cls(primary, secondary, tertiary, reasoning)
        except Exception as e:
            logger.warning(f"ContextPriority parse failed ({e}), using default.")
            return cls()

    @staticmethod
    def _extract(text: str, tier: str, default: str) -> str:
        pattern = rf"{tier}\s*:\s*([a-z_]+)"
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            val = m.group(1).strip().lower()
            return val if val in SOURCE_KEYS else default
        return default

    @staticmethod
    def _extract_reasoning(text: str) -> str:
        m = re.search(r"Reasoning\s*:\s*(.+?)(?=\n(?:Plan:|NormalizedQuestion:)|$)",
                      text, re.IGNORECASE | re.DOTALL)
        return m.group(1).strip() if m else ""

    @property
    def ordered_sources(self):
        """Return sources in priority order (primary first)."""
        return [self.primary, self.secondary, self.tertiary]

    def __repr__(self):
        return (f"ContextPriority(primary={self.primary!r}, "
                f"secondary={self.secondary!r}, tertiary={self.tertiary!r})")


class ContextAssembler:

    # Character limit for tertiary section (avoid bloating prompt)
    TERTIARY_TRUNCATE = 800

    def assemble(
        self,
        raptor_evidence: str,
        sensor_block: str,
        drone_block: str,
        priority: Optional[ContextPriority] = None,
        query_time: Optional[datetime] = None,
        last_flight_date: Optional[str] = None,
        plan_string: Optional[str] = None,
    ) -> str:
        if priority is None and plan_string:
            priority = ContextPriority.from_plan_string(plan_string)
        if priority is None:
            priority = ContextPriority()

        now_str = (query_time or datetime.now()).strftime("%Y-%m-%d %H:%M UTC")
        flight_age = self._flight_age_str(last_flight_date)

        
        metadata = {
            "sensor_data":   f"[SOURCE: Live IoT Sensors | AS OF: {now_str} | FRESHNESS: Real-time]",
            "raptor_kb":     f"[SOURCE: Agricultural Knowledge Base (RAPTOR) | FRESHNESS: Indexed]",
            "drone_imagery": f"[SOURCE: Multispectral Drone Imagery | LAST FLIGHT: {flight_age}]",
        }

       
        content = {
            "sensor_data":   sensor_block   or "[SENSOR] No live sensor data available.",
            "raptor_kb":     raptor_evidence or "[RAPTOR] No knowledge base evidence retrieved.",
            "drone_imagery": drone_block     or "[DRONE] No drone imagery available.",
        }

        sections = []

        for tier_name, source in [
            ("PRIMARY",   priority.primary),
            ("SECONDARY", priority.secondary),
            ("TERTIARY",  priority.tertiary),
        ]:
            body = content.get(source, "[No data]")
            meta = metadata.get(source, "")

            if tier_name == "TERTIARY" and len(body) > self.TERTIARY_TRUNCATE:
                body = body[:self.TERTIARY_TRUNCATE] + "\n... [truncated — lower priority]"

            tier_block = (
                f"{'═' * 60}\n"
                f"◆ {tier_name} EVIDENCE — {source.upper().replace('_', ' ')}\n"
                f"  {meta}\n"
                f"{'─' * 60}\n"
                f"{body}\n"
            )
            sections.append(tier_block)

        header = (
            f"{'═' * 60}\n"
            f"MULTI-MODAL EVIDENCE CONTEXT\n"
            f"Priority routing: {priority.primary.upper()} → "
            f"{priority.secondary.upper()} → {priority.tertiary.upper()}\n"
        )
        if priority.reasoning:
            header += f"Planner reasoning: {priority.reasoning}\n"
        header += f"{'═' * 60}\n\n"

        footer = (
            f"\n{'─' * 60}\n"
            f"SOLVER INSTRUCTION: Weight your answer primarily on {tier_name_for(priority.primary)} evidence.\n"
            f"Use SECONDARY evidence for protocols and thresholds.\n"
            f"Use TERTIARY evidence for spatial/background context only.\n"
            f"Always cite your evidence source ([LIVE SENSOR], [RAPTOR], [DRONE]).\n"
            f"{'─' * 60}"
        )

        return header + "\n".join(sections) + footer

    @staticmethod
    def _flight_age_str(last_flight_date: Optional[str]) -> str:
        if not last_flight_date:
            return "Unknown"
        try:
            dt = datetime.strptime(last_flight_date, "%Y-%m-%d")
            delta = (datetime.now() - dt).days
            return f"{last_flight_date} ({delta}d ago)"
        except Exception:
            return last_flight_date


def tier_name_for(source_key: str) -> str:
    labels = {
        "sensor_data":   "LIVE SENSOR",
        "raptor_kb":     "KNOWLEDGE BASE",
        "drone_imagery": "DRONE IMAGERY",
    }
    return labels.get(source_key, source_key.upper())




def infer_priority_from_query(query: str) -> ContextPriority:
    q = query.lower()

    sensor_keywords = {"moisture", "vwc", "temperature", "irrigat", "sensor",
                       "soil", "water stress", "et0", "evapotranspiration", "co2"}
    drone_keywords  = {"ndvi", "ndre", "drone", "imagery", "canopy", "chlorosis",
                       "spectral", "aerial", "stress map", "color", "pale"}

    sensor_score = sum(1 for kw in sensor_keywords if kw in q)
    drone_score  = sum(1 for kw in drone_keywords  if kw in q)

    if sensor_score > drone_score and sensor_score > 0:
        return ContextPriority(
            primary="sensor_data",
            secondary="raptor_kb",
            tertiary="drone_imagery",
            reasoning=f"Query contains sensor-related keywords (score={sensor_score})",
        )
    elif drone_score > sensor_score and drone_score > 0:
        return ContextPriority(
            primary="drone_imagery",
            secondary="raptor_kb",
            tertiary="sensor_data",
            reasoning=f"Query contains drone/spectral keywords (score={drone_score})",
        )
    else:
        return ContextPriority(
            primary="raptor_kb",
            secondary="sensor_data",
            tertiary="drone_imagery",
            reasoning="General agronomy query — knowledge base prioritised",
        )
