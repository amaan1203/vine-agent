"""
VINE-Agent: Sensor Stream Module
=================================
Handles real-time and simulated sensor data from IoT devices deployed at
Iron Horse Vineyards (soil moisture, temperature, CO2, humidity).

Architecture role:
  - SlidingWindowExtractor: converts raw sensor readings → SensorContextBlock
  - SensorContextBlock: a structured, LLM-ready text description of live conditions
  - generate_live_sensor_context(): POC simulator (no MQTT broker needed)
  - generate_synthetic_sensor_csv(): writes a realistic demo CSV to disk

Production path (future):
  IoT device → LoRaWAN → MQTT broker → MQTTSensorClient → SlidingWindowExtractor
  → SensorContextBlock → injected into Solver prompt (NOT into RAPTOR)

Why NOT RAPTOR for real-time data?
  RAPTOR requires UMAP + GMM re-clustering on every insert → O(N²), takes minutes.
  5-minute sensor ticks would make the tree unusable. Instead:
    - Real-time readings → direct Solver injection (this module)
    - Weekly rolling summaries → RAPTOR leaf nodes (handled in data_loader.py)
"""

from __future__ import annotations

import csv
import logging
import os
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# Vineyard block definitions (Iron Horse Vineyards simulation)
# ─────────────────────────────────────────────────────────────────────────────

VINEYARD_BLOCKS: List[Dict] = [
    {"block": "A", "variety": "Chardonnay", "area_acres": 12.4, "soil": "Goldridge sandy loam"},
    {"block": "B", "variety": "Pinot Noir",  "area_acres": 8.7,  "soil": "Sebastopol clay loam"},
    {"block": "C", "variety": "Chardonnay", "area_acres": 10.1, "soil": "Goldridge sandy loam"},
    {"block": "D", "variety": "Pinot Noir",  "area_acres": 9.3,  "soil": "Goldridge sandy loam"},
    {"block": "E", "variety": "Pinot Noir",  "area_acres": 7.8,  "soil": "Sebastopol clay loam"},
    {"block": "F", "variety": "Chardonnay", "area_acres": 11.2, "soil": "Mixed loam"},
    {"block": "G", "variety": "Pinot Noir",  "area_acres": 6.5,  "soil": "Goldridge sandy loam"},
]

# ─────────────────────────────────────────────────────────────────────────────
# SensorContextBlock: structured live-sensor representation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SensorContextBlock:
    """
    Structured representation of a 48-hour sensor window for one vineyard block.
    Converted to an LLM-ready text block for injection into the Solver prompt.
    """
    block: str
    variety: str
    window_start: str
    window_end: str

    # Soil moisture (VWC %)
    vwc_mean: float
    vwc_min: float
    vwc_max: float
    vwc_trend: str        # "rising" | "stable" | "falling"
    hours_below_threshold: float  # hours where VWC < 25%

    # Temperature (°F)
    temp_mean: float
    temp_max: float

    # CO2 (ppm)
    co2_mean: float
    co2_max: float

    # ET₀ deficit (inches accumulated over window)
    et0_deficit_inches: float

    # Derived alerts
    alerts: List[str] = field(default_factory=list)

    def compute_alerts(self):
        """Derive agronomic alerts from sensor window stats."""
        self.alerts = []
        if self.vwc_min < 20.0:
            self.alerts.append("CRITICAL_MOISTURE_DEFICIT")
        elif self.vwc_min < 25.0:
            self.alerts.append("IRRIGATION_NEEDED")
        if self.temp_max > 95.0:
            self.alerts.append("EXTREME_HEAT_STRESS")
        elif self.temp_max > 88.0:
            self.alerts.append("HEAT_STRESS")
        if self.co2_max > 500.0:
            self.alerts.append("CO2_ANOMALY")
        if self.et0_deficit_inches > 1.2:
            self.alerts.append("HIGH_ET0_DEFICIT")

    def to_prompt_string(self) -> str:
        """
        Format as a clearly labelled, LLM-readable block.
        This goes into the SOLVER prompt as PRIMARY/SECONDARY context.
        """
        alert_str = (", ".join(f"⚠ {a}" for a in self.alerts)
                     if self.alerts else "✓ No active alerts")
        trend_arrow = {"rising": "↑", "stable": "→", "falling": "↓"}.get(self.vwc_trend, "?")

        return (
            f"╔══ [LIVE SENSOR — Block {self.block} ({self.variety})] ══════════════╗\n"
            f"║  Window: {self.window_start}  →  {self.window_end}\n"
            f"║  Source: IoT sensors (LoRaWAN/MQTT, 5-min resolution)\n"
            f"║  ─────────────────────────────────────────────────────\n"
            f"║  Soil VWC:     avg={self.vwc_mean:.1f}%  min={self.vwc_min:.1f}%  "
            f"max={self.vwc_max:.1f}%  trend={trend_arrow}\n"
            f"║  Hours below 25% VWC threshold: {self.hours_below_threshold:.0f}h\n"
            f"║  Soil Temp:    avg={self.temp_mean:.1f}°F  peak={self.temp_max:.1f}°F\n"
            f"║  CO₂:         avg={self.co2_mean:.0f}ppm  peak={self.co2_max:.0f}ppm\n"
            f"║  ET₀ deficit: {self.et0_deficit_inches:.2f} inches accumulated (48h)\n"
            f"║  ─────────────────────────────────────────────────────\n"
            f"║  ALERTS: {alert_str}\n"
            f"╚══════════════════════════════════════════════════════╝"
        )


# ─────────────────────────────────────────────────────────────────────────────
# SlidingWindowExtractor: CSV / dict → SensorContextBlock
# ─────────────────────────────────────────────────────────────────────────────

class SlidingWindowExtractor:
    """
    Converts raw sensor readings (list of dicts or CSV rows) from one block
    into a SensorContextBlock summary over a configurable time window.

    Expected row keys: date, time, block, variety, vwc_pct, temp_f, co2_ppm, et0_in
    """

    def __init__(self, window_hours: int = 48):
        self.window_hours = window_hours

    def extract(self, rows: List[Dict], block_meta: Dict) -> SensorContextBlock:
        """
        rows: list of sensor reading dicts, all from the same block.
        block_meta: {"block": str, "variety": str}
        """
        if not rows:
            raise ValueError("No sensor rows provided.")

        vwcs  = [float(r["vwc_pct"]) for r in rows if r.get("vwc_pct")]
        temps = [float(r["temp_f"])  for r in rows if r.get("temp_f")]
        co2s  = [float(r["co2_ppm"]) for r in rows if r.get("co2_ppm")]
        et0s  = [float(r.get("et0_in", 0.0)) for r in rows]

        # Trend: compare first quartile avg vs last quartile avg
        q = max(1, len(vwcs) // 4)
        early_avg = sum(vwcs[:q]) / q
        late_avg  = sum(vwcs[-q:]) / q
        if late_avg - early_avg > 1.5:
            trend = "rising"
        elif early_avg - late_avg > 1.5:
            trend = "falling"
        else:
            trend = "stable"

        hours_below = sum(1 for v in vwcs if v < 25.0) * (self.window_hours / max(len(vwcs), 1))

        block = SensorContextBlock(
            block=block_meta["block"],
            variety=block_meta["variety"],
            window_start=rows[0].get("date", "?") + " " + rows[0].get("time", ""),
            window_end=rows[-1].get("date", "?") + " " + rows[-1].get("time", ""),
            vwc_mean=sum(vwcs) / len(vwcs),
            vwc_min=min(vwcs),
            vwc_max=max(vwcs),
            vwc_trend=trend,
            hours_below_threshold=hours_below,
            temp_mean=sum(temps) / len(temps) if temps else 70.0,
            temp_max=max(temps) if temps else 70.0,
            co2_mean=sum(co2s) / len(co2s) if co2s else 415.0,
            co2_max=max(co2s) if co2s else 415.0,
            et0_deficit_inches=sum(et0s),
        )
        block.compute_alerts()
        return block

    def extract_from_csv(self, csv_path: str) -> List[SensorContextBlock]:
        """Load a sensor CSV and extract one SensorContextBlock per block."""
        if not os.path.exists(csv_path):
            logger.warning(f"Sensor CSV not found: {csv_path}")
            return []
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))

        # Group by block
        grouped: Dict[str, List[Dict]] = {}
        for row in rows:
            b = row.get("block", "Unknown")
            grouped.setdefault(b, []).append(row)

        blocks_meta = {bm["block"]: bm for bm in VINEYARD_BLOCKS}

        results = []
        for block_id, block_rows in grouped.items():
            # Use the last window_hours worth of rows (approx via index)
            # Assume 5-min intervals → 12 readings/hour
            n_readings = self.window_hours * 12
            recent = block_rows[-n_readings:]
            meta = blocks_meta.get(block_id, {"block": block_id, "variety": "Unknown"})
            try:
                ctx = self.extract(recent, meta)
                results.append(ctx)
            except Exception as e:
                logger.warning(f"Could not extract block {block_id}: {e}")
        return results


# ─────────────────────────────────────────────────────────────────────────────
# POC Simulator: generate_live_sensor_context()
# ─────────────────────────────────────────────────────────────────────────────

def generate_live_sensor_context(
    n_blocks: int = 3,
    scenario: str = "mixed",   # "healthy" | "stress" | "mixed"
    as_string: bool = True,
) -> str | List[SensorContextBlock]:
    """
    Generate realistic live sensor context blocks for the POC demo.
    Returns a formatted string ready for Solver injection (or list of blocks).

    Scenarios:
      "healthy" – most blocks fine, one mild deficit
      "stress"  – multiple blocks below threshold, heat event
      "mixed"   – mix of healthy and stressed blocks (default)
    """
    now = datetime.now()
    window_start = (now - timedelta(hours=48)).strftime("%Y-%m-%d %H:%M UTC")
    window_end   = now.strftime("%Y-%m-%d %H:%M UTC")

    # Base VWC profiles per scenario
    profiles = {
        "healthy": [(32, 28, 35, "stable"),   (31, 27, 34, "rising"),  (30, 26, 33, "stable")],
        "stress":  [(19, 15, 24, "falling"),  (17, 13, 22, "falling"), (21, 17, 26, "falling")],
        "mixed":   [(22, 17, 28, "falling"),  (31, 27, 35, "stable"),  (19, 14, 24, "falling")],
    }
    vwc_configs = profiles.get(scenario, profiles["mixed"])

    blocks: List[SensorContextBlock] = []
    for i, bm in enumerate(VINEYARD_BLOCKS[:n_blocks]):
        vwc_mean, vwc_min, vwc_max, trend = vwc_configs[i % len(vwc_configs)]
        # Add small noise
        vwc_mean += random.uniform(-1.5, 1.5)
        vwc_min  += random.uniform(-1.0, 0.5)
        # Heat event more likely in stress scenario
        temp_max = 95.0 if scenario == "stress" else random.choice([76, 82, 88, 91, 93])
        temp_mean = temp_max - random.uniform(8, 14)
        et0_deficit = random.uniform(0.6, 2.1) if scenario != "healthy" else random.uniform(0.2, 0.8)

        ctx = SensorContextBlock(
            block=bm["block"],
            variety=bm["variety"],
            window_start=window_start,
            window_end=window_end,
            vwc_mean=round(vwc_mean, 1),
            vwc_min=round(vwc_min, 1),
            vwc_max=round(vwc_max, 1),
            vwc_trend=trend,
            hours_below_threshold=round(max(0, (25 - vwc_min) * 3.5), 1),
            temp_mean=round(temp_mean, 1),
            temp_max=round(temp_max, 1),
            co2_mean=round(random.uniform(412, 435), 1),
            co2_max=round(random.uniform(438, 460), 1),
            et0_deficit_inches=round(et0_deficit, 2),
        )
        ctx.compute_alerts()
        blocks.append(ctx)

    if as_string:
        header = (
            f"[LIVE SENSOR DATA — {len(blocks)} blocks — "
            f"as of {window_end}]\n\n"
        )
        return header + "\n\n".join(b.to_prompt_string() for b in blocks)
    return blocks


# ─────────────────────────────────────────────────────────────────────────────
# POC Simulator: generate_synthetic_sensor_csv()
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_sensor_csv(
    path: str = "/tmp/vine_sensor.csv",
    n_days: int = 90,
    heat_events: Optional[List[str]] = None,
) -> str:
    """
    Write a realistic 90-day sensor CSV to disk.
    Simulates: seasonal VWC decline, irrigation events, heat spikes.
    Returns the path to the written file.
    """
    heat_events = heat_events or ["2024-07-14", "2024-07-15", "2024-08-03"]

    fieldnames = ["date", "time", "block", "variety", "vwc_pct",
                  "temp_f", "co2_ppm", "et0_in", "humidity_pct"]

    start_date = datetime(2024, 5, 1)
    rows: List[Dict] = []

    for block_meta in VINEYARD_BLOCKS:
        vwc = 35.0  # start well-watered
        for day_offset in range(n_days):
            date = start_date + timedelta(days=day_offset)
            date_str = date.strftime("%Y-%m-%d")
            is_heat = date_str in heat_events
            # Daily VWC decline (ET-driven), with occasional irrigation event
            daily_et = random.uniform(0.18, 0.35) * (1.4 if is_heat else 1.0)
            vwc -= daily_et * 2.5
            if vwc < 20.0 or (day_offset % 7 == 4):  # irrigation every ~7 days or below threshold
                vwc = min(38.0, vwc + random.uniform(8, 14))
            vwc = max(12.0, min(42.0, vwc))

            # Write 3 readings per day (morning, noon, evening)
            for hour, time_str in [(6, "06:00"), (12, "12:00"), (18, "18:00")]:
                noise = random.uniform(-0.8, 0.8)
                temp_base = 65 + 18 * (hour / 24.0)
                temp = round(temp_base + random.uniform(-3, 5) + (15 if is_heat else 0), 1)
                rows.append({
                    "date": date_str,
                    "time": time_str,
                    "block": block_meta["block"],
                    "variety": block_meta["variety"],
                    "vwc_pct": round(vwc + noise, 2),
                    "temp_f": temp,
                    "co2_ppm": round(random.uniform(410, 455), 1),
                    "et0_in": round(daily_et / 3, 4),
                    "humidity_pct": round(random.uniform(55, 85), 1),
                })

    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Synthetic sensor CSV written: {path} ({len(rows)} rows)")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Production stub: MQTT client (future integration)
# ─────────────────────────────────────────────────────────────────────────────

class MQTTSensorClient:
    """
    Production MQTT client for live sensor ingestion.
    Subscribes to 'vine/sensors/#' topic and feeds SlidingWindowExtractor.

    Requires: paho-mqtt (pip install paho-mqtt)
    Usage:
        client = MQTTSensorClient(broker="mqtt.vine.nrp.ai", port=1883)
        client.start()
        ctx_blocks = client.get_latest_context()
    """

    def __init__(self, broker: str, port: int = 1883, window_hours: int = 48):
        self.broker = broker
        self.port = port
        self.extractor = SlidingWindowExtractor(window_hours=window_hours)
        self._buffer: Dict[str, List[Dict]] = {}
        self._client = None

    def start(self):
        try:
            import paho.mqtt.client as mqtt
        except ImportError:
            raise ImportError("Install paho-mqtt: pip install paho-mqtt")

        def on_message(client, userdata, msg):
            import json
            try:
                payload = json.loads(msg.payload.decode())
                block = payload.get("block", "Unknown")
                self._buffer.setdefault(block, []).append(payload)
                # Keep only last 24h of readings (~600 readings at 5min interval)
                self._buffer[block] = self._buffer[block][-600:]
            except Exception as e:
                logger.warning(f"MQTT parse error: {e}")

        self._client = mqtt.Client()
        self._client.on_message = on_message
        self._client.connect(self.broker, self.port)
        self._client.subscribe("vine/sensors/#")
        self._client.loop_start()
        logger.info(f"MQTT client started: {self.broker}:{self.port}")

    def get_latest_context(self) -> str:
        """Return formatted SensorContextBlock string for all blocks with data."""
        blocks = []
        for block_id, rows in self._buffer.items():
            meta = {"block": block_id, "variety": "Unknown"}
            try:
                ctx = self.extractor.extract(rows[-576:], meta)  # last 48h
                blocks.append(ctx)
            except Exception as e:
                logger.warning(f"Extraction failed for {block_id}: {e}")
        if not blocks:
            return "[LIVE SENSOR] No data received yet."
        return "\n\n".join(b.to_prompt_string() for b in blocks)

    def stop(self):
        if self._client:
            self._client.loop_stop()
