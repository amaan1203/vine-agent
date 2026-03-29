from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)
random.seed(42)


@dataclass
class DroneContextBlock:
    block: str
    variety: str
    flight_date: str
    drone_model: str = "DJI Phantom 4 Multispectral"

    
    ndvi_mean: float = 0.65
    ndvi_std:  float = 0.08
    ndvi_min:  float = 0.45
    ndre_mean: float = 0.38
    ndre_std:  float = 0.06
    ndwi_mean: float = 0.15   

    
    stressed_area_pct: float = 0.0  
    severe_stress_pct: float = 0.0   
    anomaly_clusters: int   = 0      
    anomaly_location: str   = ""     

    
    ndvi_status: str  = "HEALTHY"    
    ndre_status: str  = "ADEQUATE"   
    water_status: str = "NORMAL"     

    def compute_status_labels(self):
        if self.ndvi_mean >= 0.65:
            self.ndvi_status = "HEALTHY"
        elif self.ndvi_mean >= 0.50:
            self.ndvi_status = "MILD_STRESS"
        elif self.ndvi_mean >= 0.38:
            self.ndvi_status = "MODERATE_STRESS"
        else:
            self.ndvi_status = "SEVERE_STRESS"

        # NDRE thresholds (nitrogen / chlorophyll)
        if self.ndre_mean >= 0.40:
            self.ndre_status = "ADEQUATE_NITROGEN"
        elif self.ndre_mean >= 0.28:
            self.ndre_status = "MILD_N_DEFICIENCY"
        else:
            self.ndre_status = "SEVERE_N_DEFICIENCY"

        # NDWI (water status)
        if self.ndwi_mean >= 0.20:
            self.water_status = "WELL_WATERED"
        elif self.ndwi_mean >= 0.08:
            self.water_status = "MILD_WATER_STRESS"
        else:
            self.water_status = "HIGH_WATER_STRESS"

    def to_prompt_string(self) -> str:
       
        alert_parts = []
        if self.ndvi_status not in ("HEALTHY",):
            alert_parts.append(f"{self.ndvi_status}")
        if "DEFICIENCY" in self.ndre_status:
            alert_parts.append(f"{self.ndre_status}")
        if "STRESS" in self.water_status:
            alert_parts.append(f"{self.water_status}")
        alert_str = "  ".join(alert_parts) if alert_parts else "✓ No major alerts"

        anomaly_str = (
            f"Anomaly clusters: {self.anomaly_clusters} patch(es), {self.anomaly_location}"
            if self.anomaly_clusters > 0
            else "No discrete anomaly clusters detected"
        )

        return (
            f"╔══ [DRONE — Block {self.block} ({self.variety})] ═══════════════════╗\n"
            f"║  Flight date: {self.flight_date}  |  Platform: {self.drone_model}\n"
            f"║  Source: Multispectral imagery (NIR, Red, RedEdge, Blue, Green)\n"
            f"║  ─────────────────────────────────────────────────────────────\n"
            f"║  NDVI:  mean={self.ndvi_mean:.3f}  ±{self.ndvi_std:.3f}  "
            f"min={self.ndvi_min:.3f}  →  {self.ndvi_status}\n"
            f"║  NDRE:  mean={self.ndre_mean:.3f}  ±{self.ndre_std:.3f}"
            f"                →  {self.ndre_status}\n"
            f"║  NDWI:  mean={self.ndwi_mean:.3f}               →  {self.water_status}\n"
            f"║  Stressed area (NDVI<0.40): {self.stressed_area_pct:.1f}% of block\n"
            f"║  Severe stress (NDVI<0.30): {self.severe_stress_pct:.1f}% of block\n"
            f"║  {anomaly_str}\n"
            f"║  ─────────────────────────────────────────────────────────────\n"
            f"║  ALERTS: {alert_str}\n"
            f"╚══════════════════════════════════════════════════════════════╝"
        )



class DroneImageryEncoder:
    """
    Converts multispectral GeoTIFF drone imagery into DroneContextBlock objects.
    """

    def __init__(self, zone_grid: Tuple[int, int] = (4, 4)):
        """
        zone_grid: (rows, cols) to divide each block into sub-zones.
        4x4 = 16 zones per block, each approximately 0.5 acre.
        """
        self.zone_grid = zone_grid

    # ───GeoTIFF → statistics ───────────────────────────────────

    def load_geotiff_zone_stats(
        self,
        geotiff_path: str,
        block: str,
        variety: str,
        flight_date: str,
    ) -> DroneContextBlock:
        try:
            import rasterio
        except ImportError:
            raise ImportError("Install rasterio: pip install rasterio")

        with rasterio.open(geotiff_path) as src:
            if src.count < 5:
                raise ValueError(f"GeoTIFF must have ≥5 bands (Blue,Green,Red,RE,NIR). Got {src.count}.")

            blue  = src.read(1).astype(float)
            green = src.read(2).astype(float)
            red   = src.read(3).astype(float)
            re    = src.read(4).astype(float)
            nir   = src.read(5).astype(float)
            swir  = src.read(6).astype(float) if src.count >= 6 else None

        eps = 1e-10
        ndvi = (nir - red)  / (nir + red  + eps)
        ndre = (nir - re)   / (nir + re   + eps)
        ndwi = (nir - swir) / (nir + swir + eps) if swir is not None else np.zeros_like(ndvi)

        # Clip to valid range
        ndvi = np.clip(ndvi, -1, 1)
        ndre = np.clip(ndre, -1, 1)
        ndwi = np.clip(ndwi, -1, 1)

        stressed_mask       = ndvi < 0.40
        severe_stress_mask  = ndvi < 0.30

        ctx = DroneContextBlock(
            block=block,
            variety=variety,
            flight_date=flight_date,
            ndvi_mean=float(np.nanmean(ndvi)),
            ndvi_std=float(np.nanstd(ndvi)),
            ndvi_min=float(np.nanmin(ndvi)),
            ndre_mean=float(np.nanmean(ndre)),
            ndre_std=float(np.nanstd(ndre)),
            ndwi_mean=float(np.nanmean(ndwi)),
            stressed_area_pct=float(np.sum(stressed_mask) / ndvi.size * 100),
            severe_stress_pct=float(np.sum(severe_stress_mask) / ndvi.size * 100),
            anomaly_clusters=self._detect_anomaly_clusters(ndvi),
        )
        ctx.compute_status_labels()
        return ctx

    def _detect_anomaly_clusters(self, ndvi: np.ndarray, threshold: float = 0.35) -> int:
        try:
            from scipy.ndimage import label
            mask = (ndvi < threshold).astype(int)
            _, n_clusters = label(mask)
            return min(n_clusters, 20)  # cap for sanity
        except Exception:
            return int(np.sum(ndvi < threshold) > 0)


    def generate_synthetic_ndvi_zones(
        self,
        blocks: Optional[List[Dict]] = None,
        n_flights: int = 3,
        start_date: str = "2024-06-01",
    ) -> List[DroneContextBlock]:
        if blocks is None:
            from sensor_stream import VINEYARD_BLOCKS
            blocks = VINEYARD_BLOCKS

        base_date = datetime.strptime(start_date, "%Y-%m-%d")
        results: List[DroneContextBlock] = []

        for flight_num in range(n_flights):
            flight_date = (base_date + timedelta(days=flight_num * 21)).strftime("%Y-%m-%d")
            # Season: later flights = more stress as summer progresses
            stress_factor = 0.05 * flight_num

            for bm in blocks:
                # Sandy loam blocks dry out faster → more stress
                soil_factor = 0.04 if "sandy" in bm.get("soil", "").lower() else 0.0
                base_ndvi = 0.70 - stress_factor - soil_factor + random.uniform(-0.04, 0.04)
                base_ndre = 0.42 - stress_factor * 0.6 + random.uniform(-0.03, 0.03)
                base_ndwi = 0.22 - stress_factor * 0.7 + random.uniform(-0.03, 0.03)

                base_ndvi = max(0.25, min(0.85, base_ndvi))
                base_ndre = max(0.10, min(0.55, base_ndre))
                base_ndwi = max(0.02, min(0.35, base_ndwi))

                stressed_pct = max(0, (0.55 - base_ndvi) * 120 + random.uniform(-3, 3))
                severe_pct   = max(0, (0.38 - base_ndvi) * 80  + random.uniform(-2, 2))
                clusters = max(0, int(stressed_pct / 8) + random.randint(0, 2))
                locations = ["NE corner", "SW rows", "center cluster", "E border", "scattered"]
                loc_str = random.choice(locations) if clusters > 0 else ""

                ctx = DroneContextBlock(
                    block=bm["block"],
                    variety=bm["variety"],
                    flight_date=flight_date,
                    ndvi_mean=round(base_ndvi, 3),
                    ndvi_std=round(random.uniform(0.05, 0.12), 3),
                    ndvi_min=round(base_ndvi - random.uniform(0.18, 0.30), 3),
                    ndre_mean=round(base_ndre, 3),
                    ndre_std=round(random.uniform(0.04, 0.08), 3),
                    ndwi_mean=round(base_ndwi, 3),
                    stressed_area_pct=round(stressed_pct, 1),
                    severe_stress_pct=round(severe_pct, 1),
                    anomaly_clusters=clusters,
                    anomaly_location=loc_str,
                )
                ctx.compute_status_labels()
                results.append(ctx)

        logger.info(f"Synthetic drone data: {len(results)} blocks × flights generated.")
        return results

    def encode_to_text_blocks(
        self,
        drone_blocks: List[DroneContextBlock],
    ) -> List[str]:
        return [block.to_prompt_string() for block in drone_blocks]

    def get_latest_flight_summary(
        self,
        drone_blocks: List[DroneContextBlock],
        target_block: Optional[str] = None,
    ) -> str:
        """
        Return the most recent flight's DroneContextBlock(s) as formatted text.
        Optionally filter to a specific vineyard block.
        Used for direct Solver injection (most recent flight context).
        """
        if not drone_blocks:
            return "[DRONE] No drone flight data available."

        # Sort by flight date descending
        sorted_blocks = sorted(drone_blocks, key=lambda b: b.flight_date, reverse=True)
        latest_date = sorted_blocks[0].flight_date

        latest = [b for b in sorted_blocks if b.flight_date == latest_date]
        if target_block:
            latest = [b for b in latest if b.block == target_block]

        if not latest:
            return f"[DRONE] No flight data for Block {target_block}."

        header = f"[DRONE IMAGERY — Latest Flight: {latest_date} | {len(latest)} block(s)]\n\n"
        return header + "\n\n".join(b.to_prompt_string() for b in latest)
