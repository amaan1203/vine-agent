from __future__ import annotations

import csv
import logging
import os
import textwrap
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)



SYNTHETIC_AGRO_KNOWLEDGE = [
    # ── Irrigation & Soil Moisture ──────────────────────────────────────────
    textwrap.dedent("""
    [Domain] Irrigation Scheduling | [Crop] Vitis vinifera | [Stage] General
    Vineyard irrigation is typically managed using soil moisture sensors measuring
    volumetric water content (VWC). For most wine grape varieties, the critical
    threshold for irrigation initiation is when VWC drops below 25-30% in the top
    30 cm of soil. Evapotranspiration (ET0) data from nearby weather stations
    (CIMIS, AZMET) provides a reference for daily water demand. The crop coefficient
    (Kc) for grapevines varies by phenological stage: 0.15-0.25 during dormancy,
    0.45-0.55 at budbreak, 0.60-0.75 during rapid shoot growth, and 0.45-0.55
    post-harvest. Regulated deficit irrigation (RDI) is a common strategy during
    veraison and post-veraison, targeting 10-15% soil moisture deficit to improve
    anthocyanin concentration and berry quality metrics.
    """).strip(),

    textwrap.dedent("""
    [Domain] Irrigation | [Crop] Pinot Noir | [Stage] Veraison
    Pinot Noir (Vitis vinifera cv. Pinot Noir) is highly sensitive to water stress
    during veraison (color change in berries). Optimal irrigation strategy during
    this period is Regulated Deficit Irrigation (RDI), maintaining soil VWC between
    20-28% (stem water potential of -8 to -12 bar). Excessive irrigation during
    veraison (VWC > 35%) increases berry size, dilutes flavors, and elevates Botrytis
    risk due to berry skin cracking. Deficit stress of 10-12% promotes anthocyanin
    accumulation, tannin development, and improved color stability. Irrigation events
    should be 0.3-0.5 inches per application, targeting early morning (5-7 AM) to
    minimize evaporation losses. ET0 for California coastal regions during veraison
    typically ranges 0.20-0.35 inches/day.
    """).strip(),

    textwrap.dedent("""
    [Domain] Irrigation | [Crop] Chardonnay | [Stage] Berry Set
    Chardonnay vines during berry set (4-6 weeks post-bloom) require consistent
    soil moisture to minimize berry shatter and improve fruit set percentage.
    Recommended VWC range: 28-35%. The critical period extends approximately 3 weeks.
    Application of 0.25-0.35 inches every 2-3 days (adjusted for ET0 and effective
    rainfall) maintains adequate root zone moisture. Water stress (VWC < 22%) during
    this stage causes stomatal closure, reduces photosynthesis rate by up to 40%,
    and can reduce final yield by 15-25%. Soil temperature between 65-75°F promotes
    optimal root water uptake efficiency.
    """).strip(),

    textwrap.dedent("""
    [Domain] Irrigation | [Crop] Cabernet Sauvignon | [Stage] Post-Veraison
    Post-veraison deficit irrigation for Cabernet Sauvignon is well-documented as
    improving wine quality parameters. Target stem water potential: -10 to -14 bar.
    VWC should be maintained at 18-24% in the top 40 cm. Research at UC Davis
    demonstrated that moderate post-veraison water deficit increases Brix accumulation
    rate by 0.2-0.3 °Brix/week and improves phenolic concentration (total anthocyanins
    up to 25% higher). Over-stressing below VWC 15% causes leaf desiccation, premature
    defoliation, and stops sugar accumulation. Apply irrigation when predawn leaf water
    potential exceeds -0.6 MPa (measured with pressure chamber before sunrise).
    """).strip(),

    textwrap.dedent("""
    [Domain] Irrigation | [Method] Drip Irrigation | [Efficiency]
    Drip irrigation in vineyards typically achieves 90-95% application efficiency
    compared to 70-75% for sprinkler and 50-60% for flood systems. Emitter flow rates
    of 0.5-1.0 GPH per emitter, spaced 18-24 inches along the vine row, are standard.
    Pressure requirements: 15-25 PSI at emitter. Sub-surface drip at 12-18 inch depth
    reduces evaporation by 15-20% vs surface drip. Fertigation via drip lines allows
    precise nutrient delivery: nitrogen at 10-20 lbs/acre/year during growing season,
    split into 4-6 applications. Drip scheduling should be based on daily ET0 readings
    multiplied by the block-specific Kc coefficient.
    """).strip(),

    # ── Disease & Pest Management ────────────────────────────────────────────
    textwrap.dedent("""
    [Domain] Disease Management | [Pathogen] Botrytis cinerea | [Crop] Vitis vinifera
    Botrytis bunch rot (caused by Botrytis cinerea) is the most economically significant
    fungal disease in California vineyards. Infection risk is highest when relative
    humidity exceeds 90% for more than 8 hours and temperature is between 59-77°F
    (15-25°C). Pre-bunch closure spray programs using FRAC Group 7 (boscalid) or
    Group 17 (fenhexamid) fungicides are critical. NDVI values below 0.55 in affected
    blocks may indicate early Botrytis establishment. Risk periods: bloom (primary
    infection window), bunch closure (canopy creates humid microclimate), and
    pre-harvest. Tight-clustered varieties (Pinot Noir, Chardonnay, Riesling) are most
    susceptible. Canopy management practices (leaf pulling, shoot positioning) reduce
    humidity by improving airflow.
    """).strip(),

    textwrap.dedent("""
    [Domain] Pest Management | [Pest] Leafhoppers (Erythroneura spp.) | [Detection]
    Grape leafhoppers (Erythroneura elegantula, E. variabilis) cause stippling damage
    on leaves, reducing photosynthetic capacity. Economic threshold: 15-30 nymphs per
    leaf (depending on vineyard history and variety). Drone imagery can detect leafhopper
    damage as irregular chlorotic speckling (NDVI drop of 0.08-0.12 vs. healthy canopy).
    Biological control with Anagrus epos (egg parasitoid) is effective when habitat
    (hedgerows, cover crops) is maintained. Chemical controls: insect growth regulators
    (spirotetramat, buprofezin) applied at 1st generation nymph peak (late May/June
    in Northern CA). Second generation peaks in August represent higher economic risk
    due to proximity to harvest.
    """).strip(),

    textwrap.dedent("""
    [Domain] Disease Management | [Pathogen] Powdery Mildew (Erysiphe necator)
    [Crop] Vitis vinifera | [Risk Period] Bloom to Bunch Closure
    Powdery mildew is the primary disease threat in most California wine grape regions.
    The pathogen requires temperatures 50-90°F and relative humidity > 40% for spore
    germination (unlike most fungi, does NOT require free water). Critical infection
    windows: pre-bloom through bunch closure (BBCH 57-75). Disease index assessment:
    0% = clean, 5% = mild (1-5 clusters affected), 20% = moderate, 50%+ = severe.
    Sulfur applications at 3-7 day intervals (temperature-dependent) are the primary
    management tool. DMI fungicides (Group 3: tebuconazole, myclobutanil) provide
    curative and protective activity. Sensitive varieties: Chardonnay, Merlot, Cabernet Franc.
    """).strip(),

    textwrap.dedent("""
    [Domain] Pest Management | [Pest] Grape Mealybug | [Vector] Grapevine Leafroll
    Grape mealybug (Pseudococcus maritimus) is a key vector of grapevine leafroll
    associated virus-3 (GLRaV-3), causing significant yield and quality losses.
    Symptoms: leaf curl, reddening (Pinot Noir, Cabernet), reduced photosynthesis.
    Monitoring: crawlers emerge in spring (GDD 100-200, base 50°F); chemical
    control with spirotetramat (systemic) or chlorpyrifos (organophosphate, restricted
    use). NDVI analysis can detect leafroll-infected vines as lower canopy
    reflectance (NDVI 0.10-0.15 below healthy vines) starting in mid-summer.
    Vinestock certification and roguing infected vines are essential.
    """).strip(),

    # ── Remote Sensing & NDVI ────────────────────────────────────────────────
    textwrap.dedent("""
    [Domain] Remote Sensing | [Index] NDVI | [Application] Vineyard Health Monitoring
    Normalized Difference Vegetation Index (NDVI) = (NIR - Red) / (NIR + Red).
    NDVI ranges in vineyards:
    - 0.75-0.90: Vigorous, potentially over-vigorous canopy (risk of Botrytis, shading)
    - 0.55-0.75: Optimal canopy development for most wine grape varieties
    - 0.40-0.55: Mild stress (monitor closely, may indicate water stress or disease)
    - Below 0.40: Significant stress (immediate intervention required)
    Multi-spectral drone imagery (NIR, RedEdge bands at 5-10 cm resolution) allows
    per-vine health mapping. NDRE (Normalized Difference Red Edge) is superior for
    detecting early chlorophyll deficiency before visible symptoms appear. Seasonal
    baseline NDVI should be established at budbreak (BBCH 09-11) for accurate
    comparative analysis throughout the season.
    """).strip(),

    textwrap.dedent("""
    [Domain] Remote Sensing | [Index] NDRE | [Application] Nitrogen Status
    NDRE (Normalized Difference Red Edge) = (NIR - RedEdge) / (NIR + RedEdge)
    is a more sensitive indicator of vine nitrogen and chlorophyll status than NDVI,
    particularly at high canopy density. NDRE thresholds for Vitis vinifera:
    - NDRE > 0.45: Adequate to excess nitrogen; risk of excessive vigor
    - NDRE 0.30-0.45: Optimal range for most stages
    - NDRE 0.20-0.30: Mild nitrogen deficiency (petiole N < 0.8%)
    - NDRE < 0.20: Severe nitrogen deficiency; yield reduction likely
    Tissue sampling (petiole analysis at bloom) should confirm drone-detected anomalies.
    Precision variable-rate fertilization maps can be generated from NDRE rasters.
    """).strip(),

    textwrap.dedent("""
    [Domain] Remote Sensing | [Index] NDWI | [Application] Canopy Water Content
    NDWI (Normalized Difference Water Index) = (NIR - SWIR) / (NIR + SWIR) uses
    shortwave infrared (SWIR) bands to detect canopy water content. NDWI thresholds
    for grapevines: > 0.25 (well-watered), 0.10-0.25 (mild stress), < 0.10 (significant
    water stress). NDWI is complementary to soil VWC sensors: sensors measure root zone
    moisture, NDWI reflects actual vine water status (leaf water content). Useful for
    identifying blocks where vines are stressed despite adequate soil moisture (e.g.,
    root restriction, high salinity, rootstock mismatch). Multi-flight time series of
    NDWI enables early detection of chronic water deficit weeks before visible symptoms.
    """).strip(),

    # ── Harvest Timing ───────────────────────────────────────────────────────
    textwrap.dedent("""
    [Domain] Harvest Timing | [Metric] Brix, pH, TA | [Crop] Pinot Noir
    Optimal harvest for Pinot Noir wine production: Brix 23-25.5°, pH 3.3-3.5,
    titratable acidity (TA) 5.5-7.0 g/L. Sampling frequency: weekly from veraison,
    biweekly in final 4 weeks pre-harvest. Sample 100 berries per block minimum,
    from across all vine positions and cluster positions. Growing Degree Days (GDD,
    base 50°F) accumulation from April 1: harvest typically occurs at 2,800-3,400 GDD
    for Pinot Noir in California coastal regions. Sensory assessment (seed browning,
    skin tannin development, flavor profile) is essential alongside analytical data.
    pH is the best single predictor of perceived ripeness in cool-climate Pinot Noir.
    """).strip(),

    textwrap.dedent("""
    [Domain] Harvest Timing | [Metric] GDD (Growing Degree Days) | [General]
    Growing Degree Day (GDD) calculation: GDD = ((Tmax + Tmin)/2) - 50°F (base temp),
    accumulated from April 1 to harvest. California wine region GDD targets:
    - Sparkling wine base (Chardonnay): 1,800-2,200 GDD
    - Pinot Noir (light, elegant style): 2,400-2,800 GDD
    - Chardonnay (full-bodied): 2,800-3,200 GDD
    - Cabernet Sauvignon: 3,200-3,800 GDD
    Cool vintage years (< 2,800 GDD accumulated by Sept 15) often require longer
    hang time for Cabernet-family varieties. GDD tracking with IoT weather stations
    enables accurate harvest window prediction 2-3 weeks in advance.
    """).strip(),

    textwrap.dedent("""
    [Domain] Harvest Timing | [Method] Berry Sampling | [Quality Assessment]
    Automated berry sampling algorithms using machine vision on drone footage can
    estimate color development (YCbCr color space) as a proxy for anthocyanin
    accumulation. Color index correlation with HPLC anthocyanin measurements shows
    R² > 0.85 in Pinot Noir when canopy conditions are consistent. Ground-truth
    samples (100-berry random walk per block) should always validate remote-sensing
    estimates. Seed browning (75% brown seeds = physiological ripeness) and skin
    tannin polymerization (firm but not harsh) provide sensory harvest indicators
    complementing analytical data.
    """).strip(),

    # ── Soil Science ─────────────────────────────────────────────────────────
    textwrap.dedent("""
    [Domain] Soil Science | [Parameter] pH | [Impact] Nutrient Availability
    Vineyard soil pH profoundly affects nutrient availability and vine health:
    - pH 5.5-6.5: Optimal for most wine grape varieties; maximum nutrient availability
    - pH > 7.0: Reduced iron, manganese, zinc availability → chlorosis risk
    - pH > 7.5: Iron-induced chlorosis common (interveinal yellowing on young leaves)
    - pH < 5.5: Aluminum toxicity possible; reduced calcium, magnesium availability
    Iron chlorosis correction: 2-4 lbs/acre chelated iron (Fe-EDDHA for high pH soils)
    applied to soil, or foliar application of 1-2 lbs/acre iron sulfate. Soil amendment
    with elemental sulfur (2-4 tons/acre, incorporated) can reduce pH 0.5-1.0 units
    over 1-2 seasons on calcareous soils.
    """).strip(),

    textwrap.dedent("""
    [Domain] Cover Crops | [Application] Soil Health | [Vineyard Management]
    Cover crops between vineyard rows provide multiple benefits: organic matter
    addition, erosion control, beneficial insect habitat, and competition regulation
    with vines. Recommended species: Zorro fescue or annual ryegrass (low-growing,
    minimal nitrogen competition), cereal rye (winter cover, mows down easily),
    legume mixes (hairy vetch, crimson clover) for nitrogen fixation. Cover crop
    water consumption: 0.5-1.5 inches/week during spring growth requires monitoring.
    Resident flora (native annual grasses) is preferred on low-vigor sites to avoid
    excessive vine competition. Mow or roll-crimp at 50% bloom to prevent seed bank
    replenishment of problematic weeds. Maintain 18-inch bare strip under vine row for
    weed control and soil moisture conservation.
    """).strip(),

    textwrap.dedent("""
    [Domain] Frost Management | [Mechanism] Active Protection | [Crop] Vitis vinifera
    Grapevine frost damage occurs when tissue temperatures drop below 28°F (-2.2°C)
    for more than 30 minutes. Primary bud damage begins at 28°F; secondary buds more
    tolerant to 22°F (-5.5°C). Active protection methods:
    - Wind machines: effective when temperature inversion present (warm air above cold);
      protection radius 300-500 ft; activate at 34°F
    - Overhead sprinklers: apply at 34°F, must run continuously until air temp > 34°F;
      application rate 0.1 in/hr minimum; can actually increase ice load risk
    - Heaters (smudge pots): 1 heater per acre; smoke plume intercepts radiant cooling
    IoT temperature sensors at multiple heights (ground, 2 ft, 4 ft) enable real-time
    inversion detection and precise timing of frost protection activation.
    """).strip(),

    textwrap.dedent("""
    [Domain] Nutrition | [Nutrient] Potassium | [Crop] Vitis vinifera
    Potassium (K) is the most abundantly required macronutrient by grapevines.
    Petiole K at bloom: 1.5-3.0% dry weight (adequate); < 1.5% = deficiency.
    Potassium deficiency symptoms: marginal leaf scorch (particularly lower canopy),
    poor berry color development in red varieties, reduced berry set. Excess K:
    reduces tartaric acid, raises must pH (risk of microbiological instability).
    High-K soils (> 300 ppm exchangeable K) require no supplementation; sandy loam
    soils commonly require 50-100 lbs K₂O/acre/year. Apply as potassium sulfate
    (preferred for low-sulfur soils) or potassium chloride. Foliar K in late summer
    can correct deficiency if root uptake is restricted by drought or rootstock.
    """).strip(),

    textwrap.dedent("""
    [Domain] IoT Sensors | [Parameter] Soil Moisture, Temperature, CO2 | [VINE Platform]
    VINE platform deploys IoT sensors at Iron Horse Vineyards (Sonoma County, CA):
    Sensor readings: soil moisture (VWC% at 15, 30, 60 cm depth), soil temperature
    (°F), canopy temperature, CO2 (ppm), relative humidity (%), PAR (photosynthetically
    active radiation). Data logged every 5 minutes via LoRaWAN network to cloud endpoint.
    Alert thresholds: soil moisture < 25% VWC (irrigation trigger), soil temperature
    > 90°F (heat stress), CO2 > 450 ppm sustained (anomaly). Historical data enables
    ML model training for 72-hour irrigation scheduling and harvest timing prediction.
    """).strip(),

    textwrap.dedent("""
    [Domain] Water Stress | [Measurement] Stem Water Potential | [Crop] Vitis vinifera
    Stem water potential (ψstem) measured with pressure chamber is the gold standard
    for vine water status assessment. Measurements taken at midday on covered, shaded
    leaves (equilibrated to stem ψ). Thresholds for California wine grapes:
    - ψstem > -6 bar: No stress; full water replenishment needed
    - ψstem -6 to -9 bar: Mild stress; acceptable for quality-focused irrigation
    - ψstem -9 to -12 bar: Moderate stress; Regulated Deficit Irrigation target
    - ψstem -12 to -16 bar: Severe stress; risk of permanent damage
    - ψstem < -16 bar: Extreme stress; irrigation emergency
    Measurement frequency: minimum weekly during growing season, every 3-4 days
    during critical periods (fruit set, veraison, pre-harvest).
    """).strip(),

    textwrap.dedent("""
    [Domain] Iron Horse Vineyards | [Location] Sonoma County, CA | [Context] VINE Project
    Iron Horse Vineyards is a 160-acre estate winery located in the Green Valley
    appellation of Sonoma County, California. Primary varieties: Chardonnay, Pinot Noir
    (sparkling and still wine production). Average annual rainfall: 40-55 inches
    (Oct-April). Summer fog from Petaluma Gap moderates temperatures. Soil types:
    Goldridge sandy loam (low water-holding capacity, fast drainage), Sebastopol clay
    loam. Elevation 200-400 ft. The VINE project deploys IoT sensors, multi-spectral
    drone imagery, and NRP computational resources to build open precision agriculture
    datasets. AI-driven analytics have demonstrated potential for 10% water use reduction.
    """).strip(),

    textwrap.dedent("""
    [Domain] Canopy Management | [Technique] Leaf Removal | [Quality Impact]
    Leaf removal in the fruit zone (basal leaf pulling) is one of the highest-impact
    canopy practices for wine quality and disease management. Benefits:
    - Reduces Botrytis risk: improves air circulation, reduces humidity; 30-50% reduction
    - Improves spray penetration: fungicide coverage increases 40-60% in open canopy
    - Berry color: increased sun exposure (east-facing) improves anthocyanin by 15-25%
    Timing: early leaf pull at 10-15 cm shoot length vs. late (pre-harvest) provides
    different quality outcomes. Early pull: hardening effect reduces berry cracking;
    late pull: phenolic ripening boost without excessive sugar accumulation.
    Machine-assisted leaf removal (mechanical defoliators) effective for large blocks.
    """).strip(),

    textwrap.dedent("""
    [Domain] Rootstocks | [Selection] | [Soil Adaptation]
    Rootstock selection determines vine vigor, drought tolerance, and phylloxera
    resistance. Common California choices:
    - 101-14: Low to medium vigor; good water efficiency; Goldridge sandy loam
    - 3309C: Medium vigor; drought tolerant; adapts well to clay loam
    - 110R: High drought resistance; deep roots; clay soils with water restriction
    - 1616C: Excellent for wet soils and nematode pressure; low vigor
    - 5BB: High vigor and lime tolerance; calcareous soils (pH > 7.5)
    At Iron Horse (Goldridge sandy loam): 101-14 and 110R are standard choices,
    providing appropriate vigor control and water efficiency matching the region's
    low summer rainfall and fast-draining soils.
    """).strip(),

    textwrap.dedent("""
    [Domain] Phenology | [Stages] BBCH Scale | [Vitis vinifera]
    Key BBCH phenological stages for vineyard management timing:
    - BBCH 01-09: Bud swell to bud burst (dormancy break)
    - BBCH 11-19: Shoot development (1-9 unfolded leaves)
    - BBCH 55-59: Inflorescence development (visible to separated flowers)
    - BBCH 61-69: Flowering / bloom (10% to full flowering)
    - BBCH 71: Berry set (fruit set complete)
    - BBCH 75-77: Berry enlargement (pea size to bunch closure)
    - BBCH 81-85: Veraison (color change, softening)
    - BBCH 89: Harvest ripeness
    Management actions (spray timing, irrigation, leaf pull) are triggered at
    specific BBCH stages. IoT + ML models can predict BBCH stage from accumulated GDD.
    """).strip(),

    textwrap.dedent("""
    [Domain] Nutrient Management | [Timing] Spring | [Method] Fertigation
    Spring nitrogen application timing is critical for Vitis vinifera. Applications
    before or at budbreak (BBCH 01-07) support shoot development and set the
    nitrogen reservoir for the season. Recommended rates: 10-20 lbs N/acre for
    sandy soils, 5-10 lbs for clay loam (lower leaching risk). Petiole analysis
    at full bloom (BBCH 65) is the definitive diagnostic: adequate N > 0.9% dry weight.
    Split applications: 50% at budbreak, 25% at fruit set (BBCH 71), 25% at veraison
    onset — maximize uptake timing with growth demand. Excess nitrogen post-veraison
    delays color development and increases disease susceptibility.
    """).strip(),

    textwrap.dedent("""
    [Domain] Heat Stress | [Threshold] | [Management] Sunburn
    Grapevine heat stress begins at air temperatures > 95°F (35°C) and is exacerbated
    by lower relative humidity. Symptoms: bleaching/necrosis of sun-exposed fruit
    (sunburn), leaf rolling (thermoregulatory response), reduced photosynthesis.
    Management: kaolin clay applications (4-6 lbs/gal dilution) applied pre-heat event
    form a reflective particle film reducing fruit temperature 5-8°F. Overhead misting
    (evaporative cooling) lowers canopy temperature 4-6°F when humidity < 40%.
    IoT sensors monitoring soil temperature (> 85°F triggers root stress) and
    canopy temperature (> 100°F in still air = high sunburn risk) enable timely alerts.
    Post-heat-event irrigation restores vine turgor and supports recovery.
    """).strip(),

    textwrap.dedent("""
    [Domain] Digital Twin | [Platform] VINE/NRP | [Application] Precision Agriculture
    The VINE digital twin system integrates IoT sensor streams, drone imagery, weather
    forecasts, and AI model predictions into a real-time 3D representation of Iron Horse
    Vineyards. Components: sensor data stream (LoRaWAN → Kafka → InfluxDB), drone flight
    management (automated DJI flight planning, post-processing pipeline), ML model serving
    (vLLM on NRP GPU clusters for inference), geospatial visualization (GIS layers in QGIS
    or web-based Cesium). The digital twin enables scenario simulation: testing different
    irrigation strategies virtually before field application, predicting harvest timing
    under different weather scenarios, and identifying optimal spray timing windows.
    Kubernetes on NRP provides scalable compute for all ML workloads.
    """).strip(),

    textwrap.dedent("""
    [Domain] Machine Learning | [Application] Yield Prediction | [Vitis vinifera]
    Yield prediction models for grapevines integrate multiple data streams:
    - Inflorescence count (BBCH 55): manual count or computer vision on drone imagery
    - Berry set rate (BBCH 71): 30-70% of flowers set to fruit depending on conditions
    - NDVI at véraison: correlates with final crop load (R² = 0.72-0.81 in studies)
    - Historical yield records: block-specific 10-year average as baseline
    - Rainfall accumulation: spring rainfall > 10 inches typically correlates with
      higher yields on Goldridge sandy loam via improved root zone depth
    ML models (random forest, LSTM time-series) trained on 5+ years of sensor + yield
    data achieve 85-90% accuracy in predicting harvest yield 4-6 weeks in advance.
    These models require annual retraining as climate patterns shift.
    """).strip(),

    textwrap.dedent("""
    [Domain] Integrated Pest Management | [Approach] Monitoring | [VINE IoT]
    IPM monitoring in VINE-connected vineyards integrates: degree-day (GDD) models
    for pest emergence timing, pheromone trap data (Lepidoptera: grape berry moth,
    Platynota stultana at 10 moths/trap/week = action threshold), visual scouting
    calibrated with drone NDVI anomaly maps. Spray decision framework:
    - GDD < 50 (base 50°F from Jan 1): monitor only
    - Disease pressure model (DMI + temperature + wetness hours): compute spray risk index
    - Spray risk > 7: apply fungicide within 48h
    IoT-connected weather stations log wetness (leaf wetness sensor) and temperature
    continuously, feeding real-time disease pressure models (powdery mildew, downy mildew).
    """).strip(),
]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Sensor CSV → Weekly NL Summaries (for RAPTOR historical nodes)
# ─────────────────────────────────────────────────────────────────────────────

def sensor_csv_to_summaries(
    csv_path: str,
    block_col: str = "block",
    variety_col: str = "variety",
    date_col: str = "date",
    vwc_col: str = "vwc_pct",
    temp_col: str = "temp_f",
    window_days: int = 7,
) -> List[str]:
    """
    Reads a sensor CSV and converts rolling weekly windows into NL summaries.
    Each summary becomes a RAPTOR temporal leaf node (historical record).
    These are DIFFERENT from live SensorContextBlock objects — these capture
    historical patterns for the RAPTOR knowledge base.
    """
    if not os.path.exists(csv_path):
        logger.warning(f"Sensor CSV not found: {csv_path}. Skipping.")
        return []

    summaries = []
    try:
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            return []

        blocks = {}
        for row in rows:
            block = row.get(block_col, "Unknown")
            blocks.setdefault(block, []).append(row)

        for block, block_rows in blocks.items():
            variety = block_rows[0].get(variety_col, "unknown variety")
            for i in range(0, len(block_rows), window_days):
                window = block_rows[i: i + window_days]
                if len(window) < 2:
                    continue
                start_date = window[0].get(date_col, "?")
                end_date   = window[-1].get(date_col, "?")
                try:
                    vwcs  = [float(r[vwc_col]) for r in window if r.get(vwc_col)]
                    temps = [float(r[temp_col]) for r in window if r.get(temp_col)]
                    avg_vwc = sum(vwcs) / len(vwcs)
                    min_vwc = min(vwcs)
                    max_temp = max(temps)
                    summary = (
                        f"[Sensor History] Block {block} ({variety}), "
                        f"{start_date} to {end_date}: "
                        f"Average soil VWC {avg_vwc:.1f}%, minimum {min_vwc:.1f}% VWC recorded. "
                        f"Peak temperature {max_temp:.1f}°F. "
                    )
                    if min_vwc < 25:
                        summary += "VWC dropped below irrigation threshold (25%)."
                    if max_temp > 90:
                        summary += f" Heat stress event detected ({max_temp:.0f}°F)."
                    summaries.append(summary)
                except (ValueError, KeyError):
                    continue

    except Exception as e:
        logger.error(f"Error reading sensor CSV: {e}")
    return summaries



def load_text_docs(directory: str, extensions: tuple = (".txt", ".md")) -> List[str]:
    """Load plain text or markdown files from a directory, chunk into ~512-word passages."""
    chunks = []
    path = Path(directory)
    if not path.exists():
        logger.warning(f"Document directory not found: {directory}")
        return []
    for fp in path.rglob("*"):
        if fp.suffix.lower() in extensions:
            try:
                text = fp.read_text(encoding="utf-8")
                chunks.extend(_chunk_text(text, chunk_size=512))
            except Exception as e:
                logger.warning(f"Could not read {fp}: {e}")
    logger.info(f"Loaded {len(chunks)} text chunks from {directory}")
    return chunks


def load_pdf_docs(directory: str) -> List[str]:
    """Load PDFs from a directory using LangChain's PyPDFLoader."""
    try:
        from langchain_community.document_loaders import PyPDFDirectoryLoader
        loader = PyPDFDirectoryLoader(directory)
        docs = loader.load()
        texts = [doc.page_content for doc in docs]
        chunks = []
        for t in texts:
            chunks.extend(_chunk_text(t, chunk_size=512))
        logger.info(f"Loaded {len(chunks)} PDF chunks from {directory}")
        return chunks
    except ImportError:
        logger.warning("pypdf not installed. Install with: pip install pypdf")
        return []
    except Exception as e:
        logger.error(f"PDF loading error: {e}")
        return []


def _chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> List[str]:
    """Split text into overlapping word-level chunks."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i: i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        i += chunk_size - overlap
    return chunks



def load_hf_agro_dataset(
    dataset_name: str = "rag-datasets/rag-mini-bioasq",
    split: str = "passages",
    text_col: str = "passage",
    max_samples: int = 500,
) -> List[str]:
    """Load an agricultural or biology dataset from HuggingFace for testing."""
    try:
        from datasets import load_dataset
        ds = load_dataset(dataset_name, split=split, trust_remote_code=True)
        texts = []
        for i, row in enumerate(ds):
            if i >= max_samples:
                break
            text = row.get(text_col, "")
            if text and len(text) > 50:
                texts.append(text)
        logger.info(f"Loaded {len(texts)} passages from HuggingFace: {dataset_name}")
        return texts
    except Exception as e:
        logger.warning(f"HuggingFace dataset load failed ({e}). Using synthetic data only.")
        return []


def build_vine_knowledge_base(
    docs_dir: Optional[str] = None,
    sensor_csv: Optional[str] = None,
    drone_summaries: Optional[List[str]] = None,
    use_synthetic: bool = True,
    use_hf_dataset: bool = False,
    hf_dataset: str = "rag-datasets/rag-mini-bioasq",
    max_hf_samples: int = 200,
) -> tuple:
    """
    Aggregate all knowledge sources into:
      - knowledge_texts: List[str] for FAISS + RAPTOR document indexing
      - sensor_summaries: List[str] of historical weekly sensor summaries

    Returns:
        (knowledge_texts, sensor_summaries)
    """
    knowledge_texts: List[str] = []
    sensor_summaries: List[str] = []

    if use_synthetic:
        knowledge_texts.extend(SYNTHETIC_AGRO_KNOWLEDGE)
        logger.info(f"Synthetic knowledge base: {len(SYNTHETIC_AGRO_KNOWLEDGE)} docs.")

    if docs_dir:
        knowledge_texts.extend(load_text_docs(docs_dir, extensions=(".txt", ".md")))
        knowledge_texts.extend(load_pdf_docs(docs_dir))

    if sensor_csv:
        sensor_summaries = sensor_csv_to_summaries(sensor_csv)
        logger.info(f"Sensor historical summaries: {len(sensor_summaries)} weekly windows.")
        knowledge_texts.extend(sensor_summaries)  # Also index in FAISS

    if drone_summaries:
        knowledge_texts.extend(drone_summaries)
        logger.info(f"Drone text blocks added to knowledge base: {len(drone_summaries)} blocks.")

    if use_hf_dataset:
        hf_texts = load_hf_agro_dataset(hf_dataset, max_samples=max_hf_samples)
        knowledge_texts.extend(hf_texts)

    logger.info(f"Total knowledge base: {len(knowledge_texts)} text chunks.")
    return knowledge_texts, sensor_summaries
