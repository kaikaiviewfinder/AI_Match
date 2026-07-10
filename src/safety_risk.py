"""Outdoor hazard detection from image + sensor — offline, rule-based, < 10 KB.

Input:  GeoVLM field predictions (16 fields) + DK sensor readings (elev/temp/humid)
Output: risk level + specific hazards + actionable mitigation advice

No new model. No network. Pure rule-based inference from existing signals.

Risk signals by source:
  Visual (GeoVLM fields):
    - terrain_type   → cliff / landslide / avalanche terrain
    - soil_color     → landslide susceptibility (red/loess on slope = dangerous)
    - water_type     → nearby water hazard (rising river, flash flood channel)
    - sky_quality    → incoming storm, low visibility, dust storm
    - landform_detail→ specific hazard topology (canyon=bottleneck, karst=rockfall)
    - mountain_rock  → rock stability (fractured limestone ≠ solid granite)
    - vegetation     → sparse on slope = recent slide, alpine_meadow = exposure
    - scene_type     → environmental context (snow=avalanche, coastal=surge)

  Sensor (DK-2500 onboard):
    - humidity trend → rapid drop = incoming front; sustained high = saturated soil
    - temperature    → freeze-thaw risk; heat stress
    - elevation      → altitude sickness threshold
"""

from dataclasses import dataclass, field
from typing import Optional


# ═══════════════════════════════════════════════════════════════════
# Hazard lookup tables — hardcoded expert knowledge, ~3 KB total
# ═══════════════════════════════════════════════════════════════════

# Terrain hazard mapping
TERRAIN_HAZARD = {
    "sharp_mountains":       ("high_cliff",    "陡峭山体——坠落、滑坡高风险"),
    "karst_peaks":           ("rockfall",      "喀斯特峰林——岩体破碎，落石风险"),
    "sandstone_pillars":     ("rockfall",      "砂岩柱——风化严重，崩塌风险"),
    "desert_dunes":          ("disorientation","沙丘地形——易迷失方向"),
    "rolling_hills":         ("moderate",      ""),
    "farmland_plain":        ("low",           ""),
    "urban_flat":            ("low",           ""),
    "grassland_steppe":      ("exposure",      "开阔草甸——暴晒/暴风无遮蔽"),
    "plateau":               ("altitude",      "高原地形——注意高反和温差"),
}

# Soil → landslide susceptibility (on slopes)
SOIL_LANDSLIDE_RISK = {
    "red":         0.9,   # 红土 + 坡 = 极易滑坡（华南典型）
    "loess_yellow":0.85,  # 黄土 + 雨 = 泥流（西北黄土高原）
    "yellow":      0.6,
    "brown":       0.4,
    "black":       0.3,
    "grey":        0.2,
    "white":       0.1,
}

# Water hazards
WATER_HAZARD = {
    "river":      ("flash_flood",  "河道——山洪首冲通道，雨天立即远离"),
    "stream":     ("flash_flood",  "溪流——上游暴雨数分钟即可涨水"),
    "waterfall":  ("cliff",        "瀑布——湿滑悬崖，失足高风险"),
    "lake":       ("rising_water", "湖泊——持续暴雨可能漫溢"),
    "glacier":    ("crevasse",     "冰川——冰裂缝隐藏风险"),
    "ocean":      ("storm_surge",  "海岸——台风风暴潮风险"),
    "none_visible": (None, ""),
}

# Sky → weather hazard
SKY_HAZARD = {
    "clear_blue":   ("low",        ""),
    "grey_hazy":    ("rain_coming","灰蒙天空——降水前兆，准备雨具和撤离路线"),
    "thick_fog":    ("zero_vis",   "浓雾——能见度极低，就地停止移动，等待雾散"),
    "dusty_yellow": ("sandstorm",  "黄沙天——沙尘暴前兆，遮住口鼻，寻找洼地躲避"),
}

# Landform → specific hazard topology
LANDFORM_HAZARD = {
    "river_canyon":                ("flash_flood",    "河谷峡谷——山洪瓶颈，水位可骤升数米"),
    "limestone_tower_karst_scattered": ("rockfall",   "喀斯特孤峰——溶蚀裂隙发育，落石频繁"),
    "limestone_tower_karst_dense":     ("rockfall",   "喀斯特峰丛——岩体破碎，崩塌高风险"),
    "granite_spheroidal_rounded":      ("low",         ""),
    "granite_spheroidal_sharp":        ("cliff",       "花岗岩尖峰——陡峭，攀爬坠落风险"),
    "quartz_sandstone_pillars":        ("rockfall",    "石英砂岩柱——节理发育，柱体可能倾倒"),
    "red_sandstone_danxia":            ("rockfall",    "丹霞地貌——软硬岩层差异风化，崩塌带"),
    "snow_peaks_glaciers":             ("avalanche",   "雪山冰川——雪崩/冰崩/冰裂缝"),
    "alpine_meadow":                   ("exposure",    "高山草甸——无遮蔽，天气骤变无处躲避"),
    "plateau_plain":                   ("altitude",    "高原面——高海拔缺氧+温差剧烈"),
    "coastal_shore":                   ("storm_surge", "海岸——风暴潮+湿滑礁石"),
    "granite_boulder_field":           ("ankle_injury","花岗岩漂砾场——踝关节扭伤高风险"),
    "urban_flat":                      ("low",         ""),
    "none_visible":                    ("unknown",     ""),
}

# Rock type → stability
ROCK_STABILITY = {
    "limestone_karst":       0.2,  # 溶蚀裂隙多，极不稳定
    "quartz_sandstone_pillars": 0.3,
    "red_sandstone_danxia":  0.35,
    "granite_spheroidal":    0.8,  # 整体性好，稳定
    "snow_peaks_glaciers":   0.3,  # 冰川侵蚀，不稳定
    "alpine_lakes":          0.5,
    "none_visible":          0.5,
}

# Slope risk multiplier by terrain + vegetation
# Sparse vegetation on steep terrain = high erosion/landslide
VEG_SLOPE_MULTIPLIER = {
    "alpine_meadow":     0.8,  # 高山草甸——无根系固土
    "desert_scrub":      0.7,
    "sparse":            0.8,
    "grassland":         0.5,
    "cropland":          0.3,
    "broadleaf_deciduous":0.2,
    "broadleaf_evergreen":0.15,
    "conifer_forest":    0.2,
    "mixed_forest":      0.2,
    "bamboo_forest":     0.3,
    "tropical_rainforest":0.1,  # 密林根系好
}


# ═══════════════════════════════════════════════════════════════════
# Risk assessment engine
# ═══════════════════════════════════════════════════════════════════

@dataclass
class RiskResult:
    """Structured risk assessment output."""
    risk_level: str           # "low" | "moderate" | "high" | "critical"
    risk_score: float         # 0.0 — 1.0
    hazards: list = field(default_factory=list)       # [(type, description), ...]
    mitigations: list = field(default_factory=list)   # [action_string, ...]
    weather_alert: Optional[str] = None


def _terrain_to_steep(terrain: str) -> bool:
    """Is this terrain type steep enough for landslide/cliff concern?"""
    steep_types = {"sharp_mountains", "karst_peaks", "sandstone_pillars",
                   "desert_dunes", "plateau", "rolling_hills"}
    return terrain in steep_types


def assess_risk(field_predictions: dict,
                sensor_elevation_m: float = 500.0,
                sensor_temperature_c: float = 20.0,
                sensor_humidity_pct: float = 60.0,
                prev_humidity_pct: Optional[float] = None,
                ) -> RiskResult:
    """Analyze photo scene + sensor data for outdoor hazards.

    Args:
        field_predictions: dict of field_name → predicted_value
            (e.g. {"terrain_type": "sharp_mountains", "soil_color": "red", ...})
        sensor_elevation_m: current elevation in meters
        sensor_temperature_c: current temperature in Celsius
        sensor_humidity_pct: current relative humidity (0–100)
        prev_humidity_pct: humidity reading from ~30 min ago (for trend detection)

    Returns:
        RiskResult with level, score, hazard list, and actionable advice
    """
    hazards = []
    mitigations = []
    weather_alert = None

    terrain = field_predictions.get("terrain_type", "unknown")
    soil = field_predictions.get("soil_color", "unknown")
    water = field_predictions.get("water_type", "none_visible")
    sky = field_predictions.get("sky_quality", "unknown")
    landform = field_predictions.get("landform_detail", "unknown")
    rock = field_predictions.get("mountain_rock_type", "none_visible")
    vegetation = field_predictions.get("vegetation_zone", "unknown")
    scene = field_predictions.get("scene_type", "unknown")
    climate = field_predictions.get("climate_zone", "unknown")

    # ── 1. Terrain hazard ─────────────────────────────────────────
    terrain_info = TERRAIN_HAZARD.get(terrain, ("unknown", ""))
    if terrain_info[0] in ("high_cliff", "rockfall", "exposure", "altitude"):
        hazards.append(("terrain", terrain_info[1]))
        if terrain_info[0] == "high_cliff":
            mitigations.append("远离悬崖边缘，选择山脊线或山谷底部行走")
        elif terrain_info[0] == "rockfall":
            mitigations.append("注意头顶落石，快速通过崩塌区，佩戴头盔")
        elif terrain_info[0] == "exposure":
            mitigations.append("注意防风保暖，携带充足饮水，避免暴晒")

    # ── 2. Landslide risk (soil × terrain × vegetation) ────────────
    soil_risk = SOIL_LANDSLIDE_RISK.get(soil, 0.3)
    is_steep = _terrain_to_steep(terrain)
    veg_mult = VEG_SLOPE_MULTIPLIER.get(vegetation, 0.5)

    landslide_score = soil_risk * (0.8 if is_steep else 0.2) * veg_mult
    if landslide_score > 0.4:
        hazards.append(("landslide",
            f"{'陡坡' if is_steep else '斜坡'}+{soil}土壤+{vegetation}植被——滑坡风险高"))
        mitigations.append("避开陡坡底部和松散土体，沿山脊线行走，注意坡面裂缝")
    elif landslide_score > 0.2:
        hazards.append(("landslide_medium",
            f"中等滑坡风险——注意雨后坡面渗水和碎石滚落"))

    # ── 3. Water hazard ────────────────────────────────────────────
    water_info = WATER_HAZARD.get(water, (None, ""))
    if water_info[0] in ("flash_flood", "cliff", "rising_water", "crevasse", "storm_surge"):
        hazards.append(("water", water_info[1]))
        if water_info[0] == "flash_flood":
            mitigations.append("雨天立即远离河道和溪谷，向高处撤离，不要在峡谷底部扎营")
        elif water_info[0] == "cliff":
            mitigations.append("远离瀑布边缘湿滑区域，不要尝试攀爬湿岩")

    # Check if current location is in a flash-flood-prone landform
    landform_info = LANDFORM_HAZARD.get(landform, ("unknown", ""))
    if landform_info[0] == "flash_flood" and water_info[0] != "flash_flood":
        hazards.append(("terrain_flood", landform_info[1]))
        mitigations.append("当前处于河谷地形，即使无雨也应注意上游天气，预留撤离路线")

    # ── 4. Sky / weather visual signal ─────────────────────────────
    sky_info = SKY_HAZARD.get(sky, ("unknown", ""))
    if sky_info[0] in ("rain_coming", "zero_vis", "sandstorm"):
        hazards.append(("weather_visual", sky_info[1]))
        if sky_info[0] == "rain_coming":
            mitigations.append("降雨将至——寻找避雨处，检查防水装备，远离河道")
            weather_alert = "rain_imminent"
        elif sky_info[0] == "zero_vis":
            mitigations.append("浓雾天气就地停止移动，使用GPS/罗盘确认当前位置，不要尝试探路")
            weather_alert = "fog"
        elif sky_info[0] == "sandstorm":
            mitigations.append("沙尘暴来临——用湿布遮住口鼻，寻找背风洼地蹲下，背部迎风")

    # ── 5. Rock stability ──────────────────────────────────────────
    rock_stab = ROCK_STABILITY.get(rock, 0.5)
    if rock_stab < 0.4 and is_steep:
        hazards.append(("rock_unstable",
            f"岩体稳定性低（{rock}），在陡坡上落石风险高"))
        mitigations.append("快速通过，不要在岩壁下方停留")

    # ── 6. Scene-specific hazards ──────────────────────────────────
    if scene == "mountain_trail_snow":
        hazards.append(("avalanche", "雪山路线——雪崩/冰崩风险"))
        mitigations.append("避开30°以上积雪坡，携带雪崩信标和探杆")
    elif scene == "coastal":
        hazards.append(("tide", "海岸——注意潮汐时间和湿滑礁石"))
        mitigations.append("查询潮汐表，不要在涨潮时穿越礁石区")

    # ── 7. Sensor-based weather risk ───────────────────────────────
    if prev_humidity_pct is not None:
        humidity_delta = sensor_humidity_pct - prev_humidity_pct
        # Rapid humidity drop = incoming weather front
        if humidity_delta < -15:
            weather_alert = "storm_front"
            hazards.append(("weather_sensor",
                f"湿度骤降 {abs(humidity_delta):.0f}%——强对流天气前兆"))
            mitigations.append("天气将急剧恶化，立即寻找安全避难处，不要在山脊或树下停留")
        elif humidity_delta < -8:
            weather_alert = "weather_change"
            hazards.append(("weather_sensor",
                f"湿度快速下降 {abs(humidity_delta):.0f}%——天气转差可能性高"))

    # Sustained high humidity = soil saturation
    if sensor_humidity_pct > 85:
        hazards.append(("soil_saturated",
            f"当前湿度 {sensor_humidity_pct:.0f}%——土壤水分饱和，滑坡概率大幅上升"))
        mitigations.append("避免在陡坡和土质路面行走，注意脚下泥土松动")

    # ── 8. Elevation alerts ────────────────────────────────────────
    if sensor_elevation_m > 4000:
        hazards.append(("altitude", f"海拔 {sensor_elevation_m:.0f}m——高原反应风险"))
        mitigations.append("缓慢上升，每小时爬升不超过300m，多喝水，注意头疼/恶心症状")
    elif sensor_elevation_m > 3500 and climate in ("alpine", "boreal"):
        hazards.append(("hypothermia",
            f"高海拔 {sensor_elevation_m:.0f}m + 寒冷气候——失温风险"))
        mitigations.append("注意保暖，携带应急毯，避免出汗后静止")

    # ── Score & level ──────────────────────────────────────────────
    # Weighted: terrain 0.25, landslide 0.25, water 0.2, sky 0.15, rock 0.15
    hazard_count = len(hazards)
    has_critical = any(
        h[1].startswith(("陡峭", "河道", "河谷", "浓雾", "湿度骤降", "沙尘暴"))
        for h in hazards
    )

    if hazard_count == 0:
        risk_level, risk_score = "low", 0.05
    elif hazard_count == 1 and not has_critical:
        risk_level, risk_score = "low", 0.15
    elif hazard_count <= 2 and not has_critical:
        risk_level, risk_score = "moderate", 0.35
    elif hazard_count <= 3:
        risk_level, risk_score = "high", 0.65
    else:
        risk_level, risk_score = "critical", 0.85

    # Boost score if any critical hazard exists
    if has_critical:
        risk_score = max(risk_score, 0.7)
        if risk_level == "moderate":
            risk_level = "high"

    return RiskResult(
        risk_level=risk_level,
        risk_score=round(risk_score, 2),
        hazards=hazards,
        mitigations=mitigations,
        weather_alert=weather_alert,
    )


# ═══════════════════════════════════════════════════════════════════
# Convenience wrapper (matches dk_deploy.py pattern)
# ═══════════════════════════════════════════════════════════════════

def quick_risk(field_predictions: dict,
               elev: float = 500, temp: float = 20, humid: float = 60,
               prev_humid: Optional[float] = None) -> dict:
    """One-call interface, returns dict for JSON serialization."""
    r = assess_risk(field_predictions, elev, temp, humid, prev_humid)
    return {
        "risk_level": r.risk_level,
        "risk_score": r.risk_score,
        "hazards": [{"type": h[0], "description": h[1]} for h in r.hazards],
        "mitigations": r.mitigations,
        "weather_alert": r.weather_alert,
    }
