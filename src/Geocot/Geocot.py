"""GeoCoT: Geospatial Chain-of-Thought reasoning framework.

Implements a 3-stage coarse-to-fine pipeline for image geolocation:
  Stage 1 (Macro):  Continent-level — climate, vegetation, topography
  Stage 2 (Regional): Country-level — language, architecture, traffic, plates
  Stage 3 (Local):   City-level — sidewalk patterns, landmarks, synthesis

Based on the multi-turn CoT pattern from ImageGeoLocator_TestSet.py,
extended with structured prompts and local VLM support.
Tested with Qwen2-VL-2B-Instruct on RTX 4060 Laptop (8GB VRAM).
"""

import json
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

# Load .env for HF_ENDPOINT (mirror for China), HF_HOME, etc.
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except Exception:
    pass


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class GeoCoTStage(Enum):
    MACRO = "macro"
    REGIONAL = "regional"
    LOCAL = "local"


@dataclass
class GeoPrediction:
    city: str = "Unknown"
    country: str = "Unknown"
    continent: str = "Unknown"
    latitude: float = None
    longitude: float = None

    def __str__(self) -> str:
        if self.latitude is not None and self.longitude is not None:
            return f"{self.latitude:.4f}, {self.longitude:.4f} ({self.city}, {self.country}, {self.continent})"
        return f"{self.city}, {self.country}, {self.continent}"


@dataclass
class GeoCoTResult:
    stage_outputs: dict = field(default_factory=dict)
    final_prediction: GeoPrediction = field(default_factory=GeoPrediction)
    reasoning_chain: str = ""
    explanation: object = None  # GeoExplanation from MoE pipeline
    scene_type: object = None   # inferred SceneType int or None

    @property
    def full_reasoning(self) -> str:
        parts = []
        for stage in GeoCoTStage:
            if stage.value in self.stage_outputs:
                parts.append(f"[{stage.value.upper()}]\n{self.stage_outputs[stage.value]}")
        return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Prompt management
# ---------------------------------------------------------------------------

_PROMPTS_DIR = Path(__file__).parent / "prompts"

_DEFAULT_PROMPTS = {
    GeoCoTStage.MACRO: (
        "You are an expert geolocation analyst. Examine this street view image for broad geographic context.\n\n"
        "Analyze ONLY what you can actually see. Do NOT fabricate clues that are not visible.\n\n"
        "Observe and report on:\n"
        "- Climate zone: vegetation type, sky/clouds, soil/ground color\n"
        "- Topography: flat, hilly, mountainous, coastal, inland\n"
        "- Vegetation: density, type, any distinctive species\n"
        "- Urbanization: rural, suburban, urban, megacity\n"
        "- Road surface: paved asphalt, concrete, dirt, cobblestone\n"
        "- Apparent season: bare trees, snow, dry/brown vs lush grass\n\n"
        "Based on ALL observations, narrow to the most likely continent(s) or large region(s). "
        "Rank from most to least likely. Do NOT give a vague 'could be anywhere' answer."
    ),
    GeoCoTStage.REGIONAL: (
        "Based on the macro-level assessment:\n\"{prev_output}\"\n\n"
        "Now examine this image for country-level indicators. For EACH category, report what you ACTUALLY see:\n\n"
        "1. LANGUAGE & TEXT: Script/alphabet on signs, buildings, vehicles\n"
        "2. ARCHITECTURE: Roof style, wall materials/colors, building patterns, distinctive features\n"
        "3. ROADS & VEHICLES: Driving side, road markings, vehicle makes, license plates\n"
        "4. INFRASTRUCTURE: Utility pole design, street lights, fire hydrant style\n"
        "5. PEOPLE & CLOTHING (if visible)\n\n"
        "Identify the most likely COUNTRY (max 2 candidates). List specific clues for each.\n"
        "CRITICAL: Do NOT hallucinate. If uncertain, narrow to a specific sub-region instead."
    ),
    GeoCoTStage.LOCAL: (
        "Based on the analysis so far:\n"
        "- Macro context: {macro_output}\n"
        "- Regional assessment: {regional_output}\n\n"
        "Now synthesize ALL evidence into a precise geolocation. Consider city-level details:\n"
        "- Street furniture, sidewalk material and patterns\n"
        "- Signage style (European vs American vs Asian traffic sign standards)\n"
        "- Commercial signs, chain stores, local business types\n"
        "- Any visible landmarks or distinctive buildings\n\n"
        "Write a reasoning paragraph connecting observations to inferences.\n"
        "Then end with EXACTLY this line (nothing after it):\n"
        "LOCATION: [city], [country], [continent]\n\n"
        "Continent must be one of: Asia, Africa, Europe, North America, South America, Oceania.\n"
        "If uncertain about the city, give your best estimate. Do NOT write 'Unknown'."
    ),
}


class GeoCoTPrompt:
    """Manages stage-specific prompts and few-shot examples."""

    def __init__(self, prompts_dir: Optional[str] = None, few_shot_path: Optional[str] = None,
                 enable_knowledge_injection: bool = False):
        self._prompts = {}
        self._load_prompts(prompts_dir)
        self._few_shots = self._load_few_shots(few_shot_path)
        self._enable_knowledge = enable_knowledge_injection
        # Lazy-import knowledge retriever to avoid circular imports
        self._retriever = None

    def _get_retriever(self):
        if self._retriever is None and self._enable_knowledge:
            try:
                from .knowledge_retriever import retrieve_and_inject, retrieve_by_keywords
                self._retriever = (retrieve_and_inject, retrieve_by_keywords)
            except ImportError:
                self._enable_knowledge = False
        return self._retriever

    def _load_prompts(self, prompts_dir: Optional[str]):
        # Auto-detect enhanced prompts if not explicitly configured
        if prompts_dir is None:
            default_dir = Path(__file__).parent / "prompts"
            if (default_dir / "macro.txt").exists():
                prompts_dir = str(default_dir)
        for stage in GeoCoTStage:
            path = Path(prompts_dir) / f"{stage.value}.txt" if prompts_dir else None
            if path and path.exists():
                self._prompts[stage] = path.read_text(encoding="utf-8").strip()
            else:
                self._prompts[stage] = _DEFAULT_PROMPTS[stage]

    def _load_few_shots(self, path: Optional[str]) -> list[dict]:
        if path and Path(path).exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        # Auto-detect fewshot_china.json in prompts directory
        auto_path = Path(__file__).parent / "prompts" / "fewshot_china.json"
        if auto_path.exists():
            with open(auto_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def get_prompt(self, stage: GeoCoTStage, prev_outputs: dict,
                   sensor_data: str = "") -> str:
        template = self._prompts[stage]
        fmt = {}
        if "{prev_output}" in template and prev_outputs:
            last_key = list(prev_outputs.keys())[-1]
            fmt["prev_output"] = prev_outputs[last_key]
        if "{macro_output}" in template:
            fmt["macro_output"] = prev_outputs.get("macro", "")
        if "{regional_output}" in template:
            fmt["regional_output"] = prev_outputs.get("regional", "")
        if "{sensor_data}" in template:
            fmt["sensor_data"] = sensor_data
        try:
            prompt_text = template.format(**fmt)
        except KeyError:
            prompt_text = template

        # Inject geographic knowledge at Regional and Local stages
        if self._enable_knowledge and stage in (GeoCoTStage.REGIONAL, GeoCoTStage.LOCAL):
            r = self._get_retriever()
            if r:
                retrieve_and_inject_fn, _ = r
                # Use previous outputs for keyword-based retrieval
                all_text = " ".join(prev_outputs.values()) if prev_outputs else ""
                prompt_text = retrieve_and_inject_fn(prompt_text, prev_outputs)

        return prompt_text

    def build_few_shot_prefix(self, continent_hint: str = "", prev_outputs: dict = None) -> str:
        """Build few-shot prefix with diverse examples (avoids feedback loop)."""
        if not self._few_shots:
            return ""

        # Pick 2 diverse examples: one urban, one nature (mountain/karst/etc)
        urban_examples = [e for e in self._few_shots
                          if e.get("scene_type") in ("urban", "city")]
        nature_examples = [e for e in self._few_shots
                           if e.get("scene_type") not in ("urban", "city")]

        selected = []
        if urban_examples:
            selected.append(urban_examples[0])
        if nature_examples:
            selected.append(nature_examples[0])
        if not selected:
            selected = self._few_shots[:2]

        parts = ["以下是中国地理位置推理的参考示例：\n"]
        for i, ex in enumerate(selected, 1):
            parts.append(f"Example {i}:\n{ex.get('output', '')}\n")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Prediction extraction (offline, no GPT needed)
# ---------------------------------------------------------------------------

_COUNTRY_TO_CONTINENT = {
    "china": "Asia", "japan": "Asia", "thailand": "Asia", "south korea": "Asia",
    "india": "Asia", "vietnam": "Asia", "indonesia": "Asia", "malaysia": "Asia",
    "philippines": "Asia", "singapore": "Asia", "taiwan": "Asia",
    "france": "Europe", "germany": "Europe", "italy": "Europe", "spain": "Europe",
    "united kingdom": "Europe", "uk": "Europe", "netherlands": "Europe",
    "russia": "Europe", "portugal": "Europe", "poland": "Europe",
    "united states": "North America", "usa": "North America", "us": "North America",
    "america": "North America",
    "canada": "North America", "mexico": "North America",
    "brazil": "South America", "argentina": "South America",
    "colombia": "South America", "chile": "South America",
    "australia": "Oceania", "new zealand": "Oceania",
    "nigeria": "Africa", "south africa": "Africa", "egypt": "Africa", "kenya": "Africa",
}

_LANDMARK_TO_LOCATION = {
    # World
    "brooklyn bridge": ("New York City", "United States"),
    "statue of liberty": ("New York City", "United States"),
    "eiffel tower": ("Paris", "France"),
    "big ben": ("London", "United Kingdom"),
    "tower bridge": ("London", "United Kingdom"),
    "colosseum": ("Rome", "Italy"),
    "sagrada familia": ("Barcelona", "Spain"),
    "tokyo tower": ("Tokyo", "Japan"),
    "sydney opera house": ("Sydney", "Australia"),
    "golden gate bridge": ("San Francisco", "United States"),
    "notre dame": ("Paris", "France"),
    "brandenburg gate": ("Berlin", "Germany"),
    # China: urban landmarks
    "great wall": ("Beijing", "China"),
    "forbidden city": ("Beijing", "China"),
    "temple of heaven": ("Beijing", "China"),
    "summer palace": ("Beijing", "China"),
    "oriental pearl tower": ("Shanghai", "China"),
    "the bund": ("Shanghai", "China"),
    "canton tower": ("Guangzhou", "China"),
    "west lake": ("Hangzhou", "China"),
    "potala palace": ("Lhasa", "China"),
    "terracotta warriors": ("Xi'an", "China"),
    "city wall": ("Xi'an", "China"),
    # China: mountain scenic areas
    "huangshan": ("Huangshan", "China"),
    "yellow mountain": ("Huangshan", "China"),
    "zhangjiajie": ("Zhangjiajie", "China"),
    "wulingyuan": ("Zhangjiajie", "China"),
    "tianmen mountain": ("Zhangjiajie", "China"),
    "guilin karst": ("Guilin", "China"),
    "li river": ("Guilin", "China"),
    "tiger leaping gorge": ("Tiger Leaping Gorge", "China"),
    "jade dragon snow mountain": ("Lijiang", "China"),
    "yulong snow mountain": ("Lijiang", "China"),
    "emei mountain": ("Emeishan City", "China"),
    "emeishan": ("Emeishan City", "China"),
    "mount tai": ("Tai'an", "China"),
    "taishan": ("Tai'an", "China"),
    "mount hua": ("Huayin", "China"),
    "huashan": ("Huayin", "China"),
    "lushan": ("Lushan", "China"),
    "wuyishan": ("Wuyishan City", "China"),
    "wuyi mountain": ("Wuyishan City", "China"),
    "four girls mountain": ("Siguniangshan", "China"),
    "siguniang mountain": ("Siguniangshan", "China"),
    "siguniangshan": ("Siguniangshan", "China"),
    "jiuzhaigou": ("Jiuzhaigou", "China"),
    "meili snow mountain": ("Deqin", "China"),
    "kawagebo": ("Deqin", "China"),
    "daocheng yading": ("Daocheng", "China"),
    "yading nature reserve": ("Daocheng", "China"),
    "yubeng": ("Deqin", "China"),
    "sanqingshan": ("Shangrao", "China"),
    "leshan giant buddha": ("Leshan", "China"),
    "hailuogou glacier": ("Moxi", "China"),
    "gongga mountain": ("Garze", "China"),
    "wudang mountain": ("Shiyan", "China"),
    "wudangshan": ("Shiyan", "China"),
    "huanglong": ("Songpan", "China"),
    "shennongjia": ("Shennongjia", "China"),
}

_CITY_TO_COUNTRY = {
    "new york": "United States", "new york city": "United States",
    "los angeles": "United States", "chicago": "United States",
    "san francisco": "United States", "miami": "United States",
    "london": "United Kingdom", "paris": "France", "berlin": "Germany",
    "rome": "Italy", "madrid": "Spain", "barcelona": "Spain",
    "tokyo": "Japan", "osaka": "Japan",
    "beijing": "China", "shanghai": "China", "shenzhen": "China",
    "hong kong": "China", "guangzhou": "China", "fuzhou": "China",
    "bangkok": "Thailand", "seoul": "South Korea", "singapore": "Singapore",
    "sydney": "Australia", "melbourne": "Australia",
    "toronto": "Canada", "vancouver": "Canada",
    "moscow": "Russia", "istanbul": "Turkey", "cairo": "Egypt", "dubai": "UAE",
    # Chinese provincial capitals
    "chengdu": "China", "chongqing": "China", "hangzhou": "China",
    "nanjing": "China", "wuhan": "China", "xi'an": "China",
    "kunming": "China", "changsha": "China", "hefei": "China",
    "nanchang": "China", "guiyang": "China", "lanzhou": "China",
    "nanning": "China", "harbin": "China", "changchun": "China",
    "shenyang": "China", "jinan": "China", "zhengzhou": "China",
    "shijiazhuang": "China", "taiyuan": "China", "hohhot": "China",
    "urumqi": "China", "lhasa": "China", "xining": "China",
    "yinchuan": "China", "haikou": "China", "tianjin": "China",
    "macau": "China", "taipei": "China",
    # Chinese major cities
    "suzhou": "China", "wuxi": "China", "ningbo": "China",
    "wenzhou": "China", "xiamen": "China", "qingdao": "China",
    "dalian": "China", "zhuhai": "China", "foshan": "China",
    "dongguan": "China", "luoyang": "China", "kaifeng": "China",
    "dali": "China", "lijiang": "China", "guilin": "China",
    "sanya": "China", "beihai": "China", "yan'an": "China",
    "jingdezhen": "China", "zigong": "China",
    # Chinese scenic/trail towns
    "tangkou": "China", "wulingyuan": "China", "zhangjiajie city": "China",
    "shangri-la": "China", "zhongdian": "China",
    "qiaotou": "China", "daju": "China", "hutiaoxia": "China",
    "riwa": "China", "xinduqiao": "China", "moxi": "China",
    "feilaixi": "China", "bingzhongluo": "China",
    "daocheng": "China", "litang": "China", "kangding": "China",
    "deqin": "China", "aba": "China", "garze": "China",
    "jiuzhaigou": "China", "jinghong": "China", "tengchong": "China",
    "huangshan city": "China", "wuyishan city": "China",
    "emeishan city": "China",
}


def _infer_continent(country: str) -> str:
    return _COUNTRY_TO_CONTINENT.get(country.lower(), "Unknown")


def extract_prediction(text: str) -> GeoPrediction:
    """Extract coordinates and location from reasoning text.

    Strategy (coordinates-first):
      1. COORDINATES: line → lat, lng (primary target)
      2. LOCATION: line → city, country, continent
      3. Coord→city/trail reverse lookup if only coords available
      4. City/trail→coord forward lookup if only location available
    """
    lat, lng = _extract_coordinates(text)
    city, country, continent = _extract_location_strict(text)

    # If we have coords but no proper location, reverse-lookup nearest city or trail
    if lat is not None and lng is not None and (not city or city == "Unknown"):
        city, country = _coords_to_nearest_city(lat, lng)
        if city:
            continent = _infer_continent(country)

    # If we have a trail/scenic name, forward-lookup coordinates with higher precision
    if city:
        trail_info = lookup_trail(city)
        if trail_info:
            lat, lng = trail_info[0], trail_info[1]
            if not country or country == "Unknown":
                country = "China"
                continent = "Asia"

    # If we have location but no coords, forward-lookup city or trail
    if (lat is None or lng is None) and (city or country):
        trail_info = lookup_trail(city) if city else None
        if trail_info:
            lat, lng = trail_info[0], trail_info[1]
        else:
            lat, lng = _city_to_coords(city or "Unknown", country or "Unknown")

    return GeoPrediction(
        city=city or "Unknown",
        country=country or "Unknown",
        continent=continent or "Unknown",
        latitude=lat,
        longitude=lng,
    )


def _extract_coordinates(text: str) -> tuple:
    """Extract latitude/longitude from COORDINATES: line. Returns (lat, lng) or (None, None).

    Takes only the FIRST COORDINATES: match if multiple exist.
    """
    # Pattern: COORDINATES: <lat>, <lng>
    coord_pattern = re.compile(
        r"COORDINATES:\s*([+-]?\d+\.?\d*)\s*[\s,;]+\s*([+-]?\d+\.?\d*)",
        re.IGNORECASE,
    )
    m = coord_pattern.search(text)
    if m:
        lat = max(-90.0, min(90.0, float(m.group(1))))
        lng = max(-180.0, min(180.0, float(m.group(2))))
        return lat, lng

    # Pattern: "40.7128° N, 74.0060° W"
    nswe_pattern = re.compile(
        r"(\d+\.?\d*)\s*°?\s*([NS])\s*[,;]\s*(\d+\.?\d*)\s*°?\s*([EW])",
        re.IGNORECASE,
    )
    m = nswe_pattern.search(text)
    if m:
        lat = float(m.group(1)) * (1 if m.group(2).upper() == "N" else -1)
        lng = float(m.group(3)) * (1 if m.group(4).upper() == "E" else -1)
        return max(-90.0, min(90.0, lat)), max(-180.0, min(180.0, lng))

    # Pattern: Chinese coordinate format "北纬30.10度，东经118.18度"
    cn_pattern = re.compile(
        r"北纬\s*(\d+\.?\d*)\s*度?\s*[,，\s]+\s*东经\s*(\d+\.?\d*)\s*度?",
        re.IGNORECASE,
    )
    m = cn_pattern.search(text)
    if m:
        return float(m.group(1)), float(m.group(2))

    # Pattern: Chinese label "坐标：30.10, 118.18"
    cn_label_pattern = re.compile(
        r"(?:坐标|经纬度)[：:]\s*([+-]?\d+\.?\d*)\s*[,，\s;；]+\s*([+-]?\d+\.?\d*)",
        re.IGNORECASE,
    )
    m = cn_label_pattern.search(text)
    if m:
        lat = max(-90.0, min(90.0, float(m.group(1))))
        lng = max(-180.0, min(180.0, float(m.group(2))))
        return lat, lng

    # Fallback: JSON with "latitude" and "longitude" fields
    json_pattern = re.compile(
        r'"latitude"\s*:\s*([+-]?\d+\.?\d*)\s*[,}\s].*?"longitude"\s*:\s*([+-]?\d+\.?\d*)',
        re.DOTALL,
    )
    m = json_pattern.search(text)
    if m:
        lat = max(-90.0, min(90.0, float(m.group(1))))
        lng = max(-180.0, min(180.0, float(m.group(2))))
        return lat, lng

    return None, None


def _is_template_placeholder(s: str) -> bool:
    """Check if a string looks like a template placeholder rather than actual content."""
    s = s.strip()
    if not s:
        return True
    # [placeholder], [city name], City Name, etc.
    if re.match(r"^\[.*\]$", s):
        return True
    if s.lower() in {"city name", "country name", "continent name", "major city", "unknown", "city",
                       "n/a", "na", "none", "null", "placeholder", "province name"}:
        return True
    return False


def _extract_location_strict(text: str) -> tuple:
    """Extract city, country, continent from LOCATION: line only.

    Rejects template placeholders like '[city name]'.
    Returns (city, country, continent) or (None, None, None).
    """
    # Strict LOCATION: City, Country, Continent format
    loc_pattern = re.compile(
        r"LOCATION:\s*([^,\[\]\n]+?),\s*([^,\[\]\n]+?),\s*"
        r"(Asia|Africa|Europe|North America|South America|Oceania)",
        re.IGNORECASE,
    )
    for m in loc_pattern.finditer(text):
        c1, c2, c3 = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
        # Reject template placeholders
        if _is_template_placeholder(c1) or _is_template_placeholder(c2):
            continue
        # city should not be extremely long
        if len(c1) > 40 or len(c2) > 40:
            continue
        return c1, c2, c3

    # Fallback: JSON with "location_name" and "province" fields
    json_city = re.search(r'"location_name"\s*:\s*"([^"]+)"', text)
    json_province = re.search(r'"province"\s*:\s*"([^"]+)"', text)
    if json_city:
        city = json_city.group(1).strip()
        if not _is_template_placeholder(city) and len(city) <= 40:
            province = json_province.group(1).strip() if json_province else "China"
            country = "China"
            continent = "Asia"
            return city, country, continent

    return None, None, None


def _coords_to_nearest_city(lat: float, lng: float) -> tuple:
    """Find the closest city or trail in our database to the given coordinates.

    Returns (city, country) or (None, None).
    """
    _init_city_db()
    best_name, best_country, best_dist = None, None, float("inf")

    # Check city database
    for (city_key, country_key), (clat, clng) in _CITY_COORD_DB.items():
        dist = ((lat - clat) ** 2 + (lng - clng) ** 2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_name = city_key.title()
            best_country = country_key.title()

    # Check trail database (with bonus weighting for trails → prefer them nearby)
    for trail_name, (tlat, tlng, province) in _TRAIL_COORD_DB.items():
        dist = ((lat - tlat) ** 2 + (lng - tlng) ** 2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_name = trail_name.title()
            best_country = "china"

    # Only return if within reasonable distance (~30 degrees ≈ ~3300km)
    if best_dist < 30:
        return best_name, best_country
    return None, None


def _city_to_coords(city: str, country: str) -> tuple:
    """Rough coordinate lookup for common cities. Returns (lat, lng) or (None, None)."""
    _init_city_db()
    key = (city.strip().lower(), country.strip().lower())
    return _CITY_COORD_DB.get(key, (None, None))


# Shared city↔coordinate database
_CITY_COORD_DB = {}
_TRAIL_COORD_DB = {}  # hiking trail / scenic area → (lat, lng, province)
_DB_INITIALIZED = False

# Chinese province abbreviations → full name and capital
_PROVINCE_MAP = {
    "京": ("beijing", "beijing", 39.9042, 116.4074),
    "沪": ("shanghai", "shanghai", 31.2304, 121.4737),
    "津": ("tianjin", "tianjin", 39.0842, 117.2009),
    "渝": ("chongqing", "chongqing", 29.4316, 106.9123),
    "粤": ("guangdong", "guangzhou", 23.1291, 113.2644),
    "苏": ("jiangsu", "nanjing", 32.0603, 118.7969),
    "浙": ("zhejiang", "hangzhou", 30.2741, 120.1551),
    "鲁": ("shandong", "jinan", 36.6510, 117.1201),
    "豫": ("henan", "zhengzhou", 34.7466, 113.6253),
    "冀": ("hebei", "shijiazhuang", 38.0423, 114.5149),
    "川": ("sichuan", "chengdu", 30.5728, 104.0668),
    "云": ("yunnan", "kunming", 25.0389, 102.7183),
    "贵": ("guizhou", "guiyang", 26.6477, 106.6302),
    "桂": ("guangxi", "nanning", 22.8170, 108.3665),
    "鄂": ("hubei", "wuhan", 30.5928, 114.3055),
    "湘": ("hunan", "changsha", 28.2278, 112.9389),
    "赣": ("jiangxi", "nanchang", 28.6820, 115.8582),
    "闽": ("fujian", "fuzhou", 26.0745, 119.2965),
    "皖": ("anhui", "hefei", 31.8206, 117.2272),
    "陕": ("shaanxi", "xi'an", 34.3416, 108.9398),
    "晋": ("shanxi", "taiyuan", 37.8706, 112.5489),
    "黑": ("heilongjiang", "harbin", 45.8038, 126.5350),
    "吉": ("jilin", "changchun", 43.8990, 125.2249),
    "辽": ("liaoning", "shenyang", 41.8057, 123.4315),
    "蒙": ("inner mongolia", "hohhot", 40.8424, 111.7490),
    "甘": ("gansu", "lanzhou", 36.0611, 103.8343),
    "青": ("qinghai", "xining", 36.6171, 101.7785),
    "新": ("xinjiang", "urumqi", 43.8256, 87.6168),
    "藏": ("tibet", "lhasa", 29.6500, 91.1000),
    "宁": ("ningxia", "yinchuan", 38.4872, 106.2309),
    "琼": ("hainan", "haikou", 20.0174, 110.3492),
}


def _infer_scene_type_from_geocot(text: str):
    """Infer SceneType int from GeoCoT output text (for MoE reconciliation).

    Returns int (0-3) or None if unclear.
    """
    if not text:
        return None
    text_lower = text.lower()

    urban_kw = ["城市", "建筑", "街道", "沥青", "路面", "urban", "city",
                "road", "building", "traffic", "shanghai", "beijing"]
    karst_kw = ["峰林", "花岗岩", "喀斯特", "石灰岩", "石英砂岩",
                "karst", "granite", "peak", "guilin", "huangshan",
                "zhangjiajie", "sandstone"]
    alpine_kw = ["雪山", "冰川", "高山", "海拔", "alpine", "snow",
                 "mountain", "glacier", "plateau", "tibet",
                 "siguniang", "gongga", "meili"]
    other_kw = ["峡谷", "古镇", "草原", "沙漠", "湖泊", "gorge",
                "old town", "grassland", "desert", "lake", "canyon",
                "lijiang", "tiger leaping", "dali"]

    scores = {
        0: sum(1 for kw in urban_kw if kw in text_lower),
        1: sum(1 for kw in karst_kw if kw in text_lower),
        2: sum(1 for kw in alpine_kw if kw in text_lower),
        3: sum(1 for kw in other_kw if kw in text_lower),
    }
    best = max(scores, key=scores.get)
    return int(best) if scores[best] > 0 else None


def _init_city_db():
    """Lazy-init the coordinate databases (cities + trails)."""
    global _CITY_COORD_DB, _TRAIL_COORD_DB, _DB_INITIALIZED
    if _DB_INITIALIZED:
        return

    # ── World major cities ──────────────────────────────────────────
    _CITY_COORD_DB = {
        # East Asia
        ("beijing", "china"): (39.9042, 116.4074),
        ("shanghai", "china"): (31.2304, 121.4737),
        ("shenzhen", "china"): (22.5431, 114.0579),
        ("guangzhou", "china"): (23.1291, 113.2644),
        ("hong kong", "china"): (22.3193, 114.1694),
        ("tokyo", "japan"): (35.6762, 139.6503),
        ("osaka", "japan"): (34.6937, 135.5023),
        ("seoul", "south korea"): (37.5665, 126.9780),
        ("singapore", "singapore"): (1.3521, 103.8198),
        ("bangkok", "thailand"): (13.7563, 100.5018),
        ("hanoi", "vietnam"): (21.0278, 105.8342),
        ("ho chi minh city", "vietnam"): (10.8231, 106.6297),

        # Europe
        ("london", "united kingdom"): (51.5074, -0.1278),
        ("paris", "france"): (48.8566, 2.3522),
        ("berlin", "germany"): (52.5200, 13.4050),
        ("rome", "italy"): (41.9028, 12.4964),
        ("madrid", "spain"): (40.4168, -3.7038),
        ("barcelona", "spain"): (41.3874, 2.1686),
        ("moscow", "russia"): (55.7558, 37.6173),
        ("istanbul", "turkey"): (41.0082, 28.9784),

        # Americas
        ("new york city", "united states"): (40.7128, -74.0060),
        ("new york", "united states"): (40.7128, -74.0060),
        ("los angeles", "united states"): (34.0522, -118.2437),
        ("chicago", "united states"): (41.8781, -87.6298),
        ("san francisco", "united states"): (37.7749, -122.4194),
        ("toronto", "canada"): (43.6532, -79.3832),
        ("vancouver", "canada"): (49.2827, -123.1207),
        ("mexico city", "mexico"): (19.4326, -99.1332),
        ("sao paulo", "brazil"): (-23.5505, -46.6333),
        ("rio de janeiro", "brazil"): (-22.9068, -43.1729),
        ("buenos aires", "argentina"): (-34.6037, -58.3816),
        ("lima", "peru"): (-12.0464, -77.0428),

        # South Asia / Oceania / Africa / Middle East
        ("mumbai", "india"): (19.0760, 72.8777),
        ("delhi", "india"): (28.6139, 77.2090),
        ("sydney", "australia"): (-33.8688, 151.2093),
        ("melbourne", "australia"): (-37.8136, 144.9631),
        ("dubai", "uae"): (25.2048, 55.2708),
        ("cairo", "egypt"): (30.0444, 31.2357),
        ("nairobi", "kenya"): (-1.2921, 36.8219),
        ("cape town", "south africa"): (-33.9249, 18.4241),
        ("lagos", "nigeria"): (6.5244, 3.3792),
        ("jakarta", "indonesia"): (-6.2088, 106.8456),
        ("manila", "philippines"): (14.5995, 120.9842),
        ("kuala lumpur", "malaysia"): (3.1390, 101.6869),
    }

    # ── China: provincial capitals ───────────────────────────────────
    _CITY_COORD_DB.update({
        ("chengdu", "china"): (30.5728, 104.0668),
        ("chongqing", "china"): (29.4316, 106.9123),
        ("hangzhou", "china"): (30.2741, 120.1551),
        ("nanjing", "china"): (32.0603, 118.7969),
        ("wuhan", "china"): (30.5928, 114.3055),
        ("xi'an", "china"): (34.3416, 108.9398),
        ("kunming", "china"): (25.0389, 102.7183),
        ("changsha", "china"): (28.2278, 112.9389),
        ("fuzhou", "china"): (26.0745, 119.2965),
        ("hefei", "china"): (31.8206, 117.2272),
        ("nanchang", "china"): (28.6820, 115.8582),
        ("guiyang", "china"): (26.6477, 106.6302),
        ("lanzhou", "china"): (36.0611, 103.8343),
        ("nanning", "china"): (22.8170, 108.3665),
        ("harbin", "china"): (45.8038, 126.5350),
        ("changchun", "china"): (43.8990, 125.2249),
        ("shenyang", "china"): (41.8057, 123.4315),
        ("jinan", "china"): (36.6510, 117.1201),
        ("zhengzhou", "china"): (34.7466, 113.6253),
        ("shijiazhuang", "china"): (38.0423, 114.5149),
        ("taiyuan", "china"): (37.8706, 112.5489),
        ("hohhot", "china"): (40.8424, 111.7490),
        ("urumqi", "china"): (43.8256, 87.6168),
        ("lhasa", "china"): (29.6500, 91.1000),
        ("xining", "china"): (36.6171, 101.7785),
        ("yinchuan", "china"): (38.4872, 106.2309),
        ("haikou", "china"): (20.0174, 110.3492),
        ("tianjin", "china"): (39.0842, 117.2009),
        ("macau", "china"): (22.1987, 113.5439),
        ("taipei", "china"): (25.0330, 121.5654),
    })

    # ── China: major prefecture cities ────────────────────────────────
    _CITY_COORD_DB.update({
        ("suzhou", "china"): (31.2990, 120.5853),
        ("wuxi", "china"): (31.4912, 120.3119),
        ("ningbo", "china"): (29.8683, 121.5440),
        ("wenzhou", "china"): (28.0015, 120.6989),
        ("xiamen", "china"): (24.4798, 118.0894),
        ("qingdao", "china"): (36.0671, 120.3826),
        ("dalian", "china"): (38.9140, 121.6147),
        ("zhuhai", "china"): (22.2707, 113.5767),
        ("foshan", "china"): (23.0218, 113.1214),
        ("dongguan", "china"): (23.0208, 113.7518),
        ("luoyang", "china"): (34.6181, 112.4536),
        ("kaifeng", "china"): (34.7975, 114.3076),
        ("dali", "china"): (25.6065, 100.2676),
        ("lijiang", "china"): (26.8721, 100.2299),
        ("guilin", "china"): (25.2736, 110.2900),
        ("sanya", "china"): (18.2528, 109.5120),
        ("beihai", "china"): (21.4733, 109.1192),
        ("yan'an", "china"): (36.5855, 109.4897),
        ("jingdezhen", "china"): (29.2708, 117.1784),
        ("zigong", "china"): (29.3392, 104.7784),
        ("aba", "china"): (31.8995, 102.2245),  # Ngawa, near Siguniang
        ("garze", "china"): (30.0490, 101.9625),  # Ganzi, near Gongga/Yading
        ("deqin", "china"): (28.4860, 98.9180),  # near Meili Snow Mountain
        ("kangding", "china"): (30.0530, 101.9630),
        ("litang", "china"): (30.0310, 100.2720),
        ("daocheng", "china"): (29.0529, 100.2944),
        ("jiuzhaigou", "china"): (33.2630, 103.9190),
        ("jinghong", "china"): (22.0091, 100.7970),  # Xishuangbanna
        ("tengchong", "china"): (25.0205, 98.4973),
        ("xichang", "china"): (27.8949, 102.2644),
        ("shannan", "china"): (29.2370, 91.7720),
        ("nyingchi", "china"): (29.6483, 94.3619),
        ("yanbian", "china"): (42.9094, 129.5133),
        ("hulunbuir", "china"): (49.2116, 119.7658),
        ("altay", "china"): (47.8449, 88.1370),
        ("kashgar", "china"): (39.4677, 75.9897),
        ("turpan", "china"): (42.9513, 89.1841),
        ("dunhuang", "china"): (40.1421, 94.6620),
        ("zhangye", "china"): (38.9258, 100.4498),
        ("huangshan city", "china"): (29.7153, 118.3377),  # Huangshan city (Tunxi)
        ("wuyishan city", "china"): (27.7563, 118.0353),
        ("emeishan city", "china"): (29.6012, 103.4825),
        ("qufu", "china"): (35.5807, 116.9865),
        ("datong", "china"): (40.0768, 113.3001),
        ("pingyao", "china"): (37.2020, 112.1775),
        ("yangshuo", "china"): (24.7785, 110.4965),
        ("fenghuang", "china"): (27.9482, 109.5992),
        ("wuzhen", "china"): (30.7460, 120.4866),
        ("zhouzhuang", "china"): (31.1150, 120.8455),
        ("tongli", "china"): (31.1583, 120.7188),
    })

    # ── China: county-level towns near hiking areas ──────────────────
    _CITY_COORD_DB.update({
        ("tangkou", "china"): (30.0900, 118.1800),  # 汤口镇, Huangshan south gate
        ("wulingyuan", "china"): (29.3465, 110.5500),  # 武陵源, Zhangjiajie
        ("zhangjiajie city", "china"): (29.1170, 110.4780),
        ("shangri-la", "china"): (27.8298, 99.7040),  # 香格里拉 (Zhongdian)
        ("zhongdian", "china"): (27.8298, 99.7040),
        ("qiaotou", "china"): (27.0080, 100.0900),  # 桥头镇, Tiger Leaping Gorge start
        ("daju", "china"): (27.3020, 100.2650),  # 大具, Tiger Leaping Gorge end
        ("hutiaoxia", "china"): (27.1400, 100.2000),  # 虎跳峡镇
        ("riwa", "china"): (28.4100, 100.3500),  # 日瓦镇, Yading gateway
        ("xinduqiao", "china"): (30.0430, 101.5030),  # 新都桥, Kangding area
        ("moxi", "china"): (29.6500, 102.1200),  # 磨西镇, Hailuogou / Gongga
        ("feilaixi", "china"): (28.4450, 98.8780),  # 飞来寺, Meili Snow Mountain view
        ("bingzhongluo", "china"): (28.0150, 98.6230),  # 丙中洛
        ("lugu lake", "china"): (27.7030, 100.7910),  # 泸沽湖
        ("yading", "china"): (28.4300, 100.3500),  # 亚丁村
    })

    # ── Hiking trails & scenic areas ─────────────────────────────────
    _TRAIL_COORD_DB = {
        # Anhui 安徽
        "huangshan": (30.1296, 118.1690, "anhui"),
        "yellow mountain": (30.1296, 118.1690, "anhui"),
        "tunxi old street": (29.7153, 118.3377, "anhui"),

        # Hunan 湖南
        "zhangjiajie": (29.3355, 110.4807, "hunan"),
        "wulingyuan": (29.3465, 110.5500, "hunan"),
        "tianmen mountain": (29.0516, 110.4830, "hunan"),
        "fenghuang ancient town": (27.9482, 109.5992, "hunan"),

        # Yunnan 云南
        "tiger leaping gorge": (27.1980, 100.1183, "yunnan"),
        "yubeng": (28.4100, 98.7800, "yunnan"),
        "meili snow mountain": (28.4380, 98.6850, "yunnan"),
        "kawagebo": (28.4380, 98.6850, "yunnan"),
        "jade dragon snow mountain": (27.0980, 100.1750, "yunnan"),
        "yulong snow mountain": (27.0980, 100.1750, "yunnan"),
        "lijiang old town": (26.8721, 100.2299, "yunnan"),
        "shangri-la old town": (27.8298, 99.7040, "yunnan"),
        "hutiaoxia": (27.1400, 100.2000, "yunnan"),
        "lugu lake": (27.7030, 100.7910, "yunnan"),
        "shaxi ancient town": (26.3180, 99.8530, "yunnan"),
        "weishan old town": (25.2320, 100.3100, "yunnan"),

        # Sichuan 四川
        "siguniang mountain": (31.1000, 102.9000, "sichuan"),
        "siguniangshan": (31.1000, 102.9000, "sichuan"),
        "four girls mountain": (31.1000, 102.9000, "sichuan"),
        "changping valley": (31.1200, 102.8800, "sichuan"),
        "haizi valley": (31.0300, 102.9000, "sichuan"),
        "daocheng yading": (28.4200, 100.3500, "sichuan"),
        "yading nature reserve": (28.4200, 100.3500, "sichuan"),
        "gongga mountain": (29.5950, 101.8790, "sichuan"),
        "minya konka": (29.5950, 101.8790, "sichuan"),
        "hailuogou glacier": (29.6100, 102.0700, "sichuan"),
        "jiuzhaigou": (33.2630, 103.9190, "sichuan"),
        "jiuzhaigou valley": (33.2630, 103.9190, "sichuan"),
        "huanglong": (32.7500, 103.8300, "sichuan"),
        "emeishan": (29.6012, 103.4825, "sichuan"),
        "emei mountain": (29.6012, 103.4825, "sichuan"),
        "leshan giant buddha": (29.5470, 103.7690, "sichuan"),
        "mount qingcheng": (30.9800, 103.5250, "sichuan"),
        "langzhong ancient city": (31.5600, 105.9800, "sichuan"),

        # Fujian 福建
        "wuyishan": (27.7180, 117.6820, "fujian"),
        "wuyi mountain": (27.7180, 117.6820, "fujian"),
        "tulou": (24.5900, 117.0600, "fujian"),

        # Jiangxi 江西
        "lushan": (29.5730, 115.9740, "jiangxi"),
        "sanqingshan": (28.9150, 118.0820, "jiangxi"),
        "jinggangshan": (26.5600, 114.1700, "jiangxi"),
        "wuyuan": (29.2570, 117.8620, "jiangxi"),

        # Shaanxi 陕西
        "huashan": (34.4830, 110.0890, "shaanxi"),
        "hua mountain": (34.4830, 110.0890, "shaanxi"),
        "mount hua": (34.4830, 110.0890, "shaanxi"),
        "zhongnan mountain": (33.9300, 108.9700, "shaanxi"),

        # Shandong 山东
        "taishan": (36.2610, 117.1110, "shandong"),
        "tai mountain": (36.2610, 117.1110, "shandong"),
        "laoshan": (36.1470, 120.6620, "shandong"),

        # Zhejiang 浙江
        "west lake": (30.2380, 120.1440, "zhejiang"),
        "putuoshan": (30.0100, 122.3880, "zhejiang"),
        "yandangshan": (28.3730, 121.0620, "zhejiang"),
        "mogan mountain": (30.6250, 119.8800, "zhejiang"),

        # Guangxi 广西
        "guilin karst": (25.2736, 110.2900, "guangxi"),
        "li river": (25.0300, 110.4200, "guangxi"),
        "yangshuo karst": (24.7785, 110.4965, "guangxi"),
        "longji rice terraces": (25.7600, 110.1400, "guangxi"),

        # Tibet 西藏
        "mount kailash": (31.0670, 81.3120, "tibet"),
        "lake manasarovar": (30.6700, 81.4700, "tibet"),
        "everest base camp": (28.1417, 86.8550, "tibet"),
        "potala palace": (29.6576, 91.1169, "tibet"),
        "yarlung tsangpo grand canyon": (29.6100, 94.9100, "tibet"),

        # Hubei 湖北
        "wudangshan": (32.4200, 111.0200, "hubei"),
        "wudang mountain": (32.4200, 111.0200, "hubei"),
        "three gorges": (30.8300, 111.0000, "hubei"),
        "shennongjia": (31.7400, 110.6800, "hubei"),

        # Hebei 河北
        "great wall badaling": (40.3550, 116.0160, "hebei"),
        "great wall mutianyu": (40.4310, 116.5650, "beijing"),
        "great wall jinshanling": (40.6850, 117.2330, "hebei"),
        "chengde mountain resort": (40.9880, 117.9380, "hebei"),
        "bashang grassland": (41.5500, 115.9000, "hebei"),

        # Xinjiang 新疆
        "kanas lake": (48.8200, 87.0400, "xinjiang"),
        "heavenly lake": (43.8830, 88.1330, "xinjiang"),
        "tianchi xinjiang": (43.8830, 88.1330, "xinjiang"),

        # Other famous destinations
        "mount heng shanxi": (39.6700, 113.7300, "shanxi"),
        "mount heng hunan": (27.2560, 112.7100, "hunan"),
        "mount song": (34.5020, 112.9350, "henan"),
        "shaolin temple": (34.5070, 112.9360, "henan"),
        "hanging temple": (39.6730, 113.7120, "shanxi"),
        "yungang grottoes": (40.1100, 113.1280, "shanxi"),
        "mogao caves": (40.0370, 94.8040, "gansu"),
        "rainbow mountains": (38.9258, 100.4498, "gansu"),
        "terracotta warriors": (34.3850, 109.2730, "shaanxi"),
        "mount lu": (29.5730, 115.9740, "jiangxi"),
        "dali old town": (25.6065, 100.2676, "yunnan"),
        "erce lake": (25.7590, 100.1820, "yunnan"),
        "shuhe ancient town": (26.9200, 100.2090, "yunnan"),
    }

    _DB_INITIALIZED = True


def lookup_trail(name: str):
    """Look up a hiking trail or scenic area by name. Returns (lat, lng, province) or None."""
    _init_city_db()
    key = name.strip().lower()
    if key in _TRAIL_COORD_DB:
        return _TRAIL_COORD_DB[key]
    return None


def lookup_province(abbr: str):
    """Look up province info by Chinese abbreviation. Returns (province, capital, lat, lng) or None."""
    return _PROVINCE_MAP.get(abbr)


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

MAX_IMAGE_PIXELS = 504 * 28 * 28  # ~395K pixels, fits in 8GB VRAM


def resize_image_for_vlm(image: Image.Image, max_pixels: int = 500000) -> Image.Image:
    """Resize image if too large, to avoid CUDA OOM."""
    w, h = image.size
    if w * h > max_pixels:
        scale = (max_pixels / (w * h)) ** 0.5
        image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return image


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class GeoCoTPipeline:
    """3-stage coarse-to-fine geolocation reasoning pipeline."""

    def __init__(self, model_fn, prompt_config: Optional[dict] = None):
        self._model_fn = model_fn
        self._prompt = GeoCoTPrompt(
            prompts_dir=prompt_config.get("prompts_dir") if prompt_config else None,
            few_shot_path=prompt_config.get("few_shot_path") if prompt_config else None,
            enable_knowledge_injection=prompt_config.get("enable_knowledge", True) if prompt_config else True,
        )

    def run(self, image: Image.Image,
            sensor_elevation_m: Optional[float] = None,
            sensor_temperature_c: Optional[float] = None,
            sensor_humidity_pct: Optional[float] = None) -> GeoCoTResult:
        """Run the full 3-stage GeoCoT pipeline on a single image."""
        # Build sensor hint for prompt injection
        sensor_parts = []
        if sensor_elevation_m is not None:
            sensor_parts.append(f"- 海拔：{sensor_elevation_m:.0f}m")
        if sensor_temperature_c is not None:
            sensor_parts.append(f"- 温度：{sensor_temperature_c:.0f}°C")
        if sensor_humidity_pct is not None:
            sensor_parts.append(f"- 湿度：{sensor_humidity_pct:.0f}%")
        if sensor_parts:
            # Build hard physical constraints from sensor values.
            # Only exclude what is PHYSICALLY IMPOSSIBLE — no exceptions or
            # "unless" clauses that create ambiguity.
            constraints = []
            if sensor_elevation_m is not None:
                e = sensor_elevation_m
                if e < 200:
                    constraints.append("- 海拔<200m → 排除 alpine气候、plateau地形")
                if e > 3000:
                    constraints.append("- 海拔>3000m → 排除 tropical/subtropical气候、urban_flat/farmland_plain地形")
            if sensor_temperature_c is not None:
                t = sensor_temperature_c
                if t < 10:
                    constraints.append("- 温度<10°C → 排除 tropical 气候")
                if t > 30:
                    constraints.append("- 温度>30°C → 排除 alpine/boreal 气候")
            if sensor_humidity_pct is not None:
                h = sensor_humidity_pct
                if h < 30:
                    constraints.append("- 湿度<30% → 排除 tropical_rainforest、arid气候下除外")
                if h > 70:
                    constraints.append("- 湿度>70% → 排除 desert_scrub")

            constraint_text = ""
            if constraints:
                constraint_text = (
                    "\n【硬约束 — 物理排除】\n"
                    "根据传感器数据，以下场景在物理上极不可能。\n"
                    "除非图中视觉证据极其压倒性，否则不要选择：\n"
                    + "\n".join(constraints) + "\n"
                )

            sensor_data = (
                "【物理基准】设备实测数据，气压海拔已校准（误差±15%），温度湿度精度可靠：\n"
                + "\n".join(sensor_parts)
                + constraint_text
                + "\n"
                + "海拔估计必须以传感器值为锚点（±15%内）。若图中证据与传感器严重冲突，以视觉为准但需明确标注。\n"
            )
        else:
            sensor_data = ""

        image = resize_image_for_vlm(image)
        result = GeoCoTResult()

        for stage in GeoCoTStage:
            prompt_text = self._prompt.get_prompt(stage, result.stage_outputs,
                                                   sensor_data=sensor_data)

            if stage == GeoCoTStage.LOCAL:
                few_shot_prefix = self._prompt.build_few_shot_prefix(
                    prev_outputs=result.stage_outputs
                )
                if few_shot_prefix:
                    prompt_text = few_shot_prefix + "\n\n" + prompt_text

            output = self._model_fn(image, prompt_text, result.stage_outputs)
            result.stage_outputs[stage.value] = output.strip()

        result.reasoning_chain = result.full_reasoning
        result.final_prediction = extract_prediction(result.stage_outputs.get("local", ""))

        # Infer scene type from GeoCoT output for MoE reconciliation
        geocot_text = " ".join(result.stage_outputs.values())
        result.scene_type = _infer_scene_type_from_geocot(geocot_text)

        return result


# ---------------------------------------------------------------------------
# Qwen2-VL backend (tested, working)
# ---------------------------------------------------------------------------

def create_qwen2vl_model_fn(
    model,
    processor,
    max_new_tokens: int = 4096,
    temperature: float = 0.7,
    top_p: float = 0.9,
    enable_thinking: bool = True,
):
    """Create a model_fn for GeoCoTPipeline using Qwen2-VL.

    This is the backend that has been tested and verified working.
    """
    from qwen_vl_utils import process_vision_info

    def model_fn(image: Image.Image, prompt_text: str, prev_outputs: dict) -> str:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt_text},
        ]}]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        if not enable_thinking:
            text = text.replace('<think>\n', '')
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
            )

        result = processor.batch_decode(
            output_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
        )[0]

        # Free VRAM: generate() leaves large tensors on GPU
        del inputs, output_ids
        torch.cuda.empty_cache()

        return result.strip()

    return model_fn


def _detect_model_class(model_name: str):
    """Detect the correct model class from config (supports Qwen2-VL and Qwen3-VL)."""
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_name, local_files_only=True)
    if config.model_type == "qwen3_vl":
        from transformers import Qwen3VLForConditionalGeneration
        return Qwen3VLForConditionalGeneration
    elif config.model_type == "qwen2_vl":
        from transformers import Qwen2VLForConditionalGeneration
        return Qwen2VLForConditionalGeneration
    else:
        from transformers import Qwen2VLForConditionalGeneration
        print(f"  [WARN] Unknown model_type={config.model_type}, falling back to Qwen2VL")
        return Qwen2VLForConditionalGeneration


def load_qwen2vl(
    model_name: str = "Qwen/Qwen3-VL-2B-Instruct",
    max_pixels: int = MAX_IMAGE_PIXELS,
    load_in_4bit: bool = False,
    lora_path: str = None,
    merge_lora: bool = True,
    offload_folder: str = None,
    gpu_memory: str = None,
    cpu_memory: str = "16GB",
    max_new_tokens: int = 4096,
    temperature: float = 0.7,
    top_p: float = 0.9,
    enable_thinking: bool = True,
):
    """Load Qwen2-VL / Qwen3-VL model and processor, return (model, processor, model_fn).

    Auto-detects the model class from config.model_type. Default: Qwen3-VL-2B-Instruct
    for native 2D/3D spatial perception via Interleaved-MRoPE.

    Args:
        model_name: HuggingFace model ID (Qwen2-VL or Qwen3-VL)
        max_pixels: Max image pixels for processor
        load_in_4bit: Use 4-bit quantization (fits 7B in ~4.5GB VRAM)
        lora_path: Path to LoRA adapter weights. If set, loads adapter,
                   merges into base model, and returns the merged model.
                   Inference speed is identical to base model after merge.
        offload_folder: Directory for CPU offloading (slower fallback)
        gpu_memory: Max GPU memory to use, e.g. "6GB"
        cpu_memory: Max CPU memory for offloaded layers
    """
    from transformers import AutoProcessor, BitsAndBytesConfig

    ModelClass = _detect_model_class(model_name)
    print(f"  Model class: {ModelClass.__name__}")

    processor = AutoProcessor.from_pretrained(
        model_name,
        min_pixels=256 * 28 * 28,
        max_pixels=max_pixels,
        local_files_only=True,
    )

    load_kwargs = {
        "device_map": "auto",
        "local_files_only": True,
    }

    if load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        load_kwargs["quantization_config"] = quant_config
        print(f"  4-bit quantization: enabled (nf4, double-quant)")
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16

    # For CPU offloading: limit GPU usage
    if gpu_memory or offload_folder:
        max_memory = {}
        if gpu_memory:
            max_memory[0] = gpu_memory
        if cpu_memory:
            max_memory["cpu"] = cpu_memory
        load_kwargs["max_memory"] = max_memory

    if offload_folder:
        os.makedirs(offload_folder, exist_ok=True)
        load_kwargs["offload_folder"] = offload_folder

    print(f"  Mode: {'4-bit' if load_in_4bit else 'bfloat16'}, GPU limit: {gpu_memory or 'auto'}")
    model = ModelClass.from_pretrained(model_name, **load_kwargs)
    model.eval()

    # ── Load LoRA adapter ──────────────────────────────────────────
    if lora_path:
        from peft import PeftModel
        print(f"  Loading LoRA adapter from: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
        if merge_lora:
            print(f"  Merging LoRA weights into base model...")
            model = model.merge_and_unload()
            print(f"  LoRA merged. Inference speed = base model speed.")
        else:
            print(f"  LoRA kept as adapter (no merge). 4-bit quantization preserved.")

    model_fn = create_qwen2vl_model_fn(model, processor, max_new_tokens=max_new_tokens,
                                       temperature=temperature, top_p=top_p,
                                       enable_thinking=enable_thinking)
    return model, processor, model_fn


# ---------------------------------------------------------------------------
# Legacy HuggingFace backend (for LLaMA-3.2-Vision, untested on this GPU)
# ---------------------------------------------------------------------------

def create_hf_model_fn(model, processor, device, temperature=0.7, top_p=0.9, max_new_tokens=2048):
    """Create a model_fn for MllamaForConditionalGeneration (LLaMA-3.2-Vision)."""
    def model_fn(image: Image.Image, prompt_text: str, prev_outputs: dict) -> str:
        conversation = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ]}
        ]
        for stage_key, stage_output in prev_outputs.items():
            conversation.append({"role": "assistant", "content": stage_output})

        prompt = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        inputs = processor(
            image, prompt, text_kwargs={"add_special_tokens": False}, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            output = model.generate(
                **inputs, temperature=temperature, top_p=top_p, max_new_tokens=max_new_tokens
            )

        decoded = processor.decode(output[0])
        response = decoded[len(prompt):]
        for tok in ["<|eot_id|>", "<|im_end|>", "<|end_of_text|>"]:
            response = response.replace(tok, "")
        return response.strip()

    return model_fn


# ---------------------------------------------------------------------------
# GeoVLM backend — edge-optimized geographic element extraction
# ---------------------------------------------------------------------------

def load_geovlm(
    checkpoint_path: str = None,
    hidden_dim: int = 512,
    vit_model_name: str = "google/vit-base-patch16-224",
    device: str = "cpu",
):
    """Load GeoVLM model for element extraction.

    Args:
        checkpoint_path: Path to trained .pt checkpoint (None = random init)
        hidden_dim: Hidden dimension (must match checkpoint)
        vit_model_name: HuggingFace ViT model identifier
        device: 'cpu', 'cuda', or 'npu' (OpenVINO)

    Returns:
        (model, model_fn) where model_fn(image, sensor_dict) -> elements dict
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from geovlm.model import GeoVLM, GeoVLMConfig

    config = GeoVLMConfig(
        vit_model_name=vit_model_name,
        hidden_dim=hidden_dim,
    )
    model = GeoVLM(config)

    if checkpoint_path and Path(checkpoint_path).exists():
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state["model"] if "model" in state else state)
        print(f"  GeoVLM checkpoint loaded from: {checkpoint_path}")
    elif checkpoint_path:
        print(f"  [WARN] GeoVLM checkpoint not found: {checkpoint_path}, using random weights")

    model.to(device)
    model.eval()

    def model_fn(image, sensor_elevation_m=None, sensor_temperature_c=None,
                 sensor_humidity_pct=None):
        """Extract elements from a single image. Returns elements dict."""
        return model.predict(
            image,
            sensor_elevation_m=sensor_elevation_m,
            sensor_temperature_c=sensor_temperature_c,
            sensor_humidity_pct=sensor_humidity_pct,
        )

    return model, model_fn


class GeoVLMPipeline:
    """Thin wrapper making GeoVLM usable as drop-in replacement for GeoCoT's 3-stage VLM.

    Usage in test scripts — replaces the 3-stage VLM pipeline:
        model, model_fn = load_geovlm(checkpoint_path="geovlm_final.pt")
        pipe = GeoVLMPipeline(model_fn)
        elements = pipe.extract(img, elev, temp, humid)
        # elements dict passes directly to fuse_elements_v3()
    """

    def __init__(self, model_fn):
        self._model_fn = model_fn

    def extract(self, image: "Image.Image",
                sensor_elevation_m: Optional[float] = None,
                sensor_temperature_c: Optional[float] = None,
                sensor_humidity_pct: Optional[float] = None) -> dict:
        """Run element extraction on a single image. Returns elements dict ready for fusion."""
        return self._model_fn(
            image,
            sensor_elevation_m=sensor_elevation_m,
            sensor_temperature_c=sensor_temperature_c,
            sensor_humidity_pct=sensor_humidity_pct,
        )

    def batch_extract(self, images: list, sensor_values: list) -> list[dict]:
        """Extract elements from a batch of images.

        Args:
            images: list of PIL Images
            sensor_values: list of (elev, temp, humid) tuples
        Returns:
            list of elements dicts
        """
        results = []
        for img, (elev, temp, humid) in zip(images, sensor_values):
            results.append(self._model_fn(
                img,
                sensor_elevation_m=elev,
                sensor_temperature_c=temp,
                sensor_humidity_pct=humid,
            ))
        return results
