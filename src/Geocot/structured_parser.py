"""GeoStructuredParser: extract structured geographic entities from GeoCoT text.

Parses free-text GeoCoT stage outputs into GeoEntities dataclass with:
  terrain_types, climate_zones, vegetation_types, architectural_styles,
  cultural_markers, named_locations, altitude_indicators, and confidence.

Used by the cascade pipeline to feed spatial KB search with structured signals.
"""

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GeoEntities:
    """Structured geographic entities extracted from GeoCoT reasoning text."""
    terrain_types: list[str] = field(default_factory=list)
    climate_zones: list[str] = field(default_factory=list)
    vegetation_types: list[str] = field(default_factory=list)
    architectural_styles: list[str] = field(default_factory=list)
    cultural_markers: list[str] = field(default_factory=list)
    named_locations: list[str] = field(default_factory=list)
    altitude_indicators: list[str] = field(default_factory=list)
    scene_type_hint: Optional[int] = None  # 0-3 SceneType guess
    confidence: float = 0.0

    @property
    def all_keywords(self) -> list[str]:
        """All non-location keywords for spatial KB matching."""
        return (self.terrain_types + self.climate_zones + self.vegetation_types +
                self.architectural_styles + self.cultural_markers + self.altitude_indicators)

    @property
    def is_empty(self) -> bool:
        return not any([
            self.terrain_types, self.climate_zones, self.vegetation_types,
            self.architectural_styles, self.cultural_markers, self.named_locations,
        ])


# ── Keyword lexicons ──────────────────────────────────────────────────────

_TERRAIN_KEYWORDS = {
    "花岗岩峰林": ["granite peak", "花岗岩", "granite", "峰林", "peak forest"],
    "石英砂岩峰林": ["quartz sandstone", "石英砂岩", "sandstone pillar", "柱状峰林"],
    "石灰岩喀斯特": ["limestone", "karst", "石灰岩", "喀斯特", "tower karst", "溶蚀"],
    "丹霞地貌": ["danxia", "丹霞", "red sandstone", "红色砂岩"],
    "高山峡谷": ["gorge", "canyon", "峡谷", "valley", "深切"],
    "现代冰川": ["glacier", "冰川", "ice", "冰舌"],
    "雪山群": ["snow mountain", "雪山", "snow peak", "雪峰"],
    "单峰雪山": ["pyramidal peak", "锥形", "isolated peak"],
    "黄土高原": ["loess", "黄土", "plateau"],
    "沙漠": ["desert", "沙漠", "sand dune", "沙丘"],
    "草原": ["grassland", "草原", "steppe", "pasture"],
    "高山草甸": ["alpine meadow", "高山草甸", "meadow"],
    "干热河谷": ["dry-hot valley", "干热河谷", "arid valley"],
    "钙华地貌": ["travertine", "钙华", "tufa"],
    "丹霞彩丘": ["rainbow mountain", "彩丘", "colorful"],
    "海岸线": ["coast", "coastal", "海岸", "beach", "海滩"],
    "湖泊": ["lake", "湖泊", "湖"],
    "平原": ["plain", "平原", "flatland"],
    "丘陵": ["hill", "丘陵", "rolling"],
}

_CLIMATE_KEYWORDS = {
    "亚热带湿润": ["subtropical", "亚热带", "humid subtropical"],
    "温带季风": ["temperate", "温带", "monsoon"],
    "高原寒带": ["alpine", "plateau climate", "高原气候", "cold plateau"],
    "热带": ["tropical", "热带"],
    "干旱/半干旱": ["arid", "干旱", "dry", "semi-arid"],
    "高原山地": ["mountain climate", "山地气候"],
    "大陆性": ["continental", "大陆性"],
}

_VEGETATION_KEYWORDS = {
    "黄山松": ["huangshan pine", "黄山松", "pinus hwangshanensis"],
    "马尾松": ["masson pine", "马尾松", "pinus massoniana"],
    "冷杉": ["fir", "冷杉", "abies"],
    "云杉": ["spruce", "云杉", "picea"],
    "红杉": ["larch", "红杉", "larix", "落叶松"],
    "高山杜鹃": ["rhododendron", "杜鹃", "azalea"],
    "榕树": ["banyan", "榕树", "ficus"],
    "法国梧桐": ["plane tree", "法国梧桐", "platanus", "悬铃木"],
    "银杏": ["ginkgo", "银杏"],
    "香樟": ["camphor", "香樟", "cinnamomum"],
    "竹林": ["bamboo", "竹", "竹林"],
    "针叶林": ["conifer", "针叶林", "coniferous"],
    "阔叶林": ["broadleaf", "阔叶林", "broad-leaved"],
    "常绿阔叶林": ["evergreen broadleaf", "常绿阔叶林"],
    "针阔混交": ["mixed forest", "针阔混交", "mixed coniferous"],
    "高山灌丛": ["alpine shrub", "高山灌丛", "scrub"],
    "流石滩": ["scree", "流石滩", "talus"],
    "苔藓": ["moss", "苔藓", "lichen"],
    "仙人掌": ["cactus", "仙人掌"],
    "蕨类": ["fern", "蕨类"],
}

_ARCHITECTURE_KEYWORDS = {
    "徽派建筑": ["hui-style", "徽派", "白墙黛瓦", "马头墙", "horse-head wall"],
    "藏族碉房": ["tibetan house", "藏族", "tibetan", "碉房", "stone house"],
    "纳西族建筑": ["naxi", "纳西", "三坊一照壁", "four合五天井"],
    "吊脚楼": ["stilt house", "吊脚楼", "stilted", "干栏式"],
    "土楼": ["tulou", "土楼", "earthen building", "circular"],
    "骑楼": ["arcade", "骑楼", "qilou", "colonnade"],
    "四合院": ["courtyard", "四合院", "siheyuan"],
    "现代高层": ["modern high-rise", "高层", "skyscraper", "modern building"],
    "西式建筑": ["western architecture", "西式", "colonial", "租界"],
    "石库门": ["shikumen", "石库门"],
    "江南水乡": ["water town", "水乡", "canal town", "江南"],
    "摩崖石刻": ["cliff carving", "摩崖石刻", "石刻"],
    "白塔": ["stupa", "白塔", "pagoda", "佛塔"],
    "经幡": ["prayer flag", "经幡", "prayer banner"],
}

_CULTURAL_KEYWORDS = {
    "藏传佛教": ["tibetan buddhism", "藏传佛教", "buddhist", "lama"],
    "道教": ["taoist", "道教", "taoism", "道观"],
    "纳西族": ["naxi", "纳西", "纳西族"],
    "壮族": ["zhuang", "壮族"],
    "土家族": ["tujia", "土家族"],
    "维吾尔": ["uyghur", "维吾尔"],
    "蒙古族": ["mongolian", "蒙古", "mongol"],
    "红色革命": ["communist", "revolutionary", "革命", "红色"],
    "茶文化": ["tea culture", "茶园", "tea plantation", "茶馆"],
    "竹筏": ["bamboo raft", "竹筏", "rafting"],
    "缆车": ["cable car", "缆车", "索道", "gondola"],
    "玻璃栈道": ["glass bridge", "玻璃栈道", "glass walkway"],
    "摩崖": ["cliff inscription", "摩崖", "stone inscription"],
}

_ALTITUDE_PATTERNS = [
    (re.compile(r"海拔\s*(\d{3,4})\s*[米mM]"), lambda m: f"海拔{m.group(1)}m"),
    (re.compile(r"(\d{3,4})\s*[米mM]\s*[以左海高拔]"), lambda m: f"海拔{m.group(1)}m"),
    (re.compile(r"elevation\s*(\d{3,4})\s*m", re.IGNORECASE), lambda m: f"海拔{m.group(1)}m"),
    (re.compile(r"altitude\s*(\d{3,4})\s*m", re.IGNORECASE), lambda m: f"海拔{m.group(1)}m"),
    (re.compile(r"雪线[以之]上"), lambda _: "雪线以上"),
    (re.compile(r"树线[以之]上"), lambda _: "树线以上"),
    (re.compile(r"高海拔"), lambda _: "高海拔"),
    (re.compile(r"低海拔"), lambda _: "低海拔"),
]

# Direct province name patterns
_PROVINCE_PATTERNS = [
    "安徽", "云南", "四川", "广西", "湖南", "福建", "江西", "陕西",
    "浙江", "湖北", "山东", "西藏", "新疆", "内蒙古", "甘肃", "广东",
    "贵州", "青海", "宁夏", "海南", "河南", "河北", "山西", "辽宁",
    "吉林", "黑龙江", "江苏", "台湾",
]

# Region patterns (broader than province)
_REGION_PATTERNS = [
    "华东", "华南", "华北", "华中", "西南", "西北", "东北",
    "长三角", "珠三角", "京津冀", "成渝",
    "横断山", "秦岭", "太行", "大别山",
]


def _collect_entities(text: str, lexicon: dict[str, list[str]]) -> list[str]:
    """Match lexicon entries against text. Returns list of matched category names."""
    found = []
    text_lower = text.lower()
    for category, keywords in lexicon.items():
        for kw in keywords:
            if kw.lower() in text_lower:
                found.append(category)
                break  # one match per category
    return found


def _extract_named_locations(text: str) -> list[str]:
    """Extract named locations from text using known landmark/trail/city databases.

    Imported lazily to avoid circular imports.
    """
    from .Geocot import _LANDMARK_TO_LOCATION, _init_city_db, _TRAIL_COORD_DB
    _init_city_db()

    found = set()
    text_lower = text.lower()

    # Check landmarks
    for landmark_name in _LANDMARK_TO_LOCATION:
        if len(landmark_name) >= 3 and landmark_name in text_lower:
            found.add(landmark_name.title())

    # Check trails
    for trail_name in _TRAIL_COORD_DB:
        if len(trail_name) >= 3 and trail_name in text_lower:
            found.add(trail_name.title())

    # Check GEO_KNOWLEDGE name_cn
    from .geoknowledge import GEO_KNOWLEDGE
    for name, entry in GEO_KNOWLEDGE.items():
        cn = entry.get("name_cn", "")
        if cn and len(cn) >= 2 and cn in text:
            found.add(cn)
        # Also check English key name
        if len(name) >= 4 and name.replace("_", " ") in text_lower:
            found.add(cn or name.replace("_", " ").title())

    # Check provinces
    for prov in _PROVINCE_PATTERNS:
        if prov in text:
            found.add(prov)

    # Check regions
    for region in _REGION_PATTERNS:
        if region in text:
            found.add(region)

    return sorted(found)


def _extract_altitude(text: str) -> list[str]:
    """Extract altitude-related indicators."""
    found = []
    for pattern, formatter in _ALTITUDE_PATTERNS:
        m = pattern.search(text)
        if m:
            found.append(formatter(m))
    return found


def _infer_scene(text: str) -> Optional[int]:
    """Infer SceneType from text content. 0=urban, 1=karst/granite, 2=alpine, 3=other."""
    text_lower = text.lower()

    urban_kw = ["城市", "建筑", "街道", "沥青", "路面", "urban", "city",
                "road", "building", "traffic", "shanghai", "beijing",
                "高层", "skyscraper", "street"]
    karst_kw = ["峰林", "花岗岩", "喀斯特", "石灰岩", "石英砂岩", "丹霞",
                "karst", "granite", "peak", "guilin", "huangshan",
                "zhangjiajie", "sandstone", "武陵源", "桂林", "黄山"]
    alpine_kw = ["雪山", "冰川", "高山", "alpine", "snow",
                 "mountain", "glacier", "plateau", "tibet",
                 "四姑娘", "贡嘎", "梅里", "雪峰", "海拔", "冰"]
    other_kw = ["峡谷", "古镇", "草原", "沙漠", "湖泊", "gorge",
                "old town", "grassland", "desert", "lake", "canyon",
                "lijiang", "tiger leaping", "dali", "古城"]

    scores = {
        0: sum(1 for kw in urban_kw if kw in text_lower),
        1: sum(1 for kw in karst_kw if kw in text_lower),
        2: sum(1 for kw in alpine_kw if kw in text_lower),
        3: sum(1 for kw in other_kw if kw in text_lower),
    }
    best = max(scores, key=scores.get)
    return int(best) if scores[best] > 0 else None


def parse_geocot_output(stage_outputs: dict) -> GeoEntities:
    """Parse GeoCoT stage outputs into structured geographic entities.

    Args:
        stage_outputs: dict with keys "macro", "regional", "local" → text

    Returns:
        GeoEntities with all extracted fields populated.
    """
    # Combine all stage outputs, with higher weight to later stages
    full_text_parts = []
    for stage in ["macro", "regional", "local"]:
        if stage in stage_outputs:
            text = stage_outputs[stage]
            # Repeat local/regional text for higher weight
            if stage == "local":
                full_text_parts.extend([text] * 3)
            elif stage == "regional":
                full_text_parts.extend([text] * 2)
            else:
                full_text_parts.append(text)

    combined = "\n".join(full_text_parts)

    entities = GeoEntities()
    entities.terrain_types = _collect_entities(combined, _TERRAIN_KEYWORDS)
    entities.climate_zones = _collect_entities(combined, _CLIMATE_KEYWORDS)
    entities.vegetation_types = _collect_entities(combined, _VEGETATION_KEYWORDS)
    entities.architectural_styles = _collect_entities(combined, _ARCHITECTURE_KEYWORDS)
    entities.cultural_markers = _collect_entities(combined, _CULTURAL_KEYWORDS)
    entities.named_locations = _extract_named_locations(combined)
    entities.altitude_indicators = _extract_altitude(combined)
    entities.scene_type_hint = _infer_scene(combined)

    # Compute confidence: more entities → higher confidence
    n_entities = (len(entities.terrain_types) + len(entities.climate_zones) +
                  len(entities.vegetation_types) + len(entities.architectural_styles) +
                  len(entities.cultural_markers) + len(entities.named_locations))
    entities.confidence = min(1.0, n_entities / 12.0)

    return entities


def parse_single_stage(text: str, stage_name: str = "") -> GeoEntities:
    """Parse a single GeoCoT stage output (convenience wrapper)."""
    outputs = {stage_name: text} if stage_name else {"local": text}
    return parse_geocot_output(outputs)
