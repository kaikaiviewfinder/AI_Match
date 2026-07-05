"""Spatial KB Engine: map structured geographic entities to spatial regions.

Converts GeoEntities (from structured_parser.py) into candidate spatial
ConstraintBox regions by matching against:
  1. GEO_KNOWLEDGE — 30+ entries with coordinates, geology, vegetation, etc.
  2. _TRAIL_COORD_DB — 89 hiking trails / scenic areas
  3. _CITY_COORD_DB — 139 cities in China + world
  4. _LANDMARK_TO_LOCATION — known landmarks → city/country

Output: list of ScoredRegion (center lat/lng, radius_km, confidence, source).
"""

import math
from dataclasses import dataclass, field
from typing import Optional

from .structured_parser import GeoEntities

# Will be imported lazily to avoid circular imports
_KB_IMPORTS_DONE = False
_GEO_KNOWLEDGE = {}
_TRAIL_COORD_DB = {}
_CITY_COORD_DB = {}
_LANDMARK_TO_LOCATION = {}
_PROVINCE_MAP = {}


def _ensure_imports():
    global _KB_IMPORTS_DONE, _GEO_KNOWLEDGE, _TRAIL_COORD_DB
    global _CITY_COORD_DB, _LANDMARK_TO_LOCATION, _PROVINCE_MAP
    if _KB_IMPORTS_DONE:
        return
    from .geoknowledge import GEO_KNOWLEDGE
    from .Geocot import _TRAIL_COORD_DB, _CITY_COORD_DB, _LANDMARK_TO_LOCATION
    from .Geocot import _PROVINCE_MAP, _init_city_db
    _init_city_db()
    _GEO_KNOWLEDGE = GEO_KNOWLEDGE
    _TRAIL_COORD_DB = _TRAIL_COORD_DB
    _CITY_COORD_DB = _CITY_COORD_DB
    _LANDMARK_TO_LOCATION = _LANDMARK_TO_LOCATION
    _PROVINCE_MAP = _PROVINCE_MAP
    _KB_IMPORTS_DONE = True


@dataclass
class ScoredRegion:
    """A candidate spatial region with confidence score."""
    center_lat: float
    center_lng: float
    radius_km: float          # 1-sigma radius
    confidence: float          # 0.0-1.0
    source: str                # KB entry name, trail name, city name
    source_type: str           # "knowledge_entry" | "trail" | "city" | "landmark"
    match_reason: str = ""     # which entity triggered this match

    @property
    def lat_min(self) -> float:
        deg = self.radius_km / 111.0
        return self.center_lat - deg

    @property
    def lat_max(self) -> float:
        deg = self.radius_km / 111.0
        return self.center_lat + deg

    @property
    def lng_min(self) -> float:
        deg = self.radius_km / (111.0 * math.cos(math.radians(self.center_lat)))
        return self.center_lng - deg

    @property
    def lng_max(self) -> float:
        deg = self.radius_km / (111.0 * math.cos(math.radians(self.center_lat)))
        return self.center_lng + deg


def _haversine_km(lat1, lng1, lat2, lng2):
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlng / 2) ** 2)
    return 6371.0 * 2 * math.asin(math.sqrt(a))


# ── Named location → coordinates lookup ───────────────────────────────────

def _lookup_location_coords(name: str) -> list[tuple[float, float, str, str]]:
    """Look up coordinates for a named location. Returns list of (lat, lng, source, source_type)."""
    _ensure_imports()
    results = []
    name_lower = name.lower().strip()

    # 1. GEO_KNOWLEDGE
    for key, entry in _GEO_KNOWLEDGE.items():
        cn = entry.get("name_cn", "")
        coords = entry.get("coordinates")
        if not coords:
            continue
        if name_lower == key.lower() or name == cn or name_lower in key.lower() or cn in name:
            results.append((coords[0], coords[1], key, "knowledge_entry"))

    # 2. Trail DB
    for trail_name, (tlat, tlng, _) in _TRAIL_COORD_DB.items():
        if name_lower == trail_name.lower() or name_lower in trail_name.lower() or trail_name.lower() in name_lower:
            results.append((tlat, tlng, trail_name, "trail"))

    # 3. City DB
    for (city_key, country_key), (clat, clng) in _CITY_COORD_DB.items():
        if name_lower == city_key.lower() or name_lower in city_key:
            results.append((clat, clng, city_key, "city"))

    # 4. Landmark → city → coords
    for landmark, (city_name, country_name) in _LANDMARK_TO_LOCATION.items():
        if name_lower == landmark.lower() or name_lower in landmark:
            # Lookup city coords
            for (ckey, _), (clat, clng) in _CITY_COORD_DB.items():
                if ckey.lower() == city_name.lower():
                    results.append((clat, clng, landmark, "landmark"))
                    break

    # 5. Province → capital coords
    for abbr, (prov_name, capital, clat, clng) in _PROVINCE_MAP.items():
        if name == prov_name or name == capital:
            results.append((clat, clng, prov_name, "city"))
        # Also check Chinese province name
        if name in (abbr, capital):
            results.append((clat, clng, capital, "city"))

    return results


# ── Terrain/vegetation → KB entry matching ────────────────────────────────

def _match_entities_to_kb(entities: GeoEntities) -> list[tuple[dict, float, str]]:
    """Match terrain/vegetation/cultural entities against GEO_KNOWLEDGE entries.

    Returns list of (entry_dict, score, match_reason).
    """
    _ensure_imports()
    results = []

    # Build a search text from all non-location entities
    search_terms = (entities.terrain_types + entities.vegetation_types +
                    entities.climate_zones + entities.architectural_styles +
                    entities.cultural_markers)
    search_text = " ".join(search_terms).lower()

    if not search_text.strip():
        return results

    for key, entry in _GEO_KNOWLEDGE.items():
        if entry.get("coordinates") is None:
            continue  # skip entries without coordinates (e.g. urban general hints)

        score = 0
        reasons = []

        # Match against geology field
        geology = entry.get("geology", "")
        for terrain in entities.terrain_types:
            # Check if terrain keywords appear in geology description
            for kw in _get_terrain_subkeywords(terrain):
                if kw.lower() in geology.lower():
                    score += 3
                    reasons.append(f"地质匹配: {terrain}")
                    break

        # Match against vegetation field
        vegetation = entry.get("vegetation", "")
        for veg in entities.vegetation_types:
            for kw in _get_vegetation_subkeywords(veg):
                if kw.lower() in vegetation.lower():
                    score += 3
                    reasons.append(f"植被匹配: {veg}")
                    break

        # Match against cultural_markers field
        cultural = entry.get("cultural_markers", "")
        for cm in entities.cultural_markers:
            for kw in _get_cultural_subkeywords(cm):
                if kw.lower() in cultural.lower():
                    score += 2
                    reasons.append(f"文化匹配: {cm}")
                    break

        # Bonus: name_cn appears in search text
        name_cn = entry.get("name_cn", "")
        if name_cn and name_cn in " ".join(entities.named_locations):
            score += 5
            reasons.append(f"地名匹配: {name_cn}")

        # province match
        province = entry.get("province", "")
        if province and (province.lower() in search_text or
                         any(province in loc for loc in entities.named_locations)):
            score += 4
            reasons.append(f"省份匹配: {province}")

        if score > 0:
            results.append((entry, score, "; ".join(reasons)))

    # Normalize scores to 0-1
    if results:
        max_score = max(s for _, s, _ in results)
        results = [(e, min(1.0, s / max(10, max_score)), r) for e, s, r in results]

    return results


def _get_terrain_subkeywords(terrain_type: str) -> list[str]:
    """Get sub-keywords for a terrain type to match against KB fields."""
    mapping = {
        "花岗岩峰林": ["granite", "花岗岩"],
        "石英砂岩峰林": ["quartz sandstone", "石英砂岩", "sandstone"],
        "石灰岩喀斯特": ["limestone", "karst", "石灰岩", "喀斯特"],
        "丹霞地貌": ["danxia", "丹霞", "red sandstone"],
        "高山峡谷": ["gorge", "canyon", "峡谷"],
        "现代冰川": ["glacier", "冰川"],
        "雪山群": ["snow mountain", "雪山", "snow"],
        "单峰雪山": ["pyramidal", "锥形"],
        "黄土高原": ["loess", "黄土"],
        "沙漠": ["desert", "沙漠"],
        "草原": ["grassland", "草原", "steppe"],
        "高山草甸": ["alpine meadow", "meadow"],
        "干热河谷": ["dry-hot", "arid valley", "干热"],
        "钙华地貌": ["travertine", "钙华", "tufa"],
        "丹霞彩丘": ["rainbow", "colorful"],
    }
    return mapping.get(terrain_type, [terrain_type])


def _get_vegetation_subkeywords(veg_type: str) -> list[str]:
    mapping = {
        "黄山松": ["huangshan pine", "pinus hwangshanensis", "黄山松"],
        "马尾松": ["masson pine", "pinus massoniana", "马尾松"],
        "冷杉": ["fir", "abies", "冷杉"],
        "云杉": ["spruce", "picea", "云杉"],
        "红杉": ["larch", "larix", "落叶松", "红杉"],
        "高山杜鹃": ["rhododendron", "杜鹃"],
        "榕树": ["banyan", "榕树", "ficus"],
        "法国梧桐": ["plane tree", "platanus", "法国梧桐"],
        "银杏": ["ginkgo", "银杏"],
        "香樟": ["camphor", "樟"],
        "竹林": ["bamboo", "竹"],
        "针叶林": ["conifer", "coniferous", "针叶"],
        "阔叶林": ["broadleaf", "broad-leaved", "阔叶"],
        "常绿阔叶林": ["evergreen broadleaf", "常绿阔叶"],
        "针阔混交": ["mixed", "混交"],
        "仙人掌": ["cactus", "仙人掌"],
    }
    return mapping.get(veg_type, [veg_type])


def _get_cultural_subkeywords(cultural_type: str) -> list[str]:
    mapping = {
        "徽派建筑": ["徽派", "白墙黛瓦", "马头墙"],
        "藏族碉房": ["tibetan", "藏族"],
        "纳西族建筑": ["naxi", "纳西"],
        "吊脚楼": ["stilt", "吊脚楼"],
        "土楼": ["tulou", "土楼"],
        "藏传佛教": ["tibetan buddhism", "buddhist", "lama", "藏传"],
        "道教": ["taoist", "道观", "道教"],
        "壮族": ["zhuang", "壮族"],
        "土家族": ["tujia", "土家族"],
        "竹筏": ["bamboo raft", "竹筏"],
        "摩崖": ["cliff carving", "摩崖", "石刻"],
        "经幡": ["prayer flag", "经幡"],
        "白塔": ["stupa", "白塔"],
    }
    return mapping.get(cultural_type, [cultural_type])


# ── Main entry point ──────────────────────────────────────────────────────

def entities_to_spatial_regions(
    entities: GeoEntities,
    top_k: int = 5,
    max_radius_km: float = 500.0,
    min_radius_km: float = 15.0,
) -> list[ScoredRegion]:
    """Convert GeoEntities into candidate spatial regions.

    Strategy:
      1. Named locations → direct coordinate lookup (highest confidence)
      2. Terrain/vegetation/cultural entities → KB entry matching
      3. Each match generates a ScoredRegion with radius ∝ 1/confidence

    Args:
        entities: parsed GeoEntities from structured_parser
        top_k: max regions to return
        max_radius_km: widest radius for low-confidence matches
        min_radius_km: narrowest radius for high-confidence matches

    Returns:
        list of ScoredRegion sorted by confidence descending
    """
    _ensure_imports()
    regions: list[ScoredRegion] = []

    # Strategy 1: Named location lookup
    for loc_name in entities.named_locations:
        coords_list = _lookup_location_coords(loc_name)
        for lat, lng, source, source_type in coords_list:
            # Higher confidence for exact trail/KB matches
            if source_type in ("trail", "knowledge_entry"):
                confidence = 0.9
                radius = 20.0
            elif source_type == "city":
                confidence = 0.6
                radius = 100.0
            else:
                confidence = 0.4
                radius = 200.0

            regions.append(ScoredRegion(
                center_lat=lat, center_lng=lng,
                radius_km=radius,
                confidence=confidence,
                source=source,
                source_type=source_type,
                match_reason=f"地名匹配: {loc_name}",
            ))

    # Strategy 2: KB entry matching from terrain/vegetation/cultural entities
    kb_matches = _match_entities_to_kb(entities)
    for entry, score, reason in kb_matches:
        coords = entry.get("coordinates")
        if not coords:
            continue
        # Radius scales inversely with confidence
        radius = max(min_radius_km, max_radius_km * (1.0 - score))
        regions.append(ScoredRegion(
            center_lat=coords[0], center_lng=coords[1],
            radius_km=radius,
            confidence=score,
            source=entry.get("name_cn", "unknown"),
            source_type="knowledge_entry",
            match_reason=reason,
        ))

    # Deduplicate by proximity: merge regions within 30km of each other
    regions = _merge_nearby_regions(regions, merge_radius_km=30.0)

    # Sort by confidence descending
    regions.sort(key=lambda r: r.confidence, reverse=True)

    return regions[:top_k]


def _merge_nearby_regions(
    regions: list[ScoredRegion],
    merge_radius_km: float = 30.0,
) -> list[ScoredRegion]:
    """Merge regions whose centers are within merge_radius_km of each other.

    When merging: average centers, take min radius, max confidence.
    """
    if len(regions) <= 1:
        return regions

    merged = []
    used = [False] * len(regions)

    for i, r1 in enumerate(regions):
        if used[i]:
            continue
        group_lats = [r1.center_lat]
        group_lngs = [r1.center_lng]
        group_radii = [r1.radius_km]
        group_conf = [r1.confidence]
        group_sources = [r1.source]
        group_reasons = [r1.match_reason]
        best_source_type = r1.source_type

        for j, r2 in enumerate(regions):
            if i == j or used[j]:
                continue
            dist = _haversine_km(r1.center_lat, r1.center_lng,
                                r2.center_lat, r2.center_lng)
            if dist < merge_radius_km:
                used[j] = True
                group_lats.append(r2.center_lat)
                group_lngs.append(r2.center_lng)
                group_radii.append(r2.radius_km)
                group_conf.append(r2.confidence)
                group_sources.append(r2.source)
                group_reasons.append(r2.match_reason)
                if r2.source_type in ("trail", "knowledge_entry"):
                    best_source_type = r2.source_type

        used[i] = True
        avg_lat = sum(group_lats) / len(group_lats)
        avg_lng = sum(group_lngs) / len(group_lngs)
        # Narrower radius (intersection-like) when multiple sources agree
        min_radius = min(group_radii)
        # Confidence boosted by multi-source agreement
        boosted_conf = min(1.0, max(group_conf) + 0.1 * (len(group_conf) - 1))

        merged.append(ScoredRegion(
            center_lat=avg_lat, center_lng=avg_lng,
            radius_km=min_radius,
            confidence=boosted_conf,
            source=" + ".join(group_sources[:3]),
            source_type=best_source_type,
            match_reason="; ".join(group_reasons),
        ))

    return merged


def region_to_constraint_box(region: ScoredRegion) -> dict:
    """Convert ScoredRegion to a constraint box dict usable by spatial_constraint module.

    Returns dict with keys: lat_min, lat_max, lng_min, lng_max,
      center_lat, center_lng, radius_km, confidence, source
    """
    deg_lat = region.radius_km / 111.0
    deg_lng = region.radius_km / (111.0 * math.cos(math.radians(region.center_lat)))
    return {
        "lat_min": region.center_lat - deg_lat,
        "lat_max": region.center_lat + deg_lat,
        "lng_min": region.center_lng - deg_lng,
        "lng_max": region.center_lng + deg_lng,
        "center_lat": region.center_lat,
        "center_lng": region.center_lng,
        "radius_km": region.radius_km,
        "confidence": region.confidence,
        "source": region.source,
        "source_type": region.source_type,
        "match_reason": region.match_reason,
    }
