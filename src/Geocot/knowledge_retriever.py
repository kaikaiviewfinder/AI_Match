"""Knowledge retriever: inject China geo knowledge into GeoCoT prompts.

Two retrieval strategies:
  1. Proximity-based: given (lat, lng), find the closest knowledge entry
  2. Keyword-based: given GeoCoT output text, find entries matching region names

The retrieved knowledge is formatted as a text block appended to the prompt.
"""

import math
from typing import Optional

from .geoknowledge import GEO_KNOWLEDGE, PROVINCE_HINTS


def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Approximate distance in km between two lat/lng points."""
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlng / 2) ** 2)
    return 6371.0 * 2 * math.asin(math.sqrt(a))


def retrieve_by_proximity(
    lat: float,
    lng: float,
    max_distance_km: float = 300.0,
    top_k: int = 3,
) -> list[dict]:
    """Retrieve knowledge entries closest to the given coordinates.

    Args:
        lat, lng: query coordinates
        max_distance_km: only return entries within this distance
        top_k: max number of entries to return

    Returns:
        list of dicts with keys: name, name_cn, province, distance_km, geology,
        vegetation, cultural_markers, distinguish_from, common_mistakes, urban_clues
    """
    results = []
    for name, entry in GEO_KNOWLEDGE.items():
        coords = entry.get("coordinates")
        if coords is None:
            continue  # skip entries without coordinates (e.g. urban general hints)
        elat, elng = coords
        dist = _haversine_km(lat, lng, elat, elng)
        if dist <= max_distance_km:
            results.append({
                "name": name,
                "distance_km": dist,
                **{k: v for k, v in entry.items() if k != "coordinates"},
            })
    results.sort(key=lambda x: x["distance_km"])
    return results[:top_k]


def retrieve_by_keywords(text: str, top_k: int = 3) -> list[dict]:
    """Retrieve knowledge entries whose name or province appears in the text.

    Args:
        text: GeoCoT output text (e.g., from Macro or Regional stage)
        top_k: max number of entries to return

    Returns:
        list of dicts with full knowledge entry data
    """
    text_lower = text.lower()
    results = []
    for name, entry in GEO_KNOWLEDGE.items():
        score = 0
        if name in text_lower:
            score += 10
        cn = entry.get("name_cn")
        if cn and cn in text:
            score += 10
        province = entry.get("province")
        if province and province in text:
            score += 5
        # Check for partial name matches
        parts = name.split("_")
        for part in parts:
            if len(part) > 3 and part in text_lower:
                score += 3
        if score > 0:
            results.append({
                "name": name,
                "match_score": score,
                **{k: v for k, v in entry.items() if k != "coordinates"},
            })
    results.sort(key=lambda x: x["match_score"], reverse=True)
    return results[:top_k]


def retrieve_all_urban_hints() -> str:
    """Return all urban regional differentiation hints."""
    entries = [
        e for name, e in GEO_KNOWLEDGE.items()
        if e.get("scene_type") == 0 and "urban_clues" in e
    ]
    lines = []
    for e in entries:
        if "urban_clues" in e:
            lines.append(f"  {e['name_cn']}: {e['urban_clues']}")
        if "distinguish_from_south" in e:
            lines.append(f"    南北差异: {e['distinguish_from_south']}")
        if "regional_differences" in e:
            lines.append(f"    区域差异: {e['regional_differences']}")
        if "shenzhen_specific" in e:
            lines.append(f"    深圳: {e['shenzhen_specific']}")
        if "guangzhou_specific" in e:
            lines.append(f"    广州: {e['guangzhou_specific']}")
    return "\n".join(lines) if lines else ""


def format_knowledge_block(
    entries: list[dict],
    include_urban_hints: bool = False,
) -> str:
    """Format retrieved knowledge entries as a text block for prompt injection.

    Args:
        entries: list of knowledge entry dicts from retrieve_*()
        include_urban_hints: if True, also include urban differentiation hints

    Returns:
        Formatted text block to append to a GeoCoT prompt
    """
    if not entries:
        return ""

    lines = [
        "",
        "--- GEOGRAPHIC KNOWLEDGE REFERENCE ---",
        "以下是对你可能关心的区域的详细地理知识，请结合图像实际内容审慎对照:",
        "",
    ]
    for e in entries:
        cn = e.get("name_cn", e.get("name", ""))
        province = e.get("province", "")
        if province:
            lines.append(f"[{cn}] ({province})")
        else:
            lines.append(f"[{cn}]")
        if e.get("geology"):
            lines.append(f"  地质: {e['geology']}")
        if e.get("vegetation"):
            lines.append(f"  植被: {e['vegetation']}")
        if e.get("cultural_markers"):
            lines.append(f"  文化标志: {e['cultural_markers']}")
        if e.get("urban_clues"):
            lines.append(f"  城市线索: {e['urban_clues']}")
        if e.get("distinguish_from"):
            lines.append(f"  与相似地点区分: {'; '.join(e['distinguish_from'])}")
        if e.get("common_mistakes"):
            lines.append(f"  常见误判: {'; '.join(e['common_mistakes'])}")
        lines.append("")

    if include_urban_hints:
        urban_text = retrieve_all_urban_hints()
        if urban_text:
            lines.append("[中国城市区域差异速查]")
            lines.append(urban_text)
            lines.append("")

    lines.append("--- END KNOWLEDGE REFERENCE ---")
    lines.append("请将以上知识与图像实际观察进行对照。如果知识描述与图像不符，以图像实际观察为准。")
    return "\n".join(lines)


def retrieve_and_inject(
    prompt_text: str,
    prev_outputs: dict,
    lat_hint: Optional[float] = None,
    lng_hint: Optional[float] = None,
) -> str:
    """Main entry point: retrieve relevant knowledge and inject into prompt.

    Called by GeoCoT before each stage to augment the prompt with domain knowledge.

    Args:
        prompt_text: the original stage prompt
        prev_outputs: dict of {stage_name: output_text} from previous stages
        lat_hint: optional latitude hint for proximity-based retrieval
        lng_hint: optional longitude hint

    Returns:
        prompt_text with knowledge block appended (if relevant knowledge found)
    """
    entries = []

    # Strategy 1: proximity-based (if coordinates available)
    if lat_hint is not None and lng_hint is not None:
        entries = retrieve_by_proximity(lat_hint, lng_hint, max_distance_km=500, top_k=3)

    # Strategy 2: keyword-based from all previous outputs
    if not entries:
        all_text = " ".join(prev_outputs.values()) if prev_outputs else ""
        if all_text:
            entries = retrieve_by_keywords(all_text, top_k=3)

    if not entries:
        return prompt_text

    # For Regional/Local stages, include urban hints
    is_regional_or_local = any(
        k in prev_outputs for k in ["regional", "macro"]
    )
    knowledge_block = format_knowledge_block(entries, include_urban_hints=is_regional_or_local)

    return prompt_text + "\n" + knowledge_block
